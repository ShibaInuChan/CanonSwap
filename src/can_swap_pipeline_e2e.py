import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning
import imageio
from skimage.draw import disk

import matplotlib.pyplot as plt

import cv2
from .utils.cv2_config import configure_opencv; configure_opencv()
import numpy as np
import os
import os.path as osp
import time
from itertools import islice
from rich.progress import track, Progress

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream, to_frames, StreamingVideoWriter
from .utils.crop import prepare_paste_back, paste_back
from .utils.crop import dilation_mask, erode_mask, smooth_mask, blend_images, SoftErosion
from .utils.io import load_image_rgb, load_video, stream_video, stream_video_at, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from .utils.filter import smooth
from .utils.rprint import rlog as log
from .can_swap_e2e import can_swapper
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from .utils.watermark import add_watermark_to_frame_list, add_image_watermark
def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

from insightface_func.face_detect_crop_single import Face_detect_crop
watermark_path = "watermark.png"


def _is_out_of_memory(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return 'out of memory' in msg or 'insufficient' in msg or 'can\'t allocate' in msg


class CanSwapPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.can_swapper: can_swapper = can_swapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)
        self.soft_mask = SoftErosion(kernel_size=21, threshold=0.9, iterations=3).to(self.can_swapper.device)
        self.ID_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

        # visualizer_params={"kp_size": 5, "draw_border": True, "colormap": "gist_rainbow"}
        # self.visualizer = Visualizer(**visualizer_params)

        # NOTE: the SegFormer face-parsing model (jonathandinu/face-parsing) that
        # used to be loaded here is no longer used — the paste-back mask is now
        # built from landmarks (see _build_soft_masks()). Removing this
        # avoids an unnecessary Hugging Face Hub download on first run and keeps
        # this fork fully offline after the pretrained_weights/ models are in place.

        self.cropper_insightface = Face_detect_crop(name='antelope', root='pretrained_weights/insightface/models')
        self.cropper_insightface.prepare(ctx_id=0, det_thresh=0.5, det_size=(640,640), mode='None')

    def execute_face_canonical(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.can_swapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg
        ######## load source input ########
        img_rgb = load_image_rgb(args.source)  # Assuming args.source is already a numpy array
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
        source_rgb_lst = [img_rgb]

        crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
        img_crop_256x256 = crop_info['img_crop_256x256']

        I_s = self.can_swapper.prepare_source(img_crop_256x256)
        x_s_info = self.can_swapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        f_s = self.can_swapper.extract_feature_3d(I_s)
        x_s = self.can_swapper.transform_keypoint(x_s_info)

        scale_new = x_s_info['scale']

        x_d_i_new = scale_new * x_c_s
        out = self.can_swapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p = self.can_swapper.parse_output(out['out'])[0]

        # Convert RGB to BGR
        I_p_rgb = cv2.cvtColor(I_p, cv2.COLOR_BGR2RGB)
        # wfp = args.outpath
        cv2.imwrite('source_can.jpg', I_p_rgb)
        # log(f'Animated image: {wfp}')
        return I_p_rgb, x_s_info

    def get_source_id(self, device ,args):
        # source_can, x_s_info = self.execute_face_canonical(args)
        x_s_info = None
        source_can = cv2.imread(args.source)
        source_can_crop = self.cropper_insightface.get(source_can, crop_size=112, max_num=1)
        cv2.imwrite('source_can_112.jpg', source_can_crop[0][0])
        source_can_crop = cv2.cvtColor(source_can_crop[0][0], cv2.COLOR_BGR2RGB)
        source_tensor = self.ID_transform(source_can_crop).unsqueeze(0).to(device)
        source_id = self.can_swapper.getid(source_tensor)
        return source_id, x_s_info

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        """NOTE: kept for compatibility / template dumping.

        `execute()` no longer uses it: extracting the motion in its own pass over
        the video meant a second full traversal plus a device->numpy->device
        round trip per frame, and the motion is now computed inside the batched
        swap loop instead.
        """
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.can_swapper.get_kp_info(I_i)
            x_s = self.can_swapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    ######## paste back helpers ########

    def _mask_roi(self, hull, frame_shape):
        """Bounding box of the (soft-eroded) mask, clipped to the frame.

        The soft erosion is a 21x21 convolution repeated 3 times; running it — and
        the paste-back blend — over a whole 1080p/4K frame costs orders of
        magnitude more than the face region it can actually affect. Padding the
        landmark hull by the total convolution reach gives a region outside of
        which the mask is provably zero, i.e. where paste-back would return the
        original pixels unchanged.
        """
        h_full, w_full = frame_shape[:2]
        x, y, w, h = cv2.boundingRect(hull)
        pad = self.soft_mask.padding * self.soft_mask.iterations + 1
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_full, x + w + pad)
        y1 = min(h_full, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _build_soft_masks(self, lmk_lst, frame_shape, device, dsize_scale):
        """Soft paste-back masks for a batch of frames.

        Returns a list of (roi, mask) with mask a float32 HxWx1 array covering
        `roi` only. The whole batch is eroded in one call so the device is
        synchronised once per batch instead of once per frame.
        """
        rois, mask_np_lst = [], []
        for lmk in lmk_lst:
            hull = cv2.convexHull((lmk * dsize_scale).astype(np.int32))
            roi = self._mask_roi(hull, frame_shape)
            rois.append(roi)
            if roi is None:
                mask_np_lst.append(None)
                continue
            x0, y0, x1, y1 = roi
            mask_np = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillConvexPoly(mask_np, hull - np.array([[x0, y0]], dtype=hull.dtype), 1)
            mask_np_lst.append(mask_np)

        valid = [i for i, m in enumerate(mask_np_lst) if m is not None]
        if not valid:
            return [None] * len(lmk_lst)

        # zero-pad every mask to a common size so they can be eroded as one batch;
        # the padding lies outside each mask's support and cannot change the result
        max_h = max(mask_np_lst[i].shape[0] for i in valid)
        max_w = max(mask_np_lst[i].shape[1] for i in valid)
        batch = np.zeros((len(valid), 1, max_h, max_w), dtype=np.float32)
        for slot, i in enumerate(valid):
            m = mask_np_lst[i]
            batch[slot, 0, :m.shape[0], :m.shape[1]] = m

        with torch.no_grad():
            soft, _ = self.soft_mask(torch.from_numpy(batch).to(device))
            soft = soft.cpu().numpy()

        out = [None] * len(lmk_lst)
        for slot, i in enumerate(valid):
            h, w = mask_np_lst[i].shape
            out[i] = (rois[i], soft[slot, 0, :h, :w][..., None])
        return out

    def _paste_back(self, I_p_i, M_c2o, img_ori, soft_mask):
        """Blend the swapped crop back into the frame, in place, over the mask ROI."""
        if soft_mask is None:
            return img_ori
        (x0, y0, x1, y1), mask = soft_mask

        M = np.asarray(M_c2o, dtype=np.float32)[:2, :].copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        warped = cv2.warpAffine(I_p_i, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)

        roi = img_ori[y0:y1, x0:x1]
        blended = warped.astype(np.float32) * mask + roi.astype(np.float32) * (1 - mask)
        img_ori[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return img_ori

    def execute(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.can_swapper.inference_cfg
        device = self.can_swapper.device
        crop_cfg = self.cropper.crop_cfg

        timings = {}
        t0 = time.perf_counter()
        source_id, x_s_info = self.get_source_id(device, args)
        timings['source id'] = time.perf_counter() - t0

        ######## process target info ########
        # NOTE: for a video the frames are streamed, never collected into a list.
        # Decoding a whole clip up front costs ~6MB per 1080p frame and ~25MB per
        # 4K frame, so a few thousand frames is several GB of RAM before any work
        # starts -- enough to take the machine down. The clip is decoded twice
        # instead (cheap: a fraction of a second per pass), once to track/crop and
        # once to paste back, holding only one batch of full-resolution frames.
        t0 = time.perf_counter()
        if is_video(args.driving):
            flag_is_driving_video = True
            output_fps = int(get_fps(args.driving))
            log(f"Load driving video from: {args.driving}, FPS is {output_fps}")
            driving_frames = lambda: stream_video(args.driving)
        elif is_image(args.driving):
            flag_is_driving_video = False
            driving_img_rgb = load_image_rgb(args.driving)
            output_fps = 25
            log(f"Load driving image from {args.driving}")
            driving_frames = lambda: iter([driving_img_rgb])
        else:
            raise Exception(f"{args.driving} is not a supported type!")
        timings['load target'] = time.perf_counter() - t0

        ######## crop / track the target ########
        t0 = time.perf_counter()
        target_M_c2o_lst = None
        kept_idx_lst = None
        if inf_cfg.flag_crop_driving_video or (not is_square_video(args.driving)):
            ret_d = self.cropper.crop_source_video(driving_frames(), crop_cfg)
            driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
            target_M_c2o_lst = ret_d['M_c2o_lst']
            n_frames = len(driving_rgb_crop_lst)
            # Frames with no detected face are dropped by the cropper, so results
            # are indexed by this list, not by source frame number. A cropper that
            # does not report it is assumed to have kept every frame.
            kept_idx_lst = ret_d.get('idx_lst')
            if kept_idx_lst is None:
                kept_idx_lst = list(range(n_frames))
            log(f'Target video is cropped, {n_frames} frames are processed.')
            # NOTE: the cropper already returns 256x256 crops, so only resize when
            # something else produced them (the old unconditional resize copied
            # every frame for nothing).
            driving_rgb_crop_256x256_lst = [
                _ if _.shape[:2] == (256, 256) else cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst
            ]
            lmk_to_full_scale = crop_cfg.dsize / 256
        else:
            driving_rgb_lst = list(driving_frames())
            driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256
            n_frames = len(driving_rgb_crop_256x256_lst)
            lmk_to_full_scale = 1.0
            del driving_rgb_lst
        timings['crop target'] = time.perf_counter() - t0

        if n_frames == 0:
            raise Exception(f'No face was detected in any frame of {args.driving}.')

        ######## what to produce ########
        flag_pasteback = bool(inf_cfg.flag_pasteback and inf_cfg.flag_do_crop)
        if flag_pasteback and target_M_c2o_lst is None:
            # the un-cropped branch never computes a crop->original transform
            log('No crop transform available for the target, writing the cropped result directly.')
            flag_pasteback = False
        # The side-by-side comparison video costs an extra full generator pass per
        # frame (plus a second encode of a 4x wider video), so it is opt-in.
        flag_concat = bool(getattr(inf_cfg, 'flag_write_concat_video', False))
        batch_size = max(1, int(getattr(inf_cfg, 'batch_size', 4)))

        mkdir(args.output_dir)
        if flag_is_driving_video:
            wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.mp4')
            wfp_concat = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat.mp4')
        else:
            wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.jpg')
            wfp_concat = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat.jpg')

        # Frames are encoded as they are produced; keeping the finished
        # full-resolution frames in a list (twice, once more for the watermarked
        # copy) used several GB of RAM on a 1080p clip.
        writer = StreamingVideoWriter(wfp, fps=output_fps) if flag_is_driving_video else None
        result_frames = []          # only used for the image branch
        I_p_lst, I_can_lst = [], []  # only filled for the comparison video

        if flag_is_driving_video:
            log(f"The target video consists of {n_frames} frames.")
        else:
            log(f"The output is an image.")

        ######## animate ########
        # Only the frames that paste-back actually needs are decoded, one batch at
        # a time; with paste-back off the video is not decoded a second time.
        t0 = time.perf_counter()
        t_net = 0.0
        i = 0
        if not flag_pasteback:
            full_frames = iter(())
        elif flag_is_driving_video:
            full_frames = stream_video_at(args.driving, kept_idx_lst)
        else:
            full_frames = iter((driving_img_rgb,))
        try:
            with Progress(transient=True) as progress:
                task = progress.add_task('🚀Swapping...', total=n_frames)
                while i < n_frames:
                    b = min(batch_size, n_frames - i)
                    frames = driving_rgb_crop_256x256_lst[i:i + b]

                    t_batch = time.perf_counter()
                    try:
                        I_d = self.can_swapper.prepare_frames(frames)
                        out_imgs, can_imgs = self.can_swapper.swap_batch(
                            I_d, source_id, need_canonical=flag_concat
                        )
                    except (RuntimeError, MemoryError) as e:
                        if b > 1 and _is_out_of_memory(e):
                            batch_size = max(1, b // 2)
                            log(f'Out of memory at batch size {b}, retrying with {batch_size}.')
                            try:
                                if device.startswith('cuda'):
                                    torch.cuda.empty_cache()
                                elif device == 'mps':
                                    torch.mps.empty_cache()
                            except Exception:
                                pass
                            continue
                        raise
                    t_net += time.perf_counter() - t_batch

                    if flag_pasteback:
                        batch_full = list(islice(full_frames, b))
                        if len(batch_full) != b:
                            raise Exception(
                                f'Target video ended early: expected {b} more frames at {i}, got {len(batch_full)}.')
                        soft_masks = self._build_soft_masks(
                            driving_lmk_crop_lst[i:i + b], batch_full[0].shape, device, lmk_to_full_scale
                        )
                    else:
                        batch_full, soft_masks = None, [None] * b

                    for k in range(b):
                        idx = i + k
                        I_p_i = out_imgs[k]

                        if flag_concat:
                            I_p_lst.append(I_p_i)
                            I_can_lst.append(can_imgs[k])

                        if flag_pasteback:
                            frame = self._paste_back(I_p_i, target_M_c2o_lst[idx], batch_full[k], soft_masks[k])
                            frame = add_image_watermark(frame, watermark_path, opacity=0.2, inplace=True)
                            batch_full[k] = None  # let the frame be collected
                        else:
                            frame = add_image_watermark(I_p_i, watermark_path, opacity=0.2)

                        if writer is not None:
                            writer.append(frame)
                        else:
                            result_frames.append(frame)

                    i += b
                    progress.update(task, advance=b)
        finally:
            if writer is not None:
                writer.close()
        timings['swap'] = time.perf_counter() - t0
        timings['  └ network'] = t_net

        ######## final outputs ########
        if flag_is_driving_video:
            flag_driving_has_audio = has_audio_stream(args.driving)

            if flag_concat:
                t0 = time.perf_counter()
                frames_concatenated = concat_frames(
                    driving_rgb_crop_256x256_lst[:n_frames], I_can_lst, I_p_lst, driving_rgb_crop_256x256_lst[:n_frames]
                )
                images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)
                if flag_driving_has_audio:
                    wfp_concat_with_audio = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat_with_audio.mp4')
                    log(f"Audio is selected from {args.driving}, concat mode")
                    add_audio_to_video(wfp_concat, args.driving, wfp_concat_with_audio)
                    os.replace(wfp_concat_with_audio, wfp_concat)
                timings['concat video'] = time.perf_counter() - t0
            else:
                wfp_concat = None

            ######### build the final result #########
            if flag_driving_has_audio:
                t0 = time.perf_counter()
                wfp_with_audio = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_with_audio.mp4')
                log(f"Audio is selected from {args.driving}")
                add_audio_to_video(wfp, args.driving, wfp_with_audio)
                os.replace(wfp_with_audio, wfp)
                timings['audio mux'] = time.perf_counter() - t0

            log(f'Results: {wfp}')
            if wfp_concat is not None:
                log(f'Results with concat: {wfp_concat}')
        else:
            cv2.imwrite(wfp, result_frames[0][..., ::-1])
            if flag_concat:
                frames_concatenated = concat_frames(
                    driving_rgb_crop_256x256_lst[:n_frames], I_can_lst, I_p_lst, driving_rgb_crop_256x256_lst[:n_frames]
                )
                cv2.imwrite(wfp_concat, frames_concatenated[0][..., ::-1])
            else:
                wfp_concat = None
            log(f'Animated image: {wfp}')
            if wfp_concat is not None:
                log(f'Animated image with concat: {wfp_concat}')

        from .modules.fast_conv3d import choice_summary
        log(choice_summary())

        total = sum(v for k, v in timings.items() if not k.startswith(' '))
        log('Timing: ' + ', '.join(f'{k} {v:.1f}s' for k, v in timings.items()) +
            f' | total {total:.1f}s' +
            (f' ({total / max(1, n_frames):.2f}s/frame)' if flag_is_driving_video else ''))

        return wfp, wfp_concat
