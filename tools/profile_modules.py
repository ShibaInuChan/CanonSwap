# coding: utf-8
"""Per-module timing for the CanonSwap swap loop.

Answers "where do the seconds per frame actually go" by running each network
stage on its own with the real shapes and synchronising the device in between.
Weights are optional (timing does not depend on their values), so this runs even
on a machine that has not downloaded the checkpoints yet.

    python tools/profile_modules.py                  # default: batch 1,2,4
    python tools/profile_modules.py --batch 1 --fp32 # compare against fp32
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from src.modules.adaptive_modulate import G3d
from src.modules.adaptive_modulate import transfer_model2 as transfer_model_big
from src.modules.motion_extractor import MotionExtractor
from src.modules.spade_generator import SPADEDecoder
from src.modules.warping_network import WarpingNetwork
from src.can_swap_e2e import can_swapper
from src.modules.fast_conv3d import convert_conv3d
from src.utils.camera import headpose_pred_to_degree

MODELS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'config', 'models.yaml')
WEIGHTS = 'pretrained_weights/combined_weights.pth'


def pick_device(force_cpu=False):
    if force_cpu:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda:0'
    try:
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def sync(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def build(device, half):
    cfg = yaml.safe_load(open(MODELS_YAML))['model_params']
    cfg['spade_generator_params']['upscale'] = 2

    class Cfg:
        pass
    c = Cfg()
    c.flag_use_half_precision = half
    c.device_id = 0

    s = can_swapper.__new__(can_swapper)      # skip __init__: no checkpoints needed
    s.inference_cfg = c
    s.device = device
    s.compile = False
    s.appearance_feature_extractor = AppearanceFeatureExtractor(**cfg['appearance_feature_extractor_params']).to(device).eval()
    s.motion_extractor = MotionExtractor(**cfg['motion_extractor_params']).to(device).eval()
    s.warping_module = WarpingNetwork(**cfg['warping_module_params']).to(device).eval()
    s.spade_generator = SPADEDecoder(**cfg['spade_generator_params']).to(device).eval()
    s.swap_module = transfer_model_big().to(device).eval()
    s.refine_module = G3d().to(device).eval()
    s.autocast_available = s._probe_autocast()

    if os.path.exists(WEIGHTS):
        w = torch.load(WEIGHTS, map_location='cpu')
        s.appearance_feature_extractor.load_state_dict(w['appearance_feature_extractor'])
        s.motion_extractor.load_state_dict(w['motion_extractor'])
        s.warping_module.load_state_dict(w['warping_module'])
        s.spade_generator.load_state_dict(w['spade_generator'])
        s.swap_module.load_state_dict(w['transfer'])
        s.refine_module.load_state_dict(w['refine'])
        print(f'loaded weights from {WEIGHTS}')
    else:
        print(f'{WEIGHTS} not found, using random weights (timings are unaffected)')

    if os.environ.get('CANONSWAP_CONV3D', 'auto').lower() != 'native':
        n = sum(convert_conv3d(m) for m in (s.appearance_feature_extractor, s.motion_extractor,
                                            s.warping_module, s.spade_generator,
                                            s.swap_module, s.refine_module))
        print(f'auto-tuning {n} 3D convolutions (CANONSWAP_CONV3D=native to compare without)')
    return s


def timeit(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters


def profile(s, device, bs, warmup, iters):
    ctx = s.inference_ctx
    I_d = torch.rand(bs, 3, 256, 256, device=device)
    source_id = torch.nn.functional.normalize(torch.randn(1, 512, device=device), dim=1)

    with torch.no_grad(), ctx():
        x_info = s.motion_extractor(I_d)
        x_info = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in x_info.items()}
        n = x_info['kp'].shape[0]
        for k in ('pitch', 'yaw', 'roll'):
            x_info[k] = headpose_pred_to_degree(x_info[k])[:, None]
        x_info['kp'] = x_info['kp'].reshape(n, -1, 3)
        x_info['exp'] = x_info['exp'].reshape(n, -1, 3)
        x_t = s.transform_keypoint(x_info)
        x_can = x_info['scale'][..., None] * x_info['kp']
        f_s = s.appearance_feature_extractor(I_d)
        f_can, occ = s.warping_module.warp(f_s, x_t, x_can)
        f_swap = s.swap_module(f_can, source_id)
        f_ref = s.refine_module(f_swap)
        warped = s.warping_module(f_ref, kp_source=x_can, kp_driving=x_t)['out']
        decoded = s.spade_generator(feature=warped)

    def wrap(fn):
        def run():
            with torch.no_grad(), ctx():
                fn()
        return run

    stages = [
        ('motion extractor', wrap(lambda: s.motion_extractor(I_d))),
        ('appearance extractor', wrap(lambda: s.appearance_feature_extractor(I_d))),
        ('warp #1 (target->canonical)', wrap(lambda: s.warping_module.warp(f_s, x_t, x_can))),
        ('swap module (identity)', wrap(lambda: s.swap_module(f_can, source_id))),
        ('refine module (G3d)', wrap(lambda: s.refine_module(f_swap))),
        ('warp #2 (canonical->target)', wrap(lambda: s.warping_module(f_ref, kp_source=x_can, kp_driving=x_t))),
        ('spade generator (->512)', wrap(lambda: s.spade_generator(feature=warped))),
        ('output to host (uint8)', lambda: s.parse_output(decoded)),
    ]

    print(f'\n  batch {bs}  ({iters} iterations, {warmup} warm-up)')
    print(f'  {"stage":<32}{"ms/frame":>10}{"share":>8}')
    print('  ' + '-' * 50)
    per_frame = []
    for name, fn in stages:
        per_frame.append((name, timeit(fn, device, warmup, iters) * 1000 / bs))
    total = sum(v for _, v in per_frame)
    for name, v in per_frame:
        print(f'  {name:<32}{v:>10.1f}{v / total * 100:>7.1f}%')
    print('  ' + '-' * 50)
    print(f'  {"sum of stages":<32}{total:>10.1f}')

    whole = timeit(lambda: s.swap_batch(I_d, source_id), device, warmup, iters) * 1000 / bs
    print(f'  {"swap_batch() end to end":<32}{whole:>10.1f} ms/frame')
    return whole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='1,2,4', help='comma separated batch sizes to try')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--fp32', action='store_true', help='disable half precision')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = pick_device(args.cpu)
    half = not args.fp32
    print(f'device: {device}   half precision requested: {half}   torch {torch.__version__}')

    s = build(device, half)
    print(f'half precision active: {s.half_precision_enabled()}')

    results = {}
    for bs in [int(b) for b in args.batch.split(',')]:
        try:
            results[bs] = profile(s, device, bs, args.warmup, args.iters)
        except RuntimeError as e:
            print(f'\n  batch {bs}: failed ({str(e).splitlines()[0]})')

    if len(results) > 1:
        print('\nbest batch size: -b %d (%.1f ms/frame)' % min(results.items(), key=lambda kv: kv[1]))


if __name__ == '__main__':
    main()
