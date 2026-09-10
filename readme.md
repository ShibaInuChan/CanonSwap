<h1 align="center">CanonSwap</h1>

## Environment Setup

```bash
conda create -n CanonSwap python=3.10
conda activate CanonSwap
```

**Install PyTorch (Over versions may be supported):**

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

**Install other dependencies:**

```bash
pip install -r requirements.txt
```

## Apple Silicon (macOS) Setup

This fork has been patched to run on Apple Silicon Macs (tested with Python 3.11, PyTorch 2.14) using Metal Performance Shaders (MPS) instead of CUDA. The instructions above are written for CUDA/Linux; use the following on macOS instead.

**Requirements:** Python 3.10+. PyTorch's MPS support for `grid_sampler_3d` (used by this model's motion warping) was added in PyTorch 2.9, which in turn requires Python 3.10+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio   # no --index-url: the default macOS build already includes MPS support
```

Before installing `requirements.txt`, edit it:
- Remove the `gradio` line (unused by the CLI scripts here).
- Change `onnxruntime-gpu` to `onnxruntime` (the GPU build is CUDA-only).

```bash
pip install -r requirements.txt
brew install ffmpeg   # if not already installed
```

**Running inference:** set `PYTORCH_ENABLE_MPS_FALLBACK=1` so any operator without an MPS kernel falls back to CPU instead of raising an error.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference_canswap.py -s examples/source.jpeg -t examples/target.mp4
```

**Bugs also fixed in this fork** (unrelated to the CUDA/MPS port — these were pre-existing issues):
- The SegFormer-based face-parsing mask used for paste-back produced scattered noise instead of a face-shaped region; replaced with a landmark-based convex-hull mask built directly in full-frame coordinates (`src/can_swap_pipeline_e2e.py`).
- `SoftErosion` (`src/utils/crop.py`) divided by zero when the off-threshold region was uniformly zero (common with a sharp-edged landmark mask), producing NaN that silently corrupted the entire paste-back blend; guarded against a zero denominator.
- Hardcoded `.cuda()` calls in `src/can_swap_e2e.py`, `src/can_swap_pipeline_e2e.py`, and `src/can_swap_pipeline_v2i.py` replaced with the already-resolved `self.device`.
- Wrapped the inference call in `torch.no_grad()` (`inference_canswap.py`) — several forward passes in the per-frame loop weren't already covered by an inner `no_grad()`, causing memory to grow unboundedly over the course of a video.
- Removed a redundant generator pass that only fed the debug `_concat.mp4` comparison video, for a modest speed-up.

## Performance

The face-swap pipeline (`inference_canswap.py`) has been reworked for speed. The
model and its weights are unchanged — these are all scheduling, precision and IO
changes, so the output is the same.

Measured on an M-series MacBook Air (MPS), 100 frames of 1080p:

| | before | after |
|---|---|---|
| swap loop | 421 s | 144 s |
| target cropping / landmark tracking | ~16 s | 15 s |
| **total** | **457 s** | **161 s** |
| per frame | 4.57 s | 1.61 s |

That is **2.8x**. Note the "before" run also wrote the side-by-side comparison
video, which is now opt-in; asking for it again with
`--flag_write_concat_video True` gives back some of the difference.

What is left is real arithmetic, not overhead: the network is about 2 TFLOP per
frame and now runs at roughly 1.4 TFLOPS on that machine, i.e. a decent fraction
of what the GPU can do. Going substantially faster from here means changing what
the model computes (a smaller decoder output, fewer blocks) or running on a
faster GPU, not further scheduling work.

The changes:

- **Half precision (fp16) is now actually used.** `inference_canswap.py` used to
  force it off, and autocast was skipped entirely on MPS, so every run was fp32.
  fp16 autocast is now probed once at start-up (CUDA *and* MPS) and used when it
  works, falling back to fp32 automatically. Pass
  `--flag_use_half_precision False` to force fp32.
- **Frames are processed in batches** (`--batch_size`, default 4) instead of one
  at a time, so the GPU is no longer stalled by per-frame launch overhead. The
  batch size is halved automatically if the device runs out of memory.
- **Motion extraction is fused into the swap loop.** It used to be a separate
  full pass over the video with a device→numpy→device round trip per frame.
- **The comparison video is opt-in** (`--flag_write_concat_video`). Producing it
  costs an extra full generator pass per frame plus a second video encode.
- **Paste-back only touches the face region.** The soft mask (a 21×21
  convolution applied three times) and the blend used to run over the entire
  frame; they now run over the mask's bounding box, which is a ~10× reduction at
  1080p and more at 4K. The result is identical: the mask is zero everywhere
  outside that box.
- **Full-resolution frames are no longer accumulated in RAM.** Results are
  encoded as they are produced instead of being buffered twice (once raw, once
  watermarked) — several GB saved on a 1080p clip, which on a unified-memory Mac
  is the difference between running and swapping.
- **The identity-modulated convolution weights are cached.** The source identity
  is constant for a whole video, so the 512×512×3×3 modulated kernel used by
  each of the swap module's 14 modulated convolutions is computed once instead
  of once per frame, and a plain convolution replaces the grouped one.
- **3D convolutions pick the faster of two mathematically identical paths.**
  Profiling on an M-series Mac showed the 7x7x7 mask convolution inside
  `DenseMotionNetwork` — which the authors marked `# 65G! NOTE: computation
  cost is large`, and which runs twice per frame — taking 431 ms of the
  628 ms that stage costs, i.e. 45% of the entire frame. MPS' `conv3d` is
  simply slow: expressing the same convolution as seven 2D convolutions
  summed over depth-shifted slices measured 189 ms, faster in fp32 than the
  native call is in fp16. Each 3D convolution now times both paths once for
  its input shape and keeps the winner, so CUDA (where cuDNN's conv3d is
  well tuned) is unaffected. `CANONSWAP_CONV3D=native` disables it.
- Smaller things: no more per-frame debug image writes, no per-frame re-decode of
  the watermark PNG, no device→host synchronisation inside the mask erosion, only
  the two face models that are actually used are loaded, and OpenCV is no longer
  pinned to a single thread (set `CANONSWAP_CV_THREADS=1` to restore that).

A per-stage timing summary is printed at the end of each run, so it is easy to
see where the remaining time goes. Two tools help dig further:

```bash
python tools/profile_modules.py --batch 1,2,4   # per-network-stage timings, best batch size
python tools/diagnose_warp.py                   # why the warp stages cost what they cost:
                                                # autocast coverage, CPU fallbacks, per-step breakdown
```

Note that this model is still significantly heavier per frame than lightweight
one-shot swappers (e.g. inswapper_128), and Apple Silicon remains slower than a
comparable CUDA GPU. Test on a short clip before processing a full video.

## Model Download

### 1. CanonSwap Checkpoints
Download from [here](https://drive.google.com/file/d/1uDWiIam1jziU918iOZY2ATE2dw9aqYAr/view?usp=drive_link) and move to `./pretrained_weights` folder

### 2. InsightFace Models
Download Antelope from [here](https://drive.google.com/file/d/1yXQs6Nd0_hp97UGCvceyGehlCVTPs1ZD/view?usp=sharing).

Download buffalo_l from [here](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip).

Extract both models to `./pretrained_weights/insightface/models` with the following structure:
```
/pretrained_weights/insightface/models/
├── antelope/
│   ├── glintr100.onnx
│   └── scrfd_10g_bnkps.onnx
└── buffalo_l/
│   ├── 2d106det.onnx
│   └── det_10g.onnx
```


### 3. ArcFace
Download ArcFace from [here](https://drive.google.com/file/d/1lDpbmvc7__cIfWU9rTTKNW5OXeeqohUJ/view?usp=drive_link) and extract to `./pretrained_weights` folder.

### 4. Landmark Model
Download Landmark Model from [here](https://drive.google.com/file/d/1uuee7ebWr9lBYfCmIPk8c4fk_YDbiNHz/view?usp=drive_link) and extract to `./pretrained_weights` folder.


## Project Structure After Download

After downloading all models, your project structure should look like:

```
CanonSwap/
├── pretrained_weights/
│   ├── combined_weights.pth
│   ├── arcface_checkpoint.tar
│   ├── landmark.onnx
│   └── insightface/
│       └── models/
│           ├── antelope/
│           │   ├── glintr100.onnx
│           │   └── scrfd_10g_bnkps.onnx
│           └── buffalo_l/
│               ├── 2d106det.onnx
│               └── det_10g.onnx
```

## Inference
The first inference run will automatically download the face parsing model.
### Face Swapping
```bash
python inference_canswap.py -s examples/source.jpeg -t examples/target.mp4
```
This also supports image-to-image swapping.

Useful performance options:

```bash
# larger batches are faster but need more VRAM / unified memory (default: 4)
python inference_canswap.py -s examples/source.jpeg -t examples/target.mp4 -b 8

# fall back to fp32 if fp16 produces artefacts on your GPU
python inference_canswap.py -s ... -t ... --flag_use_half_precision False

# also write the side-by-side comparison video (slower: one extra generator pass per frame)
python inference_canswap.py -s ... -t ... --flag_write_concat_video True
```

### Video-to-Image Swap
```bash
python inference_v2i.py -s examples/i2v_s.jpeg -t examples/i2v_t.mov
```

## Citation

If you find this work useful, please star and cite:

```bibtex
@article{luo2025canonswap,
   title={CanonSwap: High-Fidelity and Consistent Video Face Swapping via Canonical Space Modulation},
   author={Luo, Xiangyang and Zhu, Ye and Liu, Yunfei and Lin, Lijian and Wan, Cong and Cai, Zijian and Huang, Shao-Lun and Li, Yu},
   journal={arXiv preprint arXiv:2507.02691},
   year={2025}
}
```
It is the greatest appreciation of our work!

## License and Attention
This project is licensed under the Research Responsible AI License (ResearchRAIL-M). A full copy of the license can be found in the LICENSE file.
By using this code or model, you agree to the terms outlined in the license.
This project is intended for technical and academic purposes only. You are strictly prohibited from using this project for any illegal or unethical applications, including but not limited to creating non-consensual content, spreading misinformation, or harassing individuals. Please see the Use-Based Restrictions in the license for a non-exhaustive list of prohibited uses.
The authors are exempt from any liability arising from your violation of these terms.

## Acknowledgments

This project is based on [LivePortrait](https://github.com/KwaiVGI/LivePortrait). We thank the authors for their excellent work in efficient portrait animation.
