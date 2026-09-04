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

**Performance:** even with these fixes, this model is significantly heavier per-frame than lightweight one-shot swappers (e.g. inswapper_128). Expect notably longer processing time on Apple Silicon than on a comparable CUDA GPU. Test on a short clip before processing a full video.

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
