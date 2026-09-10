# coding: utf-8
"""Find out why the two warp stages dominate the frame time.

Reports, for this machine:
  1. which ops autocast actually converts to fp16 (an op it misses keeps running
     in fp32 no matter what --flag_use_half_precision says)
  2. whether any op silently falls back to the CPU under
     PYTORCH_ENABLE_MPS_FALLBACK=1, which would make it far slower than it looks
  3. a per-step breakdown of DenseMotionNetwork, the module both warps run
  4. isolated timings for the two suspects: the 7x7x7 mask convolution and the
     3D grid_sample

    python tools/diagnose_warp.py
"""

import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.dense_motion import DenseMotionNetwork
from src.modules.fast_conv3d import conv3d_via_conv2d, conv3d_via_conv2d_fused
from src.modules.util import kp2gaussian, make_coordinate_grid

NUM_KP, COMPRESS, D, H, W = 21, 4, 16, 64, 64
HOURGLASS_OUT = 142  # block_expansion(32) + in_features((21+1)*(4+1)=110)


def pick_device():
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


def timeit(fn, device, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def autocast_ctx(device, enabled=True):
    device_type = 'cuda' if device.startswith('cuda') else device
    if device_type == 'cpu' or not enabled:
        import contextlib
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch.float16)


# ---------------------------------------------------------------- 1. autocast coverage
def report_autocast(device):
    print('\n=== 1. what autocast actually converts to fp16 ===')
    print('   (anything still float32 here runs at full precision no matter the flag)')
    with torch.no_grad(), autocast_ctx(device):
        x2 = torch.randn(1, 8, 32, 32, device=device)
        x3 = torch.randn(1, 8, 4, 16, 16, device=device)
        probes = {
            'conv2d': lambda: F.conv2d(x2, torch.randn(8, 8, 3, 3, device=device), padding=1),
            'conv3d': lambda: F.conv3d(x3, torch.randn(8, 8, 3, 3, 3, device=device), padding=1),
            'grid_sample 2d': lambda: F.grid_sample(x2, torch.rand(1, 32, 32, 2, device=device) * 2 - 1, align_corners=False),
            'grid_sample 3d': lambda: F.grid_sample(x3, torch.rand(1, 4, 16, 16, 3, device=device) * 2 - 1, align_corners=False),
            'matmul': lambda: torch.randn(4, 8, device=device) @ torch.randn(8, 8, device=device),
            'batch_norm 3d': lambda: F.batch_norm(x3, None, None, training=True),
        }
        for name, fn in probes.items():
            try:
                print(f'   {name:<16} -> {str(fn().dtype).replace("torch.", "")}')
            except Exception as e:
                print(f'   {name:<16} -> failed: {str(e).splitlines()[0]}')


# ---------------------------------------------------------------- 2. cpu fallback
FALLBACK_PROBE = '''
import torch, torch.nn.functional as F, sys
d = "mps"
x3 = torch.randn(1, 4, 8, 16, 16, device=d)
ops = {
 "conv3d": lambda: F.conv3d(x3, torch.randn(4,4,3,3,3, device=d), padding=1),
 "conv3d k7": lambda: F.conv3d(x3, torch.randn(4,4,7,7,7, device=d), padding=3),
 "grid_sample 3d": lambda: F.grid_sample(x3, torch.rand(1,8,16,16,3, device=d)*2-1, align_corners=False),
 "grid_sample 3d fp16": lambda: F.grid_sample(x3.half(), (torch.rand(1,8,16,16,3, device=d)*2-1).half(), align_corners=False),
 "conv3d fp16": lambda: F.conv3d(x3.half(), torch.randn(4,4,3,3,3, device=d).half(), padding=1),
 "batch_norm 3d": lambda: F.batch_norm(x3, None, None, training=True),
}
for name, fn in ops.items():
    try:
        fn(); torch.mps.synchronize(); print(f"OK        {name}")
    except NotImplementedError as e:
        print(f"CPU-ONLY  {name}   <-- runs on the CPU when the fallback is enabled")
    except Exception as e:
        print(f"ERROR     {name}: {type(e).__name__}: {str(e).splitlines()[0]}")
'''


def report_fallback(device):
    print('\n=== 2. ops with no MPS kernel (they run on the CPU via the fallback) ===')
    if device != 'mps':
        print('   not applicable on this device')
        return
    env = {k: v for k, v in os.environ.items() if k != 'PYTORCH_ENABLE_MPS_FALLBACK'}
    r = subprocess.run([sys.executable, '-c', FALLBACK_PROBE], capture_output=True, text=True, env=env)
    for line in (r.stdout or '').splitlines():
        print('   ' + line)
    if r.returncode != 0 and not r.stdout:
        print('   probe failed:', (r.stderr or '').strip().splitlines()[-1:])


# ---------------------------------------------------------------- 3. dense motion breakdown
def report_dense_motion(device, half):
    print(f'\n=== 3. DenseMotionNetwork step by step (fp16 autocast: {half}) ===')
    dm = DenseMotionNetwork(block_expansion=32, num_blocks=5, max_features=1024, num_kp=NUM_KP,
                            feature_channel=32, reshape_depth=D, compress=COMPRESS,
                            estimate_occlusion_map=True).to(device).eval()
    feat = torch.randn(1, 32, D, H, W, device=device)
    kp_d = torch.randn(1, NUM_KP, 3, device=device) * 0.3
    kp_s = torch.randn(1, NUM_KP, 3, device=device) * 0.3

    with torch.no_grad(), autocast_ctx(device, half):
        compressed = F.relu(dm.norm(dm.compress(feat)))
        sparse_motion = dm.create_sparse_motions(compressed, kp_d, kp_s)
        deformed = dm.create_deformed_feature(compressed, sparse_motion)
        heatmap = dm.create_heatmap_representations(deformed, kp_d, kp_s)
        inp = torch.cat([heatmap, deformed], dim=2).view(1, -1, D, H, W)
        prediction = dm.hourglass(inp)

    def wrap(fn):
        def run():
            with torch.no_grad(), autocast_ctx(device, half):
                fn()
        return run

    steps = [
        ('compress + norm + relu', wrap(lambda: F.relu(dm.norm(dm.compress(feat))))),
        ('create_sparse_motions', wrap(lambda: dm.create_sparse_motions(compressed, kp_d, kp_s))),
        ('create_deformed_feature (grid_sample 3d)', wrap(lambda: dm.create_deformed_feature(compressed, sparse_motion))),
        ('create_heatmap_representations', wrap(lambda: dm.create_heatmap_representations(deformed, kp_d, kp_s))),
        ('hourglass', wrap(lambda: dm.hourglass(inp))),
        ('mask conv 7x7x7 (142->22)', wrap(lambda: dm.mask(prediction))),
        ('occlusion conv 7x7', wrap(lambda: dm.occlusion(prediction.view(1, -1, H, W)))),
        ('full forward', wrap(lambda: dm(feat, kp_d, kp_s))),
    ]
    results = []
    for name, fn in steps:
        try:
            results.append((name, timeit(fn, device)))
        except Exception as e:
            results.append((name, float('nan')))
            print(f'   {name}: failed {str(e).splitlines()[0]}')
    full = results[-1][1]
    print(f'   {"step":<44}{"ms":>9}{"share":>8}')
    print('   ' + '-' * 61)
    for name, ms in results[:-1]:
        print(f'   {name:<44}{ms:>9.1f}{ms / full * 100:>7.1f}%')
    print('   ' + '-' * 61)
    print(f'   {"full forward":<44}{full:>9.1f}')
    return full


# ---------------------------------------------------------------- 4. the two suspects
def report_suspects(device):
    print('\n=== 4. the 7x7x7 mask conv and the 3D grid_sample, in isolation ===')
    x = torch.randn(1, HOURGLASS_OUT, D, H, W, device=device)
    conv = nn.Conv3d(HOURGLASS_OUT, NUM_KP + 1, kernel_size=7, padding=3).to(device).eval()

    rows = []
    with torch.no_grad():
        rows.append(('mask conv, fp32', timeit(lambda: conv(x), device)))
        rows.append(('mask conv, fp16 autocast', timeit(lambda: _autocast_call(conv, x, device), device)))
        try:
            conv_h = nn.Conv3d(HOURGLASS_OUT, NUM_KP + 1, kernel_size=7, padding=3).to(device).half().eval()
            xh = x.half()
            rows.append(('mask conv, fp16 weights', timeit(lambda: conv_h(xh), device)))
        except Exception as e:
            print('   fp16 weights failed:', str(e).splitlines()[0])

        # the two rewrites: one 2D convolution per depth tap, or all taps stacked
        # into a single 2D convolution
        small = torch.randn(1, 5, 6, 12, 12, device=device)
        w_small = torch.randn(4, 5, 7, 7, 7, device=device) * 0.02
        b_small = torch.randn(4, device=device)
        ref = F.conv3d(small, w_small, b_small, padding=3)
        for label, alt in (('per-tap', conv3d_via_conv2d(small, w_small, b_small)),
                           ('fused', conv3d_via_conv2d_fused(small, w_small, b_small))):
            print(f'   [{label} matches F.conv3d: {torch.allclose(ref, alt, atol=1e-4)}, '
                  f'max diff {(ref - alt).abs().max():.2e}]')
        rows.append(('mask conv as 7x conv2d, fp32',
                     timeit(lambda: conv3d_via_conv2d(x, conv.weight, conv.bias), device)))
        rows.append(('mask conv as 7x conv2d, fp16 autocast',
                     timeit(lambda: _autocast_decomp(conv, x, device), device)))
        w_cat = conv.weight.permute(2, 0, 1, 3, 4).reshape(-1, conv.weight.shape[1], 7, 7).contiguous()
        rows.append(('mask conv as 1 fused conv2d, fp32',
                     timeit(lambda: conv3d_via_conv2d_fused(x, conv.weight, conv.bias, w_cat), device)))
        rows.append(('mask conv as 1 fused conv2d, fp16 autocast',
                     timeit(lambda: _autocast_decomp(conv, x, device, fused=True), device)))

        n = NUM_KP + 1
        feat = torch.randn(n, COMPRESS, D, H, W, device=device)
        grid = (torch.rand(n, D, H, W, 3, device=device) * 2 - 1)
        rows.append(('grid_sample 3d, fp32', timeit(lambda: F.grid_sample(feat, grid, align_corners=False), device)))
        try:
            fh, gh = feat.half(), grid.half()
            rows.append(('grid_sample 3d, fp16', timeit(lambda: F.grid_sample(fh, gh, align_corners=False), device)))
        except Exception as e:
            print('   grid_sample fp16 failed:', str(e).splitlines()[0])

    for name, ms in rows:
        print(f'   {name:<32}{ms:>9.1f} ms')
    print('\n   (each of these runs twice per frame)')



def _autocast_call(conv, x, device):
    with torch.no_grad(), autocast_ctx(device):
        return conv(x)


def _autocast_decomp(conv, x, device, fused=False):
    fn = conv3d_via_conv2d_fused if fused else conv3d_via_conv2d
    with torch.no_grad(), autocast_ctx(device):
        return fn(x, conv.weight, conv.bias)


def main():
    device = pick_device()
    print(f'device: {device}   torch {torch.__version__}')
    print(f'PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "<unset>")}')
    report_autocast(device)
    report_fallback(device)
    full16 = report_dense_motion(device, half=True)
    full32 = report_dense_motion(device, half=False)
    print(f'\n   dense motion: fp16 autocast {full16:.1f} ms vs fp32 {full32:.1f} ms '
          f'({full32 / full16:.2f}x)' if full16 else '')
    report_suspects(device)


if __name__ == '__main__':
    main()
