# coding: utf-8

"""Faster 3D convolution for backends whose conv3d is slow.

A 3D convolution is exactly equal to a sum of 2D convolutions, one per depth
tap of the kernel, applied to depth-shifted slices of the input:

    conv3d(x, W)[..., d, :, :] = sum_kd conv2d(x[..., d + kd - pd, :, :], W[..., kd, :, :])

The arithmetic is identical, but 2D convolution kernels are far better optimised
than 3D ones on some backends. On Apple Silicon (MPS) the 7x7x7 mask convolution
in DenseMotionNetwork -- which the model runs twice per frame and which the
authors flagged as "65G! computation cost is large" -- measured:

    3D conv, fp16 autocast   433 ms
    as 7 x conv2d, fp32      189 ms

so the rewrite more than doubles that layer's speed. On CUDA, where cuDNN's
conv3d is well tuned, the native path is usually the faster one, which is why
each layer measures both once for its input shape and then keeps the winner.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Layers cheaper than this are never measured; the dispatch overhead of the
# rewrite is not worth it and the native call is fine.
MIN_MACS = 5e8


def _tuple3(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v, v)


def is_eligible(conv: nn.Conv3d) -> bool:
    """Only plain 'same'-padded, unit-stride, undilated, ungrouped convs."""
    k = _tuple3(conv.kernel_size)
    return (
        _tuple3(conv.stride) == (1, 1, 1)
        and _tuple3(conv.dilation) == (1, 1, 1)
        and conv.groups == 1
        and getattr(conv, 'padding_mode', 'zeros') == 'zeros'
        and _tuple3(conv.padding) == (k[0] // 2, k[1] // 2, k[2] // 2)
    )


def macs(conv: nn.Conv3d, shape) -> float:
    k = _tuple3(conv.kernel_size)
    d, h, w = shape[2:]
    return conv.in_channels * conv.out_channels * k[0] * k[1] * k[2] * d * h * w


def conv3d_via_conv2d(x, weight, bias):
    """Same result as F.conv3d(x, weight, bias, padding=k//2), via 2D convolutions.

    The per-tap results are accumulated in fp32 even when the convolutions run in
    fp16, so this does not lose precision against conv3d's internal accumulation.
    """
    b, _, d, h, w = x.shape
    out_c, _, kd, kh, kw = weight.shape
    pd = kd // 2

    xf = x.permute(0, 2, 1, 3, 4).reshape(b * d, x.shape[1], h, w)
    out, out_dtype = None, None
    for i in range(kd):
        y = F.conv2d(xf, weight[:, :, i], bias=None, padding=(kh // 2, kw // 2))
        y = y.view(b, d, out_c, h, w)
        if out is None:
            # NOTE: the result dtype has to be the one conv2d produced, not x's.
            # Under autocast a convolution takes fp32 in and gives fp16 out, so
            # returning x's dtype here would hand the rest of the network a
            # different precision than the native conv3d would have.
            out_dtype = y.dtype
            out = torch.zeros(b, d, out_c, h, w, device=y.device, dtype=torch.float32)
        shift = i - pd
        lo, hi = max(0, -shift), min(d, d - shift)
        if lo < hi:
            out[:, lo:hi] += y[:, lo + shift:hi + shift].float()

    out = out.permute(0, 2, 1, 3, 4).to(out_dtype)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1).to(out_dtype)
    return out


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


class FastConv3d(nn.Module):
    """Wraps an nn.Conv3d and uses whichever of the two paths is faster.

    The wrapped convolution keeps its own parameters, so this must be applied
    after the checkpoint is loaded (the wrapper adds a 'conv.' prefix to the
    parameter names).
    """

    def __init__(self, conv: nn.Conv3d, mode='auto'):
        super().__init__()
        self.conv = conv
        self.mode = mode
        self._decomposed_for = {}

    def _use_decomposed(self, x):
        if self.mode == 'native':
            return False
        if not is_eligible(self.conv):
            return False
        if self.mode == 'decomposed':
            return True

        key = (tuple(x.shape), x.dtype)
        cached = self._decomposed_for.get(key)
        if cached is not None:
            return cached

        if macs(self.conv, x.shape) < MIN_MACS:
            self._decomposed_for[key] = False
            return False

        chosen = self._measure(x)
        self._decomposed_for[key] = chosen
        return chosen

    def _measure(self, x):
        w, b = self.conv.weight, self.conv.bias

        def native():
            return self.conv(x)

        def decomposed():
            return conv3d_via_conv2d(x, w, b)

        try:
            timings = []
            for fn in (native, decomposed):
                fn()  # warm up
                _sync(x.device)
                t0 = time.perf_counter()
                for _ in range(2):
                    fn()
                _sync(x.device)
                timings.append(time.perf_counter() - t0)
            return timings[1] < timings[0]
        except Exception:
            return False

    def forward(self, x):
        if self._use_decomposed(x):
            return conv3d_via_conv2d(x, self.conv.weight, self.conv.bias)
        return self.conv(x)


def convert_conv3d(module: nn.Module, mode=None) -> int:
    """Replace every nn.Conv3d under `module` with a FastConv3d, in place.

    Call this AFTER loading weights. Returns the number of layers wrapped.
    `mode` is 'auto' (measure per layer and shape), 'native' or 'decomposed';
    it defaults to $CANONSWAP_CONV3D or 'auto'.
    """
    if mode is None:
        mode = os.environ.get('CANONSWAP_CONV3D', 'auto').lower()
    if mode not in ('auto', 'native', 'decomposed'):
        mode = 'auto'
    if mode == 'native':
        return 0

    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            setattr(module, name, FastConv3d(child, mode=mode))
            n += 1
        elif isinstance(child, FastConv3d):
            continue
        else:
            n += convert_conv3d(child, mode=mode)
    return n
