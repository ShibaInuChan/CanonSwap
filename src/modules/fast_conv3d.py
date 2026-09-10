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


def _depth_shifted_sum(y_per_tap, b, d, out_c, h, w, kd):
    """Sum the per-tap 2D results over depth-shifted slices, accumulating in fp32.

    y_per_tap(i) returns the i-th tap's result shaped (b, d, out_c, h, w).
    fp32 accumulation matches conv3d's own internal accumulation, so running the
    convolutions in fp16 does not cost precision here.
    """
    pd = kd // 2
    out = None
    out_dtype = None
    for i in range(kd):
        y = y_per_tap(i)
        if out is None:
            # NOTE: the result dtype has to be the one conv2d produced, not x's.
            # Under autocast a convolution takes fp32 in and gives fp16 out, so
            # returning x's dtype would hand the rest of the network a different
            # precision than the native conv3d would have.
            out_dtype = y.dtype
            out = torch.zeros(b, d, out_c, h, w, device=y.device, dtype=torch.float32)
        shift = i - pd
        lo, hi = max(0, -shift), min(d, d - shift)
        if lo < hi:
            out[:, lo:hi] += y[:, lo + shift:hi + shift].float()
    return out.permute(0, 2, 1, 3, 4).to(out_dtype), out_dtype


def conv3d_via_conv2d(x, weight, bias):
    """Same result as F.conv3d(x, weight, bias, padding=k//2), as kd 2D convolutions."""
    b, _, d, h, w = x.shape
    out_c, _, kd, kh, kw = weight.shape
    xf = x.permute(0, 2, 1, 3, 4).reshape(b * d, x.shape[1], h, w)

    def tap(i):
        y = F.conv2d(xf, weight[:, :, i], bias=None, padding=(kh // 2, kw // 2))
        return y.view(b, d, out_c, h, w)

    out, out_dtype = _depth_shifted_sum(tap, b, d, out_c, h, w, kd)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1).to(out_dtype)
    return out


def conv3d_via_conv2d_fused(x, weight, bias, w_cat=None):
    """Same again, but as a SINGLE 2D convolution.

    The kd taps are stacked along the output-channel axis of the weight, so one
    conv2d produces all of them at once. This matters when the layer has few
    output channels -- the dense-motion mask convolution has 22, which makes each
    per-tap convolution a badly shaped GEMM; stacking gives it kd*22 instead, at
    identical arithmetic and without duplicating the input.
    """
    b, _, d, h, w = x.shape
    out_c, _, kd, kh, kw = weight.shape
    xf = x.permute(0, 2, 1, 3, 4).reshape(b * d, x.shape[1], h, w)
    if w_cat is None:
        w_cat = weight.permute(2, 0, 1, 3, 4).reshape(kd * out_c, weight.shape[1], kh, kw)
    y = F.conv2d(xf, w_cat, bias=None, padding=(kh // 2, kw // 2))
    y = y.view(b, d, kd, out_c, h, w)

    out, out_dtype = _depth_shifted_sum(lambda i: y[:, :, i], b, d, out_c, h, w, kd)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1).to(out_dtype)
    return out


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


NATIVE, PER_TAP, FUSED = 'native', 'per-tap', 'fused'

# Every auto-tuning decision, so a run can report what it actually picked.
_DECISIONS = []


def choice_summary(top=10):
    """What the auto-tuner chose, and what it measured, for each layer/shape."""
    if not _DECISIONS:
        return '3D convolutions: nothing auto-tuned (all below the cost threshold, ' \
               'ineligible, or a path was forced).'

    counts = {}
    for d in _DECISIONS:
        counts[d['chosen']] = counts.get(d['chosen'], 0) + 1
    head = '3D conv paths chosen: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items()))

    rows = sorted(_DECISIONS, key=lambda d: -min(v for v in d['ms'].values() if v is not None))[:top]
    lines = [head, f'  {"layer":<26}{"batch":>6}  ' + ''.join(f'{p:>10}' for p in (NATIVE, PER_TAP, FUSED)) + '   chosen']
    for d in rows:
        ms = ''.join((f'{d["ms"][p]:>10.1f}' if d['ms'].get(p) is not None else f'{"-":>10}')
                     for p in (NATIVE, PER_TAP, FUSED))
        lines.append(f'  {d["layer"]:<26}{d["shape"][0]:>6}  {ms}   {d["chosen"]}')
    return '\n'.join(lines)


def reset_choices():
    _DECISIONS.clear()


class FastConv3d(nn.Module):
    """Wraps an nn.Conv3d and uses whichever equivalent path is fastest here.

    The wrapped convolution keeps its own parameters, so this must be applied
    after the checkpoint is loaded (the wrapper adds a 'conv.' prefix to the
    parameter names).
    """

    def __init__(self, conv: nn.Conv3d, mode='auto'):
        super().__init__()
        self.conv = conv
        self.mode = mode
        self._path_for = {}
        self._w_cat = None
        self._w_cat_version = None

    def w_cat(self):
        """Taps stacked along the output-channel axis; the weights never change
        during inference, so this is built once."""
        w = self.conv.weight
        if self._w_cat is None or self._w_cat_version != w._version:
            out_c, in_c, kd, kh, kw = w.shape
            self._w_cat = w.permute(2, 0, 1, 3, 4).reshape(kd * out_c, in_c, kh, kw).contiguous()
            self._w_cat_version = w._version
        return self._w_cat

    def _path(self, x):
        if self.mode == NATIVE or not is_eligible(self.conv):
            return NATIVE
        if self.mode in (PER_TAP, FUSED):
            return self.mode

        key = (tuple(x.shape), x.dtype)
        cached = self._path_for.get(key)
        if cached is not None:
            return cached

        if macs(self.conv, x.shape) < MIN_MACS:
            self._path_for[key] = NATIVE
            return NATIVE

        chosen = self._measure(x)
        self._path_for[key] = chosen
        return chosen

    def _measure(self, x):
        w, b = self.conv.weight, self.conv.bias
        candidates = [
            (NATIVE, lambda: self.conv(x)),
            (PER_TAP, lambda: conv3d_via_conv2d(x, w, b)),
            (FUSED, lambda: conv3d_via_conv2d_fused(x, w, b, self.w_cat())),
        ]
        best, best_t = NATIVE, None
        measured = {}
        for name, fn in candidates:
            try:
                # Two warm-up calls: the first builds any cached weight and
                # compiles the backend's graph, the second lands on a warm
                # allocator, so the timed calls measure steady state.
                fn()
                fn()
                _sync(x.device)
                t0 = time.perf_counter()
                for _ in range(2):
                    fn()
                _sync(x.device)
                elapsed = time.perf_counter() - t0
            except Exception as e:
                measured[name] = None  # e.g. out of memory; just do not pick it
                continue
            measured[name] = elapsed / 2 * 1000
            if best_t is None or elapsed < best_t:
                best, best_t = name, elapsed

        k = _tuple3(self.conv.kernel_size)
        _DECISIONS.append({
            'layer': f'{self.conv.in_channels}->{self.conv.out_channels} k{k[0]} '
                     f'{x.shape[2]}x{x.shape[3]}x{x.shape[4]}',
            'shape': tuple(x.shape),
            'dtype': str(x.dtype),
            'ms': measured,
            'chosen': best,
        })
        return best

    def forward(self, x):
        path = self._path(x)
        if path == PER_TAP:
            return conv3d_via_conv2d(x, self.conv.weight, self.conv.bias)
        if path == FUSED:
            return conv3d_via_conv2d_fused(x, self.conv.weight, self.conv.bias, self.w_cat())
        return self.conv(x)


def convert_conv3d(module: nn.Module, mode=None) -> int:
    """Replace every nn.Conv3d under `module` with a FastConv3d, in place.

    Call this AFTER loading weights. Returns the number of layers wrapped.
    `mode` is 'auto' (measure per layer and shape), or one of 'native',
    'per-tap', 'fused' to force a path. Defaults to $CANONSWAP_CONV3D or 'auto'.
    """
    if mode is None:
        mode = os.environ.get('CANONSWAP_CONV3D', 'auto').lower()
    if mode not in ('auto', NATIVE, PER_TAP, FUSED):
        mode = 'auto'
    if mode == NATIVE:
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
