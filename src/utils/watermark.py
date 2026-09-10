# coding: utf-8

"""
Watermark compositing.

The previous implementation re-decoded and re-resized the watermark PNG for
*every* frame and blended it with a per-channel python loop.  Both the decode
and the resize only depend on the frame size, so they are cached here and the
blend is done once over all three channels.
"""

import os

import cv2
import numpy as np

from .cv2_config import configure_opencv; configure_opencv()

# (path, mtime, target_width, target_height, opacity) -> (premultiplied rgb, 1 - alpha)
_WATERMARK_CACHE = {}
_CACHE_LIMIT = 8


def _prepare_watermark(watermark_path, new_wm_w, new_wm_h, opacity):
    """Return (wm_premultiplied, inv_alpha) as float32 arrays, or None."""
    try:
        mtime = os.path.getmtime(watermark_path)
    except OSError:
        return None

    key = (watermark_path, mtime, new_wm_w, new_wm_h, float(opacity))
    cached = _WATERMARK_CACHE.get(key)
    if cached is not None:
        return cached

    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if watermark is None:
        return None

    watermark_resized = cv2.resize(watermark, (new_wm_w, new_wm_h))

    if watermark_resized.ndim == 3 and watermark_resized.shape[2] == 4:
        wm_rgb = watermark_resized[:, :, :3].astype(np.float32)
        wm_alpha = (watermark_resized[:, :, 3].astype(np.float32) / 255.0 * opacity)[..., None]
    else:
        wm_rgb = watermark_resized.reshape(new_wm_h, new_wm_w, -1)[:, :, :3].astype(np.float32)
        wm_alpha = np.full((new_wm_h, new_wm_w, 1), float(opacity), dtype=np.float32)

    prepared = (wm_rgb * wm_alpha, 1.0 - wm_alpha)

    if len(_WATERMARK_CACHE) >= _CACHE_LIMIT:
        _WATERMARK_CACHE.clear()
    _WATERMARK_CACHE[key] = prepared

    return prepared


def _watermark_placement(image_shape, watermark_path, position, scale):
    h, w = image_shape[:2]

    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if watermark is None:
        return None
    wm_h, wm_w = watermark.shape[:2]

    new_wm_w = int(w * scale)
    new_wm_h = int(wm_h * new_wm_w / wm_w)
    if new_wm_w <= 0 or new_wm_h <= 0:
        return None

    margin = 20
    if position == "bottom_left":
        x, y = margin, h - new_wm_h - margin
    elif position == "top_right":
        x, y = w - new_wm_w - margin, margin
    elif position == "top_left":
        x, y = margin, margin
    else:  # bottom_right (default)
        x, y = w - new_wm_w - margin, h - new_wm_h - margin

    x = max(0, min(x, w - new_wm_w))
    y = max(0, min(y, h - new_wm_h))
    return x, y, new_wm_w, new_wm_h


# (path, mtime, frame_h, frame_w, position, scale) -> placement
_PLACEMENT_CACHE = {}


def _cached_placement(image, watermark_path, position, scale):
    try:
        mtime = os.path.getmtime(watermark_path)
    except OSError:
        return None
    key = (watermark_path, mtime, image.shape[0], image.shape[1], position, float(scale))
    if key not in _PLACEMENT_CACHE:
        if len(_PLACEMENT_CACHE) >= _CACHE_LIMIT:
            _PLACEMENT_CACHE.clear()
        _PLACEMENT_CACHE[key] = _watermark_placement(image.shape, watermark_path, position, scale)
    return _PLACEMENT_CACHE[key]


def add_image_watermark(image, watermark_path, position="bottom_right", opacity=0.2, scale=0.3, inplace=False):
    if image is None or not os.path.exists(watermark_path):
        return image

    placement = _cached_placement(image, watermark_path, position, scale)
    if placement is None:
        return image
    x, y, new_wm_w, new_wm_h = placement

    prepared = _prepare_watermark(watermark_path, new_wm_w, new_wm_h, opacity)
    if prepared is None:
        return image
    wm_premultiplied, inv_alpha = prepared

    result = image if inplace else image.copy()
    roi = result[y:y + new_wm_h, x:x + new_wm_w]
    blended = roi.astype(np.float32) * inv_alpha + wm_premultiplied
    result[y:y + new_wm_h, x:x + new_wm_w] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def add_watermark_to_frame_list(frame_list, watermark_path, opacity=0.2):
    if not frame_list or not os.path.exists(watermark_path):
        return frame_list

    return [add_image_watermark(frame, watermark_path, opacity=opacity) for frame in frame_list]
