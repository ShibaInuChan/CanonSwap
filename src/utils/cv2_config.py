# coding: utf-8

"""
Central place for the OpenCV runtime settings used across the code base.

The original code hard-coded ``cv2.setNumThreads(0)`` in every module, which
forced every warp/resize/blend onto a single core.  Those operations run on
full-resolution frames once per video frame (cropping, paste-back, watermark),
so on a modern multi-core machine single-threading them is a large, avoidable
cost.  OpenCV's own default (use every core) is used instead; set
``CANONSWAP_CV_THREADS`` to override, e.g. ``CANONSWAP_CV_THREADS=1`` to
restore the previous single-threaded behaviour.
"""

import os

import cv2

_CONFIGURED = False


def configure_opencv():
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # The OpenCL (UMat) path is not used anywhere here and only adds
    # host<->device copies, keep it off as before.
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    n_threads = os.environ.get("CANONSWAP_CV_THREADS")
    if n_threads is None:
        return  # keep OpenCV's default thread pool (all cores)

    try:
        cv2.setNumThreads(int(n_threads))
    except (TypeError, ValueError):
        pass


configure_opencv()
