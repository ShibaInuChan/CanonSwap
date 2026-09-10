# coding: utf-8

import cv2
from .cv2_config import configure_opencv; configure_opencv()


def viz_lmk(img_, vps, **kwargs):
    """可视化点"""
    lineType = kwargs.get("lineType", cv2.LINE_8)  # cv2.LINE_AA
    img_for_viz = img_.copy()
    for pt in vps:
        cv2.circle(
            img_for_viz,
            (int(pt[0]), int(pt[1])),
            radius=kwargs.get("radius", 1),
            color=(0, 255, 0),
            thickness=kwargs.get("thickness", 1),
            lineType=lineType,
        )
    return img_for_viz
