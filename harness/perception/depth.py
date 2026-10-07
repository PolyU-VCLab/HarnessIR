#!/usr/bin/env python3
"""T3 DEPTH - monocular depth (Depth-Anything-V2-Base).

Produces an 8-bit depth map (near = white) plus a turbo-colormap preview, and
three scalars the composer actually consumes: how much of the frame is near, how
much is far, and whether the scene is layered at all. A layered scene is one
where degradation strength plausibly varies with distance, which is what lets
the composer write "sharpen the foreground, leave the distant haze alone".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ._models import depth_pipeline

# Normalised-depth cuts for the near / far masses.
NEAR_CUT = 0.66
FAR_CUT = 0.33
# A scene counts as layered only when both ends carry real mass. The near
# threshold is lower than the far one because a small, close subject against a
# deep background is the common layered case.
LAYERED_NEAR_MIN = 0.08
LAYERED_FAR_MIN = 0.15


@dataclass
class DepthOut:
    path: str                 # 8-bit depth map, near = white
    vis_path: str             # turbo colormap preview (for human inspection)
    near_ratio: float         # fraction of pixels with normalised depth > NEAR_CUT
    far_ratio: float          # fraction with normalised depth < FAR_CUT
    layered: bool             # clear foreground / background separation


def run_depth(img: Image.Image, out_dir: Path) -> DepthOut | None:
    """Estimate depth for ``img``, writing artifacts into ``out_dir``."""
    try:
        depth = depth_pipeline()(img)["depth"]
    except Exception as e:                   # noqa: BLE001 - soft degradation
        print(f"    T3 DEPTH skipped: {type(e).__name__}: {e}", flush=True)
        return None

    arr = np.asarray(depth).astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    u8 = (arr * 255).astype(np.uint8)
    if u8.shape[:2] != (img.height, img.width):
        u8 = cv2.resize(u8, (img.width, img.height), interpolation=cv2.INTER_LINEAR)
        arr = u8.astype(np.float32) / 255.0

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "depth.png"
    Image.fromarray(u8).save(path)
    vis = out_dir / "depth_vis.jpg"
    cv2.imwrite(str(vis), cv2.applyColorMap(u8, cv2.COLORMAP_TURBO),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    near = float((arr > NEAR_CUT).mean())
    far = float((arr < FAR_CUT).mean())
    return DepthOut(
        path=str(path),
        vis_path=str(vis),
        near_ratio=round(near, 4),
        far_ratio=round(far, 4),
        layered=bool(near > LAYERED_NEAR_MIN and far > LAYERED_FAR_MIN),
    )
