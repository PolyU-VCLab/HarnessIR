#!/usr/bin/env python3
"""T2 FACE - face localisation with the evidence the composer needs.

The composer does not want coordinates; it wants to be able to write "the face in
the upper left". So every detection is reported as a position phrase, a pixel
size, and the upscale factor a dedicated repair would imply.

That last number is a gate, not a description. Past roughly eight times, a
generative editor starts inventing a plausible face rather than recovering the
one that is there, so faces beyond ``FACE_UPSCALE_SAFE_MAX`` are flagged and
left out of any regional repair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import settings
from ._models import face_analyser

# YuNet is only a fallback detector: SCRFD is noticeably steadier on small,
# occluded and degraded faces. The weights are optional - when the file is absent
# the fallback is simply skipped. See settings.YUNET_CKPT.
YUNET_PATH = Path(settings.YUNET_CKPT)
FACE_MIN_PX = 24


@dataclass
class FaceEvidence:
    n_detected: int = 0
    faces: list[dict] = field(default_factory=list)
    overlay: str | None = None


def _detect_scrfd(img_path: Path, conf: float = 0.4) -> list[tuple]:
    """SCRFD detections as normalised ``(x0, y0, x1, y1)`` boxes."""
    app = face_analyser()
    if app is None:
        return []
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    try:
        found = app.get(np.asarray(img))
    except Exception:                        # noqa: BLE001 - fall back to YuNet
        return []
    out = []
    for f in found:
        if f["det_score"] < conf:
            continue
        x0, y0, x1, y1 = (float(v) for v in f["bbox"])
        out.append((max(0.0, x0 / w), max(0.0, y0 / h),
                    min(1.0, x1 / w), min(1.0, y1 / h)))
    return out


def _detect_yunet(img_path: Path, conf: float = 0.6) -> list[tuple]:
    """YuNet detections as normalised boxes; empty when weights are missing."""
    if not YUNET_PATH.is_file():
        return []
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    try:
        det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (w, h), conf, 0.3, 5000)
        det.setInputSize((w, h))
        _, faces = det.detect(img)
    except cv2.error:
        return []
    if faces is None:
        return []
    return [(x / w, y / h, (x + bw) / w, (y + bh) / h)
            for x, y, bw, bh in (f[:4] for f in faces)]


def _iou(a: tuple, b: tuple) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area if area > 0 else 0.0


def _position_phrase(cx: float, cy: float) -> str:
    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "middle")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
    return f"{vert} {horiz}"


def run_faces(img_path: str | Path, out_dir: Path) -> FaceEvidence | None:
    """Detect faces, summarise them, and draw an overlay for inspection."""
    img_path = Path(img_path)
    with Image.open(img_path) as im:
        width, height = im.size

    boxes = _detect_scrfd(img_path)
    source = ["scrfd"] * len(boxes)
    if not boxes:
        boxes = _detect_yunet(img_path)
        source = ["yunet"] * len(boxes)

    # Merge overlapping detections from either detector.
    kept: list[tuple] = []
    kept_src: list[str] = []
    for box, src in zip(boxes, source):
        if any(_iou(box, k) > 0.4 for k in kept):
            continue
        kept.append(box)
        kept_src.append(src)

    faces: list[dict] = []
    for box, src in zip(kept, kept_src):
        bw = int((box[2] - box[0]) * width)
        bh = int((box[3] - box[1]) * height)
        if min(bw, bh) < FACE_MIN_PX:
            continue
        # Square crop side = longest face edge x (1 + 2 x 0.55); the generous
        # margin gives the editor surrounding context without letting the face
        # become a small fraction of the crop.
        side = min(int(max(bw, bh) * 2.1), min(width, height))
        upscale = round(1024 / max(side, 1), 2)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        faces.append({
            "bbox_norm": [round(v, 4) for v in box],
            "bbox_desc": _position_phrase(cx, cy),
            "face_px": [bw, bh],
            "upscale": upscale,
            "upscale_safe": upscale <= settings.FACE_UPSCALE_SAFE_MAX,
            "source": src,
        })

    # Largest first: the composer reads them in order and the big faces are the
    # ones worth naming.
    faces.sort(key=lambda f: -(f["face_px"][0] * f["face_px"][1]))

    overlay = None
    if faces:
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = cv2.imread(str(img_path))
        for f in faces:
            x0, y0, x1, y1 = f["bbox_norm"]
            cv2.rectangle(arr, (int(x0 * width), int(y0 * height)),
                          (int(x1 * width), int(y1 * height)), (0, 200, 0), 2)
        overlay_path = out_dir / "faces_overlay.jpg"
        cv2.imwrite(str(overlay_path), arr)
        overlay = str(overlay_path)

    return FaceEvidence(n_detected=len(faces), faces=faces, overlay=overlay)
