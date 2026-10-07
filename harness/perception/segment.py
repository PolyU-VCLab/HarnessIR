#!/usr/bin/env python3
"""T4 SEGMENTATION - SAM3 open-vocabulary semantic segmentation.

Text-prompted: the model is asked for a fixed vocabulary of scene labels and
returns one merged mask per label it finds. Two artifacts are produced:

* ``seg_ids.png`` - a 16-bit label raster (0 = unassigned). This is the map the
  contamination gate compares against.
* ``seg_color.png`` - the same raster through a fixed palette, for humans and
  for the composer's legend.

The palette is regenerated deterministically from ``default_rng(0)`` so that a
label id always has the same colour, across images and across runs. Everything
that needs the palette (the legend, the padding colour, the contamination gate)
derives it from ``seg_palette()`` rather than hard-coding RGB values.

The raster is written with ``cv2.imwrite``, which stores BGR, so the bytes on
disk are the RGB palette values - that is what ``seg_palette()`` returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import settings
from ._models import sam3_predictor

# Classes whose visible area is below this are dropped from the summary. They
# are offcuts of a larger region, not regions in their own right.
SEG_MIN_RATIO = 0.005


@dataclass
class SegOut:
    color_path: str
    ids_path: str
    classes: list[dict] = field(default_factory=list)   # [{label, id, ratio, bbox}]
    reliable: bool = True
    unreliable_reason: str = ""


def seg_palette() -> np.ndarray:
    """Deterministic ``(256, 3)`` RGB palette indexed by label id.

    Must stay byte-identical to the palette used when writing ``seg_color.png``;
    the legend quotes exact RGB triples and a mismatch would point the composer at
    a colour that is not on the map.
    """
    rng = np.random.default_rng(0)
    # Values start at 40 so that id 0 (background) is never near-black, which
    # would be indistinguishable from unpainted pixels.
    return rng.integers(40, 255, size=(256, 3), dtype=np.uint8)


def _labels_for_masks(result, names: list[str], n_masks: int) -> list[str]:
    """Map each mask to a class label.

    SAM3's semantic path does not return masks in the same order as boxes, so
    ``boxes.cls`` cannot be indexed directly. Match each mask to the box with
    the highest area-normalised IoU and take that box's label.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or n_masks == 0:
        return ["unknown"] * n_masks
    md = result.masks.data.cpu().numpy().astype(bool)
    out: list[str] = []
    for i in range(n_masks):
        m = md[i]
        area = float(m.sum()) or 1.0
        best, best_iou = 0, -1.0
        for j, b in enumerate(boxes.xyxy.cpu().numpy()):
            x0, y0, x1, y1 = (int(v) for v in b)
            sub = m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            if sub.size == 0:
                continue
            iou = float(sub.sum()) / area
            if iou > best_iou:
                best, best_iou = j, iou
        cls_id = int(boxes.cls[best].item()) if hasattr(boxes, "cls") else 0
        out.append(names[cls_id] if 0 <= cls_id < len(names) else "unknown")
    return out


def run_segment(img: Image.Image, out_dir: Path,
                labels: list[str] | None = None) -> SegOut | None:
    """Segment ``img`` into the requested vocabulary and write both rasters."""
    try:
        predictor = sam3_predictor()
        labels = list(labels or settings.SAM3_LABELS)
        if predictor.model.names != labels:
            predictor.model.set_classes(labels)
        result = predictor(np.asarray(img))[0]
    except Exception as e:                    # noqa: BLE001 - soft degradation
        print(f"    T4 SEGMENTATION skipped: {type(e).__name__}: {e}", flush=True)
        return None

    ids = np.zeros((img.height, img.width), dtype=np.uint16)
    classes: list[dict] = []
    reliable, reason = True, ""

    masks = getattr(result, "masks", None)
    if masks is None or masks.data.shape[0] == 0:
        reliable = False
        reason = "SAM3 found no region for any label in the vocabulary"
    else:
        md = masks.data.cpu().numpy().astype(bool)
        names = list(getattr(result, "names", [])) or labels
        per_mask = _labels_for_masks(result, names, md.shape[0])

        # One merged mask per label: instances of the same class are a single
        # region for our purposes.
        by_label: dict[str, list[np.ndarray]] = {}
        for label, m in zip(per_mask, md):
            by_label.setdefault(label, []).append(m)

        for i, (label, ms) in enumerate(by_label.items(), start=1):
            union = np.logical_or.reduce(ms)
            ids[union] = i
            ratio = float(union.mean())
            if ratio < SEG_MIN_RATIO:
                continue
            ys, xs = np.where(union)
            classes.append({
                "label": label, "id": i, "ratio": round(ratio, 4),
                "bbox": [round(float(xs.min() / img.width), 3),
                         round(float(ys.min() / img.height), 3),
                         round(float(xs.max() / img.width), 3),
                         round(float(ys.max() / img.height), 3)],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / "seg_ids.png"
    cv2.imwrite(str(ids_path), ids)

    palette = seg_palette()
    color = palette[np.clip(ids, 0, 255)]
    color_path = out_dir / "seg_color.png"
    cv2.imwrite(str(color_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    from ..utils import write_json
    write_json(out_dir / "seg_summary.json",
               {"reliable": reliable, "reason": reason, "classes": classes})

    return SegOut(color_path=str(color_path), ids_path=str(ids_path),
                  classes=classes, reliable=reliable, unreliable_reason=reason)
