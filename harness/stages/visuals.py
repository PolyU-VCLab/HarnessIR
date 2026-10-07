#!/usr/bin/env python3
"""Diagnostic picture pack for the composer, plus the text evidence rendering.

The composer LOOKS at the semantic map and the depth map; it must WRITE as though
they never existed, because the executor only ever receives the photograph and
the woven text. That asymmetry is enforced in the prompt (compose.txt PART A),
not here - this module only assembles what the composer sees.

Two rules are load-bearing:

* Every diagnostic map is resampled to the exact size of the photograph. The
  composer is told the maps are pixel-aligned, and it writes regional directives
  on that basis ("the building on the left, the green region"). A map at a
  different aspect ratio silently shifts every region it names.
* Label ids are resampled with NEAREST. Bilinear interpolation between label 3
  and label 7 produces label 5 - a region that does not exist, in a colour the
  legend will happily explain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..perception.segment import seg_palette

ROLE_PHOTO = "photo"
ROLE_SEG = "seg"
ROLE_DEPTH = "depth"

# Only the largest regions reach the semantic legend. A long legend buries the
# two or three regions a restoration prompt can actually act on.
# Note: the OCR and face evidence has no such cap - every text region and every
# face is listed, because those are what a per-region directive has to name.
SEG_LEGEND_MAX = 6

_PHOTO_CAPTION = ("Picture 1 - THE DEGRADED PHOTOGRAPH. This is the image to be "
                  "restored. Every following picture is a diagnostic map of this "
                  "one, not a photograph.")


@dataclass
class VisualEntry:
    role: str
    path: str
    caption: str


@dataclass
class VisualPack:
    entries: list[VisualEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [e.path for e in self.entries]

    @property
    def n_aux(self) -> int:
        """Number of diagnostic maps, excluding the photograph itself."""
        return max(0, len(self.entries) - 1)

    def has(self, role: str) -> bool:
        return any(e.role == role for e in self.entries)

    def captions_block(self) -> str:
        """Numbered captions, matching the order the images are sent in."""
        return "\n".join(f"Picture {i}: {e.caption}" if i > 1 else e.caption
                         for i, e in enumerate(self.entries, start=1))

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for ``visual_pack.json``.

        Records what the composer was shown - the roles, in send order, and the
        notes about anything excluded. The rendered captions are not duplicated
        here; they are rebuilt by ``captions_block`` and the evidence text file
        already holds the model-facing wording.
        """
        return {
            "entries": [{"role": e.role, "path": e.path} for e in self.entries],
            "roles": [e.role for e in self.entries],
            "paths": self.paths,
            "n_aux": self.n_aux,
            "notes": self.notes,
        }


def _align_to(src: str | Path, size: tuple[int, int], dst: Path, *,
              nearest: bool) -> str:
    """Resample ``src`` to ``size`` and write it to ``dst``.

    ``nearest=True`` for label rasters, where interpolation would invent ids.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return str(src)
    if (arr.shape[1], arr.shape[0]) != size:
        arr = cv2.resize(arr, size,
                         interpolation=cv2.INTER_NEAREST if nearest
                         else cv2.INTER_LINEAR)
    cv2.imwrite(str(dst), arr)
    return str(dst)


def _bbox_phrase(bbox: Any, ratio: Any = None) -> str:
    """Describe where a region is, in words rather than coordinates."""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return ""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if (x1 - x0) > 0.8 and (y1 - y0) > 0.8:
        return "spread across most of the frame"
    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "middle")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
    phrase = f"{vert} {horiz}"
    if isinstance(ratio, (int, float)) and ratio >= 0.25:
        phrase += ", a large part of the frame"
    return phrase


def seg_legend(seg: dict[str, Any]) -> str:
    """Colour legend for the semantic map.

    Each line gives the exact RGB triple as written to disk, so the composer can
    match a colour it sees to a name. Colours come from the same deterministic
    palette the raster was painted with.
    """
    classes = [c for c in (seg.get("classes") or []) if c.get("label")]
    if not classes:
        return ""
    palette = seg_palette()
    lines = []
    for c in sorted(classes, key=lambda c: -float(c.get("ratio") or 0))[:SEG_LEGEND_MAX]:
        r, g, b = (int(v) for v in palette[int(c["id"]) % 256])
        where = _bbox_phrase(c.get("bbox"), c.get("ratio"))
        lines.append(f"  - RGB({r},{g},{b}) = {c['label']}"
                     + (f", {where}" if where else ""))
    return "\n".join(lines)


def depth_caption_tail(depth: dict[str, Any], style: str) -> str:
    """One sentence on what the depth map implies for restoration."""
    if depth.get("layered"):
        return ("the scene is clearly layered, so degradation strength may differ "
                "between the near and the far field - treat them accordingly")
    return ("depth is relatively flat, so the degradation is likely uniform "
            "across the frame")


def build_visual_pack(lq_path: str | Path, evidence: dict[str, Any] | None,
                      out_dir: str | Path, *,
                      include_seg: bool = True,
                      include_depth: bool = True,
                      depth_style: str = "gray",
                      expect_tools: bool = True) -> VisualPack:
    """Assemble the photograph plus whatever diagnostic maps are usable.

    ``depth_style`` selects the depth encoding: ``"gray"`` sends ``depth.png``
    (near = white, single channel) and ``"turbo"`` sends the false-colour
    preview. Grey is the default because a false-colour map is easier to mistake
    for content, and its colours have been seen bleeding into an output.

    ``expect_tools=False`` means this configuration never intended to run tools,
    so a missing map is not an anomaly and no note is recorded. Left at True,
    the note would reach the composer as "the tools failed", contradicting a
    configuration that deliberately has none.
    """
    lq_path, out_dir = Path(lq_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(lq_path) as im:
        ref_size = im.size

    pack = VisualPack()
    pack.entries.append(VisualEntry(ROLE_PHOTO, str(lq_path), _PHOTO_CAPTION))
    ev = evidence or {}

    seg = ev.get("segmentation") if include_seg else None
    if include_seg and not seg:
        if expect_tools:
            pack.notes.append("segmentation: tool did not run or returned nothing")
    elif seg and not seg.get("reliable", True):
        pack.notes.append(f"segmentation: excluded, marked unreliable "
                          f"({seg.get('unreliable_reason', '')})")
    elif seg:
        legend = seg_legend(seg)
        src = seg.get("color_path")
        if not legend:
            pack.notes.append("segmentation: excluded, no class survived the area filter")
        elif not (src and Path(src).is_file()):
            pack.notes.append(f"segmentation: excluded, colour map missing ({src})")
        else:
            path = _align_to(src, ref_size, out_dir / "seg" / "aligned.png",
                             nearest=True)
            pack.entries.append(VisualEntry(ROLE_SEG, path, (
                "SEMANTIC MAP of Picture 1 - a diagnostic overlay, NOT a photograph. "
                "Each flat colour marks one semantic region, pixel-aligned with "
                "Picture 1:\n" + legend)))

    depth = ev.get("depth") if include_depth else None
    if include_depth and not depth:
        if expect_tools:
            pack.notes.append("depth: tool did not run or returned nothing")
    elif depth:
        src = depth.get("vis_path") if depth_style == "turbo" else depth.get("path")
        if not (src and Path(src).is_file()):
            pack.notes.append(f"depth: excluded, map missing ({src})")
        else:
            path = _align_to(src, ref_size, out_dir / "depth" / "aligned.png",
                             nearest=False)
            scale = ("a turbo false-colour scale: red/warm = nearest to the camera, "
                     "blue/cool = farthest away" if depth_style == "turbo" else
                     "a greyscale scale: brighter = nearer the camera, darker = "
                     "farther away")
            pack.entries.append(VisualEntry(ROLE_DEPTH, path, (
                "DEPTH MAP of Picture 1 - a diagnostic overlay, NOT a photograph, "
                f"pixel-aligned with Picture 1 and encoded on {scale}. "
                f"Reading: {depth_caption_tail(depth, depth_style)}.")))

    return pack


def render_text_evidence(evidence: dict[str, Any] | None,
                         pack: VisualPack) -> str:
    """Text evidence for the composer.

    Division of labour with ``ToolEvidence.as_report``: anything spatial that
    already went through the picture channel is NOT re-rendered as text here, so
    the composer never reads two differently-worded accounts of one fact. Faces stay
    in text because "keep this identity" is a linguistic constraint, not a spatial
    one.

    OCR entries carry their location and confidence alongside the string. The
    string alone says what the characters are; the box says where to look, which
    is what turns "the text must survive" into something a per-region directive
    can act on, and the confidence says how far to trust the reading - a low-
    confidence string must be preserved as it stands rather than "corrected" into
    what the model guesses it should have said.

    Nothing is capped. An image with many text regions or many faces gets all of
    them listed; the prompt is long already and truncating the evidence would
    hide exactly the regions the prompt needs to name.
    """
    ev = evidence or {}
    lines: list[str] = []

    ocr = ev.get("ocr")
    if ocr and ocr.get("texts"):
        texts = [t for t in ocr["texts"] if str(t.get("string", "")).strip()]
        if texts:
            lines.append(f"TEXT read from the image by OCR ({ocr.get('engine', 'ocr')}):")
            for t in texts:
                string = str(t.get("string", "")).strip()
                conf = t.get("confidence")
                conf_s = (f", confidence {float(conf):.2f}"
                          if isinstance(conf, (int, float)) else "")
                where = _bbox_phrase(t.get("bbox_norm"))
                where_s = f", {where}" if where else ""
                lines.append(f"  - \"{string}\"{where_s}{conf_s}")
            lines.append("  These are the characters actually present. They must "
                         "survive restoration unchanged. Where confidence is low, "
                         "preserve the characters exactly as they stand - do not "
                         "'correct' a reading you are unsure of.")

    faces = ev.get("faces")
    if faces:
        lines.append(f"FACES: {faces.get('n_detected', 0)} detected")
        for c in (faces.get("faces") or []):
            px = c.get("face_px", [0, 0])
            flag = "" if c.get("upscale_safe", True) else (
                "  [too small to repair reliably - recover it conservatively, "
                "inventing no new facial features]")
            lines.append(f"  - {c.get('bbox_desc', '?')} of the frame, "
                         f"{px[0]}x{px[1]}px, upscale {c.get('upscale', '?')}x{flag}")

    # Semantic and depth evidence falls back to text ONLY when it could not be
    # sent as a picture.
    seg = ev.get("segmentation")
    if seg and not pack.has(ROLE_SEG) and seg.get("classes"):
        lines.append("SEMANTIC REGIONS (map unavailable as a picture, text only):")
        for c in seg["classes"]:
            where = _bbox_phrase(c.get("bbox"), c.get("ratio"))
            lines.append(f"  - {c.get('label')}" + (f": {where}" if where else ""))

    depth = ev.get("depth")
    if depth and not pack.has(ROLE_DEPTH):
        lines.append(f"DEPTH (map unavailable as a picture, text only): "
                     f"layered={depth.get('layered')}, "
                     f"near_ratio={depth.get('near_ratio')}, "
                     f"far_ratio={depth.get('far_ratio')}")

    if ev.get("unavailable"):
        lines.append(f"UNAVAILABLE TOOLS: {', '.join(ev['unavailable'])}")
    if pack.notes:
        lines.append(f"DIAGNOSTIC PICTURES NOT SUPPLIED: {'; '.join(pack.notes)}")

    return "\n".join(lines) if lines \
        else "(no auxiliary evidence available for this image)"
