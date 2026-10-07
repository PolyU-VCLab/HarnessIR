#!/usr/bin/env python3
"""Stage 2 - run the tools the diagnosis asked for and assemble the evidence.

This stage makes no judgements. It executes the plan and structures the result.

Tools run SEQUENTIALLY, even though they are independent. The four tools sit on
three different GPU frameworks (insightface on onnxruntime, depth and SAM3 on
torch, OCR on paddle), each with its own stream and allocator. Overlapping them
on one device is racy, and the race is nastier than a crash: it reproduces
reliably on a quiet machine and never on a busy one, because a loaded GPU
serialises the calls for you. The cost of sequential execution is that stage 2
takes the sum of the tools instead of the slowest one.

Failures are soft everywhere. A tool that cannot run contributes ``None`` and an
entry in ``unavailable``; the composer is written to work from whatever evidence
actually arrived, so one broken model must not lose the sample.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..perception import run_depth, run_faces, run_ocr, run_segment

TOOL_OCR = "T1_OCR"
TOOL_FACE = "T2_FACE"
TOOL_DEPTH = "T3_DEPTH"
TOOL_SEG = "T4_SEGMENTATION"


@dataclass
class ToolEvidence:
    ocr: dict[str, Any] | None = None
    faces: dict[str, Any] | None = None
    depth: dict[str, Any] | None = None
    segmentation: dict[str, Any] | None = None
    unavailable: list[str] = field(default_factory=list)
    elapsed: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr": self.ocr,
            "faces": self.faces,
            "depth": self.depth,
            "segmentation": self.segmentation,
            "unavailable": self.unavailable,
            "elapsed_sec": self.elapsed,
        }

    def as_report(self) -> str:
        """Render the evidence as text for the composer.

        Only tools that actually ran appear. A tool that was never requested
        must leave no trace, or the composer starts reasoning about evidence it
        does not have - compose.txt is explicit about that.
        """
        lines: list[str] = []

        if self.ocr:
            texts = self.ocr.get("texts") or []
            if texts:
                lines.append("OCR (verified text):")
                lines += [
                    f"  - \"{t.get('string')}\" (confidence {t.get('confidence', '?')})"
                    for t in texts[:10]
                ]

        if self.faces:
            lines.append(f"FACES: {self.faces.get('n_detected', 0)} detected")
            for c in (self.faces.get("faces") or [])[:6]:
                px = c.get("face_px", [0, 0])
                flag = "" if c.get("upscale_safe", True) \
                    else "  [too small for dedicated repair]"
                lines.append(
                    f"  - at {c.get('bbox_desc', '?')}, {px[0]}x{px[1]}px, "
                    f"upscale {c.get('upscale', '?')}x{flag}")

        if self.depth:
            d = self.depth
            lines.append(f"DEPTH: layered={d.get('layered')}, "
                         f"near_ratio={d.get('near_ratio')}, "
                         f"far_ratio={d.get('far_ratio')}")
            if d.get("interpretation"):
                lines.append(f"  {d['interpretation']}")

        if self.segmentation:
            s = self.segmentation
            if not s.get("reliable", True):
                lines.append(
                    f"SEGMENTATION: UNRELIABLE ({s.get('unreliable_reason', '')}) "
                    "- do not rely on these labels")
            else:
                lines.append("SEGMENTATION (semantic regions):")
                lines += [
                    f"  - {c.get('label')}: {c.get('ratio', 0):.0%} of frame"
                    for c in (s.get("classes") or [])[:8]
                ]

        if self.unavailable:
            lines.append(f"UNAVAILABLE TOOLS: {', '.join(self.unavailable)}")

        return "\n".join(lines) if lines \
            else "(no auxiliary tools were run for this image)"


def _face_evidence(image_path: Path, out_dir: Path) -> dict[str, Any] | None:
    ev = run_faces(image_path, out_dir)
    if ev is None:
        return None
    return {
        "n_detected": ev.n_detected,
        "faces": ev.faces,
        "overlay": ev.overlay,
    }


def _ocr_evidence(image_path: Path, out_dir: Path) -> dict[str, Any] | None:
    with Image.open(image_path) as im:
        out = run_ocr(im.convert("RGB"), out_dir)
    if out is None:
        return None
    return {"engine": out.engine, "n_texts": out.n_texts, "texts": out.texts[:30]}


def _depth_evidence(image_path: Path, out_dir: Path) -> dict[str, Any] | None:
    with Image.open(image_path) as im:
        d = run_depth(im.convert("RGB"), out_dir)
    if d is None:
        return None
    interpretation = (
        "clear foreground/background separation - degradation may differ by distance"
        if d.layered else
        "relatively flat depth - degradation likely uniform")
    return {
        "path": d.path,
        "vis_path": d.vis_path,
        "near_ratio": d.near_ratio,
        "far_ratio": d.far_ratio,
        "layered": d.layered,
        "interpretation": interpretation,
    }


def _seg_evidence(image_path: Path, out_dir: Path) -> dict[str, Any] | None:
    with Image.open(image_path) as im:
        s = run_segment(im.convert("RGB"), out_dir)
    if s is None:
        return None
    return {
        "ids_path": s.ids_path,
        "color_path": s.color_path,
        "classes": s.classes,
        "reliable": s.reliable,
        "unreliable_reason": s.unreliable_reason,
    }


_RUNNERS = {
    TOOL_FACE: _face_evidence,
    TOOL_DEPTH: _depth_evidence,
    TOOL_SEG: _seg_evidence,
    TOOL_OCR: _ocr_evidence,
}


def run_tools(image_path: str | Path, requested: list[str],
              out_dir: str | Path) -> ToolEvidence:
    """Run the requested tools in sequence and return the structured evidence."""
    image_path, out_dir = Path(image_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = ToolEvidence()
    if not requested:
        return evidence

    # Keep the stage-1 order regardless of the order diagnosis listed them in.
    jobs = [t for t in _RUNNERS if t in requested]

    for name in jobs:
        t0 = time.perf_counter()
        try:
            result = _RUNNERS[name](image_path, out_dir)
        except Exception as e:                # noqa: BLE001 - one tool cannot
            evidence.elapsed[name] = round(time.perf_counter() - t0, 1)
            evidence.unavailable.append(f"{name}: {type(e).__name__}: {e}")
            continue
        evidence.elapsed[name] = round(time.perf_counter() - t0, 1)
        if result is None:
            evidence.unavailable.append(f"{name}: unavailable")
            continue
        if name == TOOL_FACE:
            evidence.faces = result
        elif name == TOOL_DEPTH:
            evidence.depth = result
        elif name == TOOL_SEG:
            evidence.segmentation = result
        elif name == TOOL_OCR:
            evidence.ocr = result

    return evidence
