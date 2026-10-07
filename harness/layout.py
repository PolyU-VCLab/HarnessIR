#!/usr/bin/env python3
"""Output layout for one run.

    runs/<run>/
    |-- config.json                   the configuration that produced this run
    |-- summary.json                  per-image records, aggregate stats, timings
    |-- results/<id>.png              one copy of every delivered image
    `-- per_image/<id>/
        |-- seg/      seg_ids.png seg_color.png seg_summary.json aligned.png
        |-- depth/    depth.png depth_vis.jpg aligned.png
        |-- ocr/      ocr.json
        |-- face/     faces_overlay.jpg
        |-- text/     text_evidence.txt prompt.txt
        |-- json/     diagnosis.json tool_evidence.json visual_pack.json
        |             composition.json
        |-- output.png
        `-- record.json

``results/`` holds a flat copy because every scorer here wants to iterate over
delivered images and none of them want to know about per-image directories.
``per_image/`` holds the archive: what the composer saw, what it wrote, and what
the executor was sent.

Two layouts exist because the stages produce files long before the sample is
finished, and a batch that dies halfway leaves both. ``results/`` is written
last, so its presence is the resume marker; ``per_image/`` may hold partial
work.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

RESULTS_DIRNAME = "results"
PER_IMAGE_DIRNAME = "per_image"

DIR_SEG = "seg"
DIR_DEPTH = "depth"
DIR_OCR = "ocr"
DIR_FACE = "face"
DIR_TEXT = "text"
DIR_JSON = "json"
SUBDIRS = (DIR_SEG, DIR_DEPTH, DIR_OCR, DIR_FACE, DIR_TEXT, DIR_JSON)

# Tool artifact filename -> subdirectory. Listed explicitly rather than matched
# by pattern: a glob would eventually move a delivered image too.
ARCHIVE_MAP: dict[str, str] = {
    "seg_ids.png": DIR_SEG,
    "seg_color.png": DIR_SEG,
    "seg_summary.json": DIR_SEG,
    "depth.png": DIR_DEPTH,
    "depth_vis.jpg": DIR_DEPTH,
    "ocr.json": DIR_OCR,
    "_ocr_input.png": DIR_OCR,
    "faces_overlay.jpg": DIR_FACE,
}

# Evidence fields holding absolute paths, and where that file ends up.
EVIDENCE_PATH_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("segmentation", "ids_path", DIR_SEG),
    ("segmentation", "color_path", DIR_SEG),
    ("depth", "path", DIR_DEPTH),
    ("depth", "vis_path", DIR_DEPTH),
    ("faces", "overlay", DIR_FACE),
)


def ensure_layout(out_dir: Path) -> None:
    """Create the per-image skeleton.

    Empty directories are created deliberately: when the semantic map is missing
    because the segmentation model ran out of memory, an empty ``seg/`` says
    "something belongs here" in a way a missing directory does not.
    """
    for sub in SUBDIRS:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def archive_tool_outputs(out_dir: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Move stage 2 artifacts into their subdirectories and fix the evidence paths.

    Moving a file without rewriting the path that points at it makes the
    downstream consumer fail silently - the map simply never reaches the pack,
    and the only symptom is a smaller picture count. So the rewriter below is
    part of the move, not an afterthought.

    Idempotent: files already in place are left alone.
    """
    out_dir = Path(out_dir)
    ensure_layout(out_dir)

    moved: dict[str, str] = {}
    for name, sub in ARCHIVE_MAP.items():
        src = out_dir / name
        if not src.is_file():
            continue
        dst = out_dir / sub / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved[name] = f"{sub}/{name}"

    for key, field_name, sub in EVIDENCE_PATH_FIELDS:
        node = evidence.get(key)
        if not isinstance(node, dict):
            continue
        old = node.get(field_name)
        if not old:
            continue
        new = out_dir / sub / Path(str(old)).name
        if new.is_file():
            node[field_name] = str(new)

    if moved:
        evidence["_archived"] = moved
    return evidence


def collect_result(final_output: str | Path | None, results_dir: Path,
                   img_id: str) -> str | None:
    """Copy the delivered image to ``results/<id>.png``.

    A copy rather than a move: ``per_image/<id>/`` keeps the original, which is
    the sample's full archive, and the record's ``final_output`` still points at
    it. ``results/`` is only the flat view.

    ``img_id`` may contain slashes, in which case the same nesting is created
    under ``results/``.
    """
    if not final_output:
        return None
    src = Path(final_output)
    if not src.is_file():
        return None
    dst = Path(results_dir) / f"{img_id}{src.suffix or '.png'}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def result_exists(results_dir: Path, img_id: str) -> bool:
    """Whether this sample already has a delivered image (the resume test).

    Only ``results/`` counts. It is written at the very end of a sample, whereas
    ``per_image/`` accumulates partial work, so its presence means the sample
    finished.
    """
    base = Path(results_dir) / img_id
    return any(base.with_suffix(ext).is_file() for ext in (".png", ".jpg", ".jpeg"))


def text_path(out_dir: Path, name: str) -> Path:
    p = Path(out_dir) / DIR_TEXT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def json_path(out_dir: Path, name: str) -> Path:
    p = Path(out_dir) / DIR_JSON / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
