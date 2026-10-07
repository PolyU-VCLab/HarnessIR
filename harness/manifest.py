#!/usr/bin/env python3
"""Manifest loading and sample-id derivation.

A manifest is a JSON array of::

    {"lq": "/abs/path/to/input.png",
     "gt": "/abs/path/to/ground_truth.png",   # optional
     "prompt": "Restore this low-quality image to a clear, clean state.",
     "type": ["haze"]}                       # optional; [] means mix

The prompt field is what the baseline path executes; the harness path ignores it
and writes its own prompt from the diagnosis.

Sample ids are derived, not read. Deriving them from the path keeps manifests
portable across machines and keeps ids stable when a file is regenerated, and
using a path rather than a bare filename matters: a benchmark can reuse
``001.png`` across several degradation folders, and keying on the stem alone
would silently collapse those into one sample.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest.

    Accepts both a JSON array and JSONL, decided by the first non-whitespace
    character rather than by the extension - ``.json`` files holding one object
    per line occur in practice.
    """
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError(f"{path}: top-level JSON is not an array")
        return items
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sample_id(lq_path: str | Path, *, lq_root: str | Path | None = None,
              manifest_dir: str | Path | None = None) -> str:
    """Derive a stable, unique id for one sample.

    The id is the input path relative to a root, without the extension, so a
    nested dataset yields ids like ``"DRealSR/LQ/Canon_40"``. Relative ids carry
    the degradation folder, which is what makes them unique - a flat
    ``"Canon_40"`` would not distinguish the same sample name reused across
    folders.

    Roots are tried in order: an explicit ``--lq-root``, then the manifest's own
    directory. When the input is not under either (a manifest listing absolute
    paths elsewhere), fall back to the last two directory levels plus the
    filename.

    The returned id is used as a relative path under the run directory, so it
    must never begin with ``/`` or an absolute path would escape the output tree.
    """
    p = Path(lq_path)
    for root in (lq_root, manifest_dir):
        if root is None:
            continue
        try:
            cand = str(p.resolve().relative_to(Path(root).resolve()).with_suffix(""))
            return cand.replace("\\", "/")
        except ValueError:
            continue
    parents = p.parent
    cand = str(Path(parents.parent.name) / parents.name / p.stem)
    return cand.replace("\\", "/")


def build_index(items: list[dict[str, Any]], *, lq_root: str | Path | None = None,
                 manifest_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Map sample id to manifest entry, rejecting duplicates.

    Duplicates are an error rather than a last-one-wins overwrite: two entries
    mapping to one id would mean one sample is silently never processed and two
    results compete for one output path.
    """
    index: dict[str, dict[str, Any]] = {}
    for item in items:
        if "lq" not in item:
            raise ValueError(f"manifest entry without an 'lq' field: {item}")
        sid = sample_id(item["lq"], lq_root=lq_root, manifest_dir=manifest_dir)
        if sid in index:
            raise ValueError(
                f"duplicate sample id {sid!r}: two manifest entries resolve to "
                f"the same id, so one of them would never be run and both would "
                f"write to the same output path")
        index[sid] = item
    return index
