#!/usr/bin/env python3
"""Image quality metrics: no-reference and full-reference.

    NR (result only):            MANIQA  CLIP-IQA  MUSIQ  TOPIQ  AFINE-NR
    FR (result against GT):      PSNR  SSIM  LPIPS  DISTS

NR and FR are separate because they answer different questions and are not
interchangeable. NR says how good the image looks on its own, and it cannot see
fidelity at all - a confidently hallucinated result scores HIGHER than a faithful
one, because it is sharper and cleaner. FR says how close the result came to the
ground truth, which is the only way to see fabrication, and it needs paired data.

Metrics are recorded, never used to accept or reject a round. That decision
belongs to the verifier; mixing the two would let the metric set silently change
the pipeline's behaviour.

Measured models are created once per process and cached, because loading the
weights is far more expensive than the inference. A cached model must not be
called from two threads at once - several of these implementations keep mutable
state inside ``forward``, and concurrent use makes different images receive
identical scores. That failure looks like a plausible result, which is what
makes it worth a lock rather than a comment.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

NR_METRIC_ALIASES: dict[str, str] = {
    "MANIQA": "maniqa-pipal",
    "CLIPIQA": "clipiqa",
    "CLIPIQA+": "clipiqa+",
    "MUSIQ": "musiq",
    "TOPIQ": "topiq_nr",
    "NIQE": "niqe",
    "AFINE_NR": "afine_nr",
}
DEFAULT_NR_METRICS = ["MANIQA", "CLIPIQA", "MUSIQ", "TOPIQ", "AFINE_NR"]
DEFAULT_FR_METRICS = ["PSNR", "SSIM", "LPIPS", "DISTS"]

# Direction of each metric, for report ordering. AFINE_NR is None because its
# direction is not established, and a metric with an unknown sign must not enter
# any composite score.
HIGHER_IS_BETTER: dict[str, bool | None] = {
    "MANIQA": True, "CLIPIQA": True, "CLIPIQA+": True, "MUSIQ": True,
    "TOPIQ": True, "NIQE": False,
    "AFINE_NR": None,
    "PSNR": True, "SSIM": True, "LPIPS": False, "DISTS": False,
}

# Ground-truth alignment.
#
# Restoration methods crop their input to a multiple of their network's
# downsampling factor (8 or 16) and anchor that crop at the top-left, so results
# come back a few pixels smaller than the ground truth. Cropping the ground
# truth at the same anchor aligns them without any resampling.
#
# Anything beyond this tolerance is a harness-level mismatch, not a rounding
# difference, and is raised rather than patched: a result at a quarter of the
# ground-truth size means the input was never upscaled to the ground-truth
# resolution before restoration. Resampling it here would produce a plausible
# and meaningless PSNR.
_ALIGN_TOL = 32

# AFINE-NR extracts CLIP ViT-B/32 features and its forward asserts that both
# dimensions are multiples of 32. Crop at the top-left to satisfy it, never
# centre-crop and never resize: the outputs of different methods differ in size
# by a few pixels, so a centre crop would give each method a different pixel
# region and the scores would stop being comparable. Resizing would change the
# sharpness the metric is measuring.
_AFINE_MULTIPLE = 32

_MODEL_CACHE: dict[str, Any] = {}
_METRIC_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_metric(name: str):
    """Create or fetch a metric. Returns ``None`` when it cannot be built.

    The whole construction holds ``_CACHE_LOCK``: without it, two threads
    calling the same metric for the first time both build it, which loads the
    weights twice and can exhaust memory on a shared GPU.
    """
    with _CACHE_LOCK:
        if name in _MODEL_CACHE:
            return _MODEL_CACHE[name]
        import pyiqa

        device = _device()
        try:
            if name in ("PSNR", "SSIM"):
                # Both on the Y channel in YCbCr, which is the convention the
                # reference numbers were produced under.
                metric = pyiqa.create_metric(name.lower(), test_y_channel=True,
                                             color_space="ycbcr").to(device)
            elif name in ("LPIPS", "DISTS"):
                metric = pyiqa.create_metric(name.lower(), device=device)
            else:
                metric = pyiqa.create_metric(NR_METRIC_ALIASES[name],
                                             device=device)
        except Exception as e:                # noqa: BLE001 - soft degradation
            print(f"[iqa] {name} unavailable, skipping: "
                  f"{type(e).__name__}: {str(e)[:80]}")
            metric = None
        _MODEL_CACHE[name] = metric
        _METRIC_LOCKS[name] = threading.Lock()
        return metric


def _to_tensor(path: str | Path):
    import numpy as np
    import torch
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(_device())


def _align_pair(x, gt):
    """Align result and ground truth by cropping at the top-left."""
    xh, xw = x.shape[-2:]
    gh, gw = gt.shape[-2:]
    if (xh, xw) == (gh, gw):
        return x, gt
    if 0 <= gh - xh <= _ALIGN_TOL and 0 <= gw - xw <= _ALIGN_TOL:
        return x, gt[..., :xh, :xw]
    if 0 <= xh - gh <= _ALIGN_TOL and 0 <= xw - gw <= _ALIGN_TOL:
        return x[..., :gh, :gw], gt
    raise ValueError(
        f"result and ground truth differ in resolution by more than the "
        f"{_ALIGN_TOL}px alignment tolerance: GT {(gh, gw)} vs result "
        f"{(xh, xw)}. This is not aligned by cropping, so it is a data "
        f"preparation problem - upscale the input to the ground-truth "
        f"resolution before restoration rather than rescaling at scoring time.")


def _crop_to_multiple(x, multiple: int):
    """Crop the top-left to a multiple of ``multiple``; ``None`` if too small."""
    h, w = x.shape[-2:]
    nh, nw = (h // multiple) * multiple, (w // multiple) * multiple
    if nh < multiple or nw < multiple:
        return None
    if (nh, nw) == (h, w):
        return x
    return x[..., :nh, :nw]


def compute_iqa(result_path: str | Path, gt_path: str | Path | None = None, *,
                nr_metrics: list[str] | None = None,
                fr_metrics: list[str] | None = None) -> dict[str, float]:
    """Score one result. FR metrics run only when a ground truth is supplied.

    Metrics that cannot run are omitted from the result rather than set to
    ``None``, so averaging never has to filter.
    """
    import torch

    nr_metrics = DEFAULT_NR_METRICS if nr_metrics is None else nr_metrics
    fr_metrics = DEFAULT_FR_METRICS if fr_metrics is None else fr_metrics
    out: dict[str, float] = {}

    try:
        x = _to_tensor(result_path)
    except Exception as e:                    # noqa: BLE001
        print(f"[iqa] cannot read {result_path}: {e}")
        return out

    with torch.no_grad():
        for name in nr_metrics:
            metric = _get_metric(name)
            if metric is None:
                continue
            xi = x
            if name.startswith("AFINE"):
                xi = _crop_to_multiple(x, _AFINE_MULTIPLE)
                if xi is None:
                    print(f"[iqa] {name} skipped: {tuple(x.shape[-2:])} is "
                          f"below {_AFINE_MULTIPLE}px ({result_path})")
                    continue
            try:
                # A clone per metric: some implementations preprocess in place
                # and would corrupt the input for the next metric. The lock
                # serialises calls into one model instance.
                with _METRIC_LOCKS[name]:
                    out[name] = round(float(metric(xi.clone()).item()), 4)
            except Exception as e:            # noqa: BLE001
                print(f"[iqa] {name} failed: {type(e).__name__}: {str(e)[:60]}")

        if gt_path and Path(gt_path).is_file():
            try:
                gt = _to_tensor(gt_path)
                x_fr, gt = _align_pair(x, gt)
                for name in fr_metrics:
                    metric = _get_metric(name)
                    if metric is None:
                        continue
                    xi, gi = x_fr, gt
                    if name.startswith("AFINE"):
                        xi = _crop_to_multiple(x_fr, _AFINE_MULTIPLE)
                        gi = _crop_to_multiple(gt, _AFINE_MULTIPLE)
                        if xi is None or gi is None:
                            continue
                    try:
                        with _METRIC_LOCKS[name]:
                            out[name] = round(
                                float(metric(xi.clone(), gi.clone()).item()), 4)
                    except Exception as e:    # noqa: BLE001
                        print(f"[iqa] {name} failed: {type(e).__name__}: "
                              f"{str(e)[:60]}")
            except Exception as e:            # noqa: BLE001
                print(f"[iqa] alignment or read failed for {gt_path}: {e}")
    return out


def format_iqa(metrics: dict[str, float]) -> str:
    """Render metrics on one line, in a fixed order."""
    order = DEFAULT_NR_METRICS + DEFAULT_FR_METRICS
    parts = []
    for name in order:
        if name not in metrics:
            continue
        direction = HIGHER_IS_BETTER.get(name)
        arrow = "" if direction is None else ("up" if direction else "down")
        parts.append(f"{name}{arrow} {metrics[name]:.4f}")
    return " | ".join(parts) if parts else "(no metrics)"
