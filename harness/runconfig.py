#!/usr/bin/env python3
"""Run configuration: everything that varies between two runs of this harness.

One config, one output directory. The config is written into the run directory
as ``config.json``, so "which models, which stages, which metrics produced this
batch" stays answerable long after the run finished.

A JSON file supplies the starting point; command-line flags override individual
fields. Unknown keys in the JSON raise rather than being ignored, because a
misspelled key would otherwise silently leave the default in place and the run
would differ from what the file says it does.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import settings
from .metrics import DEFAULT_FR_METRICS, DEFAULT_NR_METRICS

DEFAULT_CONFIG = settings.REPO_DIR / "configs" / "default.json"
NAME = "default"


def _strip_comment_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys beginning with ``//`` (JSON has no comments; this is the idiom)."""
    return {k: v for k, v in data.items() if not k.startswith("//")}


@dataclass
class RunConfig:
    """Every knob for one run."""

    # -- identity -----------------------------------------------------------
    name: str = NAME
    note: str = ""

    # -- models -------------------------------------------------------------
    vlm_backend: str = settings.VLM_BACKEND
    vlm_model: str | None = None            # None -> settings.VLM_MODEL_NAME
    mfm_backend: str = settings.MFM_BACKEND
    vlm_max_output_tokens: int = settings.VLM_MAX_OUTPUT_TOKENS

    # -- which stages run ---------------------------------------------------
    enable_tools: bool = True               # stage 2
    enable_redo: bool = False               # stage 5
    max_rounds: int = 3                     # total attempts, including the first

    # -- metrics ------------------------------------------------------------
    enable_iqa: bool = True
    nr_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_NR_METRICS))
    fr_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_FR_METRICS))
    gt_map: dict[str, str] | None = None    # per-image ground truth, from the manifest

    # -- data ---------------------------------------------------------------
    images: list[str] = field(default_factory=list)
    lq_map: dict[str, str] | None = None    # per-image LQ path, from the manifest
    type_map: dict[str, str | list[str]] | None = None
    request_map: dict[str, str] | None = None   # per-image request, from the manifest
    user_intent: str | None = None

    def request_for(self, sample_id: str) -> str | None:
        """The restoration request for one sample.

        A manifest may carry a per-entry ``prompt`` - the user request for that
        image. Passing it to the harness as well as to the baseline is what makes
        the two variants comparable: both then answer the same request, and the
        only difference between them is the harness.

        ``--intent`` sets one request for the whole run and takes precedence when
        given, because it is an explicit choice at the command line.
        """
        if self.user_intent:
            return self.user_intent
        return (self.request_map or {}).get(sample_id)

    # -- execution ----------------------------------------------------------
    workers: int = 4
    out_root: str | None = None
    resume: bool = True

    # -- geometry -----------------------------------------------------------
    # Square 1024 input to the executor (see stages/execute.py).
    square1024: bool = True

    # -- post-processing ----------------------------------------------------
    # Align the delivered image's channel statistics to the input's after each
    # executor call, so every scorer sees the image that is actually shipped.
    # See harness/color_adjust.py for the per-type rules.
    color_adjust: bool = True

    # -- redo ---------------------------------------------------------------
    # Two independent choices, both only meaningful with enable_redo.
    #
    # redo_stop decides how many rounds get executed:
    #   "verifier"  stop as soon as the verifier accepts an attempt. This is the
    #               default: the loop is meant to succeed in one execution and to
    #               spend another round only on results that failed.
    #   "all"       ignore the verdict and run every round up to max_rounds. Use
    #               it to fill the candidate pool, e.g. when studying how much a
    #               second or third attempt could reach. It costs one execution
    #               per extra round for EVERY sample, not just the failing ones.
    redo_stop: str = "verifier"             # verifier | all
    #
    # redo_select decides which executed round is delivered:
    #   "verifier"  the round the verifier named, falling back to the first it
    #               accepted. Judged on the restoration requirements, with the
    #               measurements supplied as numbers - see selection.py.
    #   "ssim"      the round with the highest grayscale SSIM against the LQ
    #               input, among rounds that passed the deterministic gates.
    #               Free and deterministic, but biased towards the round that
    #               changed least: an attempt that barely touched the image
    #               scores highly on it.
    redo_select: str = "verifier"           # verifier | ssim

    # -- tool parameters ----------------------------------------------------
    depth_picture_style: str = "gray"       # gray | turbo

    # -- derived ------------------------------------------------------------

    def resolve_out_dir(self, base: Path) -> Path:
        """``<base>/<name>_<timestamp>`` unless an explicit root was given."""
        if self.out_root:
            return Path(self.out_root)
        return base / f"{self.name}_{time.strftime('%Y%m%d_%H%M%S')}"

    def gt_path_for(self, img_id: str) -> Path | None:
        """Ground truth for one image, if the manifest supplied a live path."""
        path = (self.gt_map or {}).get(img_id)
        if path and Path(path).is_file():
            return Path(path)
        return None

    def lq_path_for(self, img_id: str) -> Path | None:
        path = (self.lq_map or {}).get(img_id)
        if path and Path(path).is_file():
            return Path(path)
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "RunConfig":
        p = Path(path) if path else DEFAULT_CONFIG
        data = _strip_comment_keys(json.loads(p.read_text(encoding="utf-8")))
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"config {p} has unknown fields: {sorted(unknown)}; "
                             f"available: {sorted(known)}")
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        known = set(cls.__dataclass_fields__)
        data = _strip_comment_keys(data)
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    def summary_lines(self) -> list[str]:
        """Human-readable configuration dump, printed before the batch starts."""
        stages = ["stage 1 perception and diagnosis",
                  "stage 2 tool invocation" if self.enable_tools
                  else "stage 2 tool invocation (skip)",
                  "stage 3 prompt composition", "stage 4 execution"]
        if self.enable_redo:
            stops = ", stop on accept" if self.redo_stop == "verifier" \
                else ", run every round"
            stages += ["stage 5 verification",
                       f"stage 5 refinement (<= {self.max_rounds - 1} more)"
                       f"{stops}, deliver by {self.redo_select}"]
        else:
            stages.append("stage 5 skip (redo off)")

        metrics = "off" if not self.enable_iqa else (
            f"NR[{','.join(self.nr_metrics)}]"
            + (f" + FR[{','.join(self.fr_metrics)}]" if self.gt_map else
               " (no ground truth, FR skipped)"))

        return [
            f"run name : {self.name}" + (f"  - {self.note}" if self.note else ""),
            f"vlm      : {self.vlm_backend}" +
            (f" (model={self.vlm_model})" if self.vlm_model else ""),
            f"executor : {self.mfm_backend}",
            f"stages   : {' -> '.join(stages)}",
            f"metrics  : {metrics}",
            f"intent   : {self.user_intent or '(default restoration intent)'}",
        ]
