#!/usr/bin/env python3
"""Stage 1 - perception and diagnosis: read the image, name the degradations,
plan the tools.

One VLM call. It returns a content description, a ranked degradation list, the
fidelity-sensitive regions it noticed, and which auxiliary models it wants run.

The tool plan is chosen by the VLM from what it sees, not by a rule keyed on the
task type. A fixed rule cannot tell a hazy street with a readable sign from a
hazy mountain with none, and the second does not need OCR.

Lightweight classical statistics (noise sigma, Laplacian variance, mean
brightness) are measured locally and passed in. They give the model an order of
magnitude to anchor on; they never decide anything by themselves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils import load_prompt

# Tool ids, in the order the stage-1 scheduler runs them.
ALL_TOOLS = ["T1_OCR", "T2_FACE", "T3_DEPTH", "T4_SEGMENTATION"]


def estimate_degradation_stats(image_path: str | Path) -> dict[str, float]:
    """Noise level, sharpness and brightness, measured classically."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}
    f = img.astype(np.float64)
    # Robust sigma of the high-frequency residual (MAD, scaled to match a
    # Gaussian standard deviation). Less distracted by texture than a plain std.
    residual = f - cv2.GaussianBlur(f, (7, 7), 0)
    sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    return {
        "noise_sigma_est": round(sigma, 2),
        "laplacian_var": round(float(cv2.Laplacian(f, cv2.CV_64F).var()), 1),
        "mean_brightness": round(float(f.mean()), 1),
    }


@dataclass
class DiagnosisResult:
    content: str = ""
    degradations: list[dict[str, Any]] = field(default_factory=list)
    sensitive: dict[str, Any] = field(default_factory=dict)
    tool_plan: list[dict[str, str]] = field(default_factory=list)
    difficulty_hint: str = ""
    difficulty_reason: str = ""
    stats: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def requested_tools(self) -> list[str]:
        """Validated, de-duplicated tool ids, in the model's requested order.

        Accepts ``T3``, ``T3_DEPTH`` and ``DEPTH`` for the same tool; the model
        uses all three spellings and occasionally repeats an entry.
        """
        seen: set[str] = set()
        out: list[str] = []
        for item in self.tool_plan:
            name = str(item.get("tool", "")).strip().upper()
            for known in ALL_TOOLS:
                short, long = known.split("_", 1)
                if name in (known, short, long) and known not in seen:
                    seen.add(known)
                    out.append(known)
                    break
        return out

    def tool_reason(self, tool: str) -> str:
        """Why the model asked for ``tool`` - recorded in the evidence file."""
        want = tool.split("_")[0]
        for item in self.tool_plan:
            if str(item.get("tool", "")).strip().upper().startswith(want):
                return str(item.get("reason", ""))
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "degradations": self.degradations,
            "sensitive": self.sensitive,
            "tool_plan": self.tool_plan,
            "requested_tools": self.requested_tools,
            "difficulty_hint": self.difficulty_hint,
            "difficulty_reason": self.difficulty_reason,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiagnosisResult":
        return cls(
            content=str(d.get("content", "")),
            degradations=list(d.get("degradations", [])),
            sensitive=dict(d.get("sensitive", {})),
            tool_plan=list(d.get("tool_plan", [])),
            difficulty_hint=str(d.get("difficulty_hint", "")),
            difficulty_reason=str(d.get("difficulty_reason", "")),
            stats=dict(d.get("stats", {})),
            raw=d,
        )


def _user_message(stats: dict[str, float], user_intent: str | None,
                  task_type: str | None) -> str:
    """Assemble the diagnosis user turn.

    The user intent is included because it changes where the model should look,
    not just what the executor is later told: "fix the sign" should make text
    detection likely, "keep the night mood" should make it describe the existing
    lighting carefully.
    """
    parts = [
        "Assess this image and plan which auxiliary models are needed.",
        f"Auxiliary measurements for this image: "
        f"{json.dumps(stats, ensure_ascii=False)}",
    ]
    if task_type:
        parts.append(f"Restoration task type: {task_type}")
    if user_intent:
        parts.append(f"User intent: {user_intent}")
        parts.append(
            "(Note: this intent influences what you should pay attention to - "
            "e.g. if the user says 'fix the text', you should prioritise text "
            "detection and OCR; if they say 'keep the night atmosphere', note "
            "the low-light conditions carefully.)")
    else:
        parts.append("No user intent provided - use the default: clarity "
                     "restoration only, preserving all other properties.")
    parts.append("Output the JSON only.")
    return "\n".join(parts)


def run_diagnosis(vlm, image_path: str | Path, *,
               user_intent: str | None = None,
               task_type: str | None = None) -> DiagnosisResult:
    """Diagnose ``image_path`` and return the plan for stage 2."""
    stats = estimate_degradation_stats(image_path)
    obj = vlm.chat_json(_user_message(stats, user_intent, task_type),
                        [str(image_path)],
                        system=load_prompt("diagnosis.txt"))
    result = DiagnosisResult.from_dict(obj)
    result.stats = stats
    result.raw = obj
    return result
