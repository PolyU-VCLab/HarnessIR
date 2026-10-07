#!/usr/bin/env python3
"""Stage 3 - compose one restoration prompt out of the evidence.

The composer is the only stage that sees everything: the diagnosis report, the tool
evidence, and the diagnostic maps as pictures. It produces a single prompt, and
that prompt is what the executor receives.

Who sees what is the whole design, so it is worth stating plainly:

    composer    : photo + semantic map + depth map + diagnosis + tool evidence
    executor  : photo + the composed prompt. Nothing else.

The composer therefore writes regions it can locate on the maps but must describe
them in terms the executor can find on the photo alone. compose.txt PART A
states this; models still drift towards "the map shows..." so it is worth
re-reading when prompts start containing dangling references.

A truncated reply is caught here rather than downstream. The prompt is long
by design (per-region sections, several hundred words), so hitting the output
token budget is a normal failure, and a truncated reply parses into a valid JSON
object with an empty prompt. Sending that to the executor costs an image
generation to produce a picture nobody asked for.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import build_system
from ..utils import types_label
from ..utils import load_blocks, load_prompt
from .visuals import VisualPack

# The default restoration intent, injected when the user gives none.
#
# Two items from the original eight are deliberately absent: "do not change
# brightness, exposure or lighting atmosphere" and "do not correct colour casts".
# Both contradict the haze and lowlight task blocks outright - low-light
# enhancement exists to change brightness, and haze IS a colour cast - and since
# most images here carry no explicit user intent, they were injected on nearly
# every image. With both present the composer received "remove the haze" and "do
# not change colour" simultaneously, and the prohibition won: measured haze
# results came back slightly darker than the input.
#
# Colour is not left unguarded by their removal. The per-type colour ruling in
# harness/types.py states that every surface keeps its own true colour and no
# region is repainted, and the type blocks keep their own colour locks.
DEFAULT_SPEC = (
    "DEFAULT RESTORATION INTENT (applies when the user provides no explicit intent):\n"
    "1. Restore the image to a clean, clear state: remove blur, softness, noise, "
    "grain, compression artifacts, and whatever the task type names as its target "
    "degradation. Brightness, exposure and colour may change where the restoration "
    "itself requires it - see the task-specific block.\n"
    "2. Keep optical and lighting phenomena that are part of the scene: glare, lens "
    "flare, reflections, highlights and intentional bokeh must remain.\n"
    "3. Do not add, remove, move, or reshape any content.\n"
    "4. Legible text must remain exactly identical; illegible text must not be "
    "invented.\n"
    "5. Faces must keep exact identity and expression; no beautification.\n"
    "6. Target a natural clean state: no over-saturation, no over-sharpening, no "
    "stylization."
)

# Appended when the first attempt came back truncated.
_COMPACT_RETRY = (
    "\n\nIMPORTANT - YOUR PREVIOUS ATTEMPT WAS CUT OFF BEFORE IT FINISHED. "
    "The JSON was incomplete and the 'instruction' field came back empty. "
    "Produce a COMPLETE, valid JSON object this time. To fit: keep 'instruction' at "
    "the short end of the range (about 300 words), cap 'region_notes' at 4 entries, "
    "and keep every 'directive' to one short sentence. A complete instruction at 300 "
    "words is worth far more than a truncated one at 600 - the instruction field is "
    "the only part that actually reaches the editor, so it must never be empty."
)


@dataclass
class Composition:
    task_type: str = ""
    treat: list[str] = field(default_factory=list)
    preserve: list[str] = field(default_factory=list)
    prompt: str = ""
    region_notes: list[dict[str, Any]] = field(default_factory=list)
    evidence_used: list[str] = field(default_factory=list)
    truncated_retry: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_regions(self) -> int:
        return len(self.region_notes)

    @property
    def anchors(self) -> dict[str, int]:
        """Region directives counted by what each one was anchored on.

        This is the quantitative answer to "did the picture channel get used".
        The values are normalised because the prompt asks for one of
        semantic/depth/visual but models return compounds ("depth and semantic")
        and case variants; left raw, the key space is free text and never
        aggregates across a run.

        A compound counts once for EACH channel it names - one directive that
        used both maps is evidence for both, and crediting it to either alone
        understates the other. So the counts can sum to more than the number of
        regions.
        """
        out: dict[str, int] = {}
        for r in self.region_notes:
            # The model sometimes returns a region as a bare list (a positional
            # record) or as a string instead of the requested object. That is a
            # shape drift in one optional bookkeeping field, so skip the entry
            # rather than fail the sample: the prompt it produced is still valid
            # and the executor call is still worth making.
            text = r.get("anchor", "") if isinstance(r, dict) else r
            raw = str(text if text is not None else "").strip().lower()
            kinds = [k for k in ("semantic", "depth", "visual") if k in raw]
            for k in (kinds or ["unspecified"]):
                out[k] = out.get(k, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "treat": self.treat,
            "preserve": self.preserve,
            "prompt": self.prompt,
            "region_notes": self.region_notes,
            "n_regions": self.n_regions,
            "anchors": self.anchors,
            "evidence_used": self.evidence_used,
            "truncated_retry": self.truncated_retry,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Composition":
        return cls(
            task_type=str(d.get("task_type", "")),
            treat=list(d.get("treat", [])),
            preserve=list(d.get("preserve", [])),
            # The VLM contract uses "instruction" as the JSON key; the field
            # is named prompt on this side to match the paper's wording.
            prompt=str(d.get("instruction", "")),
            region_notes=list(d.get("region_notes", [])),
            evidence_used=list(d.get("evidence_used", [])),
            raw=d,
        )


def build_user_message(diagnosis: dict[str, Any], text_evidence: str,
                       pack: VisualPack, user_intent: str | None,
                       types: list[str]) -> str:
    """Assemble the composer's user turn."""
    if len(types) > 1:
        parts = [
            f"Restoration task types ({len(types)}, all apply): {types_label(types)}",
            "This image carries several degradations at once - see the conflict "
            "rules in your prompt.",
            "",
        ]
    else:
        parts = [f"Restoration task type: {types_label(types)}", ""]

    # Tell the composer what it is looking at, and - in the same breath - what the
    # executor will receive. The second half is the operative constraint: the
    # maps exist to inform the prompt, not to be referred to by it.
    parts.append("=== PICTURES YOU ARE LOOKING AT ===")
    if pack.n_aux == 0:
        parts.append("Picture 1 - the degraded photograph. No diagnostic map was "
                     "available for this image.")
    else:
        for i, e in enumerate(pack.entries, start=1):
            parts.append(f"Picture {i} - {e.caption}")
        parts.append("")
        parts.append("These maps are for YOU. The editing model will receive the "
                     "degraded photograph and your prompt, and nothing else - "
                     "it never sees these maps and cannot follow a reference to "
                     "them. Use what the maps tell you to decide where each "
                     "treatment belongs, then describe every region in terms "
                     "visible on the photograph itself.")
    parts.append("")

    parts.append("=== TRIAGE REPORT (from first-stage analysis) ===")
    parts.append(f"Content: {diagnosis.get('content', '')}")
    parts.append(f"Degradations: "
                 f"{json.dumps(diagnosis.get('degradations', []), ensure_ascii=False)}")
    parts.append(f"Sensitive elements: "
                 f"{json.dumps(diagnosis.get('sensitive', {}), ensure_ascii=False)}")
    parts.append("")

    parts.append("=== TOOL EVIDENCE (from second-stage tools) ===")
    parts.append(text_evidence)
    parts.append("")

    if user_intent:
        parts.append(f"User intent: {user_intent}")
        parts.append("(This overrides the default intent where it applies)")
    else:
        parts.append("User intent: none provided. Use the default intent below.")
    parts.append("")
    parts.append("=== RESTORATION SPECIFICATION ===")
    parts.append(DEFAULT_SPEC)
    parts.append("")
    parts.append("Now produce the restoration instruction as JSON. Output the JSON only.")
    return "\n".join(parts)


def run_compose(vlm, types: list[str], diagnosis: dict[str, Any], text_evidence: str,
              pack: VisualPack, user_intent: str | None = None) -> Composition:
    """One image-bearing VLM call: evidence in, restoration prompt out."""
    system = build_system(load_prompt("compose.txt"), types,
                          load_blocks("compose_blocks.txt"))
    user = build_user_message(diagnosis, text_evidence, pack, user_intent, types)

    def _one(msg: str) -> Composition:
        return Composition.from_dict(vlm.chat_json(msg, pack.paths, system=system))

    plan = _one(user)
    if not plan.prompt.strip():
        # Truncated. Retry once with an explicitly smaller budget before giving
        # up, because the failure is a length collision rather than a bad call.
        plan = _one(user + _COMPACT_RETRY)
        plan.truncated_retry = True

    if not plan.prompt.strip():
        raise ValueError(
            "stage 3 returned an empty prompt twice (both attempts "
            "truncated). This normally means the output token budget was hit - "
            "raise VLM_MAX_OUTPUT_TOKENS in harness/settings.py, or lower the "
            "word range in prompts/compose.txt PART E.")

    return plan
