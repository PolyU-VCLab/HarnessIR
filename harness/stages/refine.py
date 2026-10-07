#!/usr/bin/env python3
"""Stage 5 - revise the prompt for the next attempt.

This is the same composer as stage 3, run again. It receives what stage 3 received
- the degraded photograph, the semantic and depth maps, the diagnosis report, the
tool evidence, the restoration specification - plus the two things stage 3 did
not have:

  * the prompt that was used last, and
  * the result that prompt produced, as a picture, together with the defects
    the review found in it.

The system prompt is ``prompts/refine.txt``, which is ``prompts/compose.txt``
with the framing changed from "write a prompt" to "revise this one against
these defects". Everything else is shared: the same per-type task blocks, the
same rules about what the prompt may reference, the same style and length
requirements. A separate rewriter persona would be a different writer, and the
point here is the same writer with more information.

Only the global path exists: the revision is applied to the prompt, and the
next round goes back to the ORIGINAL photograph. Attempts therefore never
compound - round 2 is not a restoration of round 1's output - at the cost of the
edit having to be described in words rather than applied to pixels.

The length target is derived from the previous prompt's own word count. A
fixed ceiling would compress a long, region-organised prompt into a fraction
of its length and lose exactly the regional anchoring the approach depends on,
after which "more rounds" would just mean "a vaguer prompt each round".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import build_system, types_label
from ..utils import load_blocks, load_prompt
from .visuals import VisualPack

# Appended to the user turn. This is the only piece of revision-specific
# prompt text that lives outside the prompt file, because it quotes this round's
# numbers.
_REVISION_TAIL = """\
=== WHAT WENT WRONG IN THE PREVIOUS ATTEMPT ===

The instruction above was already used once. Its result is the LAST picture you
were given; Picture 1 is still the degraded photograph that the editor will
receive again. Compare the two before you decide anything.

Defects the review found in that result:
{defects}

REVISE the instruction. Keep what worked, and fix each defect above:
1. For every defect, add an explicit corrective directive that names its location.
2. Add a negative directive stating plainly what must NOT happen again, quoting the
   defect. e.g. "Do not leave the background out of focus", "Do not alter the colour
   of the sky".
3. Do not weaken or drop the parts of the instruction that were not implicated in a
   defect.

LENGTH AND STRUCTURE - this is an edit of the previous instruction, not a fresh one:
  * The previous instruction is about {n_words} words. Yours must be at least that
    long. Being more specific about what failed makes it longer, not shorter.
  * If it is organised into per-region sections, keep EVERY section heading and every
    region it names. Modify and extend them; delete none.
  * Do not summarise, compress or generalise. A shorter, vaguer instruction is the
    single most common way this step makes the next round worse.

Now produce the revised instruction as JSON. Output the JSON only."""


def _images(lq_path: str | Path, pack: VisualPack,
            failed_output: str | Path | None) -> list[str]:
    """The pictures for this call, in the order the user turn describes them.

    Picture 1 is the degraded photograph, then the maps the composer was given, and
    LAST the result the previous prompt produced. The result goes last so
    that it reads as an addition to the stage-2 view rather than a replacement of
    Picture 1.
    """
    paths = [str(p) for p in pack.paths]
    if failed_output and Path(str(failed_output)).is_file():
        paths.append(str(failed_output))
    return paths


def build_revision_message(diagnosis: dict[str, Any], text_evidence: str,
                           pack: VisualPack, prev_prompt: str,
                           defect_summary: str, result_index: int) -> str:
    """The revision user turn.

    Structurally the same as the composer's turn - pictures, diagnosis report, tool
    evidence, specification - so the same reader is reading the same shape of
    input. The additions are the previous prompt, the defect list, and a
    line naming the extra picture, followed by the revision requirements.
    """
    parts = [f"Restoration task type: {types_label(diagnosis.get('task_types') or [])}",
             ""]

    parts.append("=== PICTURES YOU ARE LOOKING AT ===")
    for i, entry in enumerate(pack.entries, start=1):
        parts.append(f"Picture {i} - {entry.caption}")
    if result_index:
        parts.append(
            f"Picture {result_index} - the RESTORATION RESULT the previous "
            "prompt produced. This is NOT the image to be restored; it is "
            "shown so you can see what the defect list below actually refers to. "
            "The editor will receive Picture 1 again.")
    parts.append("")
    parts.append("As in the first pass: the maps are for YOU. The editing model "
                 "will receive the degraded photograph and your prompt, and "
                 "nothing else - it never sees these maps and cannot follow a "
                 "reference to them. Never describe the result picture as the "
                 "thing to be edited.")
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

    parts.append("=== THE PREVIOUS INSTRUCTION (this is what you are revising) ===")
    parts.append(prev_prompt.strip())
    parts.append("")

    parts.append(_REVISION_TAIL.format(
        defects=defect_summary or "(quality below threshold)",
        n_words=max(1, len(str(prev_prompt).split()))))
    return "\n".join(parts)


def revise_prompt(vlm, types: list[str], diagnosis: dict[str, Any],
                       text_evidence: str, pack: VisualPack,
                       prev_prompt: str, defect_summary: str,
                       out_dir: Path, *,
                       failed_output: str | Path | None = None,
                       round_idx: int = 1) -> dict[str, Any]:
    """One image-bearing VLM call: revise the prompt against the defects.

    The system prompt is the composer's, from ``prompts/refine.txt``, assembled
    with the same per-type task blocks. The pictures are the composer's plus the
    flawed result. Nothing here is a separate role.

    Returns the new prompt and the bookkeeping for the run record.
    """
    system = build_system(load_prompt("refine.txt"), types,
                          load_blocks("compose_blocks.txt"))

    images = _images(Path(pack.entries[0].path), pack, failed_output)
    has_result = bool(failed_output) and Path(str(failed_output)).is_file()
    result_index = len(images) if has_result else 0
    user = build_revision_message(diagnosis, text_evidence, pack,
                                  prev_prompt, defect_summary, result_index)

    obj = vlm.chat_json(user, images, system=system)
    new_prompt = str(obj.get("instruction", "")).strip()
    if not new_prompt:
        # An empty revision means the reply was truncated or malformed. Falling
        # back to the previous prompt would silently spend a round
        # regenerating the same image, so fail loudly instead.
        raise ValueError(
            "stage 5 returned an empty prompt. The reply was most likely "
            "truncated - raise VLM_MAX_OUTPUT_TOKENS in harness/settings.py, or "
            "lower the word range in prompts/refine.txt PART E.")

    out_dir.mkdir(parents=True, exist_ok=True)
    # One file per round: keeping only a single filename would leave just the
    # last revision, with no way to see what an earlier round actually sent.
    (out_dir / "refined_prompt.txt").write_text(new_prompt, encoding="utf-8")
    (out_dir / f"refined_prompt_r{round_idx}.txt").write_text(new_prompt,
                                                             encoding="utf-8")

    return {
        "mode": "global",
        "prompt": "refine.txt",
        "new_prompt": new_prompt,
        "defects": defect_summary,
        "n_images": len(images),
        "prev_words": max(1, len(str(prev_prompt).split())),
        "new_words": len(new_prompt.split()),
    }
