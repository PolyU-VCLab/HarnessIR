#!/usr/bin/env python3
"""Stage 5 - verifier: deterministic gates plus a VLM verdict.

The verifier sees the degraded input, the result, and the standard for that image's
degradation types. It never sees the prompt that produced the result.
Showing it the prompt would make the system grade itself against its own
prompt: any requirement the composer omitted would quietly stop being checked, and
a dropped requirement would read as a pass.

That has a hard consequence - the per-type task block is the ONLY channel
through which restoration requirements reach the verifier. A wrong type is a wrong
ruler, which is why this stage assembles the same multi-type prompt the composer
got, colour ruling and all.

Two layers, in this order:

* gates - deterministic, no model involved. They exist to catch the failures
  that are certain: the scene was replaced, or the framing changed. Thresholds
  are deliberately loose. A false reject removes that round from best-of-N
  outright, which costs far more than letting one bad round through.
* the verdict - needs_redo and, on later rounds, best_round.

Loop mode (the default when redo is on) differs from single-round judging in
what it compares. An absolute quality score correlates with real error at
roughly zero across different images, because difficulty varies far more than
quality does. Comparing attempts at the SAME image removes the difficulty term,
and what remains is the quantity of interest. So the verifier is shown every
attempt so far, plus the measured numbers for each, and answers directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import settings
from ..types import COLOUR_CHANGING_TYPES, build_system
from ..utils import load_blocks, load_prompt

# NR metrics reported to the verifier. All seven would crowd out the actual
# judgement, and they carry almost no fidelity signal anyway; these three are
# enough to confirm "this is a field of artifacts" and not enough to rank by.
NR_FOR_JUDGE = ["MUSIQ", "MANIQA", "CLIPIQA"]


@dataclass
class VerificationResult:
    verdict: str = ""                     # correct | incorrect
    failure_mode: str = ""                # none | under_doing | over_reach | both
    reasoning: str = ""
    checklist: dict[str, str] = field(default_factory=dict)
    defects: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    iqa: dict[str, float] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # The redo decision, answered by the verifier itself.
    needs_redo: bool | None = None
    redo_priority: float | None = None    # 0-10, recorded for offline analysis
    best_round: int | None = None         # which attempt to deliver
    best_round_why: str = ""
    measurements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Whether this round stops the loop.

        Two conditions, in order:

          1. the deterministic gates passed, and
          2. the verifier said no further attempt is needed.

        That is the whole rule. ``checklist``, ``defects``, ``scores`` and
        ``failure_mode`` are recorded and used to write the defect report that
        the revision pass reads, but they do not decide acceptance: the verifier
        answers ``needs_redo`` directly, and second-guessing it from the other
        fields would mean two rules with veto power over one decision.

        ``needs_redo`` missing is an error, not a fallback. The caller validates
        it before constructing this object, so reaching here without it means
        the response was malformed and the round's verdict is unknown.
        """
        if self.gates.get("failed"):
            return False
        if self.needs_redo is None:
            raise ValueError(
                "verifier returned no usable needs_redo, so this round's verdict is "
                "unknown. The response must contain needs_redo as a boolean "
                "(true/false, or a string/1/0 the parser accepts).")
        return not self.needs_redo

    def defect_summary(self) -> str:
        """One-line defect list, for the revision pass to act on."""
        if not self.defects:
            return ""
        return "; ".join(f"{d.get('location', '?')}: {d.get('issue', '')} "
                         f"[{d.get('severity', '?')}]" for d in self.defects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "needs_redo": self.needs_redo,
            "redo_priority": self.redo_priority,
            "best_round": self.best_round,
            "best_round_why": self.best_round_why,
            "measurements": self.measurements,
            # Recorded for the defect report and for offline analysis. None of
            # these decide acceptance - needs_redo does.
            "verdict": self.verdict,
            "failure_mode": self.failure_mode,
            "reasoning": self.reasoning,
            "checklist": self.checklist,
            "defects": self.defects,
            "scores": self.scores,
            "overall": self.overall,
            "iqa": self.iqa,
            "gates": self.gates,
        }

    # -- value coercion for the model's free-form JSON -----------------------

    @staticmethod
    def as_bool(v: Any) -> bool | None:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes", "1"):
                return True
            if s in ("false", "no", "0"):
                return False
        return None

    @staticmethod
    def as_float(v: Any) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def as_round(v: Any, n: int) -> int | None:
        """Reject a ``best_round`` outside the attempts actually shown.

        Clamping would dress noise up as a decision: a model that names attempt
        seven of three was not counting pictures, so its answer carries nothing.
        Returning ``None`` sends the selector down its default path.
        """
        try:
            i = int(v)
        except (TypeError, ValueError):
            return None
        return i if 0 <= i < n else None


# ---------------------------------------------------------------------------
# Deterministic gates
# ---------------------------------------------------------------------------

def _gray_small(path: str | Path, long_edge: int = 192, shape=None):
    from PIL import Image

    im = Image.open(path).convert("L")
    if shape is not None:
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    else:
        w, h = im.size
        s = long_edge / max(w, h)
        if s < 1.0:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                           Image.BILINEAR)
    return np.asarray(im).astype("float32") / 255.0


def _ssim(a, b, win: int = 7) -> float | None:
    """Uniform-window SSIM over a downscaled grayscale pair.

    This only has to separate "restored" from "replaced", so it does not need to
    match a reference implementation numerically.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if a.shape != b.shape or min(a.shape) < win:
        return None
    n = win * win
    wa = sliding_window_view(a, (win, win)).reshape(-1, n)
    wb = sliding_window_view(b, (win, win)).reshape(-1, n)
    mu_a, mu_b = wa.mean(1), wb.mean(1)
    da, db = wa - mu_a[:, None], wb - mu_b[:, None]
    va = (da * da).sum(1) / (n - 1)
    vb = (db * db).sum(1) / (n - 1)
    cov = (da * db).sum(1) / (n - 1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return round(float(s.mean()), 4)


def _chroma_drift(lq_path: str | Path, result_path: str | Path,
                  long_edge: int = 192) -> float | None:
    """Mean chromaticity shift, luminance-independent.

    Normalising RGB by its sum removes brightness, which is what makes this
    usable at all here: dehazing and low-light lifting change brightness a great
    deal and would trip any luminance-carrying measure. What is left is hue
    movement - a white balance "correction", or a global grade.
    """
    from PIL import Image

    def chroma(p, shape=None):
        im = Image.open(p).convert("RGB")
        if shape is not None:
            im = im.resize((shape[1], shape[0]), Image.BILINEAR)
        else:
            w, h = im.size
            s = long_edge / max(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.BILINEAR)
        a = np.asarray(im).astype("float32") / 255.0
        return (a / (a.sum(2, keepdims=True) + 1e-6)).mean((0, 1))

    try:
        return round(float(np.abs(chroma(lq_path) - chroma(result_path)).sum()), 4)
    except Exception:                         # noqa: BLE001
        return None


def check_gates(result_path: str | Path, lq_path: str | Path, *,
                task_types: list[str] | None = None) -> dict[str, Any]:
    """Deterministic checks. Any failure marks the round unusable.

    Only certainties belong here: scene replacement, framing change. Everything
    with a judgement call in it belongs to the VLM.
    """
    from PIL import Image

    gates: dict[str, Any] = {"failed": False, "checks": {}}

    def fail(name: str, **kw) -> None:
        gates["checks"][name] = {"pass": False, **kw}
        gates["failed"] = True

    def ok(name: str, **kw) -> None:
        gates["checks"][name] = {"pass": True, **kw}

    # Framing sentinel. A size change means the result is not the same picture
    # at the same scale and nothing downstream can compare them.
    lq_size = Image.open(lq_path).size
    out_size = Image.open(result_path).size
    (ok if lq_size == out_size else fail)("size_match", lq=lq_size, out=out_size)

    # Structure: is this still the same scene?
    try:
        g_lq = _gray_small(lq_path)
        g_out = _gray_small(result_path, shape=g_lq.shape)
        value = _ssim(g_lq, g_out)
        if value is None:
            gates["checks"]["structure"] = {"pass": None, "note": "ssim unavailable"}
        elif value >= settings.STRUCTURE_SSIM_MIN:
            ok("structure", value=value, threshold=settings.STRUCTURE_SSIM_MIN)
        else:
            fail("structure", value=value, threshold=settings.STRUCTURE_SSIM_MIN)
    except Exception as e:                    # noqa: BLE001
        gates["checks"]["structure"] = {"pass": None, "error": str(e)}

    # Colour drift. Recorded always, enforced only where the type does not
    # license it - for haze and lowlight a large drift is the correct outcome,
    # and enforcing it would kill every successful round.
    licensed = bool(set(task_types or []) & COLOUR_CHANGING_TYPES)
    drift = _chroma_drift(lq_path, result_path)
    if drift is None:
        gates["checks"]["colour_drift"] = {"pass": None}
    elif licensed:
        gates["checks"]["colour_drift"] = {
            "pass": True, "value": drift, "enforced": False,
            "note": "colour-changing type present; recorded only"}
    elif drift <= settings.CHROMA_DRIFT_MAX:
        ok("colour_drift", value=drift, threshold=settings.CHROMA_DRIFT_MAX,
           enforced=True)
    else:
        fail("colour_drift", value=drift, threshold=settings.CHROMA_DRIFT_MAX,
             enforced=True)

    return gates


# ---------------------------------------------------------------------------
# Verifier prompt assembly
# ---------------------------------------------------------------------------

def _type_tag(task_type: str) -> str:
    """``lowlight`` -> ``LOWLIGHT``, used to namespace checklist ids."""
    return re.sub(r"[^A-Za-z0-9]+", "_", task_type).upper()


def _namespace_t_slots(block: str, tag: str) -> str:
    """Rewrite ``T5`` to ``RAIN.T5`` inside one block.

    Every block numbers its items from T1, and the ids mean different things in
    different blocks - T5 is "physical plausibility" under haze and "colour
    unchanged" under rain. Concatenating blocks would collide the ids and the
    verifier would have no way to say which T5 it meant.

    Only entry definitions at the start of a line are rewritten. Cross-
    references to the universal items (U2, U3) are left alone.
    """
    return re.sub(r"(?m)^(\s*)T(\d+)\b", rf"\1{tag}.T\2", block)


def build_verify_system(task_types: list[str], *,
                       template_name: str = "verify.txt") -> str:
    """Assemble the verifier system prompt for this image's type combination."""
    blocks = load_blocks("verify_blocks.txt")
    if len(task_types) > 1:
        # Prefix only the blocks in play; the rest stay in the dict so an
        # unknown type still errors with the full list of valid names.
        tagged = set(task_types)
        blocks = {t: (_namespace_t_slots(b, _type_tag(t)) if t in tagged else b)
                  for t, b in blocks.items()}
    return build_system(load_prompt(template_name), task_types, blocks,
                        voice="verify")


def _loop_user(n_attempts: int) -> str:
    if n_attempts == 1:
        # Most rounds are first rounds, and most first rounds end here. Say so,
        # or the model invents a second attempt to have something to choose.
        return (
            "Picture 1 is the degraded input. Picture 2 is the only attempt so far "
            "at restoring it (attempt 0).\n\n"
            "Decide whether a second attempt is worth making. `best_round` is 0 - "
            "there is nothing else to choose from. Output the JSON only.")
    return (
        f"Picture 1 is the degraded input. The next {n_attempts} pictures are "
        f"attempts at restoring it, oldest first; the last picture is the newest "
        f"attempt (attempt {n_attempts - 1}).\n\n"
        "Decide whether another attempt is worth making, and which of the "
        f"{n_attempts} attempts should be delivered. Output the JSON only.")


def measure_attempt(lq_path: str | Path, result_path: str | Path, *,
                    gates: dict[str, Any] | None = None,
                    iqa: dict[str, float] | None = None) -> dict[str, Any]:
    """Deterministic measurements for one attempt, relative to the input."""
    checks = (gates or {}).get("checks") or {}
    ssim = (checks.get("structure") or {}).get("value")
    chroma = (checks.get("colour_drift") or {}).get("value")

    if ssim is None:
        try:
            g_lq = _gray_small(lq_path)
            ssim = _ssim(g_lq, _gray_small(result_path, shape=g_lq.shape))
        except Exception:                     # noqa: BLE001
            ssim = None
    if chroma is None:
        chroma = _chroma_drift(lq_path, result_path)

    out: dict[str, Any] = {"structure_ssim": ssim, "chroma_drift": chroma}
    if iqa:
        nr = {k: iqa[k] for k in NR_FOR_JUDGE
              if isinstance(iqa.get(k), (int, float))}
        if nr:
            out["nr"] = nr
    return out


def _measurements_block(measurements: list[dict[str, Any]],
                        task_types: list[str]) -> str:
    """Fill the ``{{MEASUREMENTS}}`` slot of verify.txt.

    Every number is reported together with the way it fails. A bare number gets
    read as "higher is better", and a structure SSIM above about 0.97 usually
    means the attempt changed nothing at all - a warning, not a good score.
    """
    licensed = bool(set(task_types) & COLOUR_CHANGING_TYPES)
    has_nr = any("nr" in m for m in measurements)

    lines = [
        "-------------------------------------------------------------",
        "MEASUREMENTS (computed from the pixels - not estimates)",
        "-------------------------------------------------------------",
        "",
        "structure_ssim - grayscale structural similarity between that attempt and",
        "  Picture 1, in [0,1]. It measures HOW MUCH CHANGED, not whether the change",
        "  was right, so read it in both directions:",
        "    . Above ~0.97 is a WARNING, not a compliment: the attempt barely altered",
        "      the photograph. Before accepting such a result, confirm the degradation",
        "      is actually gone rather than merely untouched.",
        "    . A low value is expected and correct where the task legitimately rewrites",
        "      global contrast or colour; it is a red flag where the task does not,",
        "      because then something moved or was repainted.",
        "  It is the most informative single number available here, but it is a prior,",
        "  not a verdict - when it disagrees with what you can plainly see in the",
        "  pictures, your eyes win and you say so in `reasoning`.",
        "",
        "chroma_drift - mean absolute shift in chromaticity vs Picture 1, luminance-",
        "  independent, in [0,1]. Above ~0.10 the colour or white balance moved",
        "  materially.",
        "  For this image that shift is " + (
            "LICENSED by the task type: the restoration\n"
            "  necessarily changes colour here, so a high value is not by itself a\n"
            "  defect." if licensed else
            "NOT licensed: the task block locks colour for\n"
            "  this type, so a high value is a defect in its own right."),
    ]
    if has_nr:
        lines += [
            "",
            "NR scores - no-reference quality models (higher is better for all three).",
            "  Treat these with suspicion. Measured against ground-truth error on an",
            "  earlier batch of this exact task, their correlation was ~0.07 - i.e.",
            "  none. They do NOT detect fabrication: a confidently hallucinated result",
            "  scores HIGHER than a faithful one, because it is sharper and cleaner. Use",
            "  them only to corroborate something you can already see. Never let them",
            "  decide `best_round`.",
        ]

    lines += ["", "Values:"]
    for i, m in enumerate(measurements):
        s, c = m.get("structure_ssim"), m.get("chroma_drift")
        parts = [f"  attempt {i}:",
                 f"structure_ssim {s:.3f}" if isinstance(s, (int, float))
                 else "structure_ssim n/a",
                 f" chroma_drift {c:.3f}" if isinstance(c, (int, float))
                 else " chroma_drift n/a"]
        if m.get("nr"):
            parts.append("  [" + "  ".join(f"{k} {v:g}"
                                           for k, v in m["nr"].items()) + "]")
        if i == len(measurements) - 1 and len(measurements) > 1:
            parts.append("   <- newest")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def run_verify(vlm, *, task_types: list[str],
              lq_path: str | Path, result_path: str | Path,
              prev_rounds: list[dict[str, Any]] | None = None,
              iqa: dict[str, float] | None = None) -> VerificationResult:
    """Gate the result, then ask the verifier.

    Notes:

    * Picture 1 is always the ORIGINAL input, never a previous attempt. Fidelity
      is measured against the source photograph, or drift across rounds cannot
      be seen.
    * The diagnostic maps are never shown here. Their presence would create an
      information asymmetry the composer did not have, and the standard is meant
      to be reproducible across methods that have no maps at all.
    * ``prev_rounds`` entries look like ``{"path", "gates", "iqa"}``; the latter
      two may be absent, in which case the measurements are recomputed.
    """
    gates = check_gates(result_path, lq_path, task_types=task_types)
    system = build_verify_system(task_types, template_name="verify.txt")

    # Every attempt so far goes in, plus each one's measured numbers. The
    # comparison between attempts at one image is the part that carries signal;
    # an absolute score across different images does not.
    prev = list(prev_rounds or [])
    measurements = [measure_attempt(lq_path, r["path"], gates=r.get("gates"),
                                    iqa=r.get("iqa")) for r in prev]
    measurements.append(measure_attempt(lq_path, result_path, gates=gates,
                                        iqa=iqa))
    system = system.replace("{{MEASUREMENTS}}",
                            _measurements_block(measurements, task_types))
    paths = ([str(lq_path)] + [str(r["path"]) for r in prev]
             + [str(result_path)])
    obj = vlm.chat_json(_loop_user(len(measurements)), paths, system=system)

    needs_redo = VerificationResult.as_bool(obj.get("needs_redo"))
    if needs_redo is None:
        # The whole accept/redo decision rests on this field. Guessing a value
        # would turn a malformed reply into a silently different policy, so the
        # round fails instead and the run record shows it.
        raise ValueError(
            "verifier response has no usable needs_redo (got "
            f"{obj.get('needs_redo')!r}). The prompt requires it as a boolean; "
            f"the reply contained keys {sorted(obj)}.")

    result = VerificationResult(
        verdict=str(obj.get("verdict", "")),
        failure_mode=str(obj.get("failure_mode", "")),
        reasoning=str(obj.get("reasoning", "")),
        checklist=dict(obj.get("checklist", {})),
        defects=list(obj.get("defects", [])),
        scores=dict(obj.get("scores", {})),
        overall=float(obj.get("overall", 0) or 0),
        iqa=dict(iqa or {}),
        gates=gates,
        raw=obj,
        measurements=measurements,
        needs_redo=needs_redo,
        redo_priority=VerificationResult.as_float(obj.get("redo_priority")),
        best_round=VerificationResult.as_round(obj.get("best_round"), len(measurements)),
        best_round_why=str(obj.get("best_round_why", "")),
    )
    return result
