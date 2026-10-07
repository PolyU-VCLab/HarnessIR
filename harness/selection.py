#!/usr/bin/env python3
"""Select which attempt to deliver when more than one exists.

Without a selector, multi-round work has a silent failure mode: the delivered
image is simply the LAST attempt. Redoing a restoration can easily make it
worse, and each round is an independent generation with real variance - so
"enabling redo" would degrade the batch for reasons that have nothing to do with
whether the verifier or the rewriter are any good. Redo only pays off if something
chooses between the candidates.

Rules, in order:

1. Drop rounds whose deterministic gates failed. Those outputs were already
   judged unusable.
2. In loop mode, take ``best_round`` from the LAST round's verdict. Only that
   verdict saw every attempt; an earlier one named a best before the later
   candidates existed.
3. Otherwise take the EARLIEST accepted round. Stopping early is the goal - a
   later round is further work on a result already judged good, with no reason
   to expect improvement.
4. If nothing was accepted, rank by verifier ``overall`` descending, then by an NR
   composite, then by round ascending. The continuous score is steadier than the
   binary verdict, NR breaks ties, and the earliest round wins the remainder
   because it changed less.

All rounds stay in the record, so "deliver the last attempt" and "deliver the
oracle-best by ground truth" can both be computed offline from one run.
"""
from __future__ import annotations

from typing import Any, Callable

# NR terms for the tie-break composite. Each is normalised to roughly [0, 1] and
# oriented so larger is better, then averaged over whichever are present - so a
# missing metric (weights not downloaded, a timeout) still leaves a comparable
# number rather than a zero.
#
# AFINE_NR is excluded on purpose: its direction is not established, and a term
# with an unknown sign only introduces errors.
_NR_TERMS: dict[str, Callable[[float], float]] = {
    "CLIPIQA": lambda v: v,
    "CLIPIQA+": lambda v: v,
    "MANIQA": lambda v: v,
    "TOPIQ": lambda v: v,
    "MUSIQ": lambda v: v / 100.0,
    "NIQE": lambda v: 1.0 - v / 10.0,     # lower is better, so flip the sign
}


def nr_composite(iqa: dict[str, float] | None) -> float | None:
    """Collapse NR metrics into one larger-is-better scalar."""
    if not iqa:
        return None
    values = []
    for name, fn in _NR_TERMS.items():
        v = iqa.get(name)
        if v is None:
            continue
        try:
            values.append(float(fn(float(v))))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 4) if values else None


def round_output(round_rec: dict[str, Any]) -> str | None:
    """Path to a round's delivered image."""
    return round_rec.get("output") or (round_rec.get("execute") or {}).get("path")


def round_ssim(round_rec: dict[str, Any]) -> float | None:
    """Grayscale structure SSIM of this round against the input."""
    verifier = round_rec.get("verifier") or {}
    value = (((verifier.get("gates") or {}).get("checks") or {})
             .get("structure") or {}).get("value")
    if value is None:
        measurements = verifier.get("measurements") or []
        if measurements:                  # loop mode stores the same number
            value = measurements[-1].get("structure_ssim")
    return float(value) if isinstance(value, (int, float)) else None


def _gates_failed(round_rec: dict[str, Any]) -> bool:
    return bool(((round_rec.get("verifier") or {}).get("gates") or {}).get("failed"))


def _ssim_pick(passed: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    """Which round a pure "highest SSIM" rule would pick. RECORDED, NOT USED.

    Recorded because it is worth knowing: on pairs where more than one attempt
    exists, picking the round with the highest SSIM against the input captures
    most of the available headroom at zero additional cost. Not used because
    reporting the numbers to the verifier and letting it weigh them is the chosen
    protocol, for interpretability and to keep one code path across stages.

    Keeping both in the record means the difference between the two policies can
    be computed later from the same run, without regenerating anything.

    Worth knowing before switching to it: the highest SSIM is often the round
    that changed least, and "barely touched the image" scores highly here.
    """
    candidates = [(i, s) for i, r in passed
                  if (s := round_ssim(r)) is not None]
    if not candidates:
        return {"index": None, "ssim": None}
    i, s = max(candidates, key=lambda t: (t[1], -t[0]))
    return {"index": i, "ssim": round(s, 4)}


def _loop_pick(usable: list[tuple[int, dict[str, Any]]],
               passed: list[tuple[int, dict[str, Any]]]) -> int | None:
    """The round the loop verifier named as best.

    Only the last verdict counts - it is the only one that saw every attempt.
    ``best_round`` indexes the attempts the verifier was shown, which is the order
    of ``usable``; that usually matches the round index but should not be assumed.

    A round that failed a gate is not selectable even if named: those failures
    are deterministic (scene replaced, framing changed) and the verifier has no
    standing to overrule them.
    """
    if not usable:
        return None
    verifier = usable[-1][1].get("verifier") or {}
    best = verifier.get("best_round")
    if not isinstance(best, int) or not (0 <= best < len(usable)):
        return None
    index = usable[best][0]
    return index if any(i == index for i, _ in passed) else None


def select_final(rounds: list[dict[str, Any]], *,
                 rule: str = "verifier") -> dict[str, Any]:
    """Choose the delivered attempt.

    ``rule`` selects which policy decides, among the rounds that passed the
    deterministic gates:

      ``"verifier"``  the round the verifier named as best, falling back to the
                      first round it accepted, then to the highest continuous
                      score. The verifier compares the attempts against the same
                      restoration requirements the composer worked from, so this
                      is a judgement about restoration quality rather than about
                      similarity.
      ``"ssim"``      the highest grayscale SSIM against the LQ input. Free,
                      deterministic and needs no verdict, but note what it
                      measures: similarity to the input, so an attempt that
                      barely changed the image scores highest. Useful as a
                      reference point, not as a quality criterion.

    Both rules are always computed and both land in the result - ``ssim_pick``
    records what the other policy would have chosen - so the two can be compared
    offline from a single run.

    Returns ``{"index", "round", "path", "reason", "rule", "n_candidates",
    "gate_rejected", "ssim_pick"}``; ``index`` is an offset into ``rounds`` and
    is ``None`` when no round produced an output.
    """
    if rule not in ("verifier", "ssim"):
        raise ValueError(f"unknown selection rule {rule!r}; "
                         f"available: ['ssim', 'verifier']")

    usable = [(i, r) for i, r in enumerate(rounds) if round_output(r)]
    if not usable:
        return {"index": None, "round": None, "path": None,
                "reason": "no round produced an output", "rule": rule,
                "n_candidates": 0, "gate_rejected": 0, "ssim_pick": None}

    passed = [(i, r) for i, r in usable if not _gates_failed(r)]
    gate_rejected = len(usable) - len(passed)
    degraded = False
    if not passed:
        # Every round failed a gate. Still deliver one, since a sample with no
        # image cannot be compared at all - but say so in the reason, so an
        # evaluation can drop it.
        passed = usable
        degraded = True

    ssim_pick = _ssim_pick(passed)
    prefix = "gates_failed_everywhere; " if degraded else ""

    def out(i: int, r: dict[str, Any], reason: str) -> dict[str, Any]:
        return {"index": i, "round": r.get("round"), "path": round_output(r),
                "reason": prefix + reason, "rule": rule,
                "n_candidates": len(usable),
                "gate_rejected": gate_rejected, "ssim_pick": ssim_pick}

    if rule == "ssim":
        if ssim_pick["index"] is not None:
            i = ssim_pick["index"]
            return out(i, rounds[i], f"highest structure SSIM ({ssim_pick['ssim']})")
        # No round carries a structure measurement (the check errored, or the
        # gates were never run). Fall through to the verifier policy rather than
        # delivering nothing.
        prefix = prefix + "no SSIM available; "

    i = _loop_pick(usable, passed)
    if i is not None:
        return out(i, rounds[i], "verifier best_round")

    accepted = [(i, r) for i, r in passed
                if (r.get("verifier") or {}).get("accepted")]
    if accepted:
        i, r = accepted[0]
        return out(i, r, "first accepted round")

    def key(item):
        i, r = item
        verifier = r.get("verifier") or {}
        overall = float(verifier.get("overall", 0) or 0)
        nr = nr_composite(r.get("iqa"))
        return (-overall, -(nr if nr is not None else -1e9), i)

    i, r = sorted(passed, key=key)[0]
    return out(i, r, "no round accepted; best by (overall, NR)")
