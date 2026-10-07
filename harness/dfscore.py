#!/usr/bin/env python3
"""D-Score, F-Score and DF-Score: a VLM verifier over (input, result) pairs.

Two axes, each 0-100, scored by a VLM that sees the degraded input as picture 1
and the restoration result as picture 2:

    D  degradation removed and clarity gained, relative to the input
    F  content fidelity: did anything in the scene get rewritten

The two axes exist because neither alone can be satisfied cheaply. A method that
regenerates the scene scores well on D and badly on F; a method that returns its
input untouched scores perfectly on F and zero on D. The composite is therefore
a geometric mean, computed PER IMAGE and then averaged:

    DF-Score = (1/N) * sum_i sqrt(D_i * F_i)

Not ``sqrt(mean(D) * mean(F))``. Averaging first lets one image's good D cover
another image's broken F, which is exactly the trade the geometric mean is meant
to forbid; the per-image form runs 3 to 11 points lower and is the definition
reported here.

The criteria are selected by the image's DEGRADATION TYPE, never by the
prompt the method was given. A method must not be able to change the
yardstick by changing what it asks for, and a verifier reading the prompt
would be scoring compliance rather than restoration.

Python applies no score corrections. D and F are taken exactly as the model
returned them, and no rule stated in the prompt is re-imposed in code - not the
capping rules, not the scene-replaced rule, not the no-op rule. The prompt states
what those rules are and the scores are whatever the model made of them; a scorer
that quietly corrected a score would hide the disagreement instead of reporting
it. The flags this module adds (``veto_hit``, ``scene_replaced``, ``no_op_vlm``)
record what the model itself claimed, so the disagreement stays visible.
"""
from __future__ import annotations

import math
import re
import statistics
from pathlib import Path
from typing import Any

from . import settings

PROMPTS = settings.PROMPTS_DIR / "df"
CORE_FILE = "df_core.txt"
TYPES_FILE = "df_types.txt"
INTENT_FILE = "df_intent.txt"

METRIC_NAME = "VLM-DF"
SCALE_MAX = 100
PASS_SCORE = 80           # band boundary for "acceptable", for reporting only
STEP = 5                  # the prompt asks for multiples of 5

KNOWN_TYPES = ["haze", "lowlight", "rain", "snow", "mix"]

# Which type supplies the primary label when an image carries several. Must stay
# identical to the manifest builder's ordering, or the same image lands in
# different rows of the per-type table on either side of the comparison.
TYPE_PRIORITY = ["haze", "rain", "snow", "lowlight", "mix"]

# Score bands, used for distribution readouts and flip rates.
BANDS: tuple[tuple[int, int, str], ...] = (
    (95, 100, "excellent"),
    (80, 94, "acceptable"),
    (50, 79, "compromised"),
    (20, 49, "damaged"),
    (0, 19, "broken"),
)

# Properties whose violation means content was rewritten. Must stay in sync with
# the entries marked as capped in df_types.txt.
#   P1 object added or removed   P3 text fabricated   P4 face identity changed
#   P6 artificial covering introduced
VETO_UNIVERSAL = {"P1", "P3", "P4", "P6"}
VETO_BY_TYPE: dict[str, set[str]] = {
    "haze":     {"T1"},   # occluded sky invented
    "lowlight": {"T1"},   # night turned into day
    "rain":     {"T1"},   # rain turned into clear weather
    "snow":     {"T1"},   # lying snow removed
    "mix":      set(),
}

# Structural SSIM above this means the result is practically a copy of the
# input. An audit column only - "high" here is a warning, not a compliment.
NO_OP_SSIM = 0.97


def band_of(score: float | None) -> str | None:
    if score is None:
        return None
    for lo, hi, name in BANDS:
        if lo <= score <= hi:
            return name
    return None


# ---------------------------------------------------------------------------
# prompt assembly
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Drop ``#`` comment lines, keeping ``# [[type]]`` block markers."""
    keep = [ln for ln in text.splitlines()
            if not (ln.startswith("#")
                    and not ln.lstrip("#").strip().startswith("[["))]
    return "\n".join(keep).strip()


def load_text(name: str) -> str:
    return _strip_comments((PROMPTS / name).read_text(encoding="utf-8"))


def load_blocks(name: str) -> dict[str, str]:
    """Parse a ``### [[type]] ###`` sectioned prompt file."""
    blocks: dict[str, str] = {}
    cur, buf = None, []
    for line in load_text(name).splitlines():
        s = line.strip()
        if s.startswith("### [[") and s.endswith("]] ###"):
            if cur:
                blocks[cur] = "\n".join(buf).strip()
            cur, buf = s[6:-6], []
        elif cur is not None:
            buf.append(line)
    if cur:
        blocks[cur] = "\n".join(buf).strip()
    return blocks


def normalise_types(raw: Any) -> list[str]:
    """Normalise a ``type`` field - absent, empty or listed - to a list of types.

    An absent or empty field means ``mix``, not "unlabelled": there is no
    unlabelled image in these sets, and a generic rubric is what an unspecified
    type resolves to. Resolving it here rather than at each call site keeps the
    two spellings of "no type given" from diverging - one path treating it as
    mix and another raising for it.
    """
    if raw is None:
        return ["mix"]
    if isinstance(raw, str):
        items = [t.strip() for t in raw.split("|") if t.strip()]
    else:
        items = [str(t).strip() for t in raw if str(t).strip()]
    if not items:
        return ["mix"]
    unknown = [t for t in items if t not in KNOWN_TYPES]
    if unknown:
        raise ValueError(f"unknown degradation type {unknown}; known: {KNOWN_TYPES}")
    return list(dict.fromkeys(items))


def primary_type(types: list[str]) -> str:
    for t in TYPE_PRIORITY:
        if t in types:
            return t
    return types[0]


def _type_tag(t: str) -> str:
    return t.upper()


def veto_properties(types: list[str]) -> set[str]:
    """Capped properties for this image, named as the prompt names them."""
    out = set(VETO_UNIVERSAL)
    multi = len(types) > 1
    for t in types:
        for tid in VETO_BY_TYPE.get(t, set()):
            out.add(f"{_type_tag(t)}.{tid}" if multi else tid)
    return out


def _namespace_t_ids(block: str, tag: str) -> str:
    """Rewrite ``T1`` to ``HAZE.T1`` inside one type block.

    Each block numbers its own T items, so T1 means "occluded sky" under haze
    and something else under mix. Concatenated without a prefix the numbers
    collide and the veto lookup consults the wrong rule. Cross-references in the
    prose are rewritten too, or a block's "see T2" points into another type.
    """
    block = re.sub(r"(?m)^(\s*)T(\d+)\b",
                   lambda m: f"{m.group(1)}{tag}.T{m.group(2)}", block)
    return re.sub(r"\b([Ss]ee )T(\d+)\b",
                  lambda m: f"{m.group(1)}{tag}.T{m.group(2)}", block)


USER_MSG = (
    "Picture 1 is the DEGRADED INPUT. Picture 2 is the RESTORATION RESULT "
    "produced from it. Score D and F on 0-100 as instructed and return only "
    "the JSON."
)


def build_system(types: list[str], intent: str | None = None, *,
                 core: str | None = None,
                 blocks: dict[str, str] | None = None,
                 intents: dict[str, str] | None = None) -> str:
    """Fill the verifier's system prompt for one image's degradation types.

    With several types, every block is injected: D symptoms are the union, and
    protected properties are the union as well, so the stricter rule wins. A
    change is licensed only when no block protects it - if one block permits a
    colour change and another forbids it, it is forbidden.
    """
    core = load_text(CORE_FILE) if core is None else core
    blocks = load_blocks(TYPES_FILE) if blocks is None else blocks
    intents = load_blocks(INTENT_FILE) if intents is None else intents

    if not types:
        raise ValueError("types is empty: the degradation type selects this "
                         "metric's criteria and cannot be omitted")
    missing = [t for t in types if t not in blocks]
    if missing:
        raise ValueError(f"{TYPES_FILE} has no block for: {missing}")

    if len(types) == 1:
        task_type, block = types[0], blocks[types[0]]
    else:
        task_type = " + ".join(types)
        parts = [f"----- criteria for {t} -----\n"
                 + _namespace_t_ids(blocks[t], _type_tag(t)) for t in types]
        parts.append(
            "----- when the blocks above disagree -----\n"
            "This image carries several degradation types at once. Every D "
            "symptom listed by any block is in scope. Every protected property "
            "listed by any block is protected - the stricter rule wins. A change "
            "is licensed only if no block protects it: if one block says colour "
            "may change and another says colour must not, then colour must not.")
        block = "\n\n".join(parts)

    if intent is None:
        intent = intents.get(primary_type(types), "").strip()
    return (core.replace("{{TASK_TYPE}}", task_type)
                .replace("{{USER_INTENT}}", intent)
                .replace("{{TASK_BLOCK}}", block))


# ---------------------------------------------------------------------------
# audit columns
#
# Two cheap pixel measurements, recorded alongside every pair. They do not enter
# D or F: the colour rules differ by degradation type, so a large chroma drift is
# a violation under one type and the intended outcome under another, and only
# the type block knows which. They are here to catch the case the VLM is worst
# at - telling a copy of the input apart from a restoration of it.
# ---------------------------------------------------------------------------

def _gray_small(path: str | Path, shape: tuple[int, int] | None = None):
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("L")
    if shape is not None:
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    else:
        w, h = im.size
        s = 192 / max(w, h)
        if s < 1:
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                           Image.BILINEAR)
    return np.asarray(im, dtype="float64") / 255.0


def structure_ssim(lq: str | Path, out: str | Path) -> float | None:
    """Grayscale SSIM at 192px. High means the structure did not move."""
    try:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        a = _gray_small(lq)
        b = _gray_small(out, shape=a.shape)
        win = 7
        if min(a.shape) < win:
            return None
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        wa = sliding_window_view(a, (win, win)).reshape(-1, win * win)
        wb = sliding_window_view(b, (win, win)).reshape(-1, win * win)
        ma, mb = wa.mean(1), wb.mean(1)
        va, vb = wa.var(1), wb.var(1)
        cov = (wa * wb).mean(1) - ma * mb
        s = (((2 * ma * mb + c1) * (2 * cov + c2))
             / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))
        return round(float(s.mean()), 4)
    except Exception:                         # noqa: BLE001 - audit column only
        return None


def chroma_drift(lq: str | Path, out: str | Path) -> float | None:
    """Euclidean distance between mean Lab a/b of the two images, 0-1."""
    try:
        import numpy as np
        from PIL import Image

        def ab(p, shape=None):
            im = Image.open(p).convert("RGB")
            if shape is not None:
                im = im.resize((shape[1], shape[0]), Image.BILINEAR)
            else:
                w, h = im.size
                s = 192 / max(w, h)
                if s < 1:
                    im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                                   Image.BILINEAR)
            lab = np.asarray(im.convert("LAB"), dtype="float64")
            return lab[..., 1:3] / 255.0, lab.shape[:2]

        x, shp = ab(lq)
        y, _ = ab(out, shape=shp)
        d = x.reshape(-1, 2).mean(0) - y.reshape(-1, 2).mean(0)
        return round(float(np.linalg.norm(d)), 4)
    except Exception:                         # noqa: BLE001 - audit column only
        return None


def audit_pair(lq: str | Path, out: str | Path) -> dict[str, Any]:
    ssim = structure_ssim(lq, out)
    return {"structure_ssim": ssim,
            "chroma_drift": chroma_drift(lq, out),
            "no_op_ssim": ssim is not None and ssim > NO_OP_SSIM}


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------

def _as_score(v: Any) -> int | None:
    """A 0-100 score, or ``None`` when out of range.

    Out-of-range values are discarded, never clamped: clamping dresses "the
    model ignored the scale" up as a judgement.
    """
    if v is None:
        return None
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return n if 0 <= n <= SCALE_MAX else None


def _as_sev(v: Any) -> int | None:
    """A 0-3 symptom severity, or ``None``."""
    if v is None:
        return None
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return n if 0 <= n <= 3 else None


def _clean_violations(raw: Any) -> list[dict[str, Any]]:
    out = []
    for v in raw or []:
        if not isinstance(v, dict):
            continue
        bbox = v.get("bbox_norm")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                and all(isinstance(x, (int, float)) for x in bbox)
                and 0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
            bbox = None
        out.append({"property": str(v.get("property", "")).strip().upper(),
                    "where": str(v.get("where", "")).strip(),
                    "bbox_norm": list(bbox) if bbox else None,
                    "what_changed": str(v.get("what_changed", "")).strip()})
    return out


def parse_response(obj: dict[str, Any], types: list[str],
                   audit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn the VLM's JSON into one scored record.

    D and F are taken exactly as returned and never adjusted, and no rule the
    prompt states is re-imposed here. The capping rules live in the prompt
    (``prompts/df/df_types.txt`` marks the properties, and the core states the
    consequence), and whether the model followed them is answered by the scores
    themselves. A scorer that silently corrected a score would hide the
    discrepancy instead of reporting it.

    What this function adds is bookkeeping only:

      ``veto_hit``       which capped properties the model listed as violated
      ``scene_replaced`` the model's own report that the scene was replaced
      ``no_op_vlm``      the model's own report that nothing changed
      ``order_confused`` a self-check on picture order that came back wrong
      ``off_grid_*``     a score that is not a multiple of 5
      ``audit``          the SSIM / chroma columns, computed from pixels

    None of these changes D or F.
    """
    d = _as_score(obj.get("D"))
    f = _as_score(obj.get("F"))
    vio = _clean_violations(obj.get("violations"))
    flags: list[str] = []

    vetoes = veto_properties(types)
    hit = sorted({v["property"] for v in vio} & vetoes)
    if hit:
        flags.append("veto_hit")

    if obj.get("scene_replaced") is True:
        flags.append("scene_replaced")

    if obj.get("changed_at_all") is False:
        flags.append("no_op_vlm")

    order = str(obj.get("which_is_degraded", "")).strip().lower()
    if order and order not in ("picture_1", "picture1", "1"):
        flags.append("order_confused")

    for name, val in (("D", d), ("F", f)):
        if val is not None and val % STEP:
            flags.append(f"off_grid_{name}")

    return {"D": d, "F": f,
            "D_band": band_of(d), "F_band": band_of(f),
            "veto_hit": hit,
            "violations": vio,
            "symptom_before": _as_sev(obj.get("symptom_before")),
            "symptom_after": _as_sev(obj.get("symptom_after")),
            "D_why": str(obj.get("D_why", "")).strip(),
            "F_why": str(obj.get("F_why", "")).strip(),
            "notes": str(obj.get("notes", "")).strip(),
            "which_is_degraded": order,
            "changed_at_all": obj.get("changed_at_all"),
            "scene_replaced": obj.get("scene_replaced"),
            "flags": flags}


# ---------------------------------------------------------------------------
# scoring one pair
# ---------------------------------------------------------------------------

def _median(vals: list[float]) -> float | None:
    return round(statistics.median(vals), 2) if vals else None


def _images(lq: str | Path, out: str | Path, max_side: int = 0):
    """The two pictures, in a fixed order: [input, result]."""
    if max_side and max_side > 0:
        from PIL import Image

        out_imgs = []
        for p in (lq, out):
            im = Image.open(p).convert("RGB")
            w, h = im.size
            s = max_side / max(w, h)
            if s < 1:
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                               Image.LANCZOS)
            out_imgs.append(im)
        return out_imgs
    # Raw bytes by default: re-encoding would perturb the very artifacts this
    # metric is asked about.
    return [str(lq), str(out)]


def score_pair(vlm, *, lq: str | Path, output: str | Path,
               types: list[str], intent: str | None = None,
               repeats: int = 1, max_side: int = 0,
               with_audit: bool = True,
               system: str | None = None) -> dict[str, Any]:
    """Score one (input, result) pair.

    ``repeats > 1`` samples the verifier several times and takes the per-axis
    median, which is the only lever that has measurably improved
    self-consistency here (dropping the temperature to 0 helps, sampling the
    same pair repeatedly helps more).

    The returned reasoning and flags come from a single real call - the one
    closest to the median on both axes - rather than being assembled across
    calls. Attaching one call's flags to another call's median score produces
    rows that contradict themselves.
    """
    system = system if system is not None else build_system(types, intent)
    imgs = _images(lq, output, max_side)
    audit = audit_pair(lq, output) if with_audit else None

    runs = [parse_response(vlm.chat_json(USER_MSG, imgs, system=system),
                           types, audit)
            for _ in range(max(1, repeats))]

    ds = [r["D"] for r in runs if r["D"] is not None]
    fs = [r["F"] for r in runs if r["F"] is not None]
    med_d, med_f = _median(ds), _median(fs)

    def distance(r):
        a = abs(r["D"] - med_d) if r["D"] is not None and med_d is not None else 0.0
        b = abs(r["F"] - med_f) if r["F"] is not None and med_f is not None else 0.0
        return a + b

    rec: dict[str, Any] = dict(min(runs, key=distance))
    rec["D"], rec["F"] = med_d, med_f
    rec["D_band"], rec["F_band"] = band_of(med_d), band_of(med_f)
    rec["n_calls"] = len(runs)
    rec["spread_D"] = round(max(ds) - min(ds), 2) if len(ds) > 1 else 0.0
    rec["spread_F"] = round(max(fs) - min(fs), 2) if len(fs) > 1 else 0.0
    rec["band_flip"] = (len({band_of(x) for x in ds}) > 1
                        or len({band_of(x) for x in fs}) > 1)
    rec["gate_flip"] = (len({x >= PASS_SCORE for x in ds}) > 1
                        or len({x >= PASS_SCORE for x in fs}) > 1)
    if len(runs) > 1:
        rec["repeats"] = [{"D": r["D"], "F": r["F"], "flags": r["flags"]}
                          for r in runs]
    if audit is not None:
        rec["audit"] = audit
    return rec


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

PLAIN_FLAGS = ("veto_hit", "scene_replaced", "no_op_vlm", "order_confused",
               "off_grid_D", "off_grid_F")

# Five-band histogram, reported for each axis.
_BAND_NAMES = [b[2] for b in BANDS]


def _dist(vals: list[int]) -> dict[str, Any]:
    """Distribution readout: quantiles plus the band histogram."""
    if not vals:
        return {"n": 0}
    xs = sorted(vals)

    def q(p: float) -> float:
        i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        return float(xs[i])

    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 2),
        "median": float(statistics.median(xs)),
        "p10": q(0.10), "p90": q(0.90),
        "min": float(xs[0]), "max": float(xs[-1]),
        "bands": {name: sum(1 for x in xs if band_of(x) == name)
                  for name in _BAND_NAMES},
    }


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a batch of scored pairs.

    ``entries`` are per-item dicts carrying the pair under the ``"vlm"`` key.

    Reported here:

      D, F        per-axis means
      DF-Score    the headline metric, mean of the per-image geometric mean
      DF_Rate     share of images clearing PASS_SCORE on BOTH axes
      HalRate     share of images whose F falls below PASS_SCORE
      rate_*      how often each flag fired
    """
    vs = [e["vlm"] for e in entries if e.get("vlm")]
    n = len(vs)
    if not n:
        return {"n_scored": 0}

    d = [v["D"] for v in vs if v.get("D") is not None]
    f = [v["F"] for v in vs if v.get("F") is not None]
    both = [(v["D"], v["F"]) for v in vs
            if v.get("D") is not None and v.get("F") is not None]

    def rate(xs: list[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    out: dict[str, Any] = {
        "n_scored": n,
        "D": round(statistics.fmean(d), 2) if d else None,
        "F": round(statistics.fmean(f), 2) if f else None,
        # The reported composite: geometric mean per image, then averaged.
        "DF_Score": (round(statistics.fmean([math.sqrt(a * b) for a, b in both]), 2)
                     if both else None),
        "D_Rate": rate([x >= PASS_SCORE for x in d]),
        "F_Rate": rate([x >= PASS_SCORE for x in f]),
        "DF_Rate": rate([a >= PASS_SCORE and b >= PASS_SCORE for a, b in both]),
        "HalRate": rate([x < PASS_SCORE for x in f]),
        "n_D": len(d), "n_F": len(f),
        "dist_D": _dist(d), "dist_F": _dist(f),
    }
    out["band_flip_rate"] = rate([bool(v.get("band_flip")) for v in vs])
    out["gate_flip_rate"] = rate([bool(v.get("gate_flip")) for v in vs])
    for key in ("spread_D", "spread_F"):
        xs = [v[key] for v in vs if v.get(key) is not None]
        out[f"{key}_mean"] = round(statistics.fmean(xs), 2) if xs else None
        out[f"{key}_max"] = round(max(xs), 2) if xs else None

    for flag in PLAIN_FLAGS:
        out[f"rate_{flag}"] = rate([flag in (v.get("flags") or []) for v in vs])

    ssim_no_op = [bool((v.get("audit") or {}).get("no_op_ssim")) for v in vs
                  if (v.get("audit") or {}).get("structure_ssim") is not None]
    out["rate_no_op_ssim"] = rate(ssim_no_op)
    return out


def format_line(v: dict[str, Any]) -> str:
    """One-line per-image summary for progress output."""
    def s(x):
        return "-" if x is None else f"{x:g}"

    tail = ""
    if v.get("veto_hit"):
        tail += f" veto={','.join(v['veto_hit'])}"
    return f"D={s(v.get('D'))}  F={s(v.get('F'))}{tail}"


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------

def score_rows(entries: list[dict[str, Any]],
               keys: list[str] | None = None) -> dict[str, Any]:
    """Aggregate per-group as well as overall.

    ``keys`` names the grouping fields on each entry (``type``, ``group``,
    ``folder``). A per-type breakdown is the most informative view of this
    metric, because the axes do not behave the same way across degradations.
    """
    ok = [e for e in entries if e.get("vlm")]
    out: dict[str, Any] = {"overall": summarize(ok)}
    for key in (keys or []):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for e in ok:
            k = e.get(key)
            if isinstance(k, list):
                k = "|".join(k)
            buckets.setdefault(str(k), []).append(e)
        out[f"by_{key}"] = {k: summarize(v) for k, v in sorted(buckets.items())}
    return out


def geometric_mean(d: float | None, f: float | None) -> float | None:
    """Per-image DF for one record, in case a caller needs it directly."""
    if d is None or f is None:
        return None
    return round(math.sqrt(d * f), 2)
