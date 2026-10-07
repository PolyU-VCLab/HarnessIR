#!/usr/bin/env python3
"""Compute D-Score, F-Score and DF-Score over a run's delivered images.

    python compute_df.py --records runs/harness/record.json
    python compute_df.py --records runs/harness/record.json --repeats 3
    python compute_df.py --records runs/harness/record.json --aggregate-only

A VLM sees each (input, result) pair - input as picture 1, result as picture 2 -
and scores two axes on 0-100:

    D  how much of the degradation is gone and how much clarity was gained
    F  content fidelity: whether anything in the scene was rewritten

    DF-Score = (1/N) * sum_i sqrt(D_i * F_i)

The geometric mean is taken per image and then averaged, so one image's strong D
cannot offset another image's broken F.

The criteria come from each image's degradation ``type``, never from the prompt
the method was given: a method must not be able to change the yardstick by
changing what it asks for. ``type`` is therefore required - an entry without one
is scored as ``mix``, and if the manifest has no types at all, pass
``--default-type``.

``--repeats N`` scores each pair N times and takes the per-axis median, which is
the only lever that measurably improves self-consistency. ``--aggregate-only``
recomputes the aggregates from an existing output with no API calls, which is how
a definition change is applied to archived scores.

Output: ``df_<record stem>.json`` beside the record, or ``--out``.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import dfscore, settings                           # noqa: E402
from harness.clients import make_vlm                            # noqa: E402
from harness.utils import write_json                            # noqa: E402


def _print_summary(title: str, s: dict) -> None:
    if not s.get("n_scored"):
        print(f"{title}: nothing scored")
        return
    print(f"{title:<24} n={s['n_scored']:<4} "
          f"D={s['D']:<6} F={s['F']:<6} DF-Score={s['DF_Score']:<6} "
          f"DF_Rate={s['DF_Rate']} HalRate={s['HalRate']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True,
                    help="flat record.json: [{id, lq, output, type?}]")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: df_<record stem>.json beside "
                         "the record)")
    ap.add_argument("--vlm", default=settings.VLM_BACKEND,
                    choices=sorted(settings.VLM_MODELS),
                    help=f"verifier model (default {settings.VLM_BACKEND})")
    ap.add_argument("--vlm-model", default=None,
                    help="explicit model id, overriding --vlm")
    ap.add_argument("--repeats", type=int, default=1,
                    help="score each pair N times and take the per-axis median "
                         "(default 1)")
    ap.add_argument("--max-side", type=int, default=0,
                    help="downscale both pictures to this long edge before "
                         "sending; 0 sends the original bytes, which is the "
                         "default because re-encoding perturbs the artifacts "
                         "this metric is asked about")
    ap.add_argument("--default-type", default=None,
                    help="degradation type for entries whose `type` is absent "
                         f"or empty (one of {', '.join(dfscore.KNOWN_TYPES)})")
    ap.add_argument("--no-audit", action="store_true",
                    help="skip the SSIM / chroma audit columns")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent verifier calls (default 4; the API tolerates "
                         "about 10 parallel requests across all processes)")
    ap.add_argument("--limit", type=int, default=0, help="first N entries only")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="recompute the aggregates from an existing output "
                         "file without calling the API")
    args = ap.parse_args()

    records_path = Path(args.records).resolve()
    out_path = Path(args.out).resolve() if args.out else \
        records_path.parent / f"df_{records_path.stem}.json"

    if args.aggregate_only:
        if not out_path.is_file():
            raise SystemExit(f"--aggregate-only needs an existing {out_path}")
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        per_item = prev.get("per_item", [])
        result = dict(prev)
        result.update(dfscore.score_rows(per_item, keys=["type"]))
        write_json(out_path, result)
        _print_summary("overall", result["overall"])
        for t, s in sorted(result.get("by_type", {}).items()):
            _print_summary(f"  {t}", s)
        print(f"output  : {out_path}")
        return 0

    records = json.loads(records_path.read_text(encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]

    print("=" * 68)
    print(f"records : {records_path}")
    print(f"verifier   : {args.vlm_model or args.vlm}"
          + (f"  x{args.repeats} (median)" if args.repeats > 1 else ""))
    print(f"metric  : {dfscore.METRIC_NAME}  "
          f"DF-Score = mean_i sqrt(D_i * F_i)")
    print(f"output  : {out_path}")
    print("=" * 68, flush=True)

    vlm = make_vlm(args.vlm, args.vlm_model)

    # One system prompt per type combination, built once and reused: assembling
    # it is pure string work but it is the same work for every image of a type.
    systems: dict[tuple[str, ...], str] = {}

    def _one(rec: dict) -> dict:
        sid = str(rec.get("id", ""))
        lq, out = rec.get("lq"), rec.get("output")
        entry = {"id": sid, "lq": lq, "output": out}
        if not lq or not Path(str(lq)).is_file():
            entry["error"] = f"input missing: {lq or '(empty)'}"
            return entry
        if not out or not Path(str(out)).is_file():
            entry["error"] = f"output missing: {out or '(empty)'}"
            return entry
        try:
            raw = rec.get("type")
            if (raw is None or raw == [] or raw == "") and args.default_type:
                raw = args.default_type
            types = dfscore.normalise_types(raw)
            entry["type"] = types
            key = tuple(types)
            if key not in systems:
                systems[key] = dfscore.build_system(types)
            entry["vlm"] = dfscore.score_pair(
                vlm, lq=lq, output=out, types=types,
                repeats=args.repeats, max_side=args.max_side,
                with_audit=not args.no_audit, system=systems[key])
        except Exception as e:                # noqa: BLE001 - one pair, one error
            entry["error"] = f"{type(e).__name__}: {e}"
        return entry

    per_item: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_one, rec): rec for rec in records}
        for i, fut in enumerate(as_completed(futures), 1):
            entry = fut.result()
            per_item.append(entry)
            if entry.get("error"):
                print(f"[{i}/{len(records)}] {entry['id']} ERROR "
                      f"{entry['error'][:60]}", flush=True)
            else:
                print(f"[{i}/{len(records)}] {entry['id']} "
                      f"{dfscore.format_line(entry['vlm'])}", flush=True)

    result = {
        "metric": dfscore.METRIC_NAME,
        "definition": "DF-Score = (1/N) * sum_i sqrt(D_i * F_i)",
        "records": str(records_path),
        "verifier": args.vlm_model or args.vlm,
        "repeats": args.repeats,
        "max_side": args.max_side,
        "n": len(records),
        "errors": [f"{e['id']}: {e['error']}" for e in per_item if e.get("error")],
        "per_item": per_item,
    }
    result.update(dfscore.score_rows(per_item, keys=["type"]))
    write_json(out_path, result)

    print("=" * 68)
    _print_summary("overall", result["overall"])
    for t, s in sorted(result.get("by_type", {}).items()):
        _print_summary(f"  {t}", s)
    if result["errors"]:
        print(f"failed  : {len(result['errors'])}")
    print(f"output  : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
