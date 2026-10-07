#!/usr/bin/env python3
"""Compute image quality metrics over a run's delivered images.

    python compute_iqa.py --records runs/harness/record.json
    python compute_iqa.py --records runs/harness/record.json --nr-only
    python compute_iqa.py --records runs/baseline/record.json --gt

Metrics:

    NR (result only)          MANIQA  CLIP-IQA  MUSIQ  TOPIQ  AFINE-NR
    FR (result vs ground GT)  PSNR  SSIM  LPIPS  DISTS

FR metrics need a ground truth and are skipped without one. NR metrics are
reported per type and per group as well as overall, because the axes do not
behave the same way across degradations.

``--gt`` scores the ground truth itself instead of the result, giving an upper
reference for the no-reference metrics.

Output: ``iqa_<record stem>.json`` beside the record, or the path given with
``--out``.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness.metrics import (DEFAULT_FR_METRICS, DEFAULT_NR_METRICS,  # noqa: E402
                             compute_iqa, format_iqa)
from harness.utils import write_json                                  # noqa: E402


def _means(entries: list[dict], names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        vals = [e["iqa"][name] for e in entries
                if name in (e.get("iqa") or {})]
        if vals:
            out[name] = round(sum(vals) / len(vals), 4)
    return out


def _group(entries: list[dict], key: str) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for e in entries:
        v = e.get(key)
        if isinstance(v, list):
            v = "|".join(v)
        buckets.setdefault(str(v or "(unset)"), []).append(e)
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True,
                    help="flat record.json produced by run_harness.py / "
                         "run_baseline.py: [{id, lq, gt?, output, type?}]")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: iqa_<record stem>.json beside "
                         "the record)")
    ap.add_argument("--workers", type=int, default=4,
                    help="scoring threads (default 4); each metric instance is "
                         "serialised internally, so more threads mainly overlap "
                         "loading with inference")
    ap.add_argument("--nr", default=None,
                    help="comma-separated NR metrics "
                         f"(default {','.join(DEFAULT_NR_METRICS)})")
    ap.add_argument("--fr", default=None,
                    help="comma-separated FR metrics "
                         f"(default {','.join(DEFAULT_FR_METRICS)})")
    ap.add_argument("--nr-only", action="store_true",
                    help="skip the FR metrics, even when a ground truth exists")
    ap.add_argument("--gt", action="store_true",
                    help="score the ground truth itself instead of the result; "
                         "FR is skipped since a ground truth against itself is "
                         "meaningless")
    args = ap.parse_args()

    nr = [s.strip() for s in args.nr.split(",")] if args.nr else None
    fr = [] if (args.nr_only or args.gt) else (
        [s.strip() for s in args.fr.split(",")] if args.fr
        else list(DEFAULT_FR_METRICS))

    records_path = Path(args.records).resolve()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{records_path} must be a JSON array")

    out_path = Path(args.out).resolve() if args.out else \
        records_path.parent / f"iqa_{records_path.stem}.json"

    print("=" * 68)
    print(f"records : {records_path}")
    print(f"NR      : {', '.join(nr or DEFAULT_NR_METRICS)}"
          + ("  (scoring the ground truth)" if args.gt else ""))
    print(f"FR      : {', '.join(fr) if fr else 'skipped'}")
    print(f"output  : {out_path}")
    print("=" * 68, flush=True)

    def _one(rec: dict) -> dict:
        sid = str(rec.get("id", ""))
        target = rec.get("gt") if args.gt else rec.get("output")
        reference = None if args.gt else rec.get("gt")
        entry = {"id": sid, "type": rec.get("type"), "lq": rec.get("lq"),
                 "gt": rec.get("gt"), "output": rec.get("output")}
        if not target or not Path(str(target)).is_file():
            kind = "ground truth" if args.gt else "output"
            entry["error"] = f"{kind} missing: {target or '(empty)'}"
            return entry
        # GT mode skips FR: a ground truth has no reference of its own.
        entry["iqa"] = compute_iqa(target, reference if fr else None,
                                   nr_metrics=nr, fr_metrics=fr)
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
                      f"{format_iqa(entry['iqa'])}", flush=True)

    ok = [e for e in per_item if not e.get("error")]
    names = list(nr or DEFAULT_NR_METRICS) + fr

    result = {
        "records": str(records_path),
        "target": "gt" if args.gt else "output",
        "nr_metrics": list(nr or DEFAULT_NR_METRICS),
        "fr_metrics": fr,
        "n": len(records), "n_ok": len(ok),
        "errors": [f"{e['id']}: {e['error']}" for e in per_item if e.get("error")],
        "mean": _means(ok, names),
        "mean_by_type": {k: _means(v, names) for k, v in _group(ok, "type").items()},
        "per_item": per_item,
    }
    write_json(out_path, result)

    print("=" * 68)
    print(f"scored {len(ok)}/{len(records)}")
    if result["mean"]:
        print("overall : " + format_iqa(result["mean"]))
    for t, m in result["mean_by_type"].items():
        if m:
            print(f"[{t}] " + format_iqa(m))
    print(f"output  : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
