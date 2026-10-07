#!/usr/bin/env python3
"""Baseline: execute the prompt that ships in the manifest.

This is the direct-use path - read a manifest, take the ``prompt`` field of each
entry, and hand it to the executor with the input image. No diagnosis, no tools,
no prompt composition. It exists as the reference point the harness is measured
against, and it is what "just ask the editing model" produces.

    python run_baseline.py --manifest manifest.json \
        --lq-root /path/to/test-set --out runs/baseline

Output layout:

    runs/baseline/
    |-- config.json          the settings this run used
    |-- summary.json         per-sample records and aggregate statistics
    |-- record.json          flat [{id, lq, gt, prompt, output}] for the scorers
    `-- results/<id>.png     one image per sample

Re-running the same command skips samples that already have an output; pass
``--no-resume`` to force a full re-run.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import settings                                    # noqa: E402
from harness.clients import make_mfm                            # noqa: E402
from harness.layout import collect_result, result_exists        # noqa: E402
from harness.manifest import build_index, load_manifest         # noqa: E402
from harness.stages.execute import crop_black_border, pad_to_square  # noqa: E402
from harness.utils import write_json                            # noqa: E402

DEFAULT_PROMPT = ("Restore this low-quality image to a clear, clean and natural "
                  "state.")


def _run_one(mfm, sid: str, item: dict, out_dir: Path, results_dir: Path, *,
             square1024: bool, resume: bool, color_adjust: bool) -> dict:
    """Execute one sample. Failures are recorded, never raised."""
    from PIL import Image

    lq = Path(item["lq"])
    out_file = out_dir / "results" / f"{sid}.png"
    rec = {
        "id": sid, "lq": str(lq), "gt": item.get("gt"),
        "prompt": item.get("prompt") or DEFAULT_PROMPT,
    }

    if resume and result_exists(out_dir / "results", sid):
        rec["output"] = str(out_file)
        rec["skipped"] = True
        return rec

    t0 = time.perf_counter()
    try:
        with Image.open(lq) as im:
            original_size = im.size

        if square1024:
            with Image.open(lq) as im:
                send = pad_to_square(im.convert("RGB"))
            work = out_dir / "square_inputs" / sid
            work.mkdir(parents=True, exist_ok=True)
            staged = work / "input_sq1024.png"
            send.save(staged)
            input_size = (settings.SQUARE, settings.SQUARE)
        else:
            staged = lq
            input_size = original_size

        img = mfm.edit([str(staged)], rec["prompt"])

        if square1024:
            # The executor chooses its own output size from the aspect ratio, so
            # normalise to the canvas before computing the crop box.
            if img.size != input_size:
                img = img.resize(input_size, Image.LANCZOS)
            scale = settings.SQUARE / max(original_size)
            img = crop_black_border(img, original_size, scale)
        elif img.size != input_size:
            img = img.resize(input_size, Image.LANCZOS)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_file)

        # Colour alignment, applied before the image is collected so the scorers
        # see the same image that ships. The aligned image replaces the
        # executor's own output: one file per sample is kept.
        if color_adjust:
            from harness.color_adjust import adjust_in_place
            from harness.utils import parse_types
            # The colour mode is chosen from the manifest's type labels, the same
            # key the harness path uses. An entry with no type is a generic mix.
            rec["color_adjust"] = adjust_in_place(
                parse_types(item.get("type")) or ["mix"], lq, out_file)

        rec["output"] = str(out_file)
        rec["result_copy"] = collect_result(str(out_file), results_dir, sid)
        rec["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    except Exception as e:                    # noqa: BLE001 - one sample, one error
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Execute the manifest prompt directly, without the harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Output layout:")[0])
    ap.add_argument("--manifest", required=True,
                    help="JSON array or JSONL of {lq, gt?, prompt?, type?}")
    ap.add_argument("--lq-root", default=None,
                    help="root used to derive sample ids from the input paths "
                         "(default: the manifest's own directory)")
    ap.add_argument("--out", required=True,
                    help="output run directory; created if missing")
    ap.add_argument("--executor", default=settings.MFM_BACKEND,
                    choices=sorted(settings.MFM_GEMINI_ENDPOINTS),
                    help=f"image editing model (default: {settings.MFM_BACKEND})")
    ap.add_argument("--workers", type=int, default=4,
                    help="samples processed concurrently (default 4)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N samples, for a smoke test")
    ap.add_argument("--images", default=None,
                    help="comma-separated sample ids to run")
    ap.add_argument("--no-square1024", action="store_true",
                    help="send the input at its native size instead of padding "
                         "to a 1024x1024 square (default: square)")
    ap.add_argument("--no-color-adjust", action="store_true",
                    help="skip the post-executor colour alignment. Alignment is "
                         "ON by default so the scored image is the shipped image")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run samples that already have an output")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    items = load_manifest(manifest_path)
    lq_root = Path(args.lq_root).resolve() if args.lq_root else None
    index = build_index(items, lq_root=lq_root, manifest_dir=manifest_path.parent)

    if args.images:
        wanted = [s.strip() for s in args.images.split(",") if s.strip()]
        missing = [s for s in wanted if s not in index]
        if missing:
            raise SystemExit(f"--images names {len(missing)} id(s) not in the "
                             f"manifest: {missing[:5]}")
        index = {k: index[k] for k in wanted}
    if args.limit:
        index = dict(list(index.items())[:args.limit])

    out_dir = Path(args.out).resolve()
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    square1024 = not args.no_square1024
    cfg = {
        "mode": "baseline",
        "manifest": str(manifest_path),
        "lq_root": str(lq_root) if lq_root else None,
        "executor": args.executor,
        "workers": args.workers,
        "square1024": square1024,
        "color_adjust": not args.no_color_adjust,
        "prompts": "manifest `prompt` field",
    }
    write_json(out_dir / "config.json", cfg)

    print("=" * 68)
    print("Baseline: manifest prompt, executed directly")
    print(f"executor : {args.executor}")
    print(f"geometry : {'square 1024' if square1024 else 'native size'}")
    print(f"colour   : {'align to input' if not args.no_color_adjust else 'off'}")
    print(f"samples  : {len(index)}")
    print(f"output   : {out_dir}")
    print("=" * 68, flush=True)

    mfm = make_mfm(model=args.executor)
    records: list[dict] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_run_one, mfm, sid, item, out_dir, results_dir,
                             square1024=square1024, resume=not args.no_resume,
                             color_adjust=not args.no_color_adjust): sid
                   for sid, item in index.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            if rec.get("skipped"):
                print(f"[{i}/{len(index)}] {rec['id']} SKIP (already done)",
                      flush=True)
            elif rec.get("error"):
                print(f"[{i}/{len(index)}] {rec['id']} ERROR "
                      f"{rec['error'][:70]}", flush=True)
            else:
                print(f"[{i}/{len(index)}] {rec['id']} OK "
                      f"{rec.get('elapsed_sec', 0):.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    ok = [r for r in records if r.get("output") and not r.get("error")]
    write_json(out_dir / "record.json", records)
    write_json(out_dir / "summary.json", {
        "config": cfg,
        "n": len(records), "n_ok": len(ok),
        "n_skipped": sum(1 for r in records if r.get("skipped")),
        "errors": [r["id"] for r in records if r.get("error")],
        "total_elapsed_sec": round(elapsed, 1),
        "per_sample": records,
    })

    print("=" * 68)
    print(f"done {len(ok)}/{len(records)}  failed "
          f"{sum(1 for r in records if r.get('error'))}  "
          f"in {elapsed:.0f}s")
    print(f"images  : {results_dir}")
    print(f"record  : {out_dir / 'record.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
