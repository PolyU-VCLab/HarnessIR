#!/usr/bin/env python3
"""HarnessIR: diagnosis, tool invocation, prompt composition, execution.

    python run_harness.py --manifest manifest.json \
        --lq-root /path/to/test-set --out runs/harness

Stages, in order:

    1  perception and diagnosis
                 a VLM reads the photograph and names what is wrong with it,
                 then nominates which perception tools would resolve the
                 questions it cannot answer by looking
    2  tool invocation
                 OCR, face detection, depth and segmentation run locally and
                 return structured evidence
    3  prompt composition
                 a VLM sees the photograph, the maps and the evidence, and
                 writes ONE prompt for the executor
    4  execution the executor receives the photograph and that prompt

With ``--redo`` the fifth stage is enabled:

    5  verification-driven refinement
                 a VLM scores the result against the input and names defects,
                 the prompt is rewritten, and the next round re-executes from
                 the ORIGINAL photograph

Redo is off by default, which is the single-pass configuration. The executor
only ever receives the photograph and text - the maps inform the prompt but are
never sent on, because an executor shown a depth map copies its colours into the
output.

Output layout: see README.md. Re-running the same command resumes.
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
from harness.clients import make_mfm, make_vlm                  # noqa: E402
from harness.layout import PER_IMAGE_DIRNAME, RESULTS_DIRNAME, result_exists  # noqa: E402
from harness.manifest import build_index, load_manifest         # noqa: E402
from harness.pipeline import run_sample                         # noqa: E402
from harness.runconfig import RunConfig                         # noqa: E402
from harness.utils import write_json                            # noqa: E402


def build_config(args, index: dict[str, dict]) -> RunConfig:
    """Assemble the run configuration from the file plus the CLI overrides."""
    cfg = RunConfig.from_file(args.config) if args.config else RunConfig()

    cfg.name = args.name or cfg.name
    cfg.images = list(index)
    cfg.lq_map = {sid: item["lq"] for sid, item in index.items()}
    cfg.gt_map = {sid: item["gt"] for sid, item in index.items() if item.get("gt")}
    cfg.type_map = {sid: item.get("type") for sid, item in index.items()}
    cfg.request_map = {sid: item["prompt"] for sid, item in index.items()
                       if item.get("prompt")}

    if args.vlm:
        cfg.vlm_backend = args.vlm
    if args.vlm_model:
        cfg.vlm_model = args.vlm_model
    if args.executor:
        cfg.mfm_backend = args.executor
    if args.workers is not None:
        cfg.workers = args.workers
    if args.redo:
        cfg.enable_redo = True
    if args.max_rounds is not None:
        cfg.max_rounds = args.max_rounds
    if args.redo_stop:
        cfg.redo_stop = args.redo_stop
    if args.redo_select:
        cfg.redo_select = args.redo_select
    if args.no_tools:
        cfg.enable_tools = False
    if args.no_iqa:
        cfg.enable_iqa = False
    if args.no_square1024:
        cfg.square1024 = False
    if args.no_resume:
        cfg.resume = False
    if args.intent:
        cfg.user_intent = args.intent
    cfg.out_root = str(Path(args.out).resolve())
    return cfg


def _read_existing_record(path: Path) -> list[dict]:
    """The ``record.json`` already on disk, or ``[]`` if there is none yet.

    An unreadable or corrupt file is treated as absent rather than fatal: the
    run has already spent its API calls by the time this is reached, and losing
    the run to a malformed bookkeeping file would be the worse outcome.
    """
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the full HarnessIR pipeline over a manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Output layout:")[0])

    ap.add_argument("--manifest", required=True,
                    help="JSON array or JSONL of {lq, gt?, prompt?, type?}. "
                         "`prompt` is the restoration request for that image and "
                         "is passed to the harness as well as to the baseline, so "
                         "both answer the same request; `--intent` overrides it for "
                         "the whole run. `type` states the degradation, which sets "
                         "the criteria the verifier judges against")
    ap.add_argument("--lq-root", default=None,
                    help="root used to derive sample ids (default: the "
                         "manifest's directory)")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--config", default=None,
                    help="run parameters as JSON: executor, VLM, round count and "
                         "max_rounds, metric list, worker count, geometry, colour "
                         f"adjustment (default: configs/default.json)")
    ap.add_argument("--name", default=None, help="run name, recorded in config.json")

    ap.add_argument("--vlm", default=None,
                    choices=sorted(settings.VLM_MODELS),
                    help=f"VLM for diagnosis, composition and verification "
                         f"(default {settings.VLM_BACKEND})")
    ap.add_argument("--vlm-model", default=None,
                    help="explicit model id, overriding --vlm")
    ap.add_argument("--executor", default=None,
                    choices=sorted(settings.MFM_GEMINI_ENDPOINTS),
                    help="image editing model; the choices are the keys of "
                         "MFM_GEMINI_ENDPOINTS in harness/settings.py "
                         f"(default {settings.MFM_BACKEND})")

    ap.add_argument("--redo", action="store_true",
                    help="enable stage 5: verify each result and re-execute "
                         "from a rewritten prompt (off by default)")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="total attempts per sample including the first, only "
                         "meaningful with --redo (default 3)")
    ap.add_argument("--redo-stop", default=None, choices=["verifier", "all"],
                    help="how many rounds to execute: 'verifier' stops as soon "
                         "as an attempt is accepted, 'all' runs every round up to "
                         "--max-rounds regardless of the verdict (default: the "
                         "config value)")
    ap.add_argument("--redo-select", default=None, choices=["verifier", "ssim"],
                    help="which executed round to deliver: 'verifier' takes the "
                         "one the verifier named, 'ssim' the one with the highest "
                         "grayscale SSIM against the LQ input (default: the "
                         "config value)")
    ap.add_argument("--no-tools", action="store_true",
                    help="skip stage 2; the composer works from the photograph "
                         "and the diagnosis alone")
    ap.add_argument("--no-iqa", action="store_true",
                    help="skip the per-sample metrics (run compute_iqa.py later)")
    ap.add_argument("--no-square1024", action="store_true",
                    help="keep the native resolution instead of resizing the "
                         "long edge to 1024")
    ap.add_argument("--intent", default=None,
                    help="one restoration request for the whole run, shown to the "
                         "diagnoser and the composer; overrides the manifest's "
                         "per-image `prompt`")

    ap.add_argument("--workers", type=int, default=None,
                    help="samples processed concurrently (default 4; the API "
                         "tolerates about 10 parallel requests in total)")
    ap.add_argument("--limit", type=int, default=0, help="first N samples only")
    ap.add_argument("--images", default=None,
                    help="comma-separated sample ids to run")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run samples that already have a delivered image")
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

    cfg = build_config(args, index)
    out_root = Path(cfg.out_root)
    (out_root / RESULTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (out_root / PER_IMAGE_DIRNAME).mkdir(parents=True, exist_ok=True)
    write_json(out_root / "config.json", cfg.to_dict())

    todo = list(index)
    if cfg.resume:
        todo = [s for s in todo if not result_exists(out_root / RESULTS_DIRNAME, s)]
        skipped = len(index) - len(todo)
    else:
        skipped = 0

    print("=" * 68)
    for line in cfg.summary_lines():
        print(line)
    print(f"samples  : {len(todo)} to run"
          + (f", {skipped} already done" if skipped else ""))
    print(f"output   : {out_root}")
    print("=" * 68, flush=True)

    if not todo:
        print("nothing to do (every sample already has a result)")
        return 0

    # One client per role. They are separate objects so a per-stage model
    # override stays possible without re-reading the configuration.
    # vlm_model is None when the config does not name one; the client then falls
    # back to settings.VLM_MODEL_NAME.
    vlm = make_vlm(model=cfg.vlm_model,
                   max_output_tokens=cfg.vlm_max_output_tokens)
    vlms = {"diagnoser": vlm, "composer": vlm, "verifier": vlm}
    mfm = make_mfm(model=cfg.mfm_backend)

    records = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
        futures = {ex.submit(run_sample, vlms, mfm, sid, cfg, out_root): sid
                   for sid in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:            # noqa: BLE001 - keep the batch alive
                print(f"[{i}/{len(todo)}] {sid} CRASH {type(e).__name__}: {e}",
                      flush=True)
                continue
            records.append(rec)
            tail = ""
            if rec.error:
                tail = f" ERROR {rec.error[:60]}"
            elif rec.accepted is not None:
                tail = f" accepted={rec.accepted} rounds={rec.n_rounds}"
            if rec.iqa:
                from harness.metrics import format_iqa
                tail += f"  {format_iqa(rec.iqa)}"
            print(f"[{i}/{len(todo)}] {sid}{tail}", flush=True)

    elapsed = time.perf_counter() - t0
    ok = [r for r in records if not r.error]

    # Flat record for the scorers, in the shape compute_iqa.py and
    # compute_df.py expect.
    flat = [{"id": r.sample_id,
             "lq": cfg.lq_map.get(r.sample_id),
             "gt": (cfg.gt_map or {}).get(r.sample_id),
             "type": r.task_types,
             "output": r.result_copy or r.final_output,
             "error": r.error}
            for r in records]

    # A resumed run only processes the samples still outstanding, so writing
    # `flat` on its own would replace a full 200-entry record with the handful
    # that were left over - the finished samples' rows would be lost while their
    # images stayed on disk. Merge instead: this run's rows override the stored
    # ones for the same id, and every other id is carried over unchanged.
    record_path = out_root / "record.json"
    stored = _read_existing_record(record_path)
    order = [str(e.get("id")) for e in stored]
    for r in records:                          # this run's ids go last, in run order
        if r.sample_id not in order:
            order.append(r.sample_id)
    merged: dict[str, dict] = {str(e.get("id")): e for e in stored}
    for e in flat:
        merged[str(e["id"])] = e
    write_json(record_path, [merged[k] for k in order if k in merged])
    write_json(out_root / "summary.json", {
        "config": cfg.to_dict(),
        "n": len(records), "n_ok": len(ok),
        "n_skipped": skipped,
        "errors": [r.sample_id for r in records if r.error],
        "total_elapsed_sec": round(elapsed, 1),
        "api_calls": {
            k: sum(r.api_calls.get(k, 0) for r in records)
            for k in ("vlm", "mfm", "tools")},
        "per_sample": [r.to_dict() for r in records],
    })

    print("=" * 68)
    print(f"done {len(ok)}/{len(records)}  failed {len(records) - len(ok)}  "
          f"in {elapsed / 60:.1f} min")
    print(f"images  : {out_root / RESULTS_DIRNAME}")
    print(f"record  : {out_root / 'record.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
