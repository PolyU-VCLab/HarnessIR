#!/usr/bin/env python3
"""Per-sample pipeline.

Five stages, named after the paper:

    stage 1  perception and diagnosis   VLM reads the photo, names degradations,
                                        and plans the auxiliary tools
    stage 2  tool invocation            OCR / faces / depth / segmentation run
    stage 3  prompt composition         VLM sees photo + maps + evidence, and
                                        writes ONE prompt
    stage 4  execution                  executor gets the photo + that prompt
    ------- stages 1-4 always run; the next two only when redo is enabled -----
    stage 5  verification               VLM judges the result against the input
             refinement                 and revises the prompt; the next round
                                        re-executes from the ORIGINAL photo

A sample that throws is recorded and returned, not raised: one broken image
should not end a batch.

Redo is off by default. With ``enable_redo=False`` the loop runs exactly one
round and stage 5 never executes, which is the configuration the single-pass
results are produced under.
"""
from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import layout as L
from .runconfig import RunConfig
from .selection import select_final
from .stages.execute import prepare_lq, run_execute
from .stages.compose import run_compose
from .stages.diagnosis import run_diagnosis
from .stages.refine import revise_prompt
from .stages.tools import ToolEvidence, run_tools
from .stages.verify import run_verify
from .stages.visuals import build_visual_pack, render_text_evidence
from .utils import parse_types, types_key
from .utils import write_json


@dataclass
class SampleRecord:
    sample_id: str
    task_types: list[str] = field(default_factory=list)
    user_intent: str | None = None

    diagnosis: dict[str, Any] = field(default_factory=dict)
    tool_evidence: dict[str, Any] = field(default_factory=dict)
    visual_pack: dict[str, Any] = field(default_factory=dict)
    composition: dict[str, Any] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    iqa: dict[str, float] = field(default_factory=dict)

    stage_times: dict[str, float] = field(default_factory=dict)
    api_calls: dict[str, int] = field(
        default_factory=lambda: {"vlm": 0, "mfm": 0, "tools": 0})
    accepted: bool | None = None
    final_output: str | None = None
    result_copy: str | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    selected_round: int | None = None
    selection: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_type": types_key(self.task_types),
            "task_types": self.task_types,
            "user_intent": self.user_intent,
            "diagnosis": self.diagnosis,
            "tool_evidence": self.tool_evidence,
            "visual_pack": self.visual_pack,
            "composition": self.composition,
            "rounds": self.rounds,
            "iqa": self.iqa,
            "n_rounds": self.n_rounds,
            "accepted": self.accepted,
            "final_output": self.final_output,
            "result_copy": self.result_copy,
            "geometry": self.geometry,
            "selected_round": self.selected_round,
            "selection": self.selection,
            "error": self.error,
            "stage_times": self.stage_times,
            "api_calls": self.api_calls,
        }


def _md5(path: str | Path) -> str | None:
    """Fingerprint of one round's output.

    Every attempt re-executes from the original photograph with a rewritten
    prompt, so two identical outputs mean the prompt did not actually
    change - the round's budget was spent for nothing.
    """
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def run_sample(vlms: dict[str, Any], mfm, img_id: str, cfg: RunConfig,
               out_root: Path) -> SampleRecord:
    """Run every stage for one sample.

    Produces ``out_root/per_image/<img_id>/`` plus a copy of the delivered image
    in ``out_root/results/``.
    """
    # The request this image is restored under: a per-image manifest prompt if
    # the manifest carries one, else the run-wide --intent.
    request = cfg.request_for(img_id)
    rec = SampleRecord(sample_id=img_id, user_intent=request)

    lq = cfg.lq_path_for(img_id)
    if lq is None:
        raise FileNotFoundError(f"no input image for {img_id!r}: not in the "
                                f"manifest")
    # Degradation types come from the manifest. An entry without one is scored
    # as a generic mix rather than guessed at, because the type selects the
    # judge's criteria and a wrong guess silently changes the yardstick.
    rec.task_types = (parse_types((cfg.type_map or {}).get(img_id))
                      or ["mix"])

    out_dir = out_root / L.PER_IMAGE_DIRNAME / img_id
    out_dir.mkdir(parents=True, exist_ok=True)
    L.ensure_layout(out_dir)

    # Chain input geometry: long edge 1024, no padding. Padding is applied only
    # around the executor call (see stages/execute.py for why).
    crop_back = None
    if cfg.square1024:
        lq, geom = prepare_lq(lq, out_dir)
        rec.geometry = geom
        crop_back = (tuple(geom["original_size"]), geom["scale"])

    def timed(stage: str, fn: Callable, *, vlm: int = 0, mfm_calls: int = 0,
              tools: int = 0):
        t0 = time.perf_counter()
        result = fn()
        rec.stage_times[stage] = round(time.perf_counter() - t0, 1)
        rec.api_calls["vlm"] += vlm
        rec.api_calls["mfm"] += mfm_calls
        rec.api_calls["tools"] += tools
        return result

    try:
        # ---- stage 1: perception and diagnosis --------------------------
        diagnosis = timed("stage1_diagnosis",
                          lambda: run_diagnosis(vlms["diagnoser"], lq,
                                                user_intent=request,
                                                task_type=types_key(rec.task_types)),
                          vlm=1)
        rec.diagnosis = diagnosis.to_dict()
        write_json(L.json_path(out_dir, "diagnosis.json"), rec.diagnosis)

        # ---- stage 2: on-demand tool invocation ------------------------
        if cfg.enable_tools:
            requested = list(diagnosis.requested_tools)
            evidence = timed("stage2_tools",
                             lambda: run_tools(lq, requested, out_dir),
                             tools=len(requested))
        else:
            evidence = ToolEvidence()
            evidence.unavailable.append("stage 2 disabled by configuration")
        rec.tool_evidence = evidence.to_dict()

        # Move artifacts into their subdirectories and repoint the evidence at
        # them. Must happen before the pack is built: the pack reads the maps
        # back through those paths.
        rec.tool_evidence = L.archive_tool_outputs(out_dir, rec.tool_evidence)
        write_json(L.json_path(out_dir, "tool_evidence.json"), rec.tool_evidence)

        # ---- stage 3: prompt composition -------------------------------
        pack = timed("build_visual_pack",
                     lambda: build_visual_pack(
                         lq, rec.tool_evidence, out_dir,
                         include_seg=True, include_depth=True,
                         depth_style=cfg.depth_picture_style,
                         expect_tools=cfg.enable_tools))
        rec.visual_pack = pack.to_dict()
        write_json(L.json_path(out_dir, "visual_pack.json"), rec.visual_pack)

        text_evidence = render_text_evidence(rec.tool_evidence, pack)
        L.text_path(out_dir, "text_evidence.txt").write_text(text_evidence,
                                                            encoding="utf-8")

        plan = timed("stage3_compose",
                     lambda: run_compose(vlms["composer"], rec.task_types,
                                         rec.diagnosis, text_evidence, pack,
                                         request),
                     vlm=1)
        rec.composition = plan.to_dict()
        write_json(L.json_path(out_dir, "composition.json"), rec.composition)
        L.text_path(out_dir, "prompt.txt").write_text(plan.prompt,
                                                      encoding="utf-8")

        # ---- stage 4: execution, or stage 4+5 when redo is on ----------
        prompt = plan.prompt
        max_rounds = cfg.max_rounds if cfg.enable_redo else 1

        for rnd in range(max_rounds):
            fname = "output.png" if rnd == 0 else f"output_r{rnd}.png"
            entry: dict[str, Any] = {
                "round": rnd,
                "prompt": prompt,
                "prompt_chars": len(prompt),
            }

            # The executor receives the photograph and the prompt text.
            # This is the whole contract - no maps, no preamble, no legend.
            L.text_path(out_dir, fname.replace(".png", "_prompt.txt")).write_text(
                prompt, encoding="utf-8")

            ex = timed(f"stage4_execute_r{rnd}",
                       lambda: run_execute(mfm, lq, prompt, out_dir,
                                           filename=fname, crop_back=crop_back),
                       mfm_calls=1)
            entry["execute"] = ex
            if ex.get("error"):
                rec.error = f"stage 4 round {rnd}: {ex['error']}"
                rec.rounds.append(entry)
                break

            out_img = ex["path"]
            entry["output"] = out_img
            entry["execute"]["md5"] = _md5(out_img)
            for prev in rec.rounds:
                if prev.get("execute", {}).get("md5") == entry["execute"]["md5"]:
                    entry["duplicate_of"] = prev["round"]
                    break

            # ---- colour alignment -------------------------------------------
            # Runs here, before the round is scored or judged, so the judge, the
            # selector and the metrics all see the image that is actually
            # delivered. The aligned image replaces the executor's own output:
            # one file per round is kept, and it is the one that ships.
            if cfg.color_adjust:
                from .color_adjust import adjust_in_place
                adj = timed(f"color_adjust_r{rnd}",
                            lambda: adjust_in_place(rec.task_types, lq, out_img))
                entry["color_adjust"] = adj

            # Per-round NR scores (no ground truth needed, cheap). Used to
            # break ties when the judge's continuous scores tie.
            if cfg.enable_iqa:
                from .metrics import compute_iqa
                entry["iqa"] = timed(
                    f"iqa_r{rnd}",
                    lambda: compute_iqa(out_img, None, nr_metrics=cfg.nr_metrics,
                                        fr_metrics=[]))

            if not cfg.enable_redo:
                entry["verifier"] = None
                rec.rounds.append(entry)
                break

            # ---- stage 5: verification ---------------------------------
            # Picture 1 is the ORIGINAL input, never a previous attempt, so
            # fidelity is always measured against the source.
            prev_rounds = [
                {"path": e["output"],
                 "gates": (e.get("verifier") or {}).get("gates"),
                 "iqa": e.get("iqa")}
                for e in rec.rounds
                if e.get("output") and Path(e["output"]).is_file()
            ]
            judge = timed(
                f"stage5_verify_r{rnd}",
                lambda: run_verify(vlms["verifier"], task_types=rec.task_types,
                                  lq_path=lq, result_path=out_img,
                                  prev_rounds=prev_rounds,
                                  iqa=entry.get("iqa")),
                vlm=1)
            entry["verifier"] = judge.to_dict()
            rec.rounds.append(entry)

            # Stop as soon as the judge is satisfied, unless the run is set to
            # execute every round regardless (redo_stop="all").
            stop_on_accept = cfg.redo_stop != "all"
            if rnd == max_rounds - 1 or (judge.accepted and stop_on_accept):
                break

            # ---- stage 5: refinement -----------------------------------
            # Same composer, same evidence, plus the previous prompt and the
            # flawed result. The revision wants everything stage 3 saw, so the
            # diagnosis report, the text evidence and the map paths all go back in.
            refine = timed(
                f"stage5_revise_r{rnd}",
                lambda: revise_prompt(
                    vlms["composer"], rec.task_types, rec.diagnosis, text_evidence,
                    pack, prompt, judge.defect_summary(),
                    out_dir / L.DIR_TEXT, failed_output=out_img,
                    round_idx=rnd + 1),
                vlm=1)
            prompt = refine["new_prompt"]
            rec.rounds[-1]["refine"] = refine

        # ---- best-of-N selection ---------------------------------------
        selection = select_final(rec.rounds, rule=cfg.redo_select)
        rec.selection = selection
        rec.selected_round = selection.get("round")
        rec.final_output = selection.get("path")
        # Accepted describes the SELECTED round, not the last one. The two
        # differ whenever the selector picks an earlier attempt, and reporting
        # the other round's verdict would describe an image nobody receives.
        if selection.get("index") is not None:
            verdict = rec.rounds[selection["index"]].get("verifier")
            rec.accepted = (bool(verdict.get("accepted")) if verdict else None)

        # ---- IQA on the delivered image --------------------------------
        if cfg.enable_iqa and rec.final_output:
            from .metrics import compute_iqa
            gt = cfg.gt_path_for(img_id)
            rec.iqa = timed(
                "iqa_final",
                lambda: compute_iqa(rec.final_output, gt,
                                    nr_metrics=cfg.nr_metrics,
                                    fr_metrics=cfg.fr_metrics if gt else []))

    except Exception as e:                    # noqa: BLE001 - record, do not raise
        rec.error = f"{type(e).__name__}: {e}"
        (out_dir / "error.txt").write_text(traceback.format_exc(),
                                           encoding="utf-8")

    # Collect whatever exists. Outside the try block: a sample that failed
    # halfway still produced images worth keeping.
    rec.result_copy = L.collect_result(rec.final_output,
                                       out_root / L.RESULTS_DIRNAME, img_id)
    write_json(out_dir / "record.json", rec.to_dict())
    return rec
