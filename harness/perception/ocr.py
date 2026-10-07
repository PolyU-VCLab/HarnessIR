#!/usr/bin/env python3
"""T1 OCR - text transcription with per-line boxes (PP-OCRv6).

A classic two-stage detector + recogniser rather than an end-to-end document
VLM, because restoration needs to know WHERE each string is: "sharpen the sign
in the upper left without changing what it says" is only expressible with a
box. Document VLMs read better but their per-line boxes are not dependable.

Transcriptions are evidence, not targets. The composer quotes them so the
executor re-renders the same glyphs; it must never treat a low-confidence
reading as the true text and "correct" it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .. import settings
from ._models import MODEL_LOAD_LOCK

_OCR = None


@dataclass
class OcrOut:
    texts: list[dict] = field(default_factory=list)   # [{string, confidence, bbox_norm}]
    n_texts: int = 0
    engine: str = ""


def _engine():
    """Lazily build the PP-OCRv6 pipeline (one instance per process).

    PP-OCRv6 runs a static graph and is safe to call from several threads, so
    only construction is serialised.
    """
    global _OCR
    if _OCR is not None:
        return _OCR
    with MODEL_LOAD_LOCK:
        if _OCR is not None:
            return _OCR
        from paddleocr import PaddleOCR

        # paddlex accepts "gpu:N", not "cuda:N".
        dev = settings.tool_device()
        if dev.startswith("cuda:"):
            dev = "gpu:" + dev.split(":", 1)[1]
        elif dev == "cuda":
            dev = "gpu"

        t0 = time.time()
        det = settings.PP_OCR_DET_DIR
        rec = settings.PP_OCR_REC_DIR
        _OCR = PaddleOCR(
            # Pin the version explicitly; the package default may drift.
            ocr_version="PP-OCRv6",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # None lets paddlex resolve from its own cache.
            text_detection_model_dir=str(det) if det.is_dir() else None,
            text_recognition_model_dir=str(rec) if rec.is_dir() else None,
            device=dev,
        )
        print(f"    T1 OCR loaded in {time.time() - t0:.1f}s", flush=True)
    return _OCR


def _parse(res: dict, width: int, height: int) -> list[dict]:
    """Flatten a PP-OCRv6 result into ``{string, confidence, bbox_norm}``.

    ``bbox_norm`` is the axis-aligned hull of the detected polygon, normalised
    to [0, 1] - the same convention every other tool in this package uses.
    """
    out: list[dict] = []
    for text, score, poly in zip(res.get("rec_texts") or [],
                                 res.get("rec_scores") or [],
                                 res.get("rec_polys") or []):
        s = (text if isinstance(text, str) else str(text)).strip()
        if not s:
            continue
        try:
            xs = [float(x) for x, _ in poly]
            ys = [float(y) for _, y in poly]
            bbox = [round(min(xs) / width, 3), round(min(ys) / height, 3),
                    round(max(xs) / width, 3), round(max(ys) / height, 3)]
        except (TypeError, ValueError):
            bbox = [0.0, 0.0, 1.0, 1.0]
        out.append({"string": s, "confidence": float(score), "bbox_norm": bbox})
    return out


def run_ocr(img: Image.Image, out_dir: Path) -> OcrOut | None:
    """Transcribe text in ``img``. Returns ``None`` when the engine is unusable.

    An empty ``texts`` list is a real result (no text in the image) and is
    distinct from ``None`` (the tool could not run).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # The backend reads from a path, so stage the PIL image next to the outputs.
    staged = out_dir / "_ocr_input.png"
    img.save(staged)
    try:
        res = _engine().predict(str(staged))[0].json["res"]
        texts = _parse(res, img.width, img.height)
    except Exception as e:                    # noqa: BLE001 - soft degradation
        print(f"    T1 OCR skipped: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        staged.unlink(missing_ok=True)

    from ..utils import write_json
    write_json(out_dir / "ocr.json",
               {"engine": "ppocr-v6", "n_texts": len(texts), "texts": texts[:50]})
    return OcrOut(texts=texts, n_texts=len(texts), engine="ppocr-v6")
