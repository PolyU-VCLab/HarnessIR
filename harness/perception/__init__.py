"""Local perception tools (T1 OCR / T2 FACE / T3 DEPTH / T4 SEGMENTATION).

Every tool follows the same soft-degradation contract: on any failure it prints
one line and returns ``None``. A missing tool must never abort an image - the
composer is written to work from whatever evidence actually arrived, and a hard
failure here would cost the whole sample.

Models are loaded lazily and cached process-wide, so a batch run pays the load
cost once. ``MODEL_LOAD_LOCK`` serialises loading only; inference is unlocked.
"""
from __future__ import annotations

from .depth import DepthOut, run_depth
from .faces import FaceEvidence, run_faces
from .ocr import OcrOut, run_ocr
from .segment import SegOut, run_segment

__all__ = ["DepthOut", "run_depth", "FaceEvidence", "run_faces",
           "OcrOut", "run_ocr", "SegOut", "run_segment"]
