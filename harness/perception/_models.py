#!/usr/bin/env python3
"""Lazy, process-wide model cache shared by the perception tools.

Loading is serialised by ``MODEL_LOAD_LOCK``. Two reasons:

* SAM3 reads a multi-GB checkpoint; loading it concurrently with another torch
  model triggers a meta-tensor race in ultralytics (parameters come back as
  meta tensors and ``setup_model`` raises "Cannot copy out of meta tensor").
* Concurrent first-time loads would each allocate their own peak, which is what
  actually causes OOM on a shared GPU rather than steady-state inference.

Inference is deliberately NOT locked - once a model is resident, concurrent
calls are safe and the pipeline relies on that for its thread pool.
"""
from __future__ import annotations

import threading
from typing import Any

from .. import settings

MODEL_LOAD_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _cached(key: str, build):
    """Double-checked lazy init: build ``key`` at most once per process."""
    if key in _CACHE:
        return _CACHE[key]
    with MODEL_LOAD_LOCK:
        if key not in _CACHE:                 # another thread may have won
            _CACHE[key] = build()
    return _CACHE[key]


def depth_pipeline():
    """Depth-Anything-V2-Base monocular depth estimator."""
    def build():
        from transformers import pipeline
        dev = settings.tool_device()
        idx = int(dev.split(":")[1]) if dev.startswith("cuda") else -1
        return pipeline("depth-estimation", model=settings.DEPTH_MODEL, device=idx)
    return _cached("depth", build)


def sam3_predictor():
    """SAM3 open-vocabulary semantic predictor.

    The checkpoint must include the language backbone / text encoder. A
    conversion that dropped it loads without error but makes
    ``forward_grounding`` return NaN for every mask.
    """
    def build():
        from ultralytics.models.sam.build_sam3 import build_sam3_image_model
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        # clip_anytorch and the ultralytics CLIP fork install into the same
        # ``clip/`` package; whichever lands second wins. The anytorch copy of
        # simple_tokenizer.py has no ``SimpleTokenizer.__call__``, so
        # VETextEncoder.forward raises "'SimpleTokenizer' object is not
        # callable". Restore an equivalent implementation.
        import clip
        from clip.simple_tokenizer import SimpleTokenizer
        if "__call__" not in SimpleTokenizer.__dict__:
            SimpleTokenizer.__call__ = (
                lambda self, texts, context_length=77:
                clip.tokenize(texts, context_length=context_length, truncate=True))

        net = build_sam3_image_model(settings.SAM3_CKPT)
        pred = SAM3SemanticPredictor(overrides={
            "conf": settings.SAM3_CONF,
            "iou": settings.SAM3_IOU,
            "imgsz": settings.SAM3_IMGSZ,
            "device": settings.tool_device(),
        })
        pred.setup_model(net, verbose=False)
        pred.model.set_classes(list(settings.SAM3_LABELS))
        return pred
    return _cached("sam3", build)


def face_analyser():
    """insightface SCRFD-10G detector (buffalo_l).

    Returns ``None`` when insightface or its weights are unavailable, which the
    caller treats as "T2 produced no evidence".
    """
    def build():
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            # onnxruntime builds without the CUDA EP are common; asking for it
            # unconditionally fails at session creation. Pick by availability,
            # and remember that insightface needs ctx_id=-1 to match CPU.
            use_cuda = "CUDAExecutionProvider" in set(ort.get_available_providers())
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if use_cuda else ["CPUExecutionProvider"])
            dev = settings.tool_device()
            ctx = (int(dev.split(":")[1]) if use_cuda and dev.startswith("cuda")
                   else -1)
            app = FaceAnalysis(name=settings.FACE_MODEL_NAME,
                               root=settings.FACE_MODEL_ROOT,
                               allowed_modules=["detection"],
                               providers=providers)
            app.prepare(ctx_id=ctx, det_size=settings.FACE_DET_SIZE)
            return app
        except Exception as e:                # noqa: BLE001 - soft degradation
            print(f"    T2 FACE unavailable: {type(e).__name__}: {e}", flush=True)
            return None
    return _cached("face", build)
