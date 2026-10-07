#!/usr/bin/env python3
"""API clients: the VLM (text out) and the executor / MFM (image out).

Both talk to a Gemini-compatible ``:generateContent`` endpoint, configured in
``settings.API_BASE`` / ``settings.API_KEY``. Two clients rather than one because
the response shapes differ: the VLM returns JSON text, the executor returns
inline image bytes.

Retries use exponential backoff on every exception. Rate limits, timeouts and
5xx all look the same from here and all want the same treatment.
"""
from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from . import settings

# Serialises nothing by itself - only guards the usage counters below.
_USAGE_LOCK = threading.Lock()
_USAGE: dict[str, dict[str, int]] = {}


def usage_report() -> dict[str, dict[str, int]]:
    """Cumulative token counters per endpoint, for cost accounting."""
    with _USAGE_LOCK:
        return {k: dict(v) for k, v in _USAGE.items()}


def _record_usage(tag: str, data: dict[str, Any]) -> None:
    meta = data.get("usageMetadata") or {}
    if not meta:
        return
    with _USAGE_LOCK:
        slot = _USAGE.setdefault(tag, {})
        for key in ("promptTokenCount", "candidatesTokenCount",
                    "thoughtsTokenCount", "totalTokenCount"):
            if key in meta:
                slot[key] = slot.get(key, 0) + int(meta[key] or 0)
        slot["calls"] = slot.get("calls", 0) + 1


def _retry(fn, max_retries: int, base_delay: float = 2.0):
    """Exponential backoff. Raises after the last attempt."""
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:                   # noqa: BLE001 - uniform backoff
            last = e
            if attempt + 1 < max_retries:
                time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"API call failed after {max_retries} attempts: {last}") from last


def _to_inline(im: Image.Image | str | Path) -> dict[str, Any]:
    """Encode a path or a PIL image as a Gemini ``inlineData`` part."""
    if isinstance(im, (str, Path)):
        p = str(im)
        data = base64.b64encode(Path(p).read_bytes()).decode("ascii")
        mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
    else:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = "image/png"
    return {"inlineData": {"mimeType": mime, "data": data}}


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Tolerates markdown fences and surrounding prose, which both occur even with
    ``responseMimeType: application/json`` set.
    """
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# VLM - diagnosis, weaving, judging
# ---------------------------------------------------------------------------

class VLMClient:
    """Vision-language model with a JSON-out convenience method."""

    def __init__(self, model: str | None = None, *,
                 max_output_tokens: int | None = None,
                 timeout_sec: int | None = None,
                 max_retries: int | None = None,
                 temperature: float | None = None) -> None:
        self.model = model or settings.VLM_MODEL_NAME
        self.max_output_tokens = max_output_tokens or settings.VLM_MAX_OUTPUT_TOKENS
        self.timeout_sec = timeout_sec or settings.VLM_TIMEOUT_SEC
        self.max_retries = max_retries or settings.VLM_MAX_RETRIES
        self.temperature = settings.VLM_TEMPERATURE if temperature is None else temperature
        base = settings.api_base()
        self.url = f"{base}/v1beta/models/{self.model}:generateContent"

    def chat_json(self, user: str, images: list | None = None, *,
                  system: str | None = None) -> dict[str, Any]:
        """One turn, JSON object out. ``images`` are paths or PIL images."""
        parts: list[dict[str, Any]] = [_to_inline(im) for im in (images or [])]
        parts.append({"text": user})

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": self.max_output_tokens,
                "temperature": self.temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {settings.api_key()}"}

        def _call() -> dict[str, Any]:
            resp = requests.post(self.url, headers=headers, json=payload,
                                 timeout=self.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            _record_usage(f"vlm:{self.model}", data)
            cands = data.get("candidates") or []
            if not cands:
                raise ValueError(f"no candidate in response: {str(data)[:400]}")
            texts = [p["text"] for p in cands[0].get("content", {}).get("parts", [])
                     if "text" in p]
            if not texts:
                # A truncated reply arrives as a candidate with no text part at
                # all; surface finishReason so the caller sees MAX_TOKENS rather
                # than a bare parse error.
                raise ValueError(
                    f"no text part (finishReason="
                    f"{cands[0].get('finishReason')}): {str(data)[:400]}")
            return extract_json("".join(texts))

        return _retry(_call, self.max_retries)


# ---------------------------------------------------------------------------
# Executor - the generative image editing model
# ---------------------------------------------------------------------------

def _aspect_ratio(image: Image.Image) -> str:
    """Nearest supported ``imageConfig.aspectRatio`` bucket for an input."""
    w, h = image.size
    r = w / h if h else 1.0
    if r >= 1.55:
        return "16:9"
    if r >= 1.15:
        return "4:3"
    if r > 0.87:
        return "1:1"
    if r > 0.65:
        return "3:4"
    return "9:16"


class MFMClient:
    """Generative image editor (Nano-Banana-2 / Gemini image family).

    ``edit`` takes one or more images plus a prompt and returns one image.
    The first image sets the output aspect ratio; any further images are
    references. In this repository only the first image is ever sent - the
    executor receives the photograph and the woven text, nothing else.
    """

    def __init__(self, model: str | None = None, *,
                 timeout_sec: int | None = None,
                 max_retries: int | None = None,
                 image_size: str | None = None) -> None:
        name = model or settings.MFM_BACKEND
        if name not in settings.MFM_GEMINI_ENDPOINTS:
            raise ValueError(
                f"unknown executor {name!r}; known: "
                f"{sorted(settings.MFM_GEMINI_ENDPOINTS)}")
        self.model = name
        self.endpoint = settings.MFM_GEMINI_ENDPOINTS[name]
        self.timeout_sec = timeout_sec or settings.MFM_TIMEOUT_SEC
        self.max_retries = max_retries or settings.MFM_MAX_RETRIES
        self.image_size = image_size or settings.MFM_IMAGE_SIZE
        base = settings.api_base()
        self.url = f"{base}/v1beta/models/{self.endpoint}:generateContent"

    def edit(self, images: list, prompt: str, *,
             size: str | None = None) -> Image.Image:
        assert images, "at least one input image is required"
        first = images[0]
        pil = (Image.open(first).convert("RGB")
               if isinstance(first, (str, Path)) else first)

        parts: list[dict[str, Any]] = [_to_inline(im) for im in images]
        parts.append({"text": prompt})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                # Force image-only output; without this the model sometimes
                # answers with prose about what it would do.
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": _aspect_ratio(pil),
                                "imageSize": str(size or self.image_size)},
            },
        }
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {settings.api_key()}"}

        def _call() -> Image.Image:
            resp = requests.post(self.url, headers=headers, json=payload,
                                 timeout=self.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            _record_usage(f"mfm:{self.endpoint}", data)
            cands = data.get("candidates") or []
            rparts = cands[0].get("content", {}).get("parts", []) if cands else []
            for part in rparts:
                if "inlineData" in part:
                    raw = base64.b64decode(part["inlineData"]["data"])
                    return Image.open(io.BytesIO(raw)).convert("RGB")
            raise ValueError(f"no image in response: {str(data)[:400]}")

        return _retry(_call, self.max_retries)


def make_vlm(**kwargs) -> VLMClient:
    return VLMClient(**kwargs)


def make_mfm(**kwargs) -> MFMClient:
    return MFMClient(**kwargs)
