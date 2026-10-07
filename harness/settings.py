#!/usr/bin/env python3
"""Global settings: paths, API endpoints, model names, thresholds.

Everything a user of this repository might reasonably want to change lives here,
so that no other module has to be edited to point the pipeline at a different
endpoint or model.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent          # <repo>/harness
REPO_DIR = PKG_DIR.parent                          # <repo>
PROMPTS_DIR = REPO_DIR / "prompts"
DF_PROMPTS_DIR = PROMPTS_DIR / "df"
DATA_DIR = REPO_DIR / "data"
DEFAULT_OUT_ROOT = REPO_DIR / "runs"

# ---------------------------------------------------------------------------
# API endpoint and credentials
# ---------------------------------------------------------------------------
#
# FILL THESE IN BEFORE RUNNING ANYTHING.
#
# Both clients in this package (harness/clients.py) speak the Gemini
# ``generateContent`` protocol:
#
#   POST {API_BASE}/v1beta/models/{model}:generateContent
#   headers: {"Authorization": "Bearer <API_KEY>", "Content-Type": "application/json"}
#
# so API_BASE is the ORIGIN of that API - scheme plus host, no trailing path.
# Any endpoint that accepts that request and returns the documented response
# works; nothing else in this repository assumes a particular provider.
#
# Two ways to set them, and the environment wins:
#
#   1. environment variables (recommended - keeps credentials out of the tree)
#        export HARNESS_API_BASE="https://<your-endpoint-host>"
#        export HARNESS_API_KEY="<your-api-key>"
#   2. the two constants below, edited in place
#
# Leaving both unset raises at client construction with a message naming the
# variable to set, rather than failing later with an opaque HTTP error.

API_BASE = os.environ.get("HARNESS_API_BASE", "")       # e.g. "https://your-host"
API_KEY_ENV = "HARNESS_API_KEY"
API_KEY = os.environ.get(API_KEY_ENV, "")              # e.g. "sk-..."

# ---------------------------------------------------------------------------
# VLM backends (diagnosis / weave / verifier share one client, different prompts)
# ---------------------------------------------------------------------------

VLM_BACKEND = "gemini"                       # gemini | gpt | openai-compatible
VLM_MODEL_NAME = "gemini-3.7-flash"          # default composer/diagnosis/verifier model
# Selectable VLM ids, for the command-line `--vlm` / `--vlm-model` overrides.
# Any model id the endpoint accepts works; these are the ones in use.
VLM_MODELS = {
    "gemini": "gemini-3.7-flash",
    "gemini-pro": "gemini-3.1-pro-preview",
}
VLM_MAX_OUTPUT_TOKENS = 16384
VLM_TIMEOUT_SEC = 300
VLM_MAX_RETRIES = 3
VLM_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Executor (generative image editing model)
# ---------------------------------------------------------------------------

MFM_BACKEND = "nb2"                          # nb2 | gpt-image-2.5
# Short name -> the model id the endpoint expects in the URL. Add an entry for
# any other editing model the endpoint serves, then select it with --executor.
MFM_GEMINI_ENDPOINTS = {
    "nb2": "gemini-3.1-flash-image-preview",
    "gpt-image-2.5": "gpt-image-2.5-sunburst",
}
MFM_TIMEOUT_SEC = 600
MFM_MAX_RETRIES = 2
# generationConfig.imageConfig.imageSize: 512 / 1K / 2K / 4K
MFM_IMAGE_SIZE = "1K"

# ---------------------------------------------------------------------------
# Executor input geometry
# ---------------------------------------------------------------------------

# The executor receives a square 1024x1024 canvas: the long edge is resized to
# 1024 and the short edge is centre-padded with black. Outputs are cropped back
# to a long edge of 1024.
#
# Padding is applied ONLY immediately before the executor call (see
# stages/execute.py). Padding earlier in the chain would make the depth model
# read the black border as foreground and push the whole content region into
# the far field, which flips the prompt towards "leave it alone".
SQUARE = 1024
DEPTH_PAD_VALUE = (0, 0, 0)                  # 0 = farthest, matches a black border

# ---------------------------------------------------------------------------
# Verifier thresholds (hard gates and the accept policy)
# ---------------------------------------------------------------------------

# Grayscale SSIM between LQ and the result. Below this the executor replaced
# the scene or changed the geometry rather than restoring it. Deliberately
# loose: a false reject removes a round from best-of-N outright.
STRUCTURE_SSIM_MIN = 0.30
# Mean chroma drift, brightness-independent. Only enforced for colour-locking
# types; haze and lowlight necessarily change colour, so for them it is
# recorded but never fails the round.
CHROMA_DRIFT_MAX = 0.10
# There is deliberately no score threshold here. Whether a round stops is
# answered by the verifier's own needs_redo field; the scores it reports are
# recorded for analysis but do not gate anything.

# Faces whose repair would require more upscaling than this are excluded from
# regional repair; beyond ~8x the editor starts inventing new identities.
FACE_UPSCALE_SAFE_MAX = 8.0

# ---------------------------------------------------------------------------
# Tool models (paths can be overridden with HARNESS_CKPT_DIR)
# ---------------------------------------------------------------------------

CKPT_DIR = Path(os.environ.get("HARNESS_CKPT_DIR", str(REPO_DIR / "checkpoints")))

# T1 OCR - PP-OCRv6. Weights are located by paddlex automatically unless a local
# copy is present under checkpoints/paddleocr/official_models/.
PP_OCR_DET_DIR = CKPT_DIR / "paddleocr" / "official_models" / "PP-OCRv6_medium_det"
PP_OCR_REC_DIR = CKPT_DIR / "paddleocr" / "official_models" / "PP-OCRv6_medium_rec"
if PP_OCR_DET_DIR.is_dir():
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CKPT_DIR / "paddleocr"))

# T2 FACE - SCRFD (insightface buffalo_l) with YuNet as an optional fallback.
FACE_MODEL_ROOT = str(CKPT_DIR / "insightface") \
    if (CKPT_DIR / "insightface" / "models" / "buffalo_l" / "det_10g.onnx").is_file() \
    else "~/.insightface"
FACE_MODEL_NAME = "buffalo_l"
FACE_DET_SIZE = (640, 640)
FACE_MIN_PX = 24

# Optional second face detector, used only when SCRFD returns nothing. Download
# the ONNX build of YuNet into the checkpoints directory to enable it, or leave it
# absent - the fallback is skipped and SCRFD is used alone.
YUNET_CKPT = str(CKPT_DIR / "yunet.onnx")

# T3 DEPTH - Depth-Anything-V2-Base.
DEPTH_MODEL = str(CKPT_DIR / "depth_anything_v2_base") \
    if (CKPT_DIR / "depth_anything_v2_base" / "model.safetensors").is_file() \
    else "depth-anything/Depth-Anything-V2-Base-hf"

# T4 SEGMENTATION - SAM3 semantic (open vocabulary, text-prompted).
# The checkpoint MUST contain the language backbone; conversions without it
# produce NaNs in forward_grounding.
SAM3_CKPT = str(CKPT_DIR / "sam3_semantic.pt")
SAM3_LABELS = [
    "person", "face", "text", "sign", "sky", "building", "car",
    "tree", "road", "vegetation", "water", "ground", "door", "window",
    "windmill", "bicycle", "motorcycle", "truck", "bus", "traffic light",
    "pole", "fence", "cloud", "mountain", "bridge", "stairs",
    "poster", "billboard", "street light", "bench", "trash can",
]
SAM3_CONF = 0.15
SAM3_IOU = 0.7
SAM3_IMGSZ = 1024

# Offline model loading: the tool models are resolved from local weights.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def tool_device() -> str:
    """Device for the local tool models.

    Honours ``TOOL_DEVICE`` (e.g. ``cuda:1``); otherwise picks the GPU with the
    most free memory, so the tools do not compete with anything else on the box.
    """
    forced = os.environ.get("TOOL_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        best, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best, best_free = i, free
        return f"cuda:{best}"
    except Exception:                        # noqa: BLE001 - no torch -> CPU
        return "cpu"


def api_key() -> str:
    """The API key: the environment variable if set, else ``API_KEY`` above.

    Raises when neither is filled in. The failure is raised here, with the name of
    the variable to set, rather than at the first HTTP call - a missing key is a
    configuration mistake, and "401 from the endpoint" sends the reader looking in
    the wrong place.
    """
    key = os.environ.get(API_KEY_ENV, "").strip() or API_KEY.strip()
    if not key:
        raise RuntimeError(
            f"no API key configured. Set the environment variable {API_KEY_ENV}, "
            f"or fill in API_KEY in {Path(__file__).name}.")
    return key


def api_base() -> str:
    """The endpoint origin, validated the same way as the key."""
    base = (API_BASE or os.environ.get("HARNESS_API_BASE", "")).strip()
    if not base:
        raise RuntimeError(
            "no API endpoint configured. Set the environment variable "
            "HARNESS_API_BASE, or fill in API_BASE in "
            f"{Path(__file__).name}.")
    return base.rstrip("/")
