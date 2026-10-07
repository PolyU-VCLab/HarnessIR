#!/usr/bin/env python3
"""Stage 4 - call the executor with the photograph and the prompt.

Two geometries, applied only here and only around the call:

* Tool models and the composer see the photograph at its native aspect ratio with
  its long edge resized to 1024 (``prepare_lq``). Padding happens later.
* Immediately before the executor call, the photograph is padded to a 1024x1024
  square (``finalize_square``), and the result is cropped back to a long edge of
  1024 (``crop_black_border``). This is the geometry the executor works best in.

Padding was moved here from the very front of the chain for a measured reason:
a depth model treats the black border as foreground. With the border present,
the entire content region gets pushed into the far field, the depth evidence
says "mostly distant", and the composer writes prompts that leave the
distant region alone - which is the region it should have been restoring. The
bias survives cropping and renormalising afterwards, because the border changes
the model's reasoning, not just the normalisation range.

The executor does not guarantee it returns exactly the requested size, so the
output is normalised to the canvas before cropping. Without that step a
near-miss size shifts the whole crop box.
"""
from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from .. import settings


def resize_long_edge(img: Image.Image, long_edge: int = settings.SQUARE,
                     resample=Image.LANCZOS) -> tuple[Image.Image, float]:
    """Scale so the long edge is ``long_edge``. Aspect ratio preserved, no padding."""
    w, h = img.size
    scale = long_edge / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                      resample), scale


def pad_to_square(img: Image.Image, resample=Image.LANCZOS,
                  fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Scale to a long edge of 1024, then centre-pad the short edge to 1024."""
    resized, _ = resize_long_edge(img, settings.SQUARE, resample)
    w, h = resized.size
    canvas = Image.new("RGB", (settings.SQUARE, settings.SQUARE), fill)
    canvas.paste(resized, ((settings.SQUARE - w) // 2, (settings.SQUARE - h) // 2))
    return canvas


def crop_black_border(img: Image.Image, original_size, scale: float) -> Image.Image:
    """Crop the padding off a square canvas, keeping a long edge of 1024.

    The content sits centred, so the effective region is computed from the
    original aspect ratio and the scale that produced it.
    """
    ow, oh = original_size
    eff_w = max(1, round(ow * scale))
    eff_h = max(1, round(oh * scale))
    left = (img.width - eff_w) // 2
    top = (img.height - eff_h) // 2
    return img.crop((left, top, left + eff_w, top + eff_h))


def prepare_lq(lq_path: str | Path, out_dir: Path) -> tuple[Path, dict]:
    """Resize the long edge to 1024 and record the geometry.

    This is the chain input. The returned geometry is what ``crop_black_border``
    needs later, and it is written into the run record so any output can be
    traced back to the framing it came from.
    """
    lq_path = Path(lq_path)
    with Image.open(lq_path) as im:
        ow, oh = im.size
        resized, scale = resize_long_edge(im.convert("RGB"))

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "lq_resized1024.png"
    resized.save(dst)
    return dst, {
        "original_size": [ow, oh],
        "resized_size": list(resized.size),
        "scale": scale,
        "square": settings.SQUARE,
        "source": str(lq_path),
    }


def run_execute(mfm, lq_path: str | Path, prompt: str, out_dir: Path,
                *, filename: str = "output.png",
                crop_back: tuple[tuple[int, int], float] | None = None) -> dict:
    """One executor call. Returns a dict rather than raising.

    A per-image failure must not abort a batch, so the error comes back in the
    returned dict and the caller records it. The return shape mirrors a
    successful call, so callers do not need a special branch.

    ``crop_back`` is ``((original_width, original_height), scale)``. When given,
    the square geometry above is applied; when ``None`` the photograph is sent
    as-is and the result is resized back to the input size.
    """
    lq_path = Path(lq_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with Image.open(lq_path) as im:
        input_size = im.size
    out_path = out_dir / filename
    send_path = str(lq_path)

    try:
        if crop_back is not None:
            with Image.open(lq_path) as im:
                square = pad_to_square(im.convert("RGB"))
            work = out_dir / "square_inputs"
            work.mkdir(parents=True, exist_ok=True)
            send_path = str(work / f"sq1024_{Path(filename).stem}.png")
            square.save(send_path)
            input_size = (settings.SQUARE, settings.SQUARE)

        img = mfm.edit([send_path], prompt)

        if crop_back is not None:
            # The executor picks its own output size from the aspect ratio, so
            # normalise to the canvas before computing the crop box.
            if img.size != input_size:
                img = img.resize(input_size, Image.LANCZOS)
            img = crop_black_border(img, *crop_back)
        elif img.size != input_size:
            img = img.resize(input_size, Image.LANCZOS)

        img.save(out_path)
        return {
            "path": str(out_path),
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "n_pictures": 1,
            "output_size": list(img.size),
            "cropped_back": crop_back is not None,
            "error": None,
        }
    except Exception as e:                    # noqa: BLE001 - record, do not raise
        return {
            "path": None,
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "n_pictures": 1,
            "cropped_back": crop_back is not None,
            "error": f"{type(e).__name__}: {e}",
        }
