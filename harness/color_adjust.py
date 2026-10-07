#!/usr/bin/env python3
"""Post-hoc colour alignment for the delivered image.

Generative editors drift the colour and white balance of a restoration - the
result looks plausible but no longer matches the input's palette. The fix is a
lightweight AdaIN: take the mean and standard deviation of each channel from the
degraded input and transfer them onto the result, so the result keeps the input's
overall colour statistics without any learned model.

Two modes, chosen from the sample's degradation types:

  ``adain``  align the result's channel statistics to the input's. Used when the
             only type is ``mix`` - an untyped image, where nothing licenses a
             colour change and the drift is a defect to be removed.
  ``copy``   pass the result through untouched. Used for every named type.

Every named type owns its own colour behaviour. Haze and lowlight are expected to
change colour as part of the restoration, so forcing the input's statistics back
onto the result would undo part of the very correction being asked for; rain and
snow must not change colour at all, so there is no drift to remove and shifting
it would be the defect being measured for. Copy is the safe reading of both, and
it is what the reference implementation does.

The mode is a function of the type labels, not of the image and not of the file
path, so the same manifest yields the same mode on every machine.

Run after the executor and before anything that scores the round, so the verifier,
the selector and the metrics all see the image that is actually delivered.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

MODE_ADJUST = "adain"
MODE_COPY = "copy"


def mode_for(types: list[str]) -> str:
    """Pick the adjustment mode for one sample's degradation types.

    Only a pure ``mix`` is adjusted. Any named degradation - haze, rain, snow,
    lowlight - selects ``copy``, and a multi-type image selects ``copy`` because
    at least one of its types is named.
    """
    return MODE_ADJUST if types and all(t == "mix" for t in types) else MODE_COPY


def _mean_std(feat):
    b, c = feat.size()[:2]
    var = feat.reshape(b, c, -1).var(dim=2) + 1e-5
    std = var.sqrt().reshape(b, c, 1, 1)
    mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return mean, std


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """Transfer ``source``'s per-channel mean and std onto ``target``."""
    import torch
    from torchvision.transforms import ToPILImage, ToTensor

    to_tensor = ToTensor()
    t = to_tensor(target).unsqueeze(0)
    s = to_tensor(source).unsqueeze(0)
    s_mean, s_std = _mean_std(s)
    t_mean, t_std = _mean_std(t)
    size = t.size()
    normalised = (t - t_mean.expand(size)) / t_std.expand(size)
    out = normalised * s_std.expand(size) + s_mean.expand(size)
    return ToPILImage()(out.squeeze(0).clamp_(0.0, 1.0).cpu())


def adjust_image(types: list[str], lq_path: str | Path, src_path: str | Path,
                 dst_path: str | Path) -> dict[str, Any]:
    """Apply the rule for ``types`` and write the result to ``dst_path``.

    Returns ``{"mode", "types", "adjusted", "error"}``. A failure is reported
    rather than raised: the adjustment is a refinement, and a sample whose
    adjustment failed is still a usable sample. The caller keeps the unadjusted
    image in that case, and the error is recorded.

    ``dst_path`` is always written, including in ``copy`` mode, so nothing
    downstream has to branch on "was this one adjusted".
    """
    mode = mode_for(types)
    out: dict[str, Any] = {"mode": mode, "types": list(types), "adjusted": None,
                           "error": None}
    src, dst = Path(src_path), Path(dst_path)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == MODE_COPY:
            import shutil

            shutil.copy2(src, dst)
        else:
            lq = Path(lq_path)
            if not lq.is_file():
                out["error"] = f"adain needs the input image, missing: {lq}"
                return out
            img = Image.open(src).convert("RGB")
            style = Image.open(lq).convert("RGB")
            # AdaIN compares whole-image statistics, so the two must be the same
            # size for the pairing to be meaningful.
            if style.size != img.size:
                style = style.resize(img.size, Image.BICUBIC)
            adain_color_fix(img, style).save(dst)
        out["adjusted"] = str(dst)
    except Exception as e:                       # noqa: BLE001 - report, do not raise
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def adjusted_name(filename: str) -> str:
    """``output.png`` -> ``output_adjusted.png``.

    Used as the temporary name for the aligned image: the result is written
    there, then moved over the executor's own output when it succeeds.
    """
    p = Path(filename)
    return f"{p.stem}_adjusted{p.suffix}"


def adjust_in_place(types: list[str], lq_path: str | Path,
                    image_path: str | Path) -> dict[str, Any]:
    """Align ``image_path`` and replace it with the aligned version.

    The adjusted image goes to a temporary name first and is moved over the
    original only once it has been written. One file per round is therefore
    kept, and it is the one that ships.

    Writing in place is safe because the source is read and decoded fully before
    anything is written. On failure the original is left untouched and the
    temporary file is removed, so a failed adjustment never leaves a partial
    image behind for a scorer to pick up.
    """
    path = Path(image_path)
    tmp = path.parent / adjusted_name(path.name)
    out = adjust_image(types, lq_path, path, tmp)
    if not out.get("adjusted"):
        try:
            tmp.unlink()
        except OSError:
            pass                        # nothing to remove, or already gone
        return out
    Path(out["adjusted"]).replace(path)
    out["adjusted"] = str(path)
    out["replaced"] = True
    return out
