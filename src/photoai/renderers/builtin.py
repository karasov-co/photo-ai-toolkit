"""A deterministic renderer built on LibRaw, with no external binary.

Every operation is applied in linear light on float64 and in a fixed order --
white balance, exposure, highlight and shadow recovery, contrast, colour,
detail, geometry -- because the order changes the result and a search that
compares candidates needs the comparison to be about the candidates.

The point of doing this in-process rather than shelling out is availability.
rawpy is already a hard dependency, so any machine that can analyse the archive
can also render the proposals. darktable and RawTherapee produce better final
images and are wired up as adapters, but they are optional installs, and a
preview that only appears on some machines is a preview the report cannot rely
on.

**This is not a raw converter.** It is a preview engine good enough to tell two
candidate edits apart. The tone curve is a simple filmic-ish roll-off, not
anyone's colour science, and the numbers it produces should be read as
directions for a real converter rather than as final values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from photoai.renderers.base import Renderer, RenderError, register

logger = logging.getLogger(__name__)

GAMMA = 2.2
# Where the highlight and shadow sliders start biting, in display-referred
# terms. Matched roughly to Camera Raw's behaviour so the numbers transfer.
HIGHLIGHT_PIVOT = 0.72
SHADOW_PIVOT = 0.28


class BuiltinRenderer(Renderer):
    name = "builtin"

    def is_available(self) -> bool:
        try:
            import rawpy  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    def version(self) -> str:
        try:
            import rawpy

            return f"librawpy/{getattr(rawpy, '__version__', 'unknown')}"
        except ImportError:  # pragma: no cover
            return "unavailable"

    # --- decoding -----------------------------------------------------------

    def _decode(self, path: Path, max_px: int) -> np.ndarray:
        """Sensor data to a linear RGB array, camera white balance applied."""
        import rawpy

        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,   # the recipe decides exposure, not LibRaw
                    output_bps=16,
                    gamma=(1, 1),          # stay linear; the tone curve is ours
                    half_size=True,
                )
        except Exception as e:
            raise RenderError(f"could not decode {path.name}: {e}") from e

        linear = rgb.astype(np.float64) / 65535.0
        image = Image.fromarray((np.clip(linear, 0, 1) ** (1 / GAMMA) * 255).astype(np.uint8))
        image.thumbnail((max_px, max_px), Image.LANCZOS)
        return (np.asarray(image, dtype=np.float64) / 255.0) ** GAMMA

    def _decode_rendered(self, path: Path, max_px: int) -> np.ndarray:
        """Fallback for JPEG/HEIC input: what is there is already developed."""
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((max_px, max_px), Image.LANCZOS)
            return (np.asarray(rgb, dtype=np.float64) / 255.0) ** GAMMA

    # --- the pipeline -------------------------------------------------------

    def render(self, path: Path, recipe, *, max_px: int = 1024) -> Image.Image:
        from photoai import media

        path = Path(path)
        linear = (
            self._decode(path, max_px)
            if media.photo_format(path) is media.PhotoFormat.RAW
            else self._decode_rendered(path, max_px)
        )

        g = recipe.global_adjustments
        linear = _white_balance(linear, g.temperature_delta_k, g.tint_delta)
        linear = linear * (2.0 ** g.exposure_ev)

        display = np.clip(linear, 0.0, 4.0) ** (1 / GAMMA)
        display = np.clip(display, 0.0, 1.0)

        display = _highlights(display, g.highlights)
        display = _shadows(display, g.shadows)
        display = _whites_blacks(display, g.whites, g.blacks)
        display = _contrast(display, g.contrast)
        display = _clarity(display, g.clarity)
        display = _saturation(display, g.saturation, g.vibrance)

        if recipe.color.monochrome:
            luma = display @ np.array([0.299, 0.587, 0.114])
            display = np.dstack([luma, luma, luma])

        out = Image.fromarray((np.clip(display, 0, 1) * 255).astype(np.uint8))
        out = _detail(out, recipe.detail)
        out = _geometry(out, recipe.geometry)
        return out


# --- operations, each in one place ------------------------------------------


def _white_balance(linear: np.ndarray, kelvin_delta: int, tint_delta: int) -> np.ndarray:
    """Approximate: warm lifts red and drops blue, tint moves green.

    A real converter solves for the illuminant. This is a channel gain, which is
    close enough over the +/-1000K a correction actually uses and wrong beyond
    it -- hence the clamp in the generator rather than here.
    """
    if not kelvin_delta and not tint_delta:
        return linear
    warm = kelvin_delta / 4000.0
    green = tint_delta / 200.0
    gains = np.array([1.0 + warm, 1.0 + green * 0.5, 1.0 - warm])
    return linear * np.clip(gains, 0.2, 5.0)


def _highlights(display: np.ndarray, amount: int) -> np.ndarray:
    """Negative pulls the top down. Only touches values above the pivot."""
    if not amount:
        return display
    strength = amount / 100.0
    mask = np.clip((display - HIGHLIGHT_PIVOT) / (1.0 - HIGHLIGHT_PIVOT), 0, 1)
    return np.clip(display + mask * strength * (1.0 - display) * 0.9, 0, 1)


def _shadows(display: np.ndarray, amount: int) -> np.ndarray:
    if not amount:
        return display
    strength = amount / 100.0
    mask = np.clip((SHADOW_PIVOT - display) / SHADOW_PIVOT, 0, 1)
    return np.clip(display + mask * strength * (SHADOW_PIVOT - display) * 0.8, 0, 1)


def _whites_blacks(display: np.ndarray, whites: int, blacks: int) -> np.ndarray:
    if not whites and not blacks:
        return display
    out = display + (whites / 100.0) * np.clip(display - 0.5, 0, None) * 0.6
    out = out + (blacks / 100.0) * np.clip(0.5 - out, 0, None) * 0.6
    return np.clip(out, 0, 1)


def _contrast(display: np.ndarray, amount: int) -> np.ndarray:
    if not amount:
        return display
    strength = amount / 100.0
    return np.clip(display + strength * (display - 0.5) * (1.0 - np.abs(display - 0.5) * 2.0), 0, 1)


def _clarity(display: np.ndarray, amount: int) -> np.ndarray:
    """Local contrast via unsharp mask on luminance only, so hues do not shift."""
    if not amount:
        return display
    luma = display @ np.array([0.299, 0.587, 0.114])
    blurred = np.asarray(
        Image.fromarray((luma * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(12)),
        dtype=np.float64,
    ) / 255.0
    boost = (luma - blurred) * (amount / 100.0)
    return np.clip(display + boost[:, :, None], 0, 1)


def _saturation(display: np.ndarray, saturation: int, vibrance: int) -> np.ndarray:
    """Vibrance protects what is already saturated; saturation does not."""
    if not saturation and not vibrance:
        return display
    luma = (display @ np.array([0.299, 0.587, 0.114]))[:, :, None]
    chroma = display - luma
    gain = 1.0 + saturation / 100.0
    if vibrance:
        current = np.abs(chroma).max(axis=2, keepdims=True)
        gain = gain + (vibrance / 100.0) * (1.0 - np.clip(current * 2.5, 0, 1))
    return np.clip(luma + chroma * gain, 0, 1)


def _detail(image: Image.Image, detail) -> Image.Image:
    if detail.denoise_luminance > 0:
        radius = 1 + detail.denoise_luminance // 40
        image = image.filter(ImageFilter.MedianFilter(min(5, 1 + 2 * radius)))
    if detail.sharpening > 0:
        # Masking keeps the sharpening off flat areas, which is where it turns
        # into visible noise rather than into detail.
        percent = int(detail.sharpening * 1.5)
        threshold = int(detail.masking / 100.0 * 12)
        image = image.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=max(1, percent), threshold=threshold)
        )
    return image


def _inscribed_rect(width: int, height: int, degrees: float) -> tuple[int, int]:
    """Largest axis-aligned rectangle that fits inside the rotated frame.

    Every raw converter crops to this after a straighten, because the
    alternative is black wedges in the corners. Skipping it here was not merely
    cosmetic: the validator counted those wedges as newly crushed shadows and
    vetoed every candidate that contained a rotation.
    """
    import math

    angle = math.radians(abs(degrees) % 180)
    if angle > math.pi / 2:
        angle = math.pi - angle
    if width <= 0 or height <= 0:
        return width, height

    width_is_longer = width >= height
    long_side, short_side = (width, height) if width_is_longer else (height, width)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))

    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        half = 0.5 * short_side
        wr, hr = (half / sin_a, half / cos_a) if width_is_longer else (half / cos_a, half / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (width * cos_a - height * sin_a) / cos_2a
        hr = (height * cos_a - width * sin_a) / cos_2a
    return max(1, int(wr)), max(1, int(hr))


def _geometry(image: Image.Image, geometry) -> Image.Image:
    if geometry.rotation_deg and not geometry.preserve_existing_tilt:
        width, height = image.size
        image = image.rotate(
            geometry.rotation_deg, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0)
        )
        keep_w, keep_h = _inscribed_rect(width, height, geometry.rotation_deg)
        left = (width - keep_w) // 2
        top = (height - keep_h) // 2
        image = image.crop((left, top, left + keep_w, top + keep_h))
    crop = geometry.crop
    if crop and not crop.is_identity:
        width, height = image.size
        box = (
            int(crop.left * width),
            int(crop.top * height),
            max(int(crop.right * width), int(crop.left * width) + 1),
            max(int(crop.bottom * height), int(crop.top * height) + 1),
        )
        image = image.crop(box)
    return image


register(BuiltinRenderer())
