"""What a rendered edit is not allowed to have done.

The generator proposes and the optimiser ranks, but neither of them looks at the
result. This module does, by comparing the rendered candidate against the
rendered original, and it can veto.

The checks are deliberately about *damage* rather than about taste:

- **New clipping.** An edit that recovers a sky by blowing the skin has not
  improved the frame. Measured as clipped pixels created, not total.
- **Flattening a low-key frame.** The commonest way an automatic edit ruins a
  photograph: it sees a dark histogram, "corrects" it, and returns a grey image
  where a black one was intended. Detected as the shadow mass collapsing.
- **Sharpening intentional blur.** If `preserve` names the blur, any sharpening
  is a veto, not a deduction.
- **Halos.** Over-clarity and over-sharpening produce a bright rim along strong
  edges. Detected by comparing edge-adjacent brightness before and after.
- **Crop cost.** Area thrown away has to buy something; below a confidence
  threshold it is refused outright.
- **Skin hue drift.** A white-balance move that swings skin towards green or
  magenta is worse than the cast it fixed.

A veto is not a low score. A vetoed candidate is removed from consideration
entirely, because these are all ways of making the photograph worse while
scoring better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Fraction of the frame that may newly clip before the edit is refused. Some
# clipping is normal and even desirable on a specular highlight.
MAX_NEW_CLIPPING = 0.015

# Shadows get more room than highlights, and not out of laziness. Setting a
# black point is a normal, deliberate part of almost every edit -- that is what
# the Blacks slider is for -- whereas blowing a highlight destroys information
# that cannot be recovered. The two are not symmetric and should not share a
# threshold. The low-key check below is what actually guards dark frames, and it
# guards them by intent rather than by an arbitrary pixel count.
MAX_NEW_CRUSHING = 0.06

# A low-key frame keeps most of its mass in the lower third. If an edit moves
# more than this much of it out, the frame is no longer low key.
LOW_KEY_MASS_LOSS = 0.30

# Halo: mean brightening in a narrow band beside strong edges, 0..1.
MAX_HALO = 0.045

# Skin hue is roughly 15-35 degrees. Drift beyond this is visible and wrong.
MAX_SKIN_HUE_DRIFT_DEG = 8.0

MIN_CROP_CONFIDENCE = 0.6
MIN_CROP_KEEP = 0.55


@dataclass
class Violation:
    rule: str
    detail: str
    fatal: bool = True


@dataclass
class ValidationResult:
    violations: list[Violation] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(v.fatal for v in self.violations)

    @property
    def reasons(self) -> list[str]:
        return [f"{v.rule}: {v.detail}" for v in self.violations]


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def validate(original: Image.Image, edited: Image.Image, recipe) -> ValidationResult:
    """Compare the two renders and refuse anything that did damage."""
    result = ValidationResult()

    # Geometry may legitimately change the size; compare on a common grid.
    before = _array(original)
    after = _array(edited.resize(original.size, Image.LANCZOS)) if edited.size != original.size else _array(edited)

    _check_new_clipping(before, after, result)
    _check_low_key_preserved(before, after, recipe, result)
    _check_sharpening_of_intentional_blur(recipe, result)
    _check_halos(before, after, result)
    _check_crop(recipe, result)
    _check_skin_hue(before, after, result)
    return result


def _check_new_clipping(before: np.ndarray, after: np.ndarray, result: ValidationResult) -> None:
    was_white = (before >= 0.996).all(axis=2)
    now_white = (after >= 0.996).all(axis=2)
    was_black = (before <= 0.004).all(axis=2)
    now_black = (after <= 0.004).all(axis=2)

    new_white = float((now_white & ~was_white).mean())
    new_black = float((now_black & ~was_black).mean())
    result.measurements["new_clipped_highlights"] = round(new_white, 5)
    result.measurements["new_crushed_shadows"] = round(new_black, 5)

    if new_white > MAX_NEW_CLIPPING:
        result.violations.append(
            Violation("new_clipping", f"the edit blows {new_white:.1%} of the frame that had detail")
        )
    if new_black > MAX_NEW_CRUSHING:
        result.violations.append(
            Violation("new_crushing", f"the edit crushes {new_black:.1%} of the frame that had detail")
        )


def _check_low_key_preserved(
    before: np.ndarray, after: np.ndarray, recipe, result: ValidationResult
) -> None:
    """The commonest way an automatic edit ruins a photograph."""
    protects_dark = any(
        "low-key" in p.lower() or "shadow" in p.lower() for p in (recipe.preserve or [])
    )
    if not protects_dark:
        return

    shadow_mass_before = float((_luma(before) < 0.25).mean())
    shadow_mass_after = float((_luma(after) < 0.25).mean())
    result.measurements["shadow_mass_before"] = round(shadow_mass_before, 4)
    result.measurements["shadow_mass_after"] = round(shadow_mass_after, 4)

    if shadow_mass_before <= 0.01:
        return
    lost = (shadow_mass_before - shadow_mass_after) / shadow_mass_before
    if lost > LOW_KEY_MASS_LOSS:
        result.violations.append(
            Violation(
                "low_key_flattened",
                f"{lost:.0%} of the shadow mass is gone; the frame the photographer made "
                "was dark and this one is not",
            )
        )


def _check_sharpening_of_intentional_blur(recipe, result: ValidationResult) -> None:
    protects_blur = any(
        "blur" in p.lower() or "motion" in p.lower() for p in (recipe.preserve or [])
    )
    if protects_blur and recipe.detail.sharpening > 0:
        result.violations.append(
            Violation(
                "sharpening_intentional_blur",
                f"sharpening {recipe.detail.sharpening} applied to a frame whose blur is "
                "the subject",
            )
        )


def _check_halos(before: np.ndarray, after: np.ndarray, result: ValidationResult) -> None:
    """Bright rims beside strong edges: the signature of too much clarity."""
    luma_before = _luma(before)
    if min(luma_before.shape) < 16:
        return

    edges = np.asarray(
        Image.fromarray((luma_before * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
        dtype=np.float64,
    ) / 255.0
    strong = edges > np.percentile(edges, 97)
    if not strong.any():
        return

    band = np.asarray(
        Image.fromarray((strong * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)),
        dtype=np.float64,
    ) > 0
    band &= ~strong
    if not band.any():
        return

    halo = float((_luma(after)[band] - luma_before[band]).mean())
    result.measurements["halo"] = round(halo, 5)
    if halo > MAX_HALO:
        result.violations.append(
            Violation("halos", f"edges have gained a {halo:.3f} bright rim; the edit is over-sharpened")
        )


def _check_crop(recipe, result: ValidationResult) -> None:
    crop = recipe.geometry.crop
    if crop is None or crop.is_identity:
        return
    result.measurements["crop_keeps"] = round(crop.keeps, 4)

    if crop.confidence < MIN_CROP_CONFIDENCE:
        result.violations.append(
            Violation(
                "low_confidence_crop",
                f"crop confidence {crop.confidence:.2f} is below {MIN_CROP_CONFIDENCE}; "
                "recomposing somebody's photograph on a guess is not an improvement",
            )
        )
    if crop.keeps < MIN_CROP_KEEP:
        result.violations.append(
            Violation("excessive_crop", f"only {crop.keeps:.0%} of the frame would survive")
        )


def _check_skin_hue(before: np.ndarray, after: np.ndarray, result: ValidationResult) -> None:
    """Skin drifting green or magenta is worse than the cast it replaced."""
    import colorsys

    def mean_skin_hue(rgb: np.ndarray) -> float | None:
        flat = rgb.reshape(-1, 3)
        step = max(1, flat.shape[0] // 20000)
        flat = flat[::step]
        hues = np.array(
            [colorsys.rgb_to_hsv(*pixel)[0] * 360.0 for pixel in flat[:4000]]
        )
        sat = np.array([colorsys.rgb_to_hsv(*pixel)[1] for pixel in flat[:4000]])
        val = np.array([colorsys.rgb_to_hsv(*pixel)[2] for pixel in flat[:4000]])
        skin = (hues >= 5) & (hues <= 45) & (sat > 0.15) & (sat < 0.7) & (val > 0.2)
        return float(hues[skin].mean()) if skin.sum() >= 50 else None

    hue_before = mean_skin_hue(before)
    hue_after = mean_skin_hue(after)
    if hue_before is None or hue_after is None:
        return

    drift = abs(hue_after - hue_before)
    result.measurements["skin_hue_drift_deg"] = round(drift, 2)
    if drift > MAX_SKIN_HUE_DRIFT_DEG:
        result.violations.append(
            Violation("skin_hue_drift", f"skin hue moves {drift:.1f} degrees")
        )
