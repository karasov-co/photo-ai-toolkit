"""Candidate edits, derived from the sensor rather than imagined from a preview.

A language model looking at a 512px JPEG cannot know that `Highlights -47` is
right, because the JPEG has already spent the highlight it would be recovering.
It can propose a *direction* -- "pull the sky down, keep the shadows heavy" --
and that is what the intent is for. The numbers come from measurement and are
then checked by rendering.

So the split here is deliberate:

    intent      what the frame is trying to be        (model, or inferred)
    numbers     how far the data actually allows      (RAW measurements)
    winner      which of them survives rendering      (recipe_optimizer)

Three variants and no more. Faithful corrects only what is demonstrably
limiting the frame. Expressive strengthens whatever is already there.
Monochrome is offered **only** when the frame's own colour is not carrying it --
proposing black and white on every artistic frame is how a tool becomes a style
preset.

Every candidate carries a `preserve` list built from the intentionality signals,
so a deliberate tilt or a low-key structure is protected by name and the
validator can refuse any candidate that would undo it.
"""

from __future__ import annotations

import logging

from photoai.edit_schema import (
    ColorTreatment,
    Confidence,
    Detail,
    EditRecipe,
    Geometry,
    GlobalAdjustments,
    Variant,
)

logger = logging.getLogger(__name__)

# Camera Raw's exposure slider runs to +/-5, but past about a stop and a half of
# lift on a real sensor the shadows turn to colour noise. The generator stays
# inside what the measured headroom supports.
MAX_LIFT_EV = 1.5
MAX_PULL_EV = 1.0

# Below this measured saturation there is nothing for a highlight slider to do,
# and proposing one is theatre.
MIN_CLIPPING_TO_RECOVER = 0.002

TARGET_MEAN = 0.46  # display-referred, where a normally exposed frame sits


def generate(
    *,
    asset_id: str,
    asset_key: str,
    checksum: str,
    raw_stats,
    mean_luma: float,
    stddev_luma: float,
    channel_means: tuple[float, float, float],
    noise: float,
    tilt_degrees: float,
    intent_signals: list,
    is_raw: bool,
    monochrome_worth_offering: bool = False,
) -> list[EditRecipe]:
    """Build the candidate set for one frame."""
    preserve, warnings = _preserve_from(intent_signals)
    evidence = _evidence_from(raw_stats, intent_signals)
    intent = _intent_from(intent_signals, mean_luma)

    exposure = _exposure_for(mean_luma, raw_stats, is_raw)
    highlights = _highlights_for(raw_stats, is_raw)
    shadows = _shadows_for(mean_luma, raw_stats, preserve)
    temperature, tint = _white_balance_for(channel_means)
    denoise = _denoise_for(noise)
    sharpening, masking = _sharpening_for(noise, preserve)

    def base(variant: Variant) -> EditRecipe:
        return EditRecipe(
            asset_id=asset_id,
            asset_key=asset_key,
            source_checksum=checksum,
            variant=variant.value,
            intent=intent,
            preserve=list(preserve),
            warnings=list(warnings),
            evidence=list(evidence),
            confidence=Confidence(
                tone=0.85 if (is_raw and raw_stats.available) else 0.5,
                color=0.6,
                # Deliberately low. A crop is a compositional judgement, and
                # nothing here has any way to make one well.
                crop=0.35,
                detail=0.7 if noise > 0 else 0.5,
            ),
            geometry=Geometry(
                rotation_deg=0.0 if _tilt_is_deliberate(intent_signals) else -tilt_degrees,
                preserve_existing_tilt=_tilt_is_deliberate(intent_signals),
            ),
        )

    faithful = base(Variant.FAITHFUL)
    faithful.global_adjustments = GlobalAdjustments(
        exposure_ev=round(exposure, 2),
        highlights=highlights,
        shadows=shadows,
        blacks=-4 if shadows > 0 else 0,
        temperature_delta_k=temperature,
        tint_delta=tint,
        contrast=6 if stddev_luma < 45 else 0,
        vibrance=4,
    )
    faithful.detail = Detail(
        denoise_luminance=denoise,
        denoise_color=min(100, denoise * 2),
        sharpening=sharpening,
        masking=masking,
    )

    expressive = base(Variant.EXPRESSIVE)
    expressive.global_adjustments = GlobalAdjustments(
        # Less exposure, more shaping: an expressive read commits to the mood
        # the frame already has rather than normalising it towards the middle.
        exposure_ev=round(exposure * 0.45, 2),
        highlights=min(0, highlights - 8),
        shadows=max(0, shadows - 6),
        blacks=-12,
        whites=-4,
        contrast=14,
        clarity=8,
        temperature_delta_k=temperature,
        tint_delta=tint,
        vibrance=8,
        saturation=-3,
    )
    expressive.detail = Detail(
        denoise_luminance=max(0, denoise - 4),
        denoise_color=min(100, denoise * 2),
        sharpening=sharpening,
        masking=masking,
    )
    expressive.intent = f"{intent} (strengthened)"

    candidates = [faithful, expressive]

    if monochrome_worth_offering:
        mono = base(Variant.MONOCHROME)
        mono.global_adjustments = GlobalAdjustments(
            exposure_ev=round(exposure * 0.6, 2),
            highlights=highlights,
            shadows=shadows,
            blacks=-10,
            contrast=18,
            clarity=6,
        )
        mono.color = ColorTreatment(style="monochrome", monochrome=True)
        mono.detail = Detail(denoise_luminance=denoise, sharpening=sharpening, masking=masking)
        candidates.append(mono)

    return candidates


# --- each number, and where it comes from ------------------------------------


def _exposure_for(mean_luma: float, raw_stats, is_raw: bool) -> float:
    """Move the frame towards a normal mean, bounded by what the data allows.

    Lift is capped harder than pull because lifting amplifies the noise floor
    while pulling only discards headroom.
    """
    current = max(mean_luma, 1.0) / 255.0
    import math

    needed = math.log2(TARGET_MEAN / current)

    if needed > 0:
        ceiling = MAX_LIFT_EV
        if is_raw and raw_stats.available:
            # Never lift further than the shadow headroom supports: past that
            # point the lift is amplifying read noise, not recovering detail.
            ceiling = min(MAX_LIFT_EV, max(0.0, raw_stats.shadow_headroom_stops))
        return max(0.0, min(needed, ceiling))
    return max(-MAX_PULL_EV, needed)


def _highlights_for(raw_stats, is_raw: bool) -> int:
    """Only propose recovery when there is something measured to recover."""
    if not (is_raw and raw_stats.available):
        return -15  # a cautious default: a JPEG's headroom is unknown
    if raw_stats.clipped_any_channel < MIN_CLIPPING_TO_RECOVER:
        return 0
    if raw_stats.truly_blown > 0.05:
        # Every channel saturated: the slider cannot invent what is not there,
        # and pulling hard only greys the area.
        return -20
    reach = min(1.0, raw_stats.highlight_headroom_stops / 2.0)
    return -int(round(20 + 35 * reach))


def _shadows_for(mean_luma: float, raw_stats, preserve: list[str]) -> int:
    """Lift, unless the darkness is the point."""
    if any("low-key" in p or "dark" in p.lower() for p in preserve):
        return 6  # a token lift only: enough to read, not enough to flatten
    if mean_luma > 110:
        return 0
    if raw_stats.available and not raw_stats.can_lift_shadows:
        return 8
    return int(round(min(35, (110 - mean_luma) * 0.35)))


def _white_balance_for(channel_means: tuple[float, float, float]) -> tuple[int, int]:
    """Half-correct a measured cast. Full correction kills an intended one."""
    r, g, b = channel_means
    if max(r, g, b) <= 1.0:
        return 0, 0
    warmth = (r - b) / max((r + b) / 2.0, 1.0)
    kelvin = int(round(-warmth * 900))
    green = (g - (r + b) / 2.0) / max(g, 1.0)
    tint = int(round(-green * 40))
    return max(-1000, min(1000, kelvin)), max(-30, min(30, tint))


def _denoise_for(noise: float) -> int:
    if noise <= 2.0:
        return 0
    return int(round(min(45, (noise - 2.0) * 6)))


def _sharpening_for(noise: float, preserve: list[str]) -> tuple[int, int]:
    """No sharpening where blur is the subject, and heavy masking where noisy."""
    if any("blur" in p.lower() or "motion" in p.lower() for p in preserve):
        return 0, 0
    masking = int(min(85, 40 + noise * 4))
    return 20, masking


def _tilt_is_deliberate(signals: list) -> bool:
    return any(
        "tilt" in s.defect and s.verdict == "likely_intentional" for s in signals
    )


def _preserve_from(signals: list) -> tuple[list[str], list[str]]:
    """Turn intentionality findings into things an edit must not undo."""
    preserve: list[str] = []
    warnings: list[str] = []
    for signal in signals:
        if signal.verdict != "likely_intentional":
            continue
        if signal.defect == "darkness":
            preserve.append("the low-key structure and the weight of the shadows")
            warnings.append("Lifting shadows past about +15 will flatten the low-key structure")
        elif "motion" in signal.defect or signal.defect == "softness":
            preserve.append("the intentional blur, which is carrying the frame")
            warnings.append("Do not sharpen: the blur is the subject, not a defect")
        elif "tilt" in signal.defect:
            preserve.append("the deliberate tilt")
            warnings.append("Do not straighten the horizon; the angle is doing work")
        elif signal.defect == "grain":
            preserve.append("the natural grain")
            warnings.append("Heavy denoise will remove the texture and the sense of presence")
        elif "highlight" in signal.defect:
            preserve.append("the high-key treatment")
    return preserve, warnings


def _evidence_from(raw_stats, signals: list) -> list[str]:
    from photoai import raw_measurements

    evidence = list(raw_measurements.summarise(raw_stats))
    evidence.extend(s.evidence for s in signals if s.verdict == "likely_intentional")
    return evidence


def _intent_from(signals: list, mean_luma: float) -> str:
    deliberate = [s.defect for s in signals if s.verdict == "likely_intentional"]
    if "darkness" in deliberate:
        return "preserve_low_key_mood"
    if any("motion" in d for d in deliberate):
        return "preserve_motion"
    if any("tilt" in d for d in deliberate):
        return "preserve_deliberate_angle"
    if mean_luma < 70:
        return "open_the_frame_without_flattening_it"
    return "correct_only_what_limits_the_frame"
