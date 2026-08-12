"""The darkroom pass: from a scored frame to two or three previews you can pick.

Ties the pieces together in the order that keeps them honest:

    RAW measurements  ->  intent  ->  candidates  ->  render  ->  validate  ->  Pareto

**Which decision comes from which domain**, because the distinction is easy to
overstate and worth being exact about:

    display appearance   what the frame visually lacks -- how bright it reads,
                         what colour cast it has, how much grain is visible.
                         Necessarily measured on the developed preview, because
                         that is what "looks too dark" means.
    RAW capacity         how far that can safely be corrected -- highlight
                         headroom, shadow headroom before the noise floor,
                         which channels actually saturated. Measured on the
                         sensor plane.
    render validation    whether the correction actually helped, measured by
                         rendering it and comparing.

So the sensor data **bounds** the tonal moves rather than originating all of
them: the target exposure comes from how the frame reads, and the ceiling on
reaching it comes from what the sensor holds. Saying "every tonal decision is
made from the sensor plane" would be an overstatement; saying the previous
version confused a rendered preview for RAW headroom would be exactly right.

The intent comes from the deterministic signals in `artistic.py`, so "preserve
the low-key structure" is a finding rather than a sentiment. Rendering and
validation decide which candidates survive -- nothing is proposed that has not
been made and looked at.

Only frames worth editing are put through it. Rendering costs roughly a second
each, and a frame already routed to `archive_only` does not need three readings
of its shadows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import edit_schema
import raw_measurements
import recipe_generator
import recipe_optimizer
import renderers.builtin  # noqa: F401  (registers the built-in engine)
import renderers.external  # noqa: F401  (registers darktable / rawtherapee)
from renderers import base as renderer_base

logger = logging.getLogger(__name__)

# Classes worth spending a render on.
WORTH_EDITING = frozenset(
    {"flagship", "stock_strong", "stock_standard", "art_candidate", "documentary_candidate"}
)


def should_run(route_class: str, artistic_scores) -> bool:
    """Edit the frames somebody might actually print, plus anything unusual."""
    if route_class in WORTH_EDITING:
        return True
    return bool(artistic_scores and artistic_scores.has_any_artistic_signal)


def run(
    asset,
    measurement,
    artistic_scores,
    *,
    out_dir: Path,
    renderer_name: str | None = None,
    write_sidecars: bool = True,
    faces_present: bool | None = None,
) -> dict:
    """Produce recipes, previews and sidecars for one frame."""
    result: dict = {
        "edit_recipes": [],
        "rendered_variants": {},
        "recipe_confidence": {},
        "preserve_intent": [],
        "sidecars": {},
        "rejected": [],
        "engine": "",
        "engine_version": "",
    }

    try:
        engine = renderer_base.get(renderer_name)
    except renderer_base.EngineUnavailable as e:
        logger.warning("Darkroom skipped: %s", e)
        result["rejected"].append(str(e))
        return result

    result["engine"] = engine.name
    result["engine_version"] = engine.version()

    raw_stats = raw_measurements.measure_or_empty(asset.path, asset.is_raw)

    recipes = recipe_generator.generate(
        asset_id=asset.asset_id,
        asset_key=asset.key,
        checksum=asset.checksum,
        raw_stats=raw_stats,
        mean_luma=measurement.mean_luma,
        stddev_luma=measurement.stddev_luma,
        channel_means=measurement.channel_means,
        noise=measurement.noise,
        tilt_degrees=_tilt_of(measurement),
        intent_signals=artistic_scores.signals if artistic_scores else [],
        is_raw=asset.is_raw,
        monochrome_worth_offering=_monochrome_worth_offering(measurement),
    )

    candidates, original = recipe_optimizer.evaluate(
        asset.path, recipes, renderer=engine, faces_present=faces_present
    )
    chosen = recipe_optimizer.choose(candidates)

    result["rejected"] = [
        f"{c.variant}: {'; '.join(c.rejected_for)}" for c in candidates if not c.ok
    ]
    result["rendered_variants"] = recipe_optimizer.write_variants(
        chosen, original, out_dir, asset.asset_id
    )

    for candidate in chosen:
        recipe = candidate.recipe
        result["edit_recipes"].append(
            {
                **recipe.to_dict(),
                "human_readable": edit_schema.describe(recipe),
                "scores": candidate.scores,
                "validation": candidate.validation.measurements if candidate.validation else {},
            }
        )
        result["recipe_confidence"][recipe.variant] = recipe.confidence.to_dict()
        if write_sidecars:
            result["sidecars"][recipe.variant] = _write_sidecars(recipe, out_dir)

    result["preserve_intent"] = list(chosen[0].recipe.preserve) if chosen else []
    return result


def _write_sidecars(recipe, out_dir: Path) -> dict[str, str]:
    """Always under `suggestions/`. Nothing here can reach an original."""
    from exporters import adobe_xmp, darktable_xmp, rawtherapee_pp3

    return {
        "recipe": str(edit_schema.write_recipe(recipe, out_dir)),
        "adobe": str(adobe_xmp.write_suggestion(recipe, out_dir)),
        "darktable": str(darktable_xmp.write_suggestion(recipe, out_dir)),
        "rawtherapee": str(rawtherapee_pp3.write_suggestion(recipe, out_dir)),
    }


def _tilt_of(measurement) -> float:
    for step in measurement.recipe or []:
        if "Straighten" in step:
            import re

            found = re.search(r"([\d.]+)°", step)
            if found:
                degrees = float(found.group(1))
                return -degrees if "clockwise" in step and "anticlock" not in step else degrees
    return 0.0


def _monochrome_worth_offering(measurement) -> bool:
    """Only when the colour is not carrying the frame.

    Offering black and white on every artistic frame turns the tool into a style
    preset. A near-neutral frame is one where the conversion genuinely changes
    the question being asked; a frame built on a colour relationship is not.
    """
    r, g, b = measurement.channel_means
    if max(r, g, b) <= 1.0:
        return False
    spread = (max(r, g, b) - min(r, g, b)) / max(max(r, g, b), 1.0)
    return spread < 0.06


def format_report(record) -> str:
    """The DARKROOM ASSISTANT block, for the terminal and the HTML report."""
    recipes = record.edit_recipes or []
    if not recipes:
        return ""

    lines = ["", "DARKROOM ASSISTANT", ""]
    lines.append(f"Intent:\n{recipes[0].get('intent', '')}")
    if record.preserve_intent:
        lines.append("\nDo not destroy:")
        lines.extend(f"  - {item}" for item in record.preserve_intent)

    for recipe in recipes:
        variant = recipe.get("variant", "?").title()
        lines.append(f"\nVariant {variant}:")
        lines.extend(f"  {step}" for step in recipe.get("human_readable", []))

    warnings = recipes[0].get("warnings") or []
    if warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  {w}" for w in warnings)

    confidence = record.recipe_confidence or {}
    if confidence:
        first = next(iter(confidence.values()))
        lines.append(
            f"\nConfidence: tone {first.get('tone', 0):.2f}, "
            f"colour {first.get('color', 0):.2f}, crop {first.get('crop', 0):.2f}"
        )
    if record.rendered_variants:
        lines.append("\nPreviews: " + ", ".join(sorted(record.rendered_variants)))
    return "\n".join(lines)
