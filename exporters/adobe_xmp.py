"""Camera Raw / Lightroom sidecar, written where it cannot collide.

Adobe stores development settings in an XMP sidecar and leaves the RAW itself
untouched, which is why a suggestion can be offered as a file at all. The
danger is the filename: a converter looks for `<stem>.xmp`, so writing there
replaces whatever the photographer has already done to the frame with no
warning and no undo.

Everything here therefore writes to `suggestions/<asset_id>/<variant>.ai-suggested.xmp`
and `apply` refuses to overwrite an existing sidecar unless told to, after
showing a diff.

The `crs:` namespace keys are Adobe's published slider names. They are stable
in practice, and no claim is made that this reproduces Adobe's colour science --
only that the numbers land in the sliders a photographer would have moved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="photo-ai-toolkit">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:pat="https://github.com/karasov-co/photo-ai-toolkit/ns/1.0/"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
{settings}
    pat:suggestedBy="photo-ai-toolkit"
    pat:variant="{variant}"
    pat:intent="{intent}"
    pat:sourceChecksum="{checksum}"
    pat:engine="{engine}">
   <pat:preserve><rdf:Bag>
{preserve}
   </rdf:Bag></pat:preserve>
   <pat:warnings><rdf:Bag>
{warnings}
   </rdf:Bag></pat:warnings>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def to_adobe_xmp(recipe) -> str:
    """One recipe as a Camera Raw sidecar document."""
    g = recipe.global_adjustments
    settings = {
        "crs:Exposure2012": f"{g.exposure_ev:+.2f}",
        "crs:Contrast2012": str(g.contrast),
        "crs:Highlights2012": str(g.highlights),
        "crs:Shadows2012": str(g.shadows),
        "crs:Whites2012": str(g.whites),
        "crs:Blacks2012": str(g.blacks),
        "crs:Texture": str(g.texture),
        "crs:Clarity2012": str(g.clarity),
        "crs:Dehaze": str(g.dehaze),
        "crs:Vibrance": str(g.vibrance),
        "crs:Saturation": str(g.saturation),
        "crs:LuminanceSmoothing": str(recipe.detail.denoise_luminance),
        "crs:ColorNoiseReduction": str(recipe.detail.denoise_color),
        "crs:Sharpness": str(recipe.detail.sharpening),
        "crs:SharpenEdgeMasking": str(recipe.detail.masking),
        "crs:ConvertToGrayscale": "True" if recipe.color.monochrome else "False",
    }
    # Camera Raw's temperature is absolute; only a delta is known here, so it is
    # carried in the toolkit namespace rather than written into crs:Temperature,
    # where it would be read as an absolute Kelvin value and wreck the frame.
    if g.temperature_delta_k:
        settings["pat:temperatureDeltaK"] = str(g.temperature_delta_k)
    if g.tint_delta:
        settings["pat:tintDelta"] = str(g.tint_delta)

    if recipe.geometry.rotation_deg and not recipe.geometry.preserve_existing_tilt:
        settings["crs:CropAngle"] = f"{recipe.geometry.rotation_deg:.2f}"
    crop = recipe.geometry.crop
    if crop is not None and not crop.is_identity:
        settings.update(
            {
                "crs:HasCrop": "True",
                "crs:CropLeft": f"{crop.left:.6f}",
                "crs:CropTop": f"{crop.top:.6f}",
                "crs:CropRight": f"{crop.right:.6f}",
                "crs:CropBottom": f"{crop.bottom:.6f}",
            }
        )

    body = "\n".join(f'    {key}="{escape(str(value))}"' for key, value in settings.items())
    return TEMPLATE.format(
        settings=body,
        variant=escape(recipe.variant),
        intent=escape(recipe.intent),
        checksum=escape(recipe.source_checksum),
        engine=escape(f"{recipe.engine} {recipe.engine_version}".strip()),
        preserve="\n".join(f"    <rdf:li>{escape(p)}</rdf:li>" for p in recipe.preserve),
        warnings="\n".join(f"    <rdf:li>{escape(w)}</rdf:li>" for w in recipe.warnings),
    )


@dataclass
class ApplyPlan:
    """What `apply-recipe` would do. Always produced before anything is written."""

    target: Path
    exists: bool
    would_overwrite: bool
    diff: list[str]
    stale: bool = False

    @property
    def safe(self) -> bool:
        return not self.would_overwrite and not self.stale


def plan_apply(recipe, raw_path: Path, *, current_checksum: str, force: bool = False) -> ApplyPlan:
    """Work out whether writing next to the RAW would destroy existing work."""
    target = Path(raw_path).with_suffix(".xmp")
    exists = target.exists()
    stale = recipe.is_stale_for(current_checksum)

    diff: list[str] = []
    if exists:
        try:
            existing = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:  # pragma: no cover
            existing = f"<unreadable: {e}>"
        diff = _diff_settings(existing, to_adobe_xmp(recipe))

    return ApplyPlan(
        target=target,
        exists=exists,
        would_overwrite=exists and not force,
        diff=diff,
        stale=stale,
    )


def _diff_settings(existing: str, proposed: str) -> list[str]:
    """Slider-level differences, so a person can see what would change."""
    import re

    def settings_of(document: str) -> dict[str, str]:
        return dict(re.findall(r'(crs:\w+)\s*=\s*"([^"]*)"', document))

    before, after = settings_of(existing), settings_of(proposed)
    lines: list[str] = []
    for key in sorted(set(before) | set(after)):
        was, now = before.get(key), after.get(key)
        if was != now:
            lines.append(f"  {key}: {was if was is not None else '-'} -> {now if now is not None else '-'}")
    return lines


def write_suggestion(recipe, root: Path) -> Path:
    """Always safe: writes only under `suggestions/`, never beside the RAW."""
    from edit_schema import suggestion_path

    path = suggestion_path(root, recipe.asset_id or "unknown", recipe.variant, ".xmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_adobe_xmp(recipe), encoding="utf-8")
    return path
