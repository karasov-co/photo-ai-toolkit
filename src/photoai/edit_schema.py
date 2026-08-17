"""The editor-neutral edit recipe, and the rules that keep it honest.

A recipe is a *proposal*, expressed in units every raw converter understands, and
deliberately not in any one converter's file format. Adobe Camera Raw and
darktable both store development settings in a sidecar XMP, but they use
different schemas and interpret the same slider differently, so a single file
claiming to be both is a file that is correctly read by neither. Adapters
translate; the recipe itself stays neutral.

Three properties are structural rather than conventional:

- **It is bound to the bytes it was computed from.** `source_checksum` pins the
  recipe to a specific file. Edit the RAW and the recipe is stale, because the
  measurements it came from describe an image that no longer exists.
- **It never lands on an original.** Recipes are written under `suggestions/`
  with an `.ai-suggested` infix, so nothing can collide with a sidecar the
  photographer wrote.
- **It records what must not be destroyed.** `preserve` is as load-bearing as
  the slider values: "raise the shadows" and "do not raise the shadows enough
  to flatten the low-key structure" are the same instruction only when both
  halves survive.

Slider ranges follow the Camera Raw convention (-100..+100, exposure in stops)
because it is the one most photographers can read without a translation table.
That is a presentation choice, not a claim of Adobe compatibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

SCHEMA_VERSION = 1

SUGGESTION_INFIX = ".ai-suggested"
SUGGESTIONS_DIRNAME = "suggestions"


class Variant(StrEnum):
    """Three readings of one frame. Never more -- a menu is not a decision.

    Deliberately excludes a monochrome variant by default. Offering black and
    white on every artistic frame turns the tool into a style filter, and the
    frames where it genuinely helps are a minority that a person spots faster
    than a heuristic does.
    """

    FAITHFUL = "faithful"
    EXPRESSIVE = "expressive"
    MONOCHROME = "monochrome"


@dataclass
class GlobalAdjustments:
    """Camera Raw slider conventions: exposure in stops, the rest -100..+100."""

    exposure_ev: float = 0.0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    whites: int = 0
    blacks: int = 0
    temperature_delta_k: int = 0
    tint_delta: int = 0
    texture: int = 0
    clarity: int = 0
    dehaze: int = 0
    vibrance: int = 0
    saturation: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Crop:
    """Normalised to 0..1 **after** EXIF orientation has been applied.

    Storing pixels would make the recipe depend on which decoder read the file;
    storing them before orientation would rotate the crop off the subject on
    every portrait-orientation frame.
    """

    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0
    aspect_ratio: str = ""
    reason: str = ""
    confidence: float = 0.0

    @property
    def keeps(self) -> float:
        return max(0.0, (self.right - self.left)) * max(0.0, (self.bottom - self.top))

    @property
    def is_identity(self) -> bool:
        return self.keeps >= 0.999

    def to_dict(self) -> dict:
        return {**asdict(self), "keeps": round(self.keeps, 4)}


@dataclass
class Geometry:
    rotation_deg: float = 0.0
    crop: Crop | None = None
    # When the tilt was read as deliberate, straightening it destroys the point.
    preserve_existing_tilt: bool = False

    def to_dict(self) -> dict:
        return {
            "rotation_deg": self.rotation_deg,
            "crop": self.crop.to_dict() if self.crop else None,
            "preserve_existing_tilt": self.preserve_existing_tilt,
        }


@dataclass
class Detail:
    denoise_luminance: int = 0
    denoise_color: int = 0
    sharpening: int = 0
    masking: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ColorTreatment:
    style: str = "neutral"
    monochrome: bool = False
    hsl: list[dict] = field(default_factory=list)
    grading: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Confidence:
    """Per-area, because they are not equally knowable.

    Tone can be derived from the histogram and is usually well founded. Crop is
    a compositional judgement and is usually not. Reporting one number for both
    lends the weakest part the credibility of the strongest.
    """

    tone: float = 0.0
    color: float = 0.0
    crop: float = 0.0
    detail: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EditRecipe:
    asset_id: str = ""
    asset_key: str = ""
    source_checksum: str = ""
    variant: str = Variant.FAITHFUL.value
    intent: str = ""
    schema_version: int = SCHEMA_VERSION

    global_adjustments: GlobalAdjustments = field(default_factory=GlobalAdjustments)
    geometry: Geometry = field(default_factory=Geometry)
    detail: Detail = field(default_factory=Detail)
    color: ColorTreatment = field(default_factory=ColorTreatment)
    confidence: Confidence = field(default_factory=Confidence)

    # What an edit must not undo. Written from the intentionality signals, so a
    # deliberate tilt or a low-key structure is protected by name.
    preserve: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    # Provenance of the recipe itself, so a result can be traced.
    engine: str = ""
    engine_version: str = ""
    generator_version: str = SCHEMA_VERSION
    uses_generative: bool = False

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "asset_key": self.asset_key,
            "source_checksum": self.source_checksum,
            "variant": self.variant,
            "intent": self.intent,
            "global": self.global_adjustments.to_dict(),
            "geometry": self.geometry.to_dict(),
            "detail": self.detail.to_dict(),
            "color": self.color.to_dict(),
            "confidence": self.confidence.to_dict(),
            "preserve": self.preserve,
            "warnings": self.warnings,
            "evidence": self.evidence,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "uses_generative": self.uses_generative,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> EditRecipe:
        recipe = cls(
            asset_id=payload.get("asset_id", ""),
            asset_key=payload.get("asset_key", ""),
            source_checksum=payload.get("source_checksum", ""),
            variant=payload.get("variant", Variant.FAITHFUL.value),
            intent=payload.get("intent", ""),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            preserve=list(payload.get("preserve") or []),
            warnings=list(payload.get("warnings") or []),
            evidence=list(payload.get("evidence") or []),
            engine=payload.get("engine", ""),
            engine_version=payload.get("engine_version", ""),
            uses_generative=bool(payload.get("uses_generative", False)),
        )
        recipe.global_adjustments = GlobalAdjustments(
            **{k: v for k, v in (payload.get("global") or {}).items()
               if k in GlobalAdjustments.__dataclass_fields__}
        )
        geometry = payload.get("geometry") or {}
        crop_payload = geometry.get("crop")
        recipe.geometry = Geometry(
            rotation_deg=float(geometry.get("rotation_deg", 0.0)),
            crop=Crop(**{k: v for k, v in crop_payload.items()
                         if k in Crop.__dataclass_fields__}) if crop_payload else None,
            preserve_existing_tilt=bool(geometry.get("preserve_existing_tilt", False)),
        )
        recipe.detail = Detail(
            **{k: v for k, v in (payload.get("detail") or {}).items()
               if k in Detail.__dataclass_fields__}
        )
        recipe.color = ColorTreatment(
            **{k: v for k, v in (payload.get("color") or {}).items()
               if k in ColorTreatment.__dataclass_fields__}
        )
        recipe.confidence = Confidence(
            **{k: v for k, v in (payload.get("confidence") or {}).items()
               if k in Confidence.__dataclass_fields__}
        )
        return recipe

    # --- staleness ----------------------------------------------------------

    def is_stale_for(self, checksum: str) -> bool:
        """A recipe describes one specific set of bytes and no other."""
        return bool(self.source_checksum) and self.source_checksum != checksum

    def matches(self, checksum: str) -> bool:
        return not self.is_stale_for(checksum)

    # --- the protected list -------------------------------------------------

    def protects(self, what: str) -> bool:
        return any(what.lower() in item.lower() for item in self.preserve)

    @property
    def is_noop(self) -> bool:
        return self.global_adjustments == GlobalAdjustments() and self.detail == Detail()


def suggestion_path(root: Path, asset_id: str, variant: str, suffix: str) -> Path:
    """`suggestions/<asset_id>/<variant>.ai-suggested<suffix>`.

    The infix is what makes a collision with the photographer's own sidecar
    impossible: no converter writes `.ai-suggested.xmp`.
    """
    return Path(root) / SUGGESTIONS_DIRNAME / asset_id / f"{variant}{SUGGESTION_INFIX}{suffix}"


def write_recipe(recipe: EditRecipe, root: Path) -> Path:
    path = suggestion_path(root, recipe.asset_id or "unknown", recipe.variant, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_recipe(path: Path) -> EditRecipe:
    return EditRecipe.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def describe(recipe: EditRecipe) -> list[str]:
    """The human-readable form. Sliders a person can type into their converter."""
    g = recipe.global_adjustments
    lines: list[str] = []

    def add(label: str, value, unit: str = "", threshold: float = 0.0) -> None:
        if isinstance(value, int | float) and abs(value) > threshold:
            lines.append(f"{label} {value:+g}{unit}")

    add("Exposure", round(g.exposure_ev, 2), " EV", 0.04)
    add("Contrast", g.contrast)
    add("Highlights", g.highlights)
    add("Shadows", g.shadows)
    add("Whites", g.whites)
    add("Blacks", g.blacks)
    add("Temperature", g.temperature_delta_k, " K")
    add("Tint", g.tint_delta)
    add("Texture", g.texture)
    add("Clarity", g.clarity)
    add("Dehaze", g.dehaze)
    add("Vibrance", g.vibrance)
    add("Saturation", g.saturation)

    if recipe.detail.denoise_luminance:
        lines.append(f"Luminance denoise {recipe.detail.denoise_luminance}")
    if recipe.detail.denoise_color:
        lines.append(f"Colour denoise {recipe.detail.denoise_color}")
    if recipe.detail.sharpening:
        lines.append(
            f"Sharpening {recipe.detail.sharpening} (masking {recipe.detail.masking})"
        )
    if recipe.geometry.rotation_deg:
        lines.append(f"Rotate {recipe.geometry.rotation_deg:+.1f}°")
    if recipe.geometry.crop and not recipe.geometry.crop.is_identity:
        crop = recipe.geometry.crop
        lines.append(
            f"Crop to {crop.aspect_ratio or 'suggested box'}, keeping {crop.keeps:.0%}"
            f" (confidence {crop.confidence:.2f})"
        )
    if recipe.color.monochrome:
        lines.append("Convert to monochrome")

    return lines or ["No adjustment: the frame is already where it should be."]
