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
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
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


# Camera Raw's Temperature slider is two different things depending on the file.
# For a RAW it is absolute Kelvin, so a delta cannot be written into it without
# knowing what the camera shot at. For a rendered file -- JPEG, TIFF -- it is a
# relative slider from -100 to +100, and a delta is exactly what it wants.
#
# The old code handled this by writing the delta into `pat:temperatureDeltaK`,
# a namespace only this project reads. The reasoning was right and the
# consequence was never followed through: white balance is the single
# correction that matters most on a frame with a cast, and it was the one
# correction guaranteed to be dropped on the way to the editor. It now lands
# where the editor reads it whenever that is possible, and says so in a warning
# when it is not.
RELATIVE_TEMPERATURE_PER_1000K = 12.0
RELATIVE_TEMPERATURE_LIMIT = 100


def temperature_settings(
    recipe, *, is_raw: bool, as_shot_temperature_k: int | None = None
) -> tuple[dict[str, str], str]:
    """The white-balance keys Adobe will actually read, plus what was lost."""
    g = recipe.global_adjustments
    delta = int(g.temperature_delta_k or 0)
    tint = int(g.tint_delta or 0)
    out: dict[str, str] = {}
    if not delta and not tint:
        return out, ""

    if not is_raw:
        # -100..+100 relative. Roughly 12 points per 1000K, clamped: the mapping
        # is approximate by nature and a clamp keeps an extreme measurement from
        # writing a slider nobody would have moved that far.
        scaled = delta / 1000.0 * RELATIVE_TEMPERATURE_PER_1000K
        value = max(-RELATIVE_TEMPERATURE_LIMIT, min(RELATIVE_TEMPERATURE_LIMIT, round(scaled)))
        if value:
            out["crs:Temperature"] = str(value)
        if tint:
            out["crs:Tint"] = str(max(-RELATIVE_TEMPERATURE_LIMIT, min(RELATIVE_TEMPERATURE_LIMIT, tint)))
        return out, ""

    if as_shot_temperature_k:
        out["crs:Temperature"] = str(max(2000, min(50000, int(as_shot_temperature_k) + delta)))
        if tint:
            out["crs:Tint"] = str(max(-150, min(150, tint)))
        return out, ""

    # A RAW whose as-shot temperature nobody could read. Saying so is the only
    # honest option; the alternative is writing a delta into an absolute field
    # and turning a 5200K frame into a 200K one.
    return {}, (
        f"white balance measured at {delta:+d}K but not written: Camera Raw needs an "
        "absolute temperature for a RAW file and the as-shot value could not be read. "
        "Move the Temp slider by roughly this much by hand."
    )


def _settings(recipe, *, is_raw: bool = True, as_shot_temperature_k: int | None = None) -> dict:
    """Every crs: key for one recipe, in the order a person reads them."""
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
    white_balance, _lost = temperature_settings(
        recipe, is_raw=is_raw, as_shot_temperature_k=as_shot_temperature_k
    )
    settings.update(white_balance)
    # Kept alongside, not instead of: the delta is what was measured, and a
    # later exporter or a person reading the file should still be able to see it.
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

    return settings


def to_adobe_xmp(recipe, *, is_raw: bool = True, as_shot_temperature_k: int | None = None) -> str:
    """One recipe as a Camera Raw sidecar document."""
    settings = _settings(recipe, is_raw=is_raw, as_shot_temperature_k=as_shot_temperature_k)
    _, lost = temperature_settings(
        recipe, is_raw=is_raw, as_shot_temperature_k=as_shot_temperature_k
    )
    warnings = list(recipe.warnings) + ([lost] if lost else [])
    body = "\n".join(f'    {key}="{escape(str(value))}"' for key, value in settings.items())
    return TEMPLATE.format(
        settings=body,
        variant=escape(recipe.variant),
        intent=escape(recipe.intent),
        checksum=escape(recipe.source_checksum),
        engine=escape(f"{recipe.engine} {recipe.engine_version}".strip()),
        preserve="\n".join(f"    <rdf:li>{escape(p)}</rdf:li>" for p in recipe.preserve),
        warnings="\n".join(f"    <rdf:li>{escape(w)}</rdf:li>" for w in warnings),
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
    from photoai.edit_schema import suggestion_path

    path = suggestion_path(root, recipe.asset_id or "unknown", recipe.variant, ".xmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_adobe_xmp(recipe), encoding="utf-8")
    return path


# --- an actual Lightroom preset ------------------------------------------------
#
# The sidecar above is develop settings and nothing else. Lightroom will read it
# beside a RAW, and that is the only place it works: dropped into Import
# Presets it appears in the panel named `<x:xmpmet` -- the root XML tag, because
# nothing inside it is a field a preset has -- with an Amount slider that does
# nothing from 0 to 100, because there is nothing to apply. Beside a JPEG it is
# ignored outright; Lightroom does not read sidecars for rendered files.
#
# A preset is a different document: the same settings plus an identity
# (`PresetType`, `Name`, `UUID`) and a declaration of what it supports. It also
# works on any file format, which is the only route this tool has to a JPEG.
#
# Written from Adobe's preset structure as observed in exported .xmp presets.
# Nobody here has watched Lightroom load one of these, and the export says so.

PRESET_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="photo-ai-toolkit">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal"
    crs:Cluster=""
    crs:UUID="{uuid}"
    crs:SupportsAmount="True"
    crs:SupportsAmount2="True"
    crs:SupportsColor="True"
    crs:SupportsMonochrome="True"
    crs:SupportsHighDynamicRange="True"
    crs:SupportsNormalDynamicRange="True"
    crs:SupportsSceneReferred="True"
    crs:SupportsOutputReferred="True"
    crs:CameraModelRestriction=""
    crs:Copyright=""
    crs:ContactInfo=""
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
{settings}
    crs:HasSettings="True">
   <crs:Name>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{name}</rdf:li>
    </rdf:Alt>
   </crs:Name>
   <crs:ShortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{short_name}</rdf:li>
    </rdf:Alt>
   </crs:ShortName>
   <crs:SortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{name}</rdf:li>
    </rdf:Alt>
   </crs:SortName>
   <crs:Group>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{group}</rdf:li>
    </rdf:Alt>
   </crs:Group>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

PRESET_GROUP = "photo-ai-toolkit"


def preset_uuid(recipe) -> str:
    """Stable per photograph and variant, so re-exporting replaces rather than
    stacks a second copy in the panel with the same name."""
    import hashlib

    seed = f"{recipe.asset_id}:{recipe.source_checksum}:{recipe.variant}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest().upper()[:32]


def preset_name(recipe, stem: str = "") -> str:
    """What a person will read in the Presets panel.

    The name that appeared there was `<x:xmpmet`, which is what Lightroom falls
    back to when a document has no `crs:Name`. A filename and a variant is the
    least a photographer needs to tell two of these apart.
    """
    label = stem or recipe.asset_id or "photograph"
    return f"{label} — {recipe.variant}"


def to_lightroom_preset(recipe, *, stem: str = "", is_raw: bool = True,
                        as_shot_temperature_k: int | None = None) -> str:
    """One recipe as a preset Lightroom can import and apply at any amount."""
    settings = _settings(recipe, is_raw=is_raw, as_shot_temperature_k=as_shot_temperature_k)
    # `crs:` only. The provenance keys live in this project's own namespace,
    # which the preset document does not declare -- emitting them here produced
    # an undeclared-prefix document, and a malformed preset is one Lightroom
    # drops entirely rather than one it applies badly.
    settings = {k: v for k, v in settings.items() if k.startswith("crs:")}
    body = "\n".join(f'    {key}="{escape(str(value))}"' for key, value in settings.items())
    name = preset_name(recipe, stem)
    return PRESET_TEMPLATE.format(
        uuid=preset_uuid(recipe),
        settings=body,
        name=escape(name),
        short_name=escape(name[:31]),
        group=escape(PRESET_GROUP),
    )


# --- the decision, in the form a catalogue understands -------------------------
#
# A farm of symlinks and an HTML page is not how anybody culls. The decision
# lives in the catalogue: a star rating, a colour label, a pick flag. Lightroom,
# Bridge, Capture One and Photo Mechanic all read `xmp:Rating` and `xmp:Label`
# from a sidecar, so the sorting this tool did can arrive as something a
# photographer already sorts by -- rather than as a folder they have to read
# beside their editor.
#
# Deliberately not written into the recipe sidecar by default: a rating is an
# opinion about the photograph and a recipe is an opinion about the file, and
# somebody may want one without the other.

# Copying one of these next to a RAW replaces `<stem>.xmp`, which is where the
# photographer's own develop settings, crop, ratings and keywords already live.
# `plan_apply` above exists because that exact collision was thought through for
# develop settings; writing ratings straight over the same file would have
# reopened the door on the other side. `merge_rating` is the way in: it reads
# what is there, changes two attributes, and leaves everything else byte for
# byte.


def merge_rating(existing: str, record) -> str:
    """The photographer's sidecar with only `xmp:Rating` and `xmp:Label` changed.

    Everything else -- develop settings, crop, keywords, GPS, their own stars if
    this is a re-run -- is preserved exactly. A regex rather than a parse on
    purpose: round-tripping XMP through an XML library reorders attributes and
    rewrites namespaces, and a diff nobody can read is a diff nobody checks.
    """
    import re

    document = existing
    values = {
        "xmp:Rating": str(STARS.get((getattr(record, "category", "") or "").upper(), 0)),
        "xmp:Label": LABELS.get((getattr(record, "category", "") or "").upper(), ""),
    }
    for key, value in values.items():
        pattern = re.compile(rf'{re.escape(key)}\s*=\s*"[^"]*"')
        if pattern.search(document):
            document = pattern.sub(f'{key}="{escape(value)}"', document, count=1)
        else:
            # No such attribute yet: add it to the first rdf:Description, with
            # the namespace if the document does not already declare it.
            if "xmlns:xmp=" not in document:
                document = document.replace(
                    "<rdf:Description",
                    '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
                    1,
                )
            document = document.replace(
                "<rdf:Description", f'<rdf:Description {key}="{escape(value)}"', 1
            )
    return document


def plan_rating(record, target: Path) -> RatingPlan:
    """What writing this rating beside an original would do. Never writes."""
    target = Path(target)
    if not target.exists():
        return RatingPlan(target=target, exists=False, merged=to_rating_sidecar(record))
    existing = target.read_text(encoding="utf-8", errors="replace")
    merged = merge_rating(existing, record)
    return RatingPlan(
        target=target, exists=True, merged=merged,
        preserved=len(_settings_in(existing)),
        # Not `_diff_settings`: that one reads `crs:` only, and a rating change
        # is entirely in `xmp:`, so the plan showed no difference at all.
        diff=_diff_keys(_settings_in(existing), _settings_in(merged)),
    )


def _diff_keys(before: dict[str, str], after: dict[str, str]) -> list[str]:
    lines = []
    for key in sorted(set(before) | set(after)):
        was, now = before.get(key), after.get(key)
        if was != now:
            lines.append(f"  {key}: {was or '-'} -> {now or '-'}")
    return lines


def _settings_in(document: str) -> dict[str, str]:
    import re

    return dict(re.findall(r'((?:crs|xmp|dc|photoshop):\w+)\s*=\s*"([^"]*)"', document))


@dataclass
class RatingPlan:
    """A rating write, described before it happens."""

    target: Path
    exists: bool
    merged: str
    preserved: int = 0
    diff: list[str] = None

    def __post_init__(self):
        if self.diff is None:
            self.diff = []

    def write(self, *, backup: bool = True) -> Path:
        """Only called after a person has said yes. Keeps a copy of what was there."""
        if self.exists and backup:
            spare = self.target.with_suffix(self.target.suffix + ".before-photoai")
            if not spare.exists():
                spare.write_bytes(self.target.read_bytes())
        self.target.write_text(self.merged, encoding="utf-8")
        return self.target


# Stars, by pile. Nothing is ever given one star or zero: zero means unrated in
# every catalogue, and one star is what many photographers use as "reject", a
# meaning this tool has not earned the right to assign.
STARS = {
    "TOP": 5,
    "GOOD_STOCK": 4,
    "GOOD_PERSONAL": 4,
    "NEEDS_DECISION": 3,
    "WEAK": 2,
}

# Lightroom matches colour labels by name against the catalogue's set, so these
# are words rather than colours. They are also somebody else's convention: red
# already means "reject" or "to print" or "client selected" in plenty of working
# catalogues, and a tool that overwrites that meaning is worse than one that
# leaves the labels alone.
#
# Override with PHOTO_AI_LABELS, as `TOP=Yellow,WEAK=` -- an empty value writes
# no label for that pile at all.
DEFAULT_LABELS = {
    "TOP": "Yellow",
    "GOOD_STOCK": "Green",
    "GOOD_PERSONAL": "Blue",
    "NEEDS_DECISION": "Purple",
    "WEAK": "Red",
}


def _configured_labels() -> dict[str, str]:
    import os

    labels = dict(DEFAULT_LABELS)
    raw = os.environ.get("PHOTO_AI_LABELS", "").strip()
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        pile, _, colour = pair.partition("=")
        pile = pile.strip().upper()
        if pile in labels:
            labels[pile] = colour.strip()
    return labels


class _Labels(dict):
    """Reads the environment on every lookup, so a test or a run can change it."""

    def get(self, key, default=""):
        return _configured_labels().get(key, default)

    def __getitem__(self, key):
        return _configured_labels()[key]

    def __contains__(self, key):
        return key in DEFAULT_LABELS

    def __iter__(self):
        return iter(DEFAULT_LABELS)


LABELS = _Labels(DEFAULT_LABELS)

RATING_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="photo-ai-toolkit">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:pat="https://github.com/karasov-co/photo-ai-toolkit/ns/1.0/"
    xmp:Rating="{rating}"
    xmp:Label="{label}"
    pat:category="{category}"
    pat:finalScore="{score}"
    pat:suggestedBy="photo-ai-toolkit"/>
 </rdf:RDF>
</x:xmpmeta>
"""


def to_rating_sidecar(record) -> str:
    """One photograph's pile as stars and a colour, for a catalogue to import."""
    category = (getattr(record, "category", "") or "").upper()
    return RATING_TEMPLATE.format(
        rating=STARS.get(category, 0),
        label=escape(LABELS.get(category, "")),
        category=escape(category),
        score=int(getattr(record, "final_score", 0) or 0),
    )
