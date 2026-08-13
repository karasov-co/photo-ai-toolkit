"""Per-image edit recipes, written where Lightroom can pick them up.

One preset does not fit a collection. The whole premise of the analysis is that
these frames differ -- one is two stops under, one has a cast, one is tilted
three degrees and one is fine -- so shipping a single "look" would throw away
every measurement that made the report worth reading. Each photograph gets its
own sidecar, built from its own numbers.

**Nothing here calls a model.** Every adjustment comes from measurements the run
already made and from the Stage 2/Stage 3 results already on the record. A
recipe costs no tokens and no network.

**Nothing here touches an original.** Adobe reads development settings from a
sidecar and leaves the file alone, which is the only reason a suggestion can be
delivered as a file at all. But a converter looks for `<stem>.xmp`, so writing
there would silently replace work the photographer has already done. These go to
`edit_recipes/<stem>.xmp` in the output directory -- next to nothing, importable
on purpose, and incapable of overwriting anything in the source folder.

**A starting point, not an edit.** Said in the folder's own README, in the file's
own metadata, and in the report. The numbers land in the sliders a photographer
would have moved; whether they are the right numbers for this picture is a
judgement the tool does not have.

The creative directions are deliberately few and conditional. A season is not a
style you can apply to any photograph: "warm autumn" on a blue winter landscape
is a lie about the picture. So each direction has to be earned by something
measured -- the palette, the tonal distribution, the genre, the documentary
weight -- and a frame that earns none is offered none.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import raw_measurements
import recipe_generator
from edit_schema import Variant
from exporters import adobe_xmp

logger = logging.getLogger(__name__)

# Below this, an edit recipe is not worth a photographer's attention -- and
# offering one on a frame the tool has just called weak reads as a suggestion to
# spend an evening on it.
MIN_POTENTIAL = 70

README_NAME = "HOW-TO-USE.txt"


@dataclass
class Direction:
    """One optional creative reading, and the evidence that earned it."""

    key: str
    label: str
    because: str

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "because": self.because}


# --- the creative directions --------------------------------------------------
#
# Five, fixed. Not a style library: each is a direction a picture editor would
# actually suggest, and each is gated on something the analysis measured.

WARM_CAST = 1.06          # red mean over blue mean, on the developed preview
COOL_CAST = 0.95
LOW_KEY_LUMA = 92.0       # 0-255; below this the frame is already dark
FLAT_COLOUR_STDDEV = 26.0
STRONG_DOCUMENTARY = 70


def directions_for(record, measurement) -> list[Direction]:
    """Up to three, only where the frame supports them."""
    out: list[Direction] = []
    stage3 = record.stage3 or {}
    genre = (record.genre or "").lower()
    channel_means = getattr(measurement, "channel_means", None) or (0.0, 0.0, 0.0)
    red, _green, blue = channel_means
    warmth = red / blue if blue else 1.0
    luma = float(getattr(measurement, "mean_luma", 0.0) or 0.0)

    documentary = int(stage3.get("documentary_significance") or 0)
    if documentary >= STRONG_DOCUMENTARY or genre in ("reportage", "street", "architecture"):
        out.append(
            Direction(
                "documentary_neutral",
                "Documentary neutral",
                f"documentary weight {documentary} and a {genre or 'reportage'} frame: "
                "correct the exposure and the cast, and leave the rest alone",
            )
        )

    if luma and luma <= LOW_KEY_LUMA:
        out.append(
            Direction(
                "cinematic_low_key",
                "Cinematic low key",
                f"the frame is already dark (mean luminance {luma:.0f} of 255): "
                "hold the shadows down instead of lifting them flat",
            )
        )

    if warmth >= WARM_CAST and genre in ("landscape", "street", "detail", "reportage"):
        out.append(
            Direction(
                "warm_autumn",
                "Warm autumn atmosphere",
                f"the palette is already warm (red/blue {warmth:.2f}): "
                "lean into it rather than neutralising it",
            )
        )
    elif warmth <= COOL_CAST and genre in ("landscape", "architecture", "street"):
        out.append(
            Direction(
                "cool_winter",
                "Cool winter atmosphere",
                f"the palette is already cool (red/blue {warmth:.2f}): "
                "keep the blue rather than correcting it away",
            )
        )

    stddev = float(getattr(measurement, "stddev_luma", 0.0) or 0.0)
    colour_is_weak = warmth and COOL_CAST < warmth < WARM_CAST
    if colour_is_weak and stddev >= FLAT_COLOUR_STDDEV and len(out) < 3:
        out.append(
            Direction(
                "restrained_bw",
                "Restrained black and white",
                "the colour is nearly neutral and the tonal range is wide: "
                "the picture may be about light rather than colour",
            )
        )

    return out[:3]


# --- writing ------------------------------------------------------------------


def should_export(record) -> bool:
    """Worth an edit recipe. Weak frames are deliberately excluded."""
    return int(record.final_score or 0) >= MIN_POTENTIAL and record.category != "WEAK"


def export_all(records, measurements: dict, out_dir: Path, *, language: str = "en") -> dict:
    """Write one sidecar per qualifying photograph. Returns what was written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []
    for record in records:
        if not should_export(record):
            skipped.append(record.filename)
            continue
        measurement = measurements.get(record.asset_key)
        if measurement is None:
            skipped.append(record.filename)
            continue
        try:
            path = export_one(record, measurement, out_dir)
        except Exception as e:  # pragma: no cover - a malformed record, not a crash
            logger.warning("Could not write a recipe for %s: %s", record.filename, e)
            skipped.append(record.filename)
            continue
        record.recipe_path = str(path)
        record.creative_directions = [d.to_dict() for d in directions_for(record, measurement)]
        written.append(record.filename)

    if written:
        write_instructions(out_dir, language=language)
    return {"written": written, "skipped": skipped, "dir": str(out_dir)}


def export_one(record, measurement, out_dir: Path) -> Path:
    """One photograph's sidecar, named after the photograph.

    `<stem>.xmp` inside `edit_recipes/` is what Lightroom's importer expects to
    find beside a file, so a photographer can copy it next to the original when
    they have decided to -- a decision the tool deliberately leaves to them.
    """
    source = Path(record.source_path)
    raw_stats = raw_measurements.measure_or_empty(source, _is_raw(source))

    recipes = recipe_generator.generate(
        asset_id=record.asset_id,
        asset_key=record.asset_key,
        checksum=record.checksum,
        raw_stats=raw_stats,
        mean_luma=float(getattr(measurement, "mean_luma", 0.0) or 0.0),
        stddev_luma=float(getattr(measurement, "stddev_luma", 0.0) or 0.0),
        channel_means=getattr(measurement, "channel_means", None) or (0.0, 0.0, 0.0),
        noise=float(getattr(measurement, "noise", 0.0) or 0.0),
        tilt_degrees=_tilt_of(measurement),
        intent_signals=[],
        is_raw=_is_raw(source),
    )
    recipe = _faithful(recipes)
    path = out_dir / f"{source.stem}.xmp"
    path.write_text(adobe_xmp.to_adobe_xmp(recipe), encoding="utf-8")
    return path


def _faithful(recipes: list):
    """The corrective reading, not the expressive one.

    A sidecar is a starting point, and a starting point that has already made
    stylistic choices is harder to work from than one that has only undone the
    camera's mistakes.
    """
    for recipe in recipes:
        if recipe.variant == Variant.FAITHFUL.value:
            return recipe
    return recipes[0]


def _is_raw(path: Path) -> bool:
    import media

    return path.suffix.lower() in media.RAW_EXTENSIONS


def _tilt_of(measurement) -> float:
    """The straightening angle the analysis already found, in degrees."""
    import re

    for step in getattr(measurement, "recipe", None) or []:
        if "Straighten" in str(step):
            found = re.search(r"(-?\d+(?:\.\d+)?)", str(step))
            if found:
                return float(found.group(1))
    return 0.0


INSTRUCTIONS = """\
Edit recipes
============

One .xmp sidecar per photograph, built from that photograph's own measurements.
These are a STARTING POINT, not a finished edit: they undo what the camera got
wrong (exposure, colour cast, a tilted horizon, clipped highlights) and stop
there. Every creative decision is still yours.

Nothing in this folder has touched your originals, and nothing here will.

Lightroom Classic
-----------------
1. Copy the .xmp file next to the original photograph, so that `PICTURE.xmp`
   sits beside `PICTURE.RW2` / `PICTURE.JPG`. Do this only for the frames you
   want; a sidecar you copy over an existing one replaces your own work.
2. In Lightroom, select those photographs in the Library module.
3. Metadata > Read Metadata from File. Confirm when asked.

The develop sliders will move to the suggested values. Undo works normally, and
History in the Develop module shows the change as a single step you can step
back from.

Adobe Camera Raw / Bridge
-------------------------
Copy the sidecar next to the original and open the photograph. Camera Raw reads
`<name>.xmp` automatically.

Capture One, darktable, RawTherapee
-----------------------------------
These do not read Camera Raw sidecars. Open .internal/reports/analysis.json and
read `edit_recipe` for each photograph -- the same adjustments in plain words.

If something looks wrong
------------------------
Delete the sidecar and nothing has happened. The suggestion never modified your
photograph; it only ever sat in a separate file.
"""


def write_instructions(out_dir: Path, *, language: str = "en") -> Path:
    path = Path(out_dir) / README_NAME
    path.write_text(INSTRUCTIONS, encoding="utf-8")
    return path
