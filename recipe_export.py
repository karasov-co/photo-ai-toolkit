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

# Presets live in their own folder so "import everything in here" cannot pick up
# the sidecars by accident, which is how a panel fills with `<x:xmpmet` entries.
PRESETS_DIRNAME = "presets"
# Star ratings and colour labels, one sidecar per photograph, for every
# photograph in the run -- not only the ones that earned an edit recipe. This is
# the culling decision in the form a catalogue sorts by.
RATINGS_DIRNAME = "ratings"

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

    rated = write_ratings(records, out_dir / RATINGS_DIRNAME)

    removed = _clear_stale(out_dir, keep={Path(name).stem for name in written})
    if written or rated:
        write_instructions(out_dir, language=language)
    return {
        "written": written, "skipped": skipped, "removed": removed,
        "rated": rated, "dir": str(out_dir),
    }


def write_ratings(records, out_dir: Path) -> int:
    """A star rating and a colour label per photograph, as XMP.

    Every catalogue worth the name reads `xmp:Rating` and `xmp:Label` from a
    sidecar. Until this existed, the only way to act on a run was to read an
    HTML page beside the editor and click through a folder of symlinks, which is
    not how anybody culls.

    Written for every record, including the weak ones -- a two-star frame is a
    decision as much as a five-star one, and the pile is only useful as a sort
    if the whole shoot carries it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in records:
        if not record.category or record.status != "ok":
            continue
        stem = Path(record.filename).stem
        (out_dir / f"{stem}.xmp").write_text(
            adobe_xmp.to_rating_sidecar(record), encoding="utf-8"
        )
        count += 1
    return count


def _clear_stale(out_dir: Path, keep: set[str]) -> list[str]:
    """Drop sidecars for photographs this run did not choose.

    Scores move a little between runs, so without this the folder accumulates
    recipes for frames that no longer qualify -- and a photographer working
    through it would edit from a suggestion the current analysis has withdrawn.

    Only files this tool wrote are removed, identified by the marker inside
    them. Anything else in the folder is somebody's own work and is left alone.
    """
    removed: list[str] = []
    for path in out_dir.glob("*.xmp"):
        if path.stem in keep:
            continue
        try:
            if "photo-ai-toolkit" not in path.read_text(encoding="utf-8", errors="ignore"):
                continue
            path.unlink()
        except OSError as e:  # pragma: no cover - permissions
            logger.warning("Could not remove the stale recipe %s: %s", path.name, e)
            continue
        removed.append(path.name)
    return removed


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
    is_raw = _is_raw(source)
    as_shot_k = _as_shot_temperature(source)

    path = out_dir / f"{source.stem}.xmp"
    path.write_text(
        adobe_xmp.to_adobe_xmp(recipe, is_raw=is_raw, as_shot_temperature_k=as_shot_k),
        encoding="utf-8",
    )

    # And the same recipe as a preset. The sidecar only works beside a RAW; a
    # preset works on any file and is the only route this tool has to a JPEG at
    # all. It is also what a person expects when they are told "import this into
    # Lightroom" -- the sidecar, imported, shows up named after an XML tag with
    # an Amount slider that does nothing.
    presets = out_dir / PRESETS_DIRNAME
    presets.mkdir(parents=True, exist_ok=True)
    (presets / f"{source.stem}.xmp").write_text(
        adobe_xmp.to_lightroom_preset(
            recipe, stem=source.stem, is_raw=is_raw, as_shot_temperature_k=as_shot_k
        ),
        encoding="utf-8",
    )
    return path


def _as_shot_temperature(source: Path) -> int | None:
    """The camera's own colour temperature, when the file carries one.

    Needed only for RAW: Camera Raw's Temperature is absolute Kelvin there, so
    a measured delta cannot be written without it. Many cameras never record it
    -- Panasonic RW2 among them -- and returning None is the honest answer,
    which makes the exporter say what it could not write rather than writing a
    delta into an absolute field.
    """
    try:
        import exif_reader

        data = exif_reader.extract_exif(source, "RAW" if _is_raw(source) else "PHOTO") or {}
    except Exception:  # pragma: no cover - unreadable EXIF is not a crash
        return None
    for key in ("color_temperature", "as_shot_temperature_k", "white_balance_k"):
        value = data.get(key)
        if value:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return None


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

There are two files per photograph and they are not interchangeable.

  <name>.xmp            a Camera Raw sidecar. Works only when copied next to a
                        RAW. Do NOT import this as a preset -- it has no preset
                        fields, so Lightroom lists it as `<x:xmpmet` with an
                        Amount slider that does nothing.

  presets/<name>.xmp    a Lightroom preset. Works on any file, including JPEG,
                        and is the only route that works for a JPEG at all.

Lightroom (preset -- start here, and the only option for JPEG)
--------------------------------------------------------------
1. Develop module > Presets panel > the + menu > Import Presets.
2. Choose the file from `presets/`. It appears under "photo-ai-toolkit",
   named after the photograph.
3. Click it. The Amount slider dials the whole correction up and down.

Lightroom Classic (sidecar, RAW only)
-------------------------------------
1. Copy the .xmp file next to the original photograph, so that `PICTURE.xmp`
   sits beside `PICTURE.RW2`. Do this only for the frames you want; a sidecar
   you copy over an existing one replaces your own work.
2. In Lightroom, select those photographs in the Library module.
3. Metadata > Read Metadata from File. Confirm when asked.

The develop sliders will move to the suggested values. Undo works normally, and
History in the Develop module shows the change as a single step you can step
back from.

Adobe Camera Raw / Bridge
-------------------------
Copy the sidecar next to the original RAW and open the photograph. Camera Raw
reads `<name>.xmp` automatically.

Stars and colour labels
-----------------------
`ratings/<name>.xmp` carries the decision as a star rating and a colour label,
one file per photograph in the shoot:

  5 stars, yellow   top
  4 stars, green    good, and it has a market
  4 stars, blue     good, personal
  3 stars, purple   needs your decision
  2 stars, red      weak

Then sort by rating in the tool you already cull in. Nothing is ever rated one
star or zero: zero means "unrated" everywhere, and one star is what many
photographers use for "reject", which is not a decision this tool is entitled to
make for you.

Read them in one of two ways.

  Bridge or Photo Mechanic: point it at this folder. Nothing is copied and
  nothing of yours is touched.

  Lightroom Classic: the file has to sit beside the original as `<name>.xmp` --
  which is the same file your own develop settings, crop and keywords live in.
  Do not copy it over the top. `photoai apply-ratings` merges instead: it
  changes only the rating and the label, keeps a `.before-photoai` copy, and
  shows you the diff before it writes anything. Lightroom does not notice a
  changed sidecar by itself, so afterwards select the photographs and use
  Metadata > Read Metadata from File.

Two things this cannot do. It does not work for JPEG in Lightroom Classic at
all -- Lightroom keeps JPEG metadata in the catalogue and ignores a sidecar --
so for a JPEG shoot use Bridge, or read the ratings out of `analysis.json`. And
the colours are a guess at your convention: red already means "reject" or "to
print" in plenty of catalogues. Set `PHOTO_AI_LABELS` to change them, for
example `PHOTO_AI_LABELS=TOP=Orange,WEAK=` to leave weak frames unlabelled.

About white balance
-------------------
On a JPEG the correction is written to the Temperature slider directly. On a
RAW, Camera Raw wants an absolute Kelvin value rather than a shift, and most
cameras never record what they shot at -- when it cannot be read, the sidecar
says so in its warnings and gives you the number to move by hand. It is the
correction that matters most on a frame with a colour cast, so it is worth the
thirty seconds.

What these are not
------------------
Corrections, not a look. Exposure, white balance, highlight and shadow recovery,
a little contrast and clarity. No HSL, no tone curve, no split toning -- the
things a photographer's preset uses to make a frame *striking* are deliberately
absent, because a style applied to an unrelated photograph is a lie about the
picture. If you want a look, put one of your own on top of this.

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
