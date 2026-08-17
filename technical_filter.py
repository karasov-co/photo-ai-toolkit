"""Stage 0: deterministic, local, zero-cost triage.

Everything here is measured, not judged -- no model is involved. The point is to
keep obviously unusable frames and near-duplicate burst siblings out of the
paid vision stage, which is where the money goes.

Signals:

- **Focus, via blur ratio.** How much detail the frame loses when blurred
  again. This is the field to threshold on, and getting there took two wrong
  turns worth recording:

  *Global* Laplacian variance rejects shallow depth of field -- a portrait is
  mostly out-of-focus by area, so it scores like a blurred frame:

      frame                       global   max-tile
      sharp                        1300      3833
      fully blurred                   2         2
      sharp subject, blurred bg     166      1401

  Switching to the best 4x4 tile fixes that. But *absolute* sharpness, tiled or
  not, still cannot tell a soft scene from a soft lens. On this archive it
  rejected three good photographs -- fog over terraces and two hazy sunsets --
  because a foggy scene has no high-frequency content to begin with:

      frame                     max-tile   blur ratio
      fog over terraces               44        13.97   <- good
      hazy sunset                     36         7.71   <- good
      hazy sunset                     21         5.56   <- good
      ordinary sharp frame          3279       254.75
      same frame, defocused            2         1.08   <- actually unusable

  Blur ratio compares the frame against itself, so scene contrast cancels out.
  Absolute sharpness is still reported, as a diagnostic only.

- **Clipping.** Fraction of pixels crushed to black or blown to white. The
  highlight threshold is deliberately loose: a bright overcast sky legitimately
  clips a quarter of the frame.

- **Perceptual hash.** For collapsing bursts down to one frame.

Note on resolution: these run on a downscaled working image, so they answer
"is this frame usable at all", not "is it tack sharp at 100%". Missed focus
that only shows when pixel-peeping will not be caught here, by design -- Stage 0
is a garbage filter, not a focus checker.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Working size for the measurements. 512 is what the vision preview already
# uses, so the frame is decoded once and serves both.
WORK_MAX_PX = 512
TILE_GRID = 4

# Rejection runs on blur_ratio, not on raw sharpness. Measured on this archive:
#
#   fog over terraces        tile   44   ratio  13.97   <- good photograph
#   hazy sunset              tile   36   ratio   7.71   <- good photograph
#   hazy sunset              tile   21   ratio   5.56   <- good photograph
#   ordinary sharp frame     tile 3279   ratio 254.75
#   same frame, defocused    tile    2   ratio   1.08   <- actually unusable
#
# Absolute sharpness cannot tell the first three from the last: a foggy scene
# has no high-frequency content because *the scene* has none, not because the
# lens missed. Thresholding on tile sharpness rejected all three sunsets.
MIN_BLUR_RATIO = 2.0

# Highlights: a bright overcast sky legitimately clips a quarter of the frame,
# so this only catches a frame that is mostly pure white.
MAX_CLIPPED_HIGHLIGHTS = 0.55
MAX_CLIPPED_SHADOWS = 0.75

# Kept as a diagnostic in the report, deliberately not used for rejection.
LOW_SHARPNESS_HINT = 60.0

SHADOW_LEVEL = 2
HIGHLIGHT_LEVEL = 253

# Two frames are burst siblings if their hashes are within this Hamming
# distance AND they were shot within BURST_WINDOW_SECONDS of each other.
PHASH_DISTANCE = 8
BURST_WINDOW_SECONDS = 3.0


@dataclass
class TechnicalReport:
    sharpness_global: float
    sharpness_tile: float
    blur_ratio: float
    clipped_shadows: float
    clipped_highlights: float
    phash: str
    rejected_for: list[str] = field(default_factory=list)
    # Which tile on the preview grid was the sharpest, as (row, col). Kept so
    # the same region can be looked at again at native resolution, which is the
    # only place a focus miss is visible.
    tile_location: tuple[int, int] | None = None

    @property
    def passed(self) -> bool:
        return not self.rejected_for

    @property
    def looks_soft(self) -> bool:
        """Low absolute detail. Often just fog or a night sky -- a hint, not a verdict."""
        return self.sharpness_tile < LOW_SHARPNESS_HINT


def analyze(image: Image.Image) -> TechnicalReport:
    """Measure one already-decoded frame."""
    work = image.copy()
    work.thumbnail((WORK_MAX_PX, WORK_MAX_PX), Image.LANCZOS)
    gray = np.asarray(work.convert("L"), dtype=np.float64)

    tile_sharpness, tile_location = _sharpest_tile(gray)
    report = TechnicalReport(
        sharpness_global=_laplacian_variance(gray),
        sharpness_tile=tile_sharpness,
        blur_ratio=_blur_ratio(work, gray),
        clipped_shadows=float((gray <= SHADOW_LEVEL).mean()),
        clipped_highlights=float((gray >= HIGHLIGHT_LEVEL).mean()),
        phash=str(imagehash.phash(work)),
        tile_location=tile_location,
    )
    report.rejected_for = _reasons_to_reject(report)
    return report


def _blur_ratio(work: Image.Image, gray: np.ndarray) -> float:
    """How much detail the frame loses when blurred again.

    A sharp frame collapses (ratio in the tens or hundreds); a frame that was
    already out of focus barely moves (ratio near 1). Because it compares the
    frame against itself, it is independent of how contrasty the scene is --
    which is what absolute sharpness gets wrong on fog, haze and night.
    """
    blurred = np.asarray(work.filter(ImageFilter.GaussianBlur(2)).convert("L"), dtype=np.float64)
    return _laplacian_variance(gray) / max(_laplacian_variance(blurred), 1e-6)


def _reasons_to_reject(report: TechnicalReport) -> list[str]:
    reasons = []
    if report.blur_ratio < MIN_BLUR_RATIO:
        reasons.append(f"out of focus (blur ratio {report.blur_ratio:.2f} < {MIN_BLUR_RATIO:.2f})")
    if report.clipped_highlights > MAX_CLIPPED_HIGHLIGHTS:
        reasons.append(f"blown highlights ({report.clipped_highlights:.0%})")
    if report.clipped_shadows > MAX_CLIPPED_SHADOWS:
        reasons.append(f"crushed shadows ({report.clipped_shadows:.0%})")
    return reasons


def _laplacian_variance(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    lap = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(lap.var())


def _max_tile_sharpness(gray: np.ndarray, grid: int = TILE_GRID) -> float:
    """Sharpness of the sharpest region, so shallow depth of field survives."""
    return _sharpest_tile(gray, grid)[0]


def _sharpest_tile(gray: np.ndarray, grid: int = TILE_GRID) -> tuple[float, tuple[int, int]]:
    """The best tile's sharpness and which tile it was, as (row, col).

    The location is what makes a full-resolution recheck possible: it says where
    to look at 100% without decoding the whole frame.
    """
    height, width = gray.shape
    best, where = 0.0, (0, 0)
    for row in range(grid):
        for col in range(grid):
            tile = gray[
                row * height // grid : (row + 1) * height // grid,
                col * width // grid : (col + 1) * width // grid,
            ]
            if min(tile.shape) >= 3:
                value = _laplacian_variance(tile)
                if value > best:
                    best, where = value, (row, col)
    return best, where


# --- looking at 100%, which is where a focus miss lives ------------------------
#
# Everything above runs on a 512px preview. That is fast and it is enough for
# exposure, clipping and a frame that is obviously soft. It is not enough for
# focus: a portrait focused on an ear instead of an eye, or a frame back-focused
# by ten centimetres, is smooth at 512px and wrong at 100%. Professional culling
# is mostly this question, and answering it on a preview is answering a
# different one.
#
# So the sharpest tile is measured a second time, at native resolution, on that
# tile alone. One crop per frame rather than a full decode: the cost is a
# fraction of the decode that already happened, and the number it returns is
# about the pixels a person would zoom into.

# The measure is the same blur ratio used above, computed on the native crop:
# how much detail the region loses when blurred again. It is scale-free, which
# matters -- comparing Laplacian variance between a 512px preview and a 768px
# native crop compares two different pixel scales and calls every large frame
# soft, which is what the first version of this did.
#
# It is NOT noise-free, and that decides the threshold. Noise is high-frequency
# energy, so grain raises the ratio: a defocused frame at ISO 6400 can clear a
# floor that a defocused frame at ISO 200 would fail. Measured over 200 frames
# from one archive, all of them in focus:
#
#     ISO <= 200    n=146   min 11.90   p10 14.84   median  56.26   max 127.25
#     ISO <= 800    n= 15   min 14.56   p10 18.50   median  44.13   max  79.31
#     ISO > 3200    n= 39   min 12.90   p10 20.61   median  42.17   max  78.99
#
# Two things follow. The passing population bottoms out near 12, so any floor
# under ~10 never rejects a frame this archive would have kept. And the p10
# rises with ISO rather than falling, which is the grain -- so the floor has to
# rise with it, or a soft high-ISO frame passes on noise alone.
#
# Honest limit: these are 200 in-focus frames from one camera. There are no
# defocused high-ISO samples in the set, so the slope below is chosen to track
# the measured p10 rather than fitted to a separation nobody has measured.
FOCUS_BASE_RATIO = 4.0
FOCUS_RATIO_PER_ISO_STOP = 0.9
FOCUS_ISO_BASELINE = 200


def focus_floor(iso: int | None) -> float:
    """The blur ratio a sharp region has to clear, given the film speed."""
    if not iso or iso <= FOCUS_ISO_BASELINE:
        return FOCUS_BASE_RATIO
    import math

    stops = math.log2(iso / FOCUS_ISO_BASELINE)
    return round(FOCUS_BASE_RATIO + FOCUS_RATIO_PER_ISO_STOP * stops, 2)


# Kept for the callers that have no ISO to hand. Same value as the base.
FOCUS_CONFIRM_RATIO = FOCUS_BASE_RATIO

# The crop taken at native resolution, in pixels. Large enough to hold an eye
# and its lashes at 24MP, small enough that reading it costs nothing.
FOCUS_CROP_PX = 768


@dataclass
class FocusCheck:
    """What the sharpest region looks like at 100%, rather than at 512px."""

    checked: bool = False
    preview_sharpness: float = 0.0
    full_sharpness: float = 0.0
    ratio: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    note: str = ""
    # The floor this frame had to clear, which rises with ISO. Stored so a
    # person reading the JSON can see what the verdict was measured against.
    floor: float = FOCUS_BASE_RATIO

    @property
    def confirmed(self) -> bool:
        """Whether the sharpest region really is sharp at 100%."""
        return self.checked and self.ratio >= self.floor

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "preview_sharpness": round(self.preview_sharpness, 2),
            "full_sharpness": round(self.full_sharpness, 2),
            "ratio": round(self.ratio, 4),
            "confirmed": self.confirmed,
            "floor": self.floor,
            "region": list(self.region),
            "note": self.note,
        }


def confirm_focus(
    path: Path, report: TechnicalReport, *, grid: int = TILE_GRID, iso: int | None = None
) -> FocusCheck:
    """Re-measure the sharpest region at native resolution.

    Returns a `FocusCheck` rather than a verdict. A soft result is a fact about
    the frame, and what to do about it belongs to the caller -- this module has
    thrown away good photographs before by deciding on its own.
    """
    check = FocusCheck(preview_sharpness=report.sharpness_tile, floor=focus_floor(iso))
    if not report.tile_location or report.sharpness_tile <= 0:
        check.note = "no sharp region was found on the preview to re-check"
        return check

    try:
        with Image.open(path) as opened:
            opened.draft("L", opened.size)  # no-op for most, a hint for JPEG
            width, height = opened.size
            row, col = report.tile_location
            centre_x = int((col + 0.5) * width / grid)
            centre_y = int((row + 0.5) * height / grid)
            half = FOCUS_CROP_PX // 2
            box = (
                max(0, min(width - FOCUS_CROP_PX, centre_x - half)),
                max(0, min(height - FOCUS_CROP_PX, centre_y - half)),
                0, 0,
            )
            box = (box[0], box[1], min(width, box[0] + FOCUS_CROP_PX),
                   min(height, box[1] + FOCUS_CROP_PX))
            if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                check.note = "the frame is too small for a full-resolution crop to mean anything"
                return check
            crop = opened.convert("L").crop(box)
    except Exception as e:
        check.note = f"could not re-read the frame at full size: {e}"
        return check

    check.checked = True
    check.region = box
    gray = np.asarray(crop, dtype=np.float64)
    check.full_sharpness = _laplacian_variance(gray)
    check.ratio = _blur_ratio(crop, gray)
    if not check.confirmed:
        check.note = (
            "the sharpest region is softer at 100% than the preview suggested: "
            "the focus may have landed somewhere else"
        )
    return check


# --- burst collapsing -------------------------------------------------------


def group_bursts(frames: list[dict]) -> list[list[dict]]:
    """Group frames shot back-to-back that also look alike.

    Each frame needs `phash` and `date_shot` (ISO 8601, or None). Frames with no
    timestamp are never grouped -- without one there is no way to tell a burst
    from two similar photographs taken months apart.

    Returns groups in input order; a frame that belongs to no burst comes back
    as a group of one.
    """
    ordered = sorted(
        (f for f in frames if f.get("date_shot")),
        key=lambda f: f["date_shot"],
    )
    undated = [f for f in frames if not f.get("date_shot")]

    groups: list[list[dict]] = []
    for frame in ordered:
        if groups and _same_burst(groups[-1][-1], frame):
            groups[-1].append(frame)
        else:
            groups.append([frame])

    groups.extend([f] for f in undated)
    return groups


def _same_burst(previous: dict, current: dict) -> bool:
    gap = _seconds_between(previous.get("date_shot"), current.get("date_shot"))
    if gap is None or gap > BURST_WINDOW_SECONDS:
        return False
    return _hamming(previous.get("phash"), current.get("phash")) <= PHASH_DISTANCE


def pick_from_burst(group: list[dict]) -> dict:
    """The sharpest frame in a burst, by best tile."""
    return max(group, key=lambda f: f.get("sharpness_tile") or 0.0)


def _seconds_between(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    try:
        return abs((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())
    except (ValueError, TypeError):
        return None


def _hamming(a: str | None, b: str | None) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 999


def load_for_analysis(path: Path, file_type: str) -> Image.Image:
    """Decode a frame for measurement. RAW goes through rawpy, the rest Pillow."""
    if file_type == "RAW":
        import rawpy

        with rawpy.imread(str(path)) as raw:
            return Image.fromarray(raw.postprocess(use_camera_wb=True, output_bps=8))
    with Image.open(path) as img:
        return img.convert("RGB")
