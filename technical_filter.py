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

    report = TechnicalReport(
        sharpness_global=_laplacian_variance(gray),
        sharpness_tile=_max_tile_sharpness(gray),
        blur_ratio=_blur_ratio(work, gray),
        clipped_shadows=float((gray <= SHADOW_LEVEL).mean()),
        clipped_highlights=float((gray >= HIGHLIGHT_LEVEL).mean()),
        phash=str(imagehash.phash(work)),
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
    height, width = gray.shape
    best = 0.0
    for row in range(grid):
        for col in range(grid):
            tile = gray[
                row * height // grid : (row + 1) * height // grid,
                col * width // grid : (col + 1) * width // grid,
            ]
            if min(tile.shape) >= 3:
                best = max(best, _laplacian_variance(tile))
    return best


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
