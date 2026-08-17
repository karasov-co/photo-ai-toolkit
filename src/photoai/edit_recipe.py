"""What the frame could look like after a normal edit, found by trying.

The honest way to answer "how much better does this get" is to make it better
and measure. So this runs a small, bounded search: a handful of non-destructive
candidate edits on a downscaled copy, each scored by the same deterministic
quality function used for the unedited frame. The uplift is the difference.

Deliberate limits, each of which exists because the obvious version is wrong:

- **The originals are never touched.** Candidates are built in memory from a
  512px working copy. What gets stored is a *recipe* -- a list of instructions a
  human or Lightroom can apply -- not a modified pixel.

- **Uplift is penalised, not taken at face value.** An edit that crops away half
  the frame, pushes three stops, or leans on denoising has bought its
  improvement with something. Those costs come off the top, which is what stops
  "brighten everything" from being the universally optimal move.

- **A blocker caps the result.** Handled by the caller in `scoring.py`, but
  worth stating here: raising the exposure on a frame whose subject is out of
  focus produces a brighter out-of-focus frame. The search will happily report
  uplift for it, because exposure genuinely did improve; the cap is what stops
  that uplift from promoting the frame.

- **No generative editing.** Nothing here invents detail that is not in the
  file. That is both an honesty requirement for potential assessment and a
  marketplace one -- see `provenance.py`.

Everything is numpy and Pillow; no model, no network, no GPU. On a 512px
working image the whole search is a few milliseconds, which is what makes it
affordable to run on every frame before deciding which ones are worth paying a
vision model to look at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

WORK_PX = 512

# The quality function's anchors, fitted to a real 45-frame RAW sample rather
# than chosen. Measured on that sample:
#
#   laplacian variance   p10   135    p50   615    p90  1906    max 2778
#   luma stddev          p10    41    p50    55    p90    72    max   92
#
# The first version of this function used generous constants and produced
# min 51 / median 81 / max 91 on that sample -- the bottom half of the scale
# was never reached. That is the same collapse an absolute 1-1000 model score
# produces (548, 560, 694, 762 on live calls), and a scale that does not
# discriminate is not a scale. The constants below spread the same sample
# across roughly 25-90.
TARGET_MEAN = 118.0
EXPOSURE_SIGMA = 45.0

CONTRAST_FLOOR = 25.0
CONTRAST_SPAN = 55.0

DETAIL_SLOPE = 42.0
DETAIL_OFFSET = 59.5

# Search bounds. Anything outside these is not "a normal edit" any more.
MAX_EV = 2.0
MIN_CROP_KEEP = 0.45  # never propose keeping less than this fraction of area
MAX_STRAIGHTEN_DEG = 8.0

# Penalties, in quality points, applied against the raw uplift.
PENALTY_PER_EV = 3.0
PENALTY_PER_CROP_FRACTION = 26.0
PENALTY_DENOISE = 4.0
PENALTY_SHARPEN = 2.0
PENALTY_NO_RAW = 0.35  # multiplier on recovery-dependent uplift, not a subtraction


@dataclass
class EditStep:
    """One instruction, in the form a person would actually carry out."""

    action: str
    detail: str
    magnitude: float = 0.0

    def describe(self) -> str:
        return f"{self.action}: {self.detail}" if self.detail else self.action


@dataclass
class SearchResult:
    steps: list[EditStep] = field(default_factory=list)
    current_score: float = 0.0
    best_score: float = 0.0
    raw_uplift: float = 0.0
    penalties: float = 0.0
    crop_keep_fraction: float = 1.0
    ev_applied: float = 0.0
    uses_generative: bool = False
    candidate_name: str = "as shot"

    @property
    def uplift(self) -> float:
        """Realistic improvement, after the cost of getting it."""
        return max(0.0, self.raw_uplift - self.penalties)

    @property
    def is_noop(self) -> bool:
        return not self.steps

    def human_readable(self) -> list[str]:
        return [s.describe() for s in self.steps] or ["No edit needed; the frame is already balanced."]


# --- the deterministic quality function -------------------------------------


# Set to True only when `bench-quality` has been run against a labelled set and
# the correlation held. It is a claim about evidence, so it is not something the
# code may decide for itself.
#
# The circularity it flags: `frame_quality` is the objective the preview search
# hill-climbs on AND the ruler used to report the outcome. `uplift` is therefore
# the distance the search moved a number it was optimising, which is a weaker
# statement than "the photograph improved" -- and the two are not the same claim
# however plausible the first makes the second sound.
UPLIFT_VALIDATED = False


def frame_quality(rgb: np.ndarray) -> float:
    """A 0-100 technical read on one already-decoded frame.

    Four things, all of which a photographer would name looking at a histogram:
    is it exposed, does it have tonal range, is it clipping, does it hold
    detail. This is *not* an aesthetic judgement -- that comes from the vision
    model. It exists so the preview search has something objective to hill-climb
    on, and so `current_quality` and `post_edit_potential` are produced by the
    same function and are therefore comparable.
    """
    if rgb.size == 0:
        return 0.0
    luma = rgb @ np.array([0.299, 0.587, 0.114])

    mean = float(luma.mean())
    exposure = 100.0 * math.exp(-(((mean - TARGET_MEAN) / EXPOSURE_SIGMA) ** 2))

    stddev = float(luma.std())
    contrast = 100.0 * max(0.0, min(1.0, (stddev - CONTRAST_FLOOR) / CONTRAST_SPAN))

    detail = max(
        0.0,
        min(100.0, DETAIL_SLOPE * math.log10(1.0 + _laplacian_variance(luma)) - DETAIL_OFFSET),
    )

    # Clipping multiplies rather than adds. As an additive term it was almost
    # always 100 -- "nothing is blown" is the normal case -- so it acted as a
    # flat 16-point bonus on every frame and compressed the whole scale.
    clipped = float((luma >= 253).mean() + (luma <= 2).mean())
    clipping_factor = max(0.4, 1.0 - clipped * 3.0)

    base = 0.36 * exposure + 0.28 * contrast + 0.36 * detail
    return round(base * clipping_factor, 2)


def estimate_noise(rgb: np.ndarray) -> float:
    """Noise sigma, measured where the image is flat.

    Taking the residual against a median filter over the whole frame would
    measure edges, not grain -- every real photograph is full of legitimate
    high-frequency detail. Restricting to the flattest tiles is what makes the
    number mean "grain" rather than "texture".
    """
    if rgb.size == 0:
        return 0.0
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    if min(luma.shape) < 16:
        return 0.0
    smoothed = np.asarray(
        Image.fromarray(luma.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
        dtype=np.float64,
    )
    residual = luma - smoothed

    height, width = residual.shape
    tiles = []
    for row in range(4):
        for col in range(4):
            tile = residual[
                row * height // 4 : (row + 1) * height // 4,
                col * width // 4 : (col + 1) * width // 4,
            ]
            if tile.size:
                tiles.append(float(tile.std()))
    tiles.sort()
    quiet = tiles[: max(1, len(tiles) // 3)]
    return round(sum(quiet) / len(quiet), 3)


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


# --- candidate transforms ---------------------------------------------------


def _as_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def _clip8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255)


def apply_exposure(rgb: np.ndarray, ev: float) -> np.ndarray:
    """Linear-light exposure shift, the way a RAW converter does it."""
    linear = (rgb / 255.0) ** 2.2
    lifted = linear * (2.0**ev)
    return _clip8((np.clip(lifted, 0, 1) ** (1 / 2.2)) * 255.0)


def suggest_ev(rgb: np.ndarray) -> float:
    """How far off the target the frame currently sits, in stops."""
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    mean = max(float(luma.mean()), 1.0)
    ev = math.log2(TARGET_MEAN / mean)
    return max(-MAX_EV, min(MAX_EV, round(ev, 2)))


def apply_white_balance(rgb: np.ndarray) -> np.ndarray:
    """Grey-world neutralisation, damped.

    Full grey-world destroys a sunset -- the whole point of which is that it is
    orange. Damping to half the correction removes a genuine cast while leaving
    an intentional one recognisably intact.
    """
    means = rgb.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    gains = np.where(means > 1.0, target / np.maximum(means, 1.0), 1.0)
    damped = 1.0 + (gains - 1.0) * 0.5
    return _clip8(rgb * damped)


def apply_tone_recovery(rgb: np.ndarray, highlights: float = 0.5, shadows: float = 0.5) -> np.ndarray:
    """Pull the top down and lift the bottom, leaving the midtones alone."""
    x = rgb / 255.0
    high_mask = np.clip((x - 0.72) / 0.28, 0, 1)
    low_mask = np.clip((0.28 - x) / 0.28, 0, 1)
    recovered = x - high_mask * highlights * (x - 0.72) * 0.9 + low_mask * shadows * (0.28 - x) * 0.7
    return _clip8(np.clip(recovered, 0, 1) * 255.0)


def apply_contrast(rgb: np.ndarray, amount: float = 0.18) -> np.ndarray:
    x = rgb / 255.0
    curved = x + amount * (x - 0.5) * (1.0 - np.abs(x - 0.5) * 2.0) * 2.0
    return _clip8(np.clip(curved, 0, 1) * 255.0)


TILT_BLOCK = 24
TILT_MIN_COHERENCE = 0.55
# A scene needs enough oriented structure before its "horizon" means anything.
# Measured across generated scenes: compositions where the estimate was accurate
# had 25-30% of blocks coherently oriented, while the ones that returned a
# confidently wrong angle had 4-5%. Below this fraction the answer is no answer.
TILT_MIN_ORIENTED_FRACTION = 0.08


def estimate_tilt(rgb: np.ndarray) -> float:
    """Dominant edge deviation from level, in degrees.

    Uses a **structure tensor** over blocks rather than per-pixel gradient
    angles. The difference is not cosmetic: a three-point central difference
    spans one pixel, and an edge tilted four degrees only moves one pixel
    across fourteen, so per-pixel angles quantise every near-level edge to
    exactly zero. The first version of this function did that and returned
    0.5 degrees for every input from -3 to +6 -- it was measuring the pixel
    grid, not the photograph.

    Accumulating gx^2, gy^2 and gx*gy over a block first is what gives the
    angular resolution. Blocks whose structure is not clearly oriented
    (coherence below the threshold) are discarded rather than allowed to vote,
    since a patch of sky has a dominant direction only by accident.

    Real scenes are full of horizontals and verticals -- horizons, buildings,
    door frames -- so folding into +/-45 degrees puts a level camera at zero
    and a tilted one at the tilt.
    """
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    if min(luma.shape) < TILT_BLOCK * 2:
        return 0.0

    # A light blur spreads each edge over several pixels, which is what lets
    # the gradient direction carry sub-degree information at all.
    smoothed = np.asarray(
        Image.fromarray(np.clip(luma, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)),
        dtype=np.float64,
    )
    gy = smoothed[2:, 1:-1] - smoothed[:-2, 1:-1]
    gx = smoothed[1:-1, 2:] - smoothed[1:-1, :-2]

    height = (gx.shape[0] // TILT_BLOCK) * TILT_BLOCK
    width = (gx.shape[1] // TILT_BLOCK) * TILT_BLOCK
    if height < TILT_BLOCK or width < TILT_BLOCK:
        return 0.0

    def blocks(a: np.ndarray) -> np.ndarray:
        trimmed = a[:height, :width]
        return trimmed.reshape(
            height // TILT_BLOCK, TILT_BLOCK, width // TILT_BLOCK, TILT_BLOCK
        ).sum(axis=(1, 3))

    jxx = blocks(gx * gx)
    jyy = blocks(gy * gy)
    jxy = blocks(gx * gy)

    trace = jxx + jyy
    spread = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    coherence = np.divide(spread, trace, out=np.zeros_like(trace), where=trace > 1e-9)

    oriented = coherence >= TILT_MIN_COHERENCE
    if oriented.sum() < max(4, TILT_MIN_ORIENTED_FRACTION * oriented.size):
        # Not enough straight structure to have a horizon. Returning zero here
        # is a refusal, not a measurement of level: a scene of foliage or open
        # water has a dominant direction only by accident, and reporting it
        # produced confidently wrong angles on exactly those scenes.
        return 0.0

    angles = 0.5 * np.degrees(np.arctan2(2.0 * jxy[oriented], (jxx - jyy)[oriented]))
    folded = ((angles + 45.0) % 90.0) - 45.0
    weights = trace[oriented]

    hist, edges = np.histogram(folded, bins=180, range=(-45, 45), weights=weights)
    # Smoothed, so a peak split across two adjacent bins is not lost to a
    # neighbour that happens to be marginally taller.
    smoothed_hist = np.convolve(hist, np.ones(5), mode="same")
    peak_index = int(smoothed_hist.argmax())
    peak = (edges[peak_index] + edges[peak_index + 1]) / 2.0
    return round(float(peak), 2) if abs(peak) <= MAX_STRAIGHTEN_DEG else 0.0


def saliency_map(rgb: np.ndarray) -> np.ndarray:
    """Where the detail is. Blurred gradient energy, which is crude but stable."""
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    img = Image.fromarray(luma.astype(np.uint8))
    blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(3)), dtype=np.float64)
    gy = np.abs(np.diff(blurred, axis=0, prepend=blurred[:1]))
    gx = np.abs(np.diff(blurred, axis=1, prepend=blurred[:, :1]))
    energy = gx + gy
    smoothed = Image.fromarray(np.clip(energy, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(8)
    )
    return np.asarray(smoothed, dtype=np.float64)


def saliency_crops(rgb: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """A few crop boxes that keep the interesting part and drop dead edges."""
    height, width = rgb.shape[:2]
    energy = saliency_map(rgb)
    total = energy.sum()
    if total <= 0:
        return []

    rows = energy.sum(axis=1).cumsum() / total
    cols = energy.sum(axis=0).cumsum() / total

    def span(cumulative: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
        return int(np.searchsorted(cumulative, lo)), int(np.searchsorted(cumulative, hi))

    candidates: list[tuple[str, tuple[int, int, int, int]]] = []
    for name, (lo, hi) in (("tight crop", (0.10, 0.90)), ("gentle crop", (0.04, 0.96))):
        top, bottom = span(rows, lo, hi)
        left, right = span(cols, lo, hi)
        if bottom - top < 16 or right - left < 16:
            continue
        keep = ((bottom - top) * (right - left)) / (height * width)
        if keep >= MIN_CROP_KEEP:
            candidates.append((name, (left, top, right, bottom)))

    centroid_y = int(np.searchsorted(rows, 0.5))
    centroid_x = int(np.searchsorted(cols, 0.5))
    for name, ratio in (("4:5 crop", 0.8), ("1:1 crop", 1.0), ("16:9 crop", 16 / 9)):
        box = _box_for_ratio(width, height, centroid_x, centroid_y, ratio)
        if box:
            keep = ((box[3] - box[1]) * (box[2] - box[0])) / (height * width)
            if keep >= MIN_CROP_KEEP:
                candidates.append((name, box))
    return candidates


def _box_for_ratio(
    width: int, height: int, cx: int, cy: int, ratio: float
) -> tuple[int, int, int, int] | None:
    """Largest box of the given aspect ratio centred on the salient point."""
    box_w = min(width, int(height * ratio))
    box_h = min(height, int(box_w / ratio))
    box_w = min(width, int(box_h * ratio))
    if box_w < 16 or box_h < 16:
        return None
    left = max(0, min(width - box_w, cx - box_w // 2))
    top = max(0, min(height - box_h, cy - box_h // 2))
    return (left, top, left + box_w, top + box_h)


# --- the search -------------------------------------------------------------


def search(image: Image.Image, *, is_raw: bool = False, noisy: bool = False) -> SearchResult:
    """Try the plausible edits, keep the best, and charge it for what it cost."""
    work = image.copy()
    work.thumbnail((WORK_PX, WORK_PX), Image.LANCZOS)
    base = _as_array(work)

    current = frame_quality(base)
    recipe = SearchResult(current_score=current, best_score=current)

    ev = suggest_ev(base)
    tilt = estimate_tilt(base)

    candidates: list[tuple[str, np.ndarray, list[EditStep], float, float]] = []

    def offer(name: str, pixels: np.ndarray, steps: list[EditStep], keep: float = 1.0, ev_used: float = 0.0):
        candidates.append((name, pixels, steps, keep, ev_used))

    if abs(ev) >= 0.15:
        exposed = apply_exposure(base, ev)
        offer(
            "exposure",
            exposed,
            [EditStep("Adjust exposure", f"{ev:+.1f} EV", abs(ev))],
            ev_used=abs(ev),
        )

        toned = apply_tone_recovery(exposed)
        offer(
            "exposure + recovery",
            toned,
            [
                EditStep("Adjust exposure", f"{ev:+.1f} EV", abs(ev)),
                EditStep("Recover highlights and lift shadows", "moderate"),
            ],
            ev_used=abs(ev),
        )
    else:
        exposed = base

    balanced = apply_white_balance(exposed)
    offer(
        "exposure + white balance",
        balanced,
        _steps_for(ev, [EditStep("Correct white balance", "neutralise the cast, damped")]),
        ev_used=abs(ev),
    )

    full = apply_contrast(apply_tone_recovery(balanced))
    offer(
        "full tonal pass",
        full,
        _steps_for(
            ev,
            [
                EditStep("Correct white balance", "neutralise the cast, damped"),
                EditStep("Recover highlights and lift shadows", "moderate"),
                EditStep("Add contrast", "restrained S-curve"),
            ],
        ),
        ev_used=abs(ev),
    )

    for name, box in saliency_crops(base):
        cropped = full[box[1] : box[3], box[0] : box[2]]
        keep = cropped.size / max(base.size, 1)
        offer(
            f"full tonal pass + {name}",
            cropped,
            _steps_for(
                ev,
                [
                    EditStep("Correct white balance", "neutralise the cast, damped"),
                    EditStep("Recover highlights and lift shadows", "moderate"),
                    EditStep(
                        "Crop",
                        f"{name} -- keep {keep:.0%} of the frame "
                        f"(x {box[0]}-{box[2]}, y {box[1]}-{box[3]} of the working image)",
                        1.0 - keep,
                    ),
                ],
            ),
            keep=keep,
            ev_used=abs(ev),
        )

    if noisy:
        denoised = np.asarray(
            Image.fromarray(full.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
            dtype=np.float64,
        )
        offer(
            "full tonal pass + denoise",
            denoised,
            _steps_for(
                ev,
                [
                    EditStep("Correct white balance", "neutralise the cast, damped"),
                    EditStep("Apply luminance denoise", "restrained; do not smear detail"),
                ],
            ),
            ev_used=abs(ev),
        )

    best_name, best_pixels, best_steps, best_keep, best_ev = max(
        candidates,
        key=lambda c: frame_quality(c[1]) - _penalty(c[3], c[4], "denoise" in c[0]),
        default=("as shot", base, [], 1.0, 0.0),
    )

    best_score = frame_quality(best_pixels)
    penalties = _penalty(best_keep, best_ev, "denoise" in best_name)
    if not is_raw:
        # Highlight and shadow recovery on a JPEG is a guess: the data the
        # recovery would use has already been thrown away by the encoder.
        penalties += max(0.0, best_score - current) * PENALTY_NO_RAW

    steps = list(best_steps)
    if abs(tilt) >= 0.3:
        steps.append(
            EditStep(
                "Straighten",
                f"rotate {abs(tilt):.1f}° {'clockwise' if tilt < 0 else 'anticlockwise'}",
                abs(tilt),
            )
        )
    if noisy and "denoise" not in best_name:
        steps.append(EditStep("Apply luminance denoise", "restrained"))
    steps.append(EditStep("Do not apply aggressive sharpening", "halos cost more than they buy"))

    recipe.candidate_name = best_name
    recipe.steps = steps if best_score > current or abs(tilt) >= 0.3 else []
    recipe.best_score = best_score
    recipe.raw_uplift = max(0.0, best_score - current)
    recipe.penalties = round(penalties, 2)
    recipe.crop_keep_fraction = round(best_keep, 3)
    recipe.ev_applied = round(best_ev, 2)
    return recipe


def _steps_for(ev: float, rest: list[EditStep]) -> list[EditStep]:
    head = [EditStep("Adjust exposure", f"{ev:+.1f} EV", abs(ev))] if abs(ev) >= 0.15 else []
    return head + rest


def _penalty(keep_fraction: float, ev_used: float, denoised: bool) -> float:
    cost = PENALTY_PER_EV * max(0.0, ev_used - 0.5)
    cost += PENALTY_PER_CROP_FRACTION * (1.0 - keep_fraction)
    if denoised:
        cost += PENALTY_DENOISE
    return cost
