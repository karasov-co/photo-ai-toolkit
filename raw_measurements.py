"""Measuring the sensor data, not the picture of it.

This module exists to correct a specific mistake in the earlier pipeline: every
tonal decision was made from a 512px rendered preview. A preview is an already
developed, already clipped, already tone-mapped 8-bit JPEG. Asking it how much
highlight headroom the RAW holds is asking the wrong file. It will always answer
"none", because the renderer that made it already spent whatever was there.

What is actually available through LibRaw, and used here:

    black level      512 per channel  (on the sensor measured for this archive)
    white level    16319
    camera WB     [541, 256, 423]
    Bayer data    6008 x 4008 uint16

From that, the questions a recipe needs answered can be answered *properly*:

- **Highlight headroom.** How many stops sit between the brightest non-clipped
  sensor value and the point where the preview turned white. This is the number
  that decides whether `Highlights -40` recovers a sky or does nothing.
- **Per-channel clipping.** The red channel clips first on skin and sunsets, and
  the green last. A single luminance figure averages that away, and averaging it
  away is how a recipe ends up recommending a recovery that only shifts the hue.
- **Shadow floor.** Where the noise overtakes signal, which is what limits how
  far shadows can be lifted before they turn to colour mud.

Everything is measured on a decimated copy of the Bayer plane. Full resolution
would be exact and pointless: these are distribution statistics, and every one
of them is stable under 1-in-16 sampling.

**Limitation worth stating.** The headroom figure is a property of the sensor
data and of the specific rendering used for comparison. It says what is
*available*, not what any given converter will actually recover -- a different
demosaic and a different tone curve will reach different amounts of it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 1-in-N along each axis. 4 gives ~1.5M samples from a 24MP sensor, which is
# far more than any percentile needs.
DECIMATION = 4

# A channel is "clipped" within this fraction of the white level. Not exactly at
# it: sensors saturate slightly below their nominal maximum, and the last few
# levels are already non-linear.
CLIP_MARGIN = 0.995

# Shadows below this fraction of the white level are where read noise dominates.
# Lifting them is possible and looks like coloured mud.
NOISE_FLOOR_FRACTION = 0.004

# Bumped whenever the measurement changes meaning, so a cached figure from an
# older definition is visible rather than silently mixed with a newer one.
MEASUREMENT_VERSION = "1.0"


@dataclass
class ChannelStats:
    name: str
    black: int = 0
    white: int = 0
    p01: float = 0.0
    p50: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    maximum: float = 0.0
    clipped_fraction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RawMeasurements:
    """What the sensor actually holds. `available` is False for non-RAW input."""

    available: bool = False
    reason: str = ""

    black_level: int = 0
    white_level: int = 0
    camera_whitebalance: list[float] = field(default_factory=list)
    color_description: str = ""
    raw_width: int = 0
    raw_height: int = 0

    channels: list[ChannelStats] = field(default_factory=list)

    highlight_headroom_stops: float = 0.0
    shadow_headroom_stops: float = 0.0
    clipped_any_channel: float = 0.0
    clipped_all_channels: float = 0.0
    noise_floor_fraction: float = 0.0
    dynamic_range_stops: float = 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["channels"] = [c.to_dict() for c in self.channels]
        return payload

    @property
    def can_recover_highlights(self) -> bool:
        """Whether pulling the highlights will actually find anything."""
        return self.available and self.highlight_headroom_stops >= 0.25

    @property
    def can_lift_shadows(self) -> bool:
        return self.available and self.shadow_headroom_stops >= 0.5

    @property
    def truly_blown(self) -> float:
        """Fraction where *every* channel is saturated: genuinely no data left.

        The number that matters for "can this sky be recovered". A frame where
        only red is clipped still has luminance information in green and blue,
        and a converter can rebuild the highlight from it.
        """
        return self.clipped_all_channels


def measure(path: Path) -> RawMeasurements:
    """Read the sensor plane. Returns `available=False` for anything not RAW."""
    try:
        import rawpy
    except ImportError:  # pragma: no cover - rawpy is a hard dependency
        return RawMeasurements(reason="rawpy is not installed")

    try:
        with rawpy.imread(str(path)) as raw:
            return _measure_open(raw)
    except Exception as e:
        return RawMeasurements(reason=f"not a readable RAW file: {e}")


def _measure_open(raw) -> RawMeasurements:
    plane = raw.raw_image_visible.astype(np.float64)
    colors = raw.raw_colors_visible

    black_levels = list(raw.black_level_per_channel)
    black = int(np.mean([b for b in black_levels if b is not None] or [0]))
    white = int(raw.white_level or 0)
    if white <= black:
        return RawMeasurements(reason=f"implausible levels: black={black} white={white}")

    plane = plane[::DECIMATION, ::DECIMATION]
    colors = colors[::DECIMATION, ::DECIMATION]

    out = RawMeasurements(
        available=True,
        black_level=black,
        white_level=white,
        camera_whitebalance=[round(float(x), 2) for x in raw.camera_whitebalance],
        color_description=raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else str(raw.color_desc),
        raw_width=int(raw.sizes.raw_width),
        raw_height=int(raw.sizes.raw_height),
    )

    span = float(white - black)
    clip_level = black + span * CLIP_MARGIN
    names = out.color_description or "RGBG"

    saturated_masks = []
    for index in range(min(4, len(set(colors.ravel().tolist())) or 1)):
        values = plane[colors == index]
        if values.size == 0:
            continue
        channel_black = black_levels[index] if index < len(black_levels) else black
        stats = ChannelStats(
            name=names[index] if index < len(names) else str(index),
            black=int(channel_black),
            white=white,
            p01=round(float(np.percentile(values, 1)), 1),
            p50=round(float(np.percentile(values, 50)), 1),
            p99=round(float(np.percentile(values, 99)), 1),
            p999=round(float(np.percentile(values, 99.9)), 1),
            maximum=round(float(values.max()), 1),
            clipped_fraction=round(float((values >= clip_level).mean()), 5),
        )
        out.channels.append(stats)
        saturated_masks.append(values >= clip_level)

    if not out.channels:
        return RawMeasurements(reason="no colour channels could be sampled")

    # Any-channel and all-channel clipping are different questions. The first
    # says a hue has shifted; only the second says the information is gone.
    out.clipped_any_channel = round(max(c.clipped_fraction for c in out.channels), 5)
    out.clipped_all_channels = round(min(c.clipped_fraction for c in out.channels), 5)

    # Headroom: distance in stops from where the *preview* would have turned
    # white (the 99th percentile of the scene) up to real saturation.
    reference = max(np.mean([c.p99 for c in out.channels]) - black, 1.0)
    ceiling = max(span * CLIP_MARGIN, 1.0)
    out.highlight_headroom_stops = round(max(0.0, math.log2(ceiling / reference)), 2)

    floor = max(span * NOISE_FLOOR_FRACTION, 1.0)
    darkest = max(np.mean([c.p01 for c in out.channels]) - black, 0.5)
    out.shadow_headroom_stops = round(max(0.0, math.log2(max(darkest, 0.5) / floor)), 2)
    out.noise_floor_fraction = round(NOISE_FLOOR_FRACTION, 5)
    out.dynamic_range_stops = round(math.log2(ceiling / floor), 2)
    return out


def measure_or_empty(path: Path, is_raw: bool) -> RawMeasurements:
    """Measure a RAW; for anything else say so rather than guessing.

    A JPEG's headroom is not zero because it is small -- it is unknown, and the
    difference matters. A recipe built on "unknown" hedges; one built on "zero"
    confidently refuses to recover a highlight that is sitting right there.
    """
    if not is_raw:
        return RawMeasurements(reason="not a RAW file: headroom is unknown, not zero")
    return measure(path)


def summarise(measurements: RawMeasurements) -> list[str]:
    """The evidence lines that go into a recipe."""
    if not measurements.available:
        return [f"RAW measurements unavailable ({measurements.reason})"]

    lines = [
        f"{measurements.highlight_headroom_stops:.2f} stops of highlight headroom above "
        f"the rendered white point",
        f"{measurements.dynamic_range_stops:.1f} stops between saturation and the noise floor",
    ]
    worst = max(measurements.channels, key=lambda c: c.clipped_fraction, default=None)
    if worst and worst.clipped_fraction > 0.001:
        lines.append(
            f"{worst.name} clips first at {worst.clipped_fraction:.2%}; "
            f"all channels saturated on {measurements.clipped_all_channels:.2%} of the frame"
        )
    if measurements.clipped_all_channels > 0.02:
        lines.append(
            f"{measurements.clipped_all_channels:.1%} of the frame has no data in any "
            "channel and cannot be recovered by any converter"
        )
    return lines
