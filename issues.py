"""Everything wrong with a file, sorted by whether it can be undone.

This is the module the whole "potential, not current state" idea rests on. An
underexposed frame and an out-of-focus frame both look bad; one is a slider and
the other is gone forever, and no aggregate score can express that difference
after the fact. So the difference is recorded first, as a type, and the scores
are derived from it.

Three degrees:

    FIXABLE         normal editing removes it entirely
    PARTIAL         editing improves it at a cost -- crop, resolution, artifacts
    UNRECOVERABLE   the information is not in the file

The asymmetry that matters: a single UNRECOVERABLE issue caps post-edit
potential no matter how much uplift the preview search finds. Otherwise raising
the exposure on a frame whose subject is not in focus produces a brighter
photograph that is still not in focus, and a naive uplift calculation promotes
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Fixability(StrEnum):
    FIXABLE = "fixable"
    PARTIAL = "partially_fixable"
    UNRECOVERABLE = "unrecoverable"


class IssueCode(StrEnum):
    # --- fixable ---
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    FLAT_CONTRAST = "flat_contrast"
    COLOR_CAST = "color_cast"
    TILTED_HORIZON = "tilted_horizon"
    PERSPECTIVE_DISTORTION = "perspective_distortion"
    WEAK_CROP = "weak_crop"
    EDGE_CLUTTER = "edge_clutter"
    MILD_NOISE = "mild_noise"
    CHROMATIC_ABERRATION = "chromatic_aberration"
    UNUSABLE_AUDIO = "unusable_audio"
    NEEDS_TRIM = "needs_trim"

    # --- partially fixable ---
    HEAVY_NOISE = "heavy_noise"
    SOME_CLIPPED_HIGHLIGHTS = "some_clipped_highlights"
    SOME_CRUSHED_SHADOWS = "some_crushed_shadows"
    MODERATE_SHAKE = "moderate_shake"
    BACKGROUND_CLUTTER = "background_clutter"
    SOFT_FOCUS = "soft_focus"
    OVERSHARPENED = "oversharpened"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    EXPOSURE_DRIFT = "exposure_drift"
    ROLLING_SHUTTER = "rolling_shutter"
    # A near-identical sibling exists. Not damage: two takes of one moment, and
    # which is better depends on a gesture or an expression that no local
    # measurement can see.
    WEAKER_DUPLICATE = "weaker_duplicate"
    # Shorter than most marketplaces accept. A three-second minimum is a
    # submission rule, not a property of the footage.
    SHORT_CLIP = "short_clip"

    # --- unrecoverable ---
    MISSED_FOCUS = "missed_focus"
    SEVERE_MOTION_BLUR = "severe_motion_blur"
    BLOWN_HIGHLIGHTS = "blown_highlights"
    CRUSHED_SHADOWS = "crushed_shadows"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    SEVERE_ARTIFACTS = "severe_artifacts"
    CORRUPT_FILE = "corrupt_file"
    EMPTY_FRAME = "empty_frame"
    DEAD_MOMENT = "dead_moment"
    OBSTRUCTED_SUBJECT = "obstructed_subject"
    LEGAL_BLOCKER = "legal_blocker"
    NO_USABLE_SEGMENT = "no_usable_segment"
    UNUSABLE_SHAKE = "unusable_shake"
    BROKEN_FOCUS_PULL = "broken_focus_pull"
    ENCODING_CORRUPTION = "encoding_corruption"
    UNUSABLE_DURATION = "unusable_duration"


FIXABILITY: dict[IssueCode, Fixability] = {
    IssueCode.UNDEREXPOSED: Fixability.FIXABLE,
    IssueCode.OVEREXPOSED: Fixability.FIXABLE,
    IssueCode.FLAT_CONTRAST: Fixability.FIXABLE,
    IssueCode.COLOR_CAST: Fixability.FIXABLE,
    IssueCode.TILTED_HORIZON: Fixability.FIXABLE,
    IssueCode.PERSPECTIVE_DISTORTION: Fixability.FIXABLE,
    IssueCode.WEAK_CROP: Fixability.FIXABLE,
    IssueCode.EDGE_CLUTTER: Fixability.FIXABLE,
    IssueCode.MILD_NOISE: Fixability.FIXABLE,
    IssueCode.CHROMATIC_ABERRATION: Fixability.FIXABLE,
    IssueCode.UNUSABLE_AUDIO: Fixability.FIXABLE,
    IssueCode.NEEDS_TRIM: Fixability.FIXABLE,
    IssueCode.HEAVY_NOISE: Fixability.PARTIAL,
    IssueCode.SOME_CLIPPED_HIGHLIGHTS: Fixability.PARTIAL,
    IssueCode.SOME_CRUSHED_SHADOWS: Fixability.PARTIAL,
    IssueCode.MODERATE_SHAKE: Fixability.PARTIAL,
    IssueCode.BACKGROUND_CLUTTER: Fixability.PARTIAL,
    IssueCode.SOFT_FOCUS: Fixability.PARTIAL,
    IssueCode.OVERSHARPENED: Fixability.PARTIAL,
    IssueCode.COMPRESSION_ARTIFACTS: Fixability.PARTIAL,
    IssueCode.EXPOSURE_DRIFT: Fixability.PARTIAL,
    IssueCode.ROLLING_SHUTTER: Fixability.PARTIAL,
    IssueCode.WEAKER_DUPLICATE: Fixability.PARTIAL,
    IssueCode.SHORT_CLIP: Fixability.PARTIAL,
    IssueCode.MISSED_FOCUS: Fixability.UNRECOVERABLE,
    IssueCode.SEVERE_MOTION_BLUR: Fixability.UNRECOVERABLE,
    IssueCode.BLOWN_HIGHLIGHTS: Fixability.UNRECOVERABLE,
    IssueCode.CRUSHED_SHADOWS: Fixability.UNRECOVERABLE,
    IssueCode.INSUFFICIENT_RESOLUTION: Fixability.UNRECOVERABLE,
    IssueCode.SEVERE_ARTIFACTS: Fixability.UNRECOVERABLE,
    IssueCode.CORRUPT_FILE: Fixability.UNRECOVERABLE,
    IssueCode.EMPTY_FRAME: Fixability.UNRECOVERABLE,
    IssueCode.DEAD_MOMENT: Fixability.UNRECOVERABLE,
    IssueCode.OBSTRUCTED_SUBJECT: Fixability.UNRECOVERABLE,
    IssueCode.LEGAL_BLOCKER: Fixability.UNRECOVERABLE,
    IssueCode.NO_USABLE_SEGMENT: Fixability.UNRECOVERABLE,
    IssueCode.UNUSABLE_SHAKE: Fixability.UNRECOVERABLE,
    IssueCode.BROKEN_FOCUS_PULL: Fixability.UNRECOVERABLE,
    IssueCode.ENCODING_CORRUPTION: Fixability.UNRECOVERABLE,
    IssueCode.UNUSABLE_DURATION: Fixability.UNRECOVERABLE,
}

# A clip is only *unusably* short when there is essentially no footage at all.
# Below a marketplace's three-second floor is a submission fact; below this is a
# fragment. The two were conflated, and one-second clips were being proposed for
# deletion as though they were corrupt.
TRULY_UNUSABLE_DURATION = 0.4


@dataclass
class Issue:
    code: IssueCode
    detail: str = ""
    # 0..1, how sure the detector is. Feeds the confidence score, and keeps a
    # marginal detection from carrying the same weight as an obvious one.
    certainty: float = 1.0

    @property
    def fixability(self) -> Fixability:
        return FIXABILITY[self.code]

    @property
    def is_blocker(self) -> bool:
        return self.fixability is Fixability.UNRECOVERABLE

    def describe(self) -> str:
        return f"{self.code.value}: {self.detail}" if self.detail else self.code.value


@dataclass
class IssueSet:
    issues: list[Issue] = field(default_factory=list)

    def add(self, code: IssueCode, detail: str = "", certainty: float = 1.0) -> None:
        self.issues.append(Issue(code=code, detail=detail, certainty=certainty))

    def of(self, fixability: Fixability) -> list[Issue]:
        return [i for i in self.issues if i.fixability is fixability]

    @property
    def fixable(self) -> list[Issue]:
        return self.of(Fixability.FIXABLE)

    @property
    def partial(self) -> list[Issue]:
        return self.of(Fixability.PARTIAL)

    @property
    def unrecoverable(self) -> list[Issue]:
        return self.of(Fixability.UNRECOVERABLE)

    @property
    def has_blocker(self) -> bool:
        return any(i.is_blocker for i in self.issues)

    def codes(self) -> set[IssueCode]:
        return {i.code for i in self.issues}

    def __len__(self) -> int:
        return len(self.issues)

    def __iter__(self):
        return iter(self.issues)


# --- thresholds -------------------------------------------------------------
#
# Placed together so that tuning is one edit rather than a search. The blur
# thresholds come from measurements on a real 193-frame RAW archive: an
# ordinary sharp frame sits around 250, a foggy but correctly focused frame
# around 6-14, and a genuinely defocused frame at 1.08. See technical_filter.

BLUR_RATIO_MISSED_FOCUS = 2.0
BLUR_RATIO_SOFT = 6.0

# Deliberately loose. An earlier version of these thresholds rejected four good
# photographs from a real archive -- fog over terraces, two hazy sunsets, one
# bright sky -- and the only thing that caught it was building a contact sheet
# and looking at it. A backlit subject legitimately blows a third of the frame;
# a night shot is legitimately most black.
CLIP_HIGHLIGHT_SOME = 0.03
CLIP_HIGHLIGHT_SEVERE = 0.35
CLIP_SHADOW_SOME = 0.15
CLIP_SHADOW_SEVERE = 0.70

NOISE_MILD = 3.0
NOISE_HEAVY = 8.0

DARK_MEAN = 70.0
BRIGHT_MEAN = 200.0
FLAT_STDDEV = 28.0
CAST_RATIO = 1.18

MIN_MEGAPIXELS_STOCK = 4.0
MIN_MEGAPIXELS_ANY = 1.0


def detect_photo_issues(
    report,
    *,
    megapixels: float,
    mean_luma: float,
    stddev_luma: float,
    channel_means: tuple[float, float, float] | None = None,
    noise_estimate: float = 0.0,
    is_raw: bool = False,
) -> IssueSet:
    """Turn Stage 0 measurements into typed, explainable issues.

    `report` is a `technical_filter.TechnicalReport`. RAW files get more
    tolerance on highlights because the preview is a rendered JPEG and the
    RAW behind it genuinely holds one to two stops the preview cannot show --
    calling a RAW's highlights unrecoverable from a JPEG preview is a guess
    dressed up as a measurement.
    """
    found = IssueSet()

    if report.blur_ratio < BLUR_RATIO_MISSED_FOCUS:
        found.add(
            IssueCode.MISSED_FOCUS,
            f"blur ratio {report.blur_ratio:.2f} below {BLUR_RATIO_MISSED_FOCUS}",
        )
    elif report.blur_ratio < BLUR_RATIO_SOFT:
        found.add(
            IssueCode.SOFT_FOCUS,
            f"blur ratio {report.blur_ratio:.2f}",
            certainty=0.6,
        )

    highlight_severe = CLIP_HIGHLIGHT_SEVERE * (2.0 if is_raw else 1.0)
    if report.clipped_highlights > highlight_severe:
        found.add(
            IssueCode.BLOWN_HIGHLIGHTS,
            f"{report.clipped_highlights:.1%} of the frame is pure white",
        )
    elif report.clipped_highlights > CLIP_HIGHLIGHT_SOME:
        found.add(
            IssueCode.SOME_CLIPPED_HIGHLIGHTS,
            f"{report.clipped_highlights:.1%}",
            certainty=0.7,
        )

    if report.clipped_shadows > CLIP_SHADOW_SEVERE:
        found.add(
            IssueCode.CRUSHED_SHADOWS,
            f"{report.clipped_shadows:.1%} of the frame is pure black",
        )
    elif report.clipped_shadows > CLIP_SHADOW_SOME:
        found.add(IssueCode.SOME_CRUSHED_SHADOWS, f"{report.clipped_shadows:.1%}", certainty=0.7)

    # Only genuine unusability is an issue here. Being below a particular
    # marketplace's 4 MP floor is a *marketplace* fact, not a defect in the
    # photograph -- a 3 MP frame is still fine for editorial, for print at size,
    # and for the portfolio. `marketplaces.py` is what refuses it, per platform.
    if megapixels < MIN_MEGAPIXELS_ANY:
        found.add(IssueCode.INSUFFICIENT_RESOLUTION, f"{megapixels} MP is unusable at any size")

    if mean_luma < DARK_MEAN:
        found.add(IssueCode.UNDEREXPOSED, f"mean luma {mean_luma:.0f}")
    elif mean_luma > BRIGHT_MEAN:
        found.add(IssueCode.OVEREXPOSED, f"mean luma {mean_luma:.0f}")

    if stddev_luma < FLAT_STDDEV:
        found.add(IssueCode.FLAT_CONTRAST, f"luma sigma {stddev_luma:.0f}")

    if channel_means:
        r, g, b = channel_means
        neutral = max(min(r, g, b), 1.0)
        if max(r, g, b) / neutral > CAST_RATIO:
            found.add(IssueCode.COLOR_CAST, f"channel means {r:.0f}/{g:.0f}/{b:.0f}", certainty=0.5)

    if noise_estimate > NOISE_HEAVY:
        found.add(IssueCode.HEAVY_NOISE, f"sigma {noise_estimate:.1f}")
    elif noise_estimate > NOISE_MILD:
        found.add(IssueCode.MILD_NOISE, f"sigma {noise_estimate:.1f}")

    return found


def summarise(found: IssueSet) -> dict[str, list[str]]:
    """The three lists the report and the UI both print."""
    return {
        "fixable": [i.describe() for i in found.fixable],
        "partially_fixable": [i.describe() for i in found.partial],
        "unrecoverable": [i.describe() for i in found.unrecoverable],
    }
