"""Whether a "defect" was a decision, and what a frame is worth if it was.

This module exists because the rest of the pipeline is a technical filter, and a
technical filter has a specific blind spot: it can prove a file is broken, but
it cannot prove that a dark, blurred, tilted, off-centre photograph is not the
picture the photographer meant to make. Those two failure modes are not
symmetric. Leaving a bad frame in review costs a few seconds of a human's
attention; deleting an intentional one destroys work.

So everything here is built to move in one direction only. **Artistic signals
can rescue a frame from the destructive path; they can never push one onto it.**
`can_only_rescue` enforces that as an assertion rather than a convention.

## What is measured, and what that measurement is worth

`intentionality_likelihood` is the load-bearing idea: for each apparent defect,
is there deterministic evidence that it was a choice?

- **Directional blur.** Camera shake is close to isotropic -- the hand moves in
  no particular direction. Panning and intentional camera movement smear along
  one axis while leaving detail across it intact. Measured by comparing how much
  fine detail survives on each axis, *not* by a structure tensor over first
  derivatives: that version was dominated by the scene's own layout and scored a
  perfectly sharp frame of buildings at 0.45, indistinguishable from a real
  smear. Even corrected, the classes overlap, so the gate is set for precision
  rather than sensitivity -- see the constant for the measurements.
- **Selective focus.** A sharp region inside a soft frame is shallow depth of
  field, not a miss. Already measured as max-tile against global sharpness.
- **Low-key vs underexposed.** A deliberate low-key frame keeps a readable
  bright region; an accidentally dark one is uniformly dark. Measured as the
  brightness of the frame's brightest decile.
- **Dutch angle vs sloppy horizon.** Accidental tilt clusters below ~3 degrees
  because nobody means to be slightly off. A deliberate angle is usually well
  past that.
- **High-key clipping.** Blown highlights around a subject that remains legible
  is a look; a uniformly white frame is an error.

**These are heuristics, not validated models.** Each is a plausible physical
correlate of a choice, tested against generated cases and a small real archive,
and none has been validated against a labelled corpus of intentional versus
accidental frames. They are calibrated to be generous: a false "this was
intentional" costs a manual review, while a false "this was an accident" is the
error that loses a photograph. Where a signal cannot decide, it abstains and
raises `curatorial_uncertainty`, which routes to review.

The remaining dimensions -- emotional resonance, visual tension, narrative
openness, moment specificity, formal coherence, distinctiveness, documentary
significance -- are **not** computable from pixels and are not faked here. They
come from the vision model, and when it has not run they are `None`, not zero.
A dimension nobody measured must not read as a dimension that scored badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from PIL import Image, ImageFilter

# --- thresholds, all provisional --------------------------------------------
#
# Sources are stated per constant. None is fitted against a labelled set; see
# the module docstring and README "Known limitations".

# How directional the loss of fine detail is: 0 = equal on both axes, 1 = one
# axis destroyed and the other intact.
#
# Measured across three generated scenes (horizontal smear / vertical smear /
# isotropic Gaussian / sharp):
#
#   seed 2   0.910   0.776   0.445   0.153
#   seed 4   0.940   0.624   0.609   0.373
#   seed 7   0.927   0.744   0.397   0.221
#
# The classes **overlap**: one vertical pan (0.624) sits below one isotropic
# blur (0.609). A threshold that catches every pan would therefore also call
# some shake intentional. So the gate is set high deliberately -- above every
# isotropic case measured -- which makes it precise rather than sensitive.
#
# That trade is chosen because of what the signal is for. Firing means "this
# looks deliberate, keep it"; not firing means "cannot tell", which also keeps
# it. Missing a real pan costs a manual review. Calling shake deliberate would
# put a broken frame in front of a photographer as a candidate, which is the
# error that wastes their trust.
DIRECTIONAL_BLUR_ANISOTROPY = 0.70

# A sharp region this much sharper than the frame average reads as selective
# focus rather than a missed one. From the earlier Stage 0 measurements, where
# a portrait with a sharp subject scored tile 1401 against global 166.
SELECTIVE_FOCUS_RATIO = 4.0

# A deliberate low-key frame keeps something legible. Below this the frame is
# uniformly dark and there is nothing to have intended.
LOW_KEY_HIGHLIGHT_FLOOR = 90.0

# Accidental tilt clusters small: nobody means to be one degree off.
DELIBERATE_TILT_DEGREES = 3.0

# High-key: highlights blown but a legible darker subject still present.
HIGH_KEY_SUBJECT_FLOOR = 60.0


class ArtRoute(StrEnum):
    """Non-destructive destinations. There is deliberately no trash here.

    Technical failure is decided by `scoring.RouteClass`, from measurements.
    Nothing in this vocabulary can send a photograph to be deleted.
    """

    TECHNICAL_FAILURE = "technical_failure"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    REVIEW = "review"
    DOCUMENTARY_CANDIDATE = "documentary_candidate"
    ART_CANDIDATE = "art_candidate"
    EMOTION_CANDIDATE = "emotion_candidate"
    SERIES_CANDIDATE = "series_candidate"
    STOCK_STANDARD = "stock_standard"
    STOCK_STRONG = "stock_strong"
    PORTFOLIO_CANDIDATE = "portfolio_candidate"
    ARCHIVE_ONLY = "archive_only"


# Anything in this set means a human decides. None of them is a deletion.
NON_DESTRUCTIVE = frozenset(ArtRoute) - {ArtRoute.TECHNICAL_FAILURE}

# Roles a frame can play in a series even when it is weak alone. Grounded in
# editing practice: "Not every image needs to say everything. Some function as
# transitions, others as climaxes, others as closures."
SERIES_ROLES = (
    "transition",
    "pause",
    "establishing",
    "counterpoint",
    "recurring_motif",
    "closing",
    "context",
    "turn",
)


@dataclass
class IntentSignal:
    """One piece of evidence that an apparent defect was a decision."""

    defect: str
    verdict: str  # "likely_intentional" | "likely_accidental" | "cannot_tell"
    confidence: float
    evidence: str

    @property
    def rescues(self) -> bool:
        return self.verdict != "likely_accidental"


@dataclass
class ArtisticScores:
    """Twelve dimensions, kept apart on purpose.

    `None` means "not measured", which is different from 0 and must stay
    different: collapsing them lets an unrun vision pass read as a bad review.
    """

    technical_integrity: int = 0
    intentionality_likelihood: int = 0
    curatorial_uncertainty: int = 100

    emotional_resonance: int | None = None
    visual_tension: int | None = None
    narrative_openness: int | None = None
    moment_specificity: int | None = None
    formal_coherence: int | None = None
    distinctiveness: int | None = None
    documentary_significance: int | None = None
    conventional_beauty: int | None = None

    artistic_candidate: int = 0
    series_role: str = ""
    signals: list[IntentSignal] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "technical_integrity": self.technical_integrity,
            "intentionality_likelihood": self.intentionality_likelihood,
            "curatorial_uncertainty": self.curatorial_uncertainty,
            "emotional_resonance": self.emotional_resonance,
            "visual_tension": self.visual_tension,
            "narrative_openness": self.narrative_openness,
            "moment_specificity": self.moment_specificity,
            "formal_coherence": self.formal_coherence,
            "distinctiveness": self.distinctiveness,
            "documentary_significance": self.documentary_significance,
            "conventional_beauty": self.conventional_beauty,
            "artistic_candidate": self.artistic_candidate,
            "series_role": self.series_role,
        }
        payload["intent_signals"] = [
            {"defect": s.defect, "verdict": s.verdict, "evidence": s.evidence}
            for s in self.signals
        ]
        return payload

    @property
    def measured(self) -> dict[str, int]:
        """Only the dimensions something actually looked at."""
        return {
            k: v
            for k, v in self.to_dict().items()
            if isinstance(v, int) and k != "curatorial_uncertainty"
        }

    @property
    def has_any_artistic_signal(self) -> bool:
        """Whether anything at all argues for keeping this frame."""
        soft = [
            self.emotional_resonance,
            self.visual_tension,
            self.narrative_openness,
            self.moment_specificity,
            self.distinctiveness,
            self.documentary_significance,
        ]
        return any(v is not None and v >= 55 for v in soft) or self.intentionality_likelihood >= 55


# --- the deterministic half: was this a decision? ----------------------------


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def blur_anisotropy(rgb: np.ndarray) -> tuple[float, float]:
    """How directional the image's blur is, and along what angle.

    Structure tensor over the whole frame. A smeared image has gradient energy
    concentrated on one axis (high anisotropy); a defocused or shaken one
    spreads it evenly. This is the difference between intentional camera
    movement and a shaky hand, and it is measurable rather than guessed.

    Limitation: a scene that is *itself* strongly directional -- a picket fence,
    rain, a striped wall -- also reads as anisotropic. The signal is therefore
    only consulted when the frame is *already* soft, where a strong direction is
    far more likely to be motion than subject matter.
    """
    luma = _luma(rgb)
    if min(luma.shape) < 16:
        return 0.0, 0.0
    smoothed = np.asarray(
        Image.fromarray(np.clip(luma, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)),
        dtype=np.float64,
    )
    gy = smoothed[2:, 1:-1] - smoothed[:-2, 1:-1]
    gx = smoothed[1:-1, 2:] - smoothed[1:-1, :-2]

    # Measured on *second* derivatives, not first. The structure tensor over
    # first derivatives is dominated by the scene's own layout -- a frame full
    # of vertical building edges reads as 0.45 anisotropic while perfectly
    # sharp, which is the same figure a genuine smear produces, so the two
    # cannot be told apart.
    #
    # Motion destroys high-frequency detail *along the direction of travel* and
    # leaves it intact across. Comparing the second derivative along each axis
    # isolates exactly that, because it asks what fine detail survives rather
    # than which way the big edges run.
    lxx = float(np.var(gx[:, 1:] - gx[:, :-1]))
    lyy = float(np.var(gy[1:, :] - gy[:-1, :]))
    total = lxx + lyy
    if total <= 1e-9:
        return 0.0, 0.0
    anisotropy = abs(lxx - lyy) / total
    angle = 0.0 if lxx >= lyy else 90.0
    return round(anisotropy, 4), angle


def brightest_decile(rgb: np.ndarray) -> float:
    """Mean luma of the brightest tenth. Separates low-key from underexposed."""
    luma = _luma(rgb).ravel()
    if luma.size == 0:
        return 0.0
    cut = np.percentile(luma, 90)
    top = luma[luma >= cut]
    return round(float(top.mean()) if top.size else 0.0, 2)


def assess_intent(
    rgb: np.ndarray,
    *,
    blur_ratio: float,
    sharpness_global: float,
    sharpness_tile: float,
    tilt_degrees: float,
    clipped_highlights: float,
    mean_luma: float,
    iso: int | None = None,
) -> list[IntentSignal]:
    """For each apparent defect, look for evidence that it was a choice.

    Returns one signal per defect actually present. A frame with no apparent
    defects returns nothing, which is not the same as "everything was
    accidental".
    """
    signals: list[IntentSignal] = []

    if blur_ratio < 6.0:
        anisotropy, angle = blur_anisotropy(rgb)
        selective = sharpness_tile > sharpness_global * SELECTIVE_FOCUS_RATIO
        if selective:
            signals.append(
                IntentSignal(
                    "softness",
                    "likely_intentional",
                    0.75,
                    f"a region is {sharpness_tile / max(sharpness_global, 1e-6):.1f}x sharper than "
                    "the frame average: selective focus, not a missed one",
                )
            )
        elif anisotropy >= DIRECTIONAL_BLUR_ANISOTROPY:
            signals.append(
                IntentSignal(
                    "motion blur",
                    "likely_intentional",
                    0.65,
                    f"blur is directional (anisotropy {anisotropy:.2f} along {angle:.0f}deg), "
                    "which is panning or intentional camera movement rather than shake",
                )
            )
        else:
            signals.append(
                IntentSignal(
                    "softness",
                    "cannot_tell",
                    0.35,
                    f"blur is undirected (anisotropy {anisotropy:.2f}) and no region is sharp; "
                    "this looks like missed focus but a deliberately soft frame looks the same",
                )
            )

    if mean_luma < 70.0:
        highlight = brightest_decile(rgb)
        if highlight >= LOW_KEY_HIGHLIGHT_FLOOR:
            signals.append(
                IntentSignal(
                    "darkness",
                    "likely_intentional",
                    0.7,
                    f"the brightest tenth reaches {highlight:.0f}: a low-key frame with a "
                    "legible bright region, not a uniformly underexposed one",
                )
            )
        else:
            signals.append(
                IntentSignal(
                    "darkness",
                    "cannot_tell",
                    0.4,
                    f"uniformly dark (brightest tenth only {highlight:.0f}); could be "
                    "underexposure or could be the intended near-black frame",
                )
            )

    if abs(tilt_degrees) >= DELIBERATE_TILT_DEGREES:
        signals.append(
            IntentSignal(
                "tilted horizon",
                "likely_intentional",
                0.6,
                f"{abs(tilt_degrees):.1f}deg is well past the one-or-two degrees of an "
                "accidental slope: this reads as a deliberate angle",
            )
        )
    elif 0.5 <= abs(tilt_degrees) < DELIBERATE_TILT_DEGREES:
        signals.append(
            IntentSignal(
                "tilted horizon",
                "cannot_tell",
                0.3,
                f"{abs(tilt_degrees):.1f}deg could be carelessness or could be deliberate; "
                "either way a rotation is one slider and costs nothing",
            )
        )

    if clipped_highlights > 0.10 and mean_luma >= HIGH_KEY_SUBJECT_FLOOR:
            signals.append(
                IntentSignal(
                    "blown highlights",
                    "likely_intentional",
                    0.55,
                    f"{clipped_highlights:.0%} blown with the frame still averaging "
                    f"{mean_luma:.0f}: a high-key treatment rather than an exposure error",
                )
            )

    if iso and iso >= 3200:
        signals.append(
            IntentSignal(
                "grain",
                "likely_intentional",
                0.5,
                f"ISO {iso} in low light is a choice made to get the frame at all; "
                "grain is the cost of the picture existing, not a defect in it",
            )
        )

    return signals


def intentionality_score(signals: list[IntentSignal]) -> int:
    """0-100. High means the apparent defects look like decisions.

    A frame with no apparent defects scores 50 -- neutral, because there is
    nothing to have intended. The scale is deliberately generous: it exists to
    keep frames out of a destructive path, so being wrong upward costs a review
    and being wrong downward costs a photograph.
    """
    if not signals:
        return 50
    weight = {"likely_intentional": 1.0, "cannot_tell": 0.55, "likely_accidental": 0.0}
    total = sum(weight[s.verdict] * s.confidence for s in signals)
    possible = sum(s.confidence for s in signals)
    return int(round(100 * total / possible)) if possible else 50


def uncertainty_score(signals: list[IntentSignal], *, semantic_present: bool) -> int:
    """How unsure we are that we understood the frame. High routes to review."""
    score = 30 if semantic_present else 70
    undecided = [s for s in signals if s.verdict == "cannot_tell"]
    score += 12 * len(undecided)
    return int(max(0, min(100, score)))


# --- the safety property, enforced rather than intended ----------------------


def can_only_rescue(before: str, after: str) -> bool:
    """An artistic signal may move a frame off the destructive path, never onto it."""
    return not (before != ArtRoute.TECHNICAL_FAILURE.value and after == ArtRoute.TECHNICAL_FAILURE.value)


def apply_conservative_art(
    route_class: str,
    scores: ArtisticScores,
    *,
    enabled: bool = True,
) -> tuple[str, str]:
    """In conservative-art mode, any artistic signal blocks automatic trash.

    Returns the (possibly rescued) class and the reason. This is the switch the
    brief calls for: a mode in which the tool will not destroy anything that
    shows a sign of being deliberate, however low its other scores are.
    """
    if not enabled or route_class != ArtRoute.TECHNICAL_FAILURE.value:
        return route_class, ""

    if scores.curatorial_uncertainty >= 60:
        return (
            ArtRoute.REVIEW.value,
            f"conservative-art: uncertainty {scores.curatorial_uncertainty} is too high "
            "to destroy anything on this evidence",
        )
    if scores.has_any_artistic_signal:
        reasons = [s.evidence for s in scores.signals if s.verdict == "likely_intentional"]
        return (
            ArtRoute.REVIEW.value,
            "conservative-art: "
            + (reasons[0] if reasons else "an artistic or documentary signal is present"),
        )
    return route_class, ""


# --- collection level --------------------------------------------------------


@dataclass
class SeriesContext:
    """What a frame is for, inside the set rather than on its own."""

    role: str = ""
    neighbours: list[str] = field(default_factory=list)
    is_alternative: bool = False
    reason: str = ""


def similar_alternatives(
    frames: list[dict],
    *,
    quality_margin: float,
) -> bool:
    """Whether near-identical frames differ in something a Laplacian cannot see.

    `frames` carries per-frame `quality` plus, when the vision pass ran,
    `moment_specificity` and `emotional_resonance`. If the technical scores are
    within the margin but the human-visible ones differ, these are not a strong
    frame and a weak one -- they are two takes of a moment, and which is better
    is the photographer's call. They become `similar_alternatives` rather than
    a winner and a set of deletions.
    """
    if len(frames) < 2:
        return False
    qualities = [f.get("quality", 0.0) for f in frames]
    if max(qualities) - min(qualities) >= quality_margin:
        return False

    for key in ("moment_specificity", "emotional_resonance"):
        values = [f[key] for f in frames if f.get(key) is not None]
        if len(values) >= 2 and max(values) - min(values) >= 10:
            return True
    # Technically indistinguishable and nothing else was measured: still not a
    # basis for choosing, so they remain alternatives.
    return True
