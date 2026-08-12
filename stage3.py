"""Stage 3: the artistic read, and the portrait analysis a face demands.

This module exists because Stage 3 was written and never wired up. The prompt
sat in `prompts.py`, nothing called it, every artistic field in every report was
`null`, and `flagship` was being assigned on the strength of three ranked axes
that say nothing about whether a photograph is any good. A portrait with the
subject mid-blink reached `flagship` because it was sharp and hard to repeat.

Two rules hold the design together.

**A null is not a score.** Every field here is either a validated number or an
explicit status saying why there isn't one. `Stage3Status` distinguishes "not
needed", "not yet run", "ran and succeeded", "ran and failed", and "deliberately
skipped, here is the reason". Nothing downstream is allowed to read a missing
value as a low one.

**HERO requires evidence, not the absence of objections.** `hero_blockers()`
returns the reasons a frame may not be promoted, and an unfinished Stage 3 is
itself a blocker. A frame cannot become the best thing in a shoot because the
analysis that would have judged it timed out.

Portrait analysis is separate from the general artistic read for a specific
reason: a blink is not an aesthetic property. Whole-frame scoring rates the
light, the composition and the colour of a photograph whose subject has their
eyes shut, and averages the blink away. So expression is asked about directly,
with its own confidence, and gates the promotion on its own.

Face detection is optional. When OpenCV is installed a real bounding box is
used; otherwise the crops are derived from where faces sit in a portrait and the
model -- which is looking at the picture anyway, and is far better at this than
a Haar cascade -- reports the geometry back. Neither path is allowed to silently
become the other: `face_source` records which one ran.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from PIL import Image

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROMPT_VERSION = "stage3-2026-08-12b"

MAX_RETRIES = 2

# The eight dimensions, all 0-100. Named once so parsing, validation, storage
# and the tests cannot disagree about what Stage 3 produces.
ARTISTIC_FIELDS = (
    "emotional_resonance",
    "visual_tension",
    "narrative_openness",
    "moment_specificity",
    "formal_coherence",
    "distinctiveness",
    "documentary_significance",
    "conventional_beauty",
)

# Below both of these, at confidence, the model has said it would not publish
# this picture of this person. Set where a real archive put them: the frame that
# exposed the gap scored 35 and 43.
UNUSABLE_EXPRESSION_QUALITY = 45
UNUSABLE_PUBLISHABILITY = 50

PORTRAIT_SCORE_FIELDS = (
    "face_sharpness",
    "expression_quality",
    "pose_quality",
    "face_occlusion",
    "blink_probability",
    "grimace_probability",
    "portrait_publishability",
    "expression_confidence",
)


class Stage3Status(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EyesState(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    SQUINTING = "SQUINTING"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    UNCLEAR = "UNCLEAR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Expression(StrEnum):
    GOOD = "GOOD"
    NEUTRAL = "NEUTRAL"
    AWKWARD = "AWKWARD"
    GRIMACE = "GRIMACE"
    BLINK = "BLINK"
    MID_SPEECH = "MID_SPEECH"
    UNCLEAR = "UNCLEAR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class IntentReading(StrEnum):
    DELIBERATE = "DELIBERATE"
    ACCIDENTAL = "ACCIDENTAL"
    CANNOT_TELL = "CANNOT_TELL"


class SeriesRole(StrEnum):
    """What a frame does in a set, when it does little on its own."""

    TRANSITION = "TRANSITION"
    PAUSE = "PAUSE"
    ESTABLISHING = "ESTABLISHING"
    COUNTERPOINT = "COUNTERPOINT"
    RECURRING_MOTIF = "RECURRING_MOTIF"
    CLOSING = "CLOSING"
    CONTEXT = "CONTEXT"
    TURN = "TURN"
    NONE = "NONE"


# Expressions that are a failure of the moment rather than a style choice.
#
# MID_SPEECH is here because of a frame this missed: a portrait of a man with
# his eyelids lowered and his mouth open on a word. Stage 3 read it exactly
# right -- PARTIALLY_CLOSED, MID_SPEECH, expression quality 35, publishability
# 43, confidence 73 -- and the frame was still filed as good, because the label
# it chose was not on this list. A person caught mid-word is the single most
# common unusable outtake in any portrait shoot.
BAD_EXPRESSIONS = frozenset(
    {Expression.AWKWARD, Expression.GRIMACE, Expression.BLINK, Expression.MID_SPEECH}
)
UNCERTAIN_EXPRESSIONS = frozenset({Expression.UNCLEAR})

# Eye states that fail a portrait. SQUINTING is deliberately absent: squinting
# into bright light is what faces do outdoors, and it is not a failed frame.
CLOSED_EYES = frozenset({EyesState.CLOSED, EyesState.PARTIALLY_CLOSED})


@dataclass
class PortraitAssessment:
    """What the face is doing. Separate because a blink is not an aesthetic."""

    face_count: int = 0
    primary_face_visible: bool = False
    primary_face_area_ratio: float = 0.0
    face_sharpness: int = 0
    eyes_state: str = EyesState.NOT_APPLICABLE.value
    expression: str = Expression.NOT_APPLICABLE.value
    expression_quality: int = 0
    pose_quality: int = 0
    face_occlusion: int = 0
    blink_probability: int = 0
    grimace_probability: int = 0
    portrait_publishability: int = 0
    expression_confidence: int = 0
    portrait_reasoning: str = ""
    portrait_blockers: list[str] = field(default_factory=list)
    # "detector" when a real bounding box was found, "derived" when the crops
    # were positioned from portrait geometry, "none" when no crop was made.
    face_source: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def applies(self) -> bool:
        return self.face_count > 0 and self.primary_face_visible

    @property
    def expression_is_bad(self) -> bool:
        return Expression(self.expression) in BAD_EXPRESSIONS

    @property
    def eyes_are_shut(self) -> bool:
        return EyesState(self.eyes_state) in CLOSED_EYES

    @property
    def expression_is_uncertain(self) -> bool:
        return Expression(self.expression) in UNCERTAIN_EXPRESSIONS

    @property
    def reads_as_unusable(self) -> bool:
        """The model's own verdict, independent of which label it picked.

        The enums cannot enumerate every way a face fails, and the frame that
        exposed this was labelled MID_SPEECH -- accurate, and at the time not on
        any list. These two numbers are the model saying, in its own terms, that
        it would not publish this picture of this person.
        """
        return (
            self.expression_quality <= UNUSABLE_EXPRESSION_QUALITY
            and self.portrait_publishability <= UNUSABLE_PUBLISHABILITY
        )


@dataclass
class ArtisticAssessment:
    """One frame's Stage 3 result, or an explicit account of why there isn't one."""

    status: str = Stage3Status.PENDING.value

    emotional_resonance: int | None = None
    visual_tension: int | None = None
    narrative_openness: int | None = None
    moment_specificity: int | None = None
    formal_coherence: int | None = None
    distinctiveness: int | None = None
    documentary_significance: int | None = None
    conventional_beauty: int | None = None

    artistic_candidate: bool = False
    artistic_confidence: int = 0
    artistic_reasoning: str = ""
    artistic_strengths: list[str] = field(default_factory=list)
    artistic_weaknesses: list[str] = field(default_factory=list)

    # Asked for by the prompt from the start and then thrown away on parse.
    # `uncertainty` is what makes an abstention possible: without it every
    # unclear frame either gets a confident wrong answer or goes to a review
    # queue on a technicality. `intent_reading` is the only thing that can say a
    # defect was a choice. `series_role` keeps a frame that is weak alone.
    intent_reading: dict[str, str] = field(default_factory=dict)
    uncertainty: int = 0
    series_role: str = SeriesRole.NONE.value

    portrait: PortraitAssessment | None = None

    model: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: int = SCHEMA_VERSION
    analysed_at: str = ""
    parse_errors: list[str] = field(default_factory=list)
    retries: int = 0
    skip_reason: str = ""

    # --- the questions the rest of the pipeline asks --------------------

    @property
    def completed(self) -> bool:
        return self.status == Stage3Status.COMPLETED.value

    @property
    def required(self) -> bool:
        return self.status not in (
            Stage3Status.NOT_REQUIRED.value,
            Stage3Status.SKIPPED.value,
        )

    @property
    def has_all_fields(self) -> bool:
        return all(getattr(self, name) is not None for name in ARTISTIC_FIELDS)

    @property
    def usable(self) -> bool:
        """Completed *and* actually carrying the numbers it promised."""
        return self.completed and self.has_all_fields

    @property
    def says_deliberate(self) -> bool:
        """Whether the read positively claims a defect was a choice.

        One acceptable form of evidence: the structured `intent_reading` field
        naming a defect as deliberate. A high aesthetic score is not evidence of
        intent, and neither is the word "deliberate" appearing in the prose.

        That second exclusion was learned. This property used to search
        `artistic_reasoning` for "deliberate", and on a real frame it matched
        the sentence *"it does not clearly read as deliberate and may simply be
        mid-speech"* -- a read that says the exact opposite. The negation is
        invisible to a substring test, and this override is load-bearing enough
        that a keyword search is the wrong mechanism for it.
        """
        return any(v == IntentReading.DELIBERATE.value for v in self.intent_reading.values())

    @property
    def carries_the_series(self) -> bool:
        """Weak alone, doing a job in the set. Never a reason to promote."""
        return SeriesRole(self.series_role) is not SeriesRole.NONE

    def score(self, name: str) -> int:
        value = getattr(self, name, None)
        return 0 if value is None else int(value)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["portrait"] = self.portrait.to_dict() if self.portrait else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> ArtisticAssessment:
        data = dict(payload or {})
        portrait = data.pop("portrait", None)
        known = set(cls.__dataclass_fields__)
        out = cls(**{k: v for k, v in data.items() if k in known})
        if portrait:
            fields = set(PortraitAssessment.__dataclass_fields__)
            out.portrait = PortraitAssessment(
                **{k: v for k, v in portrait.items() if k in fields}
            )
        return out

    @classmethod
    def not_required(cls, reason: str) -> ArtisticAssessment:
        return cls(status=Stage3Status.NOT_REQUIRED.value, skip_reason=reason)

    @classmethod
    def skipped(cls, reason: str) -> ArtisticAssessment:
        return cls(status=Stage3Status.SKIPPED.value, skip_reason=reason)

    @classmethod
    def failed(cls, errors: list[str], *, retries: int = 0, model: str = "") -> ArtisticAssessment:
        return cls(
            status=Stage3Status.FAILED.value,
            parse_errors=list(errors),
            retries=retries,
            model=model,
            analysed_at=_now(),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- when Stage 3 has to run ------------------------------------------------


def should_run(
    *,
    route_class: str,
    has_unrecoverable: bool,
    intentionality_likelihood: int,
    curatorial_uncertainty: int,
    faces_present: bool,
    corrupt: bool = False,
) -> tuple[bool, str]:
    """Whether this frame needs an artistic read, and why.

    Deliberately generous. Stage 3 is the only thing that can promote a frame to
    HERO, so anything that might plausibly get there has to be looked at --
    including frames whose technical defects may be deliberate, which is exactly
    the population a technical filter is worst at judging.
    """
    if corrupt:
        return False, "the file does not decode"
    if has_unrecoverable and intentionality_likelihood < 40:
        return False, "confidently unrecoverable, with no sign the defect was a choice"

    if route_class in ("flagship", "stock_strong", "stock_standard"):
        return True, "a keep or hero candidate"
    if faces_present:
        return True, "a face is present and expression decides the outcome"
    if intentionality_likelihood >= 55:
        return True, "an apparent defect may be deliberate"
    if curatorial_uncertainty >= 60:
        return True, "too uncertain to route without looking"
    if route_class in ("review", "duplicate_candidate"):
        return True, "unresolved, and the artistic read may settle it"
    return False, "confidently routed without an artistic judgement"


# --- parsing, strictly --------------------------------------------------------


class Stage3ParseError(ValueError):
    pass


def _clamp_score(value, name: str, errors: list[str]) -> int | None:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        errors.append(f"{name}: not a number ({value!r})")
        return None
    if not 0 <= number <= 100:
        errors.append(f"{name}: {number} out of range, clamped")
    return max(0, min(100, number))


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _enum_or(value, enum_cls, default):
    try:
        return enum_cls(str(value).strip().upper()).value
    except (ValueError, AttributeError):
        return default.value


def parse_assessment(payload: dict, *, model: str = "") -> ArtisticAssessment:
    """Validate one object into an assessment, or raise with the reasons.

    Raising rather than returning a half-filled record is the point. A partially
    parsed Stage 3 that keeps its `COMPLETED` status is exactly the state that
    let null fields reach a routing decision.
    """
    if not isinstance(payload, dict):
        raise Stage3ParseError(f"expected an object, got {type(payload).__name__}")

    errors: list[str] = []
    out = ArtisticAssessment(model=model, analysed_at=_now())

    for name in ARTISTIC_FIELDS:
        if name not in payload:
            errors.append(f"{name}: missing")
            continue
        setattr(out, name, _clamp_score(payload[name], name, errors))

    missing = [name for name in ARTISTIC_FIELDS if getattr(out, name) is None]
    if missing:
        raise Stage3ParseError(f"unusable reply: {'; '.join(errors)}")

    out.artistic_candidate = _as_bool(payload.get("artistic_candidate", False))
    out.artistic_confidence = _clamp_score(
        payload.get("artistic_confidence", 0), "artistic_confidence", errors
    ) or 0
    out.artistic_reasoning = str(payload.get("artistic_reasoning", ""))[:600]
    out.artistic_strengths = [str(s)[:160] for s in (payload.get("artistic_strengths") or [])][:6]
    out.artistic_weaknesses = [str(s)[:160] for s in (payload.get("artistic_weaknesses") or [])][:6]

    intent = payload.get("intent_reading")
    if isinstance(intent, dict):
        out.intent_reading = {
            str(k)[:40]: _enum_or(v, IntentReading, IntentReading.CANNOT_TELL)
            for k, v in list(intent.items())[:8]
        }
    # An absent `uncertainty` is not certainty. Stage 3 that forgot to answer is
    # exactly the case where a person should look, so the default is the middle.
    out.uncertainty = _clamp_score(payload.get("uncertainty", 50), "uncertainty", errors) or 0
    out.series_role = _enum_or(payload.get("series_role"), SeriesRole, SeriesRole.NONE)

    portrait_payload = payload.get("portrait")
    if isinstance(portrait_payload, dict):
        out.portrait = parse_portrait(portrait_payload, errors)

    out.parse_errors = errors
    out.status = Stage3Status.COMPLETED.value
    return out


def parse_portrait(payload: dict, errors: list[str]) -> PortraitAssessment:
    portrait = PortraitAssessment()
    try:
        portrait.face_count = max(0, int(payload.get("face_count", 0)))
    except (TypeError, ValueError):
        errors.append("face_count: not a number")

    portrait.primary_face_visible = _as_bool(payload.get("primary_face_visible", False))
    try:
        portrait.primary_face_area_ratio = max(
            0.0, min(1.0, float(payload.get("primary_face_area_ratio", 0.0)))
        )
    except (TypeError, ValueError):
        errors.append("primary_face_area_ratio: not a number")

    for name in PORTRAIT_SCORE_FIELDS:
        setattr(portrait, name, _clamp_score(payload.get(name, 0), name, errors) or 0)

    portrait.eyes_state = _enum_or(
        payload.get("eyes_state"), EyesState, EyesState.UNCLEAR
    )
    portrait.expression = _enum_or(
        payload.get("expression"), Expression, Expression.UNCLEAR
    )
    portrait.portrait_reasoning = str(payload.get("portrait_reasoning", ""))[:400]
    portrait.portrait_blockers = [
        str(b)[:120] for b in (payload.get("portrait_blockers") or [])
    ][:6]
    portrait.face_source = str(payload.get("face_source", "derived"))
    return portrait


def parse_group(text: str, group: list[str], *, model: str = "") -> dict[str, ArtisticAssessment]:
    """Parse a whole group's reply, mapping objects back onto filenames.

    An object that cannot be placed is dropped rather than guessed at: attaching
    one frame's expression analysis to another frame's file is worse than having
    none.
    """
    import batch_runner

    items = batch_runner.parse_group_json(text or "")
    if not items:
        raise Stage3ParseError("no JSON array in the reply")

    placed = batch_runner.attach_filenames(items, group)
    out: dict[str, ArtisticAssessment] = {}
    for item in placed:
        filename = item.get("filename")
        if not filename:
            continue
        try:
            out[filename] = parse_assessment(item, model=model)
        except Stage3ParseError as e:
            logger.warning("Stage 3 object for %s unusable: %s", filename, e)
    return out


# --- the gates ----------------------------------------------------------------


@dataclass
class HeroThresholds:
    """Every number a promotion depends on, in one place and documented.

    None of these is fitted. They are chosen so that the gate fires on cases a
    person would call obvious -- eyes shut, a grimace, a face out of focus --
    and abstains everywhere else, because a gate that fires on ambiguity moves
    work to the review queue rather than deciding anything.
    """

    min_artistic_confidence: int = 55
    min_expression_quality: int = 70
    min_face_sharpness: int = 65
    min_expression_confidence: int = 60
    max_blink_probability: int = 40
    max_grimace_probability: int = 40
    # A face smaller than this is incidental -- a person in a landscape -- and
    # expression should not gate the frame.
    portrait_face_area: float = 0.04


DEFAULT_THRESHOLDS = HeroThresholds()


def face_gates_the_frame(
    portrait: PortraitAssessment | None,
    *,
    is_portrait: bool = False,
    thresholds: HeroThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    """Whether this face decides the fate of the photograph.

    The area rule exists to stop a person standing in a landscape from gating
    the frame on their expression. It is the right rule for a landscape and the
    wrong one for a portrait: an environmental portrait puts the subject small
    in a wide scene, and on a real frame a face occupying 2.3% of the picture
    was the entire subject -- eyes lowered, mouth mid-word -- and was waved
    through as incidental.

    So when the content pass says the photograph is a portrait, the face gates
    it whatever its size. Everywhere else the area rule still applies.

    This depends on the content pass having run, and that dependency is real: a
    Stage 2 group rejected over one duplicated rank once cost eight photographs
    their genre, and the gate silently stopped applying to them. The fix belongs
    there -- a near-ranking is repaired rather than discarded -- and not here.
    Widening this to "one face, confidently read" was tried and reverted: it
    cannot tell an environmental portrait from a landscape with a passer-by in
    it, and it took the landscape with it.

    So when the genre is unknown the face does not gate the frame, the
    photograph is kept, and the summary says out loud how many files had no
    content check. Kept-and-flagged is the recoverable direction.
    """
    if portrait is None or not portrait.applies:
        return False
    if is_portrait:
        return True
    return portrait.primary_face_area_ratio >= thresholds.portrait_face_area


def hero_blockers(
    assessment: ArtisticAssessment,
    *,
    thresholds: HeroThresholds = DEFAULT_THRESHOLDS,
    is_portrait: bool = False,
) -> list[str]:
    """Every reason this frame may not be promoted. Empty means no objection.

    Note what this is not: a score. A blocker is a fact that survives any amount
    of excellence elsewhere, which is why they are returned as a list rather
    than folded into an average.
    """
    blocking: list[str] = []

    if assessment.status == Stage3Status.NOT_REQUIRED.value:
        return ["artistic analysis was not required, so nothing supports a promotion"]
    if not assessment.completed:
        return [f"artistic analysis is {assessment.status}, not completed"]
    if not assessment.has_all_fields:
        missing = [n for n in ARTISTIC_FIELDS if getattr(assessment, n) is None]
        return [f"artistic analysis is missing {', '.join(missing)}"]
    if assessment.artistic_confidence < thresholds.min_artistic_confidence:
        blocking.append(
            f"artistic confidence {assessment.artistic_confidence} is below "
            f"{thresholds.min_artistic_confidence}"
        )

    portrait = assessment.portrait
    if face_gates_the_frame(portrait, is_portrait=is_portrait, thresholds=thresholds):
        blocking.extend(_portrait_blockers(portrait, thresholds, assessment))

    return blocking


def _portrait_blockers(
    portrait: PortraitAssessment,
    thresholds: HeroThresholds,
    assessment: ArtisticAssessment,
) -> list[str]:
    """A face changes what "good" means, and it is not negotiable by score."""
    blocking: list[str] = []

    # A deliberate, effective expression is the one thing that overrides the
    # eye-state gate -- and it has to be said explicitly by the artistic read,
    # not inferred from a high aesthetic score.
    deliberate = assessment.says_deliberate

    if portrait.expression_confidence < thresholds.min_expression_confidence:
        blocking.append(
            f"expression confidence {portrait.expression_confidence} is too low to promote"
        )

    if portrait.eyes_are_shut and not deliberate:
        blocking.append(
            "the subject's eyes are closed"
            if EyesState(portrait.eyes_state) is EyesState.CLOSED
            else "the subject's eyes are half closed"
        )

    if portrait.expression_is_bad and portrait.expression_confidence >= thresholds.min_expression_confidence:
        blocking.append(f"the expression is {portrait.expression.lower()}")

    if portrait.reads_as_unusable and portrait.expression_confidence >= thresholds.min_expression_confidence:
        blocking.append(
            f"the read says it would not publish this: expression quality "
            f"{portrait.expression_quality}, publishability {portrait.portrait_publishability}"
        )

    if portrait.expression_quality < thresholds.min_expression_quality:
        blocking.append(
            f"expression quality {portrait.expression_quality} is below "
            f"{thresholds.min_expression_quality}"
        )

    if portrait.face_sharpness < thresholds.min_face_sharpness and not deliberate:
        blocking.append(f"the face is soft ({portrait.face_sharpness})")

    if portrait.blink_probability > thresholds.max_blink_probability:
        blocking.append(f"blink probability {portrait.blink_probability}")
    if portrait.grimace_probability > thresholds.max_grimace_probability:
        blocking.append(f"grimace probability {portrait.grimace_probability}")

    return blocking


def portrait_verdict(
    assessment: ArtisticAssessment,
    *,
    thresholds: HeroThresholds = DEFAULT_THRESHOLDS,
    is_portrait: bool = False,
) -> tuple[str, str]:
    """For a confidently bad portrait, say so; for an unclear one, ask.

    Returns (verdict, reason) where verdict is `reject`, `review` or `keep`.
    The distinction that matters: a *confident* bad expression is a decision the
    tool can make, and an *uncertain* one is the only kind worth a person's
    time.
    """
    portrait = assessment.portrait
    if not (portrait and portrait.applies):
        return "keep", ""
    if not face_gates_the_frame(portrait, is_portrait=is_portrait, thresholds=thresholds):
        return "keep", "the face is incidental to the frame"

    # Writing a photograph off is the expensive direction, so it takes more than
    # a label. `hero_blockers` will decline to *promote* a frame on the strength
    # of the word AWKWARD alone -- abstaining from a promotion costs nothing --
    # but rejecting one needs the model's own numbers to agree that the picture
    # is unusable.
    #
    # The frame that set this line: a portrait labelled AWKWARD with confidence
    # 76, expression quality 47 and publishability 51, described as "not
    # especially flattering". That is a judgement about flattery, and it was
    # being treated as the same finding as a subject caught mid-blink.
    confident = portrait.expression_confidence >= thresholds.min_expression_confidence
    if confident and (portrait.eyes_are_shut or portrait.reads_as_unusable):
        if portrait.eyes_are_shut:
            detail = (
                "eyes closed"
                if EyesState(portrait.eyes_state) is EyesState.CLOSED
                else "eyes half closed"
            )
        else:
            detail = (
                f"{portrait.expression.lower()}: expression quality "
                f"{portrait.expression_quality}, publishability "
                f"{portrait.portrait_publishability}"
            )
        return "reject", f"confident bad expression: {detail}"

    if not confident and (portrait.expression_is_bad or portrait.expression_is_uncertain):
        return "review", (
            f"the expression may be {portrait.expression.lower()} but confidence is only "
            f"{portrait.expression_confidence}: this changes whether the frame is kept"
        )
    return "keep", ""


# --- crops --------------------------------------------------------------------


def face_crops(image: Image.Image, *, max_px: int = 640) -> list[tuple[str, Image.Image]]:
    """The views the model needs to judge an expression.

    Full frame, then the region a portrait's face occupies, then a wider
    head-and-shoulders context. Padding is generous on purpose: a tight crop of
    a face removes the pose and the gaze direction, which are half of what makes
    an expression readable.
    """
    views: list[tuple[str, Image.Image]] = []
    full = image.copy()
    full.thumbnail((max_px, max_px), Image.LANCZOS)
    views.append(("full frame", full))

    box = detect_primary_face(image)
    width, height = image.size
    if box is None:
        # No detector, or nothing found. A portrait's face sits in the upper
        # middle; that is a weak prior, and `face_source` records that it was
        # used rather than a measurement.
        box = (int(width * 0.28), int(height * 0.08), int(width * 0.72), int(height * 0.55))

    for name, pad in (("face", 0.35), ("head and shoulders", 1.1)):
        crop = _padded(image, box, pad)
        if crop is None:
            continue
        crop.thumbnail((max_px, max_px), Image.LANCZOS)
        views.append((name, crop))
    return views


def _padded(image: Image.Image, box: tuple[int, int, int, int], pad: float) -> Image.Image | None:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return None
    grow_x, grow_y = int(width * pad), int(height * pad)
    return image.crop(
        (
            max(0, left - grow_x),
            max(0, top - grow_y),
            min(image.size[0], right + grow_x),
            min(image.size[1], bottom + grow_y),
        )
    )


def detector_available() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def detect_primary_face(image: Image.Image) -> tuple[int, int, int, int] | None:
    """The largest face, when a detector is installed. None otherwise.

    OpenCV is an optional dependency and a heavy one, so its absence is a normal
    state rather than an error. When it is missing the model reports the
    geometry instead, which it is better at anyway.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    try:
        grey = np.asarray(image.convert("L"))
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)
    except Exception as e:  # pragma: no cover - depends on the local install
        logger.debug("Face detection unavailable: %s", e)
        return None

    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(x + w), int(y + h))


def cache_key(checksum: str, model: str) -> str:
    """Stage 3 results are keyed apart from Stage 2, and by prompt version.

    A valid Stage 2 entry must never suppress missing Stage 3 work, which is
    what a shared key would do.
    """
    return f"stage3:{checksum}:{model}:{PROMPT_VERSION}:{SCHEMA_VERSION}"


def summarise(assessment: ArtisticAssessment) -> str:
    """One line for the console."""
    if not assessment.completed:
        return f"artistic: {assessment.status}" + (
            f" ({assessment.skip_reason})" if assessment.skip_reason else ""
        )
    parts = [f"{name.split('_')[0]} {assessment.score(name)}" for name in ARTISTIC_FIELDS[:4]]
    line = "artistic: " + ", ".join(parts) + f", confidence {assessment.artistic_confidence}"
    if assessment.portrait and assessment.portrait.applies:
        line += (
            f" | face: {assessment.portrait.expression.lower()}, "
            f"eyes {assessment.portrait.eyes_state.lower()}, "
            f"quality {assessment.portrait.expression_quality}"
        )
    return line


def to_json(assessment: ArtisticAssessment) -> str:
    return json.dumps(assessment.to_dict(), ensure_ascii=False, indent=2)
