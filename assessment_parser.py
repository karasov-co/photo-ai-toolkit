"""Parsing one Stage 2 reply into a checked object.

This was `routing.py`, which also decided where a frame belonged in an archive:
a `Destination` enum, a config, a flagship picker, a summariser. None of it had
a caller -- routing moved into `scoring` and then into `curation`, and the old
module stayed behind because deleting code that still imports cleanly requires
somebody to check. The parser was the only living part.

The one idea worth keeping from it was `Destination.EDITORIAL`: a frame that
cannot be sold as commercial stock is not thereby worthless, because editorial
is a market. That now lives in `curation.PhotoCategory.GOOD_EDITORIAL`, where it
affects what a photographer is told.

What is parsed here is a ranking, not a score. Unknown genres fall back rather
than failing a photograph; out-of-range ranks are clamped for the same reason; a
missing required key raises, because a reply that does not answer the question
is not a reply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# The model ranks within a group, so a rank is clamped to the group's range
# rather than rejected: a misnumbered frame is not worth discarding.
AXIS_MIN, AXIS_MAX = 0, 100
NOTE_MAX_WORDS = 12


# --- vocabulary the model is held to ----------------------------------------


class Genre(StrEnum):
    STREET = "street"
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    DETAIL = "detail"
    REPORTAGE = "reportage"
    NIGHT = "night"
    ARCHITECTURE = "architecture"
    OTHER = "other"


class Recover(StrEnum):
    EASY = "easy"
    MODERATE = "moderate"
    HOPELESS = "hopeless"




@dataclass
class Assessment:
    """One frame's parsed model output, plus what Stage 0 measured."""

    filename: str
    genre: Genre
    axis_a: int
    axis_b: int
    axis_c: int
    recover: Recover
    faces: bool
    # Split from a single `logos` flag. A registered mark dominating the frame
    # and a street full of shop signs are different facts with different
    # consequences, and collapsing them blocked ordinary street photography
    # from commercial use.
    brand_mark: bool = False
    signage_text: bool = False
    note: str = ""
    model_destination: str | None = None
    is_video: bool = False
    technically_rejected_for: list[str] = field(default_factory=list)

    # Whether this frame was taken on purpose, and whether anything is happening
    # in it. Defaults are deliberately benign: a frame nobody asked about is not
    # an accident and does not hold a dead moment. The asymmetry matters -- being
    # wrong here writes off a photograph, which is the expensive direction.
    intended_frame: bool = True
    subject_strength: int = 50
    accidental_probability: int = 0
    dead_moment_probability: int = 0

    @property
    def technically_rejected(self) -> bool:
        return bool(self.technically_rejected_for)



class AssessmentParseError(ValueError):
    pass


def parse_assessment(
    payload: dict,
    filename: str,
    *,
    is_video: bool = False,
    technically_rejected_for: list[str] | None = None,
) -> Assessment:
    """Turn one model JSON object into an Assessment, or raise.

    Unknown genres fall back to `other` rather than failing the frame -- a
    misspelled enum is not worth discarding a photograph over. Out-of-range
    axis values are clamped for the same reason. A missing required key is a
    real failure and does raise.
    """
    # `logos` is deliberately absent: a reply that splits it into brand_mark
    # and signage_text is the current shape, and an older reply that still
    # carries `logos` is read below for compatibility.
    required = {"genre", "axis_a", "axis_b", "axis_c", "recover", "faces"}
    missing = required - payload.keys()
    if missing:
        raise AssessmentParseError(f"{filename}: missing keys {sorted(missing)}")

    return Assessment(
        filename=filename,
        genre=_enum_or_default(Genre, payload["genre"], Genre.OTHER),
        axis_a=_clamp_axis(payload["axis_a"]),
        axis_b=_clamp_axis(payload["axis_b"]),
        axis_c=_clamp_axis(payload["axis_c"]),
        recover=_enum_or_default(Recover, payload["recover"], Recover.MODERATE),
        faces=_as_bool(payload["faces"]),
        brand_mark=_brand_mark_of(payload),
        signage_text=_as_bool(payload.get("signage_text", False), default=False),
        note=_trim_note(payload.get("note", "")),
        intended_frame=_as_bool(payload.get("intended_frame", True)),
        subject_strength=_clamp_0_100(payload.get("subject_strength", 50), 50),
        accidental_probability=_clamp_0_100(payload.get("accidental_probability", 0), 0),
        dead_moment_probability=_clamp_0_100(payload.get("dead_moment_probability", 0), 0),
        model_destination=payload.get("destination"),
        is_video=is_video,
        technically_rejected_for=list(technically_rejected_for or []),
    )


def _brand_mark_of(payload: dict) -> bool:
    """A dominant registered mark, from either the new field or an old reply.

    A cached or replayed reply from before the split carries `logos`, which
    conflated a Coca-Cola wall with a noodle-shop sign. Reading it as a brand
    mark is the conservative choice for that old data and costs nothing going
    forward, because new replies never contain the key.
    """
    if "brand_mark" in payload:
        return _as_bool(payload["brand_mark"], default=False)
    if "logos" in payload:
        return _as_bool(payload["logos"], default=False)
    return False


def _clamp_0_100(value, default: int) -> int:
    """A missing or unparseable answer becomes the benign default, not zero.

    Zero on `subject_strength` reads as "there is no subject", which is a claim.
    A model that did not answer has not made one.
    """
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _enum_or_default(enum_cls, value, default):
    try:
        return enum_cls(str(value).strip().lower())
    except (ValueError, AttributeError):
        return default


def _clamp_axis(value) -> int:
    try:
        return max(AXIS_MIN, min(AXIS_MAX, int(round(float(value)))))
    except (TypeError, ValueError):
        return AXIS_MIN


def _as_bool(value, *, default: bool = True) -> bool:
    """Parse a flag, with the caller choosing which way an absence falls.

    It used to default to True for everything, which was right for `faces` and
    wrong for a brand mark. The prompt already said "when in doubt answer true",
    so a model that was unsure said yes, and a model that said nothing at all
    was recorded as yes too -- two fail-safes stacked on the same question. The
    result was that ordinary street photographs, full of shop signs and no
    brands at all, were blocked from commercial use.

    `faces` keeps the pessimistic default: guessing "no face" on a frame nobody
    checked is the expensive direction. `brand_mark` does not: guessing "brand"
    on every unanswered frame removes an entire genre.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "no", "0"}:
            return False
        if lowered in {"true", "yes", "1"}:
            return True
        return default
    if value is None:
        return default
    return bool(value)


def _trim_note(note) -> str:
    words = str(note or "").split()
    return " ".join(words[:NOTE_MAX_WORDS])
