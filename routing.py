"""Where each frame goes, and why.

This module is pure logic: it takes parsed model output plus Stage 0
measurements and returns a destination. No API calls, no filesystem.

Two things here are load-bearing enough to say out loud.

**The stock block is enforced in code, not asked for in a prompt.** A frame
with a recognisable face or a logo cannot reach `10_stock_commercial` no matter
what the model returned in its `destination` field. Releases are a legal
requirement for commercial stock, and a model that gets this wrong once earns a
batch of rejections. The model's own `destination` is kept as
`model_destination` for comparison and is never trusted.

**Flagship is a collection-level decision.** "Top by axis_b within genre" is a
rank, not a threshold, so it cannot be computed from one frame in isolation --
`assign_destinations` needs the whole population. Everything else routes
per-frame.

Thresholds are placeholders until `--bench` produces real ones from the
150-frame labelled set. They live in `RoutingConfig` precisely so that swapping
them is a config change, not a code change.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

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


class Destination(StrEnum):
    DELETE_CANDIDATES = "00_delete_candidates"
    STOCK_COMMERCIAL = "10_stock_commercial"
    EDITORIAL = "20_editorial"
    FLAGSHIP = "30_flagship"
    VIDEO_STOCK = "40_video_stock"
    HOLD = "90_hold"


AXIS_MIN, AXIS_MAX = 0, 100
NOTE_MAX_WORDS = 12


@dataclass
class RoutingConfig:
    """Placeholders. Replace from the `--bench` winner before trusting output."""

    axis_a_stock: int = 70
    axis_c_editorial: int = 65
    # Flagship is a rank within genre, not a cutoff: this fraction of each
    # genre's frames, and never more than flagship_max_per_genre.
    flagship_top_fraction: float = 0.10
    flagship_max_per_genre: int = 25
    flagship_min_axis_b: int = 60


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
    logos: bool
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


@dataclass
class Routed:
    assessment: Assessment
    destination: Destination
    reason: str

    @property
    def filename(self) -> str:
        return self.assessment.filename


# --- the blocking rule ------------------------------------------------------


def blocks_commercial_stock(assessment: Assessment) -> bool:
    """A release is required for either, so neither may be sold as stock."""
    return assessment.faces or assessment.logos


# --- routing ----------------------------------------------------------------


def assign_destinations(
    assessments: list[Assessment],
    config: RoutingConfig | None = None,
) -> list[Routed]:
    """Route every frame. Order of the returned list matches the input."""
    config = config or RoutingConfig()
    flagship = _pick_flagship(assessments, config)
    return [_route_one(a, config, a.filename in flagship) for a in assessments]


def _route_one(assessment: Assessment, config: RoutingConfig, is_flagship: bool) -> Routed:
    def out(destination: Destination, reason: str) -> Routed:
        return Routed(assessment=assessment, destination=destination, reason=reason)

    # 1. Nothing survives a technical reject or an unrecoverable frame.
    if assessment.technically_rejected:
        return out(
            Destination.DELETE_CANDIDATES,
            "; ".join(assessment.technically_rejected_for),
        )
    if assessment.recover is Recover.HOPELESS:
        return out(Destination.DELETE_CANDIDATES, "model marked recovery hopeless")

    # 2. Video has its own bucket regardless of axes.
    if assessment.is_video:
        return out(Destination.VIDEO_STOCK, "video clip")

    # 3. Flagship: ranked within its genre, so it outranks the value buckets.
    if is_flagship:
        return out(
            Destination.FLAGSHIP,
            f"top axis_b within {assessment.genre.value} (b={assessment.axis_b})",
        )

    # 4. Commercial stock -- only if nothing needs a release.
    if assessment.axis_a >= config.axis_a_stock:
        if blocks_commercial_stock(assessment):
            return out(
                Destination.EDITORIAL,
                _blocked_reason(assessment),
            )
        return out(
            Destination.STOCK_COMMERCIAL,
            f"axis_a {assessment.axis_a} >= {config.axis_a_stock}, no release needed",
        )

    # 5. Editorial: documentary value, or anything needing a release.
    if assessment.axis_c >= config.axis_c_editorial:
        return out(
            Destination.EDITORIAL,
            f"axis_c {assessment.axis_c} >= {config.axis_c_editorial}",
        )
    if blocks_commercial_stock(assessment):
        return out(Destination.EDITORIAL, _blocked_reason(assessment))

    return out(Destination.HOLD, "below every threshold")


def _blocked_reason(assessment: Assessment) -> str:
    present = [n for n, v in (("faces", assessment.faces), ("logos", assessment.logos)) if v]
    return f"{' and '.join(present)} present -- release required, commercial stock blocked"


def _pick_flagship(assessments: list[Assessment], config: RoutingConfig) -> set[str]:
    """Top axis_b within each genre.

    Ranking inside the genre is the point: street loses to landscape on any
    shared scale because landscape is tidier, so a global cut would empty the
    reportage and street buckets and fill flagship with sunsets.
    """
    by_genre: dict[Genre, list[Assessment]] = defaultdict(list)
    for a in assessments:
        if a.is_video or a.technically_rejected or a.recover is Recover.HOPELESS:
            continue
        if a.axis_b >= config.flagship_min_axis_b:
            by_genre[a.genre].append(a)

    chosen: set[str] = set()
    for genre_frames in by_genre.values():
        quota = min(
            config.flagship_max_per_genre,
            max(1, math.ceil(len(genre_frames) * config.flagship_top_fraction)),
        )
        ranked = sorted(genre_frames, key=lambda a: (-a.axis_b, a.filename))
        chosen.update(a.filename for a in ranked[:quota])
    return chosen


# --- parsing the model's JSON ----------------------------------------------


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
    required = {"genre", "axis_a", "axis_b", "axis_c", "recover", "faces", "logos"}
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
        logos=_as_bool(payload["logos"]),
        note=_trim_note(payload.get("note", "")),
        intended_frame=_as_bool(payload.get("intended_frame", True)),
        subject_strength=_clamp_0_100(payload.get("subject_strength", 50), 50),
        accidental_probability=_clamp_0_100(payload.get("accidental_probability", 0), 0),
        dead_moment_probability=_clamp_0_100(payload.get("dead_moment_probability", 0), 0),
        model_destination=payload.get("destination"),
        is_video=is_video,
        technically_rejected_for=list(technically_rejected_for or []),
    )


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


def _as_bool(value) -> bool:
    """Absent or unparseable reads as True for faces/logos -- fail safe.

    Callers only use this for the two release flags, where guessing "no face"
    on a frame the model was unsure about is the expensive direction to be
    wrong in.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "no", "0", ""}
    if value is None:
        return True
    return bool(value)


def _trim_note(note) -> str:
    words = str(note or "").split()
    return " ".join(words[:NOTE_MAX_WORDS])


# --- reporting --------------------------------------------------------------


def summarise(routed: list[Routed]) -> dict[str, int]:
    counts: dict[str, int] = {d.value: 0 for d in Destination}
    for r in routed:
        counts[r.destination.value] += 1
    return counts
