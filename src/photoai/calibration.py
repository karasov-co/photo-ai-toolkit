"""Weights and thresholds, kept as data so they can be changed without a deploy.

Two things live here, and they exist for the same reason: raw model output is
not trustworthy enough to hard-code decisions from.

**Profiles** carry the weights that combine the score dimensions and the
thresholds that turn the result into a class. They are versioned and separate
per media type, because a video's quality signals are not a photograph's and a
threshold tuned on stills routes clips wrongly. They are also separate per
priority -- a stock-first profile and a portfolio-first profile disagree about
what a good frame is, and both are right for their own purpose.

**Reclassification** is the reason the split matters at all. Routing reads only
the stored dimensions, never the pixels or the model, so changing a threshold
and re-running is instant and free. The expensive half of the pipeline --
decoding, measuring, and paying a vision model -- does not repeat.

The shipped numbers are starting points measured against a small archive, not
truth. `--bench` against a hand-labelled set is what turns them into truth, and
until that has been run they should be treated as provisional.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_SCHEMA_VERSION = 1

DEFAULT_WEIGHTS: dict[str, float] = {
    "post_edit_potential": 0.34,
    "aesthetic_potential": 0.20,
    "stock_potential": 0.18,
    "portfolio_potential": 0.12,
    "uniqueness": 0.10,
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Below this realistic post-edit potential, nothing is worth keeping.
    "trash_potential": 30.0,
    # Below this confidence, a human decides rather than the tool.
    "review_confidence": 55.0,
    "stock_standard": 45.0,
    "stock_strong": 68.0,
    # Flagship needs BOTH an absolute floor and a place near the top of its
    # genre. A pure top-N% rule promotes the best of a bad shoot; a pure
    # threshold promotes nothing at all from a modest one.
    "flagship_portfolio": 72.0,
    "flagship_potential_floor": 60.0,
    "flagship_top_fraction": 0.10,
    "flagship_max_per_genre": 25.0,
    "flagship_max_total": 60.0,
    # Two frames closer than this in perceptual-hash distance are the same
    # photograph for selection purposes.
    "duplicate_distance": 8.0,
    # How much better the winner of a cluster has to be before the others are
    # called weaker duplicates. Below this the two frames are a tie as far as
    # local measurement can tell, and the difference between them is
    # compositional -- which is a human's call, not a Laplacian's. Observed on a
    # real archive: sibling pairs at 42/38, 40/39 and 69/66, where only the
    # first is a real difference.
    "duplicate_margin": 5.0,
    # Minimum spread between flagship picks, as a fraction. Higher = more
    # aggressive diversity enforcement.
    "diversity_lambda": 0.65,
}

VIDEO_THRESHOLD_OVERRIDES: dict[str, float] = {
    # A clip is worth more work than a still: extracting it was expensive and
    # there is usually a usable segment inside a mediocre whole.
    "trash_potential": 24.0,
    "stock_standard": 40.0,
    "stock_strong": 64.0,
    "flagship_portfolio": 75.0,
}

VIDEO_WEIGHT_OVERRIDES: dict[str, float] = {
    "post_edit_potential": 0.40,
    "aesthetic_potential": 0.18,
    "stock_potential": 0.20,
    "portfolio_potential": 0.08,
    "uniqueness": 0.08,
}


@dataclass
class CalibrationProfile:
    name: str = "default-photo"
    version: str = "0.1.0-provisional"
    media: str = "photo"
    schema_version: int = PROFILE_SCHEMA_VERSION
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    notes: str = "Provisional. Not yet fitted against a labelled set."

    def threshold(self, key: str) -> float:
        return float(self.thresholds.get(key, DEFAULT_THRESHOLDS.get(key, 0.0)))

    def weight(self, key: str) -> float:
        return float(self.weights.get(key, 0.0))

    @property
    def is_fitted(self) -> bool:
        """Whether these numbers came from data or from an author's guess."""
        return "provisional" not in self.version.lower()

    def normalised_weights(self) -> dict[str, float]:
        total = sum(self.weights.values())
        if total <= 0:
            return dict(DEFAULT_WEIGHTS)
        return {k: v / total for k, v in self.weights.items()}

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict) -> CalibrationProfile:
        """Unknown keys are dropped, missing ones defaulted.

        A profile is a file a user edits by hand, so it will eventually contain
        a typo. Failing the whole run over one is worse than ignoring it.
        """
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in payload.items() if k in known}
        profile = cls(**clean)
        profile.weights = {**DEFAULT_WEIGHTS, **(profile.weights or {})}
        base = dict(DEFAULT_THRESHOLDS)
        if profile.media == "video":
            base.update(VIDEO_THRESHOLD_OVERRIDES)
        profile.thresholds = {**base, **(profile.thresholds or {})}
        return profile

    @classmethod
    def load(cls, path: Path) -> CalibrationProfile:
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read calibration profile %s (%s); using defaults", path, e)
            return cls()


def default_photo_profile() -> CalibrationProfile:
    return CalibrationProfile(name="default-photo", media="photo")


def default_video_profile() -> CalibrationProfile:
    return CalibrationProfile(
        name="default-video",
        media="video",
        weights={**DEFAULT_WEIGHTS, **VIDEO_WEIGHT_OVERRIDES},
        thresholds={**DEFAULT_THRESHOLDS, **VIDEO_THRESHOLD_OVERRIDES},
    )


def stock_first_profile() -> CalibrationProfile:
    """Commercial usability weighted above everything else."""
    return CalibrationProfile(
        name="stock-first",
        media="photo",
        weights={
            "post_edit_potential": 0.30,
            "aesthetic_potential": 0.10,
            "stock_potential": 0.36,
            "portfolio_potential": 0.04,
            "uniqueness": 0.08,
        },
        notes="Prioritises sellable, legible frames.",
    )


def portfolio_first_profile() -> CalibrationProfile:
    """The opposite trade: memorable over marketable."""
    return CalibrationProfile(
        name="portfolio-first",
        media="photo",
        weights={
            "post_edit_potential": 0.30,
            "aesthetic_potential": 0.30,
            "stock_potential": 0.05,
            "portfolio_potential": 0.25,
            "uniqueness": 0.08,
        },
        thresholds={**DEFAULT_THRESHOLDS, "flagship_portfolio": 66.0, "flagship_top_fraction": 0.15},
        notes="Prioritises distinctive work over sellable work.",
    )


BUILTIN_PROFILES = {
    "default-photo": default_photo_profile,
    "default-video": default_video_profile,
    "stock-first": stock_first_profile,
    "portfolio-first": portfolio_first_profile,
}


@dataclass
class CalibrationSet:
    """The photo and video profiles a run is using, resolved together."""

    photo: CalibrationProfile = field(default_factory=default_photo_profile)
    video: CalibrationProfile = field(default_factory=default_video_profile)

    def for_kind(self, kind: str) -> CalibrationProfile:
        return self.video if str(kind) == "video" else self.photo

    @property
    def fingerprint(self) -> str:
        """Goes in every report, so a result can be traced to what produced it."""
        return f"{self.photo.name}@{self.photo.version}+{self.video.name}@{self.video.version}"


def resolve(name: str | None = None, path: Path | None = None) -> CalibrationSet:
    """Pick the profiles for this run: a file, a built-in name, or the defaults."""
    if path is not None:
        loaded = CalibrationProfile.load(path)
        if loaded.media == "video":
            return CalibrationSet(photo=default_photo_profile(), video=loaded)
        return CalibrationSet(photo=loaded, video=default_video_profile())
    if name:
        factory = BUILTIN_PROFILES.get(name)
        if factory is None:
            logger.warning("Unknown calibration profile %r; using defaults", name)
            return CalibrationSet()
        chosen = factory()
        if chosen.media == "video":
            return CalibrationSet(video=chosen)
        return CalibrationSet(photo=chosen)
    return CalibrationSet()
