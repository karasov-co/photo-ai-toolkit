"""Ten numbers per asset, and the class that follows from them.

The rule this module exists to enforce: **current quality and realistic
potential are different questions and must never be collapsed into one score.**
A flat, dark, tilted RAW and a sharp, bright, correctly-exposed JPEG of nothing
can have the same overall impression and opposite futures. So each dimension is
computed from its own evidence and stored separately, and only the last one --
`routing_score` -- is allowed to be a blend.

    A current_quality        what the unedited file looks like now
    B recoverability         how safely normal editing can move it
    C post_edit_potential    what it becomes after a realistic edit
    D aesthetic_potential    whether the result is worth looking at
    E stock_potential        whether it is sellable, findable and legal
    F portfolio_potential    whether it represents the photographer's best
    G legal_readiness        releases, marks, identifiable people
    H uniqueness             against the rest of this collection
    I confidence             how much of the above is actually evidenced
    J routing_score          the single blend, from the calibration profile

Two asymmetries are load-bearing:

**An unrecoverable issue caps potential.** Not penalises -- caps. The preview
search will find genuine uplift on an out-of-focus frame, because raising its
exposure genuinely does improve it, and without a hard ceiling that uplift
promotes a photograph whose subject will never be sharp.

**Faces or logos physically cannot reach commercial stock.** Not a weighting,
not a recommendation: a branch that runs before the thresholds do. Both need a
release, and submitting them without one earns rejections in batches. The model
is asked for these two flags and is *not* trusted to act on them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

from calibration import CalibrationProfile
from issues import Fixability, IssueCode, IssueSet

SCHEMA_VERSION = 2
ANALYZER_VERSION = "2.0.0"


class RouteClass(StrEnum):
    TRASH = "trash"
    REVIEW = "review"
    # Keep, but do nothing with. The class that exists so that "not worth
    # selling" never has to be expressed as "worth destroying".
    ARCHIVE_ONLY = "archive_only"
    # A near-identical sibling scored higher. A proposal for a person to compare,
    # never a deletion: the difference between two takes is usually a gesture or
    # an expression, and nothing local can see either.
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    STOCK_STANDARD = "stock_standard"
    STOCK_STRONG = "stock_strong"
    FLAGSHIP = "flagship"


class AssetTag(StrEnum):
    """Secondary destinations. An asset may carry several at once."""

    PORTFOLIO = "portfolio"
    COMMERCIAL_OK = "commercial_ok"
    EDITORIAL_ONLY = "editorial_only"
    NEEDS_MODEL_RELEASE = "needs_model_release"
    NEEDS_PROPERTY_RELEASE = "needs_property_release"
    LEGAL_REVIEW = "legal_review"
    ARCHIVE_ONLY = "archive_only"
    # A near-identical sibling scored higher. A proposal for a person to compare,
    # never a deletion: the difference between two takes is usually a gesture or
    # an expression, and nothing local can see either.
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    LOW_STOCK_DEMAND = "low_stock_demand"
    BEST_IN_CLUSTER = "best_in_cluster"
    WEAKER_DUPLICATE = "weaker_duplicate"


class Route(StrEnum):
    COMMERCIAL = "commercial"
    EDITORIAL = "editorial"


# Issues that end the conversation regardless of anything else measured.
FATAL_CODES = frozenset(
    {
        IssueCode.CORRUPT_FILE,
        IssueCode.ENCODING_CORRUPTION,
        IssueCode.EMPTY_FRAME,
        IssueCode.NO_USABLE_SEGMENT,
        IssueCode.INSUFFICIENT_RESOLUTION,
        IssueCode.LEGAL_BLOCKER,
    }
)

# Deliberately NOT fatal, and each for its own reason:
#
#   WEAKER_DUPLICATE   two takes of one moment. Which is better depends on an
#                      expression or a gesture, and the local measurement that
#                      picks a winner cannot see either.
#   SHORT_CLIP         below a marketplace's duration floor, which is a
#                      submission rule and not a property of the footage.
#   UNUSABLE_DURATION  a fragment of a second. Still routed for review rather
#                      than destroyed, because the frame count is the only thing
#                      wrong with it.

# One unrecoverable issue ceilings potential here; each further one lowers it.
# Set below every profile's `trash_potential` (30 for photos, 24 for video) so
# that a single confirmed unrecoverable problem is decisive on both media types.
# At 28 it sat above the video floor, and clips with unusable camera shake were
# landing in review rather than trash.
BLOCKER_CAP = 22.0
BLOCKER_CAP_STEP = 8.0
BLOCKER_CAP_FLOOR = 4.0


@dataclass
class Semantic:
    """What the vision model said. `present=False` means it never ran.

    Defaults are deliberately pessimistic on the two release flags: a frame
    nobody looked at is assumed to need a release, because the cost of being
    wrong in that direction is a rejected submission rather than a missed sale.
    """

    present: bool = False
    genre: str = "other"
    secondary_genres: list[str] = field(default_factory=list)
    axis_a: int = 50
    axis_b: int = 50
    axis_c: int = 50
    recover: str = "moderate"
    faces: bool = True
    logos: bool = True
    identifiable_people: bool = True
    recognizable_property: bool = False
    people_count: int = 0
    description: str = ""
    concepts: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    copy_space: str = "unknown"
    shot_type: str = "unknown"
    camera_angle: str = "unknown"
    note: str = ""

    # Whether the frame was taken on purpose and whether anything is in it.
    # Benign defaults, for the same reason the release flags have pessimistic
    # ones: each default is set to the side where being wrong is cheap. Assuming
    # a release is needed costs a sale; assuming a frame was an accident costs
    # the photograph.
    intended_frame: bool = True
    subject_strength: int = 50
    accidental_probability: int = 0
    dead_moment_probability: int = 0

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> Semantic:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (payload or {}).items() if k in known})

    @property
    def needs_release(self) -> bool:
        return self.faces or self.identifiable_people or self.recognizable_property

    @property
    def blocks_commercial(self) -> bool:
        """The hard rule. A release is required for either, so neither sells."""
        return self.faces or self.logos


UNKNOWN_AXIS = 50


def rank_to_percentile(rank: int, group_size: int) -> int:
    """A within-group rank (1 = best) onto the 0-100 scale (100 = best).

    These two conventions are opposite, and mixing them silently inverts the
    meaning of every axis. Stage 2 is *asked* for ranks because an absolute
    scale collapses; scoring works in percentiles because thresholds need one.
    The conversion belongs in exactly one place, which is here.
    """
    if group_size < 2:
        return 100
    clamped = max(1, min(group_size, int(rank)))
    return int(round(100 * (group_size - clamped) / (group_size - 1)))


def semantic_from_assessment(assessment, *, group_size: int | None = None) -> Semantic:
    """Adapter from the Stage 2 `routing.Assessment` this repo already produces.

    `group_size` is required to interpret the axes, because they arrive as
    ranks. Without it the axes are set to "unknown" rather than to the raw rank:
    a rank of 1 means *best*, and copying it into a field where 1 means *worst*
    turned the strongest frame of a group into the weakest. That happened in the
    fallback path whenever a group failed to aggregate -- a timeout or a
    malformed reply -- which is precisely when nobody is watching.
    """
    if group_size is None:
        axis_a = axis_b = axis_c = UNKNOWN_AXIS
    else:
        axis_a = rank_to_percentile(assessment.axis_a, group_size)
        axis_b = rank_to_percentile(assessment.axis_b, group_size)
        axis_c = rank_to_percentile(assessment.axis_c, group_size)

    return Semantic(
        present=True,
        genre=str(assessment.genre),
        axis_a=axis_a,
        axis_b=axis_b,
        axis_c=axis_c,
        recover=str(assessment.recover),
        faces=bool(assessment.faces),
        logos=bool(assessment.logos),
        identifiable_people=bool(assessment.faces),
        intended_frame=bool(getattr(assessment, "intended_frame", True)),
        subject_strength=int(getattr(assessment, "subject_strength", 50)),
        accidental_probability=int(getattr(assessment, "accidental_probability", 0)),
        dead_moment_probability=int(getattr(assessment, "dead_moment_probability", 0)),
        note=str(assessment.note or ""),
    )


@dataclass
class AssetScores:
    current_quality: int = 0
    recoverability: int = 0
    post_edit_potential: int = 0
    aesthetic_potential: int = 0
    stock_potential: int = 0
    portfolio_potential: int = 0
    legal_readiness: int = 0
    uniqueness: int = 0
    confidence: int = 0
    routing_score: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def potential_gain(self) -> int:
        """The number the whole design exists to expose."""
        return self.post_edit_potential - self.current_quality


@dataclass
class ScoreInput:
    """Everything scoring needs, already measured. No I/O happens in here."""

    asset_id: str
    filename: str
    kind: str = "photo"
    technical_quality: float = 0.0
    uplift: float = 0.0
    issues: IssueSet = field(default_factory=IssueSet)
    semantic: Semantic = field(default_factory=Semantic)
    is_raw: bool = False
    megapixels: float = 0.0
    cluster_size: int = 1
    is_best_in_cluster: bool = True
    cluster_similarity: float = 0.0
    cluster_margin: float = 0.0
    evidence_completeness: float = 1.0
    # Without a content check, a duplicate decision rests on sharpness alone --
    # which cannot see the expression, the gesture or the moment that usually
    # decides which of two takes is the keeper.
    semantic_ran: bool = False
    # The artistic read. `None` means Stage 3 has not been attached, which is
    # itself a blocker: a frame cannot become the best thing in a shoot because
    # the analysis that would have judged it never finished.
    artistic: object | None = None


@dataclass
class Reason:
    """A stable key plus its numbers, so the same reason renders in any language.

    The English sentence stays in `text` for the JSON, where other software
    reads it; the console and the HTML render `key` through the catalogue. That
    is what stops a Russian run from printing a paragraph of English next to a
    Russian summary.
    """

    key: str
    params: dict = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "params": self.params, "text": self.text}

    def localise(self, language: str = "en") -> str:
        from i18n import t as translate

        rendered = translate(self.key, language, **self.params)
        return self.text or rendered if rendered == self.key else rendered


@dataclass
class ScoredAsset:
    asset_id: str
    filename: str
    kind: str
    scores: AssetScores
    route_class: RouteClass
    route: Route
    tags: list[AssetTag] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    reason_keys: list[Reason] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    analyzer_version: str = ANALYZER_VERSION

    @property
    def is_commercial(self) -> bool:
        return self.route is Route.COMMERCIAL


# --- the individual dimensions ----------------------------------------------


def _clamp(value: float) -> int:
    return int(max(0, min(100, round(value))))


def current_quality_score(technical_quality: float) -> int:
    """A: the file as it sits on the card. Deterministic, no model involved."""
    return _clamp(technical_quality)


# Issues that describe this asset's relationship to *other* assets rather than
# the asset itself. They are detected, reported and routed on -- and they are
# excluded from every intrinsic score, because a photograph does not become
# harder to edit when a similar one is imported next to it. Leaving
# WEAKER_DUPLICATE in the recoverability arithmetic made an unchanged
# photograph's final score move by two points when a new batch arrived.
COLLECTION_RELATIVE_CODES = frozenset({IssueCode.WEAKER_DUPLICATE})


def intrinsic(found: IssueSet) -> IssueSet:
    """Only the problems that are properties of this file."""
    return IssueSet(
        issues=[i for i in found if i.code not in COLLECTION_RELATIVE_CODES]
    )


def recoverability_score(found: IssueSet, *, is_raw: bool) -> int:
    """B: how much room there is to move, and how safely.

    RAW starts higher because it genuinely holds one to two stops the rendered
    preview cannot show. That is a fact about the format, not a preference.
    """
    found = intrinsic(found)
    score = 100.0 if is_raw else 78.0
    score -= 11.0 * len(found.partial)
    score -= 34.0 * len(found.unrecoverable)
    return _clamp(score)


def post_edit_potential_score(
    current: int,
    uplift: float,
    found: IssueSet,
    *,
    recoverability: int,
) -> int:
    """C: what a realistic edit actually gets you.

    The cap is the important half. Uplift is measured by actually applying
    plausible edits and re-scoring, so it is real -- but "real" and "sufficient"
    are different things, and a sharp-looking histogram on a frame that missed
    focus is exactly the trap this prevents.
    """
    projected = current + uplift * (0.55 + 0.45 * recoverability / 100.0)

    blockers = found.unrecoverable
    if blockers:
        cap = max(BLOCKER_CAP_FLOOR, BLOCKER_CAP - BLOCKER_CAP_STEP * (len(blockers) - 1))
        projected = min(projected, cap)

    return _clamp(projected)


UNKNOWN_LEGAL_READINESS = 50


def legal_readiness_score(semantic: Semantic) -> int:
    """G: how close this is to being publishable without paperwork.

    "Nobody has looked" and "there is a face and a logo in it" are different
    states and must not produce the same number. The pessimistic defaults on
    `Semantic` exist to make `blocks_commercial` fail safe -- an unchecked frame
    is never sold as commercial stock -- but scoring an unchecked frame as
    though both problems were confirmed drags every other dimension down with
    it and makes the whole stock axis meaningless before the vision pass runs.

    So an unchecked frame sits in the middle, and the doubt is carried by
    `confidence`, which is the dimension that exists to carry doubt.
    """
    if not semantic.present:
        return UNKNOWN_LEGAL_READINESS

    score = 100.0
    if semantic.faces or semantic.identifiable_people:
        score -= 45.0
    if semantic.logos:
        score -= 40.0
    if semantic.recognizable_property:
        score -= 20.0
    return _clamp(score)


def uniqueness_score(*, cluster_size: int, is_best: bool, similarity: float) -> int:
    """H: against this collection, not against photography in general."""
    if not is_best:
        return _clamp(12.0 - similarity * 10.0)
    if cluster_size <= 1:
        return 100
    return _clamp(100.0 - 9.0 * (cluster_size - 1) - similarity * 12.0)


def aesthetic_potential_score(semantic: Semantic, potential: int) -> int:
    """D: worth looking at, once edited.

    Leans on the model's unrepeatability axis rather than its commercial one --
    the frames that break the rules score low on axis_a by construction, and
    those are frequently the interesting ones.
    """
    if not semantic.present:
        return _clamp(potential * 0.75)
    return _clamp(0.55 * semantic.axis_b + 0.20 * semantic.axis_c + 0.25 * potential)


def stock_potential_score(
    semantic: Semantic,
    potential: int,
    legal: int,
) -> int:
    """E: sellable, findable, publishable.

    Legal readiness multiplies rather than subtracts. A frame needing a release
    is not "slightly less sellable"; without the paperwork it is not sellable at
    all, and a subtraction lets a strong enough image climb back over the line.

    When no vision pass has run, the legal multiplier is a flat mild discount
    rather than the unknown-state 50. Compounding "we did not look" into both
    the base and the multiplier pushed every unanalysed frame under every stock
    threshold, which collapsed the offline mode into one class.
    """
    if semantic.present:
        base = 0.58 * semantic.axis_a + 0.42 * potential
        factor = 0.35 + 0.65 * legal / 100.0
    else:
        base = potential * 0.80
        factor = 0.85
    return _clamp(base * factor)


def portfolio_potential_score(semantic: Semantic, potential: int, aesthetic: int) -> int:
    """F: does this belong in the photographer's best work.

    Gated by potential, because a memorable frame that cannot be brought to a
    printable state is not a portfolio piece however good the moment was.
    """
    base = 0.62 * aesthetic + 0.38 * potential
    if semantic.present:
        base = 0.70 * base + 0.30 * semantic.axis_b
    gate = min(1.0, potential / 55.0)
    return _clamp(base * gate)


def confidence_score(
    *,
    semantic: Semantic,
    found: IssueSet,
    is_raw: bool,
    evidence_completeness: float,
) -> int:
    """I: how much of this assessment rests on evidence rather than defaults."""
    found = intrinsic(found)
    score = 100.0
    if not semantic.present:
        score -= 30.0
    if not is_raw:
        score -= 8.0
    score *= max(0.0, min(1.0, evidence_completeness))

    uncertain = [i for i in found if i.certainty < 0.8]
    score -= 5.0 * len(uncertain)

    if found.unrecoverable and found.fixable and not found.partial:
        # Signals pointing opposite ways: something is unrecoverable while
        # everything else reads as a routine edit.
        score -= 10.0
    return _clamp(score)


def routing_score(scores: AssetScores, profile: CalibrationProfile) -> int:
    """J: the one permitted blend, with the weights coming from data."""
    weights = profile.normalised_weights()
    values = scores.to_dict()
    return _clamp(sum(weights.get(k, 0.0) * values.get(k, 0) for k in weights))


# --- putting it together ----------------------------------------------------


def score(inp: ScoreInput, profile: CalibrationProfile) -> AssetScores:
    scores = AssetScores()
    scores.current_quality = current_quality_score(inp.technical_quality)
    scores.recoverability = recoverability_score(inp.issues, is_raw=inp.is_raw)
    scores.post_edit_potential = post_edit_potential_score(
        scores.current_quality,
        inp.uplift,
        inp.issues,
        recoverability=scores.recoverability,
    )
    scores.legal_readiness = legal_readiness_score(inp.semantic)
    scores.uniqueness = uniqueness_score(
        cluster_size=inp.cluster_size,
        is_best=inp.is_best_in_cluster,
        similarity=inp.cluster_similarity,
    )
    scores.aesthetic_potential = aesthetic_potential_score(inp.semantic, scores.post_edit_potential)
    scores.stock_potential = stock_potential_score(
        inp.semantic, scores.post_edit_potential, scores.legal_readiness
    )
    scores.portfolio_potential = portfolio_potential_score(
        inp.semantic, scores.post_edit_potential, scores.aesthetic_potential
    )
    scores.confidence = confidence_score(
        semantic=inp.semantic,
        found=inp.issues,
        is_raw=inp.is_raw,
        evidence_completeness=inp.evidence_completeness,
    )
    scores.routing_score = routing_score(scores, profile)
    return scores


def route_for(semantic: Semantic) -> Route:
    """The blocking rule, in one place so it cannot drift between call sites."""
    return Route.EDITORIAL if semantic.blocks_commercial else Route.COMMERCIAL


def classify(
    inp: ScoreInput,
    scores: AssetScores,
    profile: CalibrationProfile,
    *,
    flagship_selected: bool = False,
) -> ScoredAsset:
    """Scores plus issues to a class, with the reasons that produced it."""
    # The reason for the *class* has to lead. These strings are what a plan
    # prints next to a file it proposes to move, and "release status unchecked"
    # is not why a corrupt clip is being quarantined.
    class_reasons: list[str] = []
    reasons: list[str] = []
    keys: list[Reason] = []
    tags: list[AssetTag] = []
    semantic = inp.semantic
    route = route_for(semantic)

    if route is Route.EDITORIAL:
        tags.append(AssetTag.EDITORIAL_ONLY)
        if not semantic.present:
            reasons.append(
                "release status unchecked (no vision pass ran): commercial stock is "
                "blocked until a face and trademark check has actually been done"
            )
        else:
            present = [n for n, v in (("faces", semantic.faces), ("logos", semantic.logos)) if v]
            reasons.append(
                f"{' and '.join(present)} present: a release is required, "
                "so commercial stock is blocked in code"
            )
            if semantic.faces or semantic.identifiable_people:
                tags.append(AssetTag.NEEDS_MODEL_RELEASE)
            if semantic.logos:
                tags.append(AssetTag.LEGAL_REVIEW)
    else:
        tags.append(AssetTag.COMMERCIAL_OK)
    if semantic.recognizable_property:
        tags.append(AssetTag.NEEDS_PROPERTY_RELEASE)

    tags.append(AssetTag.BEST_IN_CLUSTER if inp.is_best_in_cluster else AssetTag.WEAKER_DUPLICATE)

    fatal = [i for i in inp.issues.unrecoverable if i.code in FATAL_CODES]
    route_class = _decide(inp, scores, profile, fatal, flagship_selected, class_reasons, keys)
    reasons = class_reasons + reasons

    if route_class is RouteClass.FLAGSHIP:
        tags.append(AssetTag.PORTFOLIO)
    is_stock = route_class in (RouteClass.STOCK_STANDARD, RouteClass.STOCK_STRONG)
    if is_stock and scores.stock_potential < profile.threshold("stock_strong"):
        tags.append(AssetTag.LOW_STOCK_DEMAND)
    if route_class is RouteClass.REVIEW and scores.post_edit_potential >= profile.threshold("trash_potential"):
        tags.append(AssetTag.ARCHIVE_ONLY)

    return ScoredAsset(
        asset_id=inp.asset_id,
        filename=inp.filename,
        kind=inp.kind,
        scores=scores,
        route_class=route_class,
        route=route,
        tags=tags,
        reasons=reasons,
        reason_keys=keys,
        strengths=_strengths(inp, scores),
    )


def _decide(
    inp: ScoreInput,
    scores: AssetScores,
    profile: CalibrationProfile,
    fatal: list,
    flagship_selected: bool,
    reasons: list[str],
    keys: list[Reason] | None = None,
) -> RouteClass:
    """Precedence, top to bottom. Order here *is* the policy."""
    if fatal:
        reasons.append("unrecoverable: " + "; ".join(i.describe() for i in fatal))
        _key(keys, "reason.unrecoverable", {"detail": "; ".join(i.describe() for i in fatal)}, reasons[-1])
        return RouteClass.TRASH

    # A near-identical sibling is a comparison for a person, not a deletion.
    # Sharpness picks the winner, and sharpness cannot see which take has the
    # better expression. Without a content check it is barely a signal at all.
    if not inp.is_best_in_cluster:
        # The number, taken from the measurement rather than parsed back out of
        # an English sentence -- the regex that did that picked the frame number
        # out of "cluster P1019370.JPG" and reported a margin of 1019370.
        margin = f"{inp.cluster_margin:.0f}" if inp.cluster_margin else "?"
        reasons.append(
            f"a sharper frame exists in this group ({margin} points higher); "
            + (
                "compare them by hand"
                if inp.semantic_ran
                else "no content check ran, so only sharpness separates them -- compare by hand"
            )
        )
        _key(
            keys,
            "reason.duplicate_candidate" if inp.semantic_ran else "reason.duplicate_unchecked",
            {"margin": margin},
            reasons[-1],
        )
        return RouteClass.DUPLICATE_CANDIDATE

    blockers = inp.issues.unrecoverable
    below_floor = scores.post_edit_potential < profile.threshold("trash_potential")

    # Confirmed blockers outrank low confidence. Both are checked before the
    # confidence gate because the blockers come from deterministic local
    # measurement, not from a model: not knowing what is *in* a frame does not
    # make an out-of-focus, pure-black frame recoverable. With the gate first, a
    # lens-cap shot scoring 0 on every dimension was routed to manual review.
    if blockers and below_floor:
        reasons.append(
            "realistic potential is capped by unrecoverable problems: "
            + "; ".join(i.describe() for i in blockers)
        )
        return RouteClass.TRASH

    if scores.confidence < profile.threshold("review_confidence"):
        reasons.append(
            f"confidence {scores.confidence} below {profile.threshold('review_confidence'):.0f}: "
            "a human should decide"
        )
        _key(
            keys, "reason.low_confidence",
            {"value": scores.confidence, "threshold": int(profile.threshold("review_confidence"))},
            reasons[-1],
        )
        return RouteClass.REVIEW

    if below_floor:
        # A low score is not evidence of anything. Underexposure, flat contrast,
        # softness, grain, an odd crop -- every one of these drags potential down
        # and not one of them proves the photograph is worthless. A real
        # reproduction: a dark frame with *only* fixable issues scored 16 and was
        # routed to trash purely because 16 < 30.
        #
        # Destroying a file needs a demonstrable fact about it, which is what the
        # unrecoverable-issue branch above requires. Everything else is kept.
        reasons.append(
            f"realistic post-edit potential {scores.post_edit_potential} is below "
            f"{profile.threshold('trash_potential'):.0f}, but nothing about this frame is "
            "unrecoverable: keeping it"
        )
        _key(
            keys, "reason.below_trash",
            {"value": scores.post_edit_potential,
             "threshold": int(profile.threshold("trash_potential"))},
            reasons[-1],
        )
        return RouteClass.ARCHIVE_ONLY

    if flagship_selected:
        blocking = hero_blockers(inp)
        if blocking:
            # The invariant. Ranking a frame top of its genre is necessary and
            # nowhere near sufficient: a portrait taken mid-blink can lead every
            # ranked axis, because not one of those axes is looking at the face.
            # An absent or failed artistic read blocks promotion for the same
            # reason -- the analysis that would have judged it did not finish.
            reasons.append("held back from flagship: " + "; ".join(blocking))
            _key(keys, "reason.hero_blocked", {"detail": "; ".join(blocking)}, reasons[-1])
            if scores.stock_potential >= profile.threshold("stock_strong"):
                return RouteClass.STOCK_STRONG
            if scores.stock_potential >= profile.threshold("stock_standard"):
                return RouteClass.STOCK_STANDARD
            return RouteClass.REVIEW

        reasons.append(
            f"portfolio potential {scores.portfolio_potential} clears the absolute floor, "
            "ranks near the top of its genre, and the artistic read raises no objection"
        )
        _key(keys, "reason.flagship", {"value": scores.portfolio_potential}, reasons[-1])
        return RouteClass.FLAGSHIP

    if scores.stock_potential >= profile.threshold("stock_strong"):
        reasons.append(f"stock potential {scores.stock_potential} is strong")
        _key(keys, "reason.stock_strong", {"value": scores.stock_potential}, reasons[-1])
        return RouteClass.STOCK_STRONG

    if scores.stock_potential >= profile.threshold("stock_standard"):
        reasons.append(f"stock potential {scores.stock_potential} is usable after the suggested edit")
        _key(keys, "reason.stock_standard", {"value": scores.stock_potential}, reasons[-1])
        return RouteClass.STOCK_STANDARD

    reasons.append(
        f"recoverable ({scores.post_edit_potential} potential) but below the stock floor "
        f"({profile.threshold('stock_standard'):.0f}): keep for the archive or decide by hand"
    )
    _key(
        keys, "reason.archive",
        {"value": scores.post_edit_potential,
         "threshold": int(profile.threshold("stock_standard"))},
        reasons[-1],
    )
    return RouteClass.REVIEW


def _key(keys: list[Reason] | None, name: str, params: dict, text: str) -> None:
    if keys is not None:
        keys.append(Reason(name, params, text))


def hero_blockers(inp: ScoreInput) -> list[str]:
    """Every reason this frame may not be promoted, from the artistic read.

    An absent assessment is a blocker rather than a neutral. That is the whole
    point: the previous behaviour promoted frames whose Stage 3 fields were all
    `null`, because nothing checked that they were not.
    """
    import stage3

    assessment = inp.artistic
    if assessment is None:
        return ["no artistic analysis is attached to this frame"]
    if not isinstance(assessment, stage3.ArtisticAssessment):
        return ["the artistic analysis is not in a form that can be trusted"]
    return stage3.hero_blockers(assessment)


def eligible_for_flagship(scores: AssetScores, profile: CalibrationProfile) -> bool:
    """The absolute half of the flagship test.

    Flagship is deliberately not "the top 5%": on a weak shoot the top 5% is
    still weak. An asset must clear a fixed bar first, and only then compete for
    a place.
    """
    return (
        scores.portfolio_potential >= profile.threshold("flagship_portfolio")
        and scores.post_edit_potential >= profile.threshold("flagship_potential_floor")
    )


def _strengths(inp: ScoreInput, scores: AssetScores) -> list[str]:
    """The positive half of the explanation, which is otherwise easy to omit."""
    out: list[str] = []
    if scores.potential_gain >= 8:
        out.append(f"editing realistically gains {scores.potential_gain} points")
    if inp.is_raw:
        out.append("RAW: full highlight and shadow latitude available")
    if scores.uniqueness >= 85:
        out.append("no near-duplicate in this collection")
    if inp.semantic.present:
        if inp.semantic.axis_b >= 70:
            out.append("hard to repeat: moment, light or subject will not return")
        if inp.semantic.axis_a >= 70:
            out.append("clean, legible composition with commercial usability")
        if inp.semantic.axis_c >= 70:
            out.append("documentary value: place, event or cultural context")
    if scores.legal_readiness >= 90:
        out.append("no release required")
    if not inp.issues.unrecoverable and not inp.issues.partial:
        out.append("every detected problem is a routine edit")
    return out


def explain(asset: ScoredAsset, found: IssueSet) -> dict:
    """One asset's full, human-readable account of itself."""
    return {
        "filename": asset.filename,
        "class": asset.route_class.value,
        "route": asset.route.value,
        "scores": asset.scores.to_dict(),
        "potential_gain": asset.scores.potential_gain,
        "strengths": asset.strengths,
        "reasons": asset.reasons,
        "problems": {
            "fixable": [i.describe() for i in found.of(Fixability.FIXABLE)],
            "partially_fixable": [i.describe() for i in found.of(Fixability.PARTIAL)],
            "unrecoverable": [i.describe() for i in found.of(Fixability.UNRECOVERABLE)],
        },
        "tags": [t.value for t in asset.tags],
    }
