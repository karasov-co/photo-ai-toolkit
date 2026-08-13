"""One number for how good the photograph is, and the five piles it sorts into.

Everything before this module answers a *component* question: how clean is the
file, how much does an edit gain, would a marketplace take it, is the face doing
something. None of those is the question a photographer actually asks, which is
"is this any good, and what do I do with it".

The design has three load-bearing separations.

**Quality is not saleability.** A release, a crowd of strangers, a private
context or a subject nobody wants to license are facts about *distribution*.
They decide GOOD_STOCK versus GOOD_PERSONAL and they never touch the score. A
photograph of your family that no agency would ever accept can outscore
everything else in the run, and it must be able to. This is why
`legal_readiness` and `stock_potential` appear nowhere in the blend.

**Technical excellence cannot rescue a failed photograph.** Sharpness, exposure
and resolution are entry conditions, not merits. So defects that mean the
picture failed -- eyes shut, a frame taken by accident, a dead moment, no
subject, an unrecoverable fault -- apply a *ceiling* rather than a penalty. A
penalty can be outvoted by a big enough number somewhere else. A ceiling cannot,
which is the entire point of using one.

**Evidence can rescue an unconventional photograph.** The inverse asymmetry, and
just as deliberate. A frame that is dark, blurred, off-balance and technically
poor may be the best thing in the shoot, and the only thing that can say so is
Stage 3. So a strong artistic or documentary read applies a *floor* -- but only a
completed, confident one, and only against the soft signals. It never overrides
an observed fact like a closed eye.

Stage 3 is a real term in the arithmetic, not a veto bolted on the end.
`FinalScore.stage3_delta` reports exactly how many points it moved the frame,
positive or negative, and a run where that column is all zeros is a run where
Stage 3 did not happen.

Nothing here can delete anything. WEAK is a shelf, not a bin: the deletion path
still requires a demonstrable unrecoverable fault recorded as evidence, and no
category is admissible grounds. A photograph you disagree with the tool about
costs you a look in a folder; a photograph the tool destroys costs you the
photograph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

import stage3 as stage3_module
from calibration import CalibrationProfile
from scoring import AssetScores, ScoreInput, Semantic

SCHEMA_VERSION = 1


class PhotoCategory(StrEnum):
    """The five piles. `route_class` still exists and still drives the
    filesystem; this is the answer to "what is this photograph"."""

    TOP = "TOP"
    GOOD_STOCK = "GOOD_STOCK"
    GOOD_PERSONAL = "GOOD_PERSONAL"
    WEAK = "WEAK"
    NEEDS_DECISION = "NEEDS_DECISION"


# --- thresholds ---------------------------------------------------------------


@dataclass
class CurationThresholds:
    """Every number the categories depend on, named and in one place.

    None of these is fitted. They are set so the boundaries land where a person
    would put them on the archive this was built against, and they are data so
    that changing one does not require reading the code that uses it.
    """

    top: int = 85
    weak: int = 45
    # Without a completed artistic read a frame cannot reach TOP, however good
    # its numbers are. Expressed as a ceiling rather than a branch so that the
    # score itself carries the doubt.
    no_stage3_ceiling: int = 79
    # A TOP frame needs the read to be confident as well as complete.
    top_artistic_confidence: int = 60
    top_artistic_floor: int = 62
    # Ambiguity has to be genuine to be worth a person's time.
    decision_uncertainty: int = 70
    decision_band: int = 6


DEFAULT_THRESHOLDS = CurationThresholds()


# The eight Stage 3 dimensions, weighted for one question: is this photograph
# worth keeping. `conventional_beauty` is deliberately the smallest -- it is
# recorded because it is informative, and kept small because prettiness is the
# thing a model reaches for when it has nothing else to say. `documentary_
# significance` is small here only because it is also its own component below,
# and would otherwise be counted twice.
ARTISTIC_WEIGHTS: dict[str, float] = {
    "emotional_resonance": 0.22,
    "moment_specificity": 0.18,
    "distinctiveness": 0.16,
    "formal_coherence": 0.15,
    "visual_tension": 0.12,
    "narrative_openness": 0.09,
    "documentary_significance": 0.05,
    "conventional_beauty": 0.03,
}

# With a completed artistic read.
WEIGHTS_WITH_STAGE3: dict[str, float] = {
    "technical": 0.22,
    "artistic": 0.34,
    "semantic": 0.16,
    "portrait": 0.10,
    "documentary": 0.08,
    "scene": 0.10,
}

# Without one. Technical and semantic absorb the missing weight, and the
# `no_stage3_ceiling` stops the result being mistaken for a judged frame.
WEIGHTS_WITHOUT_STAGE3: dict[str, float] = {
    "technical": 0.42,
    "semantic": 0.34,
    "documentary": 0.10,
    "scene": 0.14,
}


# --- defects that cap ---------------------------------------------------------


class DefectCode(StrEnum):
    BAD_EXPRESSION = "bad_expression"
    EYES_CLOSED = "eyes_closed"
    ACCIDENTAL_FRAME = "accidental_frame"
    DEAD_MOMENT = "dead_moment"
    NO_SUBJECT = "no_subject"
    INFERIOR_DUPLICATE = "inferior_duplicate"
    UNRECOVERABLE = "unrecoverable"


@dataclass
class Defect:
    """A reason the photograph failed, and the ceiling that follows.

    `vetoable` separates the two kinds. An observed fact -- the eyes are shut,
    the file will not decode -- stands whatever else is true. A probabilistic
    read -- "this looks like a dead moment" -- is a guess, and a confident
    artistic read that says otherwise is better evidence than the guess.
    """

    code: str
    detail: str
    ceiling: int
    vetoable: bool = False
    vetoed_by: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Ceilings. Each sits below the WEAK threshold so that the defect decides the
# category on its own, and they differ from each other so that the score still
# ranks failures against each other -- an accident is further from keepable than
# a duplicate that lost by four points.
CEILINGS: dict[str, int] = {
    DefectCode.UNRECOVERABLE.value: 30,
    DefectCode.ACCIDENTAL_FRAME.value: 22,
    DefectCode.NO_SUBJECT.value: 28,
    DefectCode.EYES_CLOSED.value: 32,
    DefectCode.BAD_EXPRESSION.value: 34,
    DefectCode.DEAD_MOMENT.value: 38,
    DefectCode.INFERIOR_DUPLICATE.value: 42,
}

# Confidence a soft semantic signal needs before it is allowed to write off a
# photograph. Below this it is an opinion, and it is recorded but does not cap.
ACCIDENTAL_CERTAINTY = 65
DEAD_MOMENT_CERTAINTY = 65
WEAK_SUBJECT = 25


def critical_defects(
    inp: ScoreInput,
    artistic: stage3_module.ArtisticAssessment | None,
) -> list[Defect]:
    """Everything about this frame that means it failed as a photograph.

    Order is severity, worst first, because the caller reports the first one.
    """
    found: list[Defect] = []
    semantic = inp.semantic

    if inp.issues.unrecoverable:
        found.append(
            Defect(
                DefectCode.UNRECOVERABLE.value,
                "; ".join(i.describe() for i in inp.issues.unrecoverable),
                CEILINGS[DefectCode.UNRECOVERABLE.value],
            )
        )

    # Only from a semantic pass that ran. An unchecked frame is not an accident,
    # and the defaults on `Semantic` are set so this branch cannot fire on one.
    if semantic.present:
        accidental = not semantic.intended_frame or (
            semantic.accidental_probability >= ACCIDENTAL_CERTAINTY
        )
        if accidental:
            found.append(
                Defect(
                    DefectCode.ACCIDENTAL_FRAME.value,
                    f"read as an accidental frame ({semantic.accidental_probability}% likely, "
                    f"intended={semantic.intended_frame})",
                    CEILINGS[DefectCode.ACCIDENTAL_FRAME.value],
                    vetoable=True,
                )
            )
        if semantic.subject_strength <= WEAK_SUBJECT:
            found.append(
                Defect(
                    DefectCode.NO_SUBJECT.value,
                    f"no legible subject and no crop that would create one "
                    f"(subject strength {semantic.subject_strength})",
                    CEILINGS[DefectCode.NO_SUBJECT.value],
                    vetoable=True,
                )
            )
        if semantic.dead_moment_probability >= DEAD_MOMENT_CERTAINTY:
            found.append(
                Defect(
                    DefectCode.DEAD_MOMENT.value,
                    f"a dead moment ({semantic.dead_moment_probability}% likely)",
                    CEILINGS[DefectCode.DEAD_MOMENT.value],
                    vetoable=True,
                )
            )

    found.extend(_portrait_defects(artistic, is_portrait=is_portrait(semantic)))

    # Measurably weaker than a sibling, not merely different from one. The
    # margin test happened upstream: inside it, two frames are a tie as far as
    # local measurement can tell, and both survive.
    if not inp.is_best_in_cluster:
        found.append(
            Defect(
                DefectCode.INFERIOR_DUPLICATE.value,
                f"a sharper frame exists in this group ({inp.cluster_margin:.0f} points higher)",
                CEILINGS[DefectCode.INFERIOR_DUPLICATE.value],
            )
        )

    return sorted(found, key=lambda d: d.ceiling)


# The genres in which a face is the photograph rather than an element of it.
PORTRAIT_GENRES = frozenset({"portrait"})


def is_portrait(semantic: Semantic) -> bool:
    """Whether the content pass called this a portrait.

    Used to decide whether the face gates the frame. It is a weaker signal than
    measuring the face, and it is the right one: an environmental portrait puts
    the subject small in a wide scene, so face size cannot distinguish it from a
    landscape with a passer-by in it. The genre can.
    """
    return semantic.present and str(semantic.genre).lower() in PORTRAIT_GENRES


def _portrait_defects(
    artistic: stage3_module.ArtisticAssessment | None, *, is_portrait: bool = False
) -> list[Defect]:
    """A face that failed, stated as a fact rather than as a low score.

    Requires a *completed* read. A missing Stage 3 says nothing about the
    subject's eyes, and inferring a defect from an absent analysis is the same
    error as inferring quality from one.
    """
    if artistic is None or not artistic.usable or not artistic.portrait:
        return []
    portrait = artistic.portrait
    if not stage3_module.face_gates_the_frame(portrait, is_portrait=is_portrait):
        # A person in a landscape. Their expression is not what the photograph
        # is about, and gating on it would write off half of street photography.
        return []
    verdict, reason = stage3_module.portrait_verdict(artistic, is_portrait=is_portrait)
    if verdict != "reject":
        return []
    if artistic.says_deliberate:
        return []

    code = DefectCode.EYES_CLOSED if portrait.eyes_are_shut else DefectCode.BAD_EXPRESSION
    return [Defect(code.value, reason, CEILINGS[code.value])]


# --- evidence that lifts ------------------------------------------------------


@dataclass
class Uplift:
    """A floor, and the completed Stage 3 finding that justifies it."""

    code: str
    detail: str
    floor: int

    def to_dict(self) -> dict:
        return asdict(self)


STRONG_DOCUMENTARY = 78
STRONG_DISTINCTIVENESS = 78
UPLIFT_CONFIDENCE = 65
DOCUMENTARY_FLOOR = 55
ARTISTIC_FLOOR = 58


def artistic_uplifts(artistic: stage3_module.ArtisticAssessment | None) -> list[Uplift]:
    """The floors a confident artistic read is allowed to set.

    Deliberately hard to earn: a completed read, high confidence, and a specific
    dimension in the top fifth. This is the mechanism that keeps a photograph
    which is technically poor and commercially useless from being filed as a
    failure -- and it is the only mechanism, because a soft nudge to the score
    would be swamped by the technical terms it is meant to overrule.
    """
    if artistic is None or not artistic.usable:
        return []
    if artistic.artistic_confidence < UPLIFT_CONFIDENCE:
        return []

    out: list[Uplift] = []
    if artistic.score("documentary_significance") >= STRONG_DOCUMENTARY:
        out.append(
            Uplift(
                "documentary_value",
                f"documentary significance {artistic.score('documentary_significance')} with "
                f"confidence {artistic.artistic_confidence}: preserves a place, event or practice",
                DOCUMENTARY_FLOOR,
            )
        )
    if artistic.artistic_candidate and artistic.score("distinctiveness") >= STRONG_DISTINCTIVENESS:
        out.append(
            Uplift(
                "artistic_value",
                f"distinctiveness {artistic.score('distinctiveness')} and named a candidate with "
                f"confidence {artistic.artistic_confidence}",
                ARTISTIC_FLOOR,
            )
        )
    return out


# --- the components -----------------------------------------------------------


def _clamp(value: float) -> int:
    return int(max(0, min(100, round(value))))


def artistic_component(artistic: stage3_module.ArtisticAssessment | None) -> int | None:
    """The eight dimensions, weighted. None when there is no usable read."""
    if artistic is None or not artistic.usable:
        return None
    return _clamp(sum(w * artistic.score(name) for name, w in ARTISTIC_WEIGHTS.items()))


def semantic_component(semantic: Semantic) -> int | None:
    """What the content pass found, excluding everything about releases.

    `axis_a` is commercial usability and carries the least weight of the three:
    it is the axis that rewards the tidiest frame in the set, which is precisely
    the judgement this score is trying not to make.
    """
    if not semantic.present:
        return None
    ranked = 0.40 * semantic.axis_b + 0.30 * semantic.axis_c + 0.30 * semantic.axis_a
    return _clamp(0.70 * ranked + 0.30 * semantic.subject_strength)


def portrait_component(
    artistic: stage3_module.ArtisticAssessment | None, *, is_portrait: bool = False
) -> int | None:
    """How the face came out. None when there is no face to judge."""
    if artistic is None or not artistic.usable or not artistic.portrait:
        return None
    portrait = artistic.portrait
    if not stage3_module.face_gates_the_frame(portrait, is_portrait=is_portrait):
        return None
    return _clamp(
        0.40 * portrait.portrait_publishability
        + 0.30 * portrait.expression_quality
        + 0.15 * portrait.pose_quality
        + 0.15 * portrait.face_sharpness
    )


def documentary_component(
    semantic: Semantic, artistic: stage3_module.ArtisticAssessment | None
) -> int:
    """The best available reading of what the frame preserves.

    Stage 3 wins where it exists: it looked at the photograph and answered the
    question directly, while `axis_c` is a rank inside an arbitrary group of
    twelve. Where neither ran, the honest answer is the middle of the scale.
    """
    if artistic is not None and artistic.usable:
        return artistic.score("documentary_significance")
    if semantic.present:
        return semantic.axis_c
    return 50


def scene_component(inp: ScoreInput, scores: AssetScores) -> int:
    """Standing against the rest of this collection, not against photography.

    A frame in a burst of nine near-identical takes is worth less than the same
    frame shot once, and the winner of a big cluster is worth more than a frame
    with no competition -- it beat something.
    """
    base = float(scores.uniqueness)
    if inp.is_best_in_cluster and inp.cluster_size > 1:
        base = min(100.0, base + min(10.0, 2.5 * (inp.cluster_size - 1)))
    return _clamp(base)


# --- the score ----------------------------------------------------------------


@dataclass
class FinalScore:
    value: int = 0
    # The blend before any ceiling or floor was applied. Kept because a capped
    # score and a genuinely low one look identical afterwards, and they are not
    # the same finding.
    blended: int = 0
    # What the same frame would have scored with no artistic read at all. The
    # difference is Stage 3's actual contribution, in points.
    without_stage3: int = 0
    stage3_delta: int = 0
    components: dict[str, int] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    defects: list[dict] = field(default_factory=list)
    uplifts: list[dict] = field(default_factory=list)
    applied_ceiling: int | None = None
    applied_floor: int | None = None
    stage3_status: str = stage3_module.Stage3Status.PENDING.value
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _blend(components: dict[str, int | None], weights: dict[str, float]) -> tuple[int, dict]:
    """Weighted mean over the components that exist, renormalised.

    A missing component is dropped and its weight redistributed rather than
    scored as zero. Zero would mean "this frame has no face and is therefore
    bad at having a face", which is not a finding about anything.
    """
    live = {k: v for k, v in components.items() if v is not None and weights.get(k)}
    total = sum(weights[k] for k in live)
    if not live or total <= 0:
        return 0, {}
    used = {k: weights[k] / total for k in live}
    return _clamp(sum(used[k] * live[k] for k in live)), used


@dataclass
class _Pass:
    """One evaluation of a frame, with or without its artistic read."""

    value: int
    blended: int
    components: dict[str, int]
    weights: dict[str, float]
    defects: list[Defect]
    uplifts: list[Uplift]
    ceiling: int | None
    floor: int | None


def _evaluate(
    inp: ScoreInput,
    scores: AssetScores,
    artistic: stage3_module.ArtisticAssessment | None,
    thresholds: CurationThresholds,
) -> _Pass:
    """Score the frame using exactly the evidence passed in.

    Called twice per photograph: once with the artistic read and once without.
    Running the *same* function both times is what makes `stage3_delta` a
    measurement -- an approximation of the counterfactual would drift from the
    real path the moment either changed.
    """
    has_read = artistic is not None and artistic.usable

    components: dict[str, int | None] = {
        "technical": scores.post_edit_potential,
        "artistic": artistic_component(artistic),
        "semantic": semantic_component(inp.semantic),
        "portrait": portrait_component(artistic, is_portrait=is_portrait(inp.semantic)),
        "documentary": documentary_component(inp.semantic, artistic),
        "scene": scene_component(inp, scores),
    }
    blended, used = _blend(components, WEIGHTS_WITH_STAGE3 if has_read else WEIGHTS_WITHOUT_STAGE3)

    defects = _apply_vetoes(critical_defects(inp, artistic), artistic)
    uplifts = artistic_uplifts(artistic)
    value = blended

    # Floors first, ceilings second, so a ceiling always wins. A photograph of
    # real documentary value whose subject blinked is still a photograph of
    # somebody blinking.
    floor = max((u.floor for u in uplifts), default=None)
    if floor is not None and floor > value:
        value = floor

    ceilings = [d.ceiling for d in defects if not d.vetoed_by]
    if not has_read:
        # Nothing judged this photograph, so nothing may call it one of the best.
        ceilings.append(thresholds.no_stage3_ceiling)
    ceiling = min(ceilings, default=None)
    if ceiling is not None and ceiling < value:
        value = ceiling

    return _Pass(
        value=_clamp(value),
        blended=blended,
        components={k: v for k, v in components.items() if v is not None},
        weights={k: round(v, 4) for k, v in used.items()},
        defects=defects,
        uplifts=uplifts,
        ceiling=ceiling,
        floor=floor,
    )


def final_score(
    inp: ScoreInput,
    scores: AssetScores,
    *,
    thresholds: CurationThresholds = DEFAULT_THRESHOLDS,
) -> FinalScore:
    """One number, 0-100, for how good the photograph is.

    Deliberately not a function of `stock_potential` or `legal_readiness`. Those
    describe what a marketplace will accept, and a photograph does not get worse
    because a stranger walked into it.
    """
    artistic = inp.artistic if isinstance(inp.artistic, stage3_module.ArtisticAssessment) else None

    result = _evaluate(inp, scores, artistic, thresholds)
    counterfactual = _evaluate(inp, scores, None, thresholds)

    return FinalScore(
        value=result.value,
        blended=result.blended,
        without_stage3=counterfactual.value,
        stage3_delta=result.value - counterfactual.value,
        components=result.components,
        weights=result.weights,
        defects=[d.to_dict() for d in result.defects],
        uplifts=[u.to_dict() for u in result.uplifts],
        applied_ceiling=result.ceiling,
        applied_floor=result.floor,
        stage3_status=(
            artistic.status if artistic is not None else stage3_module.Stage3Status.PENDING.value
        ),
    )


def _apply_vetoes(
    defects: list[Defect],
    artistic: stage3_module.ArtisticAssessment | None,
) -> list[Defect]:
    """Let a confident artistic read overrule a *guess* about the frame.

    "This looks like a dead moment" and "the subject's eyes are closed" are not
    the same kind of statement, and only the first can be argued with. A frame
    Stage 3 read as strong is a frame where the semantic pass's probability was
    most likely describing an unconventional photograph.

    Only an *uplift* vetoes -- a completed, confident read with a dimension in
    the top fifth. A bare claim that some defect was deliberate does not, and
    used not to be so narrow: a portrait whose eyes were half shut had a
    68%-likely dead moment waved through on an intent claim about something
    else entirely. An intent reading answers for the defect it names, and
    `hero_blockers` is where it is allowed to do that.
    """
    if artistic is None or not artistic.usable:
        return defects

    uplifts = artistic_uplifts(artistic)
    if not uplifts:
        return defects

    for defect in defects:
        if defect.vetoable:
            defect.vetoed_by = uplifts[0].detail
    return defects


# --- the categories -----------------------------------------------------------


@dataclass
class Verdict:
    category: str
    final_score: int
    reasons: list[str] = field(default_factory=list)
    reason_keys: list[dict] = field(default_factory=list)
    commercial_blockers: list[str] = field(default_factory=list)
    score: FinalScore = field(default_factory=FinalScore)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["score"] = self.score.to_dict()
        return payload


def commercial_blockers(inp: ScoreInput, scores: AssetScores, profile: CalibrationProfile) -> list[str]:
    """Why this photograph cannot be sold. Never why it is not good.

    Kept as a separate list, reported separately, and excluded from the score,
    because the failure this replaces was a good family photograph being filed
    as weak on the grounds that its subjects had not signed anything.
    """
    blocking: list[str] = []
    semantic = inp.semantic
    if not semantic.present:
        blocking.append("no content check has run, so faces and trademarks are unverified")
        return blocking
    if semantic.faces or semantic.identifiable_people:
        blocking.append("a model release is required before this can be licensed")
    if semantic.logos:
        blocking.append("a readable trademark is present: editorial only unless cleared")
    if semantic.recognizable_property:
        blocking.append("recognisable property: a property release may be required")
    if scores.stock_potential < profile.threshold("stock_standard"):
        blocking.append(
            f"limited commercial demand (stock potential {scores.stock_potential} below "
            f"{profile.threshold('stock_standard'):.0f})"
        )
    return blocking


def categorise(
    inp: ScoreInput,
    scores: AssetScores,
    profile: CalibrationProfile,
    *,
    thresholds: CurationThresholds = DEFAULT_THRESHOLDS,
    score: FinalScore | None = None,
) -> Verdict:
    """The photograph's own verdict, and then what may be done with it."""
    final = score or final_score(inp, scores, thresholds=thresholds)
    artistic = inp.artistic if isinstance(inp.artistic, stage3_module.ArtisticAssessment) else None
    blockers = commercial_blockers(inp, scores, profile)
    reasons: list[str] = []
    keys: list[dict] = []

    live_defects = [d for d in final.defects if not d["vetoed_by"]]
    vetoed = [d for d in final.defects if d["vetoed_by"]]

    if live_defects:
        worst = live_defects[0]
        reasons.append(f"{worst['detail']} -- a technical score cannot make up for this")
        keys.append({"key": f"category.defect.{worst['code']}", "params": {"detail": worst["detail"]}})
        for uplift in final.uplifts:
            reasons.append(f"noted but not enough to save it: {uplift['detail']}")
        return Verdict(PhotoCategory.WEAK.value, final.value, reasons, keys, blockers, final)

    for defect in vetoed:
        reasons.append(f"{defect['detail']} -- overruled: {defect['vetoed_by']}")

    if final.value >= thresholds.top and _clears_top(
        artistic, thresholds, portrait_frame=is_portrait(inp.semantic)
    ):
        reasons.append(
            f"final score {final.value} with a completed artistic read "
            f"(confidence {artistic.artistic_confidence}) and no critical defect"
        )
        keys.append({"key": "category.top", "params": {"value": final.value}})
        return Verdict(PhotoCategory.TOP.value, final.value, reasons, keys, blockers, final)

    if final.value < thresholds.weak:
        reasons.append(
            f"final score {final.value} is below {thresholds.weak}: "
            "not a photograph worth returning to"
        )
        keys.append({"key": "category.weak", "params": {"value": final.value}})
        return Verdict(PhotoCategory.WEAK.value, final.value, reasons, keys, blockers, final)

    if final.uplifts and final.applied_floor is not None and final.applied_floor > final.blended:
        # Only when the floor actually did the work. A strong frame that also
        # happens to qualify for an uplift was not rescued by it, and saying so
        # would put a claim in the report that the arithmetic does not support.
        reasons.append(f"held above the line by the artistic read: {final.uplifts[0]['detail']}")

    if _is_ambiguous(inp, artistic, final, thresholds):
        reasons.append(
            _ambiguity_reason(
                artistic, final, thresholds, portrait_frame=is_portrait(inp.semantic)
            )
        )
        keys.append({"key": "category.needs_decision", "params": {"value": final.value}})
        return Verdict(PhotoCategory.NEEDS_DECISION.value, final.value, reasons, keys, blockers, final)

    if not blockers:
        # Written for a photographer to read. The stock/personal split is
        # already expressed by the category, so the sentence describes the
        # photograph rather than its licensing position -- which is also what
        # keeps it out of the filter that strips legal language from the report.
        reasons.append(
            f"a clean, legible photograph that also works as stock material "
            f"(scores {final.value} after editing)"
        )
        keys.append({"key": "category.good_stock", "params": {"value": final.value}})
        return Verdict(PhotoCategory.GOOD_STOCK.value, final.value, reasons, keys, blockers, final)

    reasons.append(
        f"a photograph worth keeping and printing (scores {final.value} after editing)"
    )
    keys.append({"key": "category.good_personal", "params": {"value": final.value}})
    return Verdict(PhotoCategory.GOOD_PERSONAL.value, final.value, reasons, keys, blockers, final)


def _clears_top(
    artistic: stage3_module.ArtisticAssessment | None,
    thresholds: CurationThresholds,
    *,
    portrait_frame: bool = False,
) -> bool:
    """TOP needs the artistic read to be present, confident and positive.

    A high blended score is not sufficient on its own: without this, a frame
    could reach 85 on technical quality and scene standing while Stage 3 was
    still reporting that there is nothing in it.
    """
    if artistic is None or not artistic.usable:
        return False
    if artistic.artistic_confidence < thresholds.top_artistic_confidence:
        return False
    if stage3_module.hero_blockers(artistic, is_portrait=portrait_frame):
        return False
    component = artistic_component(artistic) or 0
    return component >= thresholds.top_artistic_floor


def _is_ambiguous(
    inp: ScoreInput,
    artistic: stage3_module.ArtisticAssessment | None,
    final: FinalScore,
    thresholds: CurationThresholds,
) -> bool:
    """Genuinely undecidable, and rare on purpose.

    A review queue that fills up is a queue nobody reads, so the bar is: the
    frame sits on the keep-or-not boundary *and* the analysis says it does not
    know. Either one alone is not enough -- plenty of frames sit near a
    threshold and are perfectly obvious, and plenty of uncertain reads are of
    frames that are clearly fine.

    Only the WEAK boundary counts. Whether a photograph is TOP or merely good is
    also a boundary, and asking a person about it is a waste of their attention:
    both answers are "keep", and the difference shows up in a sorted list
    anyway. An earlier version tested both boundaries and sent every strong,
    slightly-uncertain frame to the decision queue.
    """
    if artistic is None or not artistic.usable:
        return False

    if abs(final.value - thresholds.weak) > thresholds.decision_band:
        return False

    if artistic.uncertainty >= thresholds.decision_uncertainty:
        return True
    verdict, _ = stage3_module.portrait_verdict(artistic, is_portrait=is_portrait(inp.semantic))
    return verdict == "review"


def _ambiguity_reason(
    artistic: stage3_module.ArtisticAssessment | None,
    final: FinalScore,
    thresholds: CurationThresholds,
    *,
    portrait_frame: bool = False,
) -> str:
    verdict, detail = (
        stage3_module.portrait_verdict(artistic, is_portrait=portrait_frame)
        if artistic
        else ("keep", "")
    )
    if verdict == "review" and detail:
        return f"on the boundary at {final.value}, and {detail}"
    return (
        f"on the boundary at {final.value}, and the artistic read is "
        f"{artistic.uncertainty if artistic else 100}% unsure it understood this frame"
    )


# --- reporting helpers --------------------------------------------------------

CATEGORY_ORDER = (
    PhotoCategory.TOP,
    PhotoCategory.GOOD_STOCK,
    PhotoCategory.GOOD_PERSONAL,
    PhotoCategory.NEEDS_DECISION,
    PhotoCategory.WEAK,
)


def counts(verdicts: list[str]) -> dict[str, int]:
    """Every category present, in a fixed order, including the empty ones.

    A category missing from a summary reads as "none of these happened"; a zero
    reads as "none of these happened". Only one of those is a fact the reader
    can rely on, so all five are always printed.
    """
    tally = dict.fromkeys((c.value for c in CATEGORY_ORDER), 0)
    for value in verdicts:
        if value in tally:
            tally[value] += 1
    return tally
