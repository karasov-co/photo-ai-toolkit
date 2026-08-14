"""The ten dimensions, and the precedence that turns them into a class."""

import pytest

from calibration import CalibrationProfile, default_photo_profile, default_video_profile
from issues import IssueCode, IssueSet
from scoring import (
    AssetScores,
    AssetTag,
    Route,
    RouteClass,
    ScoreInput,
    Semantic,
    classify,
    confidence_score,
    current_quality_score,
    eligible_for_flagship,
    explain,
    portfolio_potential_score,
    post_edit_potential_score,
    recoverability_score,
    route_for,
    routing_score,
    score,
    uniqueness_score,
)


@pytest.fixture
def profile():
    return default_photo_profile()


def blocking(code: IssueCode, detail: str = "") -> IssueSet:
    found = IssueSet()
    found.add(code, detail)
    return found


def clean_semantic(**kwargs) -> Semantic:
    """A frame somebody actually looked at, with nothing needing a release."""
    base = {
        "present": True,
        "axis_a": 75,
        "axis_b": 70,
        "axis_c": 60,
        "genre": "landscape",
    }
    return Semantic(**{**base, **kwargs})


# --- the score schema -------------------------------------------------------


def test_every_dimension_is_present_and_in_range(profile):
    inp = ScoreInput(asset_id="a", filename="a.RW2", technical_quality=55, uplift=10, is_raw=True)
    scores = score(inp, profile)
    payload = scores.to_dict()
    assert set(payload) == {
        "current_quality", "recoverability", "post_edit_potential", "aesthetic_potential",
        "stock_potential", "portfolio_potential", "uniqueness",
        "confidence", "routing_score",
    }
    assert all(0 <= v <= 100 for v in payload.values())
    assert all(isinstance(v, int) for v in payload.values())


def test_scores_survive_absurd_inputs_without_leaving_the_scale(profile):
    inp = ScoreInput(asset_id="a", filename="a", technical_quality=999, uplift=999, is_raw=True)
    assert all(0 <= v <= 100 for v in score(inp, profile).to_dict().values())


# --- current quality vs potential -------------------------------------------


def test_current_quality_is_the_file_as_it_sits():
    assert current_quality_score(62.4) == 62


def test_a_dark_recoverable_frame_scores_higher_on_potential_than_on_quality():
    """The headline requirement of the whole feature."""
    current = current_quality_score(38)
    potential = post_edit_potential_score(current, uplift=30, found=IssueSet(), recoverability=100)
    assert potential > current
    assert potential - current >= 20


def test_a_frame_whose_subject_missed_focus_is_not_promoted_by_exposure_uplift():
    """Brightening an out-of-focus frame produces a brighter out-of-focus frame."""
    found = blocking(IssueCode.MISSED_FOCUS, "blur ratio 1.1")
    potential = post_edit_potential_score(45, uplift=40, found=found, recoverability=30)
    assert potential <= 25


def test_the_cap_applies_however_large_the_uplift_is():
    found = blocking(IssueCode.MISSED_FOCUS)
    assert post_edit_potential_score(90, uplift=999, found=found, recoverability=100) <= 25


def test_more_blockers_lower_the_ceiling_further():
    one = IssueSet()
    one.add(IssueCode.MISSED_FOCUS)
    two = IssueSet()
    two.add(IssueCode.MISSED_FOCUS)
    two.add(IssueCode.BLOWN_HIGHLIGHTS)
    assert post_edit_potential_score(80, 20, two, recoverability=50) < post_edit_potential_score(
        80, 20, one, recoverability=50
    )


def test_potential_gain_is_exposed_directly():
    scores = AssetScores(current_quality=40, post_edit_potential=72)
    assert scores.potential_gain == 32


# --- recoverability ---------------------------------------------------------


def test_raw_has_more_latitude_than_jpeg():
    assert recoverability_score(IssueSet(), is_raw=True) > recoverability_score(
        IssueSet(), is_raw=False
    )


def test_unrecoverable_problems_cost_more_than_partial_ones():
    partial = IssueSet()
    partial.add(IssueCode.HEAVY_NOISE)
    fatal = IssueSet()
    fatal.add(IssueCode.MISSED_FOCUS)
    assert recoverability_score(fatal, is_raw=True) < recoverability_score(partial, is_raw=True)


def test_routine_fixable_problems_do_not_reduce_recoverability():
    """A fixable problem is the reason recoverability matters, not a deduction."""
    fixable = IssueSet()
    fixable.add(IssueCode.UNDEREXPOSED)
    fixable.add(IssueCode.COLOR_CAST)
    assert recoverability_score(fixable, is_raw=True) == recoverability_score(
        IssueSet(), is_raw=True
    )


# --- nothing blocks any more -------------------------------------------------
#
# Nine tests lived here: a face routed a frame to EDITORIAL, a logo did, an
# unexamined frame did, and three dimensions of `legal_readiness` scored the
# result. All of it asked a vision model a legal question it cannot answer, and
# the answer decided which pile a photograph landed in.


def test_every_frame_is_commercial_now():
    assert route_for(clean_semantic()) is Route.COMMERCIAL
    assert route_for(Semantic()) is Route.COMMERCIAL


def test_the_release_vocabulary_is_gone_from_the_module():
    import scoring

    for gone in ("legal_readiness_score", "UNKNOWN_LEGAL_READINESS",
                 "LEGAL_READINESS_NOT_ASSESSED"):
        assert not hasattr(scoring, gone), gone
    for gone in ("EDITORIAL_ONLY", "NEEDS_MODEL_RELEASE",
                 "NEEDS_PROPERTY_RELEASE", "LEGAL_REVIEW"):
        assert gone not in AssetTag.__members__, gone
    assert "legal_readiness" not in score(
        ScoreInput(asset_id="a", filename="a", technical_quality=70),
        default_photo_profile(),
    ).to_dict()



# --- uniqueness -------------------------------------------------------------


def test_a_frame_with_no_near_duplicate_is_unique():
    assert uniqueness_score(cluster_size=1, is_best=True, similarity=0.0) == 100


def test_a_weaker_duplicate_scores_near_zero():
    assert uniqueness_score(cluster_size=6, is_best=False, similarity=0.9) < 10


def test_a_large_burst_dilutes_even_its_best_frame():
    assert uniqueness_score(cluster_size=8, is_best=True, similarity=0.5) < uniqueness_score(
        cluster_size=2, is_best=True, similarity=0.5
    )


# --- confidence -------------------------------------------------------------


def test_an_unexamined_frame_is_less_confident():
    assert confidence_score(
        semantic=Semantic(), found=IssueSet(), is_raw=True, evidence_completeness=1.0
    ) < confidence_score(
        semantic=clean_semantic(), found=IssueSet(), is_raw=True, evidence_completeness=1.0
    )


def test_incomplete_evidence_reduces_confidence():
    assert confidence_score(
        semantic=clean_semantic(), found=IssueSet(), is_raw=True, evidence_completeness=0.3
    ) < confidence_score(
        semantic=clean_semantic(), found=IssueSet(), is_raw=True, evidence_completeness=1.0
    )


def test_uncertain_detections_reduce_confidence():
    unsure = IssueSet()
    unsure.add(IssueCode.COLOR_CAST, certainty=0.4)
    assert confidence_score(
        semantic=clean_semantic(), found=unsure, is_raw=True, evidence_completeness=1.0
    ) < confidence_score(
        semantic=clean_semantic(), found=IssueSet(), is_raw=True, evidence_completeness=1.0
    )


# --- routing precedence -----------------------------------------------------


def routed(profile, **kwargs):
    defaults = {
        "asset_id": "a",
        "filename": "a.RW2",
        "technical_quality": 70,
        "uplift": 8,
        "is_raw": True,
        "semantic": clean_semantic(),
    }
    inp = ScoreInput(**{**defaults, **kwargs})
    return classify(inp, score(inp, profile), profile, flagship_selected=kwargs.pop("flag", False))


def test_a_corrupt_file_is_trash(profile):
    assert routed(profile, issues=blocking(IssueCode.CORRUPT_FILE)).route_class is RouteClass.TRASH


def test_a_weaker_burst_sibling_is_a_comparison_not_a_deletion(profile):
    """Sharpness picks the winner and cannot see which take has the better face."""
    result = routed(profile, issues=blocking(IssueCode.WEAKER_DUPLICATE), is_best_in_cluster=False)
    assert result.route_class is RouteClass.DUPLICATE_CANDIDATE
    assert AssetTag.WEAKER_DUPLICATE in result.tags


def test_a_duplicate_says_whether_a_content_check_ran(profile):
    without = routed(profile, issues=blocking(IssueCode.WEAKER_DUPLICATE), is_best_in_cluster=False)
    assert any("no content check ran" in r for r in without.reasons)


def test_a_clip_with_no_usable_segment_is_trash(profile):
    result = routed(profile, kind="video", issues=blocking(IssueCode.NO_USABLE_SEGMENT))
    assert result.route_class is RouteClass.TRASH


def test_low_confidence_routes_to_review_rather_than_guessing(profile):
    result = routed(profile, evidence_completeness=0.4)
    assert result.route_class is RouteClass.REVIEW
    assert any("confidence" in r for r in result.reasons)


def test_a_hard_blocker_beats_low_confidence(profile):
    """A corrupt file is certain, however little else is known about it."""
    result = routed(profile, issues=blocking(IssueCode.CORRUPT_FILE), evidence_completeness=0.4)
    assert result.route_class is RouteClass.TRASH


def test_a_strong_clean_frame_reaches_strong_stock(profile):
    result = routed(profile, technical_quality=85, uplift=10, semantic=clean_semantic(axis_a=95))
    assert result.route_class is RouteClass.STOCK_STRONG


def test_a_middling_frame_reaches_standard_stock(profile):
    result = routed(profile, technical_quality=60, uplift=6, semantic=clean_semantic(axis_a=55))
    assert result.route_class is RouteClass.STOCK_STANDARD


def test_flagship_requires_being_selected_not_merely_scoring_well(profile):
    """Flagship is a collection-level decision, so a single frame cannot claim it."""
    strong = {"technical_quality": 92, "uplift": 6, "semantic": clean_semantic(axis_b=98)}
    assert routed(profile, **strong).route_class is not RouteClass.FLAGSHIP


def completed_artistic(**overrides):
    """A Stage 3 result good enough to raise no objection."""
    import stage3

    payload = {**dict.fromkeys(stage3.ARTISTIC_FIELDS, 80),
               "artistic_candidate": True, "artistic_confidence": 80}
    payload.update(overrides)
    return stage3.parse_assessment(payload)


def test_a_selected_frame_becomes_flagship(profile):
    """Selection is necessary; a completed artistic read is what makes it sufficient."""
    inp = ScoreInput(
        asset_id="a",
        filename="a.RW2",
        technical_quality=92,
        uplift=6,
        is_raw=True,
        semantic=clean_semantic(axis_b=98),
        artistic=completed_artistic(),
    )
    result = classify(inp, score(inp, profile), profile, flagship_selected=True)
    assert result.route_class is RouteClass.FLAGSHIP
    assert AssetTag.PORTFOLIO in result.tags


def test_selection_alone_is_not_enough_without_an_artistic_read(profile):
    """The invariant: a null Stage 3 cannot be promoted."""
    inp = ScoreInput(
        asset_id="a", filename="a.RW2", technical_quality=92, uplift=6,
        is_raw=True, semantic=clean_semantic(axis_b=98),
    )
    result = classify(inp, score(inp, profile), profile, flagship_selected=True)
    assert result.route_class is not RouteClass.FLAGSHIP
    assert any("no artistic analysis" in r for r in result.reasons)


def test_the_class_reason_comes_first(profile):
    """A plan prints the first reason; for a corrupt file it must be the blocker."""
    result = routed(
        profile,
        issues=blocking(IssueCode.CORRUPT_FILE, "unreadable"),
        semantic=clean_semantic(),
    )
    assert "unrecoverable" in result.reasons[0]


# --- flagship eligibility ---------------------------------------------------


def test_flagship_needs_an_absolute_floor_not_only_a_rank(profile):
    """The top 5% of a weak shoot is still weak."""
    weak = AssetScores(portfolio_potential=40, post_edit_potential=45)
    assert not eligible_for_flagship(weak, profile)


def test_a_memorable_frame_that_cannot_be_saved_is_not_flagship(profile):
    unfixable = AssetScores(portfolio_potential=95, post_edit_potential=20)
    assert not eligible_for_flagship(unfixable, profile)


def test_a_strong_recoverable_frame_is_eligible(profile):
    assert eligible_for_flagship(
        AssetScores(portfolio_potential=80, post_edit_potential=75), profile
    )


# --- configurable thresholds ------------------------------------------------


def test_lowering_a_threshold_changes_the_class_without_touching_the_scores():
    generous = CalibrationProfile(thresholds={"stock_standard": 10.0, "stock_strong": 20.0})
    strict = CalibrationProfile(thresholds={"stock_standard": 95.0, "stock_strong": 99.0})
    inp = ScoreInput(
        asset_id="a", filename="a", technical_quality=60, uplift=5,
        is_raw=True, semantic=clean_semantic(axis_a=60),
    )
    scores = score(inp, generous)
    assert classify(inp, scores, generous).route_class is RouteClass.STOCK_STRONG
    assert classify(inp, scores, strict).route_class is RouteClass.REVIEW


def test_video_and_photo_profiles_carry_different_thresholds():
    assert default_video_profile().threshold("trash_potential") != default_photo_profile().threshold(
        "trash_potential"
    )


def test_the_routing_score_is_the_only_blend(profile):
    """Every other dimension must be traceable to its own evidence."""
    scores = AssetScores(
        current_quality=50, recoverability=60, post_edit_potential=70,
        aesthetic_potential=80, stock_potential=40, portfolio_potential=30,
        uniqueness=90, confidence=70,
    )
    blended = routing_score(scores, profile)
    assert 0 <= blended <= 100
    assert blended != scores.post_edit_potential


def test_a_profile_with_broken_weights_falls_back_rather_than_dividing_by_zero():
    broken = CalibrationProfile(weights={"post_edit_potential": 0.0})
    assert routing_score(AssetScores(post_edit_potential=80), broken) >= 0


# --- derived dimensions -----------------------------------------------------


def test_portfolio_potential_is_gated_by_whether_the_frame_can_be_saved():
    rescuable = portfolio_potential_score(clean_semantic(axis_b=90), potential=80, aesthetic=85)
    doomed = portfolio_potential_score(clean_semantic(axis_b=90), potential=15, aesthetic=85)
    assert doomed < rescuable


# --- explanation ------------------------------------------------------------


def test_every_asset_can_explain_itself(profile):
    found = IssueSet()
    found.add(IssueCode.UNDEREXPOSED, "mean luma 44")
    found.add(IssueCode.HEAVY_NOISE, "sigma 9")
    inp = ScoreInput(
        asset_id="a", filename="a.RW2", technical_quality=45, uplift=20,
        issues=found, is_raw=True, semantic=clean_semantic(),
    )
    account = explain(classify(inp, score(inp, profile), profile), found)
    assert account["problems"]["fixable"]
    assert account["problems"]["partially_fixable"]
    assert account["problems"]["unrecoverable"] == []
    assert account["reasons"]
    assert account["potential_gain"] > 0


def test_strengths_are_reported_not_only_faults(profile):
    inp = ScoreInput(
        asset_id="a", filename="a.RW2", technical_quality=45, uplift=25,
        is_raw=True, semantic=clean_semantic(axis_b=85),
    )
    result = classify(inp, score(inp, profile), profile)
    assert result.strengths
    assert any("RAW" in s for s in result.strengths)
