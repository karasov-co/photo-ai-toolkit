"""Platform eligibility, release requirements and provenance conflicts."""

import json

import pytest

import marketplaces
from provenance import Provenance, ProvenanceRecord, conflicts_for, label_for_submission, warn_for_edit
from scoring import AssetScores, Route, RouteClass, ScoredAsset, Semantic


@pytest.fixture
def rules():
    marketplaces.load_rules.cache_clear()
    return marketplaces.load_rules()


def asset(route=Route.COMMERCIAL, stock=70, aesthetic=70):
    return ScoredAsset(
        asset_id="a",
        filename="a.jpg",
        kind="photo",
        scores=AssetScores(stock_potential=stock, aesthetic_potential=aesthetic),
        route_class=RouteClass.STOCK_STRONG,
        route=route,
    )


def clean(**kwargs):
    base = {"present": True, "faces": False, "logos": False, "identifiable_people": False}
    return Semantic(**{**base, **kwargs})


def photo_facts(**kwargs):
    return marketplaces.TechnicalFacts(**{"kind": "photo", "megapixels": 24.0, "file_format": "JPEG", **kwargs})


def video_facts(**kwargs):
    return marketplaces.TechnicalFacts(
        **{
            "kind": "video", "width": 3840, "height": 2160,
            "duration": 12.0, "container": "mov,mp4,m4a", **kwargs,
        }
    )


CAMERA = ProvenanceRecord(value=Provenance.CAMERA_ORIGINAL, declared_by="metadata")
GENERATED = ProvenanceRecord(value=Provenance.FULLY_GENERATED, declared_by="user")


# --- the rules file ---------------------------------------------------------


def test_the_rules_load_and_carry_a_version(rules):
    assert rules.platforms
    assert rules.ruleset_version != "unknown"


def test_every_platform_records_where_its_rules_came_from(rules):
    """A stale rule must be visible rather than merely wrong."""
    for platform in rules.platforms:
        assert platform.get("sources"), platform.get("id")
        assert platform.get("verified_on"), platform.get("id")


def test_the_ruleset_carries_a_disclaimer(rules):
    assert "change" in rules.disclaimer.lower()


def test_rules_live_in_data_not_in_code():
    """Editing a resolution floor must not require touching Python."""
    payload = json.loads(marketplaces.RULES_PATH.read_text(encoding="utf-8"))
    assert payload["platforms"]


def test_a_missing_rules_file_degrades_rather_than_crashing():
    marketplaces.load_rules.cache_clear()
    empty = marketplaces.load_rules("/nonexistent/rules.json")
    assert empty.platforms == []
    marketplaces.load_rules.cache_clear()


# --- technical eligibility --------------------------------------------------


def test_a_photo_below_the_resolution_floor_is_refused(rules):
    adobe = rules.by_id("adobe_stock")
    blockers = marketplaces.check_technical(adobe, photo_facts(megapixels=2.0))
    assert blockers and "minimum" in blockers[0]


def test_a_photo_above_the_floor_passes(rules):
    assert marketplaces.check_technical(rules.by_id("adobe_stock"), photo_facts()) == []


def test_an_unaccepted_file_format_is_refused(rules):
    blockers = marketplaces.check_technical(
        rules.by_id("adobe_stock"), photo_facts(file_format="RAW")
    )
    assert any("not accepted" in b for b in blockers)


def test_a_clip_shorter_than_the_platform_minimum_is_refused(rules):
    blockers = marketplaces.check_technical(rules.by_id("adobe_stock"), video_facts(duration=2.0))
    assert any("minimum" in b for b in blockers)


def test_a_clip_longer_than_the_platform_maximum_is_told_to_trim(rules):
    blockers = marketplaces.check_technical(rules.by_id("adobe_stock"), video_facts(duration=180.0))
    assert any("trim" in b for b in blockers)


def test_a_vertical_clip_is_not_refused_for_being_vertical(rules):
    """1080x1920 satisfies a published 1920x1080 floor."""
    assert marketplaces.check_technical(
        rules.by_id("adobe_stock"), video_facts(width=1080, height=1920)
    ) == []


def test_undersized_footage_is_refused(rules):
    blockers = marketplaces.check_technical(
        rules.by_id("adobe_stock"), video_facts(width=1280, height=720)
    )
    assert any("below" in b for b in blockers)


def test_an_unaccepted_container_is_refused(rules):
    blockers = marketplaces.check_technical(
        rules.by_id("adobe_stock"), video_facts(container="avi")
    )
    assert any("container" in b for b in blockers)


def test_a_comma_joined_container_list_is_matched(rules):
    """ffprobe reports 'mov,mp4,m4a,3gp' for one file."""
    assert marketplaces.check_technical(rules.by_id("adobe_stock"), video_facts()) == []


def test_pond5_accepts_shorter_footage_than_adobe(rules):
    """The whole point of a per-platform matrix rather than one global rule."""
    short = video_facts(duration=3.5)
    assert marketplaces.check_technical(rules.by_id("pond5"), short) == []
    assert marketplaces.check_technical(rules.by_id("adobe_stock"), short) != []


# --- commercial vs editorial ------------------------------------------------


def test_a_clean_frame_is_offered_commercially(rules):
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), CAMERA, rules=rules)
    assert any(r.eligible for r in recs)
    assert all(r.route == "commercial" for r in recs)


def test_a_frame_with_a_face_is_offered_editorially_only(rules):
    recs = marketplaces.evaluate(
        asset(route=Route.EDITORIAL), photo_facts(), clean(faces=True), CAMERA, rules=rules
    )
    assert all(r.route == "editorial" for r in recs)


def test_a_missing_model_release_is_named(rules):
    recs = marketplaces.evaluate(
        asset(route=Route.EDITORIAL), photo_facts(), clean(faces=True), CAMERA, rules=rules
    )
    assert any("model release" in r.missing_releases for r in recs)


def test_a_held_release_is_not_reported_as_missing(rules):
    recs = marketplaces.evaluate(
        asset(route=Route.EDITORIAL), photo_facts(), clean(faces=True), CAMERA,
        rules=rules, has_model_release=True,
    )
    assert all("model release" not in r.missing_releases for r in recs)


def test_a_missing_property_release_is_named(rules):
    recs = marketplaces.evaluate(
        asset(route=Route.EDITORIAL), photo_facts(),
        clean(recognizable_property=True, faces=True), CAMERA, rules=rules,
    )
    assert any("property release" in r.missing_releases for r in recs)


def test_a_trademark_is_reported_as_a_policy_concern(rules):
    recs = marketplaces.evaluate(
        asset(route=Route.EDITORIAL), photo_facts(), clean(logos=True), CAMERA, rules=rules
    )
    assert any(any("trademark" in c for c in r.policy_conflicts) for r in recs)


def test_alamy_is_ranked_up_for_editorial_work(rules):
    """Its editorial market is stronger, which is a real ordering difference."""
    recs = marketplaces.evaluate(
        asset(route=Route.EDITORIAL), photo_facts(), clean(faces=True), CAMERA, rules=rules
    )
    alamy = next(r for r in recs if r.platform_id == "alamy")
    assert "editorial" in " ".join(alamy.reasons).lower()


# --- provenance -------------------------------------------------------------


def test_generative_content_is_blocked_where_the_platform_refuses_it(rules):
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), GENERATED, rules=rules)
    blocked = {r.platform_id for r in recs if not r.eligible}
    assert {"shutterstock", "alamy", "getty_istock"} <= blocked


def test_generative_content_is_allowed_where_it_is_accepted_with_a_label(rules):
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), GENERATED, rules=rules)
    adobe = next(r for r in recs if r.platform_id == "adobe_stock")
    assert adobe.eligible
    assert any("declared" in c for c in adobe.policy_conflicts)


def test_a_camera_original_conflicts_with_nothing(rules):
    assert conflicts_for(CAMERA, rules.platforms) == []


def test_undeclared_provenance_is_advisory_not_blocking(rules):
    """Absence of AI metadata proves nothing -- it is trivially stripped."""
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), ProvenanceRecord(), rules=rules)
    shutterstock = next(r for r in recs if r.platform_id == "shutterstock")
    assert shutterstock.eligible
    assert shutterstock.policy_conflicts


def test_ai_denoise_is_not_treated_as_generative(rules):
    """Alamy names denoise explicitly as acceptable on an original photograph."""
    retouched = ProvenanceRecord(value=Provenance.AI_ASSISTED_RETOUCH)
    assert conflicts_for(retouched, rules.platforms) == []


def test_a_generative_edit_is_warned_about_before_it_is_applied():
    warnings = warn_for_edit(CAMERA, uses_generative=True)
    assert warnings
    assert any("Alamy" in w for w in warnings)
    assert any("editorial" in w for w in warnings)


def test_a_normal_edit_produces_no_warning():
    assert warn_for_edit(CAMERA, uses_generative=False) == []


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        (Provenance.FULLY_GENERATED, "Generative AI"),
        (Provenance.AI_ASSISTED_RETOUCH, "retouching"),
        (Provenance.CAMERA_ORIGINAL, "Not generative"),
        (Provenance.UNKNOWN, "Undeclared"),
    ],
)
def test_the_submission_label_matches_the_provenance(value, fragment):
    assert fragment in label_for_submission(ProvenanceRecord(value=value))


# --- recommendations --------------------------------------------------------


def test_ineligible_platforms_are_kept_with_their_reason(rules):
    """'Blocked because X' is more useful to a contributor than silence."""
    recs = marketplaces.evaluate(asset(), photo_facts(megapixels=1.0), clean(), CAMERA, rules=rules)
    assert recs
    assert all(not r.eligible for r in recs)
    assert all(r.blockers for r in recs)


def test_eligible_platforms_sort_ahead_of_blocked_ones(rules):
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), GENERATED, rules=rules)
    eligibility = [r.eligible for r in recs]
    assert eligibility == sorted(eligibility, reverse=True)


def test_nothing_is_export_ready_without_complete_metadata(rules):
    recs = marketplaces.evaluate(
        asset(), photo_facts(), clean(), CAMERA, rules=rules, metadata_complete=False
    )
    assert all(not r.export_ready for r in recs)
    assert all(r.missing_metadata for r in recs if r.eligible)


def test_complete_metadata_makes_a_clean_asset_export_ready(rules):
    recs = marketplaces.evaluate(
        asset(), photo_facts(), clean(), CAMERA, rules=rules, metadata_complete=True
    )
    assert any(r.export_ready for r in recs)


def test_a_platform_requiring_manual_submission_says_so(rules):
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), CAMERA, rules=rules)
    getty = next(r for r in recs if r.platform_id == "getty_istock")
    assert getty.manual_submission_required


def test_recommendations_carry_the_date_the_rules_were_checked(rules):
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), CAMERA, rules=rules)
    assert all(r.verified_on for r in recs)


def test_exclusivity_is_costed_rather_than_ignored(rules):
    """Going exclusive removes the asset from every other platform."""
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), CAMERA, rules=rules)
    getty = next(r for r in recs if r.platform_id == "getty_istock")
    adobe = next(r for r in recs if r.platform_id == "adobe_stock")
    assert getty.suitability < adobe.suitability


def test_the_summary_names_the_top_recommendations(rules):
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), CAMERA, rules=rules)
    summary = marketplaces.summarise(recs)
    assert summary["eligible_count"] > 0
    assert len(summary["recommended"]) <= 3


def test_no_recommendation_ever_promises_acceptance(rules):
    """Acceptance is a human reviewer's decision at every one of these."""
    recs = marketplaces.evaluate(asset(), photo_facts(), clean(), CAMERA, rules=rules)
    text = " ".join(line for r in recs for line in (*r.reasons, *r.policy_conflicts)).lower()
    assert "guarantee" not in text
    assert "will be accepted" not in text
    assert "will sell" not in text
