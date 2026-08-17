"""Versioned profiles, and manual decisions the tool is not allowed to undo."""

import json

import pytest

from photoai.calibration import (
    BUILTIN_PROFILES,
    CalibrationProfile,
    CalibrationSet,
    default_photo_profile,
    default_video_profile,
    portfolio_first_profile,
    resolve,
    stock_first_profile,
)
from photoai.overrides import Override, OverrideStore, apply_to, capture
from photoai.reports import AssetRecord

# --- profiles ---------------------------------------------------------------


def test_a_default_profile_has_every_threshold_routing_reads():
    profile = default_photo_profile()
    for key in ("trash_potential", "review_confidence", "stock_standard", "stock_strong",
                "flagship_portfolio", "flagship_potential_floor"):
        assert profile.threshold(key) > 0


def test_the_shipped_profiles_admit_they_are_not_fitted():
    """Provisional numbers presented as fitted are worse than no numbers."""
    for factory in BUILTIN_PROFILES.values():
        assert not factory().is_fitted


def test_a_fitted_version_string_reads_as_fitted():
    assert CalibrationProfile(version="1.0.0").is_fitted


def test_video_thresholds_differ_from_photo_thresholds():
    assert default_video_profile().threshold("stock_strong") != default_photo_profile().threshold(
        "stock_strong"
    )


def test_the_stock_and_portfolio_profiles_disagree_about_what_matters():
    """Both are right for their own purpose, which is why both exist."""
    stock = stock_first_profile()
    portfolio = portfolio_first_profile()
    assert stock.weight("stock_potential") > portfolio.weight("stock_potential")
    assert portfolio.weight("portfolio_potential") > stock.weight("portfolio_potential")


def test_weights_are_normalised_so_a_profile_cannot_inflate_scores():
    profile = CalibrationProfile(weights={"post_edit_potential": 10.0, "uniqueness": 10.0})
    assert sum(profile.normalised_weights().values()) == pytest.approx(1.0)


def test_zero_weights_fall_back_rather_than_dividing_by_zero():
    assert CalibrationProfile(weights={"post_edit_potential": 0.0}).normalised_weights()


def test_an_unknown_threshold_falls_back_to_the_default():
    assert CalibrationProfile(thresholds={}).threshold("stock_strong") > 0


def test_a_profile_round_trips_through_disk(tmp_path):
    original = stock_first_profile()
    loaded = CalibrationProfile.load(original.save(tmp_path / "p.json"))
    assert loaded.name == original.name
    assert loaded.thresholds["stock_strong"] == original.thresholds["stock_strong"]


def test_a_typo_in_a_hand_edited_profile_does_not_fail_the_run(tmp_path):
    """These are files users edit by hand; one will eventually contain a typo."""
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"name": "mine", "nonsense_key": 1, "thresholds": {"stock_strong": 80}}))
    profile = CalibrationProfile.load(path)
    assert profile.name == "mine"
    assert profile.threshold("stock_strong") == 80
    assert profile.threshold("trash_potential") > 0


def test_an_unparseable_profile_falls_back_to_the_defaults(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json")
    assert CalibrationProfile.load(path).name == "default-photo"


def test_a_missing_profile_file_falls_back(tmp_path):
    assert CalibrationProfile.load(tmp_path / "absent.json").threshold("stock_strong") > 0


def test_a_video_profile_loaded_from_disk_lands_in_the_video_slot(tmp_path):
    path = default_video_profile().save(tmp_path / "v.json")
    resolved = resolve(path=path)
    assert resolved.video.media == "video"
    assert resolved.photo.media == "photo"


def test_an_unknown_profile_name_falls_back_with_a_warning():
    assert resolve("no-such-profile").photo.name == "default-photo"


def test_the_calibration_fingerprint_identifies_the_run():
    """Every report carries it, so a result can be traced to what produced it."""
    fingerprint = CalibrationSet().fingerprint
    assert "default-photo" in fingerprint and "default-video" in fingerprint


def test_the_right_profile_is_chosen_per_media_type():
    calibration = CalibrationSet()
    assert calibration.for_kind("video").media == "video"
    assert calibration.for_kind("photo").media == "photo"


# --- overrides --------------------------------------------------------------


def record(asset_id="a1", route_class="trash", filename="P1042721.RW2"):
    return AssetRecord(
        asset_id=asset_id,
        source_path=f"/archive/{filename}",
        filename=filename,
        media_type="photo",
        checksum="c" * 64,
        route_class=route_class,
        scores={"routing_score": 40},
        reasons=["below every threshold"],
        proposed_action="quarantine",
    )


def test_an_override_survives_a_fresh_analysis(tmp_path):
    """The failure that destroys trust: a rescued frame quietly re-condemned."""
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="flagship", note="fog, I want it"))
    store.save()

    reloaded = OverrideStore(tmp_path / "o.json")
    records = [record()]
    assert apply_to(records, reloaded) == 1
    assert records[0].route_class == "flagship"


def test_the_tools_own_conclusion_is_preserved_beside_the_override(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="flagship", note="fog"))
    records = [record()]
    apply_to(records, store)
    assert any("the tool had said trash" in r for r in records[0].reasons)
    assert "manual_override" in records[0].tags


def test_an_override_changes_the_proposed_filesystem_action(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="flagship"))
    records = [record()]
    apply_to(records, store)
    assert records[0].proposed_action == "keep_in_place"


def test_an_excluded_asset_is_marked_and_not_re_routed(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", excluded=True))
    records = [record()]
    apply_to(records, store)
    assert records[0].proposed_action == "excluded_by_user"


def test_an_override_can_change_the_genre(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", genre="reportage"))
    records = [record()]
    apply_to(records, store)
    assert records[0].genre == "reportage"


def test_records_without_an_override_are_untouched(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    records = [record()]
    assert apply_to(records, store) == 0
    assert records[0].route_class == "trash"


def test_an_override_agreeing_with_the_tool_is_not_counted_as_a_change(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="trash"))
    records = [record()]
    assert apply_to(records, store) == 0


def test_an_override_is_keyed_by_content_so_a_rename_does_not_lose_it(tmp_path):
    """Keying on the filename would drop the decision the first time a file moves."""
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="flagship"))
    renamed = record(filename="temple_light.RW2")
    apply_to([renamed], store)
    assert renamed.route_class == "flagship"


def test_an_edited_file_is_deliberately_a_different_asset(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="flagship"))
    different = record(asset_id="a2")
    apply_to([different], store)
    assert different.route_class == "trash"


def test_capture_remembers_what_the_tool_said(tmp_path):
    """The pair a personalised calibration would later need to learn from."""
    captured = capture(record())
    assert captured.tool_said == "trash"
    assert captured.tool_scores == {"routing_score": 40}


def test_an_override_can_be_removed(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="flagship"))
    assert store.remove("a1")
    assert store.get("a1") is None


def test_removing_an_absent_override_is_not_an_error(tmp_path):
    assert not OverrideStore(tmp_path / "o.json").remove("nope")


def test_a_corrupt_override_file_is_preserved_rather_than_overwritten(tmp_path):
    """Starting from empty would silently discard every decision ever made."""
    path = tmp_path / "o.json"
    path.write_text("{not json")
    store = OverrideStore(path)
    assert len(store) == 0
    assert (tmp_path / "o.json.corrupt").exists()


def test_the_store_writes_a_schema_version(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    store.set(Override(asset_id="a1", route_class="flagship"))
    payload = json.loads(store.save().read_text(encoding="utf-8"))
    assert payload["schema_version"] >= 1
    assert payload["updated_at"]


def test_an_override_records_when_it_was_decided(tmp_path):
    store = OverrideStore(tmp_path / "o.json")
    saved = store.set(Override(asset_id="a1", route_class="flagship"))
    assert saved.decided_at
