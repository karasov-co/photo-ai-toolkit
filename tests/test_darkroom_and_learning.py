"""The eighteen guarantees the darkroom and the learning loop must keep.

Two failure modes run through all of them. An edit assistant destroys work by
overwriting a sidecar or by "correcting" a deliberate choice. A learning loop
destroys work by teaching itself the owner's current blind spots and then acting
on them. Every test here is one of those.
"""

import json

import numpy as np
import pytest
from synthetic import photo_like, write_jpeg

import active_learning
import edit_schema
import model_monitoring
import preference_model
import raw_measurements
import recipe_generator
import recipe_validator
import selective_policy
from artistic import ArtisticScores, IntentSignal
from edit_schema import Crop, Detail, EditRecipe, GlobalAdjustments, Variant
from exporters import adobe_xmp, darktable_xmp, rawtherapee_pp3
from preference_store import Decision, PreferenceStore, Signal
from renderers import base as renderer_base
from renderers.builtin import BuiltinRenderer, _inscribed_rect


def recipe(**kwargs) -> EditRecipe:
    base = EditRecipe(
        asset_id="a1", asset_key="a/1.RW2", source_checksum="c" * 64,
        variant=Variant.FAITHFUL.value, intent="preserve_low_key_mood",
    )
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def image(seed=1, size=(240, 180)):
    return photo_like(*size, seed=seed)


# --- 1. an existing sidecar is never overwritten -----------------------------


def test_an_existing_sidecar_is_never_overwritten_by_default(tmp_path):
    raw = tmp_path / "P1042675.RW2"
    raw.write_bytes(b"raw bytes")
    mine = tmp_path / "P1042675.xmp"
    mine.write_text("<x:xmpmeta>my own edits</x:xmpmeta>", encoding="utf-8")

    import media

    plan = adobe_xmp.plan_apply(recipe(), raw, current_checksum=media.checksum_file(raw))

    assert plan.exists
    assert plan.would_overwrite
    assert not plan.safe
    assert mine.read_text(encoding="utf-8") == "<x:xmpmeta>my own edits</x:xmpmeta>"


def test_suggestions_are_written_where_no_converter_looks(tmp_path):
    """`<stem>.xmp` is what Lightroom reads; `.ai-suggested.xmp` is not."""
    path = adobe_xmp.write_suggestion(recipe(), tmp_path)
    assert edit_schema.SUGGESTION_INFIX in path.name
    assert path.parent.parent.name == "suggestions"
    assert not (tmp_path / "P1042675.xmp").exists()


def test_the_diff_shows_what_would_change_before_anything_is_written(tmp_path):
    raw = tmp_path / "shot.RW2"
    raw.write_bytes(b"raw")
    existing = recipe()
    existing.global_adjustments = GlobalAdjustments(exposure_ev=0.0, shadows=0)
    (tmp_path / "shot.xmp").write_text(adobe_xmp.to_adobe_xmp(existing), encoding="utf-8")

    proposed = recipe()
    proposed.global_adjustments = GlobalAdjustments(exposure_ev=0.8, shadows=30)

    import media

    plan = adobe_xmp.plan_apply(proposed, raw, current_checksum=media.checksum_file(raw))
    joined = "\n".join(plan.diff)
    assert "crs:Exposure2012" in joined
    assert "crs:Shadows2012" in joined


# --- 2. a recipe is bound to the bytes it came from --------------------------


def test_a_recipe_is_stale_once_the_source_changes():
    assert recipe().is_stale_for("a different checksum")
    assert recipe().matches("c" * 64)


def test_applying_a_stale_recipe_is_refused(tmp_path):
    raw = tmp_path / "shot.RW2"
    raw.write_bytes(b"the file as it is now")

    import media

    plan = adobe_xmp.plan_apply(recipe(), raw, current_checksum=media.checksum_file(raw))
    assert plan.stale
    assert not plan.safe


def test_a_recipe_round_trips_through_disk(tmp_path):
    original = recipe(preserve=["the low-key structure"], warnings=["do not lift the shadows"])
    original.global_adjustments = GlobalAdjustments(exposure_ev=0.35, highlights=-32)
    path = edit_schema.write_recipe(original, tmp_path)
    restored = edit_schema.read_recipe(path)

    assert restored.global_adjustments.exposure_ev == 0.35
    assert restored.global_adjustments.highlights == -32
    assert restored.preserve == ["the low-key structure"]
    assert restored.source_checksum == original.source_checksum


# --- 3. auto-enhance does not create new clipping ----------------------------


def test_an_edit_that_blows_the_highlights_is_vetoed():
    original = image()
    blown = np.asarray(original.convert("RGB"), dtype=np.float64)
    blown[:] = 255.0
    from PIL import Image

    result = recipe_validator.validate(original, Image.fromarray(blown.astype(np.uint8)), recipe())
    assert not result.ok
    assert any("new_clipping" in r for r in result.reasons)


def test_a_gentle_edit_passes_validation():
    original = image()
    gentle = np.clip(np.asarray(original.convert("RGB"), dtype=np.float64) * 1.03, 0, 255)
    from PIL import Image

    result = recipe_validator.validate(original, Image.fromarray(gentle.astype(np.uint8)), recipe())
    assert result.ok


def test_setting_a_black_point_is_not_treated_as_damage():
    """Blacks is a normal slider; sharing a threshold with highlights vetoed it."""
    assert recipe_validator.MAX_NEW_CRUSHING > recipe_validator.MAX_NEW_CLIPPING


# --- 4. a low-key frame does not become flat HDR -----------------------------


def test_flattening_a_low_key_frame_is_vetoed():
    from PIL import Image

    dark = np.asarray(image(), dtype=np.float64) * 0.18
    lifted = np.clip(dark * 3.4 + 60, 0, 255)

    result = recipe_validator.validate(
        Image.fromarray(dark.astype(np.uint8)),
        Image.fromarray(lifted.astype(np.uint8)),
        recipe(preserve=["the low-key structure and the weight of the shadows"]),
    )
    assert not result.ok
    assert any("low_key_flattened" in r for r in result.reasons)


def test_a_low_key_frame_is_given_only_a_token_shadow_lift():
    lift = recipe_generator._shadows_for(
        mean_luma=40.0, raw_stats=raw_measurements.RawMeasurements(),
        preserve=["the low-key structure and the weight of the shadows"],
    )
    assert 0 < lift <= 10


def test_an_ordinary_dark_frame_is_lifted_properly():
    lift = recipe_generator._shadows_for(
        mean_luma=40.0, raw_stats=raw_measurements.RawMeasurements(available=True,
                                                                  shadow_headroom_stops=3.0),
        preserve=[],
    )
    assert lift > 15


# --- 5. intentional blur is never sharpened ----------------------------------


def test_sharpening_a_frame_whose_blur_is_the_subject_is_vetoed():
    original = image()
    result = recipe_validator.validate(
        original, original,
        recipe(preserve=["the intentional blur, which is carrying the frame"],
               detail=Detail(sharpening=25)),
    )
    assert not result.ok
    assert any("sharpening_intentional_blur" in r for r in result.reasons)


def test_the_generator_proposes_no_sharpening_when_blur_is_protected():
    sharpening, masking = recipe_generator._sharpening_for(
        noise=1.0, preserve=["the intentional blur, which is carrying the frame"]
    )
    assert sharpening == 0
    assert masking == 0


def test_intent_signals_become_a_preserve_list():
    signals = [
        IntentSignal("motion blur", "likely_intentional", 0.7, "blur is directional"),
        IntentSignal("tilted horizon", "likely_intentional", 0.6, "9 degrees"),
    ]
    preserve, warnings = recipe_generator._preserve_from(signals)
    assert any("blur" in p for p in preserve)
    assert any("tilt" in p for p in preserve)
    assert any("Do not sharpen" in w for w in warnings)
    assert any("Do not straighten" in w for w in warnings)


def test_an_undecided_signal_protects_nothing():
    """Only a positive finding creates an obligation."""
    preserve, _ = recipe_generator._preserve_from(
        [IntentSignal("softness", "cannot_tell", 0.4, "")]
    )
    assert preserve == []


# --- 6. a low-confidence crop is refused -------------------------------------


def test_a_crop_below_the_confidence_floor_is_refused():
    original = image()
    result = recipe_validator.validate(
        original, original,
        recipe(geometry=edit_schema.Geometry(crop=Crop(0.1, 0.1, 0.9, 0.9, confidence=0.3))),
    )
    assert not result.ok
    assert any("low_confidence_crop" in r for r in result.reasons)


def test_an_excessive_crop_is_refused_even_when_confident():
    original = image()
    result = recipe_validator.validate(
        original, original,
        recipe(geometry=edit_schema.Geometry(crop=Crop(0.3, 0.3, 0.7, 0.7, confidence=0.95))),
    )
    assert not result.ok
    assert any("excessive_crop" in r for r in result.reasons)


def test_the_generator_never_claims_a_confident_crop():
    """Nothing here can compose a photograph, and the number says so."""
    recipes = recipe_generator.generate(
        asset_id="a", asset_key="a", checksum="c", raw_stats=raw_measurements.RawMeasurements(),
        mean_luma=120.0, stddev_luma=50.0, channel_means=(120.0, 118.0, 116.0), noise=1.0,
        tilt_degrees=0.0, intent_signals=[], is_raw=False,
    )
    assert all(r.confidence.crop < recipe_validator.MIN_CROP_CONFIDENCE for r in recipes)


# --- 7. geometry is stored after orientation ---------------------------------


def test_a_crop_is_stored_normalised_so_no_decoder_can_change_it():
    crop = Crop(left=0.1, top=0.0, right=0.9, bottom=1.0, aspect_ratio="4:5")
    assert 0.0 <= crop.left < crop.right <= 1.0
    assert crop.keeps == pytest.approx(0.8)
    assert "keeps" in crop.to_dict()


def test_rotation_crops_to_the_inscribed_rectangle():
    """Without this, black corners were counted as newly crushed shadows."""
    width, height = _inscribed_rect(1000, 800, 5.0)
    assert width < 1000 and height < 800
    assert width > 800 and height > 600


def test_no_rotation_keeps_the_whole_frame():
    assert _inscribed_rect(1000, 800, 0.0) == (1000, 800)


# --- 8. rendering is deterministic -------------------------------------------


def test_the_same_recipe_renders_identically_twice(tmp_path):
    source = write_jpeg(photo_like(400, 300, seed=7), tmp_path / "frame.jpg")
    engine = BuiltinRenderer()
    proposal = recipe()
    proposal.global_adjustments = GlobalAdjustments(exposure_ev=0.4, contrast=10, shadows=20)

    first = np.asarray(engine.render(source, proposal, max_px=300))
    second = np.asarray(engine.render(source, proposal, max_px=300))
    assert np.array_equal(first, second)


def test_a_different_recipe_renders_differently(tmp_path):
    source = write_jpeg(photo_like(400, 300, seed=7), tmp_path / "frame.jpg")
    engine = BuiltinRenderer()
    a, b = recipe(), recipe()
    a.global_adjustments = GlobalAdjustments(exposure_ev=0.0)
    b.global_adjustments = GlobalAdjustments(exposure_ev=0.8)
    assert not np.array_equal(
        np.asarray(engine.render(source, a, max_px=300)),
        np.asarray(engine.render(source, b, max_px=300)),
    )


def test_an_unavailable_engine_is_reported_rather_than_substituted():
    with pytest.raises(renderer_base.EngineUnavailable):
        renderer_base.get("no-such-engine")


# --- 9. Adobe and darktable sidecars never mix -------------------------------


def test_adobe_and_darktable_schemas_stay_separate():
    """One file claiming both namespaces is read correctly by neither."""
    adobe = adobe_xmp.to_adobe_xmp(recipe())
    dark = darktable_xmp.to_darktable_xmp(recipe())

    assert "crs:" in adobe and "darktable:" not in adobe
    assert "darktable:" in dark and "crs:" not in dark


def test_rawtherapee_writes_an_ini_not_xmp():
    profile = rawtherapee_pp3.to_pp3(recipe())
    assert "[Exposure]" in profile
    assert "<x:xmpmeta" not in profile


def test_a_temperature_delta_is_not_written_as_an_absolute_kelvin():
    """crs:Temperature is absolute; writing a delta there wrecks the frame."""
    proposal = recipe()
    proposal.global_adjustments = GlobalAdjustments(temperature_delta_k=-480)
    document = adobe_xmp.to_adobe_xmp(proposal)
    assert "pat:temperatureDeltaK" in document
    assert "crs:Temperature=" not in document


def test_every_sidecar_carries_the_source_checksum():
    for render in (adobe_xmp.to_adobe_xmp, darktable_xmp.to_darktable_xmp, rawtherapee_pp3.to_pp3):
        assert "c" * 64 in render(recipe())


# --- 10. an unmeasured dimension stays None ----------------------------------


def test_raw_headroom_for_a_jpeg_is_unknown_not_zero(tmp_path):
    source = write_jpeg(photo_like(200, 150), tmp_path / "frame.jpg")
    measured = raw_measurements.measure_or_empty(source, is_raw=False)
    assert not measured.available
    assert "unknown, not zero" in measured.reason


def test_unmeasured_artistic_dimensions_are_none():
    scores = ArtisticScores()
    assert scores.emotional_resonance is None
    assert scores.documentary_significance is None


def test_a_jpeg_gets_a_cautious_highlight_default_rather_than_a_confident_one():
    unknown = recipe_generator._highlights_for(raw_measurements.RawMeasurements(), is_raw=False)
    nothing_to_recover = recipe_generator._highlights_for(
        raw_measurements.RawMeasurements(available=True, clipped_any_channel=0.0), is_raw=True
    )
    assert unknown < 0
    assert nothing_to_recover == 0


# --- 11. a restore is a loud signal ------------------------------------------


def test_a_restore_outweighs_every_other_signal():
    weights = preference_model.__dict__  # noqa: F841 - readability only
    restore = Decision(signal=Signal.RESTORED_FROM_QUARANTINE.value, asset_id="a")
    quick = Decision(signal=Signal.QUICK_REJECT.value, asset_id="b")
    assert restore.weight > quick.weight * 5


def test_absence_of_action_teaches_nothing():
    assert Decision(signal=Signal.NOT_OPENED.value, asset_id="a").weight == 0.0


def test_a_restore_is_recorded_as_a_correction(tmp_path):
    store = PreferenceStore(tmp_path / "p.jsonl")
    store.record(
        Decision(signal=Signal.RESTORED_FROM_QUARANTINE.value, asset_id="a1",
                 tool_said="trash", answer="keep")
    )
    assert len(store.corrections()) == 1


# --- 12. an unfamiliar genre forces abstention -------------------------------


def test_an_unseen_genre_makes_the_model_abstain(tmp_path):
    store = PreferenceStore(tmp_path / "p.jsonl")
    for i in range(80):
        store.record(
            Decision(signal=Signal.PAIRWISE_KEEP.value, winner=f"w{i}", loser=f"l{i}",
                     genre="landscape")
        )
    model = preference_model.fit(store)

    assert model.knows_genre("landscape")
    prediction = model.predict("w1", genre="underwater")
    assert prediction.abstained
    assert not prediction.in_distribution
    assert "unfamiliar" in prediction.reason


def test_too_few_decisions_means_no_prediction(tmp_path):
    store = PreferenceStore(tmp_path / "p.jsonl")
    store.record(Decision(signal=Signal.PAIRWISE_KEEP.value, winner="a", loser="b"))
    prediction = preference_model.fit(store).predict("a")
    assert prediction.abstained


def test_a_prediction_near_the_boundary_abstains(tmp_path):
    store = PreferenceStore(tmp_path / "p.jsonl")
    for i in range(60):
        store.record(
            Decision(signal=Signal.PAIRWISE_KEEP.value, winner=f"w{i}", loser=f"l{i}", genre="street")
        )
    model = preference_model.fit(store)
    assert model.predict("never-seen-before", genre="street").abstained


# --- 13. disagreement always goes to review ----------------------------------


def test_disagreement_between_the_personal_model_and_the_prior_is_detected():
    keeps = preference_model.Prediction("a", probability=0.9, abstained=False)
    assert preference_model.disagreement(keeps, curator_says_keep=False)
    assert not preference_model.disagreement(keeps, curator_says_keep=True)


def test_an_abstaining_model_cannot_disagree():
    assert not preference_model.disagreement(
        preference_model.Prediction("a", abstained=True), curator_says_keep=False
    )


def test_disagreement_holds_the_asset_back():
    model = preference_model.PersonalModel(decisions=5000, genres={"street": 500})
    decision = selective_policy.decide(
        asset_id="a1", route_class="review", technical_evidence="",
        prediction=preference_model.Prediction("a1", probability=0.02, abstained=False),
        model=model,
        artistic_scores=ArtisticScores(emotional_resonance=90, curatorial_uncertainty=20),
        genre="street", holdout_checks=10_000, shadow_mode=False,
    )
    assert decision.bucket != selective_policy.Bucket.SAFE_QUARANTINE_CANDIDATE.value


# --- 14. the personal model cannot override an artistic rescue ---------------


def test_an_artistic_signal_blocks_the_personal_model_entirely():
    model = preference_model.PersonalModel(decisions=5000, genres={"street": 500})
    decision = selective_policy.decide(
        asset_id="a1", route_class="review", technical_evidence="",
        prediction=preference_model.Prediction("a1", probability=0.01, abstained=False),
        model=model,
        artistic_scores=ArtisticScores(emotional_resonance=95, curatorial_uncertainty=15),
        genre="street", holdout_checks=10_000, shadow_mode=False,
    )
    assert not decision.acts_on_files
    assert any("no_artistic_rescue" in gate for gate in decision.failed_gates)


# --- 15. no model can cause a purge ------------------------------------------


def test_no_policy_bucket_can_reach_a_purge():
    import quarantine

    for bucket in selective_policy.Bucket:
        assert not quarantine.is_purgeable_evidence(bucket.value)


def test_the_strongest_thing_the_policy_can_do_is_quarantine():
    acting = [b for b in selective_policy.Bucket if b not in selective_policy.NON_ACTING]
    assert acting == [selective_policy.Bucket.SAFE_QUARANTINE_CANDIDATE]


def test_technical_evidence_is_the_only_path_that_needs_no_model():
    decision = selective_policy.decide(
        asset_id="a1", route_class="trash", technical_evidence="corrupt_file",
        prediction=preference_model.Prediction("a1"),
        model=preference_model.PersonalModel(),
        artistic_scores=ArtisticScores(), shadow_mode=True,
    )
    assert decision.acts_on_files
    assert "corrupt_file" in decision.reasons[0]


# --- 16. shadow mode moves nothing -------------------------------------------


def test_shadow_mode_holds_everything_back():
    model = preference_model.PersonalModel(decisions=5000, genres={"street": 500})
    decision = selective_policy.decide(
        asset_id="a1", route_class="review", technical_evidence="",
        prediction=preference_model.Prediction("a1", probability=0.01, abstained=False),
        model=model, artistic_scores=ArtisticScores(curatorial_uncertainty=10),
        genre="street", holdout_checks=10_000, shadow_mode=True,
    )
    assert not decision.acts_on_files
    assert any("shadow" in gate.lower() for gate in decision.failed_gates)


def test_every_gate_is_recorded_so_the_refusal_is_auditable():
    decision = selective_policy.decide(
        asset_id="a1", route_class="review", technical_evidence="",
        prediction=preference_model.Prediction("a1"),
        model=preference_model.PersonalModel(),
        artistic_scores=ArtisticScores(curatorial_uncertainty=10), shadow_mode=True,
    )
    assert len(decision.gates) >= 8
    assert decision.failed_gates


# --- 17. an audit sample is always held back ---------------------------------


def test_a_deterministic_slice_is_always_held_back_for_checking():
    sampled = [selective_policy._in_audit_sample(f"asset{i}") for i in range(400)]
    fraction = sum(sampled) / len(sampled)
    assert 0.01 < fraction < 0.12


def test_the_audit_sample_is_stable_across_runs():
    """A reshuffling audit is an unreadable audit."""
    first = [selective_policy._in_audit_sample(f"a{i}") for i in range(100)]
    second = [selective_policy._in_audit_sample(f"a{i}") for i in range(100)]
    assert first == second


# --- 18. drift switches automation off ---------------------------------------


def test_drift_turns_automation_off_by_itself(tmp_path):
    monitor = model_monitoring.Monitor(tmp_path / "m.json")
    monitor.state.automation_enabled = True
    for i in range(60):
        monitor.observe(
            model_monitoring.Observation(
                asset_id=f"a{i}", predicted="keep", actual="keep",
                confidence=0.9, in_distribution=False,
            )
        )
    report = monitor.evaluate()
    assert not report["automation_enabled"]
    assert "outside the model's experience" in report["disabled_reason"]


def test_a_false_trash_turns_automation_off(tmp_path):
    monitor = model_monitoring.Monitor(tmp_path / "m.json")
    monitor.state.automation_enabled = True
    for i in range(50):
        monitor.observe(
            model_monitoring.Observation(asset_id=f"a{i}", predicted="keep", actual="keep",
                                         confidence=0.9)
        )
    monitor.observe(
        model_monitoring.Observation(asset_id="bad", predicted="trash", actual="restored",
                                     confidence=0.95)
    )
    report = monitor.evaluate()
    assert report["false_trash_rate"] > 0
    assert not report["automation_enabled"]


def test_automation_cannot_be_enabled_without_enough_evidence(tmp_path):
    monitor = model_monitoring.Monitor(tmp_path / "m.json")
    granted, message = monitor.enable_automation(holdout_checks=100)
    assert not granted
    assert "3000" in message


def test_automation_can_be_enabled_with_enough_clean_evidence(tmp_path):
    monitor = model_monitoring.Monitor(tmp_path / "m.json")
    for i in range(60):
        monitor.observe(
            model_monitoring.Observation(asset_id=f"a{i}", predicted="keep", actual="keep",
                                         confidence=0.9)
        )
    granted, _ = monitor.enable_automation(holdout_checks=5000)
    assert granted


def test_lost_monitoring_state_leaves_automation_off(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json")
    monitor = model_monitoring.Monitor(path)
    assert not monitor.state.automation_enabled
    assert "lost" in monitor.state.disabled_reason


# --- active learning ----------------------------------------------------------


class FakeRecord:
    def __init__(self, **kwargs):
        self.filename = kwargs.get("filename", "a.jpg")
        self.asset_id = kwargs.get("asset_id", "a1")
        self.genre = kwargs.get("genre", "street")
        self.artistic = kwargs.get("artistic", {})
        self.cluster_id = kwargs.get("cluster_id", "")
        self.cluster_size = kwargs.get("cluster_size", 1)
        self.scores = kwargs.get("scores", {})


def test_an_undecidable_defect_is_the_most_valuable_question():
    records = [
        FakeRecord(
            artistic={"intent_signals": [{"defect": "motion blur", "verdict": "cannot_tell"}]}
        )
    ]
    questions = active_learning.propose(records, preference_model.PersonalModel())
    assert questions[0].kind == "intent"
    assert "deliberate" in questions[0].options


def test_a_near_tie_in_a_burst_is_worth_more_than_a_clear_gap():
    close = [
        FakeRecord(asset_id="a", cluster_id="c", cluster_size=2, scores={"current_quality": 60}),
        FakeRecord(asset_id="b", cluster_id="c", cluster_size=2, scores={"current_quality": 62}),
    ]
    apart = [
        FakeRecord(asset_id="x", cluster_id="d", cluster_size=2, scores={"current_quality": 20}),
        FakeRecord(asset_id="y", cluster_id="d", cluster_size=2, scores={"current_quality": 80}),
    ]
    close_value = active_learning._burst_pairs(close)[0].value
    apart_value = active_learning._burst_pairs(apart)[0].value
    assert close_value > apart_value


def test_the_answer_format_is_buttons_not_a_rating():
    records = [FakeRecord(genre="never-seen")]
    for question in active_learning.propose(records, preference_model.PersonalModel()):
        assert 3 <= len(question.options) <= 4
        assert all(not option.isdigit() for option in question.options)


def test_no_question_is_asked_twice():
    records = [FakeRecord(asset_id="a1", genre="unseen") for _ in range(5)]
    questions = active_learning.propose(records, preference_model.PersonalModel())
    keys = [(q.kind, tuple(q.assets)) for q in questions]
    assert len(keys) == len(set(keys))


# --- the end-to-end shape -----------------------------------------------------


def test_the_policy_reports_how_much_attention_it_saves():
    decisions = [
        selective_policy.PolicyDecision(bucket="auto_keep"),
        selective_policy.PolicyDecision(bucket="burst_winner"),
        selective_policy.PolicyDecision(bucket="manual_review"),
    ]
    summary = selective_policy.summarise(decisions)
    assert summary["needs_full_human_decision"] == 1
    assert summary["automated_fraction"] == pytest.approx(2 / 3, abs=0.01)


def test_the_recipe_describes_itself_in_sliders_a_person_can_type():
    proposal = recipe()
    proposal.global_adjustments = GlobalAdjustments(exposure_ev=0.35, highlights=-32, shadows=14)
    described = " ".join(edit_schema.describe(proposal))
    assert "Exposure +0.35 EV" in described
    assert "Highlights -32" in described


def test_a_no_op_recipe_says_so():
    assert "No adjustment" in edit_schema.describe(recipe())[0]


def test_recipes_are_json_serialisable(tmp_path):
    proposal = recipe(preserve=["shadows"], warnings=["careful"])
    json.dumps(proposal.to_dict())
