"""The fifteen scenarios that must never regress.

Each of these is a way a culling tool destroys a photograph it should have kept.
The asymmetry is the whole point: a false `trash` is unrecoverable, a false
`review` costs a few seconds of attention.
"""

import numpy as np
import pytest
from synthetic import blurred, dark_but_recoverable, near_black, photo_like, write_jpeg

from photoai import artistic, pipeline, prompts, scoring
from photoai import quarantine as quarantine_module
from photoai.artistic import ArtisticScores, ArtRoute, IntentSignal, apply_conservative_art
from photoai.issues import IssueCode, IssueSet
from photoai.media import FileState
from photoai.scoring import RouteClass, ScoreInput


def array(image):
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def directional_smear(image, length=16):
    """Motion along one axis, which is what panning and ICM actually produce."""
    src = np.asarray(image.convert("RGB"), dtype=np.float64)
    out = np.zeros_like(src)
    for shift in range(length):
        out += np.roll(src, shift, axis=1)
    return out / length


def run_on(tmp_path, files: dict):
    root = tmp_path / "archive"
    for name, image in files.items():
        write_jpeg(image, root / name)
    return pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))


def by_name(result, name):
    return next(r for r in result.records if r.filename == name)


# --- 1. darkness is not grounds for destruction ------------------------------


def test_a_dark_emotional_frame_is_not_trashed_for_its_exposure(tmp_path):
    result = run_on(tmp_path, {"dusk.jpg": dark_but_recoverable(seed=3, size=(1600, 1200))})
    assert by_name(result, "dusk.jpg").route_class != RouteClass.TRASH.value


def test_darkness_alone_never_appears_as_an_unrecoverable_issue(tmp_path):
    record = by_name(
        run_on(tmp_path, {"dusk.jpg": dark_but_recoverable(seed=3, size=(1600, 1200))}), "dusk.jpg"
    )
    assert not any("underexposed" in i for i in record.issues["unrecoverable"])


def test_a_low_key_frame_is_read_as_deliberate_not_underexposed():
    """A low-key frame keeps a legible bright region; an underexposed one does not."""
    low_key = array(photo_like(600, 400, seed=2)) * 0.28
    low_key[:200, :200] = 210.0  # a legible bright region, as a low-key frame has

    signals = artistic.assess_intent(
        low_key, blur_ratio=40.0, sharpness_global=400.0, sharpness_tile=500.0,
        tilt_degrees=0.0, clipped_highlights=0.0, mean_luma=float(artistic._luma(low_key).mean()),
    )
    darkness = [s for s in signals if s.defect == "darkness"]
    assert darkness and darkness[0].verdict == "likely_intentional"


# --- 2. intentional motion blur ---------------------------------------------


def test_directional_blur_is_told_apart_from_camera_shake():
    """Panning smears along one axis; a shaky hand smears in every direction."""
    for seed in (2, 4, 7):
        panned = artistic.blur_anisotropy(directional_smear(photo_like(600, 400, seed=seed)))[0]
        shaken = artistic.blur_anisotropy(array(blurred(seed=seed, radius=6.0, size=(600, 400))))[0]
        sharp = artistic.blur_anisotropy(array(photo_like(600, 400, seed=seed)))[0]

        assert panned > artistic.DIRECTIONAL_BLUR_ANISOTROPY, seed
        assert shaken < artistic.DIRECTIONAL_BLUR_ANISOTROPY, seed
        assert sharp < artistic.DIRECTIONAL_BLUR_ANISOTROPY, seed


def test_an_undecidable_blur_abstains_instead_of_condemning():
    """The gate is precise, not sensitive; a missed pan must still be kept."""
    shaken = array(blurred(seed=4, radius=6.0, size=(600, 400)))
    signals = artistic.assess_intent(
        shaken, blur_ratio=1.5, sharpness_global=20.0, sharpness_tile=22.0,
        tilt_degrees=0.0, clipped_highlights=0.0, mean_luma=120.0,
    )
    softness = [s for s in signals if s.defect == "softness"]
    assert softness and softness[0].verdict == "cannot_tell"
    assert softness[0].rescues, "abstaining must never condemn the frame"


def test_intentional_motion_blur_is_not_an_automatic_severe_failure():
    signals = artistic.assess_intent(
        directional_smear(photo_like(600, 400, seed=4)),
        blur_ratio=1.5, sharpness_global=20.0, sharpness_tile=25.0,
        tilt_degrees=0.0, clipped_highlights=0.0, mean_luma=120.0,
    )
    motion = [s for s in signals if "motion" in s.defect]
    assert motion and motion[0].verdict == "likely_intentional"
    assert artistic.intentionality_score(signals) >= 55


def test_selective_focus_is_not_a_missed_one():
    signals = artistic.assess_intent(
        array(photo_like(600, 400)), blur_ratio=3.0,
        sharpness_global=166.0, sharpness_tile=1401.0,
        tilt_degrees=0.0, clipped_highlights=0.0, mean_luma=120.0,
    )
    assert signals[0].verdict == "likely_intentional"
    assert "selective focus" in signals[0].evidence


# --- 3. grain ----------------------------------------------------------------


def test_a_grainy_night_frame_can_still_read_as_deliberate():
    signals = artistic.assess_intent(
        array(photo_like(600, 400)) * 0.3, blur_ratio=20.0,
        sharpness_global=300.0, sharpness_tile=400.0, tilt_degrees=0.0,
        clipped_highlights=0.0, mean_luma=48.0, iso=6400,
    )
    grain = [s for s in signals if s.defect == "grain"]
    assert grain and grain[0].verdict == "likely_intentional"
    assert "the cost of the picture existing" in grain[0].evidence


# --- 4. tilt -----------------------------------------------------------------


def test_a_pronounced_tilt_reads_as_a_choice_rather_than_carelessness():
    signals = artistic.assess_intent(
        array(photo_like(600, 400)), blur_ratio=40.0, sharpness_global=400.0,
        sharpness_tile=500.0, tilt_degrees=9.0, clipped_highlights=0.0, mean_luma=120.0,
    )
    tilt = [s for s in signals if "tilt" in s.defect]
    assert tilt and tilt[0].verdict == "likely_intentional"


def test_a_slight_tilt_is_not_claimed_either_way():
    signals = artistic.assess_intent(
        array(photo_like(600, 400)), blur_ratio=40.0, sharpness_global=400.0,
        sharpness_tile=500.0, tilt_degrees=1.4, clipped_highlights=0.0, mean_luma=120.0,
    )
    tilt = [s for s in signals if "tilt" in s.defect]
    assert tilt and tilt[0].verdict == "cannot_tell"


# --- 5 & 6. crop, emptiness, and the one frame that IS empty -----------------


def test_an_empty_minimal_frame_is_not_the_same_as_a_lens_cap(tmp_path):
    """Both are 'nothing much'. Only one of them has no information at all."""
    minimal = np.full((1200, 1600, 3), 150.0)
    minimal[900:960, 200:260] = 40.0  # a single small dark element
    from PIL import Image

    result = run_on(
        tmp_path,
        {
            "minimal.jpg": Image.fromarray(minimal.astype(np.uint8)),
            "lens_cap.jpg": near_black(1600, 1200),
        },
    )
    assert by_name(result, "lens_cap.jpg").route_class == RouteClass.TRASH.value
    minimal_record = by_name(result, "minimal.jpg")
    assert minimal_record.route_class != RouteClass.TRASH.value
    assert not minimal_record.issues["unrecoverable"]


def test_a_lens_cap_frame_has_no_intent_signal_to_rescue_it():
    black = np.zeros((400, 600, 3))
    signals = artistic.assess_intent(
        black, blur_ratio=0.0, sharpness_global=0.0, sharpness_tile=0.0,
        tilt_degrees=0.0, clipped_highlights=0.0, mean_luma=0.0,
    )
    darkness = [s for s in signals if s.defect == "darkness"]
    assert darkness and darkness[0].verdict != "likely_intentional"


# --- 7. unpleasant emotion ---------------------------------------------------


def rubric() -> str:
    """The prompt with line wrapping normalised, so phrases match across breaks."""
    return " ".join(prompts.STAGE3_SYSTEM.lower().split())


def test_the_rubric_forbids_penalising_discomfort():
    text = rubric()
    assert "do not treat discomfort as failure" in text
    assert "an unpleasant emotion is still an emotion" in text


def test_the_rubric_forbids_scoring_by_convention():
    text = rubric()
    for convention in ("rule of thirds", "symmetry", "level horizon", "clean background"):
        assert convention in text
    assert "do not award points for the rule of thirds" in text


def test_the_rubric_forbids_equating_technical_perfection_with_worth():
    text = rubric()
    assert "do not treat technical perfection as evidence of artistic strength" in text
    assert "sharpness is not a virtue" in text


def test_the_rubric_forbids_prestige_vocabulary():
    text = rubric()
    assert "do not call anything genius, masterful or iconic" in text


def test_the_rubric_demands_visible_evidence_rather_than_jargon():
    text = rubric()
    assert "the rule of thirds is broken" in text and "is not an argument" in text
    assert "every judgement must rest on something visible" in text


def test_the_rubric_forbids_inventing_intention():
    assert "do not invent the photographer's intention as though it were fact" in rubric()


def test_the_rubric_forbids_penalising_commercial_uselessness():
    text = rubric()
    assert "never lower a frame because it is hard to place" in text


def test_conventional_beauty_is_kept_separate_from_every_other_dimension():
    assert "must not raise or lower any score above" in prompts.STAGE3_SYSTEM


def test_a_note_that_argues_from_a_textbook_is_detectable():
    assert prompts.reads_like_a_textbook("Breaks the rule of thirds")
    assert prompts.reads_like_a_textbook("A masterful, iconic frame")
    assert not prompts.reads_like_a_textbook(
        "The figure at the extreme edge makes the space feel like pressure"
    )


# --- 8. commercially useless is not worthless --------------------------------


def test_stock_and_artistic_signals_are_not_mutually_exclusive():
    scores = ArtisticScores(
        emotional_resonance=88, documentary_significance=80, conventional_beauty=10
    )
    assert scores.has_any_artistic_signal
    assert scores.conventional_beauty == 10


def test_low_conventional_beauty_cannot_by_itself_reach_the_destructive_path():
    scores = ArtisticScores(conventional_beauty=2, emotional_resonance=90, curatorial_uncertainty=20)
    route, reason = apply_conservative_art(ArtRoute.TECHNICAL_FAILURE.value, scores)
    assert route == ArtRoute.REVIEW.value
    assert reason


# --- 9. burst: sharper is not automatically better ---------------------------


def test_near_identical_frames_are_alternatives_rather_than_a_winner_and_losers():
    frames = [
        {"quality": 61.0, "moment_specificity": 40},
        {"quality": 59.5, "moment_specificity": 85},
    ]
    assert artistic.similar_alternatives(frames, quality_margin=5.0)


def test_a_real_quality_gap_is_still_a_real_gap():
    frames = [{"quality": 70.0, "moment_specificity": 50}, {"quality": 30.0, "moment_specificity": 55}]
    assert not artistic.similar_alternatives(frames, quality_margin=5.0)


def test_a_technically_weaker_frame_with_a_better_moment_is_not_condemned(tmp_path):
    """Two takes of one scene, technically within noise of each other."""
    result = run_on(
        tmp_path,
        {
            "take_a.jpg": photo_like(1600, 1200, seed=21),
            "take_b.jpg": photo_like(1600, 1200, seed=21),
        },
    )
    assert all(r.route_class != RouteClass.TRASH.value for r in result.records)


# --- 10. low confidence goes to review --------------------------------------


def test_high_uncertainty_routes_to_review_not_to_trash():
    scores = ArtisticScores(curatorial_uncertainty=85)
    route, reason = apply_conservative_art(ArtRoute.TECHNICAL_FAILURE.value, scores)
    assert route == ArtRoute.REVIEW.value
    assert "uncertainty" in reason


def test_undecided_signals_raise_uncertainty():
    undecided = [IntentSignal("softness", "cannot_tell", 0.4, "")]
    decided = [IntentSignal("softness", "likely_intentional", 0.8, "")]
    assert artistic.uncertainty_score(undecided, semantic_present=True) > artistic.uncertainty_score(
        decided, semantic_present=True
    )


def test_no_vision_pass_means_higher_uncertainty():
    assert artistic.uncertainty_score([], semantic_present=False) > artistic.uncertainty_score(
        [], semantic_present=True
    )


# --- 11. no model means no destruction ---------------------------------------


def test_a_missing_semantic_model_does_not_produce_a_destructive_classification(tmp_path):
    """The whole offline run: nothing may be trashed on an aesthetic basis."""
    result = run_on(
        tmp_path,
        {
            "a.jpg": photo_like(1600, 1200, seed=31),
            "b.jpg": dark_but_recoverable(seed=32, size=(1600, 1200)),
            "c.jpg": photo_like(1600, 1200, seed=33),
        },
    )
    for record in result.records:
        assert record.semantic_present is False
        if record.route_class == RouteClass.TRASH.value:
            assert record.issues["unrecoverable"], record.filename


def test_unmeasured_artistic_dimensions_are_none_not_zero():
    """A dimension nobody looked at must not read as one that scored badly."""
    scores = ArtisticScores()
    assert scores.emotional_resonance is None
    assert scores.documentary_significance is None
    assert "emotional_resonance" not in scores.measured


# --- 12. stability across runs ------------------------------------------------


def test_repeating_the_run_does_not_change_the_destructive_route(tmp_path):
    files = {
        "a.jpg": photo_like(1600, 1200, seed=41),
        "b.jpg": blurred(seed=42, size=(1600, 1200)),
        "c.jpg": near_black(1600, 1200),
    }
    first = run_on(tmp_path / "one", files)
    second = run_on(tmp_path / "two", files)
    assert {r.filename: r.route_class for r in first.records} == {
        r.filename: r.route_class for r in second.records
    }


# --- 13. one model is not grounds for purge ----------------------------------


@pytest.mark.parametrize(
    "evidence",
    ["", "low_stock_potential", "dead_moment", "weaker_duplicate", "conventional_beauty"],
)
def test_an_aesthetic_judgement_is_never_grounds_for_permanent_deletion(evidence):
    assert not quarantine_module.is_purgeable_evidence(evidence)


@pytest.mark.parametrize("evidence", ["corrupt_file", "empty_frame", "no_usable_segment"])
def test_only_demonstrable_technical_failure_may_be_purged(evidence):
    assert quarantine_module.is_purgeable_evidence(evidence)


def test_purge_refuses_a_quarantined_file_whose_grounds_were_aesthetic(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    victim = archive / "strange.jpg"
    victim.write_bytes(b"a photograph")

    bin_dir = tmp_path / "bin"
    q = quarantine_module.Quarantine(bin_dir, source_roots=[archive])
    q.apply(
        q.plan(
            [
                quarantine_module.PlannedMove(
                    asset_id="a1",
                    files=[victim],
                    destination_dir=bin_dir,
                    reason="model thought it was weak",
                    evidence="dead_moment",
                    states={str(victim): FileState.of(victim).to_dict()},
                )
            ]
        ),
        dry_run=False,
    )

    report = q.purge(
        confirmation=quarantine_module.PURGE_CONFIRMATION, older_than_days=0, dry_run=False
    )
    assert report["purged"] == 0
    assert report["refused"]
    assert (bin_dir / "strange.jpg").exists()


# --- 14. duplicate basenames --------------------------------------------------


def test_two_cards_with_the_same_filename_stay_two_photographs(tmp_path):
    """A good frame once inherited a black frame's measurement and was trashed."""
    root = tmp_path / "archive"
    write_jpeg(photo_like(1600, 1200, seed=1), root / "card_a" / "P1000001.jpg")
    write_jpeg(near_black(1600, 1200), root / "card_b" / "P1000001.jpg")

    result = pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))

    assert len(result.records) == 2
    good = next(r for r in result.records if "card_a" in r.asset_key)
    black = next(r for r in result.records if "card_b" in r.asset_key)
    assert good.scores["current_quality"] > 20
    assert good.route_class != RouteClass.TRASH.value
    assert black.route_class == RouteClass.TRASH.value


def test_asset_keys_are_unique_across_a_scan(tmp_path):
    root = tmp_path / "archive"
    for card in ("card_a", "card_b", "card_c"):
        write_jpeg(photo_like(800, 600, seed=hash(card) % 50), root / card / "P1000001.jpg")
    result = pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))
    keys = [r.asset_key for r in result.records]
    assert len(keys) == len(set(keys)) == 3


# --- 15. a failed move rolls the whole group back ----------------------------


def test_a_failure_partway_through_a_group_restores_every_file(tmp_path, monkeypatch):
    """A RAW in quarantine and its sidecar in the archive is an orphan."""
    archive = tmp_path / "archive"
    archive.mkdir()
    raw = archive / "P1042675.RW2"
    sidecar = archive / "P1042675.xmp"
    raw.write_bytes(b"raw bytes")
    sidecar.write_bytes(b"<xmp/>")

    bin_dir = tmp_path / "bin"
    q = quarantine_module.Quarantine(bin_dir, source_roots=[archive])
    planned = q.plan(
        [
            quarantine_module.PlannedMove(
                asset_id="a1",
                files=[raw, sidecar],
                destination_dir=bin_dir,
                reason="corrupt",
                evidence="corrupt_file",
                states={
                    str(raw): FileState.of(raw).to_dict(),
                    str(sidecar): FileState.of(sidecar).to_dict(),
                },
            )
        ]
    )

    real_move = quarantine_module.shutil.move
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # the sidecar, after the RAW has already moved
            raise OSError("No space left on device")
        return real_move(src, dst)

    monkeypatch.setattr(quarantine_module.shutil, "move", flaky)
    results = q.apply(planned, dry_run=False)

    assert raw.exists(), "the RAW must be put back when its sidecar cannot follow"
    assert sidecar.exists()
    assert not list(bin_dir.rglob("*.RW2"))
    assert any(r.status == quarantine_module.OperationStatus.FAILED.value for r in results)


def test_a_file_edited_after_analysis_is_not_moved(tmp_path):
    """The decision was made about different contents."""
    archive = tmp_path / "archive"
    archive.mkdir()
    victim = archive / "edited.jpg"
    victim.write_bytes(b"original contents")

    bin_dir = tmp_path / "bin"
    q = quarantine_module.Quarantine(bin_dir, source_roots=[archive])
    planned = q.plan(
        [
            quarantine_module.PlannedMove(
                asset_id="a1",
                files=[victim],
                destination_dir=bin_dir,
                reason="corrupt",
                evidence="corrupt_file",
                states={str(victim): FileState.of(victim).to_dict()},
            )
        ]
    )

    victim.write_bytes(b"the user re-exported this in the meantime")
    results = q.apply(planned, dry_run=False)

    assert victim.exists()
    assert results[0].status == quarantine_module.OperationStatus.SKIPPED.value
    assert "changed since analysis" in results[0].error


# --- the structural guarantee -------------------------------------------------


def test_an_artistic_signal_can_never_create_a_destructive_route():
    assert artistic.can_only_rescue(ArtRoute.TECHNICAL_FAILURE.value, ArtRoute.REVIEW.value)
    assert artistic.can_only_rescue(ArtRoute.REVIEW.value, ArtRoute.ART_CANDIDATE.value)
    assert not artistic.can_only_rescue(ArtRoute.REVIEW.value, ArtRoute.TECHNICAL_FAILURE.value)


def test_no_artistic_route_is_destructive():
    assert ArtRoute.TECHNICAL_FAILURE not in artistic.NON_DESTRUCTIVE
    assert len(artistic.NON_DESTRUCTIVE) == len(ArtRoute) - 1


def test_conservative_art_leaves_a_non_destructive_route_alone():
    scores = ArtisticScores(emotional_resonance=90, curatorial_uncertainty=10)
    route, reason = apply_conservative_art(ArtRoute.STOCK_STRONG.value, scores)
    assert route == ArtRoute.STOCK_STRONG.value
    assert reason == ""


def test_conservative_art_can_be_switched_off_but_defaults_on():
    scores = ArtisticScores(emotional_resonance=90, curatorial_uncertainty=10)
    assert apply_conservative_art(ArtRoute.TECHNICAL_FAILURE.value, scores)[0] == ArtRoute.REVIEW.value
    assert (
        apply_conservative_art(ArtRoute.TECHNICAL_FAILURE.value, scores, enabled=False)[0]
        == ArtRoute.TECHNICAL_FAILURE.value
    )


def test_a_frame_with_nothing_going_for_it_is_still_allowed_to_fail():
    """The rescue is evidence-based, not unconditional."""
    scores = ArtisticScores(curatorial_uncertainty=10, intentionality_likelihood=5)
    route, _ = apply_conservative_art(ArtRoute.TECHNICAL_FAILURE.value, scores)
    assert route == ArtRoute.TECHNICAL_FAILURE.value


# --- the ranking inversion ----------------------------------------------------


def test_a_rank_of_one_becomes_the_top_of_the_scale_not_the_bottom():
    assert scoring.rank_to_percentile(1, 12) == 100
    assert scoring.rank_to_percentile(12, 12) == 0


def test_an_unknown_group_size_yields_unknown_axes_rather_than_a_raw_rank():
    """A raw rank of 1 in a 0-100 field turns the best frame into the worst."""
    from photoai import assessment_parser as routing

    assessment = routing.parse_assessment(
        {
            "n": 1, "genre": "street", "axis_a": 1, "axis_b": 1, "axis_c": 1,
            "recover": "easy", },
        "best.RW2",
    )
    semantic = scoring.semantic_from_assessment(assessment)
    assert semantic.axis_a == scoring.UNKNOWN_AXIS

    converted = scoring.semantic_from_assessment(assessment, group_size=12)
    assert converted.axis_a == 100


def test_a_malformed_ranking_is_rejected_rather_than_aggregated():
    from photoai import batch_runner

    repeated = [{"n": 1, "axis_a": 1, "axis_b": 1, "axis_c": 1},
                {"n": 2, "axis_a": 1, "axis_b": 2, "axis_c": 2}]
    assert batch_runner.validate_group_ranks(repeated, 2)


def test_a_confirmed_blocker_still_outranks_everything(tmp_path):
    """The rescue must not become a way to keep a corrupt file."""
    found = IssueSet()
    found.add(IssueCode.CORRUPT_FILE, "does not decode")
    profile = pipeline.resolve().photo
    inp = ScoreInput(asset_id="a", filename="a.jpg", issues=found, technical_quality=0)
    result = scoring.classify(inp, scoring.score(inp, profile), profile)
    assert result.route_class is RouteClass.TRASH
