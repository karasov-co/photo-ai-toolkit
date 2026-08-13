"""Stage 3, and the promotion it is the only thing allowed to authorise.

Stage 3 was written and never wired in: the prompt sat in `prompts.py`, nothing
called it, every artistic field in every report was `null`, and `flagship` was
assigned from three ranked axes that say nothing about whether a photograph is
any good. A portrait taken mid-blink reached `flagship` because it was sharp and
hard to repeat.

Everything here holds one line: a missing score is not a low score, and a frame
cannot become the best thing in a shoot because the analysis that would have
judged it never finished.
"""

import json

import pytest
from synthetic import photo_like

import calibration
import prompts
import scoring
import stage3
from stage3 import ArtisticAssessment, Expression, EyesState, Stage3Status


def artistic_payload(**overrides) -> dict:
    payload = {
        **dict.fromkeys(stage3.ARTISTIC_FIELDS, 75),
        "artistic_candidate": True,
        "artistic_confidence": 80,
        "artistic_reasoning": "the figure at the edge makes the space feel like pressure",
        "artistic_strengths": ["directional light"],
        "artistic_weaknesses": ["busy edge"],
    }
    payload.update(overrides)
    return payload


def portrait_payload(**overrides) -> dict:
    payload = {
        "face_count": 1,
        "primary_face_visible": True,
        "primary_face_area_ratio": 0.22,
        "face_sharpness": 88,
        "eyes_state": "OPEN",
        "expression": "GOOD",
        "expression_quality": 82,
        "pose_quality": 75,
        "face_occlusion": 0,
        "blink_probability": 3,
        "grimace_probability": 2,
        "portrait_publishability": 85,
        "expression_confidence": 85,
        "portrait_reasoning": "settled expression, eyes engaged with the camera",
    }
    payload.update(overrides)
    return payload


# --- the status model --------------------------------------------------------


def test_a_fresh_assessment_is_pending_not_completed():
    assert ArtisticAssessment().status == Stage3Status.PENDING.value
    assert not ArtisticAssessment().completed


@pytest.mark.parametrize(
    ("factory", "status"),
    [
        (lambda: ArtisticAssessment.not_required("no candidate"), Stage3Status.NOT_REQUIRED),
        (lambda: ArtisticAssessment.skipped("no preview"), Stage3Status.SKIPPED),
        (lambda: ArtisticAssessment.failed(["timeout"]), Stage3Status.FAILED),
    ],
)
def test_every_non_completed_state_is_explicit(factory, status):
    assessment = factory()
    assert assessment.status == status.value
    assert not assessment.completed


def test_a_skipped_assessment_records_why():
    assert ArtisticAssessment.skipped("no preview was generated").skip_reason


def test_a_failed_assessment_records_the_errors_and_retries():
    assessment = ArtisticAssessment.failed(["bad json", "bad json again"], retries=2)
    assert assessment.parse_errors == ["bad json", "bad json again"]
    assert assessment.retries == 2
    assert assessment.analysed_at


def test_null_fields_are_never_a_completed_analysis():
    """The bug in one line."""
    assessment = ArtisticAssessment(status=Stage3Status.COMPLETED.value)
    assert assessment.completed
    assert not assessment.has_all_fields
    assert not assessment.usable


# --- parsing ------------------------------------------------------------------


def test_a_clean_reply_parses_into_every_field():
    assessment = stage3.parse_assessment(artistic_payload(), model="test-model")
    assert assessment.completed
    assert assessment.has_all_fields
    assert assessment.usable
    for name in stage3.ARTISTIC_FIELDS:
        assert assessment.score(name) == 75
    assert assessment.model == "test-model"
    assert assessment.prompt_version == stage3.PROMPT_VERSION


def test_out_of_range_scores_are_clamped_and_recorded():
    assessment = stage3.parse_assessment(artistic_payload(emotional_resonance=180))
    assert assessment.emotional_resonance == 100
    assert any("out of range" in e for e in assessment.parse_errors)


def test_a_missing_dimension_is_a_parse_failure_not_a_zero():
    payload = artistic_payload()
    del payload["visual_tension"]
    with pytest.raises(stage3.Stage3ParseError, match="visual_tension"):
        stage3.parse_assessment(payload)


def test_a_non_numeric_score_is_a_parse_failure():
    with pytest.raises(stage3.Stage3ParseError):
        stage3.parse_assessment(artistic_payload(distinctiveness="quite high"))


def test_a_non_object_reply_is_refused():
    with pytest.raises(stage3.Stage3ParseError):
        stage3.parse_assessment(["not", "an", "object"])


@pytest.mark.parametrize("value", [True, "true", "yes", 1])
def test_booleans_are_normalised(value):
    assert stage3.parse_assessment(artistic_payload(artistic_candidate=value)).artistic_candidate


def test_an_unknown_enum_falls_back_to_unclear():
    payload = artistic_payload(portrait=portrait_payload(eyes_state="sort of open"))
    portrait = stage3.parse_assessment(payload).portrait
    assert portrait.eyes_state == EyesState.UNCLEAR.value


def test_a_group_reply_maps_onto_filenames():
    text = json.dumps(
        [{**artistic_payload(), "n": 1}, {**artistic_payload(emotional_resonance=40), "n": 2}]
    )
    parsed = stage3.parse_group(text, ["a.jpg", "b.jpg"])
    assert set(parsed) == {"a.jpg", "b.jpg"}
    assert parsed["b.jpg"].emotional_resonance == 40


def test_an_unusable_object_is_dropped_rather_than_misattributed():
    """Attaching one frame's expression to another file is worse than none."""
    text = json.dumps([{**artistic_payload(), "n": 1}, {"n": 2, "emotional_resonance": 50}])
    parsed = stage3.parse_group(text, ["a.jpg", "b.jpg"])
    assert set(parsed) == {"a.jpg"}


def test_a_reply_with_no_array_raises():
    with pytest.raises(stage3.Stage3ParseError):
        stage3.parse_group("I could not analyse these", ["a.jpg"])


def test_an_assessment_round_trips_through_json():
    original = stage3.parse_assessment(artistic_payload(portrait=portrait_payload()))
    restored = ArtisticAssessment.from_dict(json.loads(stage3.to_json(original)))
    assert restored.usable
    assert restored.portrait.expression == Expression.GOOD.value


# --- when it runs -------------------------------------------------------------


@pytest.mark.parametrize("route", ["flagship", "stock_strong", "stock_standard"])
def test_keep_and_hero_candidates_are_always_read(route):
    needed, reason = stage3.should_run(
        route_class=route, has_unrecoverable=False, intentionality_likelihood=50,
        curatorial_uncertainty=20, faces_present=False,
    )
    assert needed and reason


def test_a_face_always_triggers_the_read():
    needed, reason = stage3.should_run(
        route_class="archive_only", has_unrecoverable=False, intentionality_likelihood=50,
        curatorial_uncertainty=10, faces_present=True,
    )
    assert needed
    assert "expression" in reason


def test_a_possibly_deliberate_defect_triggers_the_read():
    """The population a technical filter judges worst."""
    needed, _ = stage3.should_run(
        route_class="trash", has_unrecoverable=True, intentionality_likelihood=70,
        curatorial_uncertainty=30, faces_present=False,
    )
    assert needed


def test_a_confidently_broken_frame_is_not_read():
    needed, reason = stage3.should_run(
        route_class="trash", has_unrecoverable=True, intentionality_likelihood=10,
        curatorial_uncertainty=20, faces_present=False,
    )
    assert not needed
    assert "unrecoverable" in reason


def test_a_corrupt_file_is_never_read():
    needed, reason = stage3.should_run(
        route_class="trash", has_unrecoverable=True, intentionality_likelihood=90,
        curatorial_uncertainty=90, faces_present=True, corrupt=True,
    )
    assert not needed
    assert "decode" in reason


# --- the hero gate ------------------------------------------------------------


def test_a_completed_confident_read_raises_no_objection():
    assert stage3.hero_blockers(stage3.parse_assessment(artistic_payload())) == []


def test_a_pending_read_blocks_promotion():
    blocking = stage3.hero_blockers(ArtisticAssessment())
    assert blocking and "not completed" in blocking[0]


def test_a_failed_read_blocks_promotion():
    blocking = stage3.hero_blockers(ArtisticAssessment.failed(["timeout"]))
    assert blocking and "failed" in blocking[0]


def test_missing_fields_block_promotion_even_when_marked_completed():
    assessment = ArtisticAssessment(status=Stage3Status.COMPLETED.value, emotional_resonance=90)
    blocking = stage3.hero_blockers(assessment)
    assert blocking and "missing" in blocking[0]


def test_low_artistic_confidence_blocks_promotion():
    blocking = stage3.hero_blockers(stage3.parse_assessment(artistic_payload(artistic_confidence=20)))
    assert any("confidence" in b for b in blocking)


# --- portrait gates -----------------------------------------------------------


def parsed_portrait(**overrides):
    return stage3.parse_assessment(
        artistic_payload(portrait=portrait_payload(**overrides))
    )


def test_a_good_portrait_is_not_blocked():
    assert stage3.hero_blockers(parsed_portrait()) == []


def test_closed_eyes_block_promotion():
    blocking = stage3.hero_blockers(parsed_portrait(eyes_state="CLOSED", expression="BLINK"))
    assert any("eyes are closed" in b for b in blocking)


@pytest.mark.parametrize("expression", ["AWKWARD", "GRIMACE", "BLINK"])
def test_a_confident_bad_expression_blocks_promotion(expression):
    blocking = stage3.hero_blockers(
        parsed_portrait(expression=expression, expression_quality=20, expression_confidence=85)
    )
    assert any(expression.lower() in b for b in blocking)


def test_a_soft_face_blocks_promotion():
    blocking = stage3.hero_blockers(parsed_portrait(face_sharpness=30))
    assert any("soft" in b for b in blocking)


def test_low_expression_confidence_blocks_promotion():
    blocking = stage3.hero_blockers(parsed_portrait(expression_confidence=20))
    assert any("confidence" in b for b in blocking)


def test_a_high_blink_probability_blocks_promotion():
    assert any("blink" in b for b in stage3.hero_blockers(parsed_portrait(blink_probability=90)))


def test_an_incidental_face_does_not_gate_a_landscape():
    """A person in a landscape is not a portrait."""
    assessment = parsed_portrait(
        primary_face_area_ratio=0.005, eyes_state="CLOSED", expression="BLINK",
        expression_quality=10, expression_confidence=90,
    )
    assert stage3.hero_blockers(assessment) == []


def test_a_deliberate_expression_can_override_the_eye_gate():
    """Claimed in the structured field, never inferred from a high score."""
    assessment = stage3.parse_assessment(
        artistic_payload(
            intent_reading={"closed eyes": "deliberate"},
            portrait=portrait_payload(
                eyes_state="CLOSED", expression="NEUTRAL",
                expression_quality=80, expression_confidence=85,
            ),
        )
    )
    assert not any("eyes are closed" in b for b in stage3.hero_blockers(assessment))


def test_the_word_deliberate_in_the_prose_does_not_override_anything():
    """A real reply: "it does not clearly read as deliberate ... may be mid-speech".

    The keyword search that used to back this override matched that sentence and
    waved the frame through. Negation is invisible to a substring test, which is
    why the claim has to arrive in a field.
    """
    assessment = stage3.parse_assessment(
        artistic_payload(
            artistic_reasoning=(
                "That awkwardness adds tension, though it does not clearly read as "
                "deliberate and may simply be mid-speech."
            ),
            portrait=portrait_payload(
                eyes_state="CLOSED", expression="NEUTRAL",
                expression_quality=80, expression_confidence=85,
            ),
        )
    )
    assert not assessment.says_deliberate
    assert any("eyes are closed" in b for b in stage3.hero_blockers(assessment))


def test_an_accidental_intent_reading_is_not_a_deliberate_one():
    assessment = stage3.parse_assessment(
        artistic_payload(intent_reading={"blur": "accidental", "tilt": "cannot_tell"})
    )
    assert not assessment.says_deliberate


def test_a_high_aesthetic_score_alone_does_not_override_the_eye_gate():
    assessment = stage3.parse_assessment(
        artistic_payload(
            **dict.fromkeys(stage3.ARTISTIC_FIELDS, 100),
            artistic_reasoning="beautiful light and colour",
            portrait=portrait_payload(eyes_state="CLOSED", expression_confidence=85),
        )
    )
    assert any("eyes are closed" in b for b in stage3.hero_blockers(assessment))


# --- the portrait verdict -----------------------------------------------------


def test_a_confident_bad_expression_is_a_decision_not_a_review():
    verdict, reason = stage3.portrait_verdict(
        parsed_portrait(
            expression="GRIMACE", expression_quality=15,
            portrait_publishability=12, expression_confidence=88,
        )
    )
    assert verdict == "reject"
    assert "confident" in reason


def test_an_unflattering_expression_is_not_a_rejection():
    """Real numbers from the archive: AWKWARD, quality 47, publishability 51.

    Declining to promote a frame costs nothing, so a label is enough for that.
    Writing one off costs the photograph, so it takes the model's own numbers --
    and here they say "not especially flattering", not "unusable".
    """
    assessment = parsed_portrait(
        expression="AWKWARD", expression_quality=47,
        portrait_publishability=51, expression_confidence=76,
    )
    assert stage3.portrait_verdict(assessment)[0] == "keep"
    # Still not promotable, on the strength of the label alone.
    assert any("awkward" in b for b in stage3.hero_blockers(assessment))


def test_an_uncertain_expression_is_the_kind_worth_a_persons_time():
    verdict, reason = stage3.portrait_verdict(
        parsed_portrait(expression="UNCLEAR", expression_confidence=30)
    )
    assert verdict == "review"
    assert "confidence" in reason


def test_a_good_portrait_needs_no_decision():
    assert stage3.portrait_verdict(parsed_portrait())[0] == "keep"


def test_a_frame_with_no_face_needs_no_portrait_decision():
    assert stage3.portrait_verdict(stage3.parse_assessment(artistic_payload()))[0] == "keep"


# --- crops --------------------------------------------------------------------


def test_crops_include_the_full_frame_and_context():
    views = stage3.face_crops(photo_like(800, 600, seed=3))
    names = [name for name, _ in views]
    assert names[0] == "full frame"
    assert "face" in names
    assert "head and shoulders" in names


def test_crops_are_bounded_in_size():
    for _, view in stage3.face_crops(photo_like(2000, 1500, seed=3), max_px=400):
        assert max(view.size) <= 400


def test_the_context_crop_is_wider_than_the_face_crop():
    """A tight crop removes the pose, which is half of what makes an expression."""
    views = dict(stage3.face_crops(photo_like(1200, 900, seed=4)))
    assert views["head and shoulders"].size >= views["face"].size


def test_a_missing_detector_is_a_normal_state():
    assert isinstance(stage3.detector_available(), bool)
    if not stage3.detector_available():
        assert stage3.detect_primary_face(photo_like(400, 300)) is None


# --- the cache ----------------------------------------------------------------


def test_stage3_is_keyed_apart_from_stage2():
    """A valid Stage 2 entry must never answer for a missing Stage 3."""
    key = stage3.cache_key("abc", "a-model")
    assert key.startswith("stage3:")
    assert "abc" in key and "a-model" in key


def test_the_prompt_version_is_part_of_the_key():
    assert stage3.PROMPT_VERSION in stage3.cache_key("abc", "m")


def test_a_different_model_is_a_different_entry():
    assert stage3.cache_key("abc", "one") != stage3.cache_key("abc", "two")


def test_the_pipeline_cache_stores_stage3_separately(tmp_path):
    import pipeline

    cache = pipeline.AnalysisCache(tmp_path / "c.json")
    cache.put("abc", pipeline.Measurement(quality=50.0).to_dict())
    assert cache.get_stage3("abc", "a-model") is None

    cache.put_stage3("abc", "a-model", stage3.parse_assessment(artistic_payload()).to_dict())
    assert cache.get_stage3("abc", "a-model") is not None
    assert cache.get_stage3("abc", "another-model") is None


# --- the routing invariant ----------------------------------------------------


@pytest.fixture
def profile():
    return calibration.default_photo_profile()


def routed(profile, artistic):
    inp = scoring.ScoreInput(
        asset_id="a", filename="a.RW2", technical_quality=92, uplift=6, is_raw=True,
        artistic=artistic,
        semantic=scoring.Semantic(
            present=True, faces=False, brand_mark=False, identifiable_people=False, axis_b=98
        ),
    )
    return scoring.classify(inp, scoring.score(inp, profile), profile, flagship_selected=True)


def test_no_flagship_without_an_artistic_read(profile):
    result = routed(profile, None)
    assert result.route_class is not scoring.RouteClass.FLAGSHIP
    assert any("no artistic analysis" in r for r in result.reasons)


def test_no_flagship_when_stage3_failed(profile):
    result = routed(profile, ArtisticAssessment.failed(["timeout"]))
    assert result.route_class is not scoring.RouteClass.FLAGSHIP


def test_no_flagship_when_fields_are_null(profile):
    result = routed(profile, ArtisticAssessment(status=Stage3Status.COMPLETED.value))
    assert result.route_class is not scoring.RouteClass.FLAGSHIP
    assert any("missing" in r for r in result.reasons)


def test_flagship_is_reachable_with_a_completed_read(profile):
    assert routed(profile, stage3.parse_assessment(artistic_payload())).route_class is (
        scoring.RouteClass.FLAGSHIP
    )


def test_a_blinking_portrait_cannot_reach_flagship(profile):
    """The named regression: sharp, hard to repeat, eyes shut."""
    assessment = stage3.parse_assessment(
        artistic_payload(
            portrait=portrait_payload(
                eyes_state="CLOSED", expression="BLINK", expression_quality=12,
                blink_probability=94, portrait_publishability=8, expression_confidence=90,
            )
        )
    )
    result = routed(profile, assessment)
    assert result.route_class is not scoring.RouteClass.FLAGSHIP
    assert any("eyes are closed" in r for r in result.reasons)


def test_a_blocked_frame_lands_somewhere_sensible_rather_than_review(profile):
    """Blocking a promotion is not a reason to demand human attention."""
    result = routed(profile, ArtisticAssessment.failed(["timeout"]))
    assert result.route_class in (
        scoring.RouteClass.STOCK_STRONG,
        scoring.RouteClass.STOCK_STANDARD,
        scoring.RouteClass.REVIEW,
    )


# --- the prompt ---------------------------------------------------------------


def test_the_prompt_asks_for_every_field_the_schema_validates():
    for name in stage3.ARTISTIC_FIELDS:
        assert name in prompts.STAGE3_SYSTEM
    for name in stage3.PORTRAIT_SCORE_FIELDS:
        assert name in prompts.STAGE3_SYSTEM


def test_the_prompt_treats_the_face_as_a_separate_question():
    text = " ".join(prompts.STAGE3_SYSTEM.lower().split())
    assert "a blink is not an aesthetic property" in text
    assert "a beautiful photograph of a bad moment is a bad moment" in text


def test_the_prompt_requires_deliberate_to_be_said_explicitly():
    assert "deliberate" in prompts.STAGE3_SYSTEM


def test_multiple_views_are_labelled_as_one_photograph():
    frames = [{"key": "a.jpg", "views": [("full frame", "AA"), ("face", "BB")], "encoded": "AA"}]
    content = prompts.stage3_user_content(frames)
    text = " ".join(block.get("text", "") for block in content)
    assert "views of the SAME photograph" in text
    assert "(face)" in text


def test_a_single_view_frame_needs_no_labelling():
    frames = [{"key": "a.jpg", "views": [("full frame", "AA")], "encoded": "AA"}]
    text = " ".join(b.get("text", "") for b in prompts.stage3_user_content(frames))
    assert "views of the SAME photograph" not in text


# --- end to end ---------------------------------------------------------------
#
# Nothing below makes a network call. The client is a stub, so what these prove
# is that the wiring carries a Stage 3 verdict all the way from the reply to the
# route -- the judgement itself belongs to the model.


class _Reply:
    def __init__(self, text):
        self.output_text = text


class _Responses:
    def __init__(self, stage2, stage3_payload):
        self.stage2 = stage2
        self.stage3_payload = stage3_payload
        self.stage2_calls = 0
        self.stage3_calls = 0

    def create(self, **kwargs):
        if "emotional_resonance" in kwargs.get("instructions", ""):
            self.stage3_calls += 1
            return _Reply(self.stage3_payload)
        self.stage2_calls += 1
        return _Reply(self.stage2)


class _Client:
    def __init__(self, stage2, stage3_payload):
        self.responses = _Responses(stage2, stage3_payload)


def stage2_ranking(count: int) -> str:
    return json.dumps(
        [
            {
                "n": i + 1, "genre": "portrait", "axis_a": i + 1,
                "axis_b": count - i, "axis_c": i + 1, "recover": "easy",
                "faces": True, "brand_mark": False, "note": "lift shadows",
            }
            for i in range(count)
        ]
    )


def stage3_replies(count: int, **portrait_overrides) -> str:
    return json.dumps(
        [
            {
                **artistic_payload(
                    **dict.fromkeys(stage3.ARTISTIC_FIELDS, 92),
                    portrait=portrait_payload(**portrait_overrides),
                ),
                "n": i + 1,
            }
            for i in range(count)
        ]
    )


@pytest.fixture
def archive(tmp_path):
    from synthetic import write_jpeg

    root = tmp_path / "archive"
    for i in range(3):
        write_jpeg(photo_like(1200, 900, seed=i + 1), root / f"p{i}.jpg")
    return root


def run(archive, tmp_path, client, **options):
    import pipeline

    return pipeline.run(
        pipeline.PipelineOptions(
            input_dir=archive, output_dir=tmp_path / "out", semantic=True, **options
        ),
        client=client,
    )


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch, tmp_path):
    import bootstrap

    monkeypatch.delenv(bootstrap.API_KEY_VAR, raising=False)
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "PROJECT_ENV", tmp_path / ".env")
    monkeypatch.setattr(bootstrap, "_loaded_from", None)


def test_stage3_actually_runs(archive, tmp_path):
    client = _Client(stage2_ranking(3), stage3_replies(3))
    result = run(archive, tmp_path, client)
    assert client.responses.stage3_calls >= 1
    assert result.stage3_completed >= 1


def test_a_completed_read_reaches_the_record(archive, tmp_path):
    """The reported bug: every artistic field null in every report."""
    result = run(archive, tmp_path, _Client(stage2_ranking(3), stage3_replies(3)))
    completed = [r for r in result.records if r.stage3.get("status") == "completed"]
    assert completed
    assert all(r.stage3.get("emotional_resonance") is not None for r in completed)


def test_a_blinking_portrait_is_not_promoted_end_to_end(archive, tmp_path):
    client = _Client(
        stage2_ranking(3),
        stage3_replies(
            3, eyes_state="CLOSED", expression="BLINK", expression_quality=10,
            blink_probability=95, portrait_publishability=8, expression_confidence=90,
        ),
    )
    result = run(archive, tmp_path, client)
    assert not [r for r in result.records if r.route_class == "flagship"]
    # The gate itself is asserted directly in the routing tests above; what this
    # checks is that the verdict survives the trip from reply to record, which is
    # exactly the leg that was missing.
    assert {r.portrait_verdict for r in result.records} == {"reject"}


def test_the_same_frames_read_as_keepable_when_the_eyes_are_open(archive, tmp_path):
    """The other half: the gate has to be capable of not firing."""
    result = run(archive, tmp_path, _Client(stage2_ranking(3), stage3_replies(3)))
    assert {r.portrait_verdict for r in result.records} == {"keep"}


def test_nothing_is_promoted_when_stage3_is_off(archive, tmp_path):
    result = run(archive, tmp_path, _Client(stage2_ranking(3), ""), stage3=False)
    assert not [r for r in result.records if r.route_class == "flagship"]
    assert all(r.stage3.get("status") == "not_required" for r in result.records)


def test_an_unavailable_read_is_skipped_not_declared_unnecessary(archive, tmp_path):
    """`not_required` reads as a judgement about the frame. This wasn't one."""
    import pipeline

    result = pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out", semantic=False)
    )
    assert all(r.stage3.get("status") == "skipped" for r in result.records)
    assert all(r.stage3.get("skip_reason") for r in result.records)


def test_a_broken_reply_leaves_a_failure_not_a_silent_pass(archive, tmp_path):
    client = _Client(stage2_ranking(3), "the model declined to answer")
    result = run(archive, tmp_path, client)
    assert result.stage3_failed >= 1
    assert not [r for r in result.records if r.route_class == "flagship"]


def test_the_second_run_reuses_the_cached_read(archive, tmp_path):
    client = _Client(stage2_ranking(3), stage3_replies(3))
    run(archive, tmp_path, client)
    first = client.responses.stage3_calls
    run(archive, tmp_path, client)
    assert client.responses.stage3_calls == first


def test_a_new_prompt_version_invalidates_the_cache(archive, tmp_path, monkeypatch):
    client = _Client(stage2_ranking(3), stage3_replies(3))
    run(archive, tmp_path, client)
    first = client.responses.stage3_calls
    monkeypatch.setattr(stage3, "PROMPT_VERSION", stage3.PROMPT_VERSION + "-next")
    run(archive, tmp_path, client)
    assert client.responses.stage3_calls > first


# --- truncation ---------------------------------------------------------------
#
# Found by running against the archive: the per-frame token budget was under
# half what a frame actually costs, groups of six truncated mid-JSON, and the
# retry sent the identical request twice more.


class _Truncated:
    status = "incomplete"
    output_text = '[{"n": 1, "emotional_res'


class _TruncatingResponses:
    """Truncates any group larger than `fits`, answers cleanly at or below it."""

    def __init__(self, fits: int):
        self.fits = fits
        self.budgets: list[int] = []
        self.group_sizes: list[int] = []

    def create(self, **kwargs):
        content = kwargs["input"][0]["content"]
        size = sum(1 for block in content if block.get("type") == "input_image")
        self.budgets.append(kwargs["max_output_tokens"])
        self.group_sizes.append(size)
        if size > self.fits:
            return _Truncated()
        return _Reply(stage3_replies(size))


def _stage3_frames(count):
    return [{"key": f"f{i}.jpg", "views": [("full frame", "AA")], "encoded": "AA"} for i in range(count)]


def test_a_truncated_reply_splits_the_group_instead_of_repeating_it():
    import pipeline

    responses = _TruncatingResponses(fits=2)
    out = pipeline._stage3_call(
        _stage3_frames(4), model="m",
        client=type("C", (), {"responses": responses})(), stage3_module=stage3,
    )
    assert len(out) == 4
    assert all(a.completed for a in out.values())
    assert min(responses.group_sizes) <= 2


def test_a_single_frame_that_truncates_gets_a_wider_budget():
    import pipeline

    responses = _TruncatingResponses(fits=0)
    pipeline._stage3_call(
        _stage3_frames(1), model="m",
        client=type("C", (), {"responses": responses})(), stage3_module=stage3,
    )
    assert responses.budgets[-1] > responses.budgets[0]


def test_widening_the_budget_is_bounded():
    import pipeline

    responses = _TruncatingResponses(fits=0)
    out = pipeline._stage3_call(
        _stage3_frames(1), model="m",
        client=type("C", (), {"responses": responses})(), stage3_module=stage3,
    )
    assert all(a.status == "failed" for a in out.values())
    assert max(responses.budgets) <= (
        pipeline.STAGE3_BUDGET_LIMIT
        * (prompts.STAGE3_BASE_OUTPUT_TOKENS + prompts.STAGE3_MAX_OUTPUT_TOKENS_PER_FRAME)
    )


def test_the_budget_covers_what_a_frame_actually_costs():
    """Measured live at ~315 output tokens per frame with no portrait block."""
    assert prompts.STAGE3_MAX_OUTPUT_TOKENS_PER_FRAME >= 400
    assert prompts.STAGE3_BASE_OUTPUT_TOKENS >= 600


def test_truncation_is_told_apart_from_prose():
    import pipeline

    assert pipeline._truncated(_Truncated())
    assert not pipeline._truncated(_Reply("I could not analyse these"))


# --- a near-ranking is repaired, not discarded --------------------------------
#
# From the archive: one Stage 2 group came back [1,2,3,4,4,5,6,7,8,9,10,12] on
# one axis. The strict check rejected the group, and eight photographs lost
# their genre, their faces and their portrait gate to a single miscounted
# integer -- while the summary still read "local + semantic".


def _ranked(values, axis="axis_c"):
    return [
        {"n": i + 1, "axis_a": i + 1, "axis_b": i + 1, "axis_c": i + 1, **{axis: v}}
        for i, v in enumerate(values)
    ]


def test_the_exact_reply_that_cost_eight_photographs_their_genre():
    import batch_runner

    items = _ranked([1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 12])
    repaired, notes = batch_runner.repair_group_ranks(items, 12)
    assert notes
    assert batch_runner.validate_group_ranks(repaired, 12) == []


def test_a_repair_keeps_the_order_the_model_expressed():
    import batch_runner

    items = _ranked([1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 12])
    repaired, _ = batch_runner.repair_group_ranks(items, 12)
    ranks = [item["axis_c"] for item in repaired]
    assert ranks == sorted(ranks), "a frame the model put first must stay first"
    assert ranks[0] == 1 and ranks[-1] == 12


def test_a_clean_ranking_is_left_alone():
    import batch_runner

    items = _ranked(list(range(1, 13)))
    repaired, notes = batch_runner.repair_group_ranks(items, 12)
    assert notes == []
    assert [i["axis_c"] for i in repaired] == list(range(1, 13))


def test_a_reply_that_ranked_nothing_is_not_repaired():
    """Half the group tied is not a miscount, and no repair invents an order."""
    import batch_runner

    items = _ranked([1] * 12)
    _, notes = batch_runner.repair_group_ranks(items, 12)
    assert notes == []
    assert batch_runner.validate_group_ranks(items, 12)


def test_a_short_reply_is_not_repaired():
    import batch_runner

    items = _ranked([1, 2, 3])
    _, notes = batch_runner.repair_group_ranks(items, 12)
    assert notes == []
