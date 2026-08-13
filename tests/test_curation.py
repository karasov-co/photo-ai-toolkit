"""The final score, and the five piles.

Seven cases are named in the specification and each has a test below marked with
its number. The rest of the file defends the two asymmetries the score exists to
express, because those are the ones that silently stop holding:

  a technical number must never rescue a photograph that failed, and
  a missing release must never make a photograph bad.

The second one is not hypothetical. Before this, a good picture of your family
was filed alongside genuinely broken frames, because the only thing separating
piles was a stock threshold and stock is exactly what that photograph is not.
"""

import pytest

import calibration
import curation
import scoring
import stage3
from curation import PhotoCategory
from issues import IssueCode, IssueSet


@pytest.fixture
def profile():
    return calibration.default_photo_profile()


def artistic(**overrides) -> stage3.ArtisticAssessment:
    payload = {
        **dict.fromkeys(stage3.ARTISTIC_FIELDS, 75),
        "artistic_candidate": True,
        "artistic_confidence": 80,
        "artistic_reasoning": "a figure pushed to the edge, the space reading as pressure",
        "uncertainty": 20,
        "series_role": "none",
    }
    payload.update(overrides)
    return stage3.parse_assessment(payload)


def flat(value: int, **overrides) -> stage3.ArtisticAssessment:
    """An artistic read that scores everything the same. Useful for isolating."""
    return artistic(**{**dict.fromkeys(stage3.ARTISTIC_FIELDS, value), **overrides})


def face(**overrides) -> dict:
    payload = {
        "face_count": 1,
        "primary_face_visible": True,
        "primary_face_area_ratio": 0.25,
        "face_sharpness": 88,
        "eyes_state": "OPEN",
        "expression": "GOOD",
        "expression_quality": 82,
        "pose_quality": 78,
        "face_occlusion": 0,
        "blink_probability": 3,
        "grimace_probability": 2,
        "portrait_publishability": 85,
        "expression_confidence": 85,
        "portrait_reasoning": "a settled expression, eyes engaged",
    }
    payload.update(overrides)
    return payload


def semantic(**overrides) -> scoring.Semantic:
    payload = {
        "present": True,
        "genre": "landscape",
        "axis_a": 70,
        "axis_b": 70,
        "axis_c": 60,
        "faces": False,
        "logos": False,
        "identifiable_people": False,
        "subject_strength": 75,
    }
    payload.update(overrides)
    return scoring.Semantic(**payload)


def judge(profile, *, tech=82, uplift=6, sem=None, art=None, issues=None,
          best=True, margin=0.0, cluster=1) -> curation.Verdict:
    inp = scoring.ScoreInput(
        asset_id="a", filename="frame.jpg", technical_quality=tech, uplift=uplift,
        is_raw=True, semantic=sem if sem is not None else scoring.Semantic(),
        artistic=art, issues=issues or IssueSet(), is_best_in_cluster=best,
        cluster_margin=margin, cluster_size=cluster,
        semantic_ran=bool(sem is not None and sem.present),
    )
    return curation.categorise(inp, scoring.score(inp, profile), profile)


# --- the seven named cases ----------------------------------------------------


def test_1_a_blinking_portrait_is_weak(profile):
    verdict = judge(
        profile,
        tech=94,  # technically excellent, which is the point
        sem=semantic(genre="portrait", faces=True, identifiable_people=True, axis_a=88, axis_b=85),
        art=artistic(
            portrait=face(
                eyes_state="CLOSED", expression="BLINK", expression_quality=10,
                blink_probability=95, portrait_publishability=8, expression_confidence=90,
            )
        ),
    )
    assert verdict.category == PhotoCategory.WEAK.value
    assert "eyes closed" in verdict.reasons[0]


def test_2_an_accidental_frame_with_no_subject_is_weak(profile):
    verdict = judge(
        profile,
        tech=88,
        sem=semantic(
            genre="other", axis_a=20, axis_b=15, axis_c=10,
            intended_frame=False, subject_strength=8, accidental_probability=90,
        ),
        art=flat(15, artistic_candidate=False),
    )
    assert verdict.category == PhotoCategory.WEAK.value
    assert "accidental" in verdict.reasons[0]


def test_3_a_strong_landscape_reaches_top(profile):
    verdict = judge(
        profile,
        tech=88, uplift=9,
        sem=semantic(axis_a=95, axis_b=92, axis_c=85, subject_strength=90),
        art=flat(90, artistic_confidence=85),
    )
    assert verdict.category == PhotoCategory.TOP.value
    assert verdict.final_score >= curation.DEFAULT_THRESHOLDS.top


def test_4_losing_to_a_sibling_does_not_decide_a_category(profile):
    """Superseded by a later requirement, and the reversal was right.

    This used to assert that the weaker of two near-identical takes became
    WEAK. That made a photograph's category depend on what else was in the
    folder: import a sharper frame of the same scene and an unchanged picture
    turned from good to weak; delete it again and it turned back. Being the
    second-best of a pair is a fact about the pair.

    It is still reported -- as `duplicate_candidate` on the route class, which
    is a comparison for a person -- and it no longer touches the score.
    """
    alone = judge(profile, sem=semantic(axis_a=60, axis_b=55), art=artistic())
    beaten = judge(
        profile, best=False, margin=14, cluster=3,
        sem=semantic(axis_a=60, axis_b=55), art=artistic(),
    )
    assert beaten.final_score == alone.final_score
    assert beaten.category == alone.category
    assert not any(d["code"] == "inferior_duplicate" for d in beaten.score.defects)


def test_5_a_good_portrait_without_a_release_is_personal_not_weak(profile):
    verdict = judge(
        profile,
        tech=85, uplift=7,
        sem=semantic(
            genre="portrait", faces=True, identifiable_people=True,
            axis_a=70, axis_b=80, subject_strength=85,
        ),
        art=flat(78, artistic_confidence=80, portrait=face()),
    )
    assert verdict.category == PhotoCategory.GOOD_PERSONAL.value
    assert any("release" in b for b in verdict.commercial_blockers)


def test_6_a_commercially_usable_good_image_is_good_stock(profile):
    verdict = judge(
        profile,
        tech=84, uplift=7,
        sem=semantic(genre="detail", axis_a=88, axis_b=70, subject_strength=80),
        art=flat(72, artistic_confidence=75),
    )
    assert verdict.category == PhotoCategory.GOOD_STOCK.value
    assert not verdict.commercial_blockers


def test_7_stage3_changes_both_the_score_and_the_category(profile):
    """The requirement in one test: same pixels, same content pass, two reads."""
    kwargs = {"tech": 84, "uplift": 8,
              "sem": semantic(axis_a=85, axis_b=85, axis_c=70, subject_strength=85)}

    strong = judge(profile, art=flat(95, artistic_confidence=90), **kwargs)
    weak = judge(profile, art=flat(25, artistic_candidate=False, artistic_confidence=80), **kwargs)

    assert strong.final_score > weak.final_score + 20
    assert strong.category != weak.category
    assert strong.category == PhotoCategory.TOP.value
    assert strong.score.stage3_delta > 0
    assert weak.score.stage3_delta < 0


# --- Stage 3 is a term in the arithmetic, not a veto --------------------------


def test_the_delta_is_measured_against_the_same_frame(profile):
    verdict = judge(profile, sem=semantic(), art=flat(95, artistic_confidence=90))
    assert verdict.score.stage3_delta == verdict.final_score - verdict.score.without_stage3


def test_without_stage3_the_delta_is_exactly_zero(profile):
    """Not "small". Zero -- there is nothing for it to have contributed."""
    assert judge(profile, sem=semantic()).score.stage3_delta == 0


def test_a_failed_read_contributes_nothing_rather_than_a_low_score(profile):
    failed = judge(profile, sem=semantic(), art=stage3.ArtisticAssessment.failed(["timeout"]))
    none_at_all = judge(profile, sem=semantic())
    assert failed.final_score == none_at_all.final_score


def test_nothing_without_an_artistic_read_can_reach_top(profile):
    verdict = judge(profile, tech=99, uplift=12, sem=semantic(axis_a=100, axis_b=100, axis_c=100))
    assert verdict.category != PhotoCategory.TOP.value
    assert verdict.final_score <= curation.DEFAULT_THRESHOLDS.no_stage3_ceiling


def test_a_high_score_with_an_unconfident_read_is_not_top(profile):
    verdict = judge(
        profile, tech=90, uplift=9,
        sem=semantic(axis_a=95, axis_b=95, axis_c=90, subject_strength=95),
        art=flat(92, artistic_confidence=30),
    )
    assert verdict.category != PhotoCategory.TOP.value


def test_the_artistic_component_moves_with_the_eight_dimensions():
    assert curation.artistic_component(flat(90)) > curation.artistic_component(flat(40))


def test_prettiness_is_the_smallest_term():
    """`conventional_beauty` is what a model reaches for when it has nothing."""
    beauty = curation.ARTISTIC_WEIGHTS["conventional_beauty"]
    assert beauty == min(curation.ARTISTIC_WEIGHTS.values())
    assert beauty < curation.ARTISTIC_WEIGHTS["emotional_resonance"] / 5


# --- technical excellence cannot rescue ---------------------------------------


@pytest.mark.parametrize(
    ("label", "sem_kwargs", "art"),
    [
        ("accidental", {"intended_frame": False, "subject_strength": 10}, None),
        ("no subject", {"subject_strength": 5}, None),
        ("dead moment", {"dead_moment_probability": 88}, None),
    ],
)
def test_a_perfect_technical_score_does_not_lift_a_failed_frame(profile, label, sem_kwargs, art):
    verdict = judge(
        profile, tech=100, uplift=15,
        sem=semantic(axis_a=95, axis_b=95, axis_c=95, **sem_kwargs),
        art=art or flat(30, artistic_candidate=False),
    )
    assert verdict.category == PhotoCategory.WEAK.value, label


def test_a_ceiling_is_recorded_so_a_capped_score_is_not_mistaken_for_a_low_one(profile):
    verdict = judge(
        profile, tech=95,
        sem=semantic(intended_frame=False, subject_strength=10, accidental_probability=90),
        art=flat(20, artistic_candidate=False),
    )
    assert verdict.score.applied_ceiling is not None
    assert verdict.score.blended > verdict.final_score


def test_an_unrecoverable_fault_caps_the_score(profile):
    found = IssueSet()
    found.add(IssueCode.MISSED_FOCUS, "the subject is not in focus")
    verdict = judge(profile, tech=90, sem=semantic(), art=flat(85), issues=found)
    assert verdict.category == PhotoCategory.WEAK.value


# --- evidence can rescue ------------------------------------------------------


def test_strong_documentary_evidence_keeps_an_unconventional_frame(profile):
    """Technically poor, commercially useless, and worth keeping anyway."""
    verdict = judge(
        profile, tech=34, uplift=2,
        sem=semantic(axis_a=15, axis_b=40, axis_c=70, subject_strength=45),
        art=artistic(documentary_significance=92, artistic_confidence=80),
    )
    assert verdict.category != PhotoCategory.WEAK.value
    assert verdict.score.applied_floor == curation.DOCUMENTARY_FLOOR


def test_a_confident_read_overrules_a_guess_about_a_dead_moment(profile):
    verdict = judge(
        profile, tech=70,
        sem=semantic(dead_moment_probability=80, subject_strength=60),
        art=artistic(distinctiveness=90, artistic_candidate=True, artistic_confidence=85),
    )
    assert verdict.category != PhotoCategory.WEAK.value
    assert any(d["vetoed_by"] for d in verdict.score.defects)


def test_a_confident_read_does_not_overrule_a_closed_eye(profile):
    """The line between a guess and an observation."""
    verdict = judge(
        profile, tech=88,
        sem=semantic(genre="portrait", faces=True),
        art=artistic(
            documentary_significance=95, distinctiveness=95, artistic_confidence=90,
            portrait=face(
                eyes_state="CLOSED", expression="BLINK", expression_quality=8,
                expression_confidence=88, portrait_publishability=5,
            ),
        ),
    )
    assert verdict.category == PhotoCategory.WEAK.value


def test_a_deliberate_closed_eye_is_not_a_defect(profile):
    verdict = judge(
        profile, tech=84, uplift=6,
        sem=semantic(genre="portrait", faces=True, subject_strength=80),
        art=flat(
            82, artistic_confidence=85,
            intent_reading={"closed eyes": "deliberate"},
            portrait=face(
                eyes_state="CLOSED", expression="NEUTRAL",
                expression_quality=78, expression_confidence=85,
            ),
        ),
    )
    assert verdict.category != PhotoCategory.WEAK.value


def test_an_uplift_needs_a_confident_read(profile):
    assert curation.artistic_uplifts(artistic(documentary_significance=95, artistic_confidence=30)) == []


def test_an_uplift_needs_a_completed_read():
    assert curation.artistic_uplifts(stage3.ArtisticAssessment.failed(["timeout"])) == []
    assert curation.artistic_uplifts(None) == []


# --- saleability is not quality -----------------------------------------------


def test_a_release_never_lowers_the_score(profile):
    """The rule stated as an equality, because an inequality would drift."""
    art = flat(80, artistic_confidence=80, portrait=face())
    common = {"tech": 85, "uplift": 7, "art": art}
    private = judge(profile, sem=semantic(genre="portrait", faces=True,
                                          identifiable_people=True, subject_strength=80), **common)
    sellable = judge(profile, sem=semantic(genre="portrait", faces=False,
                                           identifiable_people=False, subject_strength=80), **common)
    assert private.final_score == sellable.final_score
    assert private.category == PhotoCategory.GOOD_PERSONAL.value
    assert sellable.category in (PhotoCategory.GOOD_STOCK.value, PhotoCategory.TOP.value)


def test_a_crowd_of_strangers_never_lowers_the_score(profile):
    art = flat(78, artistic_confidence=80)
    alone = judge(profile, sem=semantic(people_count=0), art=art)
    crowded = judge(profile, sem=semantic(people_count=40, faces=True,
                                          identifiable_people=True), art=art)
    assert crowded.final_score == alone.final_score


def test_the_score_ignores_the_two_commercial_dimensions(profile):
    """Stated structurally: neither name may appear among the components."""
    verdict = judge(profile, sem=semantic(), art=artistic())
    assert "stock_potential" not in verdict.score.components
    assert "legal_readiness" not in verdict.score.components


def test_a_personal_photograph_can_still_be_top(profile):
    """TOP is about the photograph. Nothing about stock enters it."""
    verdict = judge(
        profile, tech=90, uplift=9,
        sem=semantic(genre="portrait", faces=True, identifiable_people=True,
                     axis_a=90, axis_b=95, axis_c=85, subject_strength=92),
        art=flat(93, artistic_confidence=88, portrait=face()),
    )
    assert verdict.category == PhotoCategory.TOP.value
    assert verdict.commercial_blockers


def test_a_release_is_never_a_reason_a_photograph_is_bad(profile):
    """It may explain GOOD_PERSONAL. It may never explain WEAK."""
    weak = judge(
        profile, tech=40,
        sem=semantic(faces=True, identifiable_people=True, intended_frame=False,
                     subject_strength=8, accidental_probability=90),
        art=flat(15, artistic_candidate=False),
    )
    assert weak.category == PhotoCategory.WEAK.value
    assert weak.commercial_blockers
    assert not any("release" in r for r in weak.reasons)


def test_limited_demand_is_a_commercial_fact_not_a_quality_one(profile):
    verdict = judge(profile, tech=60, sem=semantic(axis_a=20, subject_strength=60), art=artistic())
    assert any("demand" in b for b in verdict.commercial_blockers)
    assert verdict.category != PhotoCategory.WEAK.value


# --- needs_decision stays rare ------------------------------------------------


def test_an_uncertain_read_on_the_keep_boundary_asks_a_person(profile):
    verdict = judge(
        profile, tech=50, uplift=2,
        sem=semantic(axis_a=50, axis_b=50, axis_c=45, subject_strength=50),
        art=flat(46, artistic_confidence=70, uncertainty=85),
    )
    assert verdict.category == PhotoCategory.NEEDS_DECISION.value
    assert abs(verdict.final_score - curation.DEFAULT_THRESHOLDS.weak) <= (
        curation.DEFAULT_THRESHOLDS.decision_band
    )


def test_an_uncertain_read_far_from_the_boundary_does_not(profile):
    verdict = judge(
        profile, tech=86, uplift=8,
        sem=semantic(axis_a=85, axis_b=85, subject_strength=85),
        art=flat(75, artistic_confidence=70, uncertainty=95),
    )
    assert verdict.category != PhotoCategory.NEEDS_DECISION.value


def test_top_or_merely_good_is_not_a_question_for_a_person(profile):
    """Both answers are "keep", so the frame is filed, not queued."""
    verdict = judge(
        profile, tech=88, uplift=8,
        sem=semantic(axis_a=88, axis_b=88, axis_c=80, subject_strength=88),
        art=flat(82, artistic_confidence=70, uncertainty=95),
    )
    assert abs(verdict.final_score - curation.DEFAULT_THRESHOLDS.top) <= 6
    assert verdict.category != PhotoCategory.NEEDS_DECISION.value


def test_a_confident_read_on_the_boundary_does_not(profile):
    verdict = judge(
        profile, tech=50, uplift=2,
        sem=semantic(axis_a=50, axis_b=50, axis_c=45, subject_strength=50),
        art=flat(46, artistic_confidence=75, uncertainty=10),
    )
    assert verdict.category != PhotoCategory.NEEDS_DECISION.value


def test_an_unclear_expression_on_the_boundary_asks_a_person(profile):
    verdict = judge(
        profile, tech=50, uplift=2,
        sem=semantic(genre="portrait", faces=True, axis_a=50, axis_b=50, subject_strength=50),
        art=flat(
            46, artistic_confidence=70, uncertainty=30,
            portrait=face(expression="UNCLEAR", expression_confidence=25, expression_quality=46,
                          portrait_publishability=52, pose_quality=46, face_sharpness=46),
        ),
    )
    assert verdict.category == PhotoCategory.NEEDS_DECISION.value


# --- the categories themselves ------------------------------------------------


def test_every_category_is_counted_even_at_zero():
    tally = curation.counts([PhotoCategory.TOP.value, PhotoCategory.TOP.value])
    assert set(tally) == {c.value for c in curation.CATEGORY_ORDER}
    assert tally["TOP"] == 2
    assert tally["WEAK"] == 0


def test_the_counts_ignore_a_value_that_is_not_a_category():
    assert curation.counts(["TOP", "nonsense"])["TOP"] == 1


def test_weak_is_a_shelf_and_not_a_bin(profile):
    """No category is grounds for deletion; only recorded evidence is.

    Worth asserting rather than assuming, because the cheapest way to make a
    culling tool look decisive is to wire a low score to a delete list.
    """
    found = IssueSet()
    found.add(IssueCode.MISSED_FOCUS, "the subject is not in focus")
    verdict = judge(profile, tech=40, sem=semantic(), art=artistic(), issues=found)
    assert verdict.category == PhotoCategory.WEAK.value

    inp = scoring.ScoreInput(
        asset_id="a", filename="frame.jpg", technical_quality=82, uplift=6,
        is_raw=True, semantic=semantic(), artistic=artistic(), is_best_in_cluster=False,
        cluster_margin=20, cluster_size=4,
    )
    scored = scoring.classify(inp, scoring.score(inp, profile), profile)
    assert scored.route_class is not scoring.RouteClass.TRASH


def test_the_score_survives_a_round_trip(profile):
    verdict = judge(profile, sem=semantic(), art=artistic())
    payload = verdict.to_dict()
    assert payload["category"] == verdict.category
    assert payload["score"]["stage3_delta"] == verdict.score.stage3_delta


# --- end to end ---------------------------------------------------------------
#
# Mocked clients throughout. What these prove is that a category reaches the
# record, the report, the console and the symlink farm -- not that the model's
# judgement is any good, which no test can establish.


class _Reply:
    def __init__(self, text):
        self.output_text = text


class _Responses:
    def __init__(self, stage2, stage3_text):
        self.stage2, self.stage3_text = stage2, stage3_text
        self.stage2_calls = self.stage3_calls = 0

    def create(self, **kwargs):
        if "emotional_resonance" in kwargs.get("instructions", ""):
            self.stage3_calls += 1
            return _Reply(self.stage3_text)
        self.stage2_calls += 1
        return _Reply(self.stage2)


class _Client:
    def __init__(self, stage2, stage3_text):
        self.responses = _Responses(stage2, stage3_text)


def stage2_reply(count: int, **overrides) -> str:
    import json

    return json.dumps(
        [
            {
                "n": i + 1, "genre": "landscape", "axis_a": i + 1,
                "axis_b": count - i, "axis_c": i + 1, "recover": "easy",
                "faces": False, "logos": False, "note": "lift shadows",
                "intended_frame": True, "subject_strength": 75,
                "accidental_probability": 2, "dead_moment_probability": 3,
                **overrides,
            }
            for i in range(count)
        ]
    )


def stage3_reply(count: int, value: int = 85, **overrides) -> str:
    import json

    return json.dumps(
        [
            {
                **dict.fromkeys(stage3.ARTISTIC_FIELDS, value),
                "artistic_candidate": True, "artistic_confidence": 85,
                "artistic_reasoning": "the light falls across one edge and nothing else",
                "uncertainty": 15, "n": i + 1, **overrides,
            }
            for i in range(count)
        ]
    )


@pytest.fixture
def archive(tmp_path):
    from synthetic import photo_like, write_jpeg

    root = tmp_path / "archive"
    for i in range(3):
        write_jpeg(photo_like(1400, 1000, seed=i + 1), root / f"p{i}.jpg")
    return root


def run(archive, tmp_path, client=None, **options):
    import pipeline

    return pipeline.run(
        pipeline.PipelineOptions(
            input_dir=archive, output_dir=tmp_path / "out",
            semantic=client is not None, **options,
        ),
        client=client,
    )


def run_cli(archive, tmp_path, client, monkeypatch):
    """The whole command, including everything it writes to disk.

    `pipeline.run` returns records; the reports, the HTML and the symlink farm
    are written by the CLI, so anything asserting about those has to go through
    it. The client is injected rather than built from a credential.
    """
    import cli
    import pipeline

    real = pipeline.run
    monkeypatch.setattr(pipeline, "run", lambda o, **kw: real(o, **{**kw, "client": client}))
    monkeypatch.setattr("bootstrap.has_credentials", lambda: True)
    monkeypatch.setattr("bootstrap.credential_status", lambda lang="en": "credentials: stubbed")

    import preflight

    monkeypatch.setattr(
        preflight, "run",
        lambda model, client=None: preflight.PreflightResult(ok=True, model=model),
    )
    assert cli.main(["analyze", "--input", str(archive), "--output", str(tmp_path / "out")]) == 0


def test_every_record_carries_a_category(archive, tmp_path):
    result = run(archive, tmp_path, _Client(stage2_reply(3), stage3_reply(3)))
    assert all(r.category for r in result.records)
    assert all(r.final_score > 0 for r in result.records)


def test_the_category_survives_into_the_json_and_csv(archive, tmp_path, monkeypatch):
    import json

    run_cli(archive, tmp_path, _Client(stage2_reply(3), stage3_reply(3)), monkeypatch)
    reports_dir = tmp_path / "out" / ".internal" / "reports"
    data = json.loads((reports_dir / "analysis.json").read_text())
    assets = data["assets"] if isinstance(data, dict) else data
    assert all(a["category"] for a in assets)
    assert "category" in (reports_dir / "analysis.csv").read_text().splitlines()[0]


def test_a_blinking_portrait_is_weak_end_to_end(archive, tmp_path):
    client = _Client(
        stage2_reply(3, genre="portrait", faces=True),
        stage3_reply(
            3, 80,
            portrait=face(
                eyes_state="CLOSED", expression="BLINK", expression_quality=8,
                blink_probability=96, portrait_publishability=5, expression_confidence=92,
            ),
        ),
    )
    result = run(archive, tmp_path, client)
    assert {r.category for r in result.records} == {PhotoCategory.WEAK.value}
    assert all("eyes closed" in r.category_reasons[0] for r in result.records)


def test_an_accidental_frame_is_weak_end_to_end(archive, tmp_path):
    client = _Client(
        stage2_reply(3, intended_frame=False, subject_strength=6, accidental_probability=92),
        stage3_reply(3, 20, artistic_candidate=False),
    )
    result = run(archive, tmp_path, client)
    assert {r.category for r in result.records} == {PhotoCategory.WEAK.value}


def test_stage3_moves_the_score_in_a_real_run(archive, tmp_path):
    strong = run(archive, tmp_path / "s", _Client(stage2_reply(3), stage3_reply(3, 95)))
    weak = run(archive, tmp_path / "w", _Client(stage2_reply(3), stage3_reply(3, 20)))
    assert min(r.final_score for r in strong.records) > max(r.final_score for r in weak.records)
    assert all(r.final_score_detail["stage3_delta"] > 0 for r in strong.records)
    assert all(r.final_score_detail["stage3_delta"] < 0 for r in weak.records)


def test_a_local_only_run_still_categorises_but_reaches_no_top(archive, tmp_path):
    result = run(archive, tmp_path)
    assert all(r.category for r in result.records)
    assert not [r for r in result.records if r.category == PhotoCategory.TOP.value]
    assert all(r.final_score_detail["stage3_delta"] == 0 for r in result.records)


def test_the_summary_counts_every_category(archive, tmp_path):
    import reports

    result = run(archive, tmp_path, _Client(stage2_reply(3), stage3_reply(3)))
    summary = reports.summarise(result.records)
    assert set(summary["by_category"]) == {c.value for c in curation.CATEGORY_ORDER}
    assert sum(summary["by_category"].values()) == len(result.records)


def test_the_console_summary_names_the_categories(archive, tmp_path):
    import reports

    result = run(archive, tmp_path, _Client(stage2_reply(3), stage3_reply(3)))
    text = reports.format_summary(reports.summarise(result.records))
    for name in ("Top", "Good — stock", "Good — personal", "Needs decision", "Weak"):
        assert name in text, name


def test_the_html_has_a_top_photos_section(archive, tmp_path, monkeypatch):
    run_cli(archive, tmp_path, _Client(stage2_reply(3), stage3_reply(3, 95)), monkeypatch)
    assert "Top photos" in (
        tmp_path / "out" / ".internal" / "reports" / "full_report.html"
    ).read_text()


def test_the_farm_has_a_category_view(archive, tmp_path, monkeypatch):
    run_cli(archive, tmp_path, _Client(stage2_reply(3), stage3_reply(3, 95)), monkeypatch)
    out = tmp_path / "out"
    import workspace

    # Every folder exists whether or not it has anything in it, so that an empty
    # pile is visibly empty rather than absent.
    for folder in workspace.CATEGORY_DIRS.values():
        assert (out / folder).is_dir(), folder

    linked = sum(
        1
        for folder in workspace.CATEGORY_DIRS.values()
        for entry in (out / folder).iterdir()
        if entry.is_symlink()
    )
    assert linked == 3


def test_analyze_is_always_a_full_analysis():
    """There is no longer a flag that turns `analyze` into a local-only run.

    A downgrade that keeps the command name is how a run with no content check
    and no artistic read came to be presented as a finished analysis.
    """
    import cli

    parser = cli.build_parser()
    analyze = parser.parse_args(["analyze", "--input", "i", "--output", "o"])
    assert not hasattr(analyze, "no_semantic")
    assert not hasattr(analyze, "allow_semantic_fallback")
    assert not hasattr(analyze, "semantic")


def test_local_only_lives_under_its_own_command():
    import cli

    args = cli.build_parser().parse_args(["measure", "--input", "i", "--output", "o"])
    assert args.func is cli.cmd_measure


def test_the_suite_cannot_reach_the_real_env_file():
    """A guard on the guard. This is how a real key got spent once."""
    import bootstrap

    assert not bootstrap.PROJECT_ENV.exists()
    assert bootstrap.api_key() is None


# --- what the archive caught --------------------------------------------------
#
# Both of these were found by running the finished pipeline over 47 real
# photographs and looking at what came back. Neither was visible by inspection:
# the artistic read was correct in both cases and the gating threw it away.


def test_a_small_face_still_decides_a_portrait(profile):
    """P1019417: eyelids lowered, mouth mid-word, face 2.3% of the frame.

    The area rule exists so that a passer-by in a landscape does not gate the
    photograph. An environmental portrait puts the subject small in a wide
    scene, and the rule read that as incidental -- so the strongest evidence in
    the run, a model saying it would not publish this picture of this person,
    was discarded on a geometric technicality.
    """
    verdict = judge(
        profile, tech=85,
        sem=semantic(genre="portrait", faces=True, identifiable_people=True, subject_strength=60),
        art=flat(
            55, artistic_confidence=58,
            portrait=face(
                primary_face_area_ratio=0.023, eyes_state="PARTIALLY_CLOSED",
                expression="MID_SPEECH", expression_quality=35,
                portrait_publishability=43, expression_confidence=73,
            ),
        ),
    )
    assert verdict.category == PhotoCategory.WEAK.value


def test_the_same_small_face_in_a_landscape_still_gates_nothing(profile):
    """The rule it replaced has to keep working, or street photography dies."""
    verdict = judge(
        profile, tech=85, uplift=6,
        sem=semantic(genre="landscape", faces=True, subject_strength=80),
        art=flat(
            75, artistic_confidence=75,
            portrait=face(
                primary_face_area_ratio=0.023, eyes_state="PARTIALLY_CLOSED",
                expression="MID_SPEECH", expression_quality=35,
                portrait_publishability=43, expression_confidence=73,
            ),
        ),
    )
    assert verdict.category != PhotoCategory.WEAK.value


def test_mid_speech_is_a_failed_moment(profile):
    """The most common unusable outtake in any portrait shoot."""
    verdict = judge(
        profile, tech=90,
        sem=semantic(genre="portrait", faces=True),
        art=flat(
            60, artistic_confidence=70,
            portrait=face(
                expression="MID_SPEECH", expression_quality=30,
                portrait_publishability=35, expression_confidence=80,
            ),
        ),
    )
    assert verdict.category == PhotoCategory.WEAK.value


def test_half_closed_eyes_fail_like_closed_ones(profile):
    verdict = judge(
        profile, tech=90,
        sem=semantic(genre="portrait", faces=True),
        art=flat(
            60, artistic_confidence=70,
            portrait=face(eyes_state="PARTIALLY_CLOSED", expression_confidence=80),
        ),
    )
    assert verdict.category == PhotoCategory.WEAK.value
    assert "half closed" in verdict.reasons[0]


def test_squinting_is_not_a_failure(profile):
    """Faces squint outdoors. That is not a bad photograph."""
    verdict = judge(
        profile, tech=85, uplift=6,
        sem=semantic(genre="portrait", faces=True, subject_strength=80),
        art=flat(
            78, artistic_confidence=78,
            portrait=face(eyes_state="SQUINTING", expression_confidence=80),
        ),
    )
    assert verdict.category != PhotoCategory.WEAK.value


def test_the_verdict_can_rest_on_the_numbers_rather_than_the_label(profile):
    """No enum can list every way a face fails, so the model's own scores count."""
    verdict = judge(
        profile, tech=88,
        sem=semantic(genre="portrait", faces=True),
        art=flat(
            65, artistic_confidence=70,
            portrait=face(
                expression="NEUTRAL", eyes_state="OPEN",
                expression_quality=32, portrait_publishability=30,
                expression_confidence=82,
            ),
        ),
    )
    assert verdict.category == PhotoCategory.WEAK.value
    assert "publishability" in verdict.reasons[0]


def test_a_merely_neutral_expression_is_not_a_failure(profile):
    """A calm face is not a broken one; both numbers have to be low."""
    verdict = judge(
        profile, tech=85, uplift=6,
        sem=semantic(genre="portrait", faces=True, subject_strength=80),
        art=flat(
            78, artistic_confidence=78,
            portrait=face(
                expression="NEUTRAL", expression_quality=55,
                portrait_publishability=70, expression_confidence=80,
            ),
        ),
    )
    assert verdict.category != PhotoCategory.WEAK.value


def test_an_unusable_reading_still_needs_confidence(profile):
    verdict = judge(
        profile, tech=85, uplift=6,
        sem=semantic(genre="portrait", faces=True, subject_strength=80),
        art=flat(
            78, artistic_confidence=78,
            portrait=face(
                expression_quality=30, portrait_publishability=30,
                expression_confidence=25,
            ),
        ),
    )
    assert verdict.category != PhotoCategory.WEAK.value
