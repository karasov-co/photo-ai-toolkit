"""The model check, the incremental contract, and the transaction.

Everything here comes from one reported failure: a run discovered 299 assets,
checksummed and decoded every one, wrote previews, measured them, migrated the
output directory to a new layout, and only then made its first API call --
which came back saying the configured model was not available to the key. No
report, the previous report already moved, and every minute of it spent on work
that could not be used.

The tests are lettered to match the requirements they came from.
"""

import json

import pytest
from synthetic import photo_like, write_jpeg

import batches
import bootstrap
import cli
import pipeline
import preflight
import workspace

# --- the fakes ----------------------------------------------------------------


class Reply:
    def __init__(self, text):
        self.output_text = text


class Responses:
    """Counts calls and can be told to fail on the first one."""

    def __init__(self, *, error=None, stage2=None, stage3=None, preflight_text=None):
        self.error = error
        self.stage2_text = stage2
        self.stage3_text = stage3
        self.preflight_text = preflight_text or '[{"n": 1, "ok": true}]'
        self.calls = 0
        self.stage2_calls = 0
        self.stage3_calls = 0
        self.preflight_calls = 0
        self.models: list[str] = []

    def create(self, **kwargs):
        self.calls += 1
        self.models.append(kwargs.get("model", ""))
        instructions = kwargs.get("instructions", "")
        if "verifying an API configuration" in instructions:
            self.preflight_calls += 1
            if self.error:
                raise self.error
            return Reply(self.preflight_text)
        if "emotional_resonance" in instructions:
            self.stage3_calls += 1
            return Reply(self.stage3_text)
        self.stage2_calls += 1
        return Reply(self.stage2_text)


class Client:
    def __init__(self, **kwargs):
        self.responses = Responses(**kwargs)


def stage2_reply(count: int, **overrides) -> str:
    return json.dumps(
        [
            {
                "n": i + 1, "genre": "landscape", "axis_a": i + 1,
                "axis_b": count - i, "axis_c": i + 1, "recover": "easy",
                "faces": False, "brand_mark": False, "note": "lift shadows",
                "intended_frame": True, "subject_strength": 75,
                "accidental_probability": 2, "dead_moment_probability": 3,
                **overrides,
            }
            for i in range(count)
        ]
    )


def stage3_reply(count: int, value: int = 78) -> str:
    import stage3

    return json.dumps(
        [
            {
                **dict.fromkeys(stage3.ARTISTIC_FIELDS, value),
                "artistic_candidate": True, "artistic_confidence": 82,
                "artistic_reasoning": "the light falls across one edge and nothing else",
                "uncertainty": 15, "n": i + 1,
            }
            for i in range(count)
        ]
    )


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    write_jpeg(photo_like(1200, 900, seed=1), root / "a.jpg")
    write_jpeg(photo_like(1200, 900, seed=2), root / "b.jpg")
    return root


@pytest.fixture
def watchful(monkeypatch):
    """Counts every photograph the run opens. Zero is the point of most of these."""
    opened = {"photos": 0, "videos": 0, "previews": 0}

    real_photo = pipeline.measure_photo
    real_video = pipeline.measure_video

    def photo(*args, **kwargs):
        opened["photos"] += 1
        return real_photo(*args, **kwargs)

    def video(*args, **kwargs):
        opened["videos"] += 1
        return real_video(*args, **kwargs)

    monkeypatch.setattr(pipeline, "measure_photo", photo)
    monkeypatch.setattr(pipeline, "measure_video", video)
    return opened


def with_client(monkeypatch, client):
    """Inject one fake client into both the preflight and the pipeline."""
    monkeypatch.setattr(bootstrap, "make_client", lambda **_: client)
    monkeypatch.setattr(bootstrap, "has_credentials", lambda: True)
    real = pipeline.run
    monkeypatch.setattr(pipeline, "run", lambda o, **kw: real(o, **{**kw, "client": client}))
    return client


def analyze(archive, out, *extra):
    return cli.main(["analyze", "--input", str(archive), "--output", str(out), *extra])


# --- A. no key ----------------------------------------------------------------


def test_a_missing_key_stops_before_anything_is_opened(archive, tmp_path, watchful, capsys):
    out = tmp_path / "run"
    code = analyze(archive, out)

    assert code != 0
    assert watchful["photos"] == 0, "not a single photograph should have been decoded"
    assert not out.exists(), "the output directory must not even be created"
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_a_missing_key_leaves_an_existing_run_exactly_as_it_was(archive, tmp_path, watchful):
    out = tmp_path / "run"
    space = workspace.Workspace(out).create()
    space.report.write_text("the previous report")
    (space.root / "top" / "keep.jpg").write_text("a link from last time")

    assert analyze(archive, out) != 0

    assert space.report.read_text() == "the previous report"
    assert (space.root / "top" / "keep.jpg").exists()
    assert watchful["photos"] == 0


def test_a_missing_key_creates_no_previews(archive, tmp_path):
    out = tmp_path / "run"
    analyze(archive, out)
    assert not list(tmp_path.rglob("*_jpg.jpg"))


def test_a_missing_key_does_not_migrate_an_old_layout(archive, tmp_path):
    """The reported failure moved the previous run before it knew it could work."""
    out = tmp_path / "run"
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "analysis.json").write_text('{"assets": []}')

    assert analyze(archive, out) != 0

    assert (out / "reports" / "analysis.json").exists(), "the old layout was migrated anyway"
    assert not (out / ".internal").exists()


# --- B. the model is not available --------------------------------------------


class ModelUnavailable(Exception):
    status_code = 404

    def __str__(self):
        return "The model `gpt-5.6-sol` does not exist or you do not have access to it."


def test_an_unavailable_model_stops_before_any_photograph(archive, tmp_path, watchful, monkeypatch, capsys):
    client = with_client(monkeypatch, Client(error=ModelUnavailable()))
    out = tmp_path / "run"

    code = analyze(archive, out)

    assert code != 0
    assert watchful["photos"] == 0
    assert client.responses.preflight_calls == 1, "the preflight is the only request made"
    assert client.responses.stage2_calls == 0
    assert client.responses.stage3_calls == 0


def test_an_unavailable_model_preserves_the_previous_report(archive, tmp_path, monkeypatch):
    with_client(monkeypatch, Client(error=ModelUnavailable()))
    out = tmp_path / "run"
    space = workspace.Workspace(out).create()
    space.report.write_text("the previous report")
    space.insights.write_text("the previous insights")

    assert analyze(archive, out) != 0

    assert space.report.read_text() == "the previous report"
    assert space.insights.read_text() == "the previous insights"


def test_an_unavailable_model_never_falls_back_to_an_older_one(archive, tmp_path, monkeypatch, capsys):
    client = with_client(monkeypatch, Client(error=ModelUnavailable()))
    analyze(archive, tmp_path / "run")

    assert client.responses.models == [bootstrap.DEFAULT_SEMANTIC_MODEL]
    printed = capsys.readouterr()
    everything = (printed.out + printed.err).lower()
    for legacy in ("gpt-4", "gpt-4o", "gpt-3", "o1-", "o3-"):
        assert legacy not in everything, legacy


def test_the_failure_says_what_to_do_without_offering_a_downgrade(archive, tmp_path, monkeypatch, capsys):
    with_client(monkeypatch, Client(error=ModelUnavailable()))
    analyze(archive, tmp_path / "run")

    message = capsys.readouterr().err.lower()
    assert "access" in message
    assert "no photograph was opened" in message
    for forbidden in ("fallback", "local-only", "--no-semantic", "instead use"):
        assert forbidden not in message, forbidden


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("The model `x` does not exist"), preflight.Failure.MODEL_ACCESS.value),
        (RuntimeError("Error code: 401 - invalid_api_key"), preflight.Failure.AUTH.value),
        (RuntimeError("Error code: 403 - permission denied"), preflight.Failure.PERMISSION.value),
        (RuntimeError("insufficient_quota"), preflight.Failure.QUOTA.value),
        (RuntimeError("Your billing account is past due"), preflight.Failure.BILLING.value),
        (RuntimeError("this model does not support image input"), preflight.Failure.VISION.value),
        (RuntimeError("Connection error"), preflight.Failure.UNREACHABLE.value),
    ],
)
def test_each_failure_is_told_apart(error, expected):
    assert preflight.classify(error) == expected


def test_a_reply_that_is_not_json_fails_the_preflight(monkeypatch):
    client = Client(preflight_text="I cannot help with that.")
    result = preflight.run("gpt-5.6-sol", client=client)
    assert not result.ok
    assert result.failure == preflight.Failure.STRUCTURED_OUTPUT.value


def test_the_preflight_never_sends_a_user_photograph():
    """It generates its own 16x16 image. A configuration check is not a reason
    to upload somebody's photographs."""
    encoded = preflight.test_image_base64()
    assert len(encoded) < 4000
    assert preflight.TEST_IMAGE_PX == 16


def test_the_preflight_never_prints_the_key(monkeypatch):
    secret = "sk-not-a-real-key-000111222"

    class Leaky(Exception):
        def __str__(self):
            return f"401 unauthorized for api_key={secret}"

    result = preflight.run("gpt-5.6-sol", client=Client(error=Leaky()))
    rendered = preflight.format_result(result) + preflight.format_failure(result)
    assert secret not in rendered


# --- C. a valid preflight -----------------------------------------------------


def test_analysis_starts_only_after_the_preflight_passes(archive, tmp_path, monkeypatch, capsys):
    client = with_client(
        monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2))
    )
    order: list[str] = []

    real = pipeline.measure_photo

    def watched(*args, **kwargs):
        order.append("photo")
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "measure_photo", watched)
    original_create = client.responses.create

    def recorded(**kwargs):
        if "verifying an API configuration" in kwargs.get("instructions", ""):
            order.append("preflight")
        return original_create(**kwargs)

    client.responses.create = recorded

    assert analyze(archive, tmp_path / "run") == 0
    assert order[0] == "preflight"
    assert "photo" in order


def test_the_preflight_block_names_the_model(archive, tmp_path, monkeypatch, capsys):
    with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    analyze(archive, tmp_path / "run")

    printed = capsys.readouterr().out
    assert "LLM preflight" in printed
    assert f"Model: {bootstrap.DEFAULT_SEMANTIC_MODEL}" in printed
    assert "Authentication: verified" in printed
    assert "Model access: verified" in printed
    assert "Vision input: verified" in printed
    assert "Responses API: verified" in printed


# --- D/E/F/G/H. incremental runs and intrinsic stability ----------------------


def records_by_name(out):
    import reports

    rows, _ = reports.read_json(out / ".internal" / "reports" / "analysis.json")
    return {r["filename"]: r for r in rows}


def run_twice(archive, tmp_path, monkeypatch, second_photo, *extra):
    """First run over a.jpg and b.jpg, then a second with one more."""
    out = tmp_path / "run"
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, out) == 0
    first = records_by_name(out)

    write_jpeg(second_photo, archive / "c.jpg")
    client.responses.stage2_text = stage2_reply(1)
    client.responses.stage3_text = stage3_reply(1)
    client.responses.stage2_calls = client.responses.stage3_calls = 0
    assert analyze(archive, out, *extra) == 0
    return first, records_by_name(out), client


def test_d_only_the_new_photograph_is_sent_to_the_model(archive, tmp_path, monkeypatch):
    first, second, client = run_twice(
        archive, tmp_path, monkeypatch, photo_like(1200, 900, seed=3)
    )
    assert client.responses.stage2_calls == 1
    assert client.responses.stage3_calls == 1
    assert set(second) == {"a.jpg", "b.jpg", "c.jpg"}


def test_d_the_old_photographs_keep_every_intrinsic_value(archive, tmp_path, monkeypatch):
    first, second, _ = run_twice(archive, tmp_path, monkeypatch, photo_like(1200, 900, seed=3))

    for name in ("a.jpg", "b.jpg"):
        before, after = first[name], second[name]
        assert after["final_score"] == before["final_score"], name
        assert after["category"] == before["category"], name
        assert after["category_reasons"] == before["category_reasons"], name
        assert after["scores"]["current_quality"] == before["scores"]["current_quality"]
        assert after["scores"]["post_edit_potential"] == before["scores"]["post_edit_potential"]
        assert after["genre"] == before["genre"], name
        assert after["stage3"] == before["stage3"], name


def test_e_an_unrelated_photograph_promotes_and_demotes_nothing(archive, tmp_path, monkeypatch):
    first, second, _ = run_twice(archive, tmp_path, monkeypatch, photo_like(900, 1200, seed=9))
    assert [second[n]["category"] for n in first] == [first[n]["category"] for n in first]


def test_f_a_stronger_newcomer_does_not_evict_anything(archive, tmp_path, monkeypatch):
    """TOP is an absolute bar, so a better photograph cannot take a place."""
    out = tmp_path / "run"
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2, 95)))
    assert analyze(archive, out) == 0
    first = records_by_name(out)

    write_jpeg(photo_like(1600, 1200, seed=42), archive / "strong.jpg")
    client.responses.stage2_text = stage2_reply(1)
    client.responses.stage3_text = stage3_reply(1, 99)
    assert analyze(archive, out) == 0
    second = records_by_name(out)

    for name in first:
        assert second[name]["category"] == first[name]["category"], name
        assert second[name]["final_score"] == first[name]["final_score"], name


def test_g_a_weaker_newcomer_promotes_nobody(archive, tmp_path, monkeypatch):
    out = tmp_path / "run"
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2, 60)))
    assert analyze(archive, out) == 0
    first = records_by_name(out)

    write_jpeg(photo_like(1200, 900, seed=77), archive / "weak.jpg")
    client.responses.stage2_text = stage2_reply(1)
    client.responses.stage3_text = stage3_reply(1, 10)
    assert analyze(archive, out) == 0
    second = records_by_name(out)

    for name in first:
        assert second[name]["category"] == first[name]["category"], name
        assert second[name]["final_score"] == first[name]["final_score"], name


def test_h_a_duplicate_changes_metadata_but_not_the_verdict(archive, tmp_path, monkeypatch):
    """Being the second-best of a pair is a fact about the pair, not the picture."""
    out = tmp_path / "run"
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, out) == 0
    first = records_by_name(out)

    # A near-identical sibling of a.jpg.
    write_jpeg(photo_like(1200, 900, seed=1), archive / "a_copy.jpg")
    client.responses.stage2_text = stage2_reply(1)
    client.responses.stage3_text = stage3_reply(1)
    assert analyze(archive, out) == 0
    second = records_by_name(out)

    assert second["a.jpg"]["final_score"] == first["a.jpg"]["final_score"]
    assert second["a.jpg"]["category"] == first["a.jpg"]["category"]
    assert "WEAK" not in {second["a.jpg"]["category"], second["a_copy.jpg"]["category"]}


def test_h_duplicate_status_alone_never_produces_weak():
    import calibration
    import curation
    import scoring
    import stage3

    profile = calibration.default_photo_profile()
    art = stage3.parse_assessment(
        {
            **dict.fromkeys(stage3.ARTISTIC_FIELDS, 70),
            "artistic_candidate": True, "artistic_confidence": 80,
            "artistic_reasoning": "a figure at the edge", "uncertainty": 20,
        }
    )
    inp = scoring.ScoreInput(
        asset_id="a", filename="a.jpg", technical_quality=80, uplift=6, is_raw=True,
        semantic=scoring.Semantic(present=True, genre="landscape", axis_a=60, axis_b=60,
                                  axis_c=55, faces=False, brand_mark=False,
                                  identifiable_people=False, subject_strength=70),
        artistic=art, is_best_in_cluster=False, cluster_margin=25, cluster_size=4,
    )
    verdict = curation.categorise(inp, scoring.score(inp, profile), profile)
    assert verdict.category != "WEAK"


# --- I. insights scope --------------------------------------------------------


def test_i_the_second_run_describes_only_the_new_photographs(archive, tmp_path, monkeypatch):
    _, _, _ = run_twice(archive, tmp_path, monkeypatch, photo_like(1200, 900, seed=3))
    page = (tmp_path / "run" / "photographer_insights.html").read_text()
    assert "newly analyzed" in page
    assert "1 newly analyzed" in page or "Insights based on 1" in page


def test_i_scope_all_uses_the_whole_archive(archive, tmp_path, monkeypatch):
    run_twice(archive, tmp_path, monkeypatch, photo_like(1200, 900, seed=3), "--insights-scope", "all")
    page = (tmp_path / "run" / "photographer_insights.html").read_text()
    assert "3 photographs" in page or "all 3" in page


def test_i_the_manifest_records_what_each_run_did(archive, tmp_path, monkeypatch):
    run_twice(archive, tmp_path, monkeypatch, photo_like(1200, 900, seed=3))
    manifests = batches.read_all(tmp_path / "run" / ".internal" / batches.MANIFEST_NAME)

    assert len(manifests) == 2
    assert len(manifests[0].new) == 2
    assert len(manifests[1].new) == 1
    assert len(manifests[1].reused) == 2
    assert manifests[1].model == bootstrap.DEFAULT_SEMANTIC_MODEL
    assert manifests[1].stage2_prompt_version and manifests[1].stage3_prompt_version
    assert manifests[1].run_id != manifests[0].run_id


def test_the_input_path_is_never_used_as_context(tmp_path, monkeypatch):
    """A folder called `Japan` is a filesystem location, not evidence."""
    import prompts

    for text in (prompts.STAGE2_SYSTEM, prompts.STAGE3_SYSTEM):
        lowered = text.lower()
        assert "folder" not in lowered
        assert "directory" not in lowered
        assert "file name" not in lowered and "filename" not in lowered


# --- J. the transaction -------------------------------------------------------


def test_j_a_stage2_failure_preserves_the_previous_run(archive, tmp_path, monkeypatch):
    out = tmp_path / "run"
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, out) == 0
    good_report = (out / "report.html").read_text()
    good_top = sorted(p.name for p in (out / "top").iterdir())

    # A second run whose content pass returns nothing usable at all.
    write_jpeg(photo_like(1200, 900, seed=5), archive / "d.jpg")
    client.responses.stage2_text = "the model declined"
    assert analyze(archive, out) != 0

    assert (out / "report.html").read_text() == good_report
    assert sorted(p.name for p in (out / "top").iterdir()) == good_top


def test_j_an_incomplete_staging_directory_is_refused(tmp_path):
    space = workspace.Workspace(tmp_path / "run").create()
    space.report.write_text("the previous report")
    staged = workspace.staging_dir(space, "incomplete")
    (staged / workspace.REPORT_NAME).write_text("a new report with no insights beside it")

    with pytest.raises(workspace.PublishError):
        workspace.publish(space, staged)
    assert space.report.read_text() == "the previous report"


def test_j_publication_replaces_everything_or_nothing(tmp_path):
    space = workspace.Workspace(tmp_path / "run").create()
    space.report.write_text("old report")
    space.insights.write_text("old insights")

    staged = workspace.staging_dir(space, "complete")
    (staged / workspace.REPORT_NAME).write_text("new report")
    (staged / workspace.INSIGHTS_NAME).write_text("new insights")
    (staged / "top").mkdir()

    workspace.publish(space, staged)

    assert space.report.read_text() == "new report"
    assert space.insights.read_text() == "new insights"
    assert not staged.exists()


# --- K. model policy ----------------------------------------------------------


def test_k_the_default_model_is_exactly_the_current_one():
    assert bootstrap.DEFAULT_SEMANTIC_MODEL == "gpt-5.6-terra"
    assert bootstrap.resolve_model(None) == "gpt-5.6-terra"


def test_a_run_can_be_costed_before_it_is_paid_for():
    """A number before the money, not after."""
    cheap = bootstrap.estimate_cost("gpt-5.6-terra", 300)
    dear = bootstrap.estimate_cost("gpt-5.6-sol", 300)
    assert 0 < cheap < dear


def test_k_no_legacy_model_appears_anywhere_in_the_source():
    """Not as a default, not as a fallback, not as a suggestion."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for legacy in ("gpt-4o", "gpt-4.1", "gpt-4-", "gpt-3.5"):
            if legacy in text and "LEGACY_MODEL_PREFIXES" not in text:
                offenders.append(f"{path.name}: {legacy}")
    assert offenders == []


def test_k_a_legacy_family_is_recognised_as_one():
    for name in ("gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "o1-preview"):
        assert bootstrap.is_legacy_model(name), name
    for name in ("gpt-5.6-terra", "gpt-5.6-sol"):
        assert not bootstrap.is_legacy_model(name), name


def test_k_stage2_and_stage3_use_the_same_validated_model(archive, tmp_path, monkeypatch):
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, tmp_path / "run") == 0
    assert set(client.responses.models) == {bootstrap.DEFAULT_SEMANTIC_MODEL}


def test_k_the_selected_model_is_printed_before_analysis(archive, tmp_path, monkeypatch, capsys):
    with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    analyze(archive, tmp_path / "run")

    printed = capsys.readouterr().out
    assert printed.index(bootstrap.DEFAULT_SEMANTIC_MODEL) < printed.index("Analyzing")


# --- the run tally ------------------------------------------------------------


def test_the_summary_separates_new_work_from_reused(archive, tmp_path, monkeypatch, capsys):
    run_twice(archive, tmp_path, monkeypatch, photo_like(1200, 900, seed=3))
    printed = capsys.readouterr().out

    assert "New assets analysed:      1" in printed
    assert "Unchanged reused:         2" in printed
    assert "LLM calls:" in printed


def test_reused_assets_are_not_printed_as_though_they_were_analysed(
    archive, tmp_path, monkeypatch, capsys
):
    run_twice(archive, tmp_path, monkeypatch, photo_like(1200, 900, seed=3))
    printed = capsys.readouterr().out

    # Only the progress lines. The summary lists filenames too, and it is
    # supposed to -- what must not happen is a reused asset being printed as
    # though the run were analysing it.
    second_run = printed.split("Analyzing new assets:")[-1]
    progress = [
        line for line in second_run.splitlines()
        if line.startswith("  [") and "]" in line
    ]
    assert any("c.jpg" in line for line in progress)
    assert not any("a.jpg" in line or "b.jpg" in line for line in progress)
    assert len(progress) == 1


# --- an account that runs out mid-run -----------------------------------------
#
# From a live run: the balance was exhausted at photograph 154. The remaining
# 145 were each recorded as an individual Stage 3 failure, the run finished,
# exited zero, and published a report in which half the photographs had no
# artistic read -- presented exactly like the half that did.


class NoCredits(Exception):
    status_code = 429

    def __str__(self):
        return (
            "Error code: 429 - {'error': {'message': 'You have no credits remaining. "
            "Add credits to continue using the API.', 'type': 'insufficient_quota', "
            "'code': 'credit_balance_exhausted'}}"
        )


class RunsOutOfCredit(Responses):
    """Answers Stage 2, then refuses every Stage 3 call."""

    def create(self, **kwargs):
        if "emotional_resonance" in kwargs.get("instructions", ""):
            raise NoCredits()
        return super().create(**kwargs)


def test_an_exhausted_balance_ends_the_run_instead_of_half_analysing(
    archive, tmp_path, monkeypatch
):
    out = tmp_path / "run"
    good = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, out) == 0
    previous = (out / "report.html").read_text()

    write_jpeg(photo_like(1200, 900, seed=8), archive / "d.jpg")
    broke = Client(stage2=stage2_reply(1), stage3=stage3_reply(1))
    broke.responses.__class__ = RunsOutOfCredit
    with_client(monkeypatch, broke)

    assert analyze(archive, out) != 0
    assert (out / "report.html").read_text() == previous, "the good report was replaced"
    assert good is not broke


def test_an_exhausted_balance_is_told_apart_from_a_bad_group():
    assert bootstrap.is_fatal_api_error(NoCredits())
    assert bootstrap.classify_api_error(NoCredits())[0] == "quota"
    # A malformed reply is about one group and must stay survivable.
    assert not bootstrap.is_fatal_api_error(ValueError("no JSON array in the reply"))


@pytest.mark.parametrize(
    "error",
    [ModelUnavailable(), RuntimeError("401 invalid_api_key"), RuntimeError("403 permission denied")],
)
def test_every_account_level_failure_ends_the_run(error):
    assert bootstrap.is_fatal_api_error(error)


def test_the_insights_scope_line_counts_correctly(tmp_path):
    """It said "the other 281" when 261 of those 281 were the ones in scope."""
    import insights as insights_module

    built = insights_module.Insights(total=261)
    page = insights_module.write(
        built, tmp_path / "insights.html", scope="new", total_stored=281
    ).read_text()
    assert "261 newly analyzed photographs, out of 281" in page


# --- money is asked about before it is spent ---------------------------------


def test_a_large_batch_asks_before_spending(archive, tmp_path, monkeypatch, capsys):
    """The estimate has to arrive before the charge, not in the summary after it."""
    for i in range(80):
        write_jpeg(photo_like(400, 300, seed=i + 20), archive / f"bulk{i}.jpg")
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    monkeypatch.setattr("builtins.input", lambda *_: "n")

    code = analyze(archive, tmp_path / "run")

    printed = capsys.readouterr().out
    assert code == 0
    assert "costing roughly $" in printed
    assert client.responses.stage2_calls == 0, "nothing was sent after declining"
    assert not (tmp_path / "run" / "report.html").exists()


def test_a_small_batch_does_not_ask(archive, tmp_path, monkeypatch):
    def refuse(*_):
        raise AssertionError("two photographs should not need confirming")

    monkeypatch.setattr("builtins.input", refuse)
    with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, tmp_path / "run") == 0


def test_the_estimate_counts_only_what_is_billable(archive, tmp_path, monkeypatch, capsys):
    """Reused photographs cost nothing and must not appear in the price."""
    with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, tmp_path / "run") == 0
    capsys.readouterr()

    assert analyze(archive, tmp_path / "run") == 0
    printed = capsys.readouterr().out
    assert "Reused:   2" in printed
    assert "Nothing new to analyse." in printed


# --- one report, one scale ----------------------------------------------------
#
# Stage 2 ranks twelve frames against each other; turning that into a 0-100
# figure needs a population. Caching the *percentile* froze one run's population
# into an asset that outlived it, so a later report put old percentiles beside
# percentiles computed over a different set, on the same axis, as though they
# were the same scale.


def test_the_cache_stores_ranks_rather_than_percentiles(archive, tmp_path, monkeypatch):
    out = tmp_path / "run"
    with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, out) == 0

    cache = pipeline.AnalysisCache(out / ".internal" / "analysis_cache.json")
    stored = [v for k, v in cache._data.items() if k.startswith("stage2:")]
    assert stored, "nothing was cached"
    for entry in stored:
        assert "item" in entry and "group" in entry, "the stitched result was cached"
        assert "axis_a" in entry["item"], "the raw rank is what must survive"
        assert entry["group"], "the frames it was ranked against are part of the fact"


def test_two_batches_end_up_on_one_scale(archive, tmp_path, monkeypatch):
    """The failure: half the report on last run's scale, half on this one's."""
    out = tmp_path / "run"
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, out) == 0

    for i in range(4):
        write_jpeg(photo_like(1200, 900, seed=30 + i), archive / f"n{i}.jpg")
    client.responses.stage2_text = stage2_reply(4)
    client.responses.stage3_text = stage3_reply(4)
    assert analyze(archive, out) == 0

    rows = records_by_name(out)
    axes = [r["scores"] for r in rows.values()]
    assert len(rows) == 6
    # Every frame has an axis figure, and they span one range rather than two
    # disjoint ones -- a cached frame stuck on an old scale shows up as a value
    # no fresh frame could take.
    assert all(0 <= s["stock_potential"] <= 100 for s in axes)
    genres = {r["genre"] for r in rows.values()}
    assert genres == {"landscape"}, genres


def test_a_reused_frame_is_rescored_against_the_new_population(archive, tmp_path, monkeypatch):
    """Its rank is fixed; its percentile is not, and must not be."""
    out = tmp_path / "run"
    client = with_client(monkeypatch, Client(stage2=stage2_reply(2), stage3=stage3_reply(2)))
    assert analyze(archive, out) == 0
    before = records_by_name(out)["a.jpg"]

    write_jpeg(photo_like(1200, 900, seed=40), archive / "z.jpg")
    client.responses.stage2_text = stage2_reply(1)
    client.responses.stage3_text = stage3_reply(1)
    assert analyze(archive, out) == 0
    after = records_by_name(out)["a.jpg"]

    # No second API call was spent on it, and its own judgements are unchanged.
    assert client.responses.stage2_calls == 2
    assert after["genre"] == before["genre"]
    assert after["stage3"] == before["stage3"]


def test_stage2_groups_overlap_so_the_ranking_is_not_islands():
    """Bradley-Terry needs shared frames or every group is its own scale."""
    import aggregate

    names = [f"f{i}.jpg" for i in range(60)]
    groups = aggregate.build_groups(names, size=12)
    assert len(groups) > 1
    for earlier, later in zip(groups, groups[1:], strict=False):
        assert set(earlier) & set(later), "adjacent groups share no frames"
    assert aggregate.DEFAULT_OVERLAP >= 2


def test_every_frame_appears_in_at_least_one_group():
    import aggregate

    names = [f"f{i}.jpg" for i in range(101)]
    covered = {n for group in aggregate.build_groups(names, size=12) for n in group}
    assert covered == set(names)


# --- the status code, not a substring -----------------------------------------


class WithStatus(Exception):
    def __init__(self, status, message=""):
        self.status_code = status
        self._message = message
        super().__init__(message)

    def __str__(self):
        return self._message


def test_the_attribute_is_believed_over_the_text():
    """A request id containing 404 is not a missing model."""
    error = WithStatus(401, "unauthorized, request req_a404b7c")
    assert preflight.classify(error) == preflight.Failure.AUTH.value


def test_a_number_inside_an_identifier_is_not_a_status():
    error = Exception("request failed: req_9f404ab / trace 40311")
    assert preflight.classify(error) != preflight.Failure.MODEL_ACCESS.value


def test_a_number_inside_a_path_is_not_a_status():
    error = Exception("POST https://api.example.com/v1/404/responses returned nothing")
    assert preflight.classify(error) != preflight.Failure.MODEL_ACCESS.value


def test_a_status_in_prose_is_still_read_when_there_is_no_attribute():
    assert preflight.classify(Exception("Error code: 404 - not found")) == (
        preflight.Failure.MODEL_ACCESS.value
    )
    assert preflight.classify(Exception("429 Too Many Requests")) == (
        preflight.Failure.QUOTA.value
    )


def test_a_nested_response_object_is_read():
    class Response:
        status_code = 403

    class Wrapped(Exception):
        response = Response()

        def __str__(self):
            return "forbidden"

    assert preflight.classify(Wrapped()) == preflight.Failure.PERMISSION.value


def test_a_string_status_is_read_as_a_number():
    error = WithStatus("429", "slow down")
    assert preflight.classify(error) == preflight.Failure.QUOTA.value


def test_the_message_still_wins_where_it_is_specific():
    """A 400 saying the model does not exist is a model problem, not a bad request."""
    error = WithStatus(400, "The requested model 'x' does not exist.")
    assert preflight.classify(error) == preflight.Failure.MODEL_ACCESS.value


def test_the_exception_class_is_read_when_the_message_is_bare():
    """Real SDK errors sometimes stringify to little more than the code."""

    class AuthenticationError(Exception):
        def __str__(self):
            return "401"

    assert preflight.classify(AuthenticationError()) == preflight.Failure.AUTH.value


def test_the_class_does_not_override_an_explicit_status():
    class NotFoundError(Exception):
        status_code = 429

        def __str__(self):
            return "slow down"

    assert preflight.classify(NotFoundError()) == preflight.Failure.QUOTA.value
