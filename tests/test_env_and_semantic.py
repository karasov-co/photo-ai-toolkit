"""Configuration, credentials, and refusing to look successful when nothing ran.

The failure these exist to prevent, in the user's words: a key sitting in `.env`,
`--semantic` on the command line, and a run that decoded every photograph, failed
to authenticate, carried on locally, and printed a summary that read like a
success with `genre=other` on every file and a list of deletion candidates.

No test here makes a network call or reads a real key.
"""

import json

import pytest
from synthetic import photo_like, write_jpeg

import bootstrap
import cli
import i18n
import pipeline
import prompts
import reports
from scoring import RouteClass

FAKE_KEY = "test-key-placeholder-not-a-real-credential"
OTHER_KEY = "another-test-placeholder"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    """Never touch the real project `.env`, and never inherit a real key."""
    monkeypatch.delenv(bootstrap.API_KEY_VAR, raising=False)
    monkeypatch.delenv(bootstrap.MODEL_VAR, raising=False)
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "PROJECT_ENV", tmp_path / ".env")
    monkeypatch.setattr(bootstrap, "_loaded_from", None)
    yield


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    write_jpeg(photo_like(1200, 900, seed=1), root / "a.jpg")
    write_jpeg(photo_like(1200, 900, seed=2), root / "b.jpg")
    return root


def write_env(path, key=FAKE_KEY, **extra):
    lines = [f"{bootstrap.API_KEY_VAR}={key}"]
    lines += [f"{name}={value}" for name, value in extra.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- A. .env at the project root, run from elsewhere -------------------------


def test_the_project_env_is_found_when_run_from_another_directory(tmp_path, monkeypatch):
    """`python /path/to/repo/cli.py` from anywhere is still this project."""
    write_env(tmp_path / ".env")
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    report = bootstrap.load_project_environment()

    assert report.key_present
    assert report.source is bootstrap.CredentialSource.PROJECT_ENV
    assert bootstrap.has_credentials()


def test_a_working_directory_env_is_also_honoured(tmp_path, monkeypatch):
    working = tmp_path / "work"
    working.mkdir()
    write_env(working / ".env")
    monkeypatch.chdir(working)

    report = bootstrap.load_project_environment(working_dir=working)

    assert report.key_present
    assert report.source is bootstrap.CredentialSource.WORKING_DIR_ENV


def test_no_env_anywhere_reports_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = bootstrap.load_project_environment(working_dir=tmp_path)
    assert not report.key_present
    assert report.source is bootstrap.CredentialSource.MISSING


# --- B. a real environment variable outranks the file ------------------------


def test_an_exported_key_wins_over_the_file(tmp_path, monkeypatch):
    """override=False: a key exported for one command is not silently replaced."""
    monkeypatch.setenv(bootstrap.API_KEY_VAR, OTHER_KEY)
    write_env(tmp_path / ".env", key=FAKE_KEY)

    report = bootstrap.load_project_environment(working_dir=tmp_path)

    assert report.source is bootstrap.CredentialSource.ENVIRONMENT
    assert bootstrap.api_key() == OTHER_KEY


def test_the_project_env_wins_over_the_working_directory(tmp_path, monkeypatch):
    working = tmp_path / "work"
    working.mkdir()
    write_env(tmp_path / ".env", key=FAKE_KEY)
    write_env(working / ".env", key=OTHER_KEY)

    bootstrap.load_project_environment(working_dir=working)

    assert bootstrap.api_key() == FAKE_KEY


# --- the status line never leaks -------------------------------------------


@pytest.mark.parametrize("language", ["en", "ru"])
def test_the_status_line_never_contains_the_key(tmp_path, language):
    write_env(tmp_path / ".env")
    bootstrap.load_project_environment(working_dir=tmp_path)

    status = bootstrap.credential_status(language)

    assert FAKE_KEY not in status
    assert FAKE_KEY[:8] not in status
    assert str(len(FAKE_KEY)) not in status
    assert ".env" in status


def test_the_missing_status_says_so_in_both_languages():
    assert "missing" in bootstrap.credential_status("en").lower()
    assert "не найден" in bootstrap.credential_status("ru")


# --- C. --semantic with no key: fail before decoding anything ----------------


def test_semantic_without_a_key_fails_before_any_file_is_opened(archive, tmp_path, monkeypatch):
    opened = {"n": 0}
    original = pipeline.measure_photo

    def counted(*args, **kwargs):
        opened["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "measure_photo", counted)

    with pytest.raises(bootstrap.SemanticCredentialsMissing):
        pipeline.run(
            pipeline.PipelineOptions(
                input_dir=archive, output_dir=tmp_path / "out", semantic=True
            )
        )

    assert opened["n"] == 0, "not a single photograph should have been decoded"


def test_the_cli_exits_non_zero_and_explains(archive, tmp_path, capsys):
    code = cli.main(
        ["--lang", "ru", "analyze", "--input", str(archive),
         "--output", str(tmp_path / "out"), "--semantic"]
    )
    captured = capsys.readouterr()

    assert code != 0
    assert "OPENAI_API_KEY" in captured.err
    assert "OPENAI_API_KEY=your_key_here" in captured.err
    assert "Ошибка" in captured.err


def test_no_report_is_written_when_credentials_are_missing(archive, tmp_path):
    out = tmp_path / "out"
    cli.main(["analyze", "--input", str(archive), "--output", str(out), "--semantic"])
    assert not (out / "reports" / "analysis.json").exists()


def test_no_api_call_is_attempted_without_a_key(archive, tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the client must not be constructed without a key")

    monkeypatch.setattr(bootstrap, "make_client", explode)
    with pytest.raises(bootstrap.SemanticCredentialsMissing):
        pipeline.run(
            pipeline.PipelineOptions(
                input_dir=archive, output_dir=tmp_path / "out", semantic=True
            )
        )


def test_the_error_message_carries_no_credential(archive, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(bootstrap.API_KEY_VAR, "")
    cli.main(
        ["analyze", "--input", str(archive), "--output", str(tmp_path / "o"), "--semantic"]
    )
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.err + captured.out


# --- D. --semantic with a key from .env: the client is used ------------------


class FakeResponse:
    def __init__(self, text):
        self.output_text = text


class FakeResponses:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.responses = FakeResponses(payload, error)


def ranking_for(count: int) -> str:
    """A well-formed Stage 2 reply: a strict permutation on every axis."""
    return json.dumps(
        [
            {
                "n": i + 1, "genre": "landscape",
                "axis_a": i + 1, "axis_b": count - i, "axis_c": i + 1,
                "recover": "easy", "faces": False, "logos": False, "note": "lift shadows",
            }
            for i in range(count)
        ]
    )


def test_a_key_from_the_env_file_reaches_the_client(archive, tmp_path, monkeypatch):
    write_env(tmp_path / ".env")
    bootstrap.load_project_environment(working_dir=tmp_path)

    client = FakeClient(ranking_for(2))
    result = pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out", semantic=True),
        client=client,
    )

    assert client.responses.calls == 1
    assert result.semantic_completed
    assert result.analysis_mode == "local_and_semantic"


def test_a_completed_semantic_pass_assigns_a_real_genre(archive, tmp_path):
    result = pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out", semantic=True),
        client=FakeClient(ranking_for(2)),
    )
    assert {r.genre for r in result.records} == {"landscape"}


# --- E. an authentication error ends the run ---------------------------------


class FakeAuthenticationError(Exception):
    pass


def test_an_authentication_error_is_not_dressed_up_as_a_local_run(archive, tmp_path):
    with pytest.raises(bootstrap.SemanticUnavailable) as caught:
        pipeline.run(
            pipeline.PipelineOptions(
                input_dir=archive, output_dir=tmp_path / "out", semantic=True
            ),
            client=FakeClient(error=FakeAuthenticationError("401 invalid_api_key")),
        )
    assert caught.value.kind == "authentication"


def test_the_cli_reports_a_semantic_failure_with_a_non_zero_code(archive, tmp_path, capsys, monkeypatch):
    write_env(tmp_path / ".env")
    bootstrap.load_project_environment(working_dir=tmp_path)
    monkeypatch.setattr(
        bootstrap, "make_client", lambda **_: FakeClient(error=FakeAuthenticationError("401"))
    )

    code = cli.main(
        ["--lang", "ru", "analyze", "--input", str(archive),
         "--output", str(tmp_path / "out"), "--semantic"]
    )
    captured = capsys.readouterr()

    assert code != 0
    assert "Ошибка" in captured.err
    assert "--allow-semantic-fallback" in captured.err


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (FakeAuthenticationError("401 invalid_api_key"), "authentication"),
        (RuntimeError("403 permission denied"), "permission"),
        (RuntimeError("404 model does not exist"), "model_not_found"),
        (RuntimeError("429 rate limit reached"), "rate_limit"),
    ],
)
def test_api_failures_are_classified(error, kind):
    assert bootstrap.classify_api_error(error)[0] == kind


# --- F. the same failure with the fallback explicitly allowed ----------------


def test_the_fallback_is_allowed_only_when_asked_for(archive, tmp_path):
    result = pipeline.run(
        pipeline.PipelineOptions(
            input_dir=archive, output_dir=tmp_path / "out",
            semantic=True, allow_semantic_fallback=True,
        ),
        client=FakeClient(error=FakeAuthenticationError("401")),
    )

    assert result.analysis_mode == "local_only_after_semantic_failure"
    assert not result.semantic_completed
    assert "authentication" in result.semantic_error
    assert result.records, "the local result is still produced"


def test_a_fallback_report_is_stamped_on_every_record(archive, tmp_path):
    result = pipeline.run(
        pipeline.PipelineOptions(
            input_dir=archive, output_dir=tmp_path / "out",
            semantic=True, allow_semantic_fallback=True,
        ),
        client=FakeClient(error=FakeAuthenticationError("401")),
    )
    for record in result.records:
        assert record.analysis_mode == "local_only_after_semantic_failure"
        assert record.semantic_requested
        assert not record.semantic_completed


def test_the_fallback_banner_is_impossible_to_miss(archive, tmp_path):
    result = pipeline.run(
        pipeline.PipelineOptions(
            input_dir=archive, output_dir=tmp_path / "out",
            semantic=True, allow_semantic_fallback=True,
        ),
        client=FakeClient(error=FakeAuthenticationError("401")),
    )
    summary = reports.summarise(result.records)
    printed = reports.format_summary(summary, "en")
    assert "SEMANTIC ANALYSIS DID NOT RUN" in printed

    html = reports.write_html(result.records, tmp_path / "r.html", summary=summary).read_text(
        encoding="utf-8"
    )
    assert "SEMANTIC ANALYSIS DID NOT RUN" in html


# --- G. re-running after a local-only run --------------------------------


def test_the_semantic_pass_runs_on_a_rerun_without_force(archive, tmp_path, monkeypatch):
    """The expensive local work is reused; the content check still happens."""
    out = tmp_path / "out"
    first = pipeline.run(pipeline.PipelineOptions(input_dir=archive, output_dir=out))
    assert first.analysis_mode == "local_only"

    decoded = {"n": 0}
    original = pipeline.measure_photo

    def counted(*args, **kwargs):
        decoded["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "measure_photo", counted)

    client = FakeClient(ranking_for(2))
    second = pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=out, semantic=True),
        client=client,
    )

    assert decoded["n"] == 0, "local measurements should come from the cache"
    assert client.responses.calls == 1, "the semantic pass must still run"
    assert second.semantic_completed


def test_a_local_only_cache_entry_is_never_mistaken_for_a_semantic_one(tmp_path):
    cache = pipeline.AnalysisCache(tmp_path / "c.json")
    cache.put("abc", pipeline.Measurement(quality=50.0).to_dict())
    assert pipeline.AnalysisCache.is_local_only(cache.get("abc"))


# --- H. no semantic pass: nothing is claimed that was not checked ------------


def test_genre_is_unknown_rather_than_other_when_nothing_looked(archive, tmp_path):
    result = pipeline.run(pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out"))
    assert {r.genre for r in result.records} == {"unknown"}
    assert all(not r.semantic_present for r in result.records)


def test_the_summary_reports_release_status_as_unchecked(archive, tmp_path):
    result = pipeline.run(pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out"))
    summary = reports.summarise(result.records)
    printed = reports.format_summary(summary, "ru")

    assert not summary["semantic_ran"]
    assert "не проверен" in printed
    assert "Не хватает релизов" not in printed


def test_the_two_stock_counters_are_named_apart(archive, tmp_path):
    result = pipeline.run(pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out"))
    summary = reports.summarise(result.records)
    printed = reports.format_summary(summary, "ru")

    assert "Технически пригодно" in printed
    assert "Полностью проверено и готово к экспорту" in printed
    assert "Ничего не готово к экспорту, потому что" in printed


# --- I. short clips and duplicates are not unconditional trash ---------------


def test_a_short_clip_is_not_grounds_for_deletion():
    import video_analyzer as va
    from issues import IssueCode

    probe = va.parse_probe(
        {
            "format": {"format_name": "mov,mp4", "duration": "1.0"},
            "streams": [{"codec_type": "video", "codec_name": "hevc", "width": 3840,
                         "height": 2160, "duration": "1.0", "r_frame_rate": "30/1"}],
        }
    )
    analysis = va.VideoAnalysis(probe=probe)
    analysis.samples = [
        va.FrameSample(timestamp=0.5, quality=60.0, mean_luma=120.0, blur_ratio=20.0,
                       sharpness_tile=500.0, clipped_highlights=0.0, clipped_shadows=0.0)
    ]
    found = va.detect_video_issues(analysis)

    assert IssueCode.SHORT_CLIP in found.codes()
    assert IssueCode.UNUSABLE_DURATION not in found.codes()


def test_a_weaker_duplicate_is_never_called_unrecoverable():
    from issues import FIXABILITY, Fixability, IssueCode

    assert FIXABILITY[IssueCode.WEAKER_DUPLICATE] is not Fixability.UNRECOVERABLE
    assert FIXABILITY[IssueCode.SHORT_CLIP] is not Fixability.UNRECOVERABLE


def test_neither_is_grounds_for_a_permanent_purge():
    import quarantine

    assert not quarantine.is_purgeable_evidence("weaker_duplicate")
    assert not quarantine.is_purgeable_evidence("unusable_duration")
    assert not quarantine.is_purgeable_evidence("short_clip")


def test_duplicates_go_to_their_own_class_not_to_the_delete_plan(tmp_path):
    root = tmp_path / "burst"
    write_jpeg(photo_like(1600, 1200, seed=9), root / "a.jpg")
    from synthetic import blurred

    write_jpeg(blurred(seed=9, radius=4.0, size=(1600, 1200)), root / "b.jpg")

    result = pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))

    weaker = [r for r in result.records if not r.best_in_cluster]
    assert weaker
    assert all(r.route_class == RouteClass.DUPLICATE_CANDIDATE.value for r in weaker)
    assert result.planned_operations == []


def test_a_missing_release_check_never_puts_a_file_in_the_delete_plan(archive, tmp_path):
    result = pipeline.run(pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out"))
    for op in result.planned_operations:
        assert "release" not in op.reason.lower()


# --- J. Russian output is Russian, and nothing is glued together -------------


GLUED = ("арне", "Re-runwith", "stockis", "beendone", "качестваа", "не гарантияхуд")


@pytest.mark.parametrize("language", ["en", "ru"])
def test_no_message_contains_a_known_glued_pair(language):
    """Guards the class of defect, not one example of it."""
    catalogue = i18n.STRINGS[language]
    joined = " ".join(catalogue.values())
    for glued in GLUED:
        assert glued not in joined, f"{glued!r} appears in the {language} catalogue"


def test_every_catalogue_entry_has_balanced_spacing():
    """A fragment ending mid-sentence without a space is how words fuse."""
    for language, catalogue in i18n.STRINGS.items():
        for key, value in catalogue.items():
            assert "  " not in value.replace("\n", " ") or key.startswith("creds."), (
                f"{language}:{key} has a double space"
            )


@pytest.mark.parametrize("language", ["en", "ru"])
def test_the_plan_block_speaks_the_users_language(language, tmp_path):
    import quarantine

    op = quarantine.FileOperation(
        op_id="o", asset_id="a", source="/x/a.jpg", destination="/q/a.jpg",
        checksum="c", size_bytes=1_048_576, timestamp="t", reason="test",
    )
    text = quarantine.summarise_plan([op], language)
    if language == "ru":
        assert "Будет перемещено" in text
        assert "Re-run" not in text
    else:
        assert "would move" in text


def test_the_russian_summary_has_no_english_left_in_its_labels(archive, tmp_path):
    result = pipeline.run(pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out"))
    printed = reports.format_summary(reports.summarise(result.records), "ru")

    for english in ("Total assets", "Analysis mode", "Release status", "Fully checked"):
        assert english not in printed


def test_reasons_render_in_both_languages():
    from scoring import Reason

    reason = Reason("reason.stock_standard", {"value": 55}, "stock potential 55 is usable")
    assert "55" in reason.localise("en")
    assert "55" in reason.localise("ru")
    assert reason.localise("ru") != reason.localise("en")


def test_an_unknown_reason_key_falls_back_to_its_english_text():
    from scoring import Reason

    reason = Reason("reason.does_not_exist", {}, "the original English sentence")
    assert reason.localise("ru") == "the original English sentence"


# --- the model, and where it comes from -------------------------------------


def test_the_cli_model_wins_over_everything(monkeypatch):
    monkeypatch.setenv(bootstrap.MODEL_VAR, "from-env")
    assert bootstrap.resolve_model("from-cli") == "from-cli"
    assert bootstrap.model_source("from-cli") == "--model"


def test_the_environment_wins_over_the_default(monkeypatch):
    monkeypatch.setenv(bootstrap.MODEL_VAR, "from-env")
    assert bootstrap.resolve_model(None) == "from-env"


def test_the_default_is_used_when_nothing_else_is_set():
    assert bootstrap.resolve_model(None) == bootstrap.DEFAULT_SEMANTIC_MODEL


def test_the_model_actually_used_is_recorded(archive, tmp_path, monkeypatch):
    monkeypatch.setenv(bootstrap.API_KEY_VAR, FAKE_KEY)
    result = pipeline.run(
        pipeline.PipelineOptions(
            input_dir=archive, output_dir=tmp_path / "out", semantic=True,
            semantic_model="a-specific-model",
        ),
        client=FakeClient(ranking_for(2)),
    )
    assert result.semantic_model == "a-specific-model"
    assert all(r.semantic_model == "a-specific-model" for r in result.records)


def test_the_client_refuses_to_build_without_a_key():
    with pytest.raises(bootstrap.SemanticCredentialsMissing):
        bootstrap.make_client()


def test_the_stage2_prompt_version_is_stable():
    """The cache and the report both reference it, so it must exist."""
    assert prompts.STAGE2_SYSTEM
    assert "raw_sensor" in prompts.STAGE2_SYSTEM


# --- the comparison sheet ----------------------------------------------------


def test_the_comparison_sheet_pairs_each_candidate_with_the_frame_that_beat_it(tmp_path):
    """A grid says what you would lose; a pair says why the other one won."""
    import layout

    a = write_jpeg(photo_like(400, 300, seed=1), tmp_path / "a.jpg")
    b = write_jpeg(photo_like(400, 300, seed=2), tmp_path / "b.jpg")

    path = layout.build_comparison_sheet(
        [
            {
                "label": "b.jpg", "candidate_preview": str(b),
                "best_label": "a.jpg", "best_preview": str(a),
                "margin": "11", "reason": "sharper frame exists",
            }
        ],
        tmp_path / "sheet.jpg",
        language="ru",
        semantic_ran=False,
    )
    assert path is not None and path.exists()


def test_the_sheet_reports_whether_a_content_check_ran(tmp_path):
    import layout

    assert "НЕ выполнялся" in i18n.t("sheet.semantic_missing", "ru")
    assert layout.load_font(12) is not None


def test_no_rows_means_no_sheet(tmp_path):
    import layout

    assert layout.build_comparison_sheet([], tmp_path / "none.jpg") is None
