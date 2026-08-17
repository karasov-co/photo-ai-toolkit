"""Reports, localisation, and the rule that no credential reaches disk."""

import csv
import json
import logging

import pytest

from photoai import i18n, reports
from photoai.reports import AssetRecord, RedactingFilter, redact


def record(**kwargs):
    base = {
        "asset_id": "a1",
        "source_path": "/archive/P1042675.RW2",
        "filename": "P1042675.RW2",
        "media_type": "photo",
        "checksum": "c" * 64,
        "route_class": "stock_strong",
        "route": "commercial",
        "confidence": 78,
        "genre": "landscape",
        "scores": {
            "current_quality": 44, "recoverability": 100, "post_edit_potential": 81,
            "aesthetic_potential": 62, "stock_potential": 55, "portfolio_potential": 69,
            "uniqueness": 100, "confidence": 78, "routing_score": 71,
        },
        "expected_gain": 37,
        "issues": {"fixable": ["underexposed: mean luma 44"], "partially_fixable": [], "unrecoverable": []},
        "edit_recipe": ["Adjust exposure: +1.4 EV"],
        "marketplaces": [
            {
                "platform": "Adobe Stock", "platform_id": "adobe_stock",
                "eligible": True, "export_ready": True,
            }
        ],
        "reasons": ["stock potential 55 is usable after the suggested edit"],
        "tags": ["commercial_ok"],
    }
    return AssetRecord(**{**base, **kwargs})


# --- credential hygiene -----------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "ghp_MJxKEwNLcazW33mGPS1r6P3EkBQ3H627Hp",
        "github_pat_11ABCDEFG0abcdefghijklmnop",
    ],
)
def test_a_credential_never_survives_redaction(secret):
    """A key in a traceback ends up in the log the user attaches to a bug report."""
    assert secret not in redact(f"request failed with token {secret}")
    assert "[REDACTED]" in redact(f"request failed with token {secret}")


def test_a_labelled_secret_is_redacted():
    assert "hunter2" not in redact("api_key: hunter2")


def test_a_credential_in_a_url_is_redacted():
    assert "s3cret" not in redact("cloning https://user:s3cret@github.com/org/repo.git")


def test_ordinary_text_is_left_alone():
    assert redact("Processing P1042675.RW2") == "Processing P1042675.RW2"


def test_empty_text_is_handled():
    assert redact("") == ""


def test_the_logging_filter_scrubs_the_message(caplog):
    logger = logging.getLogger("test_redaction")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO):
        logger.info("failed with sk-proj-abcdefghijklmnopqrstuvwxyz123456")
    assert "sk-proj" not in caplog.text


def test_the_logging_filter_scrubs_interpolated_arguments(caplog):
    logger = logging.getLogger("test_redaction_args")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO):
        logger.info("token was %s", "ghp_MJxKEwNLcazW33mGPS1r6P3EkBQ3H627Hp")
    assert "ghp_" not in caplog.text


def test_an_error_stored_in_a_report_is_redacted():
    payload = record(error="auth failed for sk-proj-abcdefghijklmnopqrstuvwxyz123456").to_dict()
    assert "sk-proj" not in payload["error"]


def test_a_credential_cannot_reach_the_json_file(tmp_path):
    path = reports.write_json(
        [record(error="key sk-proj-abcdefghijklmnopqrstuvwxyz123456 rejected")],
        tmp_path / "analysis.json",
    )
    assert "sk-proj" not in path.read_text(encoding="utf-8")


def test_a_credential_cannot_reach_the_csv_file(tmp_path):
    path = reports.write_csv(
        [record(error="key sk-proj-abcdefghijklmnopqrstuvwxyz123456 rejected")],
        tmp_path / "analysis.csv",
    )
    assert "sk-proj" not in path.read_text(encoding="utf-8")


# --- JSON -------------------------------------------------------------------


def test_the_json_report_round_trips(tmp_path):
    path = reports.write_json([record()], tmp_path / "analysis.json", summary={"total": 1})
    rows, summary = reports.read_json(path)
    assert len(rows) == 1
    assert rows[0]["filename"] == "P1042675.RW2"
    assert summary["total"] == 1


def test_the_json_report_carries_a_schema_version(tmp_path):
    path = reports.write_json([record()], tmp_path / "analysis.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] >= 1


def test_the_json_report_carries_the_disclaimer(tmp_path):
    path = reports.write_json([record()], tmp_path / "analysis.json")
    assert "not guarantees" in json.loads(path.read_text(encoding="utf-8"))["disclaimer"]


def test_the_json_is_written_atomically(tmp_path):
    """An interrupt mid-write once silently discarded every result so far."""
    path = tmp_path / "analysis.json"
    reports.write_json([record()], path)
    assert not path.with_name(path.name + ".tmp").exists()


def test_every_stored_record_keeps_what_re_routing_needs(tmp_path):
    path = reports.write_json([record()], tmp_path / "analysis.json")
    row = reports.read_json(path)[0][0]
    for key in ("scores", "issues", "route_class", "best_in_cluster", "media_type", "phash"):
        assert key in row


# --- CSV --------------------------------------------------------------------


def test_the_csv_has_one_row_per_asset(tmp_path):
    path = reports.write_csv([record(), record(asset_id="a2", filename="b.RW2")], tmp_path / "a.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["filename"] for r in rows] == ["P1042675.RW2", "b.RW2"]


def test_the_csv_separates_current_quality_from_potential(tmp_path):
    """Two columns, because they are two different questions."""
    path = reports.write_csv([record()], tmp_path / "a.csv")
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["current_quality"] == "44"
    assert row["post_edit_potential"] == "81"
    assert row["expected_gain"] == "37"


def test_the_csv_lists_the_three_problem_classes_separately(tmp_path):
    path = reports.write_csv([record()], tmp_path / "a.csv")
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["fixable"] == "underexposed: mean luma 44"
    assert row["unrecoverable"] == ""


# --- summary ----------------------------------------------------------------


def test_the_summary_counts_each_class():
    summary = reports.summarise(
        [record(), record(asset_id="a2", route_class="trash"), record(asset_id="a3", route_class="trash")]
    )
    assert summary["by_class"]["trash"] == 2
    assert summary["by_class"]["stock_strong"] == 1


def test_the_summary_separates_photos_from_videos():
    summary = reports.summarise([record(), record(asset_id="a2", media_type="video")])
    assert summary["photos"] == 1 and summary["videos"] == 1


def test_the_summary_counts_failures_and_low_confidence():
    summary = reports.summarise(
        [record(asset_id="a1", status="error"), record(asset_id="a2", confidence=30)]
    )
    assert summary["failed"] == 1
    assert summary["low_confidence"] == 1


def test_the_summary_reports_recoverable_space():
    summary = reports.summarise([record()], recoverable_bytes=20 * 1_048_576)
    assert summary["recoverable_mb"] == 20.0


def test_the_summary_names_the_strongest_assets():
    summary = reports.summarise([record()])
    assert summary["strongest"][0]["filename"] == "P1042675.RW2"


def test_the_summary_counts_duplicate_clusters():
    summary = reports.summarise(
        [record(cluster_id="c1", cluster_size=3), record(asset_id="a2", cluster_id="c1", cluster_size=3)]
    )
    assert summary["duplicate_clusters"] == 1


def test_an_empty_collection_summarises_without_error():
    assert reports.summarise([])["total"] == 0


# --- HTML -------------------------------------------------------------------


def test_the_html_report_is_written(tmp_path):
    path = reports.write_html([record()], tmp_path / "report.html")
    body = path.read_text(encoding="utf-8")
    assert "P1042675.RW2" in body
    assert "<!doctype html>" in body.lower()


def test_the_html_report_escapes_hostile_filenames(tmp_path):
    body = reports.write_html(
        [record(filename="<script>alert(1)</script>.jpg")], tmp_path / "r.html"
    ).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in body


def test_the_html_report_shows_the_disclaimer(tmp_path):
    body = reports.write_html([record()], tmp_path / "r.html").read_text(encoding="utf-8")
    assert "not guarantees" in body


def test_the_html_report_is_localised(tmp_path):
    body = reports.write_html([record()], tmp_path / "r.html", language="ru").read_text(
        encoding="utf-8"
    )
    assert "Текущее качество" in body


def test_an_empty_html_report_still_renders(tmp_path):
    assert reports.write_html([], tmp_path / "r.html").exists()


# --- localisation -----------------------------------------------------------


def test_english_and_russian_cover_the_same_keys():
    """Adding a string in one language and forgetting the other fails here."""
    assert i18n.missing_keys("ru") == set()


def test_russian_strings_are_actually_russian():
    for key in ("class.trash", "class.flagship", "score.post_edit_potential", "summary.title"):
        assert i18n.t(key, "ru") != i18n.t(key, "en")
        assert any("Ѐ" <= c <= "ӿ" for c in i18n.t(key, "ru"))


def test_every_route_class_has_a_label_in_both_languages():
    for route_class in ("trash", "review", "stock_standard", "stock_strong", "flagship"):
        for language in ("en", "ru"):
            assert i18n.t(f"class.{route_class}", language) != f"class.{route_class}"


def test_every_score_dimension_has_a_label_in_both_languages():
    for dimension in (
        "current_quality", "recoverability", "post_edit_potential", "aesthetic_potential",
        "stock_potential", "portfolio_potential", "uniqueness",
        "confidence", "routing_score",
    ):
        for language in ("en", "ru"):
            assert i18n.t(f"score.{dimension}", language) != f"score.{dimension}"


def test_an_unknown_key_returns_itself_rather_than_ending_the_run():
    """A missing translation should degrade a label, not lose a paid analysis."""
    assert i18n.t("no.such.key") == "no.such.key"


def test_an_unknown_language_falls_back_to_english():
    assert i18n.t("class.trash", "klingon") == i18n.t("class.trash", "en")


@pytest.mark.parametrize(
    ("given", "expected"), [("ru", "ru"), ("ru_RU", "ru"), ("ru-RU", "ru"), ("en-GB", "en"), (None, "en"), ("zz", "en")]
)
def test_locale_tags_are_normalised(given, expected):
    assert i18n.normalise(given) == expected


def test_placeholders_are_filled():
    assert "5" in i18n.t("misc.best_in_cluster", "en", n=5)


def test_a_missing_placeholder_does_not_raise():
    assert i18n.t("misc.best_in_cluster", "en")


def test_the_disclaimer_exists_in_both_languages():
    for language in ("en", "ru"):
        assert len(i18n.t("warn.disclaimer", language)) > 80


def test_the_printed_summary_is_localised():
    summary = reports.summarise([record()])
    assert "СВОДКА" in reports.format_summary(summary, "ru")
    assert "COLLECTION SUMMARY" in reports.format_summary(summary, "en")
