import csv
import json

import pytest

from output_writer import CSV_FIELDNAMES, OutputWriter


@pytest.fixture
def writer(tmp_path):
    return OutputWriter(tmp_path / "results")


def read_csv(writer):
    with open(writer.csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(writer):
    return json.loads(writer.json_path.read_text(encoding="utf-8"))


# --- directory setup --------------------------------------------------------


def test_output_and_preview_directories_are_created(tmp_path):
    w = OutputWriter(tmp_path / "a" / "b" / "results")
    assert w.output_dir.is_dir()
    assert w.previews_dir.is_dir()
    assert w.get_previews_dir() == w.output_dir / "previews"


def test_constructing_twice_over_the_same_directory_is_safe(tmp_path):
    OutputWriter(tmp_path / "results")
    OutputWriter(tmp_path / "results")  # must not raise


# --- CSV schema -------------------------------------------------------------


def test_csv_header_matches_the_declared_fieldnames(writer, sample_record):
    writer.append_record(sample_record)
    with open(writer.csv_path, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == CSV_FIELDNAMES


def test_csv_row_carries_every_declared_field(writer, sample_record):
    writer.append_record(sample_record)
    assert set(read_csv(writer)[0]) == set(CSV_FIELDNAMES)


def test_csv_values_round_trip_as_text(writer, sample_record):
    writer.append_record(sample_record)
    row = read_csv(writer)[0]
    assert row["filename"] == "P1042675.RW2"
    assert row["quality_score"] == "762"
    assert row["tags_str"] == "temple; stone; shadow"
    assert row["status"] == "ok"


def test_header_is_written_once_across_appends(writer, sample_record):
    writer.append_record(sample_record)
    writer.append_record({**sample_record, "filename": "second.RW2"})
    assert writer.csv_path.read_text(encoding="utf-8").count("filename,filepath") == 1
    assert len(read_csv(writer)) == 2


def test_keys_outside_the_schema_are_dropped(writer, sample_record):
    writer.append_record({**sample_record, "unexpected_key": "ignored"})
    assert "unexpected_key" not in read_csv(writer)[0]


def test_a_record_missing_optional_keys_still_writes(writer):
    writer.append_record({"filename": "sparse.jpg", "status": "error"})
    row = read_csv(writer)[0]
    assert row["filename"] == "sparse.jpg"
    assert row["camera_make"] == ""


# --- JSON schema ------------------------------------------------------------


def test_json_is_a_list_of_records(writer, sample_record):
    writer.append_record(sample_record)
    data = read_json(writer)
    assert isinstance(data, list) and len(data) == 1
    assert isinstance(data[0], dict)


def test_json_drops_tags_str_but_keeps_the_tags_list(writer, sample_record):
    writer.append_record(sample_record)
    record = read_json(writer)[0]
    assert "tags_str" not in record
    assert record["tags"] == ["temple", "stone", "shadow"]


def test_json_preserves_native_types(writer, sample_record):
    writer.append_record(sample_record)
    record = read_json(writer)[0]
    assert record["quality_score"] == 762
    assert record["aperture"] == 2.8
    assert record["error_message"] is None


def test_json_accumulates_across_appends(writer, sample_record):
    writer.append_record(sample_record)
    writer.append_record({**sample_record, "filename": "second.RW2"})
    assert [r["filename"] for r in read_json(writer)] == ["P1042675.RW2", "second.RW2"]


def test_unreadable_json_is_replaced_rather_than_crashing(writer, sample_record):
    writer.json_path.write_text("{ this is not json", encoding="utf-8")
    writer.append_record(sample_record)
    assert len(read_json(writer)) == 1


def test_a_json_file_holding_a_non_list_is_not_appended_to(writer, sample_record):
    writer.json_path.write_text('{"not": "a list"}', encoding="utf-8")
    writer.append_record(sample_record)
    assert read_json(writer) == [
        {k: v for k, v in sample_record.items() if k != "tags_str"}
    ]


# --- regression: a partial write used to wipe every previous result ----------


def test_corrupt_json_is_preserved_not_silently_discarded(writer, sample_record):
    writer.json_path.write_text('[{"filename": "earlier.RW2"}', encoding="utf-8")  # truncated
    writer.append_record(sample_record)

    salvage = writer.json_path.with_name(writer.json_path.name + ".corrupt")
    assert salvage.exists(), "the unparseable file must be kept, not dropped"
    assert "earlier.RW2" in salvage.read_text(encoding="utf-8")


def test_json_is_never_left_truncated_by_a_failed_write(writer, sample_record, monkeypatch):
    """The old code truncated results.json in place, so an interrupt lost everything."""
    writer.append_record(sample_record)
    writer.append_record({**sample_record, "filename": "second.RW2"})
    good = writer.json_path.read_text(encoding="utf-8")

    real_dump = json.dump

    def die_midway(obj, fp, **kwargs):
        real_dump(obj[:1], fp, **kwargs)
        raise KeyboardInterrupt("killed mid-write")

    monkeypatch.setattr(json, "dump", die_midway)
    with pytest.raises(KeyboardInterrupt):
        writer.append_record({**sample_record, "filename": "third.RW2"})

    # The real file is untouched and still parses.
    assert writer.json_path.read_text(encoding="utf-8") == good
    assert [r["filename"] for r in read_json(writer)] == ["P1042675.RW2", "second.RW2"]


def test_no_temp_file_is_left_behind(writer, sample_record):
    writer.append_record(sample_record)
    leftovers = [p.name for p in writer.output_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


# --- unicode ----------------------------------------------------------------


UNICODE_NAMES = ["Café_Ω.jpg", "Улица-Мира.RW2", "写真_001.jpg", "naïve—dash.jpeg", "🌅_sunrise.jpg"]


@pytest.mark.parametrize("filename", UNICODE_NAMES)
def test_unicode_filenames_round_trip_through_csv(writer, sample_record, filename):
    writer.append_record({**sample_record, "filename": filename})
    assert read_csv(writer)[0]["filename"] == filename


@pytest.mark.parametrize("filename", UNICODE_NAMES)
def test_unicode_filenames_round_trip_through_json(writer, sample_record, filename):
    writer.append_record({**sample_record, "filename": filename})
    assert read_json(writer)[0]["filename"] == filename


def test_json_stores_unicode_literally_not_as_escapes(writer, sample_record):
    writer.append_record({**sample_record, "description": "Улица на рассвете"})
    assert "Улица на рассвете" in writer.json_path.read_text(encoding="utf-8")
    assert "\\u0423" not in writer.json_path.read_text(encoding="utf-8")


def test_unicode_survives_the_processed_filename_lookup(writer, sample_record):
    writer.append_record({**sample_record, "filename": "Улица-Мира.RW2"})
    assert "Улица-Мира.RW2" in OutputWriter(writer.output_dir).load_processed_filenames()


def test_embedded_commas_and_newlines_are_quoted_correctly(writer, sample_record):
    messy = "a, b\nc \"quoted\""
    writer.append_record({**sample_record, "quality_reasoning": messy})
    assert read_csv(writer)[0]["quality_reasoning"] == messy


# --- empty result set -------------------------------------------------------


def test_no_files_are_written_before_the_first_record(writer):
    assert not writer.csv_path.exists()
    assert not writer.json_path.exists()


def test_processed_filenames_is_empty_when_no_csv_exists(writer):
    assert writer.load_processed_filenames() == set()


def test_processed_filenames_is_empty_for_a_header_only_csv(writer):
    writer.csv_path.write_text(",".join(CSV_FIELDNAMES) + "\n", encoding="utf-8")
    assert writer.load_processed_filenames() == set()


def test_is_already_processed_against_an_empty_set(writer):
    assert writer.is_already_processed("anything.jpg", set()) is False


def test_unreadable_csv_yields_an_empty_set_rather_than_raising(writer):
    writer.csv_path.write_bytes(b"\x00\x01\x02 binary garbage")
    assert writer.load_processed_filenames() == set()


# --- resume behaviour -------------------------------------------------------


def test_only_ok_rows_count_as_processed(writer, sample_record):
    writer.append_record(sample_record)
    writer.append_record({**sample_record, "filename": "failed.RW2", "status": "error"})
    writer.append_record({**sample_record, "filename": "skipped.RW2", "status": "skipped"})
    assert OutputWriter(writer.output_dir).load_processed_filenames() == {"P1042675.RW2"}


def test_is_already_processed_reflects_the_loaded_set(writer, sample_record):
    writer.append_record(sample_record)
    processed = OutputWriter(writer.output_dir).load_processed_filenames()
    assert writer.is_already_processed("P1042675.RW2", processed) is True
    assert writer.is_already_processed("other.RW2", processed) is False
