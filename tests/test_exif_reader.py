from types import SimpleNamespace

import pytest
from conftest import (
    CORRUPT_EXIF,
    EXPECTED_EXIF,
    EXPECTED_RAW_EXIF,
    NO_EXIF,
    RAW_EXIF,
    TRUNCATED_RAW,
    WITH_EXIF,
)

from photoai import exif_reader
from photoai.exif_reader import (
    EXIF_EMPTY,
    _dms_to_decimal,
    _int_or_none,
    _parse_date,
    _parse_gps,
    _parse_rational,
    _parse_shutter,
    _str_or_none,
    _tag,
    extract_exif,
)

# --- parsing a real fixture -------------------------------------------------


@pytest.mark.parametrize(("field", "expected"), sorted(EXPECTED_EXIF.items()))
def test_reads_every_field_from_jpeg(field, expected):
    assert extract_exif(WITH_EXIF, "JPEG")[field] == expected


def test_returns_exactly_the_documented_keys():
    assert set(extract_exif(WITH_EXIF, "JPEG")) == set(EXIF_EMPTY)


# --- graceful degradation ---------------------------------------------------


def test_jpeg_without_exif_returns_all_none():
    assert extract_exif(NO_EXIF, "JPEG") == EXIF_EMPTY


def test_corrupt_exif_segment_does_not_raise():
    assert extract_exif(CORRUPT_EXIF, "JPEG") == EXIF_EMPTY


def test_missing_file_returns_empty_instead_of_raising(tmp_path):
    assert extract_exif(tmp_path / "nope.jpg", "JPEG") == EXIF_EMPTY


def test_non_image_bytes_return_empty(tmp_path):
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"this is not an image")
    assert extract_exif(junk, "JPEG") == EXIF_EMPTY


def test_undecodable_raw_returns_empty_without_raising():
    # Neither exifread nor LibRaw can read it; both failures must be swallowed.
    assert extract_exif(TRUNCATED_RAW, "RAW") == EXIF_EMPTY


def test_missing_raw_file_returns_empty(tmp_path):
    assert extract_exif(tmp_path / "absent.rw2", "RAW") == EXIF_EMPTY


# --- RAW metadata (regression: this returned all-None for every RAW file) ----


@pytest.mark.parametrize(("field", "expected"), sorted(EXPECTED_RAW_EXIF.items()))
def test_reads_every_field_from_raw(field, expected):
    assert extract_exif(RAW_EXIF, "RAW")[field] == expected


def test_raw_metadata_is_not_empty():
    result = extract_exif(RAW_EXIF, "RAW")
    assert result != EXIF_EMPTY
    assert sum(v is not None for v in result.values()) == 8


def test_raw_returns_exactly_the_documented_keys():
    assert set(extract_exif(RAW_EXIF, "RAW")) == set(EXIF_EMPTY)


def test_pillow_is_not_used_for_raw(monkeypatch):
    # Pillow cannot decode RW2 at all; routing RAW through it was the bug.
    def explode(*args, **kwargs):
        raise AssertionError("RAW must not be routed through Pillow")

    monkeypatch.setattr(exif_reader.Image, "open", explode)
    assert extract_exif(RAW_EXIF, "RAW")["camera_make"] == "Panasonic"


def test_gps_tags_are_mapped_from_raw(monkeypatch):
    """This camera does not geotag, so stub exifread to cover the GPS branch."""
    tags = {
        "Image Make": SimpleNamespace(values="Nikon"),
        "GPS GPSLatitude": SimpleNamespace(values=[(33, 1), (52, 1), (0, 1)]),
        "GPS GPSLatitudeRef": SimpleNamespace(values="S"),
        "GPS GPSLongitude": SimpleNamespace(values=[(151, 1), (12, 1), (0, 1)]),
        "GPS GPSLongitudeRef": SimpleNamespace(values="E"),
    }
    monkeypatch.setattr(exif_reader.exifread, "process_file", lambda f, **kw: tags)
    result = extract_exif(RAW_EXIF, "RAW")
    assert (round(result["gps_lat"], 4), round(result["gps_lon"], 4)) == (-33.8667, 151.2)


def test_libraw_fills_fields_exifread_could_not_read(monkeypatch):
    """Covers containers exifread cannot parse, e.g. CR3."""
    monkeypatch.setattr(exif_reader.exifread, "process_file", lambda f, **kw: {})

    def fake_fill(path, result):
        result["iso"] = 3200
        result["lens"] = "RF 50mm F1.2 L USM"

    monkeypatch.setattr(exif_reader, "_fill_from_rawpy", fake_fill)
    result = extract_exif(RAW_EXIF, "RAW")
    assert result["iso"] == 3200
    assert result["lens"] == "RF 50mm F1.2 L USM"


def test_libraw_does_not_overwrite_what_exifread_found(monkeypatch):
    def clobber(path, result):
        result["camera_make"] = "SHOULD NOT WIN"

    monkeypatch.setattr(exif_reader, "_fill_from_rawpy", clobber)
    # exifread supplies camera_make, so the fallback must leave it alone.
    assert extract_exif(RAW_EXIF, "RAW")["camera_make"] == "Panasonic"


def test_libraw_failure_does_not_break_exifread_results(monkeypatch):
    def explode(path, result):
        raise OSError("LibRaw exploded")

    monkeypatch.setattr(exif_reader, "_fill_from_rawpy", explode)
    assert extract_exif(RAW_EXIF, "RAW")["camera_model"] == "DC-S5M2"


# --- the tag unwrapper ------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "names", "expected"),
    [
        ({"A": SimpleNamespace(values="x")}, ("A",), "x"),
        ({"A": SimpleNamespace(values=[7])}, ("A",), 7),
        ({"A": SimpleNamespace(values=[1, 2, 3])}, ("A",), (1, 2, 3)),
        ({"B": SimpleNamespace(values="y")}, ("A", "B"), "y"),
        ({"A": SimpleNamespace(values=[])}, ("A",), None),
        ({}, ("A",), None),
    ],
    ids=["scalar", "single-item-list", "multi-item-list", "falls-through", "empty", "absent"],
)
def test_tag_unwraps_exifread_values(tags, names, expected):
    assert _tag(tags, *names) == expected


def test_result_is_a_copy_not_the_shared_empty_dict():
    result = extract_exif(NO_EXIF, "JPEG")
    result["camera_make"] = "mutated"
    assert EXIF_EMPTY["camera_make"] is None


# --- rational / shutter parsing ---------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ((28, 10), 2.8),
        ((1, 0), None),
        (3.5, 3.5),
        ("nonsense", None),
    ],
)
def test_parse_rational(value, expected):
    assert _parse_rational(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((1, 250), "1/250"),
        ((1, 1), "1.0s"),
        ((5, 2), "2.5s"),
        ((0, 1), None),
        ((-1, 250), None),
        (None, None),
    ],
)
def test_parse_shutter(value, expected):
    assert _parse_shutter(value) == expected


# --- date parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026:03:14 09:26:53", "2026-03-14T09:26:53"),
        ("2026-03-14 09:26:53", "2026-03-14T09:26:53"),
        ("2026/03/14 09:26:53", "2026-03-14T09:26:53"),
        ("  2026:03:14 09:26:53  ", "2026-03-14T09:26:53"),
        ("14 March 2026", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_date(value, expected):
    assert _parse_date(value) == expected


# --- GPS --------------------------------------------------------------------


def test_gps_south_and_west_are_negated():
    lat, lon = _parse_gps(
        {
            "GPSLatitude": ((33, 1), (52, 1), (0, 1)),
            "GPSLatitudeRef": "S",
            "GPSLongitude": ((151, 1), (12, 1), (0, 1)),
            "GPSLongitudeRef": "W",
        }
    )
    assert lat < 0 and lon < 0
    assert (round(lat, 4), round(lon, 4)) == (-33.8667, -151.2)


def test_gps_returns_none_when_a_component_is_missing():
    assert _parse_gps({"GPSLatitude": ((33, 1), (52, 1), (0, 1))}) == (None, None)


def test_dms_to_decimal_rejects_short_tuples():
    assert _dms_to_decimal(((33, 1),)) is None


# --- small helpers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("  Panasonic  ", "Panasonic"), ("   ", None), (42, "42")],
)
def test_str_or_none(value, expected):
    assert _str_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("400", 400), ((800, 800), 800), ([200], 200), ("iso", None)],
)
def test_int_or_none(value, expected):
    assert _int_or_none(value) == expected
