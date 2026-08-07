import pytest
from conftest import CORRUPT_EXIF, EXPECTED_EXIF, NO_EXIF, TRUNCATED_RAW, WITH_EXIF

from exif_reader import (
    EXIF_EMPTY,
    _dms_to_decimal,
    _int_or_none,
    _parse_date,
    _parse_gps,
    _parse_rational,
    _parse_shutter,
    _str_or_none,
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


def test_raw_file_pillow_cannot_open_returns_empty():
    # Pillow has no RW2 decoder, so the RAW branch degrades to empty metadata
    # rather than raising. See the known-limitations note in CONTRIBUTING.md.
    assert extract_exif(TRUNCATED_RAW, "RAW") == EXIF_EMPTY


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
