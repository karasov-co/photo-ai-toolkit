"""Shared fixtures, plus guards that keep the suite hermetic.

Nothing in tests/ may open a socket or read a real API key. The two autouse
fixtures below enforce that rather than trusting each test to behave.
"""

import socket
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

WITH_EXIF = FIXTURES / "sample_with_exif.jpg"
NO_EXIF = FIXTURES / "sample_no_exif.jpg"
CORRUPT_EXIF = FIXTURES / "corrupt_exif.jpg"
TRUNCATED_RAW = FIXTURES / "truncated.rw2"
RAW_EXIF = FIXTURES / "raw_header.rw2"

# What raw_header.rw2 carries. This camera does not geotag, so GPS stays None;
# the GPS mapping is covered separately with a stubbed exifread.
EXPECTED_RAW_EXIF = {
    "camera_make": "Panasonic",
    "camera_model": "DC-S5M2",
    "lens": "LUMIX S 20-60/F3.5-5.6",
    "iso": 640,
    "shutter_speed": "1/60",
    "aperture": 11.0,
    "focal_length": 33.0,
    "date_shot": "2026-03-16T15:44:16",
    "gps_lat": None,
    "gps_lon": None,
}

# What sample_with_exif.jpg was generated with. See tests/generate_fixtures.py.
EXPECTED_EXIF = {
    "camera_make": "Panasonic",
    "camera_model": "DC-S5M2",
    "lens": "LUMIX S 35mm F1.8",
    "iso": 400,
    "shutter_speed": "1/250",
    "aperture": 2.8,
    "focal_length": 35.0,
    "date_shot": "2026-03-14T09:26:53",
    "gps_lat": 41.38835,
    "gps_lon": 2.173983,
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Turn any outbound connection into a loud failure."""

    def deny(*args, **kwargs):
        raise RuntimeError("network access is not allowed in the test suite")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Make sure nothing picks up a real key from the developer's shell."""
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sample_record():
    """One fully-populated record, shaped the way main.process_photo builds it."""
    return {
        "filename": "P1042675.RW2",
        "filepath": "/photos/P1042675.RW2",
        "file_type": "RAW",
        "camera_make": "Panasonic",
        "camera_model": "DC-S5M2",
        "lens": "LUMIX S 35mm F1.8",
        "iso": 400,
        "shutter_speed": "1/250",
        "aperture": 2.8,
        "focal_length": 35.0,
        "date_shot": "2026-03-14T09:26:53",
        "gps_lat": 41.38835,
        "gps_lon": 2.173983,
        "description": "A stone relief lit by a narrow shaft of light.",
        "tags": ["temple", "stone", "shadow"],
        "tags_str": "temple; stone; shadow",
        "quality_score": 762,
        "quality_reasoning": "Strong atmosphere and directional light.",
        "preview_path": "/results/previews/P1042675.jpg",
        "status": "ok",
        "error_message": None,
    }
