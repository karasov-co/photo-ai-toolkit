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
def _no_api_key(monkeypatch, tmp_path_factory):
    """Make sure nothing picks up a real key, from the shell OR from `.env`.

    Clearing the environment is not enough and never was. `bootstrap` reads the
    project's own `.env` from disk and injects what it finds, so any test that
    asked whether credentials exist -- which is now every test, because the
    vision passes default to on -- loaded the developer's real key and tried to
    spend it. The socket guard caught the call, but a guard catching it is not
    the same as the key never being read.

    So the project root is redirected at a directory that has no `.env` in it,
    for the whole suite. Tests that need to exercise credential loading point it
    somewhere of their own.
    """
    import bootstrap

    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)

    empty = tmp_path_factory.mktemp("no_env")
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", empty)
    monkeypatch.setattr(bootstrap, "PROJECT_ENV", empty / ".env")
    monkeypatch.setattr(bootstrap, "_loaded_from", None)
    # The working directory is the second place `.env` is looked for, and pytest
    # runs from the repository -- so redirecting the project root alone still
    # left the real file one lookup away.
    monkeypatch.chdir(empty)


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


@pytest.fixture
def passing_preflight(monkeypatch):
    """Let a test past the model check without a key or a network.

    The check itself is exercised directly in `test_preflight.py`. Everywhere
    else it is a gate to get through, and stubbing it here keeps that gate from
    being quietly disabled in the code to make tests pass.
    """
    import preflight

    def ok(model, **kwargs):
        result = preflight.PreflightResult(ok=True, model=model)
        result.checks = [
            preflight.Check(name, passed=True, detail="verified")
            for name in ("Authentication", "Model access", "Vision input", "Structured reply")
        ]
        return result

    monkeypatch.setattr(preflight, "run", ok)
    return ok
