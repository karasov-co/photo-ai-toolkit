"""The two commands a stranger runs first, and the privacy check before sharing.

`doctor` exists so the first failure costs nothing: no key, no FFmpeg, no LibRaw
wheel, nowhere to write, no space. Finding any of those on frame 1,700 of 2,000
is the difference between a bug report and a deleted repository. It never calls
a paid endpoint.

`demo` exists because nobody points a tool at their own archive on a promise.
"""

from __future__ import annotations

import json
import pathlib
import re

from photoai import doctor


def test_doctor_never_calls_a_paid_endpoint(monkeypatch):
    """The whole point is that it is free. A preflight belongs to `analyze`."""

    def explode(*args, **kwargs):
        raise AssertionError("doctor must not build a client")

    from photoai import bootstrap

    monkeypatch.setattr(bootstrap, "make_client", explode)
    report = doctor.run()
    assert report.checks


def test_doctor_reports_a_missing_key_without_failing(monkeypatch):
    """No key is not an error: the local pass runs and writes a full report."""
    from photoai import bootstrap

    monkeypatch.setattr(bootstrap, "api_key", lambda: None)
    key = next(c for c in doctor.run().checks if c.name == "API key")
    assert not key.ok
    assert not key.fatal
    assert "local pass" in key.fix


def test_doctor_fails_on_a_folder_it_cannot_write(tmp_path):
    target = tmp_path / "file-not-a-folder"
    target.write_text("x")
    report = doctor.run(output_dir=target)
    assert not report.ok


def test_doctor_says_what_to_do_about_each_problem(tmp_path):
    for check in doctor.run(input_dir=tmp_path, output_dir=tmp_path).checks:
        if not check.ok:
            assert check.fix, f"{check.name} failed with no advice"


def test_doctor_writes_a_dump_for_a_bug_report(tmp_path):
    payload = doctor.run(output_dir=tmp_path).to_dict()
    assert json.dumps(payload)
    assert set(payload) == {"ok", "checks"}


# --- the demo -----------------------------------------------------------------

DEMO = pathlib.Path(__file__).resolve().parent.parent / "demo" / "photos"


def test_the_demo_frames_are_committed():
    assert DEMO.is_dir()
    frames = sorted(DEMO.glob("*.jpg"))
    assert len(frames) >= 10


def test_the_demo_frames_clear_the_resolution_gate():
    """The first version was 0.54 MP and every frame landed in 'unusable at any
    size', which is a poor first thing for a stranger to see."""
    from PIL import Image

    from photoai import issues

    for frame in sorted(DEMO.glob("*.jpg")):
        with Image.open(frame) as image:
            megapixels = image.width * image.height / 1_000_000
        assert megapixels >= issues.MIN_MEGAPIXELS_ANY, frame.name


def test_the_demo_carries_nobody_s_location():
    """A committed photograph with GPS in it is a coordinate in a public repo."""
    from photoai import exif_reader

    for frame in sorted(DEMO.glob("*.jpg")):
        data = exif_reader.extract_exif(frame, "PHOTO") or {}
        assert not data.get("gps_lat"), frame.name
        assert not data.get("gps_lon"), frame.name


# --- and what a shared report gives away --------------------------------------


def test_no_coordinate_reaches_a_shared_artefact(tmp_path):
    """Somebody shares a report and hands out the coordinates of their home."""
    from photoai import reports

    record = reports.AssetRecord(
        asset_id="a", source_path="/gone/a.jpg", filename="a.jpg",
        media_type="photo", checksum="c", category="TOP", final_score=90,
        status="ok",
        stock_metadata={"location": "", "keywords": []},
    )
    payload = tmp_path / "analysis.json"
    reports.write_json([record], payload)
    text = payload.read_text()
    assert "gps_lat" not in text
    assert "gps_lon" not in text
    assert not re.search(r"\b\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}\b", text)
