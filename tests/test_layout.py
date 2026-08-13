import csv
import subprocess

import pytest
from PIL import Image

from layout import (
    build_contact_sheet,
    build_symlink_farm,
    report_token_spend,
    write_delete_candidates,
    write_manifest,
)
from routing import Assessment, Destination, Genre, Recover, assign_destinations


def make(filename, **kw):
    base = {
        "genre": Genre.LANDSCAPE, "axis_a": 50, "axis_b": 50, "axis_c": 50,
        "recover": Recover.EASY, "faces": False, "brand_mark": False,
    }
    base.update(kw)
    return Assessment(filename=filename, **base)


@pytest.fixture
def archive(tmp_path):
    """Real files on disk, so symlink and delete behaviour can be checked."""
    src = tmp_path / "originals"
    src.mkdir()
    names = ["keep.RW2", "sell.RW2", "face.RW2", "junk.RW2"]
    for n in names:
        Image.new("RGB", (80, 60), (90, 110, 140)).save(src / n.replace(".RW2", ".jpg"))
        (src / n).write_bytes(b"raw bytes " + n.encode())
    return src, {n: src / n for n in names}


@pytest.fixture
def routed(archive):
    _, sources = archive
    return assign_destinations([
        make("sell.RW2", axis_a=100),
        make("face.RW2", axis_a=100, faces=True),
        make("junk.RW2", technically_rejected_for=["out of focus (blur ratio 1.02 < 2.00)"]),
        make("keep.RW2", axis_a=10, axis_b=10, axis_c=10),
    ]), sources


# --- manifest ---------------------------------------------------------------


def test_manifest_has_a_destination_column(routed, tmp_path):
    r, sources = routed
    with open(write_manifest(r, sources, tmp_path / "out"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {row["destination"] for row in rows} <= {d.value for d in Destination}


def test_manifest_records_the_reason_and_the_axes(routed, tmp_path):
    r, sources = routed
    with open(write_manifest(r, sources, tmp_path / "out"), encoding="utf-8") as f:
        rows = {row["filename"]: row for row in csv.DictReader(f)}
    assert rows["face.RW2"]["destination"] == Destination.EDITORIAL.value
    assert "release required" in rows["face.RW2"]["reason"]
    assert rows["sell.RW2"]["axis_a"] == "100"


def test_manifest_keeps_the_models_own_suggestion_for_comparison(tmp_path, archive):
    _, sources = archive
    r = assign_destinations([make("face.RW2", axis_a=100, faces=True,
                                  model_destination="10_stock_commercial")])
    with open(write_manifest(r, sources, tmp_path / "out"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["model_destination"] == "10_stock_commercial"
    assert rows[0]["destination"] == Destination.EDITORIAL.value


# --- symlink farm -----------------------------------------------------------


def test_every_frame_gets_a_symlink_not_a_copy(routed, tmp_path):
    r, sources = routed
    out = tmp_path / "out"
    build_symlink_farm(r, sources, out)
    links = [p for d in Destination for p in (out / d.value).iterdir()]
    assert len(links) == 4
    assert all(p.is_symlink() for p in links)


def test_symlinks_resolve_to_the_originals(routed, tmp_path):
    r, sources = routed
    out = tmp_path / "out"
    build_symlink_farm(r, sources, out)
    link = out / Destination.STOCK_COMMERCIAL.value / "sell.RW2"
    assert link.resolve() == sources["sell.RW2"].resolve()
    assert link.read_bytes() == sources["sell.RW2"].read_bytes()


def test_originals_are_never_moved_or_modified(routed, tmp_path, archive):
    src, sources = archive
    r, _ = routed
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    build_symlink_farm(r, sources, tmp_path / "out")
    write_manifest(r, sources, tmp_path / "out")
    write_delete_candidates(r, sources, tmp_path / "out")
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before


def test_a_rerun_does_not_leave_stale_links(routed, tmp_path):
    r, sources = routed
    out = tmp_path / "out"
    build_symlink_farm(r, sources, out)
    # Re-route the same frame somewhere else and rebuild.
    moved = assign_destinations([make("sell.RW2", axis_a=0, axis_b=0, axis_c=0)])
    build_symlink_farm(moved, sources, out)
    assert not (out / Destination.STOCK_COMMERCIAL.value / "sell.RW2").exists()
    assert (out / Destination.HOLD.value / "sell.RW2").is_symlink()


def test_a_missing_source_is_skipped_not_fatal(tmp_path):
    r = assign_destinations([make("ghost.RW2")])
    counts = build_symlink_farm(r, {}, tmp_path / "out")
    assert sum(counts.values()) == 0


# --- delete candidates ------------------------------------------------------


def test_nothing_is_deleted_by_writing_the_candidates(routed, tmp_path, archive):
    src, sources = archive
    r, _ = routed
    write_delete_candidates(r, sources, tmp_path / "out")
    assert (src / "junk.RW2").exists()


def test_the_listing_names_the_file_and_the_reason(routed, tmp_path):
    r, sources = routed
    result = write_delete_candidates(r, sources, tmp_path / "out")
    text = result["listing"].read_text(encoding="utf-8")
    assert "junk.RW2" in text
    assert "blur ratio" in text
    assert result["count"] == 1


def test_the_script_is_executable_and_valid_shell(routed, tmp_path):
    r, sources = routed
    script = write_delete_candidates(r, sources, tmp_path / "out")["script"]
    assert script.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_the_script_moves_to_trash_rather_than_running_rm(routed, tmp_path):
    """A wrong call here must stay recoverable."""
    r, sources = routed
    body = write_delete_candidates(r, sources, tmp_path / "out")["script"].read_text()
    assert "$HOME/.Trash" in body
    assert "rm " not in body
    assert "rm -" not in body


def test_the_script_quotes_paths_with_spaces(tmp_path):
    src = tmp_path / "my photos"
    src.mkdir()
    target = src / "a frame.RW2"
    target.write_bytes(b"x")
    r = assign_destinations([make("a frame.RW2", technically_rejected_for=["out of focus"])])
    script = write_delete_candidates(r, {"a frame.RW2": target}, tmp_path / "out")["script"]
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert "'a frame.RW2'" in script.read_text() or shlex_ok(script.read_text(), target)


def shlex_ok(body, target):
    return str(target) in body and body.count("'") >= 2


def test_running_the_script_moves_the_file_and_leaves_others_alone(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".Trash").mkdir(parents=True)
    src = tmp_path / "originals"
    src.mkdir()
    doomed, spared = src / "junk.RW2", src / "keep.RW2"
    doomed.write_bytes(b"junk")
    spared.write_bytes(b"keep")

    r = assign_destinations([
        make("junk.RW2", technically_rejected_for=["out of focus"]),
        make("keep.RW2"),
    ])
    script = write_delete_candidates(
        r, {"junk.RW2": doomed, "keep.RW2": spared}, tmp_path / "out"
    )["script"]

    subprocess.run(["bash", str(script)], capture_output=True, env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
    assert not doomed.exists(), "candidate should have moved"
    assert (home / ".Trash" / "junk.RW2").read_bytes() == b"junk", "must be recoverable"
    assert spared.read_bytes() == b"keep", "non-candidate must be untouched"


def test_an_empty_candidate_list_still_produces_a_runnable_script(tmp_path, archive):
    _, sources = archive
    r = assign_destinations([make("keep.RW2")])
    result = write_delete_candidates(r, sources, tmp_path / "out")
    assert result["count"] == 0
    assert subprocess.run(["bash", "-n", str(result["script"])], capture_output=True).returncode == 0


# --- contact sheet ----------------------------------------------------------


def test_contact_sheet_shows_every_candidate(tmp_path):
    previews = []
    for i in range(7):
        p = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (200, 150), (30 * i, 80, 120)).save(p)
        previews.append((f"frame{i}.RW2", p))
    sheet = build_contact_sheet(previews, tmp_path / "sheet.jpg", columns=4)
    assert sheet.exists()
    with Image.open(sheet) as img:
        assert img.width > 0 and img.height > 0


def test_no_sheet_is_written_when_there_is_nothing_to_review(tmp_path):
    assert build_contact_sheet([], tmp_path / "sheet.jpg") is None
    assert not (tmp_path / "sheet.jpg").exists()


def test_an_unreadable_preview_does_not_break_the_sheet(tmp_path):
    good = tmp_path / "good.jpg"
    Image.new("RGB", (120, 90), (10, 20, 30)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert build_contact_sheet([("ok.RW2", good), ("broken.RW2", bad)], tmp_path / "s.jpg")


# --- token reporting --------------------------------------------------------


def test_token_report_totals_every_stage():
    usage = {"stage1 luna": {"input_tokens": 1000, "output_tokens": 40},
             "stage2 sol": {"input_tokens": 8000, "output_tokens": 900}}
    out = report_token_spend(usage, frames=100)
    assert "9,000" in out
    assert "940" in out
    assert "per frame" in out


def test_token_report_survives_zero_frames():
    assert "TOKEN SPEND" in report_token_spend({}, frames=0)
