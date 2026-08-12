"""Moving files reversibly, and refusing everything else.

This is the module where a bug loses somebody's photographs, so the tests lean
towards asserting what must *not* happen.
"""

import json

import pytest

from media import FileState
from quarantine import (
    PURGE_CONFIRMATION,
    Lock,
    Manifest,
    OperationLocked,
    OperationStatus,
    PlannedMove,
    Quarantine,
    UnsafePath,
    _next_free_name,
    assert_within,
    summarise_plan,
)


@pytest.fixture
def archive(tmp_path):
    source = tmp_path / "archive"
    source.mkdir()
    return source


@pytest.fixture
def bin_dir(tmp_path):
    return tmp_path / "quarantine"


@pytest.fixture
def quarantine(archive, bin_dir):
    return Quarantine(bin_dir, source_roots=[archive])


def photo(archive, name="P1042675.RW2", body=b"raw bytes", subdir=""):
    folder = archive / subdir if subdir else archive
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(body)
    return path


def move_for(path, bin_dir, reason="test", extra=(), evidence="", states=None):
    files = [path, *extra]
    return PlannedMove(
        asset_id="asset1",
        files=files,
        destination_dir=bin_dir,
        reason=reason,
        route_class="trash",
        evidence=evidence,
        states=states if states is not None else {
            str(p): FileState.of(p).to_dict() for p in files if p.exists()
        },
    )


# --- path safety ------------------------------------------------------------


def test_a_path_inside_the_root_is_allowed(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert assert_within(root, root / "file.jpg")


def test_traversal_out_of_the_root_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(UnsafePath):
        assert_within(root, root / ".." / ".." / "etc" / "passwd")


def test_an_absolute_path_elsewhere_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(UnsafePath):
        assert_within(root, tmp_path / "somewhere_else" / "file.jpg")


def test_a_symlink_pointing_outside_the_root_is_refused(tmp_path):
    """Checked after resolve(), so a link and a `..` are the same rejected case."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.jpg").write_bytes(b"x")
    (root / "link.jpg").symlink_to(outside / "target.jpg")
    with pytest.raises(UnsafePath):
        assert_within(root, root / "link.jpg")


def test_a_source_outside_every_configured_root_is_not_planned(tmp_path, archive, bin_dir):
    stranger = tmp_path / "elsewhere" / "photo.jpg"
    stranger.parent.mkdir(parents=True)
    stranger.write_bytes(b"x")
    quarantine = Quarantine(bin_dir, source_roots=[archive])
    assert quarantine.plan([move_for(stranger, bin_dir)]) == []


# --- dry run ----------------------------------------------------------------


def test_planning_alone_moves_nothing(quarantine, archive, bin_dir):
    path = photo(archive)
    quarantine.plan([move_for(path, bin_dir)])
    assert path.exists()
    assert not bin_dir.exists() or not any(bin_dir.rglob("*.RW2"))


def test_apply_defaults_to_a_dry_run(quarantine, archive, bin_dir):
    """The default must never be the destructive one."""
    path = photo(archive)
    planned = quarantine.plan([move_for(path, bin_dir)])
    quarantine.apply(planned)
    assert path.exists()


def test_a_dry_run_still_reports_what_it_would_do(quarantine, archive, bin_dir):
    planned = quarantine.plan([move_for(photo(archive), bin_dir, reason="out of focus")])
    text = summarise_plan(planned)
    assert "out of focus" in text
    assert "Nothing has been moved" in text


def test_an_empty_plan_says_so():
    assert summarise_plan([]) == "Nothing to move."


# --- moving -----------------------------------------------------------------


def test_applying_moves_the_file(quarantine, archive, bin_dir):
    path = photo(archive)
    results = quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)
    assert not path.exists()
    assert results[0].status == OperationStatus.MOVED.value
    assert (bin_dir / "P1042675.RW2").exists()


def test_the_original_folder_structure_is_preserved(quarantine, archive, bin_dir):
    """A restore must be possible even if the manifest is lost."""
    path = photo(archive, subdir="2026/march")
    quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)
    assert (bin_dir / "2026" / "march" / "P1042675.RW2").exists()


def test_a_raw_moves_together_with_its_sidecar(quarantine, archive, bin_dir):
    """Quarantining the RAW alone leaves an XMP no software can interpret."""
    raw = photo(archive, "P1042675.RW2")
    sidecar = photo(archive, "P1042675.xmp", b"<xmp/>")
    quarantine.apply(
        quarantine.plan([move_for(raw, bin_dir, extra=(sidecar,))]), dry_run=False
    )
    assert not raw.exists() and not sidecar.exists()
    assert (bin_dir / "P1042675.RW2").exists()
    assert (bin_dir / "P1042675.xmp").exists()


def test_a_symlink_is_never_followed_into_a_move(quarantine, archive, bin_dir):
    """Following it would relocate the original while believing it moved a link."""
    real = photo(archive, "real.RW2")
    link = archive / "link.RW2"
    link.symlink_to(real)
    results = quarantine.apply(quarantine.plan([move_for(link, bin_dir)]), dry_run=False)
    assert results[0].status == OperationStatus.SKIPPED.value
    assert real.exists()


# --- collisions -------------------------------------------------------------


def test_a_different_file_with_the_same_name_does_not_overwrite(quarantine, archive, bin_dir):
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1042675.RW2").write_bytes(b"a different photograph")
    path = photo(archive, body=b"the new one")

    quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)

    assert (bin_dir / "P1042675.RW2").read_bytes() == b"a different photograph"
    assert (bin_dir / "P1042675_1.RW2").read_bytes() == b"the new one"


def test_collision_naming_walks_upward(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "a_1.jpg").write_bytes(b"")
    assert _next_free_name(tmp_path / "a.jpg").name == "a_2.jpg"


# --- idempotency ------------------------------------------------------------


def test_an_identical_file_already_quarantined_is_skipped_not_duplicated(
    quarantine, archive, bin_dir
):
    """Re-running an interrupted operation is normal, not an error."""
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1042675.RW2").write_bytes(b"raw bytes")
    path = photo(archive, body=b"raw bytes")

    results = quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)

    assert results[0].status == OperationStatus.SKIPPED.value
    assert not list(bin_dir.glob("*_1.RW2"))


def test_replaying_the_whole_operation_changes_nothing(quarantine, archive, bin_dir):
    path = photo(archive)
    planned = quarantine.plan([move_for(path, bin_dir)])
    quarantine.apply(planned, dry_run=False)
    before = sorted(p.name for p in bin_dir.rglob("*") if p.is_file())
    quarantine.apply(planned, dry_run=False)
    assert sorted(p.name for p in bin_dir.rglob("*") if p.is_file()) == before


def test_a_missing_source_is_skipped_rather_than_failing_the_group(
    quarantine, archive, bin_dir
):
    """A resumed run must be a no-op, not a failure that rolls back good work."""
    path = photo(archive)
    planned = quarantine.plan([move_for(path, bin_dir)])
    path.unlink()
    results = quarantine.apply(planned, dry_run=False)
    assert results[0].status == OperationStatus.SKIPPED.value


# --- restore ----------------------------------------------------------------


def test_a_quarantined_file_can_be_restored_exactly_where_it_was(
    quarantine, archive, bin_dir
):
    path = photo(archive, subdir="2026/march")
    quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)
    assert not path.exists()

    quarantine.restore(dry_run=False)

    assert path.exists()
    assert path.read_bytes() == b"raw bytes"


def test_restore_defaults_to_a_dry_run(quarantine, archive, bin_dir):
    path = photo(archive)
    quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)
    quarantine.restore()
    assert not path.exists()


def test_restore_does_not_overwrite_a_file_that_reappeared(quarantine, archive, bin_dir):
    path = photo(archive)
    quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)
    path.write_bytes(b"something the user put back by hand")

    results = quarantine.restore(dry_run=False)

    assert results[0].status == OperationStatus.SKIPPED.value
    assert path.read_bytes() == b"something the user put back by hand"


def test_restoring_twice_is_harmless(quarantine, archive, bin_dir):
    path = photo(archive)
    quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)
    quarantine.restore(dry_run=False)
    second = quarantine.restore(dry_run=False)
    assert all(r.status != OperationStatus.RESTORED.value for r in second)


def test_a_single_operation_can_be_restored_by_id(quarantine, archive, bin_dir):
    first = quarantine.plan([move_for(photo(archive, "a.RW2"), bin_dir)])
    second = quarantine.plan([move_for(photo(archive, "b.RW2"), bin_dir)])
    quarantine.apply(first, dry_run=False)
    quarantine.apply(second, dry_run=False)

    quarantine.restore(first[0].op_id, dry_run=False)

    assert (archive / "a.RW2").exists()
    assert not (archive / "b.RW2").exists()


# --- the manifest -----------------------------------------------------------


def test_every_move_is_recorded_with_what_it_needs_to_be_undone(
    quarantine, archive, bin_dir
):
    path = photo(archive)
    quarantine.apply(quarantine.plan([move_for(path, bin_dir, reason="blurred")]), dry_run=False)

    rows = quarantine.manifest.load()
    assert len(rows) == 1
    row = rows[0]
    assert row.source == str(path)
    assert row.checksum
    assert row.reason == "blurred"
    assert row.timestamp
    assert row.op_id


def test_the_manifest_is_append_only_so_a_crash_cannot_lose_history(tmp_path):
    """A rewritten document interrupted halfway loses every earlier restore path."""
    manifest = Manifest(tmp_path / "m.jsonl")
    from quarantine import FileOperation

    for i in range(3):
        manifest.append(
            FileOperation(
                op_id="op", asset_id=f"a{i}", source=f"/s/{i}", destination=f"/d/{i}",
                checksum="c", size_bytes=1, timestamp="t", reason="r",
            )
        )
    assert len(manifest.path.read_text().strip().split("\n")) == 3
    assert len(manifest.load()) == 3


def test_a_corrupt_manifest_line_is_skipped_rather_than_failing_the_load(tmp_path):
    manifest = Manifest(tmp_path / "m.jsonl")
    manifest.path.write_text(
        "{not json\n"
        + json.dumps(
            {
                "op_id": "o", "asset_id": "a", "source": "/s", "destination": "/d",
                "checksum": "c", "size_bytes": 1, "timestamp": "t", "reason": "r",
            }
        )
        + "\n"
    )
    assert len(manifest.load()) == 1


def test_recoverable_space_is_reported(quarantine, archive, bin_dir):
    quarantine.apply(
        quarantine.plan([move_for(photo(archive, body=b"x" * 500), bin_dir)]), dry_run=False
    )
    assert quarantine.recoverable_bytes() == 500


# --- purge ------------------------------------------------------------------


def test_purge_refuses_without_the_exact_phrase(quarantine):
    with pytest.raises(ValueError, match="confirmation"):
        quarantine.purge(confirmation="yes", dry_run=False)


def test_purge_refuses_an_empty_confirmation(quarantine):
    with pytest.raises(ValueError):
        quarantine.purge(confirmation="", dry_run=False)


def test_purge_defaults_to_a_dry_run(quarantine, archive, bin_dir):
    path = photo(archive)
    quarantine.apply(quarantine.plan([move_for(path, bin_dir)]), dry_run=False)
    report = quarantine.purge(confirmation=PURGE_CONFIRMATION, older_than_days=0)
    assert report["dry_run"]
    assert (bin_dir / "P1042675.RW2").exists()


def test_purge_will_not_touch_a_recently_quarantined_file(quarantine, archive, bin_dir):
    """The age requirement is the window in which a mistake can be undone."""
    quarantine.apply(quarantine.plan([move_for(photo(archive), bin_dir)]), dry_run=False)
    report = quarantine.purge(confirmation=PURGE_CONFIRMATION, older_than_days=30, dry_run=False)
    assert report["purged"] == 0
    assert (bin_dir / "P1042675.RW2").exists()


def test_purge_removes_only_what_the_manifest_records(quarantine, archive, bin_dir):
    """A file the user dropped into the folder by hand is not ours to delete."""
    quarantine.apply(
        quarantine.plan([move_for(photo(archive), bin_dir, evidence="corrupt_file")]),
        dry_run=False,
    )
    stranger = bin_dir / "not_ours.jpg"
    stranger.write_bytes(b"someone else's file")

    quarantine.purge(confirmation=PURGE_CONFIRMATION, older_than_days=0, dry_run=False)

    assert stranger.exists()
    assert not (bin_dir / "P1042675.RW2").exists()


def test_purge_refuses_while_another_operation_is_running(quarantine, bin_dir):
    quarantine.lock.acquire()
    try:
        with pytest.raises(OperationLocked):
            quarantine.purge(confirmation=PURGE_CONFIRMATION, dry_run=False)
    finally:
        quarantine.lock.release()


# --- locking ----------------------------------------------------------------


def test_a_second_operation_cannot_start_while_one_is_held(tmp_path):
    lock = Lock(tmp_path / ".lock")
    with lock, pytest.raises(OperationLocked):
        Lock(tmp_path / ".lock").acquire()


def test_a_stale_lock_does_not_wedge_the_directory_forever(tmp_path):
    path = tmp_path / ".lock"
    path.write_text(json.dumps({"pid": 1, "at": 0}))
    assert not Lock(path).is_held()


def test_the_lock_is_released_on_the_way_out(tmp_path):
    lock = Lock(tmp_path / ".lock")
    with lock:
        pass
    assert not lock.is_held()
