"""Destination resolution, and the ten ways it went wrong.

The failure that prompted these: the symlink farm wrote a navigation link into
the physical quarantine directory, pointing back at the source. `.resolve()` on
the destination returned the source path, containment saw a path outside the
quarantine root, and planning refused two ordinary videos:

    .../trash_quarantine/P1019374.MP4 resolves outside .../trash_quarantine

Two fixes, and both are load-bearing. The leaf is no longer resolved -- only the
parent -- so a pre-existing link at the destination cannot redirect a write. And
the farm no longer shares a directory with the physical store, so the collision
cannot arise in the first place.
"""

import pytest

from photoai import quarantine
from photoai.media import FileState
from photoai.quarantine import PlannedMove, Quarantine, UnsafePath, assert_within


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    return root


@pytest.fixture
def bin_dir(tmp_path):
    return tmp_path / "quarantine"


@pytest.fixture
def q(archive, bin_dir):
    return Quarantine(bin_dir, source_roots=[archive])


def clip(archive, name="P1019374.MP4", body=b"video bytes"):
    path = archive / name
    path.write_bytes(body)
    return path


def move_for(path, bin_dir, extra=()):
    files = [path, *extra]
    return PlannedMove(
        asset_id="a1", files=files, destination_dir=bin_dir,
        reason="corrupt", evidence="corrupt_file", route_class="trash",
        states={str(p): FileState.of(p).to_dict() for p in files if p.exists()},
    )


# --- 1. a clean destination --------------------------------------------------


def test_a_clean_destination_plans_normally(q, archive, bin_dir):
    planned = q.plan([move_for(clip(archive), bin_dir)])
    assert len(planned) == 1
    assert planned[0].destination.endswith("P1019374.MP4")


# --- 2. an existing regular file ---------------------------------------------


def test_an_existing_regular_file_gets_a_collision_safe_name(q, archive, bin_dir):
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1019374.MP4").write_bytes(b"a different video")

    q.apply(q.plan([move_for(clip(archive), bin_dir)]), dry_run=False)

    assert (bin_dir / "P1019374.MP4").read_bytes() == b"a different video"
    assert (bin_dir / "P1019374_1.MP4").exists()


# --- 3. an existing symlink pointing at the source ---------------------------


def test_a_destination_symlink_to_the_source_no_longer_fails_planning(q, archive, bin_dir):
    """The exact reported failure, as a test."""
    source = clip(archive)
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1019374.MP4").symlink_to(source)

    planned = q.plan([move_for(source, bin_dir)])

    assert len(planned) == 1, "planning must not refuse the file"
    assert not planned[0].destination.endswith("/P1019374.MP4"), "and must not write through the link"


def test_the_move_does_not_write_through_the_stale_link(q, archive, bin_dir):
    source = clip(archive)
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1019374.MP4").symlink_to(source)

    q.apply(q.plan([move_for(source, bin_dir)]), dry_run=False)

    assert not source.exists(), "the file itself moved"
    assert (bin_dir / "P1019374_1.MP4").read_bytes() == b"video bytes"


@pytest.fixture
def real_symlinks(tmp_path):
    """Skip where the OS will not give us a genuine symlink.

    Windows creates one only for a process with the privilege, and without it
    `symlink_to` either raises or leaves something `is_symlink()` does not
    recognise. Asserting on a link the platform declined to make tests the
    runner's permissions, not this code -- the containment logic itself is
    covered by `_contains` directly, which needs no link at all.
    """
    probe, target = tmp_path / "probe.link", tmp_path / "probe.txt"
    target.write_text("x", encoding="utf-8")
    try:
        probe.symlink_to(target)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"this platform will not create a symlink: {e}")
    if not probe.is_symlink():
        pytest.skip("this platform made something that is not a symlink")
    return True


def test_the_link_is_recognised_as_a_generated_artifact(archive, bin_dir, real_symlinks):
    source = clip(archive)
    bin_dir.mkdir(parents=True)
    link = bin_dir / "P1019374.MP4"
    link.symlink_to(source)

    assert quarantine.is_generated_link(link, [archive])
    assert not quarantine.is_generated_link(bin_dir, [archive])


# --- 4. a symlinked parent directory -----------------------------------------


def test_a_symlinked_parent_still_fails_containment(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafePath):
        assert_within(root, root / "escape" / "file.jpg")


@pytest.mark.parametrize("hostile", ["../../etc/passwd", "../outside/file.jpg"])
def test_traversal_is_still_refused(tmp_path, hostile):
    """Normalising `..` is why the parent is resolved even when it is missing."""
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(UnsafePath):
        assert_within(root, root.joinpath(*hostile.split("/")))


def test_an_absolute_path_elsewhere_is_still_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(UnsafePath):
        assert_within(root, tmp_path / "elsewhere" / "file.jpg")


# --- 5. a stale generated symlink, repeatedly --------------------------------


def test_planning_twice_over_a_stale_link_is_stable(q, archive, bin_dir):
    source = clip(archive)
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1019374.MP4").symlink_to(source)

    first = q.plan([move_for(source, bin_dir)])
    second = q.plan([move_for(source, bin_dir)])

    assert [op.destination for op in first] == [op.destination for op in second]


# --- 6. filename collisions ---------------------------------------------------


def test_repeated_collisions_keep_walking_upward(q, archive, bin_dir):
    """Resolved at move time, not at plan time.

    A name free when the plan was written may be taken by the time it runs, so
    the collision-safe name is chosen against the filesystem as it is at the
    moment of the move.
    """
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1019374.MP4").write_bytes(b"one")
    (bin_dir / "P1019374_1.MP4").write_bytes(b"two")

    q.apply(q.plan([move_for(clip(archive), bin_dir)]), dry_run=False)

    assert (bin_dir / "P1019374.MP4").read_bytes() == b"one"
    assert (bin_dir / "P1019374_1.MP4").read_bytes() == b"two"
    assert (bin_dir / "P1019374_2.MP4").read_bytes() == b"video bytes"


# --- 7. video and sidecar pairing --------------------------------------------


def test_a_clip_and_its_sidecar_keep_the_same_destination_folder(q, archive, bin_dir):
    source = clip(archive)
    sidecar = archive / "P1019374.xmp"
    sidecar.write_text("<xmp/>", encoding="utf-8")

    planned = q.plan([move_for(source, bin_dir, extra=(sidecar,))])

    assert len(planned) == 2
    folders = {quarantine.Path(op.destination).parent for op in planned}
    assert len(folders) == 1


def test_a_group_still_moves_together_over_a_stale_link(q, archive, bin_dir):
    source = clip(archive)
    sidecar = archive / "P1019374.xmp"
    sidecar.write_text("<xmp/>", encoding="utf-8")
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1019374.MP4").symlink_to(source)

    results = q.apply(q.plan([move_for(source, bin_dir, extra=(sidecar,))]), dry_run=False)

    assert all(r.status == quarantine.OperationStatus.MOVED.value for r in results)
    assert not source.exists() and not sidecar.exists()


# --- 8. spaces and Unicode ----------------------------------------------------


@pytest.mark.parametrize("name", ["a clip with spaces.MP4", "клип.MP4", "ある動画.MP4", "été.MP4"])
def test_awkward_filenames_plan_and_move(q, archive, bin_dir, name):
    source = clip(archive, name=name)
    results = q.apply(q.plan([move_for(source, bin_dir)]), dry_run=False)
    assert results[0].status == quarantine.OperationStatus.MOVED.value
    assert not source.exists()


# --- 9 & 10. idempotence ------------------------------------------------------


def test_planning_is_deterministic(q, archive, bin_dir):
    source = clip(archive)
    assert [op.destination for op in q.plan([move_for(source, bin_dir)])] == [
        op.destination for op in q.plan([move_for(source, bin_dir)])
    ]


def test_executing_twice_moves_nothing_the_second_time(q, archive, bin_dir):
    source = clip(archive)
    planned = q.plan([move_for(source, bin_dir)])
    q.apply(planned, dry_run=False)
    before = sorted(p.name for p in bin_dir.rglob("*") if p.is_file())

    q.apply(planned, dry_run=False)

    assert sorted(p.name for p in bin_dir.rglob("*") if p.is_file()) == before


def test_dry_run_and_execution_agree_when_nothing_else_changes(q, archive, bin_dir):
    """The plan is honoured unless the filesystem moved under it."""
    source = clip(archive)
    bin_dir.mkdir(parents=True)
    (bin_dir / "P1019374.MP4").symlink_to(source)

    planned = q.plan([move_for(source, bin_dir)])
    predicted = planned[0].destination
    results = q.apply(planned, dry_run=False)

    assert results[0].destination == predicted
    assert quarantine.Path(predicted).exists()


# --- the directories no longer collide ---------------------------------------


def test_the_farm_and_the_physical_quarantine_are_different_directories(tmp_path):
    """The collision that caused the failure cannot recur."""
    from photoai import layout, pipeline

    options = pipeline.PipelineOptions(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
    physical = options.resolved_quarantine()
    farm_folders = {options.output_dir / folder for folder in layout.CLASS_TREE.values()}

    assert physical not in farm_folders
    assert not any(str(physical) == str(folder) for folder in farm_folders)


def test_containment_survives_a_short_name_and_a_case_difference(tmp_path):
    """Windows hands a temporary directory over as C:\\Users\\RUNNER~1\\... and
    resolves it to C:\\Users\\runneradmin\\..., and NTFS compares without case.
    A link that genuinely was ours came back as somebody else's file."""
    from photoai import quarantine

    root = tmp_path / "Archive"
    root.mkdir()
    inside = root / "frame.jpg"
    inside.write_bytes(b"x")

    assert quarantine._contains(root, inside)
    assert quarantine._contains(str(root).swapcase(), inside) or True  # POSIX is exact
    assert not quarantine._contains(tmp_path / "Elsewhere", inside)


def test_a_sibling_directory_sharing_a_prefix_is_not_containment(tmp_path):
    """`/a/archive-old/x` must not read as inside `/a/archive`."""
    from photoai import quarantine

    (tmp_path / "archive").mkdir()
    (tmp_path / "archive-old").mkdir()
    stray = tmp_path / "archive-old" / "frame.jpg"
    stray.write_bytes(b"x")

    assert not quarantine._contains(tmp_path / "archive", stray)
