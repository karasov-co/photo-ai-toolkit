"""The vertical slice, run end to end on generated files.

No network, no API key, no ffmpeg required: the default pipeline is entirely
local and deterministic, which is the property these tests exist to hold onto.
A culling tool that cannot run without a network and a paid key is not a culling
tool.
"""

import json
from concurrent.futures import ProcessPoolExecutor

import pytest
from synthetic import blurred, dark_but_recoverable, near_black, photo_like, write_jpeg

import cli
import pipeline
from calibration import CalibrationSet, portfolio_first_profile
from pipeline import AnalysisCache, Measurement, PipelineOptions
from scoring import RouteClass


@pytest.fixture
def archive(tmp_path):
    """A small collection with a known composition."""
    root = tmp_path / "archive"
    write_jpeg(photo_like(1600, 1200, seed=1), root / "good_one.jpg")
    write_jpeg(photo_like(1600, 1200, seed=2), root / "good_two.jpg")
    write_jpeg(dark_but_recoverable(seed=3, size=(1600, 1200)), root / "dark.jpg")
    write_jpeg(blurred(seed=4, size=(1600, 1200)), root / "soft.jpg")
    write_jpeg(near_black(1600, 1200), root / "lens_cap.jpg")
    return root


@pytest.fixture
def options(archive, tmp_path):
    return PipelineOptions(input_dir=archive, output_dir=tmp_path / "out")


def run(options, **kwargs):
    return pipeline.run(options, **kwargs)


def by_name(result, name):
    return next(r for r in result.records if r.filename == name)


# --- the run ----------------------------------------------------------------


def test_every_asset_is_analysed(options):
    assert len(run(options).records) == 5


def test_the_run_works_offline_with_no_api_key(options):
    """conftest denies sockets and removes the key, so reaching either fails."""
    result = run(options)
    assert all(r.status == "ok" for r in result.records)


def test_every_record_carries_all_ten_dimensions(options):
    for record in run(options).records:
        assert len(record.scores) == 10
        assert all(0 <= v <= 100 for v in record.scores.values())


def test_a_dark_frame_is_rated_higher_on_potential_than_on_current_quality(options):
    """The headline behaviour, asserted through the whole pipeline."""
    dark = by_name(run(options), "dark.jpg")
    assert dark.scores["post_edit_potential"] > dark.scores["current_quality"]
    assert dark.expected_gain > 0


def test_a_blurred_frame_is_not_rescued_by_the_edit_search(options):
    soft = by_name(run(options), "soft.jpg")
    assert soft.issues["unrecoverable"]
    assert soft.scores["post_edit_potential"] < 30


def test_an_empty_frame_is_trash(options):
    assert by_name(run(options), "lens_cap.jpg").route_class == RouteClass.TRASH.value


def test_a_good_frame_is_not_trash(options):
    assert by_name(run(options), "good_one.jpg").route_class != RouteClass.TRASH.value


def test_every_record_explains_itself(options):
    for record in run(options).records:
        assert record.reasons
        assert isinstance(record.issues, dict)
        assert set(record.issues) == {"fixable", "partially_fixable", "unrecoverable"}


def test_every_record_carries_a_proposed_action(options):
    for record in run(options).records:
        assert record.proposed_action


def test_previews_are_generated_for_every_frame(options):
    for record in run(options).records:
        assert record.preview_path
        assert pipeline.Path(record.preview_path).exists()


def test_the_run_records_which_analyzer_and_calibration_produced_it(options):
    record = run(options).records[0]
    assert record.analyzer_version
    assert "default-photo" in record.calibration


def test_a_limit_is_respected(options):
    options.limit = 2
    assert len(run(options).records) == 2


def test_an_empty_directory_produces_an_empty_run(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run(PipelineOptions(input_dir=empty, output_dir=tmp_path / "out"))
    assert result.records == []
    assert result.summary["total"] == 0


# --- progress and cancellation ----------------------------------------------


def test_progress_is_reported_as_the_run_goes(options):
    """Results have to appear while the run is still going."""
    seen = []
    run(options, progress=lambda name, i, total, reused=False: seen.append((name, i, total, reused)))
    assert len(seen) == 5
    assert seen[0][1] == 1 and seen[-1][2] == 5
    # Nothing was in store on a first run, so nothing is reported as reused.
    assert not any(row[3] for row in seen)


def test_a_run_can_be_cancelled_partway(options):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    result = run(options, should_cancel=cancel)
    assert result.cancelled
    assert len(result.records) < 5


def test_a_cancelled_run_still_reports_what_it_finished(options):
    result = run(options, should_cancel=lambda: True)
    assert result.cancelled
    assert result.summary["total"] == 0


# --- caching ----------------------------------------------------------------


def test_the_second_run_reuses_the_cached_measurements(options, monkeypatch):
    run(options)

    calls = {"n": 0}
    original = pipeline.measure_photo

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "measure_photo", counted)
    run(options)
    assert calls["n"] == 0


def test_force_ignores_the_cache(options, monkeypatch):
    run(options)
    calls = {"n": 0}
    original = pipeline.measure_photo

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "measure_photo", counted)
    options.force = True
    run(options)
    assert calls["n"] == 5


def test_a_new_analyzer_version_invalidates_the_cache(tmp_path):
    """Without this, a code change silently keeps serving the old results."""
    path = tmp_path / "cache.json"
    old = AnalysisCache(path, version="1.0.0")
    old.put("abc", Measurement(quality=50.0).to_dict())
    old.save()

    assert AnalysisCache(path, version="1.0.0").get("abc") is not None
    assert AnalysisCache(path, version="2.0.0").get("abc") is None


def test_the_cache_key_includes_the_checksum(tmp_path):
    cache = AnalysisCache(tmp_path / "c.json", version="1.0.0")
    cache.put("checksum-a", Measurement(quality=10.0).to_dict())
    assert cache.get("checksum-b") is None


def test_a_corrupt_cache_does_not_fail_the_run(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    assert AnalysisCache(path).get("anything") is None


def test_a_measurement_round_trips_through_the_cache(tmp_path):
    measurement = Measurement(quality=61.5, uplift=12.0, phash="ff00", channel_means=(1.0, 2.0, 3.0))
    restored = Measurement.from_dict(json.loads(json.dumps(measurement.to_dict())))
    assert restored.quality == 61.5
    assert restored.channel_means == (1.0, 2.0, 3.0)


# --- clustering through the pipeline ----------------------------------------


def test_a_burst_is_collapsed_to_its_best_frame(tmp_path):
    """One sharp frame and two progressively softer siblings of the same scene."""
    root = tmp_path / "burst"
    write_jpeg(photo_like(1600, 1200, seed=9), root / "burst_a.jpg")
    write_jpeg(blurred(seed=9, radius=2.5, size=(1600, 1200)), root / "burst_b.jpg")
    write_jpeg(blurred(seed=9, radius=4.0, size=(1600, 1200)), root / "burst_c.jpg")

    result = run(PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))

    assert {r.cluster_size for r in result.records} == {3}
    keeper = by_name(result, "burst_a.jpg")
    assert keeper.best_in_cluster
    assert keeper.route_class != RouteClass.TRASH.value

    weaker = [r for r in result.records if not r.best_in_cluster]
    assert weaker
    # A comparison for a person, never a deletion.
    assert all(r.route_class == RouteClass.DUPLICATE_CANDIDATE.value for r in weaker)
    assert all(not r.issues["unrecoverable"] for r in weaker)
    assert all("weaker_duplicate" in "".join(r.issues["partially_fixable"]) for r in weaker)


def test_two_frames_too_close_to_separate_are_both_kept(tmp_path):
    """Inside the margin, which frame is better is a compositional judgement.

    A Laplacian cannot make it, so neither frame is condemned. On a real
    archive this was the difference between proposing to delete a 66 that lost
    to a 69 and only proposing the 38 that lost to a 42.
    """
    root = tmp_path / "tie"
    write_jpeg(photo_like(1600, 1200, seed=11), root / "twin_a.jpg")
    write_jpeg(photo_like(1600, 1200, seed=11), root / "twin_b.jpg")

    result = run(PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))

    assert {r.cluster_size for r in result.records} == {2}
    assert all(r.route_class != RouteClass.TRASH.value for r in result.records)
    assert all(not r.issues["unrecoverable"] for r in result.records)


def test_the_margin_that_condemned_a_duplicate_is_recorded(tmp_path):
    root = tmp_path / "burst"
    write_jpeg(photo_like(1600, 1200, seed=9), root / "sharp.jpg")
    write_jpeg(blurred(seed=9, radius=4.0, size=(1600, 1200)), root / "soft.jpg")

    result = run(PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))
    assert by_name(result, "soft.jpg").cluster_margin > 0


def test_distinct_photographs_are_not_clustered(options):
    result = run(options)
    assert all(r.cluster_size == 1 for r in result.records)


# --- flagship selection -----------------------------------------------------


def test_flagship_is_selective_rather_than_everything_that_qualifies(tmp_path):
    """A `* 10` slip once made the quota equal to the candidate count."""
    root = tmp_path / "many"
    for i in range(12):
        write_jpeg(photo_like(1600, 1200, seed=100 + i), root / f"frame_{i:02d}.jpg")

    result = run(PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))
    flagship = result.by_class(RouteClass.FLAGSHIP)
    assert len(flagship) < len(result.records) / 2


def test_no_frame_is_flagship_when_none_clears_the_absolute_floor(tmp_path):
    root = tmp_path / "weak"
    for i in range(6):
        write_jpeg(blurred(seed=i, size=(1600, 1200)), root / f"soft_{i}.jpg")
    result = run(PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))
    assert result.by_class(RouteClass.FLAGSHIP) == []


# --- the plan ---------------------------------------------------------------


def test_the_run_plans_but_never_performs_a_move(options, archive):
    result = run(options)
    assert result.planned_operations
    assert sorted(p.name for p in archive.iterdir()) == [
        "dark.jpg", "good_one.jpg", "good_two.jpg", "lens_cap.jpg", "soft.jpg",
    ]


def test_only_trash_is_planned_for_quarantine(options):
    result = run(options)
    trashed = {r.filename for r in result.by_class(RouteClass.TRASH)}
    planned = {pipeline.Path(op.source).name for op in result.planned_operations}
    assert planned <= trashed


# --- re-routing without re-analysis -----------------------------------------


def test_routing_can_be_redone_from_a_stored_run(options, tmp_path):
    """The payoff of storing every dimension rather than only the class."""
    result = run(options)
    path = tmp_path / "analysis.json"
    pipeline.reports.write_json(result.records, path, summary=result.summary)

    again = pipeline.reclassify(path, CalibrationSet())

    assert len(again) == len(result.records)
    assert not any(c["changed"] for c in again)


def test_changing_a_threshold_changes_the_routing_without_decoding(options, tmp_path):
    """Retuning is a sub-second operation on a run that cost an hour and money."""
    result = run(options)
    path = tmp_path / "analysis.json"
    pipeline.reports.write_json(result.records, path, summary=result.summary)
    before = {r.filename: r.route_class for r in result.records}
    assert RouteClass.REVIEW.value in before.values()

    generous = CalibrationSet(photo=portfolio_first_profile())
    generous.photo.thresholds["stock_standard"] = 5.0
    generous.photo.thresholds["stock_strong"] = 10.0

    after = {c["filename"]: c["route_class"] for c in pipeline.reclassify(path, generous)}

    promoted = [
        name
        for name, was in before.items()
        if was == RouteClass.REVIEW.value and after[name].startswith("stock")
    ]
    assert promoted, f"nothing was promoted: {before} -> {after}"


def test_trash_survives_a_generous_profile(options, tmp_path):
    """Thresholds move the middle. Unrecoverable is not a threshold."""
    result = run(options)
    path = tmp_path / "analysis.json"
    pipeline.reports.write_json(result.records, path, summary=result.summary)

    generous = CalibrationSet()
    generous.photo.thresholds["trash_potential"] = 1.0
    generous.photo.thresholds["stock_standard"] = 1.0

    after = {c["filename"]: c["route_class"] for c in pipeline.reclassify(path, generous)}
    assert after["lens_cap.jpg"] == RouteClass.TRASH.value


# --- the CLI ----------------------------------------------------------------


def test_the_analyze_command_writes_every_report(archive, tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(["measure", "--input", str(archive), "--output", str(out)]) == 0

    assert (out / ".internal" / "reports" / "analysis.json").exists()
    assert (out / ".internal" / "reports" / "analysis.csv").exists()
    assert (out / ".internal" / "reports" / "full_report.html").exists()
    # What a photographer actually opens, at the root and nowhere else.
    assert (out / "report.html").exists()
    assert (out / "photographer_insights.html").exists()
    assert (out / ".internal" / "reports" / "distribution.csv").exists()
    assert (out / ".internal" / "reports" / "delete_candidates.txt").exists()
    assert (out / ".internal" / "reports" / "delete.sh").exists()


def test_the_delete_script_contains_no_filenames_at_all(archive, tmp_path):
    """The only dependable defence against filename injection is not to interpolate.

    A previous version quoted paths with shlex.quote but wrote one bare
    `echo 'already gone: {name}'`, and a file named `it's a photo'; rm -rf $HOME`
    closed the quote and left an executable command in a script the user is told
    to run by hand.
    """
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    script = (out / ".internal" / "reports" / "delete.sh").read_text(encoding="utf-8")

    assert "rm -" not in script and "rm " not in script
    for record in _load(out):
        assert record["filename"] not in script
        assert record["source_path"] not in script
    assert "delete_plan.json" in script


def _load(out):
    rows, _ = pipeline.reports.read_json(out / ".internal" / "reports" / "analysis.json")
    return rows


def test_a_hostile_filename_never_reaches_the_shell(tmp_path):
    """The list is data in JSON; Python carries it out."""
    import layout

    class Record:
        route_class = "trash"
        filename = "x.jpg"
        checksum = "c" * 64
        asset_key = "a/x.jpg"
        source_path = "/archive/it's a photo'; rm -rf $HOME; echo '.jpg"
        reasons = ["corrupt_file: unreadable"]
        all_files = [source_path]
        evidence = "corrupt_file"

    written = layout.write_record_delete_candidates([Record()], tmp_path)
    script = written["script"].read_text(encoding="utf-8")
    assert "rm -rf" not in script
    assert "$HOME; echo" not in script

    plan = json.loads(written["plan"].read_text(encoding="utf-8"))
    assert plan["candidates"][0]["source_path"] == Record.source_path


def test_the_trash_command_is_a_dry_run_by_default(tmp_path, capsys):
    """It moves to the Trash, and only when explicitly told to."""
    import layout

    victim = tmp_path / "archive" / "doomed.jpg"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"jpeg")

    class Record:
        route_class = "trash"
        filename = "doomed.jpg"
        checksum = "c" * 64
        asset_key = "doomed.jpg"
        source_path = str(victim)
        reasons = ["corrupt_file"]
        all_files = [str(victim)]
        evidence = "corrupt_file"

    written = layout.write_record_delete_candidates([Record()], tmp_path / "reports")
    bin_dir = tmp_path / "Trash"

    cli.main(["trash", "--plan", str(written["plan"]), "--trash", str(bin_dir)])
    assert victim.exists()
    assert "Nothing has been moved" in capsys.readouterr().out

    cli.main(["trash", "--plan", str(written["plan"]), "--trash", str(bin_dir), "--apply"])
    assert not victim.exists()
    assert (bin_dir / "doomed.jpg").exists()


def test_the_analyze_command_builds_a_symlink_farm_without_copying(archive, tmp_path):
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    links = [p for p in out.rglob("*.jpg") if p.is_symlink()]
    assert links
    assert all(p.resolve().parent == archive.resolve() for p in links)


def test_the_analyze_command_moves_nothing(archive, tmp_path):
    before = sorted(p.name for p in archive.iterdir())
    cli.main(["measure", "--input", str(archive), "--output", str(tmp_path / "out")])
    assert sorted(p.name for p in archive.iterdir()) == before


def test_a_missing_input_directory_is_an_error(tmp_path, capsys):
    assert cli.main(["measure", "--input", str(tmp_path / "nope"), "--output", str(tmp_path / "o")]) == 1


def test_the_report_command_filters(archive, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    capsys.readouterr()

    analysis = str(out / ".internal" / "reports" / "analysis.json")
    assert cli.main(["report", "--analysis", analysis, "--route-class", "trash"]) == 0
    printed = capsys.readouterr().out
    assert "lens_cap.jpg" in printed
    assert "good_one.jpg" not in printed


def test_the_report_command_renders_russian(archive, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    capsys.readouterr()
    cli.main(["--lang", "ru", "report", "--analysis", str(out / ".internal" / "reports" / "analysis.json")])
    assert "СВОДКА" in capsys.readouterr().out


def test_the_quarantine_command_is_a_dry_run_by_default(archive, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    capsys.readouterr()

    cli.main([
        "quarantine",
        "--analysis", str(out / ".internal" / "reports" / "analysis.json"),
        "--quarantine", str(tmp_path / "bin"),
        "--input", str(archive),
    ])

    assert "Nothing has been moved" in capsys.readouterr().out
    assert (archive / "lens_cap.jpg").exists()


def test_the_quarantine_and_restore_round_trip(archive, tmp_path, capsys):
    out = tmp_path / "out"
    bin_dir = tmp_path / "bin"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    analysis = str(out / ".internal" / "reports" / "analysis.json")

    cli.main(["quarantine", "--analysis", analysis, "--quarantine", str(bin_dir),
              "--input", str(archive), "--apply"])
    assert not (archive / "lens_cap.jpg").exists()

    cli.main(["restore", "--quarantine", str(bin_dir), "--apply"])
    assert (archive / "lens_cap.jpg").exists()


def test_the_purge_command_refuses_without_the_typed_phrase(tmp_path, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    assert cli.main(["purge", "--quarantine", str(bin_dir), "--apply"]) == 2
    assert "Refusing to purge" in capsys.readouterr().err


def test_an_override_recorded_by_the_cli_is_respected_by_the_next_run(archive, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    analysis = str(out / ".internal" / "reports" / "analysis.json")

    cli.main(["override", "--analysis", analysis, "lens_cap.jpg",
              "--set-class", "review", "--note", "keep it"])
    capsys.readouterr()

    cli.main(["measure", "--input", str(archive), "--output", str(out), "--force"])
    assert "Applied 1 manual override" in capsys.readouterr().out

    rows, _ = pipeline.reports.read_json(pipeline.Path(analysis))
    rescued = next(r for r in rows if r["filename"] == "lens_cap.jpg")
    assert rescued["route_class"] == "review"


def test_a_rescued_file_disappears_from_the_quarantine_plan(archive, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    analysis = str(out / ".internal" / "reports" / "analysis.json")
    cli.main(["override", "--analysis", analysis, "lens_cap.jpg", "--set-class", "review"])
    capsys.readouterr()

    cli.main(["measure", "--input", str(archive), "--output", str(out), "--force"])
    printed = capsys.readouterr().out

    # A genuinely blurred frame legitimately stays in the plan; the point is
    # that the rescued one does not.
    rows, _ = pipeline.reports.read_json(pipeline.Path(analysis))
    rescued = next(r for r in rows if r["filename"] == "lens_cap.jpg")
    assert rescued["route_class"] == "review"
    assert "lens_cap.jpg" not in printed.split("would move")[-1]


def test_the_profiles_command_lists_the_built_ins(capsys):
    assert cli.main(["profiles"]) == 0
    assert "portfolio-first" in capsys.readouterr().out


def test_the_reclassify_command_says_nothing_was_spent(archive, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["measure", "--input", str(archive), "--output", str(out)])
    capsys.readouterr()
    cli.main(["reclassify", "--analysis", str(out / ".internal" / "reports" / "analysis.json")])
    assert "no tokens were spent" in capsys.readouterr().out


# --- parallel decoding produces the same run ---------------------------------


def _archive_of(tmp_path, n):
    root = tmp_path / "many"
    for i in range(n):
        write_jpeg(photo_like(300, 200, seed=i), root / f"f{i:03d}.jpg")
    return root


def test_parallel_and_single_process_runs_are_identical(tmp_path):
    """The whole point: faster, bit for bit the same.

    Uses enough files to cross the threshold where a pool is actually used,
    so this is comparing the two code paths rather than one path twice.
    """
    archive = _archive_of(tmp_path, pipeline.PARALLEL_THRESHOLD + 6)

    serial = pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "a", jobs=1)
    )
    parallel = pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "b", jobs=4)
    )

    def snapshot(result):
        return [
            (r.filename, r.category, r.final_score, r.scores, r.issues, r.phash,
             r.cluster_id, r.best_in_cluster)
            for r in sorted(result.records, key=lambda r: r.filename)
        ]

    assert snapshot(serial) == snapshot(parallel)


def test_a_small_run_stays_in_one_process(tmp_path, monkeypatch):
    """Spawning workers for five photographs costs more than it saves."""
    used = {"pool": False}

    class Boom:
        def __init__(self, *a, **kw):
            used["pool"] = True
            raise AssertionError("a pool was started for a handful of files")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Boom)
    archive = _archive_of(tmp_path, 4)
    pipeline.run(pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out"))
    assert not used["pool"]


def test_jobs_one_never_starts_a_pool(tmp_path, monkeypatch):
    class Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("--jobs 1 must stay in this process")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Boom)
    archive = _archive_of(tmp_path, pipeline.PARALLEL_THRESHOLD + 2)
    result = pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out", jobs=1)
    )
    assert len(result.records) == pipeline.PARALLEL_THRESHOLD + 2


def test_the_worker_count_leaves_the_machine_usable():
    import os

    cores = os.cpu_count() or 1
    assert pipeline._worker_count(None, 1000) <= cores
    assert pipeline._worker_count(None, 2) <= 2, "never more workers than work"
    assert pipeline._worker_count(3, 1000) == 3


def test_progress_still_reports_every_asset_in_order(tmp_path):
    seen = []
    archive = _archive_of(tmp_path, pipeline.PARALLEL_THRESHOLD + 2)
    pipeline.run(
        pipeline.PipelineOptions(input_dir=archive, output_dir=tmp_path / "out", jobs=2),
        progress=lambda name, i, total, reused=False: seen.append(name),
    )
    assert seen == sorted(seen), "results arrived out of order"


# --- the socket guard blocks the network, not IPC -----------------------------


def test_the_guard_still_refuses_a_real_network_connection():
    """Loosening it for AF_UNIX must not loosen it for anything that can leave."""
    import socket as socket_module

    with pytest.raises(RuntimeError, match="network access is not allowed"):
        socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM).connect(
            ("example.com", 80)
        )
    with pytest.raises(RuntimeError, match="network access is not allowed"):
        socket_module.create_connection(("example.com", 80))


def test_the_guard_refuses_localhost_too():
    """A test that needs a local server should say so, not get one silently."""
    import socket as socket_module

    with pytest.raises(RuntimeError, match="network access is not allowed"):
        socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM).connect(
            ("127.0.0.1", 8080)
        )


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_the_pool_works_under_every_start_method(method, tmp_path):
    """Python 3.14 changed the Linux default to forkserver, which talks to its
    worker factory over a Unix socket -- and the guard was refusing that.

    Both methods are exercised here so the next default change is caught by a
    test rather than by CI on one of three Python versions.
    """
    import multiprocessing

    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not available on this platform")

    context = multiprocessing.get_context(method)
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        assert list(pool.map(abs, [-1, -2, -3])) == [1, 2, 3]
