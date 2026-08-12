"""Video, sampled across time rather than judged from its first frame.

Everything that shapes a decision is a pure function here -- probe parsing,
timestamp spacing, segment selection, motion analysis -- so the suite covers the
logic without ffmpeg. The one test that needs a real decode is skipped when
ffmpeg is absent and runs in CI, where it is installed.
"""

import numpy as np
import pytest
from synthetic import gray_frames, photo_like, shifted

import video_analyzer as va
from issues import Fixability, IssueCode

needs_ffmpeg = pytest.mark.skipif(not va.ffmpeg_available(), reason="ffmpeg is not installed")


def probe_payload(**overrides):
    stream = {
        "codec_type": "video",
        "codec_name": "hevc",
        "profile": "Main 10",
        "width": 3840,
        "height": 2160,
        "duration": "12.5",
        "r_frame_rate": "30000/1001",
        "avg_frame_rate": "30000/1001",
        "bit_rate": "100000000",
        "pix_fmt": "yuv420p10le",
        "color_transfer": "bt709",
        "nb_frames": "374",
    }
    stream.update(overrides)
    return {"format": {"format_name": "mov,mp4,m4a", "duration": "12.5"}, "streams": [stream]}


# --- probe parsing ----------------------------------------------------------


def test_container_facts_are_read():
    probe = va.parse_probe(probe_payload())
    assert probe.codec == "hevc"
    assert probe.width == 3840
    assert probe.duration == 12.5
    assert probe.frame_rate == pytest.approx(29.97, abs=0.01)
    assert not probe.corrupt


def test_a_file_with_no_video_stream_is_corrupt():
    assert va.parse_probe({"format": {}, "streams": []}).corrupt


def test_a_stream_with_no_dimensions_is_corrupt():
    assert va.parse_probe(probe_payload(width=0, height=0)).corrupt


def test_a_stream_with_no_duration_is_corrupt():
    payload = probe_payload(duration=None)
    payload["format"]["duration"] = None
    assert va.parse_probe(payload).corrupt


@pytest.mark.parametrize(
    ("width", "height", "orientation"),
    [(3840, 2160, "horizontal"), (1080, 1920, "vertical"), (1080, 1080, "square")],
)
def test_orientation_is_derived(width, height, orientation):
    assert va.parse_probe(probe_payload(width=width, height=height)).orientation == orientation


def test_variable_frame_rate_is_detected():
    """VFR on a CFR timeline drifts out of sync; it is a submission problem."""
    assert va.parse_probe(probe_payload(avg_frame_rate="24000/1001")).is_variable_frame_rate


def test_constant_frame_rate_is_not_flagged():
    assert not va.parse_probe(probe_payload()).is_variable_frame_rate


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_hdr_transfer_characteristics_are_recognised(transfer):
    assert va.parse_probe(probe_payload(color_transfer=transfer)).looks_log_or_hdr


def test_a_log_profile_is_recognised():
    assert va.parse_probe(probe_payload(profile="V-Log")).looks_log_or_hdr


def test_rec709_is_not_treated_as_log():
    assert not va.parse_probe(probe_payload()).looks_log_or_hdr


def test_high_frame_rate_hints_at_slow_motion():
    assert va.parse_probe(probe_payload(r_frame_rate="120/1")).slow_motion_hint


def test_very_low_frame_rate_hints_at_a_time_lapse():
    assert va.parse_probe(probe_payload(r_frame_rate="5/1")).time_lapse_hint


def test_audio_is_read_when_present():
    payload = probe_payload()
    payload["streams"].append(
        {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"}
    )
    probe = va.parse_probe(payload)
    assert probe.audio.present
    assert probe.audio.channels == 2


def test_a_silent_clip_reports_no_audio():
    assert not va.parse_probe(probe_payload()).audio.present


def test_a_malformed_frame_rate_does_not_crash_parsing():
    assert va.parse_probe(probe_payload(r_frame_rate="not/a/ratio")).frame_rate == 0.0


# --- temporal sampling ------------------------------------------------------


def test_samples_span_the_whole_clip():
    """Scoring from the first frame scores the part the editor was going to cut."""
    stamps = va.sample_timestamps(30.0, count=9)
    assert len(stamps) == 9
    assert stamps[0] < 2.0
    assert stamps[-1] > 28.0


def test_samples_are_evenly_spaced():
    stamps = va.sample_timestamps(60.0, count=5)
    gaps = [round(b - a, 2) for a, b in zip(stamps, stamps[1:], strict=False)]
    assert len(set(gaps)) == 1


def test_the_first_and_last_moments_are_not_sampled():
    """The operator is still settling the camera in the first half second."""
    stamps = va.sample_timestamps(20.0, count=5)
    assert stamps[0] > 0.0
    assert stamps[-1] < 20.0


def test_a_very_short_clip_gives_one_sample():
    assert len(va.sample_timestamps(1.2, count=9)) == 1


def test_a_zero_length_clip_gives_no_samples():
    assert va.sample_timestamps(0.0) == []


def test_asking_for_one_sample_takes_the_middle():
    assert va.sample_timestamps(10.0, count=1) == [5.0]


# --- motion -----------------------------------------------------------------


def sample(ts, quality=60.0, luma=120.0):
    return va.FrameSample(
        timestamp=ts, quality=quality, mean_luma=luma, blur_ratio=20.0,
        sharpness_tile=500.0, clipped_highlights=0.0, clipped_shadows=0.0,
    )


def test_a_known_shift_is_recovered():
    base = np.asarray(photo_like(256, 192).convert("L"), dtype=np.float64)
    moved = np.asarray(shifted(photo_like(256, 192).convert("L"), 5, -3), dtype=np.float64)
    dx, dy = va.phase_shift(base, moved)
    assert (round(dx), round(dy)) == (-5, 3)


def test_identical_frames_report_no_movement():
    frame = np.asarray(photo_like(128, 96).convert("L"), dtype=np.float64)
    assert va.phase_shift(frame, frame) == (0.0, 0.0)


def test_mismatched_shapes_report_no_movement():
    assert va.phase_shift(np.zeros((10, 10)), np.zeros((20, 20))) == (0.0, 0.0)


def test_a_tripod_shot_is_static():
    report = va.analyse_motion(gray_frames(6, jitter=0.0, pan=0.0))
    assert report.is_static
    assert report.camera_movement == "static"


def test_a_deliberate_pan_is_not_reported_as_shake():
    """A pan and a shaky hold can average the same displacement."""
    report = va.analyse_motion(gray_frames(8, jitter=0.0, pan=3.0))
    assert report.pan_magnitude > 1.0
    assert report.jitter < report.pan_magnitude
    assert report.camera_movement == "pan or tilt"


def test_handheld_jitter_is_separated_from_panning():
    report = va.analyse_motion(gray_frames(8, jitter=4.0, pan=0.0, seed=2))
    assert report.jitter > 1.0
    assert report.camera_movement == "handheld"


def test_a_single_frame_yields_no_motion():
    assert va.analyse_motion([np.zeros((32, 32))]).shifts == []


def test_frozen_frames_are_counted():
    frame = np.asarray(photo_like(128, 96).convert("L"), dtype=np.float64)
    assert va.analyse_motion([frame, frame, frame]).frozen_pairs == 2


def test_bursts_are_combined_worst_case_not_averaged():
    """A clip steady for eight seconds and unusable for two still needs a trim."""
    calm = va.MotionReport(jitter=0.2, pan_magnitude=0.1, shear=0.0)
    wild = va.MotionReport(jitter=9.0, pan_magnitude=1.0, shear=6.0)
    combined = va.combine_motion([calm, wild])
    assert combined.jitter == 9.0
    assert combined.shear == 6.0


def test_combining_nothing_gives_a_still_report():
    assert va.combine_motion([]).is_static


# --- usable segments --------------------------------------------------------


def test_a_uniformly_good_clip_is_one_usable_segment():
    samples = [sample(t) for t in (1.0, 3.0, 5.0, 7.0, 9.0)]
    segments = va.usable_segments(samples, duration=10.0)
    assert len(segments) == 1
    assert segments[0].duration >= 3.0


def test_a_clip_that_is_bad_throughout_has_no_usable_segment():
    samples = [sample(t, quality=5.0) for t in (1.0, 3.0, 5.0, 7.0)]
    assert va.usable_segments(samples, duration=8.0) == []


def test_black_frames_do_not_count_as_usable():
    samples = [sample(t, quality=70.0, luma=2.0) for t in (1.0, 3.0, 5.0)]
    assert va.usable_segments(samples, duration=6.0) == []


def test_a_segment_shorter_than_the_minimum_is_not_offered():
    samples = [sample(1.0), sample(9.0, quality=5.0), sample(17.0, quality=5.0)]
    assert va.usable_segments(samples, duration=20.0, min_duration=5.0) == []


def test_segments_carry_handles_so_an_editor_can_cut_on_them():
    """A segment usable for exactly its own length has no room for a transition."""
    samples = [sample(t) for t in (2.0, 4.0, 6.0, 8.0)]
    segment = va.usable_segments(samples, duration=10.0)[0]
    assert segment.start > 0.0
    assert segment.end < 10.0


def test_no_samples_gives_no_segments():
    assert va.usable_segments([], duration=10.0) == []


# --- derived quality over time ----------------------------------------------


def analysis_with(samples, probe=None, motion=None):
    result = va.VideoAnalysis(probe=probe or va.parse_probe(probe_payload()))
    result.samples = samples
    if motion:
        result.motion = motion
    result.segments = va.usable_segments(samples, result.probe.duration)
    return result


def test_focus_that_holds_scores_consistent():
    assert analysis_with([sample(t) for t in (1.0, 5.0, 9.0)]).focus_consistency > 0.9


def test_focus_that_wanders_scores_inconsistent():
    result = va.VideoAnalysis(probe=va.parse_probe(probe_payload()))
    result.samples = [sample(1.0), sample(5.0), sample(9.0)]
    result.samples[1].blur_ratio = 1.2
    assert result.focus_consistency < 0.9


def test_exposure_range_is_the_spread_across_the_clip():
    result = va.VideoAnalysis(probe=va.parse_probe(probe_payload()))
    result.samples = [sample(1.0, luma=60.0), sample(5.0, luma=160.0)]
    assert result.exposure_range == 100.0


def test_a_clip_with_no_samples_has_no_mean_quality():
    assert va.VideoAnalysis(probe=va.parse_probe(probe_payload())).mean_quality == 0.0


# --- issues -----------------------------------------------------------------


def test_a_corrupt_clip_reports_encoding_corruption():
    result = va.VideoAnalysis(probe=va.parse_probe({"format": {}, "streams": []}))
    assert IssueCode.ENCODING_CORRUPTION in va.detect_video_issues(result).codes()


def test_a_clip_shorter_than_the_marketplace_floor_is_not_called_damaged():
    """Three seconds is a submission rule, not a property of the footage."""
    probe = va.parse_probe(probe_payload(duration="1.5"))
    result = analysis_with([sample(0.75)], probe=probe)
    found = va.detect_video_issues(result)

    assert IssueCode.SHORT_CLIP in found.codes()
    assert IssueCode.UNUSABLE_DURATION not in found.codes()
    assert not any(i.code is IssueCode.SHORT_CLIP and i.is_blocker for i in found)


def test_a_fragment_of_a_second_is_still_unusable():
    probe = va.parse_probe(probe_payload(duration="0.2"))
    result = analysis_with([sample(0.1)], probe=probe)
    assert IssueCode.UNUSABLE_DURATION in va.detect_video_issues(result).codes()


def test_a_clip_with_no_usable_segment_says_so():
    result = analysis_with([sample(t, quality=5.0) for t in (1.0, 5.0, 9.0)])
    assert IssueCode.NO_USABLE_SEGMENT in va.detect_video_issues(result).codes()


def test_severe_shake_is_unrecoverable():
    result = analysis_with(
        [sample(t) for t in (1.0, 5.0, 9.0)], motion=va.MotionReport(jitter=9.0)
    )
    found = va.detect_video_issues(result)
    assert IssueCode.UNUSABLE_SHAKE in found.codes()
    assert found.has_blocker


def test_moderate_shake_is_only_partially_fixable():
    """Stabilisation is available but costs a crop."""
    result = analysis_with(
        [sample(t) for t in (1.0, 5.0, 9.0)], motion=va.MotionReport(jitter=2.5)
    )
    found = va.detect_video_issues(result)
    assert IssueCode.MODERATE_SHAKE in found.codes()
    assert not found.has_blocker
    assert found.partial[0].fixability is Fixability.PARTIAL


def test_a_steady_clip_reports_no_shake():
    result = analysis_with(
        [sample(t) for t in (1.0, 5.0, 9.0)], motion=va.MotionReport(jitter=0.2)
    )
    codes = va.detect_video_issues(result).codes()
    assert IssueCode.UNUSABLE_SHAKE not in codes
    assert IssueCode.MODERATE_SHAKE not in codes


def test_low_resolution_footage_is_flagged():
    probe = va.parse_probe(probe_payload(width=640, height=480))
    result = analysis_with([sample(t) for t in (1.0, 5.0, 9.0)], probe=probe)
    assert IssueCode.INSUFFICIENT_RESOLUTION in va.detect_video_issues(result).codes()


def test_an_all_black_clip_is_an_empty_frame():
    result = analysis_with([sample(t, luma=1.0) for t in (1.0, 5.0, 9.0)])
    result.black_frames = 3
    assert IssueCode.EMPTY_FRAME in va.detect_video_issues(result).codes()


def test_rolling_shutter_is_advisory_rather_than_certain():
    result = analysis_with(
        [sample(t) for t in (1.0, 5.0, 9.0)], motion=va.MotionReport(shear=5.0)
    )
    shear_issues = [i for i in va.detect_video_issues(result) if i.code is IssueCode.ROLLING_SHUTTER]
    assert shear_issues and shear_issues[0].certainty < 1.0


# --- the one test that decodes ----------------------------------------------


def encode_clip(tmp_path, *, frames=16, pan=6, size=(1920, 1080), fps=3):
    """A real file, built from photograph-like frames panning steadily.

    ffmpeg's own `testsrc` is not usable here: it is a flat colour-bar pattern
    that scores 32 on the quality function, below the usable-segment floor, so a
    clip made from it correctly reports no usable segment and tests nothing.
    """
    import subprocess

    source = photo_like(size[0], size[1], seed=5)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in range(frames):
        shifted(source, i * pan, 0).save(frame_dir / f"f{i:03d}.png")

    clip = tmp_path / "generated.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(frame_dir / "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(clip),
        ],
        check=True,
        timeout=180,
    )
    return clip


@needs_ffmpeg
def test_a_real_clip_is_probed_sampled_and_segmented(tmp_path):
    """End to end against a generated clip, so the subprocess wiring is covered."""
    clip = encode_clip(tmp_path)
    result = va.analyze_video(clip, sample_count=5, burst_positions=1)

    assert result.probe.width == 1920
    assert result.probe.duration > 4.0
    assert len(result.samples) >= 3
    assert len({s.timestamp for s in result.samples}) == len(result.samples)
    assert result.has_usable_segment
    assert result.poster_timestamp > 0


@needs_ffmpeg
def test_a_real_pan_is_read_as_a_pan_rather_than_as_shake(tmp_path):
    """The distinction that decides whether a clip is usable, end to end."""
    clip = encode_clip(tmp_path, pan=8)
    result = va.analyze_video(clip, sample_count=5, burst_positions=1)

    assert result.motion.pan_magnitude > result.motion.jitter
    assert IssueCode.UNUSABLE_SHAKE not in va.detect_video_issues(result).codes()


@needs_ffmpeg
def test_a_clip_too_short_for_any_marketplace_is_rejected(tmp_path):
    clip = encode_clip(tmp_path, frames=4, size=(640, 480), fps=4)
    result = va.analyze_video(clip, sample_count=3, burst_positions=1)
    codes = va.detect_video_issues(result).codes()
    assert IssueCode.UNUSABLE_DURATION in codes or IssueCode.NO_USABLE_SEGMENT in codes
