"""Video, judged over time rather than from its first frame.

A clip is not a photograph with a duration. The first frame is routinely the
worst one in the file -- the operator is still settling the camera, the exposure
is still adapting, the focus is still hunting -- so anything that scores a clip
from frame zero scores the part the editor was always going to cut. Everything
here samples across the whole timeline instead.

Three passes, in increasing cost:

1. **`ffprobe`** for the container facts: codec, dimensions, duration, frame
   rate, bitrate, colour metadata, audio streams. Cheap, and enough on its own
   to reject a clip that is too short or too small for any marketplace.

2. **Sparse samples** across the duration, each measured with the same Stage 0
   function used on stills. This gives focus, exposure and quality *over time*,
   which is what separates a clip that is soft throughout from one that is soft
   for two seconds in the middle.

3. **Dense bursts** -- runs of consecutive frames at a few positions -- for
   anything that only exists between frames: camera shake, motion, rolling
   shutter, frozen frames. Consecutive frames are the only way to see these, and
   they are also the expensive thing to extract, hence only a few short runs.

The camera-motion estimate is phase correlation between consecutive frames,
which yields the global pixel shift. That distinction is the useful one: a
smooth accumulating shift is a pan and is *intentional*; the same magnitude of
shift jittering around zero is shake and is a defect. Anything that scores raw
inter-frame difference calls a good pan unusable.

Every external call goes through an argument array with a timeout. No shell
string is ever built, because these paths come from a user-supplied directory.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

PROBE_TIMEOUT = 60
EXTRACT_TIMEOUT = 300

SAMPLE_COUNT = 9
BURST_POSITIONS = 3
BURST_FRAMES = 8
MOTION_WORK_PX = 128

# Marketplace floors, used to mark a clip unusable before anything is measured.
MIN_USABLE_DURATION = 3.0
MAX_USABLE_DURATION = 60.0
MIN_SEGMENT_DURATION = 3.0
SEGMENT_HANDLE = 0.5

BLACK_LUMA = 12.0
FROZEN_DIFF = 0.6
SHAKE_JITTER_PX = 1.6
SEVERE_SHAKE_JITTER_PX = 4.5
FLICKER_STDDEV = 9.0
EXPOSURE_DRIFT_RANGE = 42.0
FOCUS_INCONSISTENCY = 0.45
ROLLING_SHUTTER_SHEAR = 2.5

LOG_TRANSFERS = {"arib-std-b67", "smpte2084", "log", "bt2020-10", "bt2020-12"}
LOG_HINTS = ("log", "slog", "vlog", "clog", "hlg", "flat")


class FFmpegMissing(RuntimeError):
    """ffprobe/ffmpeg are not installed. Video support is optional by design."""


class VideoProbeError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which(FFPROBE) is not None and shutil.which(FFMPEG) is not None


@dataclass
class AudioTrack:
    codec: str = ""
    channels: int = 0
    sample_rate: int = 0
    duration: float = 0.0

    @property
    def present(self) -> bool:
        return bool(self.codec)


@dataclass
class VideoProbe:
    """Container facts, straight from ffprobe. No judgement here."""

    container: str = ""
    codec: str = ""
    profile: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    frame_rate: float = 0.0
    avg_frame_rate: float = 0.0
    bit_rate: int = 0
    pix_fmt: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    color_space: str = ""
    audio: AudioTrack = field(default_factory=AudioTrack)
    nb_frames: int = 0
    corrupt: bool = False
    error: str = ""

    @property
    def megapixels(self) -> float:
        return round(self.width * self.height / 1_000_000, 2)

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / self.height, 3) if self.height else 0.0

    @property
    def orientation(self) -> str:
        if not self.height or not self.width:
            return "unknown"
        if abs(self.aspect_ratio - 1.0) < 0.05:
            return "square"
        return "horizontal" if self.width > self.height else "vertical"

    @property
    def is_variable_frame_rate(self) -> bool:
        """r_frame_rate and avg_frame_rate disagree on a VFR file.

        Phone footage is routinely VFR, and a VFR source dropped onto a CFR
        timeline drifts out of sync -- which is a real submission problem, not
        a cosmetic one.
        """
        if not self.frame_rate or not self.avg_frame_rate:
            return False
        return abs(self.frame_rate - self.avg_frame_rate) / max(self.frame_rate, 1e-6) > 0.02

    @property
    def looks_log_or_hdr(self) -> bool:
        transfer = (self.color_transfer or "").lower()
        profile = (self.profile or "").lower()
        return transfer in LOG_TRANSFERS or any(h in profile for h in LOG_HINTS)

    @property
    def slow_motion_hint(self) -> bool:
        return self.frame_rate >= 100.0

    @property
    def time_lapse_hint(self) -> bool:
        return 0 < self.frame_rate <= 12.0


def parse_probe(payload: dict) -> VideoProbe:
    """Pure: ffprobe JSON in, VideoProbe out. Tested without ffmpeg present."""
    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        return VideoProbe(
            container=str(fmt.get("format_name", "")),
            corrupt=True,
            error="no video stream",
        )

    probe = VideoProbe(
        container=str(fmt.get("format_name", "")),
        codec=str(video.get("codec_name", "")),
        profile=str(video.get("profile", "")),
        width=_int(video.get("width")),
        height=_int(video.get("height")),
        duration=_float(video.get("duration") or fmt.get("duration")),
        frame_rate=_ratio(video.get("r_frame_rate")),
        avg_frame_rate=_ratio(video.get("avg_frame_rate")),
        bit_rate=_int(video.get("bit_rate") or fmt.get("bit_rate")),
        pix_fmt=str(video.get("pix_fmt", "")),
        color_transfer=str(video.get("color_transfer", "")),
        color_primaries=str(video.get("color_primaries", "")),
        color_space=str(video.get("color_space", "")),
        nb_frames=_int(video.get("nb_frames")),
    )
    if audio:
        probe.audio = AudioTrack(
            codec=str(audio.get("codec_name", "")),
            channels=_int(audio.get("channels")),
            sample_rate=_int(audio.get("sample_rate")),
            duration=_float(audio.get("duration")),
        )
    if not probe.width or not probe.height or probe.duration <= 0:
        probe.corrupt = True
        probe.error = "missing dimensions or duration"
    return probe


def probe(path: Path, timeout: int = PROBE_TIMEOUT) -> VideoProbe:
    """Run ffprobe. Raises FFmpegMissing when the binary is not installed."""
    if shutil.which(FFPROBE) is None:
        raise FFmpegMissing("ffprobe not found; install FFmpeg for video support")
    cmd = [
        FFPROBE,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise VideoProbeError(f"ffprobe timed out on {path.name}") from e
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        return VideoProbe(corrupt=True, error=stderr[:300] or f"ffprobe exit {result.returncode}")
    try:
        return parse_probe(json.loads(result.stdout))
    except json.JSONDecodeError as e:
        return VideoProbe(corrupt=True, error=f"unparseable ffprobe output: {e}")


# --- sampling ---------------------------------------------------------------


@dataclass
class FrameSample:
    timestamp: float
    quality: float
    mean_luma: float
    blur_ratio: float
    sharpness_tile: float
    clipped_highlights: float
    clipped_shadows: float
    path: Path | None = None


def sample_timestamps(duration: float, count: int = SAMPLE_COUNT) -> list[float]:
    """Equal intervals, inset from both ends.

    The inset is not cosmetic: the first and last half-second of a handheld clip
    contain the hand reaching for the record button, and including them drags
    down the score of a clip whose usable body is fine.
    """
    if duration <= 0 or count < 1:
        return []
    if duration <= 2.0:
        return [duration / 2.0]
    inset = min(0.5, duration * 0.05)
    span = duration - 2 * inset
    if count == 1:
        return [duration / 2.0]
    return [round(inset + span * i / (count - 1), 3) for i in range(count)]


def extract_frames(path: Path, timestamps: list[float], out_dir: Path, width: int = 512) -> list[Path]:
    """One seek + one frame per timestamp, written as JPEGs.

    Seeking before `-i` is the fast path -- ffmpeg jumps to the nearest keyframe
    instead of decoding from the start, which is the difference between seconds
    and minutes on a 200 MB clip, and the difference between bounded and
    unbounded memory on a long one.
    """
    if shutil.which(FFMPEG) is None:
        raise FFmpegMissing("ffmpeg not found; install FFmpeg for video support")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, ts in enumerate(timestamps):
        out_path = out_dir / f"sample_{i:03d}.jpg"
        cmd = [
            FFMPEG, "-nostdin", "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(path),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            "-y", str(out_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=EXTRACT_TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            logger.warning("Frame extraction timed out at %.2fs in %s", ts, path.name)
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            written.append(out_path)
    return written


def extract_burst(path: Path, start: float, count: int, out_dir: Path, width: int = MOTION_WORK_PX) -> list[Path]:
    """Consecutive frames from one point, small, for motion analysis."""
    if shutil.which(FFMPEG) is None:
        raise FFmpegMissing("ffmpeg not found; install FFmpeg for video support")
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "burst_%03d.png"
    cmd = [
        FFMPEG, "-nostdin", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(path),
        "-frames:v", str(count),
        "-vf", f"scale={width}:-2,format=gray",
        "-y", str(pattern),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=EXTRACT_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("Burst extraction timed out at %.2fs in %s", start, path.name)
        return []
    return sorted(out_dir.glob("burst_*.png"))


# --- motion, between frames -------------------------------------------------


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Global pixel shift from `a` to `b`, by phase correlation.

    Frequency-domain rather than pixel-domain because it is O(n log n), immune
    to a brightness change between the two frames, and gives a single dominant
    translation instead of a blur of local matches.
    """
    if a.shape != b.shape or min(a.shape) < 8:
        return (0.0, 0.0)
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    correlation = np.fft.ifft2(cross / np.where(magnitude < 1e-9, 1e-9, magnitude)).real
    peak = np.unravel_index(int(correlation.argmax()), correlation.shape)
    dy = peak[0] if peak[0] <= a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] <= a.shape[1] // 2 else peak[1] - a.shape[1]
    return (float(dx), float(dy))


@dataclass
class MotionReport:
    shifts: list[tuple[float, float]] = field(default_factory=list)
    pan_magnitude: float = 0.0
    jitter: float = 0.0
    shear: float = 0.0
    frozen_pairs: int = 0

    @property
    def is_static(self) -> bool:
        return self.pan_magnitude < 0.5 and self.jitter < 0.5

    @property
    def has_motion(self) -> bool:
        return self.pan_magnitude >= 0.5 or self.jitter >= 0.5

    @property
    def camera_movement(self) -> str:
        if self.is_static:
            return "static"
        if self.pan_magnitude > self.jitter * 1.5:
            return "pan or tilt"
        return "handheld"


def analyse_motion(frames: list[np.ndarray]) -> MotionReport:
    """Split camera movement into intended (pan) and unintended (jitter).

    The mean shift is the pan; the spread around it is the shake. A tripod pan
    and a shaky handheld hold can produce the same average displacement, and
    only the second one is a defect.
    """
    report = MotionReport()
    if len(frames) < 2:
        return report

    for a, b in zip(frames, frames[1:], strict=False):
        report.shifts.append(phase_shift(a, b))
        if float(np.abs(a - b).mean()) < FROZEN_DIFF:
            report.frozen_pairs += 1

    dxs = np.array([s[0] for s in report.shifts])
    dys = np.array([s[1] for s in report.shifts])
    report.pan_magnitude = round(float(math.hypot(dxs.mean(), dys.mean())), 3)
    report.jitter = round(float(math.hypot(dxs.std(), dys.std())), 3)
    report.shear = round(_shear(frames), 3)
    return report


def combine_motion(bursts: list[MotionReport]) -> MotionReport:
    """Merge per-burst motion into one report, worst case winning.

    Taking the maximum rather than the mean is deliberate: a clip that is
    rock-steady for eight seconds and unusable for two is a clip that needs a
    trim, and averaging hides exactly the part an editor has to deal with.
    """
    combined = MotionReport()
    if not bursts:
        return combined
    for burst in bursts:
        combined.shifts.extend(burst.shifts)
        combined.frozen_pairs += burst.frozen_pairs
    combined.pan_magnitude = round(max(b.pan_magnitude for b in bursts), 3)
    combined.jitter = round(max(b.jitter for b in bursts), 3)
    combined.shear = round(max(b.shear for b in bursts), 3)
    return combined


def _shear(frames: list[np.ndarray]) -> float:
    """Difference between the top and bottom halves' horizontal shift.

    A rolling shutter reads the sensor top to bottom, so during a fast pan the
    bottom of the frame lags the top and the image leans. Comparing the two
    halves' independently-measured shifts is the cheapest way to see it.
    """
    worst = 0.0
    for a, b in zip(frames, frames[1:], strict=False):
        half = a.shape[0] // 2
        if half < 8:
            continue
        top_dx, _ = phase_shift(a[:half], b[:half])
        bottom_dx, _ = phase_shift(a[half:], b[half:])
        worst = max(worst, abs(top_dx - bottom_dx))
    return worst


# --- usable segments --------------------------------------------------------


@dataclass
class Segment:
    start: float
    end: float
    mean_quality: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


def usable_segments(
    samples: list[FrameSample],
    duration: float,
    *,
    min_quality: float = 35.0,
    min_duration: float = MIN_SEGMENT_DURATION,
) -> list[Segment]:
    """Contiguous runs of acceptable samples, trimmed by a handle at each end.

    The handle is what an editor needs to cut on: a segment that is usable for
    exactly its own length has no room for a transition and is not usable in
    practice.
    """
    if not samples or duration <= 0:
        return []

    good = [s for s in samples if s.quality >= min_quality and s.mean_luma > BLACK_LUMA]
    if not good:
        return []

    step = duration / max(len(samples), 1)
    runs: list[list[FrameSample]] = []
    for sample in samples:
        if sample in good:
            if runs and abs(runs[-1][-1].timestamp - sample.timestamp) <= step * 1.6:
                runs[-1].append(sample)
            else:
                runs.append([sample])

    segments = []
    for run in runs:
        start = max(0.0, run[0].timestamp - step / 2 + SEGMENT_HANDLE)
        end = min(duration, run[-1].timestamp + step / 2 - SEGMENT_HANDLE)
        if end - start >= min_duration:
            segments.append(
                Segment(
                    start=round(start, 3),
                    end=round(end, 3),
                    mean_quality=round(sum(s.quality for s in run) / len(run), 2),
                )
            )
    return segments


@dataclass
class VideoAnalysis:
    probe: VideoProbe
    samples: list[FrameSample] = field(default_factory=list)
    motion: MotionReport = field(default_factory=MotionReport)
    segments: list[Segment] = field(default_factory=list)
    poster_timestamp: float = 0.0
    poster_path: Path | None = None
    black_frames: int = 0
    frozen_frames: int = 0

    @property
    def mean_quality(self) -> float:
        return round(sum(s.quality for s in self.samples) / len(self.samples), 2) if self.samples else 0.0

    @property
    def focus_consistency(self) -> float:
        """1.0 = focus holds all the way through; 0 = it wanders badly."""
        ratios = [s.blur_ratio for s in self.samples]
        if len(ratios) < 2:
            return 1.0
        mean = sum(ratios) / len(ratios)
        if mean <= 0:
            return 0.0
        spread = (sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5
        return round(max(0.0, 1.0 - spread / mean), 3)

    @property
    def exposure_range(self) -> float:
        lumas = [s.mean_luma for s in self.samples]
        return round(max(lumas) - min(lumas), 2) if lumas else 0.0

    @property
    def flicker(self) -> float:
        lumas = [s.mean_luma for s in self.samples]
        if len(lumas) < 3:
            return 0.0
        deltas = [abs(b - a) for a, b in zip(lumas, lumas[1:], strict=False)]
        mean = sum(deltas) / len(deltas)
        return round((sum((d - mean) ** 2 for d in deltas) / len(deltas)) ** 0.5, 2)

    @property
    def longest_segment(self) -> Segment | None:
        return max(self.segments, key=lambda s: s.duration, default=None)

    @property
    def has_usable_segment(self) -> bool:
        return bool(self.segments)


def measure_sample(path: Path, timestamp: float) -> FrameSample | None:
    """Measure one extracted still with the same code the photo path uses."""
    import technical_filter
    from edit_recipe import frame_quality

    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            report = technical_filter.analyze(rgb)
            array = np.asarray(rgb, dtype=np.float64)
    except Exception as e:
        logger.warning("Could not measure %s: %s", path.name, e)
        return None

    luma = array @ np.array([0.299, 0.587, 0.114])
    return FrameSample(
        timestamp=timestamp,
        quality=frame_quality(array),
        mean_luma=round(float(luma.mean()), 2),
        blur_ratio=round(report.blur_ratio, 3),
        sharpness_tile=round(report.sharpness_tile, 2),
        clipped_highlights=round(report.clipped_highlights, 4),
        clipped_shadows=round(report.clipped_shadows, 4),
        path=path,
    )


def analyze_video(
    path: Path,
    *,
    sample_count: int = SAMPLE_COUNT,
    burst_positions: int = BURST_POSITIONS,
    work_dir: Path | None = None,
) -> VideoAnalysis:
    """Probe, sample across the timeline, then look between frames.

    Temporary frames go to a scratch directory that is removed on the way out,
    including when something raises -- otherwise a 5000-clip run leaves tens of
    thousands of JPEGs behind.
    """
    info = probe(path)
    analysis = VideoAnalysis(probe=info)
    if info.corrupt or info.duration <= 0:
        return analysis

    owned_dir = work_dir is None
    scratch = Path(tempfile.mkdtemp(prefix="pat_video_")) if owned_dir else work_dir
    try:
        timestamps = sample_timestamps(info.duration, sample_count)
        frame_paths = extract_frames(path, timestamps, scratch / "samples")
        for ts, frame_path in zip(timestamps, frame_paths, strict=False):
            sample = measure_sample(frame_path, ts)
            if sample:
                analysis.samples.append(sample)

        analysis.black_frames = sum(1 for s in analysis.samples if s.mean_luma <= BLACK_LUMA)

        # Each burst is analysed on its own and the results combined. Chaining
        # them into one list compares the last frame of one burst with the
        # first frame of the next -- frames seconds apart -- and reports the
        # resulting jump as camera shake. That produced 22px of "jitter" on a
        # 128px frame, which is not a number any real handheld shot can reach.
        per_burst: list[MotionReport] = []
        for i, start in enumerate(_burst_starts(info.duration, burst_positions)):
            burst_dir = scratch / f"burst_{i}"
            frames: list[np.ndarray] = []
            for burst_path in extract_burst(path, start, BURST_FRAMES, burst_dir):
                try:
                    with Image.open(burst_path) as img:
                        frames.append(np.asarray(img.convert("L"), dtype=np.float64))
                except Exception as e:  # pragma: no cover - unreadable temp frame
                    logger.debug("Skipping burst frame %s: %s", burst_path, e)
            if len(frames) >= 2:
                per_burst.append(analyse_motion(frames))
        analysis.motion = combine_motion(per_burst)
        analysis.frozen_frames = analysis.motion.frozen_pairs

        analysis.segments = usable_segments(analysis.samples, info.duration)

        if analysis.samples:
            best = max(analysis.samples, key=lambda s: s.quality)
            analysis.poster_timestamp = best.timestamp
            if best.path and owned_dir:
                # The scratch directory is about to go; keep the poster.
                analysis.poster_path = None
            else:
                analysis.poster_path = best.path
    finally:
        if owned_dir:
            shutil.rmtree(scratch, ignore_errors=True)
    return analysis


def _burst_starts(duration: float, count: int) -> list[float]:
    if duration <= 0 or count < 1:
        return []
    usable = max(0.0, duration - 1.0)
    if count == 1:
        return [usable / 2.0]
    return [round(usable * (i + 0.5) / count, 3) for i in range(count)]


# --- turning all that into typed issues -------------------------------------


def detect_video_issues(analysis: VideoAnalysis):
    """Video measurements to the same Issue vocabulary the photos use."""
    import issues as issues_module
    from issues import IssueCode, IssueSet

    found = IssueSet()
    info = analysis.probe

    if info.corrupt:
        found.add(IssueCode.ENCODING_CORRUPTION, info.error or "unreadable stream")
        return found

    if info.duration < issues_module.TRULY_UNUSABLE_DURATION:
        found.add(IssueCode.UNUSABLE_DURATION, f"{info.duration:.2f}s is barely any footage")
    elif info.duration < MIN_USABLE_DURATION:
        # Below the usual marketplace floor, which is a submission rule rather
        # than a defect. A two-second clip can still be the only footage of
        # something, and deleting it because Adobe wants five seconds is the
        # tool substituting a stock policy for the photographer's judgement.
        found.add(
            IssueCode.SHORT_CLIP,
            f"{info.duration:.1f}s is below the usual {MIN_USABLE_DURATION:.0f}s marketplace floor",
            certainty=0.9,
        )
    if info.megapixels < 1.9:  # below roughly 1920x1080
        found.add(IssueCode.INSUFFICIENT_RESOLUTION, f"{info.width}x{info.height}")

    if not analysis.samples:
        found.add(IssueCode.ENCODING_CORRUPTION, "no frame could be decoded")
        return found

    if not analysis.has_usable_segment:
        found.add(
            IssueCode.NO_USABLE_SEGMENT,
            f"no run of {MIN_SEGMENT_DURATION}s meets the quality floor",
        )
    elif info.duration > MAX_USABLE_DURATION or analysis.segments[0].start > SEGMENT_HANDLE:
        found.add(IssueCode.NEEDS_TRIM, f"usable body is {_segment_summary(analysis)}")

    if analysis.motion.jitter >= SEVERE_SHAKE_JITTER_PX:
        found.add(
            IssueCode.UNUSABLE_SHAKE,
            f"inter-frame jitter {analysis.motion.jitter:.1f}px",
        )
    elif analysis.motion.jitter >= SHAKE_JITTER_PX:
        found.add(
            IssueCode.MODERATE_SHAKE,
            f"jitter {analysis.motion.jitter:.1f}px; stabilisation needs a crop",
            certainty=0.7,
        )

    if analysis.motion.shear >= ROLLING_SHUTTER_SHEAR:
        found.add(
            IssueCode.ROLLING_SHUTTER,
            f"top/bottom shear {analysis.motion.shear:.1f}px during motion",
            certainty=0.5,
        )

    if analysis.focus_consistency < FOCUS_INCONSISTENCY:
        found.add(
            IssueCode.BROKEN_FOCUS_PULL,
            f"focus consistency {analysis.focus_consistency:.2f}",
        )

    if analysis.exposure_range > EXPOSURE_DRIFT_RANGE:
        found.add(IssueCode.EXPOSURE_DRIFT, f"luma swings {analysis.exposure_range:.0f}")
    if analysis.flicker > FLICKER_STDDEV:
        found.add(IssueCode.EXPOSURE_DRIFT, f"flicker sigma {analysis.flicker:.1f}", certainty=0.6)

    if analysis.black_frames and analysis.black_frames == len(analysis.samples):
        found.add(IssueCode.EMPTY_FRAME, "every sampled frame is black")
    elif analysis.black_frames:
        found.add(IssueCode.NEEDS_TRIM, f"{analysis.black_frames} black sample(s)")

    if info.audio.present and info.audio.channels == 0:
        found.add(IssueCode.UNUSABLE_AUDIO, "audio stream declares no channels")

    return found


def _segment_summary(analysis: VideoAnalysis) -> str:
    best = analysis.longest_segment
    return f"{best.start:.1f}-{best.end:.1f}s ({best.duration:.1f}s)" if best else "unknown"


# --- helpers ----------------------------------------------------------------


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ratio(value) -> float:
    """ffprobe frame rates arrive as '30000/1001'."""
    if not value:
        return 0.0
    text = str(value)
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            den = float(denominator)
            return round(float(numerator) / den, 4) if den else 0.0
        except ValueError:
            return 0.0
    return _float(text)
