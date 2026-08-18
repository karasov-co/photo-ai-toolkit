"""The vertical slice: a directory in, decisions and reports out.

Ordered so that the cheapest thing that can eliminate work happens first, and
so that a user sees results while the run is still going:

    1. discover        extension + checksum + sidecar grouping     ~free
    2. measure         decode, technical metrics, preview search   local, no cost
    3. cluster         near-duplicates, best frame per cluster     local
    4. semantic        vision model, opt-in                        paid
    5. score           ten dimensions per asset                    free
    6. select          diversity-aware flagship pass               free
    7. market          platform eligibility and metadata           free
    8. propose         filesystem actions, written but not run     free

Steps 5 through 8 read only stored numbers, which is what makes
`reclassify()` possible: change a threshold and redo the routing in
milliseconds, without decoding a pixel or spending a token.

The semantic step is optional and off by default. Everything before it is
deterministic, local, and free, and the tool produces a complete, useful,
explainable result without it -- at reduced confidence, which the confidence
score states rather than hides. That is deliberate: a culling tool that cannot
run without a network and an API key is not a culling tool.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from photoai import (
    batches,
    curation,
    duplicates,
    edit_recipe,
    llm_provider,
    marketplaces,
    media,
    raw_measurements,
    reports,
    scoring,
    stock_metadata,
    technical_filter,
)
from photoai import issues as issues_module
from photoai import provenance as provenance_module
from photoai import quarantine as quarantine_module
from photoai.calibration import CalibrationSet, resolve
from photoai.exif_reader import extract_exif
from photoai.preview_generator import PreviewGenerationError, generate_preview, preview_name
from photoai.reports import AssetRecord
from photoai.scoring import RouteClass, ScoreInput, Semantic

logger = logging.getLogger(__name__)

CACHE_NAME = "analysis_cache.json"
PREVIEW_DIRNAME = "previews"

# Which class each route maps onto in the output tree.
CLASS_DIRS = {
    RouteClass.TRASH: Path("trash_quarantine"),
    RouteClass.REVIEW: Path("manual_review"),
    RouteClass.STOCK_STANDARD: Path("stock/standard"),
    RouteClass.STOCK_STRONG: Path("stock/strong"),
    RouteClass.FLAGSHIP: Path("portfolio/flagship"),
}


# The API calls are network-bound, so threads are the right tool: the GIL is
# released for the whole of a request and four in flight is four times less
# waiting. Four rather than more because a rate limit costs more than it saves,
# and because the archive this was measured on is one person's shoot, not a
# fleet.
DEFAULT_CONCURRENCY = 4

# 429s are not failures, they are the server pacing the client. Retrying
# immediately is how a burst becomes a ban.
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 2.0


def _is_rate_limited(e: Exception) -> bool:
    text = str(e).lower()
    return (
        getattr(e, "status_code", None) == 429
        or "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    )


def _with_backoff(call, *, what: str):
    """Run `call`, backing off exponentially while the server says 429.

    Deliberately narrow: only a rate limit is retried here. Everything else --
    a bad key, an exhausted balance, a malformed reply -- is somebody else's
    decision, and retrying it is how a run spends money on the same error
    thirty-one times.
    """
    import random
    import time

    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return call()
        except Exception as e:
            if not _is_rate_limited(e) or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            # Full jitter: without it, four threads throttled together retry
            # together, and the burst that caused the 429 repeats on a timer.
            delay = RATE_LIMIT_BASE_DELAY * (2**attempt) * (0.5 + random.random() / 2)
            logger.warning(
                "%s rate-limited; waiting %.1fs (attempt %d of %d)",
                what, delay, attempt + 1, RATE_LIMIT_RETRIES,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _in_parallel(work: list, run, *, concurrency: int, what: str):
    """`run` over `work`, in order, with `concurrency` calls in flight.

    Results come back in the order of `work`, not the order they finished, so
    nothing downstream can depend on which group happened to be quickest. One
    worker runs inline: a pool of one costs a thread and buys nothing, and it
    is what to use when a traceback matters more than the wall clock.

    An account-level failure stops the remaining submissions dead. Sequentially
    a revoked key cost exactly one call; with four in flight it costs at most
    four, because three were already on the wire when the first came back. It
    does not cost all thirty-one, which is what this is guarding.
    """
    from concurrent.futures import ThreadPoolExecutor

    stop: list[bool] = [False]
    if concurrency <= 1 or len(work) <= 1:
        return [_result(run, item, what, stop) for item in work]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda item: _result(run, item, what, stop), work))


def _result(run, item, what, stop):
    """One unit of work, with its exception carried rather than raised.

    Raised inside a pool, an exception loses the rest of the batch: work already
    paid for is thrown away with the map. Carried, it reaches the caller in
    order, and the caller decides whether it ends the run.
    """
    from photoai import bootstrap

    if stop[0]:
        return (None, bootstrap.SemanticUnavailable("the run already stopped", kind="stopped"))
    try:
        return (_with_backoff(lambda: run(item), what=what), None)
    except Exception as e:  # noqa: BLE001 - re-raised by the caller, in order
        if bootstrap.is_fatal_api_error(e):
            stop[0] = True
        return (None, e)



@dataclass
class PipelineOptions:
    input_dir: Path
    output_dir: Path
    quarantine_dir: Path | None = None
    internal_dir: Path | None = None
    language: str = "en"
    profile_name: str | None = None
    profile_path: Path | None = None
    semantic: bool = False
    semantic_model: str | None = None
    # Without this, a semantic failure ends the run. A report that looks like a
    # full analysis but had no content check is worse than no report.
    allow_semantic_fallback: bool = False
    include_video: bool = True
    video_samples: int = 9
    force: bool = False
    limit: int | None = None
    copyright_holder: str = ""
    follow_symlinks: bool = False
    # The darkroom pass renders candidate edits, which costs about a second per
    # frame, so it is opt-in and by default runs only on frames worth editing.
    # Stage 3 is the artistic read. It is what makes a HERO promotion possible at
    # all, so it runs whenever `--semantic` does unless explicitly disabled.
    stage3: bool = True
    stage3_model: str | None = None
    darkroom: bool = False
    darkroom_renderer: str | None = None
    conservative_art: bool = True
    shadow_mode: bool = True
    # Which photographs the insights page describes: the ones this run analysed,
    # or everything ever stored here. "new" by default -- adding a batch to an
    # archive should tell you about the batch, not repeat what the archive as a
    # whole has always said.
    insights_scope: str = "new"
    # Worker processes for the decode pass. None means "one per core, minus
    # one"; 1 keeps the single-process path, which is what to use when a file
    # is crashing the run and you need a traceback rather than a dead pool.
    jobs: int | None = None
    # How hard the model is asked to think. "low" by default and forwarded to
    # the provider: it used to be dropped on the way to xAI, so every call
    # reasoned at the vendor's default and the bill was two and a half times
    # the quote. None means "do not send the parameter at all".
    reasoning: str | None = "low"
    # API calls in flight at once. Separate from `jobs`, which is local decode:
    # one is network latency, the other is CPU, and a machine with four cores
    # has no business making four API calls for that reason.
    concurrency: int = DEFAULT_CONCURRENCY

    def resolved_quarantine(self) -> Path:
        """The *physical* quarantine directory, deliberately not the farm folder.

        These used to be the same path. `build_class_farm` writes navigation
        symlinks into `<output>/trash_quarantine`, and quarantine planning used
        the same directory as its destination -- so a link pointing back at the
        source sat exactly where the file was about to be moved, and containment
        refused the move. Keeping the physical store separate removes the
        collision at its source rather than teaching the check to tolerate it.
        """
        if self.quarantine_dir:
            return self.quarantine_dir
        # An older run put this at the output root. Keep using it when it is
        # there rather than starting a second store somewhere else and leaving
        # the first one orphaned with files inside it.
        legacy = self.output_dir / "quarantine"
        if legacy.is_dir():
            return legacy
        return self.internal / "quarantine"

    @property
    def internal(self) -> Path:
        """Everything the tool keeps for itself: cache, previews, diagnostics.

        Hidden so that the output directory shows a photographer their
        photographs rather than the software's filing system. Nothing is
        withheld -- the full JSON, the previews and the log are all in here,
        and every other command reads them from here.
        """
        return self.internal_dir or (self.output_dir / ".internal")


@dataclass
class RunResult:
    records: list[AssetRecord] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    planned_operations: list = field(default_factory=list)
    calibration: CalibrationSet = field(default_factory=CalibrationSet)
    cancelled: bool = False
    semantic_requested: bool = False
    semantic_completed: bool = False
    semantic_model: str = ""
    semantic_error: str = ""
    stage3_completed: int = 0
    stage3_failed: int = 0
    stage3_skipped: int = 0
    # How many requests each stage actually made, and how many assets were
    # answered from store. Printed at the end, because "252 analysed, 47 reused,
    # 26 calls" is the only honest account of what a run cost.
    llm_calls: dict = field(default_factory=dict)
    # Tokens and dollars, measured from the provider's own usage figures rather
    # than estimated from a per-photo constant.
    usage_total: dict = field(default_factory=dict)
    new_assets: list = field(default_factory=list)
    reused_assets: list = field(default_factory=list)
    modified_assets: list = field(default_factory=list)
    run_id: str = ""
    # Keyed by asset key. Kept on the result so that everything written after
    # the run -- edit recipes, insights -- can use what was measured without
    # decoding a second time.
    measurements: dict = field(default_factory=dict)
    # Which of the two similarity measurements the diversity pass actually used,
    # and why, when it was not the better one. "perceptual_hash" is the honest
    # default and the fallback both; a run must never leave which one it was to
    # be inferred from whether onnxruntime happens to be installed.
    similarity_mode: str = "perceptual_hash"
    similarity_detail: str = ""

    @property
    def analysis_mode(self) -> str:
        if not self.semantic_requested:
            return "local_only"
        if self.semantic_completed:
            return "local_and_semantic"
        return "local_only_after_semantic_failure"

    def by_class(self, route_class: RouteClass) -> list[AssetRecord]:
        return [r for r in self.records if r.route_class == route_class.value]


# --- the analysis cache -----------------------------------------------------


class AnalysisCache:
    """Keyed by content checksum *and* analyzer version.

    Both halves matter. Without the checksum a renamed file is re-analyzed;
    without the version a code change silently keeps serving results the old
    code produced, which is the harder bug to notice.
    """

    def __init__(self, path: Path, *, version: str = scoring.ANALYZER_VERSION) -> None:
        self.path = path
        self.version = version
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("analyzer_version") == self.version:
                self._data = payload.get("entries") or {}
            else:
                logger.info(
                    "Analyzer version changed (%s -> %s); discarding cache",
                    payload.get("analyzer_version"),
                    self.version,
                )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read analysis cache: %s", e)

    def key(self, checksum: str) -> str:
        return f"{checksum}:{self.version}"

    @staticmethod
    def is_local_only(payload: dict) -> bool:
        """Cached entries never carry semantic state, by construction."""
        return "semantic" not in payload

    # --- Stage 3, keyed apart on purpose ------------------------------------
    #
    # A shared key would let a valid Stage 2 entry answer "already analysed" for
    # a frame whose artistic read never ran. Stage 3 is keyed by checksum, model
    # AND prompt version, so changing the prompt invalidates exactly the work
    # whose meaning changed.

    def get_stage3(
        self, checksum: str, model: str, reasoning: str | None = "low"
    ) -> dict | None:
        from photoai import stage3

        return self._data.get(stage3.cache_key(checksum, model, reasoning))

    def put_stage3(
        self, checksum: str, model: str, payload: dict, reasoning: str | None = "low"
    ) -> None:
        from photoai import stage3

        self._data[stage3.cache_key(checksum, model, reasoning)] = payload

    # Stage 2 is cached per asset for a reason that is not about cost.
    #
    # The axes are ranks stitched into percentiles across whatever was in the
    # run, so re-ranking an unchanged photograph next to 252 new ones changes
    # its genre, its axes, its score and possibly its category -- while nothing
    # about the photograph changed. A stored result is that asset's answer,
    # fixed at the moment it was analysed, and a later batch cannot rewrite it.

    def semantic_key(self, checksum: str, model: str, reasoning: str | None = "low") -> str:
        from photoai import prompts

        # Model, prompt version AND reasoning effort. The effort was missing:
        # a run at "low" served answers a run at "high" had paid for, and the
        # two are different answers from a different amount of thinking.
        return (
            f"stage2:{checksum}:{model}:{prompts.STAGE2_PROMPT_VERSION}:{reasoning or 'none'}"
        )

    def get_semantic(
        self, checksum: str, model: str, reasoning: str | None = "low"
    ) -> dict | None:
        return self._data.get(self.semantic_key(checksum, model, reasoning))

    def put_semantic(
        self, checksum: str, model: str, payload: dict, reasoning: str | None = "low"
    ) -> None:
        self._data[self.semantic_key(checksum, model, reasoning)] = payload

    # --- semantic vectors, keyed by checksum AND model id -------------------
    #
    # The model id has to be in the key for the same reason the analyzer version
    # is in the one above: a vector from one encoder is not comparable with a
    # vector from another, and serving a stored ViT-B/32 vector to a run using a
    # different tower would produce cosines that mean nothing. Not keyed on the
    # analyzer version, though -- a scoring change does not alter what a
    # photograph looks like, and re-encoding an archive is a minute of CPU.

    @staticmethod
    def embedding_key(checksum: str, model_id: str) -> str:
        return f"embedding:{checksum}:{model_id}"

    def get_embedding(self, checksum: str, model_id: str) -> tuple[float, ...] | None:
        stored = self._data.get(self.embedding_key(checksum, model_id))
        if not stored:
            return None
        vector = stored.get("vector")
        return tuple(float(v) for v in vector) if vector else None

    def put_embedding(self, checksum: str, model_id: str, vector) -> None:
        if not vector:
            return
        # Six decimals on a unit vector is under a part in 10^5 of angle, which
        # is far below anything the similarity threshold can see, and it roughly
        # halves the size of the cache file.
        self._data[self.embedding_key(checksum, model_id)] = {
            "vector": [round(float(v), 6) for v in vector],
            "dimensions": len(vector),
        }

    def get(self, checksum: str) -> dict | None:
        return self._data.get(self.key(checksum))

    def put(self, checksum: str, payload: dict) -> None:
        self._data[self.key(checksum)] = payload

    def known_checksums(self) -> set[str]:
        """Every checksum this cache holds a measurement for."""
        suffix = f":{self.version}"
        return {k[: -len(suffix)] for k in self._data if k.endswith(suffix)}

    def has_full_result(self, checksum: str, model: str) -> bool:
        """Whether this asset can be served entirely from store.

        All three parts, not one: a measurement without a content check would
        be reused as though it had been analysed, and the photograph would show
        up in the report with no genre and no artistic read while the summary
        counted it as reused.
        """
        if self.get(checksum) is None:
            return False
        if self.get_semantic(checksum, model) is None:
            return False
        return self.get_stage3(checksum, model) is not None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps({"analyzer_version": self.version, "entries": self._data}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)


# --- step 2: local measurement ----------------------------------------------


@dataclass
class Measurement:
    """Everything the local pass learns. Cacheable, JSON-round-trippable."""

    quality: float = 0.0
    uplift: float = 0.0
    recipe: list[str] = field(default_factory=list)
    crop_keep: float = 1.0
    phash: str = ""
    blur_ratio: float = 0.0
    clipped_highlights: float = 0.0
    clipped_shadows: float = 0.0
    # Where the clipping figures above came from. A rendered preview has already
    # spent whatever headroom the RAW held, so calling its numbers "RAW ground
    # truth" to a model is a lie the model has no way to catch.
    measurement_domain: str = "rendered_image"
    raw_available: bool = False
    raw_highlight_headroom_stops: float = 0.0
    raw_shadow_headroom_stops: float = 0.0
    raw_clipped_any_channel: float = 0.0
    raw_clipped_all_channels: float = 0.0
    raw_noise_floor_fraction: float = 0.0
    raw_measurement_version: str = ""
    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    noise: float = 0.0
    mean_luma: float = 0.0
    stddev_luma: float = 0.0
    channel_means: tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration: float = 0.0
    container: str = ""
    video: dict = field(default_factory=dict)
    exif: dict = field(default_factory=dict)
    preview_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["channel_means"] = list(self.channel_means)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> Measurement:
        data = dict(payload)
        means = data.get("channel_means") or [0.0, 0.0, 0.0]
        data["channel_means"] = tuple(means)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def measure_photo(asset: media.Asset, previews_dir: Path) -> Measurement:
    """Decode once; every local signal comes off that one decode."""
    out = Measurement()
    try:
        image = media.open_photo(asset.path)
    except media.UnreadableMedia as e:
        out.error = str(e)
        return out

    out.width, out.height = image.size
    out.megapixels = media.megapixels(*image.size)

    report = technical_filter.analyze(image)
    out.phash = report.phash
    out.blur_ratio = round(report.blur_ratio, 3)

    out.clipped_highlights = round(report.clipped_highlights, 4)
    out.clipped_shadows = round(report.clipped_shadows, 4)

    work = image.copy()
    work.thumbnail((edit_recipe.WORK_PX, edit_recipe.WORK_PX), 1)
    array = np.asarray(work.convert("RGB"), dtype=np.float64)
    luma = array @ np.array([0.299, 0.587, 0.114])
    out.mean_luma = round(float(luma.mean()), 2)
    out.stddev_luma = round(float(luma.std()), 2)
    out.channel_means = tuple(round(float(m), 2) for m in array.reshape(-1, 3).mean(axis=0))
    out.noise = edit_recipe.estimate_noise(array)
    out.quality = edit_recipe.frame_quality(array)

    recipe = edit_recipe.search(
        image, is_raw=asset.is_raw, noisy=out.noise > issues_module.NOISE_MILD
    )
    out.uplift = round(recipe.uplift, 2)
    out.recipe = recipe.human_readable()
    out.crop_keep = recipe.crop_keep_fraction

    # The sensor plane, not the render. Stage 2 is told which of the two it is
    # looking at, and the prompt words itself accordingly.
    raw_stats = raw_measurements.measure_or_empty(asset.path, asset.is_raw)
    out.raw_available = raw_stats.available
    if raw_stats.available:
        out.measurement_domain = "raw_sensor"
        out.raw_highlight_headroom_stops = raw_stats.highlight_headroom_stops
        out.raw_shadow_headroom_stops = raw_stats.shadow_headroom_stops
        out.raw_clipped_any_channel = raw_stats.clipped_any_channel
        out.raw_clipped_all_channels = raw_stats.clipped_all_channels
        out.raw_noise_floor_fraction = raw_stats.noise_floor_fraction
        out.raw_measurement_version = raw_measurements.MEASUREMENT_VERSION

    out.exif = extract_exif(asset.path, _legacy_file_type(asset))

    try:
        out.preview_path = str(generate_preview(asset.path, _legacy_file_type(asset), previews_dir))
    except PreviewGenerationError as e:
        logger.warning("Preview failed for %s: %s", asset.filename, e)

    image.close()
    return out


def _legacy_file_type(asset: media.Asset) -> str:
    """The three-way string the existing exif/preview modules expect."""
    fmt = asset.format
    if fmt is media.PhotoFormat.RAW:
        return "RAW"
    if fmt is media.PhotoFormat.TIFF:
        return "TIFF"
    return "JPEG"


def measure_video(asset: media.Asset, previews_dir: Path, *, samples: int = 9) -> Measurement:
    """Probe, sample across the timeline, keep the best frame as the poster."""
    from photoai import video_analyzer

    out = Measurement()
    scratch = Path(tempfile.mkdtemp(prefix="pat_clip_"))
    try:
        analysis = video_analyzer.analyze_video(asset.path, sample_count=samples, work_dir=scratch)
    except video_analyzer.FFmpegMissing as e:
        shutil.rmtree(scratch, ignore_errors=True)
        out.error = str(e)
        return out
    except Exception as e:
        shutil.rmtree(scratch, ignore_errors=True)
        out.error = f"video analysis failed: {e}"
        return out

    try:
        info = analysis.probe
        out.width, out.height = info.width, info.height
        out.megapixels = info.megapixels
        out.duration = round(info.duration, 3)
        out.container = info.container
        out.quality = analysis.mean_quality
        out.phash = ""

        if analysis.samples:
            out.blur_ratio = round(
                sum(s.blur_ratio for s in analysis.samples) / len(analysis.samples), 3
            )
            out.clipped_highlights = round(
                max(s.clipped_highlights for s in analysis.samples), 4
            )
            out.clipped_shadows = round(max(s.clipped_shadows for s in analysis.samples), 4)
            out.mean_luma = round(
                sum(s.mean_luma for s in analysis.samples) / len(analysis.samples), 2
            )

        best = analysis.longest_segment
        out.video = {
            "codec": info.codec,
            "profile": info.profile,
            "frame_rate": info.frame_rate,
            "variable_frame_rate": info.is_variable_frame_rate,
            "bit_rate": info.bit_rate,
            "pix_fmt": info.pix_fmt,
            "color_transfer": info.color_transfer,
            "log_or_hdr": info.looks_log_or_hdr,
            "orientation": info.orientation,
            "slow_motion_hint": info.slow_motion_hint,
            "time_lapse_hint": info.time_lapse_hint,
            "audio": {
                "present": info.audio.present,
                "codec": info.audio.codec,
                "channels": info.audio.channels,
                "sample_rate": info.audio.sample_rate,
            },
            "mean_quality": analysis.mean_quality,
            "focus_consistency": analysis.focus_consistency,
            "exposure_range": analysis.exposure_range,
            "flicker": analysis.flicker,
            "camera_movement": analysis.motion.camera_movement,
            "pan_magnitude": analysis.motion.pan_magnitude,
            "jitter": analysis.motion.jitter,
            "rolling_shutter_shear": analysis.motion.shear,
            "black_frames": analysis.black_frames,
            "frozen_frames": analysis.frozen_frames,
            "segments": [
                {"start": s.start, "end": s.end, "duration": s.duration, "quality": s.mean_quality}
                for s in analysis.segments
            ],
            "best_segment": (
                {"start": best.start, "end": best.end, "duration": best.duration} if best else None
            ),
            "poster_timestamp": analysis.poster_timestamp,
            "sample_count": len(analysis.samples),
        }
        out.video["issues"] = issues_module.summarise(video_analyzer.detect_video_issues(analysis))

        if analysis.poster_path and Path(analysis.poster_path).exists():
            previews_dir.mkdir(parents=True, exist_ok=True)
            poster = previews_dir / preview_name(asset.path)
            shutil.copy2(analysis.poster_path, poster)
            out.preview_path = str(poster)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return out


def issues_for(asset: media.Asset, measurement: Measurement) -> issues_module.IssueSet:
    """Typed issues from a measurement, for either media type."""
    if measurement.error:
        found = issues_module.IssueSet()
        found.add(issues_module.IssueCode.CORRUPT_FILE, measurement.error)
        return found

    if asset.kind is media.MediaKind.VIDEO:
        found = issues_module.IssueSet()
        stored = measurement.video.get("issues") or {}
        for group, codes in (
            ("fixable", issues_module.Fixability.FIXABLE),
            ("partially_fixable", issues_module.Fixability.PARTIAL),
            ("unrecoverable", issues_module.Fixability.UNRECOVERABLE),
        ):
            for described in stored.get(group, []):
                code_name, _, detail = str(described).partition(": ")
                try:
                    found.add(issues_module.IssueCode(code_name), detail)
                except ValueError:
                    logger.debug("Unknown stored issue code %r", code_name)
            del codes
        return found

    class _Report:
        blur_ratio = measurement.blur_ratio
        clipped_highlights = measurement.clipped_highlights
        clipped_shadows = measurement.clipped_shadows

    return issues_module.detect_photo_issues(
        _Report(),
        megapixels=measurement.megapixels,
        mean_luma=measurement.mean_luma,
        stddev_luma=measurement.stddev_luma,
        channel_means=measurement.channel_means,
        noise_estimate=measurement.noise,
        is_raw=asset.is_raw,
    )


# --- step 4: the optional paid pass -----------------------------------------


def semantic_pass(
    assets: list[media.Asset],
    measurements: dict[str, Measurement],
    *,
    model: str,
    client=None,
    group_size: int = 12,
    cache: AnalysisCache | None = None,
    calls: dict | None = None,
    reasoning: str | None = "low",
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Semantic]:
    """Rank frames against each other in groups, then stitch to a global order.

    Reuses the prompts, group builder and Bradley-Terry aggregation that already
    exist in this repository. Absolute scoring is not used and is not an option:
    every live absolute call made against this archive returned 548, 560, 694,
    762 -- a scale that does not discriminate is not a scale.
    """

    from photoai import aggregate, batch_runner, prompts

    # No client is built here. It used to fall back to `bootstrap.make_client()`,
    # which returns an OpenAI client with no base_url -- hard-wired to
    # api.openai.com. `_provider` then saw a non-None client and wrapped it in
    # OpenAIProvider, so a run configured for grok sent an xai- key to OpenAI
    # and collected a 401 per group. The preflight went through
    # `llm_provider.build` and reached x.ai correctly, which is why the two
    # disagreed. Leaving this None lets `_provider` build from configuration.
    by_name = {a.key: a for a in assets}
    out: dict[str, Semantic] = {}
    # Raw model replies, keyed by asset. Both the ones answered from store and
    # the ones just fetched end up here, and the scale is computed over all of
    # them together so that one report never mixes two populations.
    cached_raw: dict[str, dict] = {}
    photo_names: list[str] = []

    for asset in assets:
        if asset.kind is not media.MediaKind.PHOTO:
            continue
        if not measurements.get(asset.key, Measurement()).preview_path:
            continue
        stored = (
            cache.get_semantic(asset.checksum, model, reasoning) if cache is not None else None
        )
        if stored is not None:
            cached_raw[asset.key] = stored
            continue
        photo_names.append(asset.key)

    if not photo_names:
        return _stitch(cached_raw, by_name, out)
    groups = aggregate.build_groups(photo_names, size=group_size)
    parsed_groups: list[list[dict]] = []
    per_frame: dict[str, dict] = {}
    # A per-group failure is survivable; every group failing is not. An
    # authentication or model error hits all of them identically, and swallowing
    # it per group turned a hard failure into a silent empty result.
    first_error: Exception | None = None
    succeeded = 0

    # The loop is wrapped rather than left bare: a fatal API error, a Ctrl-C or
    # a bug in here all used to end the process with every answer bought in this
    # pass still only in memory. `cache.save()` ran once, after both passes
    # completed, which is the one moment a failing run never reaches.
    # The frames are built up front, in one thread: reading previews and
    # base64-encoding them is local work, and doing it inside the pool would put
    # file I/O and API latency in the same budget.
    payloads = [_stage2_frames(group, measurements) for group in groups]

    def ask(frames):
        return _provider(model, client).complete_vision(
            llm_provider.from_openai_content(
                prompts.STAGE2_SYSTEM,
                prompts.stage2_user_content(frames),
                max_tokens=900 + 260 * len(frames),
                reasoning_effort=reasoning,
                stage="stage2",
            )
        )

    try:
        replies = _in_parallel(payloads, ask, concurrency=concurrency, what="Stage 2")
        for index, (group, (text, error)) in enumerate(zip(groups, replies, strict=True)):
            if error is not None:
                from photoai import bootstrap

                # An account-level failure hits every remaining group identically.
                # Without this, a wrong key produced thirty-one identical 401s --
                # one per group, each logged in full -- before the run gave up. On
                # a paid key the same loop would keep hammering a rate limit or an
                # exhausted balance to the end of the archive.
                if bootstrap.is_fatal_api_error(error):
                    kind, message = bootstrap.classify_api_error(error)
                    logger.error("Semantic pass stopped at group %d: %s", index, message)
                    raise bootstrap.SemanticUnavailable(message, kind=kind) from error
                first_error = first_error or error
                logger.error(
                    "Semantic group %d failed: %s", index, reports.redact(str(error))
                )
                continue

            try:
                items = batch_runner.parse_group_json(text)
            except Exception as e:
                first_error = first_error or e
                logger.error("Semantic group %d failed: %s", index, reports.redact(str(e)))
                continue
            succeeded += 1
            if calls is not None:
                calls["stage2"] = calls.get("stage2", 0) + 1

            # A near-ranking is repaired before it is judged. Discarding a group
            # costs twelve photographs their genre, their faces and their subject
            # strength -- and on this archive a single duplicated rank did exactly
            # that to eight frames, taking the portrait gate with it. The repair is
            # recorded, never silent.
            items, repairs = batch_runner.repair_group_ranks(items, len(group))
            if repairs:
                logger.info("Group %d repaired: %s", index, "; ".join(repairs))

            problems = batch_runner.validate_group_ranks(items, len(group))
            if problems:
                # Still not a ranking. Feeding it to Bradley-Terry would manufacture
                # a confident order out of a malformed reply, so the group is
                # dropped and its frames simply go unranked -- which lowers their
                # confidence and sends them to review, the safe direction.
                logger.warning("Group %d rejected: %s", index, "; ".join(problems))
                continue

            placed = batch_runner.attach_filenames(items, group)
            parsed_groups.append(placed)
            for item in placed:
                per_frame.setdefault(item["filename"], item)
                item["_group_size"] = len(group)

            _store_group(placed, by_name, cached_raw, cache, model, reasoning)
            # Every third group, not at the end. The end is exactly where a run
            # stops when a balance runs out, and everything paid for up to that
            # point used to be discarded with the process.
            if cache is not None and (index + 1) % CACHE_SAVE_EVERY == 0:
                cache.save()

    except KeyboardInterrupt:
        logger.warning("Interrupted; keeping the %d group(s) already paid for", succeeded)
        raise
    finally:
        if cache is not None:
            cache.save()

    if first_error is not None and succeeded == 0:
        # Nothing got through. Re-raise so the caller decides whether that ends
        # the run, rather than returning an empty dict that reads like "the
        # model had no opinion".
        raise first_error

    return _stitch(cached_raw, by_name, out)


def _stage2_frames(group, measurements) -> list[dict]:
    """The per-frame payload for one group: the preview, plus what was measured."""
    import base64

    frames = []
    for name in group:
        measurement = measurements[name]
        with open(measurement.preview_path, "rb") as f:
            encoded = base64.standard_b64encode(f.read()).decode()
        # For a RAW, hand over what the *sensor* saturated at. For anything else,
        # hand over the rendered figure and say so. The two mean different things
        # and the prompt must not conflate them.
        if measurement.raw_available:
            highlights = measurement.raw_clipped_all_channels
        else:
            highlights = measurement.clipped_highlights
        frames.append(
            {
                "filename": name,
                "clipped_highlights": highlights,
                "clipped_shadows": measurement.clipped_shadows,
                "measurement_domain": measurement.measurement_domain,
                "headroom_stops": measurement.raw_highlight_headroom_stops,
                "encoded": encoded,
            }
        )
    return frames


def _store_group(placed, by_name, cached_raw, cache, model, reasoning) -> None:
    """One parsed group, into memory and into the cache, immediately.

    Store the raw reply and the group it was ranked in, never the stitched
    result. Percentiles are a property of the population, so caching them
    freezes one run's population into an asset that outlives it -- and the next
    run then puts those old percentiles beside percentiles computed over a
    different set, in one report, on one axis, as though they were the same
    scale. The ranks are the durable fact; the scale is rebuilt later.

    This used to run after every group had been ranked. A run that died on group
    nineteen of thirty-one therefore paid for eighteen groups and kept none.
    """
    members = [item.get("filename", "") for item in placed]
    for item in placed:
        name = item.get("filename", "")
        asset = by_name.get(name)
        if asset is None:
            continue
        entry = {"item": item, "group": members}
        cached_raw.setdefault(name, entry)
        if cache is not None:
            cache.put_semantic(asset.checksum, model, entry, reasoning)


def _measure_one(args):
    """One asset, measured. Top-level so a process pool can pickle it."""
    asset, previews_dir, video_samples = args
    if asset.kind is media.MediaKind.VIDEO:
        return measure_video(asset, previews_dir, samples=video_samples)
    return measure_photo(asset, previews_dir)


# How often a pass flushes the cache to disk, in groups. Three rather than one
# because a save is a full JSON rewrite, and three rather than thirty because
# the thing being protected is money already spent.
CACHE_SAVE_EVERY = 3


# Below this, a process pool costs more than it saves: starting an interpreter
# and pickling the work is tens of milliseconds per worker, and a handful of
# photographs are decoded in less than that. It also keeps small runs -- and
# every test -- in one process, where a traceback is a traceback.
PARALLEL_THRESHOLD = 24


def _measure_all(pending, previews_dir, options, progress, total):
    """Measure everything not already in store, in parallel where it helps.

    Decoding is where a run spends its time and it is embarrassingly parallel:
    each photograph is independent, and the results are written into a dict
    keyed by asset afterwards, so nothing depends on the order they finish in.

    `--jobs 1` keeps the old single-process path exactly, which matters because
    a process pool turns a crash in one file into a broken pool rather than one
    bad measurement -- and when something is wrong, being able to run the
    sequential version is how it gets diagnosed.
    """
    jobs = _worker_count(options.jobs, len(pending))
    if jobs <= 1 or len(pending) < PARALLEL_THRESHOLD:
        for index, asset in pending:
            if progress:
                progress(asset.filename, index, total, False)
            yield index, asset, _measure_one((asset, previews_dir, options.video_samples))
        return

    from concurrent.futures import ProcessPoolExecutor

    logger.info("Measuring %d assets across %d processes", len(pending), jobs)
    payloads = [(asset, previews_dir, options.video_samples) for _, asset in pending]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        # Ordered, so progress reads the way it did and the run is reproducible.
        for (index, asset), measurement in zip(
            pending, pool.map(_measure_one, payloads, chunksize=1), strict=True
        ):
            if progress:
                progress(asset.filename, index, total, False)
            yield index, asset, measurement


def _worker_count(requested: int | None, work: int) -> int:
    """One fewer than the cores, so the machine stays usable during a long run."""
    if requested is not None and requested >= 1:
        return min(requested, max(1, work))
    return max(1, min(os.cpu_count() or 1, work))


def _stitch(raw: dict[str, dict], by_name: dict, out: dict[str, Semantic]) -> dict[str, Semantic]:
    """Rebuild one scale over every asset in the run, cached and fresh alike.

    This is the whole point of storing ranks instead of percentiles. The model
    ranks twelve frames against each other; turning that into a 0-100 figure
    requires a population, and the population changes every time somebody adds
    photographs. Recomputing here costs no API call and keeps a single report
    on a single scale.

    Groups are reconstructed from what was stored, so Bradley-Terry still sees
    the comparison structure it needs -- an asset remembers which frames it was
    ranked against, not merely where it came.
    """
    from photoai import aggregate
    from photoai import assessment_parser as routing

    if not raw:
        return out

    groups: dict[tuple, list[dict]] = {}
    for name, entry in raw.items():
        item = dict(entry.get("item") or {})
        item.setdefault("filename", name)
        members = entry.get("group") or [name]
        groups.setdefault(tuple(members), []).append(item)

    scores = aggregate.aggregate_all_axes(list(groups.values()))

    for name, entry in raw.items():
        item = entry.get("item") or {}
        try:
            assessment = routing.parse_assessment(item, name)
        except routing.AssessmentParseError as e:
            logger.warning("Unusable model output for %s: %s", name, e)
            continue
        members = entry.get("group") or []
        semantic = scoring.semantic_from_assessment(
            assessment, group_size=len(members) or item.get("_group_size")
        )
        semantic.axis_a = scores["axis_a"].get(name, semantic.axis_a)
        semantic.axis_b = scores["axis_b"].get(name, semantic.axis_b)
        semantic.axis_c = scores["axis_c"].get(name, semantic.axis_c)
        semantic.description = str(item.get("note") or "")
        out[name] = semantic
    return out


def stage3_pass(
    assets,
    measurements,
    routed: dict[str, str],
    artistic_hints: dict,
    *,
    model: str,
    client,
    cache: AnalysisCache | None = None,
    group_size: int = 6,
    reasoning: str | None = "low",
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict:
    """The artistic read, in small groups, with crops for anything with a face.

    Groups are smaller than Stage 2's twelve because each frame may carry three
    views and the reply is far longer per frame; a group of twelve portraits is
    thirty-six high-detail images and an output budget nobody should spend in
    one call.

    A per-group failure is recorded against every frame in that group as
    `FAILED` rather than dropped, so a frame can never end up looking like it
    was never a candidate when in fact the analysis broke.
    """

    from photoai import stage3 as stage3_module

    out: dict[str, stage3_module.ArtisticAssessment] = {}
    pending: list[str] = []

    for asset in assets:
        key = asset.key
        measurement = measurements.get(key)
        if measurement is None:
            continue

        hint = artistic_hints.get(key, {})
        needed, reason = stage3_module.should_run(
            route_class=routed.get(key, "review"),
            has_unrecoverable=bool(hint.get("has_unrecoverable")),
            intentionality_likelihood=int(hint.get("intentionality_likelihood", 50)),
            curatorial_uncertainty=int(hint.get("curatorial_uncertainty", 100)),
            faces_present=False,
            corrupt=bool(measurement.error),
        )
        if not needed:
            out[key] = stage3_module.ArtisticAssessment.not_required(reason)
            continue

        if cache is not None:
            cached = cache.get_stage3(asset.checksum, model, reasoning)
            if cached is not None:
                out[key] = stage3_module.ArtisticAssessment.from_dict(cached)
                continue
        pending.append(key)

    if not pending:
        return out

    from photoai import bootstrap

    by_key = {a.key: a for a in assets}
    done = 0

    # Every group's crops are cut before any call goes out, for the same reason
    # Stage 2 encodes up front: image work and network latency should not share
    # a budget.
    batches = []
    for start in range(0, len(pending), group_size):
        frames = []
        for key in pending[start : start + group_size]:
            preview = Path(measurements[key].preview_path)
            if not preview.exists():
                out[key] = stage3_module.ArtisticAssessment.skipped(
                    "no preview was generated"
                )
                continue
            views = _stage3_views(by_key[key], preview, artistic_hints.get(key, {}))
            frames.append({"key": key, "views": views, "encoded": views[0][1]})
        if frames:
            batches.append(frames)

    def ask(frames):
        return _stage3_call(
            frames, model=model, client=client, stage3_module=stage3_module,
            reasoning=reasoning,
        )

    try:
        replies = _in_parallel(batches, ask, concurrency=concurrency, what="Stage 3")
        for frames, (assessments, failure) in zip(batches, replies, strict=True):
            if failure is not None:
                # Stage 2 has had this check since a wrong key produced
                # thirty-one identical 401s. Stage 3 did not, so an exhausted
                # balance was retried once per group to the end of the archive.
                # An account-level failure hits every remaining group the same
                # way; the run ends and what it already bought is saved first.
                if not bootstrap.is_fatal_api_error(failure):
                    raise failure
                kind, message = bootstrap.classify_api_error(failure)
                logger.error("Stage 3 stopped after %d group(s): %s", done, message)
                raise bootstrap.SemanticUnavailable(message, kind=kind) from failure

            for frame in frames:
                key = frame["key"]
                assessment = assessments.get(key)
                out[key] = assessment or stage3_module.ArtisticAssessment.failed(
                    ["the model returned nothing usable for this frame"], model=model
                )
                if cache is not None and out[key].completed:
                    cache.put_stage3(
                        by_key[key].checksum, model, out[key].to_dict(), reasoning
                    )
            done += 1
            if cache is not None and done % CACHE_SAVE_EVERY == 0:
                cache.save()
    except KeyboardInterrupt:
        logger.warning("Interrupted; keeping the %d group(s) already paid for", done)
        raise
    finally:
        if cache is not None:
            cache.save()

    return out


def _stage3_views(asset, preview: Path, hint: dict) -> list[tuple[str, str]]:
    """Base64 views for one frame: the whole picture, plus crops when a face is in it."""
    import base64

    from photoai import stage3 as stage3_module

    def encode(image) -> str:
        import io

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=88)
        return base64.standard_b64encode(buffer.getvalue()).decode()

    with Image.open(preview) as opened:
        image = opened.convert("RGB")
        if not hint.get("faces_present"):
            return [("full frame", encode(image))]
        return [(name, encode(view)) for name, view in stage3_module.face_crops(image)]


def _provider(model: str, client=None):
    """The provider for this run, from configuration -- not from a stray client.

    An injected client is honoured only when the configured provider is the one
    it speaks. A client object carries an endpoint and a key inside it, and
    wrapping whatever turns up in `OpenAIProvider` silently overrode
    `PHOTO_AI_PROVIDER`: a run set to grok sent an xai- key to api.openai.com
    and got a 401 back for every group.

    The manufactured client is gone from both callers, so in a real run this is
    always None and the configuration decides. The check stays as the second
    line of defence, because the first one was "nobody will pass the wrong
    client" and that turned out to be exactly what the code did.
    """
    import os

    from photoai import bootstrap

    provider = bootstrap.resolve_provider()
    if client is not None and provider == llm_provider.OpenAIProvider.name:
        return llm_provider.OpenAIProvider(model, client=client)
    if client is not None:
        logger.debug(
            "Ignoring an injected client: this run is configured for %s", provider
        )
    return llm_provider.build(
        provider, model, base_url=os.environ.get("PHOTO_AI_BASE_URL", "")
    )


def _truncated(response) -> bool:
    """Whether the reply stopped because it ran out of output budget.

    Worth detecting specifically. A truncated reply presents as "no JSON array
    in the reply", which is indistinguishable from a model that answered in
    prose -- and the two need opposite responses. Prose is worth retrying;
    truncation is not, because an identical request truncates identically.
    """
    if str(getattr(response, "status", "")) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    return bool(details and "max_output_tokens" in str(getattr(details, "reason", details)))


def _stage3_call(frames, *, model, client, stage3_module, budget_factor: float = 1.0,
                 reasoning: str | None = "low"):
    """One group, with bounded retries on a malformed reply.

    Two failure modes, handled differently. A malformed reply is retried as-is,
    because models do occasionally wrap JSON in prose and asking again fixes it.
    A *truncated* reply splits the group instead: sending the same six frames
    again with the same budget produces the same truncation, and on this archive
    that cost two minutes and three times the tokens per group before failing.
    """
    from photoai import prompts

    group = [f["key"] for f in frames]
    errors: list[str] = []
    budget = int(
        (prompts.STAGE3_BASE_OUTPUT_TOKENS
         + prompts.STAGE3_MAX_OUTPUT_TOKENS_PER_FRAME * len(frames))
        * budget_factor
    )

    for attempt in range(stage3_module.MAX_RETRIES + 1):
        try:
            try:
                text = _provider(model, client).complete_vision(
                    llm_provider.from_openai_content(
                        prompts.STAGE3_SYSTEM,
                        prompts.stage3_user_content(frames),
                        max_tokens=budget,
                        reasoning_effort=reasoning,
                        stage="stage3",
                    )
                )
            except llm_provider.Truncated as e:
                errors.append(f"attempt {attempt + 1}: {e}")
                return _split_or_widen(
                    frames, model=model, client=client, stage3_module=stage3_module,
                    budget_factor=budget_factor, errors=errors, reasoning=reasoning,
                )
            parsed = stage3_module.parse_group(text, group, model=model)
            if parsed:
                for assessment in parsed.values():
                    assessment.retries = attempt
                return parsed
            errors.append(f"attempt {attempt + 1}: no usable object in the reply")
        except stage3_module.Stage3ParseError as e:
            errors.append(f"attempt {attempt + 1}: {e}")
        except Exception as e:
            from photoai import bootstrap

            # An account-level failure is not a bad group -- it is the end of
            # the run. On a live archive the balance ran out at photograph 154
            # and the remaining 145 were each recorded as an individual Stage 3
            # failure, so the run finished, exited zero, and published a report
            # in which half the photographs had no artistic read at all.
            if bootstrap.is_fatal_api_error(e):
                kind, message = bootstrap.classify_api_error(e)
                raise bootstrap.SemanticUnavailable(message, kind=kind) from e
            # Anything else is recorded against every frame in the group.
            errors.append(f"attempt {attempt + 1}: {reports.redact(str(e))}")
            logger.warning("Stage 3 group failed: %s", reports.redact(str(e)))
            break

    logger.error("Stage 3 gave up after %d attempt(s): %s", len(errors), "; ".join(errors[:2]))
    return {
        key: stage3_module.ArtisticAssessment.failed(
            errors, retries=len(errors) - 1, model=model
        )
        for key in group
    }


# How far the budget may be raised for a single frame that still truncates.
# Bounded, because the alternative to a bound is paying for a reply that never
# ends.
STAGE3_BUDGET_LIMIT = 3.0


def _split_or_widen(frames, *, model, client, stage3_module, budget_factor, errors,
                    reasoning: str | None = "low"):
    """Halve the group, or -- when it is already one frame -- widen the budget."""
    if len(frames) > 1:
        middle = len(frames) // 2
        logger.info("Stage 3 reply truncated; splitting %d frames into two", len(frames))
        out = {}
        for half in (frames[:middle], frames[middle:]):
            out.update(
                _stage3_call(
                    half, model=model, client=client, stage3_module=stage3_module,
                    budget_factor=budget_factor, reasoning=reasoning,
                )
            )
        return out

    widened = budget_factor * 2
    if widened <= STAGE3_BUDGET_LIMIT:
        logger.info("Stage 3 reply truncated on a single frame; widening the budget")
        return _stage3_call(
            frames, model=model, client=client, stage3_module=stage3_module,
            budget_factor=widened, reasoning=reasoning,
        )

    logger.error("Stage 3 truncated even at the widest budget: %s", "; ".join(errors[:2]))
    return {
        frame["key"]: stage3_module.ArtisticAssessment.failed(errors, model=model)
        for frame in frames
    }


# --- the run ----------------------------------------------------------------


def run(
    options: PipelineOptions,
    *,
    progress: Callable[..., None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    client=None,
) -> RunResult:
    """Discover, measure, cluster, score, select, and propose. Moves nothing."""
    from photoai import bootstrap

    calibration = resolve(options.profile_name, options.profile_path)
    model = bootstrap.resolve_model(options.semantic_model)

    # Before a single file is opened. The original failure surfaced only when
    # the client was constructed, which was after every photograph had been
    # decoded and measured -- minutes of work thrown away, and a run that then
    # carried on as though nothing had happened.
    if options.semantic and client is None and not bootstrap.has_credentials():
        raise bootstrap.SemanticCredentialsMissing(
            f"{bootstrap.API_KEY_VAR} is not set in the environment or in a .env file"
        )

    previews_dir = options.internal / PREVIEW_DIRNAME
    cache = AnalysisCache(options.internal / CACHE_NAME)

    # The output tree holds previews, reports and a symlink farm. Pointed
    # inside the input, a second run would otherwise treat its own 512px
    # previews as new photographs and route them -- and a preview routed to
    # trash is a proposal to delete a file the tool itself made.
    assets = media.discover(
        options.input_dir,
        follow_symlinks=options.follow_symlinks,
        exclude=media.excluded_roots(options.output_dir, options.resolved_quarantine()),
    )
    if not options.include_video:
        assets = [a for a in assets if a.kind is not media.MediaKind.VIDEO]
    if options.limit:
        assets = assets[: options.limit]

    result = RunResult(
        calibration=calibration,
        semantic_requested=options.semantic,
        semantic_model=model if options.semantic else "",
        run_id=batches.new_run_id(),
    )

    # Split before any work, so the progress output can distinguish the 252
    # photographs about to be analysed from the 47 that will be answered from
    # store. Printing all 299 as though each were being analysed is what made a
    # 15-minute incremental run look identical to a full one.
    history = batches.read_all(options.internal / batches.MANIFEST_NAME)
    fresh, modified, reused = batches.classify_assets(
        assets, cache, model, seen_keys=batches.seen_keys(history)
    )
    if options.force:
        fresh, modified, reused = assets, [], []
    result.new_assets = [a.checksum for a in fresh]
    result.modified_assets = [a.checksum for a in modified]
    result.reused_assets = [a.checksum for a in reused]
    measurements: dict[str, Measurement] = {}

    pending: list[tuple[int, media.Asset]] = []
    for index, asset in enumerate(assets, start=1):
        if should_cancel and should_cancel():
            result.cancelled = True
            logger.info("Cancelled after %d of %d assets", index - 1, len(assets))
            # Everything after this point indexes measurements by filename, so
            # the asset list has to shrink to what was actually measured --
            # otherwise cancelling raises a KeyError instead of returning the
            # partial results the user is entitled to.
            assets = assets[: index - 1]
            break

        reused_here = asset.checksum in set(result.reused_assets)
        cached = None if options.force else cache.get(asset.checksum)
        if cached is not None:
            if progress:
                progress(asset.filename, index, len(assets), reused_here)
            measurements[asset.key] = Measurement.from_dict(cached)
            continue
        pending.append((index, asset))

    for _, asset, measurement in _measure_all(
        pending, previews_dir, options, progress, len(assets)
    ):
        cache.put(asset.checksum, measurement.to_dict())
        measurements[asset.key] = measurement

    cache.save()

    clusters = _cluster(assets, measurements)
    # Before the semantic pass, so that a run without credentials still gets the
    # better diversity measurement: this one is local and costs no money.
    vectors, result.similarity_mode, result.similarity_detail = _embed(
        assets, measurements, cache
    )
    logger.info(
        "Diversity similarity: %s%s",
        result.similarity_mode,
        f" ({result.similarity_detail})" if result.similarity_detail else "",
    )
    semantics = _semantics(assets, measurements, options, client, result, model)

    # Stage 3 needs a provisional route to decide who is worth reading, so the
    # scoring pass runs twice: once to get candidates, then again with the
    # artistic evidence that can promote or block them.
    provisional = _score_all(
        assets, measurements, clusters, semantics, calibration, options, model,
        vectors=vectors,
    )
    stage3_results = _stage3(
        assets, measurements, provisional, semantics, options, client, result, model
    )

    result.records = _score_all(
        assets, measurements, clusters, semantics, calibration, options, model, stage3_results,
        vectors=vectors,
    )
    result.measurements = measurements
    _apply_policy(result.records, options)
    if options.darkroom:
        _darkroom_pass(assets, measurements, result.records, options)
    for record in result.records:
        record.analysis_mode = result.analysis_mode
        record.semantic_requested = result.semantic_requested
        record.semantic_completed = result.semantic_completed
        record.semantic_error = result.semantic_error
        record.similarity_mode = result.similarity_mode
        record.similarity_detail = result.similarity_detail

    # Drained here rather than per pass: one file, appended once, whatever
    # combination of stages ran. An empty list writes nothing.
    if llm_provider.USAGE:
        result.usage_total = llm_provider.write_usage(
            options.internal / "usage.jsonl", llm_provider.USAGE
        )
        llm_provider.USAGE.clear()

    _write_manifest(options, result, assets, model)
    result.planned_operations = _plan_operations(assets, result.records, options)
    result.summary = reports.summarise(
        result.records,
        recoverable_bytes=sum(
            a.size_bytes
            for a in assets
            if _record_for(result.records, a.key)
            and _record_for(result.records, a.key).route_class == RouteClass.TRASH.value
        ),
    )
    return result


def _apply_policy(records, options: PipelineOptions) -> None:
    """Assign a decision bucket to every record, with every gate recorded.

    Runs in shadow mode by default: the bucket says what the tool *would* do,
    and nothing acts on it. Turning that off requires evidence the monitor has
    to certify, which is the point.
    """
    from photoai import artistic, preference_model, selective_policy
    from photoai.model_monitoring import Monitor, Observation
    from photoai.preference_store import PreferenceStore

    store = PreferenceStore(options.internal / "preferences.jsonl")
    model = preference_model.fit(store)

    # The monitor is the tenth gate. Reading its state here is what connects the
    # `monitor` command to the pipeline: without it, switching automation off
    # after a false trash would have had no effect on the next run.
    monitor = Monitor(options.internal / "model_monitoring.json")
    health = monitor.evaluate()
    holdout = health["resolved_cases"]

    for record in records:
        scores = artistic.ArtisticScores(
            technical_integrity=record.artistic.get("technical_integrity", 0),
            intentionality_likelihood=record.artistic.get("intentionality_likelihood", 0),
            curatorial_uncertainty=record.artistic.get("curatorial_uncertainty", 100),
        )
        prediction = model.predict(
            record.asset_id, genre=record.genre, camera=record.camera
        )
        decision = selective_policy.decide(
            asset_id=record.asset_id,
            route_class=record.route_class,
            technical_evidence=record.evidence,
            prediction=prediction,
            model=model,
            artistic_scores=scores,
            genre=record.genre,
            is_best_in_cluster=record.best_in_cluster,
            cluster_size=record.cluster_size,
            shadow_mode=options.shadow_mode,
            holdout_checks=holdout,
            monitor_healthy=health["automation_enabled"],
        )
        record.decision_bucket = decision.bucket
        record.abstained = decision.abstained
        record.out_of_distribution = not prediction.in_distribution
        record.personal_preference_probability = (
            None if prediction.abstained else prediction.probability
        )
        record.curatorial_disagreement = preference_model.disagreement(
            prediction, scores.has_any_artistic_signal
        )
        record.policy_evidence = decision.reasons + decision.failed_gates

        # Every prediction becomes an observation. It stays unresolved until the
        # photographer does something that settles it -- a restore, an override,
        # a portfolio pick -- which is what closes the loop the monitor needs.
        monitor.observe(
            Observation(
                asset_id=record.asset_id,
                predicted=decision.bucket,
                confidence=(
                    0.0 if prediction.abstained else abs(prediction.probability - 0.5) * 2
                ),
                in_distribution=prediction.in_distribution,
                genre=record.genre,
                camera=record.camera,
            )
        )

    monitor.save()


def _darkroom_pass(assets, measurements, records, options: PipelineOptions) -> None:
    """Render candidate edits for the frames worth editing."""
    from photoai import artistic, darkroom

    by_key = {a.key: a for a in assets}
    for record in records:
        scores = artistic.ArtisticScores(
            intentionality_likelihood=record.artistic.get("intentionality_likelihood", 0),
            curatorial_uncertainty=record.artistic.get("curatorial_uncertainty", 100),
            signals=[
                artistic.IntentSignal(
                    s.get("defect", ""), s.get("verdict", "cannot_tell"), 0.7, s.get("evidence", "")
                )
                for s in record.artistic.get("intent_signals") or []
            ],
        )
        if not darkroom.should_run(record.route_class, scores):
            continue
        asset = by_key.get(record.asset_key)
        measurement = measurements.get(record.asset_key)
        if asset is None or measurement is None:
            continue
        try:
            produced = darkroom.run(
                asset, measurement, scores,
                out_dir=options.output_dir,
                renderer_name=options.darkroom_renderer,
                # Nothing reports faces any more, and `None` is the honest
                # value: the skin check still runs, it just cannot be fatal.
                faces_present=None,
            )
        except Exception as e:
            logger.warning("Darkroom failed for %s: %s", record.filename, reports.redact(str(e)))
            continue
        record.edit_recipes = produced["edit_recipes"]
        record.rendered_variants = produced["rendered_variants"]
        record.recipe_confidence = produced["recipe_confidence"]
        record.preserve_intent = produced["preserve_intent"]
        record.suggested_sidecars = produced["sidecars"]
        record.darkroom_rejections = produced["rejected"]
        record.darkroom_engine = produced.get("engine", "")
        record.darkroom_engine_version = produced.get("engine_version", "")


def _stage3(assets, measurements, provisional, semantics, options, client, result, model):
    """Run the artistic read, or record for every frame why it did not run."""
    from photoai import bootstrap
    from photoai import stage3 as stage3_module  # noqa: F401  (used throughout this function)

    routed = {r.asset_key: r.route_class for r in provisional}
    hints = {
        r.asset_key: {
            "has_unrecoverable": bool(r.issues.get("unrecoverable")),
            "intentionality_likelihood": (r.artistic or {}).get("intentionality_likelihood", 50),
            "curatorial_uncertainty": (r.artistic or {}).get("curatorial_uncertainty", 100),
        }
        for r in provisional
    }

    if not options.stage3:
        # Asked for by the operator: no read was wanted.
        return {
            key: stage3_module.ArtisticAssessment.not_required("Stage 3 was switched off")
            for key in routed
        }
    if not options.semantic:
        # Wanted but unavailable, which is a different thing and reads differently
        # in the report: these frames were never judged, not judged and passed over.
        return {
            key: stage3_module.ArtisticAssessment.skipped(
                "the semantic pass did not run, so there was no model to read with"
            )
            for key in routed
        }

    stage3_model = bootstrap_model(options, model)
    cache = AnalysisCache(options.internal / CACHE_NAME)
    try:
        assessments = stage3_pass(
            assets, measurements, routed, hints,
            model=stage3_model, client=client, cache=cache,
            reasoning=options.reasoning, concurrency=options.concurrency,
        )
    except bootstrap.SemanticUnavailable as e:
        # Already classified as fatal downstream: an exhausted balance or a
        # rejected key will hit every remaining group identically, so the run
        # ends and the previous report survives rather than being replaced by
        # one in which every artistic field is an error string.
        #
        # The developer escape hatch still applies. `analyze` never sets it.
        if not options.allow_semantic_fallback:
            raise
        logger.error("Stage 3 unavailable, continuing on request: %s", e)
        return {
            key: stage3_module.ArtisticAssessment.failed([str(e)], model=model)
            for key in routed
        }
    except Exception as e:
        logger.error("Stage 3 failed entirely: %s", reports.redact(str(e)))
        return {
            key: stage3_module.ArtisticAssessment.failed(
                [reports.redact(str(e))], model=stage3_model
            )
            for key in routed
        }
    cache.save()

    result.stage3_completed = sum(1 for a in assessments.values() if a.completed)
    result.stage3_failed = sum(
        1 for a in assessments.values() if a.status == stage3_module.Stage3Status.FAILED.value
    )
    result.stage3_skipped = sum(
        1 for a in assessments.values()
        if a.status in (
            stage3_module.Stage3Status.SKIPPED.value,
            stage3_module.Stage3Status.NOT_REQUIRED.value,
        )
    )
    logger.info(
        "Stage 3: %d completed, %d failed, %d not required",
        result.stage3_completed, result.stage3_failed, result.stage3_skipped,
    )
    return assessments


def _write_manifest(options: PipelineOptions, result: RunResult, assets, model: str) -> None:
    """Record what this run did, so the next one can tell what it added."""
    from photoai import prompts
    from photoai import stage3 as stage3_module

    failed = [r.checksum for r in result.records if r.status != "ok"]
    manifest = batches.BatchManifest(
        run_id=result.run_id,
        started_at=batches.now(),
        finished_at=batches.now(),
        new=list(result.new_assets),
        modified=list(result.modified_assets),
        reused=list(result.reused_assets),
        failed=failed,
        keys=[a.key for a in assets],
        model=model if options.semantic else "",
        stage2_prompt_version=prompts.STAGE2_PROMPT_VERSION,
        stage3_prompt_version=stage3_module.PROMPT_VERSION,
        analyzer_version=scoring.ANALYZER_VERSION,
        llm_calls=dict(result.llm_calls),
        stage2_completed=sum(1 for r in result.records if r.semantic_present),
        stage3_completed=result.stage3_completed,
    )
    batches.append(manifest, options.internal / batches.MANIFEST_NAME)


def bootstrap_model(options, fallback: str) -> str:
    from photoai import bootstrap

    return bootstrap.resolve_model(options.stage3_model) if options.stage3_model else fallback


def _semantics(
    assets, measurements, options: PipelineOptions, client, result: RunResult, model: str
) -> dict[str, Semantic]:
    """Run the paid pass, or record precisely why it did not.

    A failure is never swallowed. Either it ends the run, or -- with the
    fallback explicitly allowed -- it is recorded on the result so that every
    report says the content was not checked.
    """
    from photoai import bootstrap

    if not options.semantic:
        return {}

    cache = AnalysisCache(options.internal / CACHE_NAME)
    try:
        semantics = semantic_pass(
            assets, measurements, model=model, client=client,
            cache=cache, calls=result.llm_calls, reasoning=options.reasoning,
            concurrency=options.concurrency,
        )
    except bootstrap.SemanticUnavailable as e:
        # Already classified where it happened -- re-running it through
        # `classify_api_error` turned "authentication" into "unknown: the API
        # call failed (SemanticUnavailable)", which reads as a bug in this tool
        # rather than a wrong key.
        result.semantic_error = f"{e.kind}: {e}"
        logger.error("Semantic pass failed: %s", e)
        if not options.allow_semantic_fallback:
            raise
        return {}
    except Exception as e:
        kind, message = bootstrap.classify_api_error(e)
        result.semantic_error = f"{kind}: {message}"
        logger.error("Semantic pass failed: %s", message)
        if not options.allow_semantic_fallback:
            raise bootstrap.SemanticUnavailable(message, kind=kind) from e
        return {}

    # Saved immediately: a Stage 3 failure after this point must not throw away
    # content results that were already paid for. `semantic_pass` has already
    # put the raw replies in; re-deriving entries from the stitched Semantic
    # here would overwrite them with percentiles from this run's population,
    # which is the thing the raw storage exists to avoid.
    if semantics:
        cache.save()

    result.semantic_completed = bool(semantics)
    if not semantics:
        result.semantic_error = "the model returned nothing usable for any group"
        if not options.allow_semantic_fallback:
            raise bootstrap.SemanticUnavailable(result.semantic_error, kind="empty")
        return semantics

    # Every photograph that needed a content check has to have got one. A run
    # where 250 assets came from store and the one new group failed used to
    # succeed: `semantics` was non-empty, so the run reported completion while
    # the only photograph it had actually been asked about had no genre, no
    # faces and no subject -- and was filed on that basis.
    unchecked = [
        asset.key
        for asset in assets
        if asset.kind is media.MediaKind.PHOTO
        and measurements.get(asset.key, Measurement()).preview_path
        and asset.key not in semantics
    ]
    if unchecked and not options.allow_semantic_fallback:
        result.semantic_error = (
            f"{len(unchecked)} photograph(s) came back without a content check"
        )
        raise bootstrap.SemanticUnavailable(result.semantic_error, kind="incomplete")
    return semantics


def _cluster(assets, measurements) -> dict[str, tuple[str, int, bool, float, float]]:
    """filename -> (cluster_id, size, is_best, mean_similarity, quality_margin)."""
    items = [
        duplicates.DupItem(
            key=asset.key,
            phash=measurements[asset.key].phash,
            date_shot=measurements[asset.key].exif.get("date_shot"),
            quality=measurements[asset.key].quality,
        )
        for asset in assets
        if asset.kind is media.MediaKind.PHOTO and measurements[asset.key].phash
    ]
    out: dict[str, tuple[str, int, bool, float, float]] = {}
    for cluster in duplicates.cluster_items(items):
        similarity = cluster.mean_similarity()
        cluster_id = cluster.best_key
        best_quality = max((i.quality for i in cluster.items), default=0.0)
        for item in cluster.items:
            out[item.key] = (
                cluster_id,
                cluster.size,
                item.key == cluster.best_key,
                similarity,
                round(best_quality - item.quality, 2),
            )
    for asset in assets:
        out.setdefault(asset.key, (asset.key, 1, True, 0.0, 0.0))
    return out


def _embed(assets, measurements, cache) -> tuple[dict[str, tuple[float, ...]], str, str]:
    """asset key -> unit vector, plus the mode that was used and why.

    Returns `({}, "perceptual_hash", reason)` whenever the encoder is not there,
    which is the normal case: `onnxruntime` is an optional extra and the weights
    are only fetched when somebody asks. Nothing here downloads unless
    `PHOTO_AI_EMBEDDINGS=1` says it may -- see `photoai.embeddings.prepare`.

    The cache lookup happens before the model is loaded, deliberately. A re-run
    over an unchanged archive answers every frame from store and never opens the
    335 MB file at all, which is what makes turning this on cheap after the
    first time.

    One file at a time rather than in batches. Batching would be perhaps twice
    as fast, and it would mean one unreadable preview taking seven good frames
    down with it; a per-file loop isolates the failure to the frame that caused
    it, and that frame simply falls back to the hash.
    """
    from photoai import embeddings

    state = embeddings.prepare()
    if not state.ok:
        return {}, "perceptual_hash", state.reason

    vectors: dict[str, tuple[float, ...]] = {}
    pending: list[tuple[media.Asset, str]] = []
    for asset in assets:
        if asset.kind is not media.MediaKind.PHOTO:
            continue
        stored = cache.get_embedding(asset.checksum, state.model_id)
        if stored:
            vectors[asset.key] = stored
            continue
        preview = measurements.get(asset.key, Measurement()).preview_path
        if preview:
            pending.append((asset, preview))

    if pending:
        try:
            embeddings.encoder()
        except embeddings.EmbeddingsUnavailable as e:
            logger.warning("Falling back to perceptual hashes: %s", e)
            return {}, "perceptual_hash", str(e)
        for asset, preview in pending:
            vector = embeddings.embed_file(preview)
            if vector:
                vectors[asset.key] = vector
                cache.put_embedding(asset.checksum, state.model_id, vector)
        cache.save()

    photos = sum(1 for a in assets if a.kind is media.MediaKind.PHOTO)
    missing = photos - len(vectors)
    detail = ""
    if missing > 0:
        # Partial coverage is not a failure, but it is not full coverage either,
        # and those frames are compared by hash while the rest are compared by
        # meaning. Said out loud rather than averaged away.
        detail = f"{missing} of {photos} photographs have no vector and fall back to the hash"
    return vectors, "embedding", detail


def _score_all(
    assets, measurements, clusters, semantics, calibration, options,
    semantic_model: str = "", stage3_results: dict | None = None,
    vectors: dict | None = None,
) -> list[AssetRecord]:
    """Score, then run the collection-level flagship pass, then classify."""
    prepared: list[tuple[media.Asset, Measurement, ScoreInput, object]] = []

    for asset in assets:
        measurement = measurements[asset.key]
        found = issues_for(asset, measurement)
        cluster_id, size, is_best, similarity, margin = clusters[asset.key]

        semantic = semantics.get(asset.key, Semantic())
        kind = "video" if asset.kind is media.MediaKind.VIDEO else "photo"
        profile = calibration.for_kind(kind)

        # A duplicate is only "weaker" when it is measurably weaker. Inside the
        # margin the two frames are a tie as far as a Laplacian can tell, and
        # which one is better is a compositional judgement the local pass cannot
        # make -- so both survive and a human chooses. On a real archive this is
        # the difference between proposing to delete a 66 that lost to a 69, and
        # only proposing the 38 that lost to a 42.
        if not is_best and margin >= profile.threshold("duplicate_margin"):
            found.add(
                issues_module.IssueCode.WEAKER_DUPLICATE,
                f"a sharper frame exists in cluster {cluster_id} "
                f"(quality {margin:.0f} points higher)",
            )

        completeness = 1.0
        if kind == "video":
            samples = (measurement.video or {}).get("sample_count", 0)
            completeness = min(1.0, samples / 5.0) if samples else 0.3

        inp = ScoreInput(
            asset_id=asset.asset_id,
            filename=asset.filename,
            kind=kind,
            technical_quality=measurement.quality,
            uplift=measurement.uplift,
            issues=found,
            semantic=semantic,
            is_raw=asset.is_raw,
            megapixels=measurement.megapixels,
            cluster_size=size,
            is_best_in_cluster=is_best or not found.codes() & {issues_module.IssueCode.WEAKER_DUPLICATE},
            cluster_similarity=similarity,
            cluster_margin=margin,
            evidence_completeness=completeness,
            semantic_ran=semantic.present,
            artistic=(stage3_results or {}).get(asset.key),
        )
        prepared.append((asset, measurement, inp, scoring.score(inp, profile)))

    flagship = _select_flagship(prepared, measurements, clusters, calibration, vectors or {})

    records: list[AssetRecord] = []
    for asset, measurement, inp, scores in prepared:
        profile = calibration.for_kind(inp.kind)
        scored = scoring.classify(
            inp, scores, profile, flagship_selected=asset.key in flagship
        )
        records.append(
            _build_record(
                asset, measurement, inp, scored, clusters, calibration, options, semantic_model
            )
        )
    return records


def _select_flagship(prepared, measurements, clusters, calibration, vectors=None) -> set[str]:
    """The absolute floor first, then a diversity-aware competition.

    Both halves are required. The floor alone promotes nothing from a modest
    shoot; the competition alone promotes twenty frames of the same sunset.

    `vectors` is empty on a run with no encoder, in which case every candidate
    reaches `select_diverse` without one and the redundancy term is measured on
    perceptual hashes, exactly as before.
    """
    vectors = vectors or {}
    profile = calibration.photo
    candidates = []
    for asset, measurement, inp, scores in prepared:
        if inp.kind == "video":
            continue
        if not inp.is_best_in_cluster or inp.issues.unrecoverable:
            continue
        if not scoring.eligible_for_flagship(scores, calibration.for_kind(inp.kind)):
            continue
        candidates.append(
            duplicates.Candidate(
                key=asset.key,
                relevance=scores.portfolio_potential,
                item=duplicates.DupItem(
                    key=asset.key,
                    phash=measurement.phash,
                    date_shot=measurement.exif.get("date_shot"),
                    quality=measurement.quality,
                    genre=inp.semantic.genre,
                    embedding=vectors.get(asset.key),
                ),
            )
        )

    if not candidates:
        return set()

    limit = max(
        1,
        min(
            int(profile.threshold("flagship_max_total")),
            round(len(candidates) * profile.threshold("flagship_top_fraction")),
        ),
    )
    chosen = duplicates.select_diverse(
        candidates,
        limit=limit,
        lambda_=profile.threshold("diversity_lambda"),
        max_per_genre=int(profile.threshold("flagship_max_per_genre")),
    )
    return set(chosen)


def _build_record(
    asset, measurement, inp, scored, clusters, calibration, options, semantic_model: str = ""
) -> AssetRecord:
    cluster_id, size, is_best, _, _ = clusters[asset.key]
    record = provenance_module.read_provenance(asset.path, measurement.exif)

    facts = marketplaces.TechnicalFacts(
        kind=inp.kind,
        megapixels=measurement.megapixels,
        width=measurement.width,
        height=measurement.height,
        duration=measurement.duration,
        container=measurement.container,
        file_format=asset.format.value if inp.kind == "photo" else "",
    )
    # Metadata is generated first so that eligibility can take its completeness
    # into account: "export ready" must mean the CSV would actually validate,
    # not merely that the pixels are big enough.
    metadata = stock_metadata.generate(
        semantic=inp.semantic,
        route=scored.route.value,
        exif=measurement.exif,
        provenance_label=provenance_module.label_for_submission(record),
        marketplaces=[],
        copyright_holder=options.copyright_holder,
    )
    recommendations = marketplaces.evaluate(
        scored, facts, inp.semantic, record, metadata_complete=metadata.is_complete
    )
    metadata.suggested_marketplaces = [r.platform_name for r in recommendations if r.eligible][:3]

    art = _artistic_for(asset, measurement, inp)
    stage3_payload, portrait_verdict = _stage3_payload(inp)
    verdict = curation.categorise(inp, scored.scores, calibration.for_kind(inp.kind))

    # Release warnings are gone: they came from a model guessing at a legal
    # question. Provenance is a fact about the file and stays.
    legal_warnings = []
    if record.is_uncertain:
        legal_warnings.append("Provenance undeclared: confirm this is a camera original")

    return AssetRecord(
        asset_id=asset.asset_id,
        source_path=str(asset.path),
        filename=asset.filename,
        asset_key=asset.key,
        all_files=[str(p) for p in asset.all_files],
        file_states=_file_states(asset),
        evidence=_evidence(inp.issues),
        media_type=inp.kind,
        checksum=asset.checksum,
        width=measurement.width,
        height=measurement.height,
        megapixels=measurement.megapixels,
        duration=measurement.duration,
        analyzer_version=scoring.ANALYZER_VERSION,
        calibration=calibration.fingerprint,
        model_versions={"semantic": semantic_model or "none"},
        analyzed_at=reports.datetime.now(reports.UTC).isoformat(timespec="seconds"),
        scores=scored.scores.to_dict(),
        category=verdict.category,
        final_score=verdict.final_score,
        final_score_detail=verdict.score.to_dict(),
        category_reasons=verdict.reasons,
        commercial_blockers=verdict.commercial_blockers,
        route_class=scored.route_class.value,
        route=scored.route.value,
        tags=[t.value for t in scored.tags],
        confidence=scored.scores.confidence,
        issues=issues_module.summarise(inp.issues),
        strengths=scored.strengths,
        reasons=scored.reasons,
        reason_keys=[r.to_dict() for r in scored.reason_keys],
        # "other" is a genre the model can assign. When nothing looked, the
        # honest value is that nobody knows -- reporting `other` presented an
        # unexamined frame as a confidently classified one.
        genre=inp.semantic.genre if inp.semantic.present else "unknown",
        camera=_camera_key(measurement.exif),
        concepts=list(inp.semantic.concepts),
        description=inp.semantic.description,
        stock_metadata=metadata.to_dict(),
        edit_recipe=measurement.recipe,
        expected_gain=scored.scores.potential_gain,
        uplift_validated=edit_recipe.UPLIFT_VALIDATED,
        marketplaces=[r.to_dict() for r in recommendations],
        provenance=record.value.value,
        legal_warnings=legal_warnings,
        cluster_id=cluster_id,
        cluster_size=size,
        best_in_cluster=is_best,
        cluster_similarity=clusters[asset.key][3],
        cluster_margin=clusters[asset.key][4],
        phash=measurement.phash,
        semantic_present=inp.semantic.present,
        semantic_model=semantic_model,
        video=measurement.video,
        preview_path=measurement.preview_path,
        proposed_action=_proposed_action(scored.route_class),
        artistic=art.to_dict(),
        stage3=stage3_payload,
        portrait_verdict=portrait_verdict,
        status="error" if measurement.error else "ok",
        error=measurement.error,
    )


def _stage3_payload(inp) -> tuple[dict, str]:
    """The stored artistic record, plus what the face says to do about it.

    `is_portrait` has to be passed here for the same reason it is passed to the
    gates: without it this reported `keep` on a frame the categoriser had just
    rejected for a half-closed eye, because the face was small enough to look
    incidental. Two answers to one question in one record.
    """
    from photoai import curation as curation_module
    from photoai import stage3 as stage3_module

    assessment = inp.artistic
    if not isinstance(assessment, stage3_module.ArtisticAssessment):
        return stage3_module.ArtisticAssessment.not_required(
            "no artistic analysis was attached"
        ).to_dict(), "keep"

    verdict, _ = stage3_module.portrait_verdict(
        assessment, is_portrait=curation_module.is_portrait(inp.semantic)
    )
    return assessment.to_dict(), verdict


def _camera_key(exif: dict) -> str:
    """`Make Model`, normalised. Empty when EXIF does not say.

    An empty key reads as "no camera information", and `knows_camera` treats
    that as familiar rather than as a new body -- the alternative would abstain
    on every scan and every file with stripped metadata.
    """
    make = str(exif.get("camera_make") or "").strip()
    model = str(exif.get("camera_model") or "").strip()
    if model.lower().startswith(make.lower()) and make:
        model = model[len(make):].strip()
    return " ".join(part for part in (make, model) if part).strip()


def _artistic_for(asset, measurement, inp):
    """Deterministic intentionality signals for one frame.

    Runs on every asset because it is cheap and because its output decides
    whether a frame may be destroyed. The model-supplied dimensions stay None
    until the vision pass fills them.
    """
    from photoai import artistic

    scores = artistic.ArtisticScores(
        technical_integrity=int(max(0, min(100, measurement.quality))),
    )
    if asset.kind is media.MediaKind.VIDEO or measurement.error:
        scores.curatorial_uncertainty = 80
        return scores

    try:
        image = media.open_photo(asset.path)
        work = image.copy()
        work.thumbnail((edit_recipe.WORK_PX, edit_recipe.WORK_PX), 1)
        array = np.asarray(work.convert("RGB"), dtype=np.float64)
        image.close()
    except Exception as e:
        logger.debug("Artistic pass skipped for %s: %s", asset.filename, e)
        scores.curatorial_uncertainty = 85
        return scores

    luma = array @ np.array([0.299, 0.587, 0.114])
    scores.signals = artistic.assess_intent(
        array,
        blur_ratio=measurement.blur_ratio,
        sharpness_global=measurement.blur_ratio,
        sharpness_tile=measurement.blur_ratio,
        tilt_degrees=edit_recipe.estimate_tilt(array),
        clipped_highlights=measurement.clipped_highlights,
        mean_luma=float(luma.mean()),
        iso=measurement.exif.get("iso"),
    )
    scores.intentionality_likelihood = artistic.intentionality_score(scores.signals)
    scores.curatorial_uncertainty = artistic.uncertainty_score(
        scores.signals, semantic_present=inp.semantic.present
    )
    return scores


def _file_states(asset) -> dict:
    """Snapshot every file of the asset, so a later move can verify it."""
    states = {}
    for path in asset.all_files:
        try:
            states[str(path)] = media.FileState.of(path).to_dict()
        except OSError as e:
            logger.warning("Could not snapshot %s: %s", path, e)
    return states


def _evidence(found) -> str:
    """The machine-readable grounds, used only by the purge gate.

    Built from the unrecoverable issue *codes* rather than from prose, so that
    `quarantine.is_purgeable_evidence` decides on a closed vocabulary instead of
    pattern-matching an English sentence.
    """
    return ",".join(sorted({i.code.value for i in found.unrecoverable}))


def _proposed_action(route_class: RouteClass) -> str:
    if route_class is RouteClass.TRASH:
        return "quarantine"
    if route_class is RouteClass.REVIEW:
        return "hold_for_review"
    if route_class is RouteClass.ARCHIVE_ONLY:
        return "keep_in_place"
    if route_class is RouteClass.DUPLICATE_CANDIDATE:
        return "compare_by_hand"
    return "keep_in_place"


def _plan_operations(assets, records, options: PipelineOptions) -> list:
    """A quarantine plan for the trash class. Written, not executed."""
    by_key = {a.key: a for a in assets}
    quarantine = quarantine_module.Quarantine(
        options.resolved_quarantine(), source_roots=[options.input_dir]
    )
    moves = []
    for record in records:
        if record.route_class != RouteClass.TRASH.value:
            continue
        asset = by_key.get(record.asset_key)
        if asset is None:
            continue
        moves.append(
            quarantine_module.PlannedMove(
                asset_id=asset.asset_id,
                files=asset.all_files,
                destination_dir=options.resolved_quarantine(),
                reason=_plan_reason(record, options.language),
                route_class=record.route_class,
                scores=record.scores,
                states=record.file_states,
                evidence=record.evidence,
            )
        )
    return quarantine.plan(moves)


def _plan_reason(record, language: str) -> str:
    """The single line a user reads beside a file proposed for removal.

    Localised, and built from the structured keys so it never mixes languages.
    """
    from photoai.i18n import t
    from photoai.scoring import Reason

    for payload in record.reason_keys or []:
        reason = Reason(payload.get("key", ""), payload.get("params") or {}, payload.get("text", ""))
        if reason.key.startswith("reason.unrecoverable"):
            return reason.localise(language)
    if record.reason_keys:
        first = record.reason_keys[0]
        return Reason(first.get("key", ""), first.get("params") or {}, first.get("text", "")).localise(
            language
        )
    return t("reason.unrecoverable", language, detail="?")


def _record_for(records, key):
    return next((r for r in records if r.asset_key == key), None)


# --- re-running routing without re-analysing --------------------------------


def reclassify(analysis_path: Path, calibration: CalibrationSet) -> list[dict]:
    """Redo routing from a stored run. No decoding, no model, no cost.

    This is the payoff of storing every dimension rather than only the class:
    tuning a threshold is a sub-second operation on a collection that took an
    hour and real money to analyze the first time.
    """
    stored, _ = reports.read_json(analysis_path)

    prepared: list[tuple[dict, ScoreInput, scoring.AssetScores]] = []
    for row in stored:
        raw_scores = {k: int(v) for k, v in (row.get("scores") or {}).items()}
        scores = scoring.AssetScores(**raw_scores)
        kind = row.get("media_type", "photo")
        profile = calibration.for_kind(kind)

        found = issues_module.IssueSet()
        for described in (row.get("issues") or {}).get("unrecoverable", []):
            code_name, _, detail = str(described).partition(": ")
            try:
                found.add(issues_module.IssueCode(code_name), detail)
            except ValueError:
                found.add(issues_module.IssueCode.DEAD_MOMENT, described)

        semantic = Semantic(
            present=bool(row.get("semantic_present", False)),
            genre=row.get("genre") or "other",
        )
        inp = ScoreInput(
            asset_id=row.get("asset_id", ""),
            filename=row.get("filename", ""),
            kind=kind,
            issues=found,
            semantic=semantic,
            is_best_in_cluster=bool(row.get("best_in_cluster", True)),
        )
        scores.routing_score = scoring.routing_score(scores, profile)
        prepared.append((row, inp, scores))

    flagship = _reselect_flagship(prepared, calibration)

    out: list[dict] = []
    for row, inp, scores in prepared:
        profile = calibration.for_kind(inp.kind)
        scored = scoring.classify(
            inp, scores, profile, flagship_selected=inp.filename in flagship
        )
        out.append(
            {
                "filename": inp.filename,
                "previous_class": row.get("route_class", ""),
                "route_class": scored.route_class.value,
                "changed": row.get("route_class") != scored.route_class.value,
                "reasons": scored.reasons,
            }
        )
    return out


def _reselect_flagship(prepared, calibration: CalibrationSet) -> set[str]:
    """The same two-stage flagship pass, run against stored numbers only."""
    profile = calibration.photo
    candidates = [
        duplicates.Candidate(
            key=inp.filename,
            relevance=scores.portfolio_potential,
            item=duplicates.DupItem(
                key=inp.filename,
                phash=row.get("phash", ""),
                quality=scores.current_quality,
                genre=inp.semantic.genre,
            ),
        )
        for row, inp, scores in prepared
        if inp.kind == "photo"
        and inp.is_best_in_cluster
        and not inp.issues.unrecoverable
        and scoring.eligible_for_flagship(scores, profile)
    ]
    if not candidates:
        return set()
    limit = max(
        1,
        min(
            int(profile.threshold("flagship_max_total")),
            round(len(candidates) * profile.threshold("flagship_top_fraction")),
        ),
    )
    return set(
        duplicates.select_diverse(
            candidates,
            limit=limit,
            lambda_=profile.threshold("diversity_lambda"),
            max_per_genre=int(profile.threshold("flagship_max_per_genre")),
        )
    )
