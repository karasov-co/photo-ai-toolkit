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

import curation
import duplicates
import edit_recipe
import issues as issues_module
import marketplaces
import media
import provenance as provenance_module
import quarantine as quarantine_module
import raw_measurements
import reports
import scoring
import stock_metadata
import technical_filter
from calibration import CalibrationSet, resolve
from exif_reader import extract_exif
from preview_generator import PreviewGenerationError, generate_preview, preview_name
from reports import AssetRecord
from scoring import RouteClass, ScoreInput, Semantic

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
    # Keyed by asset key. Kept on the result so that everything written after
    # the run -- edit recipes, insights -- can use what was measured without
    # decoding a second time.
    measurements: dict = field(default_factory=dict)

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

    def get_stage3(self, checksum: str, model: str) -> dict | None:
        import stage3

        return self._data.get(stage3.cache_key(checksum, model))

    def put_stage3(self, checksum: str, model: str, payload: dict) -> None:
        import stage3

        self._data[stage3.cache_key(checksum, model)] = payload

    def get(self, checksum: str) -> dict | None:
        return self._data.get(self.key(checksum))

    def put(self, checksum: str, payload: dict) -> None:
        self._data[self.key(checksum)] = payload

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
    import video_analyzer

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
) -> dict[str, Semantic]:
    """Rank frames against each other in groups, then stitch to a global order.

    Reuses the prompts, group builder and Bradley-Terry aggregation that already
    exist in this repository. Absolute scoring is not used and is not an option:
    every live absolute call made against this archive returned 548, 560, 694,
    762 -- a scale that does not discriminate is not a scale.
    """
    import base64

    import aggregate
    import batch_runner
    import prompts
    import routing

    if client is None:
        import bootstrap

        client = bootstrap.make_client()

    photo_names = [
        a.key
        for a in assets
        if a.kind is media.MediaKind.PHOTO and measurements.get(a.key, Measurement()).preview_path
    ]
    if not photo_names:
        return {}

    by_name = {a.key: a for a in assets}
    groups = aggregate.build_groups(photo_names, size=group_size)
    parsed_groups: list[list[dict]] = []
    per_frame: dict[str, dict] = {}
    # A per-group failure is survivable; every group failing is not. An
    # authentication or model error hits all of them identically, and swallowing
    # it per group turned a hard failure into a silent empty result.
    first_error: Exception | None = None
    succeeded = 0

    for index, group in enumerate(groups):
        frames = []
        for name in group:
            measurement = measurements[name]
            with open(measurement.preview_path, "rb") as f:
                encoded = base64.standard_b64encode(f.read()).decode()
            # For a RAW, hand over what the *sensor* saturated at. For anything
            # else, hand over the rendered figure and say so. The two mean
            # different things and the prompt must not conflate them.
            if measurement.raw_available:
                highlights = measurement.raw_clipped_all_channels
                shadows = measurement.clipped_shadows
            else:
                highlights = measurement.clipped_highlights
                shadows = measurement.clipped_shadows
            frames.append(
                {
                    "filename": name,
                    "clipped_highlights": highlights,
                    "clipped_shadows": shadows,
                    "measurement_domain": measurement.measurement_domain,
                    "headroom_stops": measurement.raw_highlight_headroom_stops,
                    "encoded": encoded,
                }
            )

        try:
            response = client.responses.create(
                model=model,
                instructions=prompts.STAGE2_SYSTEM,
                input=[{"role": "user", "content": prompts.stage2_user_content(frames)}],
                max_output_tokens=900 + 260 * len(frames),
                reasoning={"effort": "low"},
            )
            items = batch_runner.parse_group_json(response.output_text or "")
            succeeded += 1
        except Exception as e:
            first_error = first_error or e
            logger.error("Semantic group %d failed: %s", index, reports.redact(str(e)))
            continue

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

    if first_error is not None and succeeded == 0:
        # Nothing got through. Re-raise so the caller decides whether that ends
        # the run, rather than returning an empty dict that reads like "the
        # model had no opinion".
        raise first_error

    scores = aggregate.aggregate_all_axes(parsed_groups)

    out: dict[str, Semantic] = {}
    for name, item in per_frame.items():
        try:
            assessment = routing.parse_assessment(item, name)
        except routing.AssessmentParseError as e:
            logger.warning("Unusable model output for %s: %s", name, e)
            continue
        semantic = scoring.semantic_from_assessment(assessment, group_size=item.get("_group_size"))
        # Replace the within-group rank with the stitched global percentile.
        semantic.axis_a = scores["axis_a"].get(name, semantic.axis_a)
        semantic.axis_b = scores["axis_b"].get(name, semantic.axis_b)
        semantic.axis_c = scores["axis_c"].get(name, semantic.axis_c)
        semantic.description = str(item.get("note") or "")
        asset = by_name.get(name)
        if asset is not None:
            semantic.secondary_genres = []
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
    import base64

    import stage3 as stage3_module

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
            faces_present=bool(hint.get("faces_present")),
            corrupt=bool(measurement.error),
        )
        if not needed:
            out[key] = stage3_module.ArtisticAssessment.not_required(reason)
            continue

        if cache is not None:
            cached = cache.get_stage3(asset.checksum, model)
            if cached is not None:
                out[key] = stage3_module.ArtisticAssessment.from_dict(cached)
                continue
        pending.append(key)

    if not pending:
        return out

    by_key = {a.key: a for a in assets}
    for start in range(0, len(pending), group_size):
        group = pending[start : start + group_size]
        frames = []
        for key in group:
            measurement = measurements[key]
            preview = Path(measurement.preview_path)
            if not preview.exists():
                out[key] = stage3_module.ArtisticAssessment.skipped("no preview was generated")
                continue
            views = _stage3_views(by_key[key], preview, artistic_hints.get(key, {}))
            frames.append({"key": key, "views": views, "encoded": views[0][1]})

        if not frames:
            continue

        assessments = _stage3_call(
            frames, model=model, client=client, stage3_module=stage3_module
        )
        for frame in frames:
            key = frame["key"]
            assessment = assessments.get(key)
            out[key] = assessment or stage3_module.ArtisticAssessment.failed(
                ["the model returned nothing usable for this frame"], model=model
            )
            if cache is not None and out[key].completed:
                cache.put_stage3(by_key[key].checksum, model, out[key].to_dict())

    del base64
    return out


def _stage3_views(asset, preview: Path, hint: dict) -> list[tuple[str, str]]:
    """Base64 views for one frame: the whole picture, plus crops when a face is in it."""
    import base64

    import stage3 as stage3_module

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


def _stage3_call(frames, *, model, client, stage3_module, budget_factor: float = 1.0):
    """One group, with bounded retries on a malformed reply.

    Two failure modes, handled differently. A malformed reply is retried as-is,
    because models do occasionally wrap JSON in prose and asking again fixes it.
    A *truncated* reply splits the group instead: sending the same six frames
    again with the same budget produces the same truncation, and on this archive
    that cost two minutes and three times the tokens per group before failing.
    """
    import prompts

    group = [f["key"] for f in frames]
    errors: list[str] = []
    budget = int(
        (prompts.STAGE3_BASE_OUTPUT_TOKENS
         + prompts.STAGE3_MAX_OUTPUT_TOKENS_PER_FRAME * len(frames))
        * budget_factor
    )

    for attempt in range(stage3_module.MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=prompts.STAGE3_SYSTEM,
                input=[{"role": "user", "content": prompts.stage3_user_content(frames)}],
                max_output_tokens=budget,
                reasoning={"effort": "low"},
            )
            if _truncated(response):
                errors.append(f"attempt {attempt + 1}: the reply hit the {budget}-token limit")
                return _split_or_widen(
                    frames, model=model, client=client, stage3_module=stage3_module,
                    budget_factor=budget_factor, errors=errors,
                )
            parsed = stage3_module.parse_group(
                response.output_text or "", group, model=model
            )
            if parsed:
                for assessment in parsed.values():
                    assessment.retries = attempt
                return parsed
            errors.append(f"attempt {attempt + 1}: no usable object in the reply")
        except stage3_module.Stage3ParseError as e:
            errors.append(f"attempt {attempt + 1}: {e}")
        except Exception as e:
            # Not swallowed: recorded against every frame in the group.
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


def _split_or_widen(frames, *, model, client, stage3_module, budget_factor, errors):
    """Halve the group, or -- when it is already one frame -- widen the budget."""
    if len(frames) > 1:
        middle = len(frames) // 2
        logger.info("Stage 3 reply truncated; splitting %d frames into two", len(frames))
        out = {}
        for half in (frames[:middle], frames[middle:]):
            out.update(
                _stage3_call(
                    half, model=model, client=client, stage3_module=stage3_module,
                    budget_factor=budget_factor,
                )
            )
        return out

    widened = budget_factor * 2
    if widened <= STAGE3_BUDGET_LIMIT:
        logger.info("Stage 3 reply truncated on a single frame; widening the budget")
        return _stage3_call(
            frames, model=model, client=client, stage3_module=stage3_module,
            budget_factor=widened,
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
    progress: Callable[[str, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    client=None,
) -> RunResult:
    """Discover, measure, cluster, score, select, and propose. Moves nothing."""
    import bootstrap

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
    )
    measurements: dict[str, Measurement] = {}

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
        if progress:
            progress(asset.filename, index, len(assets))

        cached = None if options.force else cache.get(asset.checksum)
        if cached is not None:
            measurement = Measurement.from_dict(cached)
        else:
            if asset.kind is media.MediaKind.VIDEO:
                measurement = measure_video(asset, previews_dir, samples=options.video_samples)
            else:
                measurement = measure_photo(asset, previews_dir)
            cache.put(asset.checksum, measurement.to_dict())
        measurements[asset.key] = measurement

    cache.save()

    clusters = _cluster(assets, measurements)
    semantics = _semantics(assets, measurements, options, client, result, model)

    # Stage 3 needs a provisional route to decide who is worth reading, so the
    # scoring pass runs twice: once to get candidates, then again with the
    # artistic evidence that can promote or block them.
    provisional = _score_all(
        assets, measurements, clusters, semantics, calibration, options, model
    )
    stage3_results = _stage3(
        assets, measurements, provisional, semantics, options, client, result, model
    )

    result.records = _score_all(
        assets, measurements, clusters, semantics, calibration, options, model, stage3_results
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
    import artistic
    import preference_model
    import selective_policy
    from model_monitoring import Monitor, Observation
    from preference_store import PreferenceStore

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
    import artistic
    import darkroom

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
                faces_present=record.semantic_present and (
                    "needs_model_release" in (record.tags or [])
                ),
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
    import stage3 as stage3_module  # noqa: F401  (used throughout this function)

    routed = {r.asset_key: r.route_class for r in provisional}
    hints = {
        r.asset_key: {
            "has_unrecoverable": bool(r.issues.get("unrecoverable")),
            "intentionality_likelihood": (r.artistic or {}).get("intentionality_likelihood", 50),
            "curatorial_uncertainty": (r.artistic or {}).get("curatorial_uncertainty", 100),
            "faces_present": semantics.get(r.asset_key, Semantic()).faces,
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
            model=stage3_model, client=client or _client_for(options), cache=cache,
        )
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


def bootstrap_model(options, fallback: str) -> str:
    import bootstrap

    return bootstrap.resolve_model(options.stage3_model) if options.stage3_model else fallback


def _client_for(options):
    import bootstrap

    return bootstrap.make_client()


def _semantics(
    assets, measurements, options: PipelineOptions, client, result: RunResult, model: str
) -> dict[str, Semantic]:
    """Run the paid pass, or record precisely why it did not.

    A failure is never swallowed. Either it ends the run, or -- with the
    fallback explicitly allowed -- it is recorded on the result so that every
    report says the content was not checked.
    """
    import bootstrap

    if not options.semantic:
        return {}

    try:
        semantics = semantic_pass(
            assets, measurements, model=model, client=client
        )
    except Exception as e:
        kind, message = bootstrap.classify_api_error(e)
        result.semantic_error = f"{kind}: {message}"
        logger.error("Semantic pass failed: %s", message)
        if not options.allow_semantic_fallback:
            raise bootstrap.SemanticUnavailable(message, kind=kind) from e
        return {}

    result.semantic_completed = bool(semantics)
    if not semantics:
        result.semantic_error = "the model returned nothing usable for any group"
        if not options.allow_semantic_fallback:
            raise bootstrap.SemanticUnavailable(result.semantic_error, kind="empty")
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


def _score_all(
    assets, measurements, clusters, semantics, calibration, options,
    semantic_model: str = "", stage3_results: dict | None = None,
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

    flagship = _select_flagship(prepared, measurements, clusters, calibration)

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


def _select_flagship(prepared, measurements, clusters, calibration) -> set[str]:
    """The absolute floor first, then a diversity-aware competition.

    Both halves are required. The floor alone promotes nothing from a modest
    shoot; the competition alone promotes twenty frames of the same sunset.
    """
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

    legal_warnings = []
    if inp.semantic.faces or inp.semantic.identifiable_people:
        legal_warnings.append("Model release required before commercial licensing")
    if inp.semantic.logos:
        legal_warnings.append("Readable trademark present: editorial only unless cleared")
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
    import curation as curation_module
    import stage3 as stage3_module

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
    import artistic

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
    from i18n import t
    from scoring import Reason

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

        tags = row.get("tags") or []
        semantic = Semantic(
            present=bool(row.get("semantic_present", False)),
            genre=row.get("genre") or "other",
            faces="needs_model_release" in tags,
            logos="legal_review" in tags,
            identifiable_people="needs_model_release" in tags,
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
