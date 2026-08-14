"""The images the report shows, copied into the report so it can be moved.

The report used to point at `.internal/previews/`, which worked exactly as long
as nobody touched anything. It broke for a subtler reason than a wrong path:
the page is rendered inside a staging directory and then moved two levels up by
the transactional publish, so `../../previews/x.jpg` -- correct where it was
written -- resolved outside the run once it landed. Every tile was blank.

The fix is not a better relative path. It is not having one: `run/report/` holds
its own `assets/`, every `src` is `assets/thumbs/x.jpg`, and the folder can be
zipped, moved, emailed or served from anywhere and still work. A report you
cannot send to somebody is half a report.

Three rules the derivatives follow:

**Orientation is baked in.** A phone photograph carries its rotation in EXIF,
and a browser showing a thumbnail generated without it displays somebody's
portrait on its side.

**Metadata is stripped.** The derivatives carry no EXIF and no GPS, because the
report is the artefact people share, and sharing a folder of thumbnails should
not hand over the coordinates of somebody's house.

**Nothing is silently missing.** A frame whose derivative cannot be made gets a
placeholder with the reason, and the reasons are counted at the top of the page.
A blank tile that looks like a rendering bug is worse than a tile that says the
file would not decode.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Long side and JPEG quality for each derivative. The thumbnail is what the grid
# shows; the full size is what the lightbox opens. 1600 is enough to judge
# sharpness on a laptop and small enough that a hundred of them still zip.
THUMB_PX, THUMB_QUALITY = 480, 78
FULL_PX, FULL_QUALITY = 1600, 82

# Inlined copies are smaller again: a standalone file carries every thumbnail in
# its own body, so 30% off each one is 30% off the file.
EMBED_PX, EMBED_QUALITY = 480, 75

ASSETS_DIRNAME = "assets"
THUMBS_DIRNAME = "thumbs"
FULL_DIRNAME = "full"

# Video has no still to show. Extracting a poster frame is a job for the video
# pass, not for the report writer, and pretending otherwise produced a broken
# image tag per clip.
SKIP_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mts", ".mkv"}


@dataclass
class Derivative:
    """What the report should put in one card's `src`, or why it cannot."""

    key: str
    thumb: str = ""
    full: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.thumb)


@dataclass
class BuildResult:
    derivatives: dict[str, Derivative] = field(default_factory=dict)
    missing: list[Derivative] = field(default_factory=list)

    def fail(self, derivative: Derivative) -> None:
        """Record a card that has no image, and why.

        The reason goes into `derivatives` as well as `missing`, so the card and
        the summary line quote the same sentence. They did not, once: a video
        was "no preview was generated" on the card and "video: no still frame to
        show" in the summary, on the same page.
        """
        self.derivatives[derivative.key] = derivative
        self.missing.append(derivative)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "with_image": sum(1 for d in self.derivatives.values() if d.ok),
            "without_image": len(self.missing),
        }

    def reasons(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for item in self.missing:
            tally[item.reason] = tally.get(item.reason, 0) + 1
        return tally


def cache_key(source: Path, width: int, quality: int) -> str:
    """sha1 of the file's bytes, plus the shape asked for.

    Content-addressed rather than path-addressed: two runs over the same
    photograph reuse the derivative even if the file moved, and a photograph
    that was edited in place gets a new one.
    """
    digest = hashlib.sha1()  # noqa: S324 - a cache key, not a security boundary
    with open(source, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{digest.hexdigest()}-{width}-{quality}"


def load_image(source: Path, preview: Path | None = None) -> Image.Image:
    """The photograph, upright and in RGB.

    RAW files are not opened here. `preview_generator` already rendered one
    during the run, and re-developing a RAW to make a thumbnail would cost
    seconds per frame to produce a worse image than the one already on disk.
    """
    candidates = [p for p in (source, preview) if p and Path(p).is_file()]
    if not candidates:
        raise FileNotFoundError("neither the original nor a preview is on disk")

    last: Exception | None = None
    for candidate in candidates:
        try:
            with Image.open(candidate) as image:
                # exif_transpose applies the orientation tag and drops it, so
                # the result cannot be rotated a second time downstream.
                upright = ImageOps.exif_transpose(image)
                return upright.convert("RGB")
        except Exception as e:  # pragma: no cover - depends on the file
            last = e
    raise last or OSError("could not decode")


def encode(image: Image.Image, width: int, quality: int) -> bytes:
    """One derivative, resized and stripped of metadata.

    `Image.new` + paste rather than `save` on the original: saving a copy can
    carry EXIF forward, and the point here is that it does not.
    """
    copy = image.copy()
    copy.thumbnail((width, width), Image.LANCZOS)
    clean = Image.new("RGB", copy.size)
    clean.paste(copy)
    buffer = io.BytesIO()
    clean.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def build(records, report_dir: Path, cache_dir: Path) -> BuildResult:
    """Write `assets/thumbs` and `assets/full` beside the report.

    Returns what each record should show and, for the ones that cannot show
    anything, why -- so the page can say so instead of rendering a black box.
    """
    report_dir, cache_dir = Path(report_dir), Path(cache_dir)
    thumbs = report_dir / ASSETS_DIRNAME / THUMBS_DIRNAME
    full = report_dir / ASSETS_DIRNAME / FULL_DIRNAME
    thumbs.mkdir(parents=True, exist_ok=True)
    full.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = BuildResult()
    for record in records:
        derivative = _one(record, thumbs, full, cache_dir)
        # Failures land in `derivatives` too. The card reads its reason from
        # there, and a card that fell back to a generic "no preview" while the
        # summary line said "video" was the same missing image described two
        # different ways on one page.
        if derivative.ok:
            result.derivatives[record.asset_key or record.filename] = derivative
        else:
            result.fail(derivative)
    if result.missing:
        logger.info(
            "%d of %d cards have no image: %s",
            len(result.missing), len(records), result.reasons(),
        )
    return result


def _one(record, thumbs: Path, full: Path, cache_dir: Path) -> Derivative:
    key = record.asset_key or record.filename
    source = Path(record.source_path or "")
    preview = Path(record.preview_path) if record.preview_path else None

    if source.suffix.lower() in SKIP_SUFFIXES:
        return Derivative(key, reason="video: no still frame to show")
    if not source.is_file() and not (preview and preview.is_file()):
        return Derivative(key, reason="the file is no longer where it was analysed")

    stem = _safe_stem(record.filename, key)
    try:
        image = load_image(source, preview)
        thumb_name = _write(image, thumbs, f"{stem}.jpg", THUMB_PX, THUMB_QUALITY,
                            cache_dir, source if source.is_file() else preview)
        full_name = _write(image, full, f"{stem}.jpg", FULL_PX, FULL_QUALITY,
                           cache_dir, source if source.is_file() else preview)
    except Exception as e:
        return Derivative(key, reason=_short(e))

    return Derivative(
        key,
        thumb=f"{ASSETS_DIRNAME}/{THUMBS_DIRNAME}/{thumb_name}",
        full=f"{ASSETS_DIRNAME}/{FULL_DIRNAME}/{full_name}",
    )


def _write(image, folder: Path, name: str, width: int, quality: int,
           cache_dir: Path, source: Path) -> str:
    """Encode once per (content, width, quality) and copy from store after."""
    cached = cache_dir / f"{cache_key(source, width, quality)}.jpg"
    if not cached.is_file():
        cached.write_bytes(encode(image, width, quality))
    shutil.copyfile(cached, folder / name)
    return name


def _safe_stem(filename: str, key: str) -> str:
    """A filename safe on every filesystem, and unique across subdirectories.

    Two cards can be `IMG_0001.JPG` from different folders; the key
    disambiguates them without putting a path separator in a filename.
    """
    stem = Path(filename or key).stem
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
    suffix = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]  # noqa: S324
    return f"{cleaned}-{suffix}"


def _short(error: Exception) -> str:
    text = str(error).strip() or type(error).__name__
    return text[:120]


# --- the standalone file ------------------------------------------------------


def inline(records, width: int = EMBED_PX, quality: int = EMBED_QUALITY) -> BuildResult:
    """Thumbnails as data URIs, for a report that is one file.

    Only the thumbnail is inlined. A second, larger copy per photograph would
    double a file that is already the whole gallery, so the lightbox scales the
    inlined image up instead -- softer than a real 1600px view, and the trade a
    single shareable file is worth.
    """
    result = BuildResult()
    for record in records:
        key = record.asset_key or record.filename
        source = Path(record.source_path or "")
        preview = Path(record.preview_path) if record.preview_path else None

        if source.suffix.lower() in SKIP_SUFFIXES:
            result.fail(Derivative(key, reason="video: no still frame to show"))
            continue
        try:
            image = load_image(source, preview)
            encoded = base64.b64encode(encode(image, width, quality)).decode("ascii")
        except Exception as e:
            result.fail(Derivative(key, reason=_short(e)))
            continue
        uri = f"data:image/jpeg;base64,{encoded}"
        result.derivatives[key] = Derivative(key, thumb=uri, full=uri)
    return result
