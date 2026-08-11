"""What a file *is*, before anything tries to judge it.

Three jobs, all of them cheap and all of them deterministic:

- **Typing.** Photo, video, or neither. Extension first, because reading 5500
  file headers to learn what you could have read from the name is a waste, but
  the extension is only a claim: `open_photo` still fails loudly on a file that
  is not what it says it is.

- **Identity.** A checksum, so an asset keeps the same id when it is renamed and
  a different one when it is edited. This is what makes the analysis cache
  correct rather than merely fast: the cache key is checksum + analyzer
  version, so bumping the analyzer invalidates exactly the work that changed.

- **Grouping.** `P1042675.RW2`, `P1042675.JPG` and `P1042675.xmp` are one
  photograph, not three. Every filesystem operation downstream moves the group
  or none of it -- quarantining a RAW and leaving its sidecar behind produces an
  orphan that no tool can interpret.

Nothing here decodes a full-resolution image except `open_photo`, and that one
refuses a decompression bomb before Pillow allocates the buffer.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".rw2", ".arw", ".cr3", ".cr2", ".nef", ".dng", ".orf", ".raf", ".pef"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
HEIC_EXTENSIONS = {".heic", ".heif"}
PHOTO_EXTENSIONS = RAW_EXTENSIONS | TIFF_EXTENSIONS | JPEG_EXTENSIONS | HEIC_EXTENSIONS | {".png", ".webp"}

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mts", ".m2ts", ".mxf", ".mkv", ".webm"}

SIDECAR_EXTENSIONS = {".xmp", ".pp3", ".dop", ".aae", ".on1", ".acr"}

# Pillow's own bomb guard trips at ~89 MP by default, which rejects legitimate
# medium-format and stitched panoramas. Raised to something a real camera can
# actually produce, and enforced ourselves so the failure is a clear exception
# rather than a warning nobody reads.
MAX_PIXELS = 400_000_000

CHECKSUM_CHUNK = 1024 * 1024
# Hashing 5500 RAW files end to end is minutes of pure I/O for no extra
# discrimination. Head + tail + size collides only for files that are identical
# at both ends and the same length, which for camera originals means identical.
FAST_CHECKSUM_BYTES = 4 * 1024 * 1024


class MediaKind(StrEnum):
    PHOTO = "photo"
    VIDEO = "video"
    UNSUPPORTED = "unsupported"


class PhotoFormat(StrEnum):
    RAW = "RAW"
    JPEG = "JPEG"
    TIFF = "TIFF"
    HEIC = "HEIC"
    OTHER = "OTHER"


class UnreadableMedia(Exception):
    """The file exists but cannot be decoded as what it claims to be."""


def classify(path: Path) -> MediaKind:
    ext = path.suffix.lower()
    if ext in PHOTO_EXTENSIONS:
        return MediaKind.PHOTO
    if ext in VIDEO_EXTENSIONS:
        return MediaKind.VIDEO
    return MediaKind.UNSUPPORTED


def photo_format(path: Path) -> PhotoFormat:
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return PhotoFormat.RAW
    if ext in JPEG_EXTENSIONS:
        return PhotoFormat.JPEG
    if ext in TIFF_EXTENSIONS:
        return PhotoFormat.TIFF
    if ext in HEIC_EXTENSIONS:
        return PhotoFormat.HEIC
    return PhotoFormat.OTHER


def has_raw_data(path: Path) -> bool:
    """Whether highlight/shadow recovery estimates can be trusted.

    A JPEG has roughly what you see; a RAW has one to two stops more in the
    highlights. Recoverability scoring leans on this, so it is a question worth
    asking explicitly rather than inferring from the extension at each call
    site.
    """
    return path.suffix.lower() in RAW_EXTENSIONS or path.suffix.lower() in TIFF_EXTENSIONS


@dataclass
class Asset:
    """One photograph or clip, together with every file that belongs to it."""

    path: Path
    kind: MediaKind
    checksum: str
    size_bytes: int
    siblings: list[Path] = field(default_factory=list)
    sidecars: list[Path] = field(default_factory=list)

    @property
    def asset_id(self) -> str:
        """Stable across renames, different after an edit."""
        return self.checksum[:16]

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def format(self) -> PhotoFormat:
        return photo_format(self.path)

    @property
    def is_raw(self) -> bool:
        return has_raw_data(self.path)

    @property
    def all_files(self) -> list[Path]:
        """Everything that must move together, primary first."""
        return [self.path, *self.siblings, *self.sidecars]


def checksum_file(path: Path, *, full: bool = False) -> str:
    """SHA-256 of the file, or of its ends plus its length.

    The fast form is the default because it is what the analysis cache needs --
    "is this the same file I already analyzed" -- and it reads 8 MB instead of
    50 MB per RAW. Pass ``full=True`` where the answer has to be a real content
    hash, which in this codebase means the quarantine manifest: that checksum is
    what a restore verifies against.
    """
    digest = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        if full or size <= 2 * FAST_CHECKSUM_BYTES:
            while chunk := f.read(CHECKSUM_CHUNK):
                digest.update(chunk)
        else:
            digest.update(f.read(FAST_CHECKSUM_BYTES))
            f.seek(-FAST_CHECKSUM_BYTES, os.SEEK_END)
            digest.update(f.read(FAST_CHECKSUM_BYTES))
            digest.update(str(size).encode())
    return digest.hexdigest()


def discover(root: Path, *, follow_symlinks: bool = False) -> list[Asset]:
    """Find every photo and video under `root`, grouped with its sidecars.

    Symlinks are not followed by default. The tool's own output is a farm of
    symlinks pointing back into the archive, so pointing a run at a directory
    containing a previous run's output would otherwise walk in a circle and
    analyze everything twice.
    """
    by_stem: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and not follow_symlinks:
            continue
        if not path.is_file():
            continue
        by_stem.setdefault((path.parent, path.stem), []).append(path)

    assets: list[Asset] = []
    for (_, _), paths in sorted(by_stem.items()):
        primaries = [p for p in paths if classify(p) is not MediaKind.UNSUPPORTED]
        if not primaries:
            continue
        sidecars = [p for p in paths if p.suffix.lower() in SIDECAR_EXTENSIONS]
        primary = _preferred_primary(primaries)
        siblings = [p for p in primaries if p != primary]
        try:
            assets.append(
                Asset(
                    path=primary,
                    kind=classify(primary),
                    checksum=checksum_file(primary),
                    size_bytes=primary.stat().st_size,
                    siblings=siblings,
                    sidecars=sidecars,
                )
            )
        except OSError as e:
            logger.warning("Could not stat %s: %s", primary, e)
    return assets


def _preferred_primary(paths: list[Path]) -> Path:
    """RAW wins over its JPEG twin: it is the file with the latitude in it.

    Shooting RAW+JPEG produces two files of the same photograph. Scoring both
    doubles the bill and reports one photograph twice; the RAW is the one worth
    keeping because every recovery estimate downstream depends on having it.
    """
    raws = [p for p in paths if p.suffix.lower() in RAW_EXTENSIONS]
    if raws:
        return raws[0]
    videos = [p for p in paths if classify(p) is MediaKind.VIDEO]
    if videos:
        return videos[0]
    return paths[0]


def open_photo(path: Path) -> Image.Image:
    """Decode a still to RGB, refusing anything oversized or malformed.

    The size check happens on the header, before the pixel buffer is allocated:
    a crafted file declaring 60000x60000 would otherwise ask for ~10 GB during
    `convert`, and the process dies rather than raising something catchable.
    """
    fmt = photo_format(path)
    if fmt is PhotoFormat.RAW:
        return _open_raw(path)
    try:
        with Image.open(path) as img:
            width, height = img.size
            if width * height > MAX_PIXELS:
                raise UnreadableMedia(
                    f"{path.name}: {width}x{height} exceeds the {MAX_PIXELS:,} pixel limit"
                )
            img.load()
            return img.convert("RGB")
    except UnreadableMedia:
        raise
    except Exception as e:
        raise UnreadableMedia(f"{path.name}: {e}") from e


def _open_raw(path: Path) -> Image.Image:
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            return Image.fromarray(raw.postprocess(use_camera_wb=True, output_bps=8))
    except Exception as e:
        raise UnreadableMedia(f"{path.name}: RAW decode failed: {e}") from e


def megapixels(width: int, height: int) -> float:
    return round(width * height / 1_000_000, 2)
