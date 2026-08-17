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
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# HEIC/HEIF is what every recent iPhone shoots, and Pillow cannot decode it on
# its own. The extension was previously listed as supported with no decoder
# behind it, so every .heic in an archive failed to open and -- because an
# unreadable file is a corrupt file -- was routed to trash. Registering the
# opener is what makes the claim true; if the package is missing the format is
# removed from the supported set rather than being advertised and then failing.
try:  # pragma: no cover - exercised by whichever branch the environment has
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:  # pragma: no cover
    HEIF_AVAILABLE = False
    logger.info("pillow-heif is not installed; HEIC/HEIF files will be skipped")

RAW_EXTENSIONS = {".rw2", ".arw", ".cr3", ".cr2", ".nef", ".dng", ".orf", ".raf", ".pef"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
HEIC_EXTENSIONS = {".heic", ".heif"} if HEIF_AVAILABLE else set()
PHOTO_EXTENSIONS = RAW_EXTENSIONS | TIFF_EXTENSIONS | JPEG_EXTENSIONS | HEIC_EXTENSIONS | {".png", ".webp"}

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mts", ".m2ts", ".mxf", ".mkv", ".webm"}

SIDECAR_EXTENSIONS = {".xmp", ".pp3", ".dop", ".aae", ".on1", ".acr"}

# Pillow's own bomb guard trips at ~89 MP by default, which rejects legitimate
# medium-format and stitched panoramas. Raised to something a real camera can
# actually produce, and enforced ourselves so the failure is a clear exception
# rather than a warning nobody reads.
MAX_PIXELS = 400_000_000

CHECKSUM_CHUNK = 1024 * 1024

# There used to be a "fast" head+tail+size checksum here, on the theory that
# hashing 5500 RAW files end to end was minutes of I/O for no extra
# discrimination. Both halves of that were wrong.
#
# It collided: two files of equal length differing only in the middle produced
# the same digest, and therefore the same `asset_id`. That id keys manual
# overrides and the quarantine manifest, so a collision misidentifies a file
# inside a destructive operation -- the one place where being wrong is
# unrecoverable.
#
# And it bought nothing. Measured on this archive, SHA-256 runs at 2.3 GB/s
# (hardware accelerated): 0.24s versus 0.14s for 554 MB, about 67 seconds for
# a 5500-frame archive against a pipeline that takes over two hours to decode
# it. The optimisation saved 0.7% of the runtime in exchange for a correctness
# hole in the destructive path.


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
    relpath: str = ""

    @property
    def asset_id(self) -> str:
        """Stable across renames, different after an edit."""
        return self.checksum[:16]

    @property
    def key(self) -> str:
        """The identity every in-run mapping must use. Never the basename.

        Cameras restart their numbering, and two memory cards routinely both
        contain `P1000001.RW2`. Keying a run's measurements or clusters by
        filename silently merges them: on a real reproduction, a good frame
        (quality 49) inherited a black frame's measurement (quality 0) and was
        routed to trash. The relative path is unique by construction within a
        scan and costs nothing.

        `asset_id` is the *cross-run* identity, used for overrides and the
        quarantine manifest, and survives renames. This one does not need to.
        """
        return self.relpath or self.path.name

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


def checksum_file(path: Path, *, full: bool = True) -> str:
    """Full SHA-256 of the file contents.

    `full` is accepted and ignored; it exists so that call sites written against
    the old two-mode API keep working and keep meaning the safe thing.
    """
    del full
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHECKSUM_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileState:
    """What a file looked like at analysis time, for verifying it before a move.

    Size and mtime are the cheap screen; the checksum is the proof. All three
    are stored because a file edited in place between analysis and quarantine
    can keep its size, and a file touched by a backup tool can change mtime
    without changing content -- neither on its own is a reliable answer.
    """

    size: int
    mtime_ns: int
    checksum: str

    @classmethod
    def of(cls, path: Path) -> FileState:
        stat = path.stat()
        return cls(size=stat.st_size, mtime_ns=stat.st_mtime_ns, checksum=checksum_file(path))

    def to_dict(self) -> dict:
        return {"size": self.size, "mtime_ns": self.mtime_ns, "checksum": self.checksum}

    @classmethod
    def from_dict(cls, payload: dict) -> FileState:
        return cls(
            size=int(payload.get("size", 0)),
            mtime_ns=int(payload.get("mtime_ns", 0)),
            checksum=str(payload.get("checksum", "")),
        )

    def matches(self, path: Path) -> tuple[bool, str]:
        """Whether `path` is still the file this state describes, and why not."""
        try:
            stat = path.stat()
        except OSError as e:
            return False, f"cannot stat: {e}"
        if stat.st_size != self.size:
            return False, f"size changed {self.size} -> {stat.st_size}"
        if stat.st_mtime_ns != self.mtime_ns:
            # Not fatal on its own -- fall through to the checksum, which is the
            # question actually being asked.
            if checksum_file(path) != self.checksum:
                return False, "contents changed since analysis"
            return True, "mtime changed but contents are identical"
        if self.checksum and checksum_file(path) != self.checksum:
            return False, "contents changed since analysis"
        return True, ""


def discover(
    root: Path,
    *,
    follow_symlinks: bool = False,
    exclude: list[Path] | None = None,
) -> list[Asset]:
    """Find every photo and video under `root`, grouped with its sidecars.

    Symlinks are not followed by default. The tool's own output is a farm of
    symlinks pointing back into the archive, so pointing a run at a directory
    containing a previous run's output would otherwise walk in a circle and
    analyze everything twice.

    `exclude` keeps the run's own output and quarantine out of its own input.
    Without it, pointing `--output` inside `--input` makes the second run treat
    the first run's 512px previews as new photographs, score them, and route
    them -- and a preview routed to trash is a proposal to delete a file the
    tool itself created, sitting next to real ones.
    """
    excluded = [Path(p).resolve() for p in (exclude or [])]
    by_stem: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and not follow_symlinks:
            continue
        if not path.is_file():
            continue
        if excluded and is_inside(path, excluded):
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
                    relpath=_relative(primary, root),
                )
            )
        except OSError as e:
            logger.warning("Could not stat %s: %s", primary, e)
    return assets


def _relative(path: Path, root: Path) -> str:
    """POSIX-style path under the scan root, so keys are stable across platforms."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


# --- keeping a run's own output out of its own input ------------------------


def excluded_roots(*directories: Path | None) -> list[Path]:
    return [Path(d).resolve() for d in directories if d]


def is_inside(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in roots)


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
