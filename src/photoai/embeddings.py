"""One normalised vector per photograph, from a CLIP image tower run locally.

Why this exists
---------------

`duplicates.visual_similarity` compares 8x8 perceptual hashes. That finds bursts
and repeated framings, and it cannot tell a beige wall from a beige beach: two
different subjects with the same palette and framing land on the same hash, and
the diversity pass then treats one as a repeat of the other. This module is the
other measurement -- what the frame is *of* -- and `duplicates` uses it when it
is there.

What it is
----------

The CLIP ViT-B/32 **image tower only**, exported to ONNX and run through ONNX
Runtime on the CPU. No text tower, no tokeniser, no PyTorch: a 512-dimensional
vector per image and nothing else. `MODEL` below records the exact file, where
it came from, its SHA256 and its licence.

Three properties this had to have, in order:

- **Optional.** `onnxruntime` is an extra (`pip install -e ".[embeddings]"`),
  never a dependency. With nothing extra installed the tool behaves exactly as
  it did, and says which similarity it used rather than leaving it to be
  guessed. `available()` is the probe and `status().reason` is the one-line
  answer when it is False.
- **Never a surprise download.** 335 MB arrives only when somebody asked for it:
  `PHOTO_AI_EMBEDDINGS=1`, or `photoai models --download`, or an explicit
  `prepare(allow_download=True)`. The default mode is "use it if it is already
  here", which is why the test suite can pin the whole feature off in one
  environment variable and never touch the network.
- **Verified.** The download is checked against the SHA256 recorded here, and a
  file that does not match is refused rather than loaded. A model file is code
  as far as ONNX Runtime is concerned, and "it downloaded, so it is fine" is not
  a security position.

The model is pinned to a **commit**, not to a branch. `resolve/main/model.onnx`
would silently start serving different weights the day the repository is
updated, and the recorded checksum would then be the thing that looks broken.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingsUnavailable(RuntimeError):
    """Asked to encode when the encoder is not usable. Carries the reason."""


class ChecksumMismatch(RuntimeError):
    """A model file whose contents are not what this module recorded."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to fetch, verify, licence-check and run one encoder."""

    model_id: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    dimensions: int
    image_size: int
    input_name: str
    licence: str
    licence_url: str
    source: str
    # CLIP's own preprocessing constants. Wrong values here do not fail loudly;
    # they quietly produce vectors that are a bit worse, which is why they are
    # recorded beside the weights rather than typed in at the call site.
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


# The image tower of OpenAI's CLIP ViT-B/32, exported to ONNX by Qdrant for
# FastEmbed. MIT on both halves: OpenAI released CLIP under MIT, and the export
# carries MIT on the model card. Single file, no tokeniser, no config to parse.
#
# The URL is pinned to commit e0c24ed rather than to `main`, so the bytes behind
# it cannot change under the checksum below.
CLIP_VIT_B32_VISION = ModelSpec(
    model_id="clip-vit-b32-vision-onnx",
    filename="clip-vit-b32-vision.onnx",
    url=(
        "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/"
        "e0c24ed0fa57fa3e4f97f30de74c51d944036ace/model.onnx"
    ),
    sha256="c68d3d9a200ddd2a8c8a5510b576d4c94d1ae383bf8b36dd8c084f94e1fb4d63",
    size_bytes=351_686_194,
    dimensions=512,
    image_size=224,
    input_name="pixel_values",
    licence="MIT",
    licence_url="https://huggingface.co/Qdrant/clip-ViT-B-32-vision",
    source="Qdrant/clip-ViT-B-32-vision @ e0c24ed (export of openai/clip-vit-base-patch32)",
)

MODEL = CLIP_VIT_B32_VISION

# `1`/`on`/`true`  -- use embeddings, and download the model if it is missing.
# `0`/`off`/`false`-- do not use them at all, whatever is installed.
# unset / `auto`   -- use them if onnxruntime and a verified model are already
#                     here, and never fetch anything.
ENABLE_VAR = "PHOTO_AI_EMBEDDINGS"
# Where the weights live. Overridable so the suite can point somewhere empty and
# a shared machine can put 335 MB on the big disk.
CACHE_VAR = "PHOTO_AI_MODEL_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "photo-ai-toolkit" / "models"

DOWNLOAD_TIMEOUT = 60.0
_CHUNK = 1 << 20

INSTALL_HINT = 'pip install -e ".[embeddings]"'
DOWNLOAD_HINT = "photoai models --download"


# --- where things are -------------------------------------------------------


def cache_dir() -> Path:
    override = os.environ.get(CACHE_VAR, "").strip()
    return Path(override).expanduser() if override else DEFAULT_CACHE_DIR


def model_path(spec: ModelSpec = MODEL) -> Path:
    return cache_dir() / spec.filename


def mode() -> str:
    """`on`, `off` or `auto`. Anything unrecognised is `auto`, not an error."""
    raw = os.environ.get(ENABLE_VAR, "").strip().lower()
    if raw in ("1", "on", "true", "yes"):
        return "on"
    if raw in ("0", "off", "false", "no"):
        return "off"
    return "auto"


# --- verification -----------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_path(path: Path) -> Path:
    return path.with_name(path.name + ".verified.json")


def _write_receipt(path: Path, spec: ModelSpec) -> None:
    """Record that *these exact bytes* were hashed and matched.

    Hashing 335 MB costs about a second, which is nothing once and irritating on
    every run of a CLI. The receipt is size + mtime + the digest that was
    checked, the same shape `media.ContentCheck` uses for photographs: cheap
    fields as the screen, the digest as the proof. Any drift and the full hash
    runs again.
    """
    stat = path.stat()
    _receipt_path(path).write_text(
        json.dumps(
            {
                "sha256": spec.sha256,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "model_id": spec.model_id,
            }
        ),
        encoding="utf-8",
    )


def _receipt_matches(path: Path, spec: ModelSpec) -> bool:
    try:
        payload = json.loads(_receipt_path(path).read_text(encoding="utf-8"))
        stat = path.stat()
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("sha256") == spec.sha256
        and payload.get("size") == stat.st_size
        and payload.get("mtime_ns") == stat.st_mtime_ns
    )


def verify(path: Path, spec: ModelSpec = MODEL) -> bool:
    """True only if the file on disk is byte-for-byte the recorded model."""
    if not path.is_file():
        return False
    if _receipt_matches(path, spec):
        return True
    if path.stat().st_size != spec.size_bytes:
        return False
    if sha256_of(path) != spec.sha256:
        return False
    try:
        _write_receipt(path, spec)
    except OSError as e:  # a read-only cache is not a verification failure
        logger.debug("Could not write the verification receipt: %s", e)
    return True


# --- fetching ---------------------------------------------------------------


def download(
    spec: ModelSpec = MODEL,
    *,
    destination: Path | None = None,
    progress=None,
) -> Path:
    """Fetch the weights, verify them, and only then put them in place.

    Downloaded to `<name>.part` and renamed after the checksum matches, so an
    interrupted download can never be picked up on the next run as though it
    were a model. A mismatch deletes the partial file and raises: a wrong model
    is not something to keep around and retry against.
    """
    target = destination or model_path(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")

    logger.info(
        "Downloading %s (%.0f MB, %s) from %s",
        spec.model_id, spec.size_bytes / 1e6, spec.licence, spec.url,
    )
    digest = hashlib.sha256()
    read = 0
    try:
        with urllib.request.urlopen(spec.url, timeout=DOWNLOAD_TIMEOUT) as response:
            with open(partial, "wb") as handle:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    read += len(block)
                    if read > spec.size_bytes * 2:
                        raise ChecksumMismatch(
                            f"{spec.model_id}: the server sent more than twice the "
                            f"expected {spec.size_bytes} bytes"
                        )
                    digest.update(block)
                    handle.write(block)
                    if progress:
                        progress(read, spec.size_bytes)
    except (urllib.error.URLError, OSError, ChecksumMismatch):
        partial.unlink(missing_ok=True)
        raise

    if digest.hexdigest() != spec.sha256:
        partial.unlink(missing_ok=True)
        raise ChecksumMismatch(
            f"{spec.model_id}: expected sha256 {spec.sha256}, got {digest.hexdigest()}. "
            "The file was discarded."
        )

    os.replace(partial, target)
    _write_receipt(target, spec)
    logger.info("Model verified and stored at %s", target)
    return target


# --- the capability probe ---------------------------------------------------


@dataclass
class Status:
    """Whether embeddings can be used right now, and one line saying why not."""

    ok: bool
    reason: str = ""
    model_id: str = ""
    path: str = ""
    fix: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "model_id": self.model_id,
            "path": self.path,
            "fix": self.fix,
        }


def _onnxruntime_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("onnxruntime") is not None


def status(spec: ModelSpec = MODEL) -> Status:
    """Probe, without downloading anything and without loading the model."""
    path = model_path(spec)
    if mode() == "off":
        return Status(
            False,
            f"embeddings are switched off ({ENABLE_VAR}=0)",
            spec.model_id,
            str(path),
            fix=f"unset {ENABLE_VAR}, or set it to 1",
        )
    if not _onnxruntime_installed():
        return Status(
            False,
            "onnxruntime is not installed",
            spec.model_id,
            str(path),
            fix=INSTALL_HINT,
        )
    if not path.is_file():
        return Status(
            False,
            "the model has not been downloaded",
            spec.model_id,
            str(path),
            fix=DOWNLOAD_HINT,
        )
    if not verify(path, spec):
        return Status(
            False,
            "the model file does not match its recorded checksum and was refused",
            spec.model_id,
            str(path),
            fix=f"delete {path} and re-run {DOWNLOAD_HINT}",
        )
    return Status(True, "", spec.model_id, str(path))


def available(spec: ModelSpec = MODEL) -> bool:
    """Can a vector be produced right now, with no download and no surprises."""
    return status(spec).ok


def unavailable_reason(spec: ModelSpec = MODEL) -> str:
    """One line, or `''` when embeddings are available."""
    return status(spec).reason


def prepare(spec: ModelSpec = MODEL, *, allow_download: bool | None = None) -> Status:
    """`status()`, plus the one place a download is allowed to happen.

    `allow_download` defaults to whatever `PHOTO_AI_EMBEDDINGS` says, so callers
    inside the pipeline need no flag of their own: unset means "use it if it is
    here", `1` means "and fetch it if it is not".
    """
    if allow_download is None:
        allow_download = mode() == "on"

    current = status(spec)
    if current.ok or not allow_download:
        return current
    # Only a missing file is worth a download. A refused checksum or a missing
    # onnxruntime would not be fixed by fetching 335 MB again.
    if current.reason != "the model has not been downloaded":
        return current
    try:
        download(spec)
    except (urllib.error.URLError, OSError, ChecksumMismatch) as e:
        return Status(
            False,
            f"the model could not be downloaded: {e}",
            spec.model_id,
            str(model_path(spec)),
            fix=DOWNLOAD_HINT,
        )
    return status(spec)


# --- the encoder ------------------------------------------------------------


def preprocess(image, spec: ModelSpec = MODEL):
    """One PIL image to CLIP's expected `(1, 3, 224, 224)` float32 tensor.

    Shortest edge to 224 with bicubic resampling, centre crop, scale to 0..1,
    subtract the channel means and divide by the channel deviations -- the
    transform `CLIPImageProcessor` applies, spelled out here so that the only
    runtime dependency is ONNX Runtime.
    """
    import numpy as np
    from PIL import Image

    edge = spec.image_size
    image = image.convert("RGB")
    width, height = image.size
    scale = edge / min(width, height)
    resized = image.resize(
        (max(edge, round(width * scale)), max(edge, round(height * scale))),
        Image.BICUBIC,
    )
    width, height = resized.size
    left, top = (width - edge) // 2, (height - edge) // 2
    cropped = resized.crop((left, top, left + edge, top + edge))

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(
        spec.std, dtype=np.float32
    )
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def normalise(vector) -> tuple[float, ...]:
    """To unit length, so that a dot product *is* the cosine.

    A zero vector cannot be normalised and is returned empty rather than as a
    tuple of NaN -- `duplicates` reads an empty vector as "no embedding here"
    and falls back, which is the behaviour that keeps a broken frame from
    poisoning a comparison.
    """
    import numpy as np

    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not norm or not np.isfinite(norm):
        return ()
    return tuple(float(v) for v in (array / norm))


@dataclass
class Encoder:
    """A loaded ONNX session, plus the preprocessing that belongs to it."""

    spec: ModelSpec
    session: object = None
    _input: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.session is not None and not self._input:
            names = [i.name for i in self.session.get_inputs()]
            self._input = self.spec.input_name if self.spec.input_name in names else names[0]

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def encode(self, image) -> tuple[float, ...]:
        return self.encode_many([image])[0]

    def encode_many(self, images) -> list[tuple[float, ...]]:
        import numpy as np

        if not images:
            return []
        batch = np.concatenate([preprocess(i, self.spec) for i in images], axis=0)
        raw = self.session.run(None, {self._input: batch})[0]
        return [normalise(row) for row in np.asarray(raw)]


_ENCODER: Encoder | None = None


def encoder(spec: ModelSpec = MODEL) -> Encoder:
    """The loaded encoder, built once per process.

    Raises `EmbeddingsUnavailable` with the same one-line reason `status()`
    gives rather than an ImportError or a FileNotFoundError from three frames
    down, so a caller has something it can print.
    """
    global _ENCODER
    if _ENCODER is not None and _ENCODER.spec == spec:
        return _ENCODER

    current = prepare(spec)
    if not current.ok:
        raise EmbeddingsUnavailable(current.reason)

    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(model_path(spec)), options, providers=["CPUExecutionProvider"]
    )
    _ENCODER = Encoder(spec, session)
    return _ENCODER


def reset() -> None:
    """Drop the loaded session. For tests and for a changed cache directory."""
    global _ENCODER
    _ENCODER = None


def embed_image(image, spec: ModelSpec = MODEL) -> tuple[float, ...]:
    return encoder(spec).encode(image)


def embed_file(path, spec: ModelSpec = MODEL) -> tuple[float, ...]:
    """One image file to a unit vector. `()` if the file cannot be opened.

    An unreadable preview is a missing measurement, not a failed run: the caller
    gets an empty vector, `duplicates` falls back to the hash for that frame,
    and the other 299 photographs are unaffected.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.load()
            return embed_image(image, spec)
    except (OSError, UnidentifiedImageError) as e:
        logger.warning("Could not embed %s: %s", path, e)
        return ()
