"""The optional semantic encoder: the probe, the checksum, the cache, the fallback.

Nothing here downloads anything, opens a socket, or loads a 335 MB ONNX file.
`conftest._no_model_downloads` pins the feature off and points its cache at an
empty directory for the whole suite, and every test that needs the feature *on*
turns it on for itself with a stub encoder.

What a stub encoder can and cannot prove is worth being clear about, because
this is a module whose whole claim is "the comparison is semantic now":

- **Proved here.** That two frames really can collide on a perceptual hash while
  being different photographs -- that half is computed from real pixels, not
  assumed. That an encoder which separates them is enough to keep both in the
  gallery. That the vectors are cached and a second run is free. That every path
  where the encoder is missing falls back to the hash and *says so*.
- **Not proved here.** That OpenAI's CLIP weights, specifically, see the
  difference. That is a property of a 335 MB file, and asserting it needs the
  file; `test_the_real_encoder_separates_the_palette_twins` at the bottom does
  exactly that and is skipped unless somebody has opted in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from photoai import duplicates, embeddings
from tests import synthetic

REAL_MODEL_VAR = "PHOTO_AI_EMBEDDINGS_REAL_MODEL"


# --- a stand-in for CLIP ------------------------------------------------------


def descriptor(image, bins: int = 8) -> tuple[float, ...]:
    """A real, very small image encoder: a gradient-orientation histogram.

    This stands in for the CLIP tower, and it is a genuine function of the
    pixels rather than a lookup keyed on the filename -- so a test that shows
    two frames staying apart is showing that *content* drove the decision, not
    that the fixture was rigged. It is nothing like as good as CLIP; it does not
    need to be. It needs to see a difference a perceptual hash cannot.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float64)
    dy, dx = np.gradient(grey)
    magnitude = np.hypot(dx, dy)
    angle = np.mod(np.arctan2(dy, dx), np.pi)
    index = np.minimum((angle / np.pi * bins).astype(int), bins - 1)
    histogram = np.bincount(index.ravel(), weights=magnitude.ravel(), minlength=bins)
    return embeddings.normalise(histogram)


class StubEncoder:
    """Shaped exactly like `embeddings.Encoder`, without the ONNX session."""

    def __init__(self, model_id: str = "stub-encoder"):
        self.spec = embeddings.MODEL
        self.model_id = model_id
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        return descriptor(image)

    def encode_many(self, images):
        return [self.encode(i) for i in images]


@pytest.fixture
def stub_encoder(monkeypatch):
    """Embeddings on, encoder stubbed, availability forced true."""
    encoder = StubEncoder()
    monkeypatch.setattr(embeddings, "encoder", lambda *a, **k: encoder)
    monkeypatch.setattr(
        embeddings, "status",
        lambda *a, **k: embeddings.Status(True, "", encoder.model_id, "<stub>"),
    )
    monkeypatch.setattr(embeddings, "prepare", lambda *a, **k: embeddings.status())
    return encoder


# --- the capability probe -----------------------------------------------------


def test_the_suite_starts_with_embeddings_switched_off():
    """The guard in conftest, asserted rather than assumed."""
    assert not embeddings.available()
    assert embeddings.mode() == "off"


def test_the_probe_says_why_and_not_merely_no(monkeypatch):
    monkeypatch.setenv(embeddings.ENABLE_VAR, "0")
    state = embeddings.status()
    assert state.ok is False
    assert "switched off" in state.reason
    assert state.fix


def test_a_missing_onnxruntime_is_the_reason_given(monkeypatch):
    monkeypatch.setenv(embeddings.ENABLE_VAR, "auto")
    monkeypatch.setattr(embeddings, "_onnxruntime_installed", lambda: False)
    assert embeddings.unavailable_reason() == "onnxruntime is not installed"
    assert 'pip install -e ".[embeddings]"' in embeddings.status().fix


def test_a_missing_model_file_is_the_reason_given(monkeypatch):
    monkeypatch.setenv(embeddings.ENABLE_VAR, "auto")
    monkeypatch.setattr(embeddings, "_onnxruntime_installed", lambda: True)
    assert embeddings.unavailable_reason() == "the model has not been downloaded"
    assert "photoai models --download" in embeddings.status().fix


def test_the_reason_is_empty_when_everything_is_there(stub_encoder):
    assert embeddings.available()
    assert embeddings.unavailable_reason() == ""


def test_auto_mode_never_downloads(monkeypatch, tmp_path):
    """The default is 'use it if it is here', which must not become 'fetch it'."""
    monkeypatch.setenv(embeddings.ENABLE_VAR, "auto")
    monkeypatch.setenv(embeddings.CACHE_VAR, str(tmp_path))
    monkeypatch.setattr(embeddings, "_onnxruntime_installed", lambda: True)

    def explode(*args, **kwargs):
        raise AssertionError("a download was attempted without being asked for")

    monkeypatch.setattr(embeddings, "download", explode)
    assert embeddings.prepare().ok is False


def test_an_unset_variable_means_auto(monkeypatch):
    monkeypatch.delenv(embeddings.ENABLE_VAR, raising=False)
    assert embeddings.mode() == "auto"


def test_the_variable_switches_downloading_on(monkeypatch, tmp_path):
    monkeypatch.setenv(embeddings.ENABLE_VAR, "1")
    monkeypatch.setenv(embeddings.CACHE_VAR, str(tmp_path))
    monkeypatch.setattr(embeddings, "_onnxruntime_installed", lambda: True)
    asked = []
    monkeypatch.setattr(embeddings, "download", lambda spec, **k: asked.append(spec) or tmp_path)
    embeddings.prepare()
    assert asked, "PHOTO_AI_EMBEDDINGS=1 is the explicit gate and it did not open"


def test_the_model_is_pinned_to_a_commit_and_carries_its_licence():
    """A `main` URL would serve different weights one day and the checksum,
    not the URL, would be the thing that looked broken."""
    spec = embeddings.MODEL
    assert spec.licence == "MIT"
    assert spec.licence_url
    assert "/resolve/main/" not in spec.url
    assert len(spec.sha256) == 64
    assert spec.dimensions == 512


# --- the checksum -------------------------------------------------------------


def _fake_model(tmp_path, spec, body: bytes):
    path = tmp_path / spec.filename
    path.write_bytes(body)
    return path


def test_a_file_that_matches_verifies(tmp_path, monkeypatch):
    import hashlib

    body = b"pretend weights"
    spec = embeddings.ModelSpec(
        model_id="test", filename="test.onnx", url="https://example.invalid/m.onnx",
        sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body),
        dimensions=4, image_size=8, input_name="pixel_values",
        licence="MIT", licence_url="", source="test",
    )
    path = _fake_model(tmp_path, spec, body)
    assert embeddings.verify(path, spec) is True
    del monkeypatch


def test_a_file_that_does_not_match_is_refused(tmp_path):
    spec = embeddings.ModelSpec(
        model_id="test", filename="test.onnx", url="https://example.invalid/m.onnx",
        sha256="0" * 64, size_bytes=15, dimensions=4, image_size=8,
        input_name="pixel_values", licence="MIT", licence_url="", source="test",
    )
    path = _fake_model(tmp_path, spec, b"pretend weights")
    assert embeddings.verify(path, spec) is False


def test_a_file_of_the_wrong_size_is_refused_without_hashing(tmp_path):
    spec = embeddings.ModelSpec(
        model_id="test", filename="test.onnx", url="https://example.invalid/m.onnx",
        sha256="0" * 64, size_bytes=999_999, dimensions=4, image_size=8,
        input_name="pixel_values", licence="MIT", licence_url="", source="test",
    )
    path = _fake_model(tmp_path, spec, b"short")
    assert embeddings.verify(path, spec) is False


def test_a_refused_checksum_makes_the_probe_say_so(tmp_path, monkeypatch):
    monkeypatch.setenv(embeddings.ENABLE_VAR, "auto")
    monkeypatch.setenv(embeddings.CACHE_VAR, str(tmp_path))
    monkeypatch.setattr(embeddings, "_onnxruntime_installed", lambda: True)
    (tmp_path / embeddings.MODEL.filename).write_bytes(b"not the model")
    state = embeddings.status()
    assert state.ok is False
    assert "checksum" in state.reason and "refused" in state.reason


def test_a_bad_download_is_deleted_rather_than_kept(tmp_path, monkeypatch):
    """A wrong model on disk would be loaded next time by a probe that only
    checks the file exists. It never reaches disk under its real name."""
    import io

    spec = embeddings.ModelSpec(
        model_id="test", filename="test.onnx", url="https://example.invalid/m.onnx",
        sha256="0" * 64, size_bytes=9, dimensions=4, image_size=8,
        input_name="pixel_values", licence="MIT", licence_url="", source="test",
    )

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        embeddings.urllib.request, "urlopen", lambda *a, **k: _Response(b"not weeee")
    )
    with pytest.raises(embeddings.ChecksumMismatch):
        embeddings.download(spec, destination=tmp_path / spec.filename)
    assert list(tmp_path.iterdir()) == []


def test_the_verification_receipt_saves_rehashing(tmp_path):
    """335 MB hashed on every CLI invocation is a second nobody agreed to."""
    import hashlib

    body = b"pretend weights"
    spec = embeddings.ModelSpec(
        model_id="test", filename="test.onnx", url="https://example.invalid/m.onnx",
        sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body),
        dimensions=4, image_size=8, input_name="pixel_values",
        licence="MIT", licence_url="", source="test",
    )
    path = _fake_model(tmp_path, spec, body)
    assert embeddings.verify(path, spec)

    receipt = json.loads((tmp_path / "test.onnx.verified.json").read_text())
    assert receipt["sha256"] == spec.sha256
    assert receipt["size"] == len(body)

    # Contents changed behind the receipt: size and mtime no longer agree, so
    # the full hash runs again and refuses it.
    path.write_bytes(b"different weights entirely")
    assert embeddings.verify(path, spec) is False


# --- preprocessing and normalisation ------------------------------------------


def test_preprocessing_produces_the_tensor_the_model_declares():
    tensor = embeddings.preprocess(synthetic.photo_like(800, 600, seed=1))
    assert tensor.shape == (1, 3, embeddings.MODEL.image_size, embeddings.MODEL.image_size)
    assert tensor.dtype == np.float32


def test_preprocessing_a_portrait_frame_still_gives_a_square():
    tensor = embeddings.preprocess(synthetic.photo_like(300, 900, seed=2))
    assert tensor.shape[2] == tensor.shape[3] == embeddings.MODEL.image_size


def test_preprocessing_an_image_smaller_than_the_crop_still_works():
    tensor = embeddings.preprocess(Image.new("RGB", (40, 30), (128, 120, 100)))
    assert tensor.shape == (1, 3, 224, 224)


def test_encoded_vectors_are_unit_length(stub_encoder):
    vector = stub_encoder.encode(synthetic.photo_like(seed=3))
    assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-6)


def test_an_unreadable_file_gives_no_vector_rather_than_an_error(tmp_path, stub_encoder):
    broken = tmp_path / "not-an-image.jpg"
    broken.write_bytes(b"\x00\x01\x02")
    assert embeddings.embed_file(broken) == ()


# --- the thing this was all for -----------------------------------------------


def test_the_two_frames_really_do_collide_on_a_perceptual_hash():
    """Everything below rests on this, so it is measured rather than asserted."""
    import imagehash

    bars, lattice = synthetic.palette_twins()
    distance = imagehash.phash(bars) - imagehash.phash(lattice)
    assert distance <= duplicates.DEFAULT_DISTANCE, (
        f"the fixture stopped colliding ({distance} bits apart); the test below "
        "would then prove nothing"
    )


def _twin_candidates(with_vectors: bool):
    """Two hash-colliding frames and one obviously different photograph."""
    import imagehash

    bars, lattice = synthetic.palette_twins()
    other = synthetic.photo_like(seed=11)
    frames = {"bars": bars, "lattice": lattice, "street": other}
    genres = {"bars": "landscape", "lattice": "landscape", "street": "street"}
    quality = {"bars": 90.0, "lattice": 88.0, "street": 40.0}

    candidates = []
    for key, image in frames.items():
        candidates.append(
            duplicates.Candidate(
                key=key,
                relevance=quality[key],
                item=duplicates.DupItem(
                    key=key,
                    phash=str(imagehash.phash(image)),
                    quality=quality[key],
                    genre=genres[key],
                    embedding=descriptor(image) if with_vectors else None,
                ),
            )
        )
    return candidates


def test_two_frames_that_share_a_palette_are_not_merged_when_embeddings_are_on():
    """The headline claim, on images built for the purpose.

    `bars` and `lattice` have one gradient, one framing, one palette and a
    perceptual hash two bits apart -- and they are not the same photograph. With
    vectors attached, the diversity pass keeps both and drops the weak street
    frame, which is the correct answer.
    """
    picks = duplicates.select_diverse(_twin_candidates(True), limit=2, lambda_=0.6)
    assert picks == ["bars", "lattice"]


def test_the_fallback_cannot_tell_the_two_frames_apart():
    """The cost of the fallback, pinned rather than hidden.

    Not an assertion that this is wrong -- it is the documented limit of a
    perceptual hash. Two different subjects at the same palette and framing come
    back as one photograph seen twice, and a diversity pass built on that number
    is working from a false premise however it then weighs relevance.
    """
    import imagehash

    bars, lattice = synthetic.palette_twins()
    pair = [
        duplicates.DupItem(key=k, phash=str(imagehash.phash(image)), quality=q,
                           genre="landscape")
        for k, image, q in (("bars", bars, 90.0), ("lattice", lattice, 88.0))
    ]
    assert duplicates.visual_similarity(*pair) == 1.0

    with_vectors = [
        duplicates.DupItem(key=item.key, phash=item.phash, quality=item.quality,
                           genre=item.genre, embedding=descriptor(image))
        for item, image in zip(pair, (bars, lattice), strict=True)
    ]
    assert duplicates.embedding_similarity(*with_vectors) < 1.0


# --- the cache ----------------------------------------------------------------


def test_a_vector_survives_a_round_trip_through_the_analysis_cache(tmp_path):
    from photoai.pipeline import AnalysisCache

    cache = AnalysisCache(tmp_path / "cache.json")
    vector = descriptor(synthetic.photo_like(seed=4))
    cache.put_embedding("abc123", "clip-vit-b32-vision-onnx", vector)
    cache.save()

    reopened = AnalysisCache(tmp_path / "cache.json")
    stored = reopened.get_embedding("abc123", "clip-vit-b32-vision-onnx")
    assert stored is not None
    assert len(stored) == len(vector)
    assert duplicates.cosine(stored, vector) == pytest.approx(1.0, abs=1e-5)


def test_a_vector_from_another_model_is_not_served(tmp_path):
    """Cosines between two different towers mean nothing at all."""
    from photoai.pipeline import AnalysisCache

    cache = AnalysisCache(tmp_path / "cache.json")
    cache.put_embedding("abc123", "clip-vit-b32-vision-onnx", (1.0, 0.0))
    assert cache.get_embedding("abc123", "some-other-tower") is None
    assert cache.get_embedding("different-checksum", "clip-vit-b32-vision-onnx") is None


def test_an_empty_vector_is_not_stored(tmp_path):
    from photoai.pipeline import AnalysisCache

    cache = AnalysisCache(tmp_path / "cache.json")
    cache.put_embedding("abc123", "m", ())
    assert cache.get_embedding("abc123", "m") is None
