import hashlib
from contextlib import contextmanager

import numpy as np
import pytest
from conftest import CORRUPT_EXIF, NO_EXIF, TRUNCATED_RAW, WITH_EXIF
from PIL import Image

import preview_generator
from preview_generator import PREVIEW_MAX_PX, PreviewGenerationError, generate_preview


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- output shape and format ------------------------------------------------


def test_longest_side_is_clamped_to_the_limit(tmp_path):
    out = generate_preview(WITH_EXIF, "JPEG", tmp_path)
    assert max(Image.open(out).size) == PREVIEW_MAX_PX


def test_output_is_jpeg_regardless_of_input(tmp_path):
    out = generate_preview(WITH_EXIF, "JPEG", tmp_path)
    with Image.open(out) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


def test_output_is_named_after_the_source_stem(tmp_path):
    assert generate_preview(WITH_EXIF, "JPEG", tmp_path).name == "sample_with_exif.jpg"


def test_landscape_aspect_ratio_is_preserved(tmp_path):
    src_w, src_h = Image.open(WITH_EXIF).size
    w, h = Image.open(generate_preview(WITH_EXIF, "JPEG", tmp_path)).size
    assert w > h
    assert abs((w / h) - (src_w / src_h)) < 0.02


def test_portrait_stays_portrait(tmp_path):
    w, h = Image.open(generate_preview(NO_EXIF, "JPEG", tmp_path)).size
    assert h > w
    assert h == PREVIEW_MAX_PX


def test_images_smaller_than_the_limit_are_not_upscaled(tmp_path):
    small = tmp_path / "small.jpg"
    Image.new("RGB", (120, 90), (10, 20, 30)).save(small, "JPEG")
    assert Image.open(generate_preview(small, "JPEG", tmp_path / "out")).size == (120, 90)


def test_missing_output_directory_is_created(tmp_path):
    target = tmp_path / "deeply" / "nested" / "previews"
    assert generate_preview(WITH_EXIF, "JPEG", target).exists()
    assert target.is_dir()


# --- the source file is never touched ---------------------------------------


@pytest.mark.parametrize("fixture", [WITH_EXIF, NO_EXIF, CORRUPT_EXIF])
def test_source_file_is_never_modified(fixture, tmp_path):
    before = _sha256(fixture)
    before_mtime = fixture.stat().st_mtime_ns
    generate_preview(fixture, "JPEG", tmp_path)
    assert _sha256(fixture) == before
    assert fixture.stat().st_mtime_ns == before_mtime


def test_source_is_untouched_even_when_generation_fails(tmp_path):
    before = _sha256(TRUNCATED_RAW)
    with pytest.raises(PreviewGenerationError):
        generate_preview(TRUNCATED_RAW, "RAW", tmp_path)
    assert _sha256(TRUNCATED_RAW) == before


def test_preview_is_written_outside_the_source_directory(tmp_path):
    out = generate_preview(WITH_EXIF, "JPEG", tmp_path)
    assert out.parent == tmp_path
    assert out.parent != WITH_EXIF.parent


# --- colour mode conversion -------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "container"),
    [("RGBA", "png"), ("P", "png"), ("LA", "png"), ("L", "png"), ("CMYK", "tiff")],
)
def test_non_rgb_modes_are_flattened_to_rgb(mode, container, tmp_path):
    src = tmp_path / f"src_{mode}.{container}"
    Image.new("RGB", (300, 200), (200, 40, 90)).convert(mode).save(src)
    with Image.open(generate_preview(src, "JPEG", tmp_path / "out")) as img:
        assert img.mode == "RGB"


# --- failure paths ----------------------------------------------------------


def test_corrupt_file_raises_preview_generation_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    with pytest.raises(PreviewGenerationError, match="Pillow preview failed"):
        generate_preview(bad, "JPEG", tmp_path / "out")


def test_missing_file_raises_preview_generation_error(tmp_path):
    with pytest.raises(PreviewGenerationError):
        generate_preview(tmp_path / "absent.jpg", "JPEG", tmp_path / "out")


def test_undecodable_raw_raises_preview_generation_error(tmp_path):
    with pytest.raises(PreviewGenerationError, match="rawpy failed"):
        generate_preview(TRUNCATED_RAW, "RAW", tmp_path)


# --- RAW happy path, with rawpy stubbed out ---------------------------------


@pytest.fixture
def fake_rawpy(monkeypatch):
    """Stand in for LibRaw so the RAW branch can be exercised without a 34 MB file."""
    calls = {}

    class FakeRaw:
        def postprocess(self, **kwargs):
            calls["postprocess_kwargs"] = kwargs
            return np.full((1200, 1600, 3), 128, dtype=np.uint8)

    @contextmanager
    def fake_imread(path):
        calls["path"] = path
        yield FakeRaw()

    monkeypatch.setattr(preview_generator.rawpy, "imread", fake_imread)
    return calls


def test_raw_preview_is_resized_and_saved(fake_rawpy, tmp_path):
    out = generate_preview(TRUNCATED_RAW, "RAW", tmp_path)
    with Image.open(out) as img:
        assert img.format == "JPEG"
        assert img.size == (PREVIEW_MAX_PX, 384)


def test_raw_preview_uses_camera_white_balance_and_8_bit_output(fake_rawpy, tmp_path):
    generate_preview(TRUNCATED_RAW, "RAW", tmp_path)
    assert fake_rawpy["postprocess_kwargs"] == {"use_camera_wb": True, "output_bps": 8}


def test_raw_path_receives_the_source_as_a_string(fake_rawpy, tmp_path):
    generate_preview(TRUNCATED_RAW, "RAW", tmp_path)
    assert fake_rawpy["path"] == str(TRUNCATED_RAW)
