import numpy as np
import pytest
from conftest import WITH_EXIF
from PIL import Image, ImageDraw, ImageFilter

import technical_filter as tf
from technical_filter import (
    analyze,
    group_bursts,
    load_for_analysis,
    pick_from_burst,
)


def detailed(size=(700, 500), seed=0):
    """An image with real high-frequency content, so blurring it is measurable."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8))


# --- sharpness --------------------------------------------------------------


def test_blurring_an_image_lowers_both_measures():
    sharp = analyze(detailed())
    blurred = analyze(detailed().filter(ImageFilter.GaussianBlur(6)))
    assert blurred.sharpness_global < sharp.sharpness_global
    assert blurred.sharpness_tile < sharp.sharpness_tile


def test_a_flat_frame_has_no_sharpness():
    report = analyze(Image.new("RGB", (600, 400), (128, 128, 128)))
    assert report.sharpness_global == pytest.approx(0.0, abs=1e-6)
    assert report.sharpness_tile == pytest.approx(0.0, abs=1e-6)


def test_tile_sharpness_is_never_below_global():
    """The best region cannot be duller than the whole frame's average."""
    assert analyze(detailed()).sharpness_tile >= analyze(detailed()).sharpness_global


# --- the reason max-tile exists: shallow depth of field ---------------------


def shallow_depth_of_field():
    """Sharp subject in the centre, everything else blurred -- a portrait."""
    base = detailed()
    out = base.filter(ImageFilter.GaussianBlur(6))
    w, h = base.size
    box = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
    out.paste(base.crop(box), box)
    return out


def test_bokeh_survives_a_threshold_that_rejects_a_blurred_frame():
    """The regression this module exists to prevent."""
    bokeh = analyze(shallow_depth_of_field())
    blurred = analyze(detailed().filter(ImageFilter.GaussianBlur(6)))
    sharp = analyze(detailed())

    # By tile there is a threshold separating bokeh from a blurred frame...
    assert blurred.sharpness_tile < bokeh.sharpness_tile
    # ...and bokeh sits in the same order of magnitude as a fully sharp frame.
    assert bokeh.sharpness_tile > sharp.sharpness_tile / 10


def test_bokeh_is_not_rejected():
    assert analyze(shallow_depth_of_field()).passed


def test_a_fully_blurred_frame_is_rejected():
    report = analyze(detailed().filter(ImageFilter.GaussianBlur(8)))
    assert not report.passed
    assert any("out of focus" in r for r in report.rejected_for)


# --- blur ratio: the metric that survives low-contrast scenes ---------------


def low_contrast_scene():
    """Fog: real structure, but almost no tonal range. This is a good photo."""
    base = photo_like()
    flat = Image.blend(base, Image.new("RGB", base.size, (150, 155, 165)), 0.90)
    return flat


def test_a_soft_scene_is_not_mistaken_for_a_soft_lens():
    """Fog and haze score near zero on absolute sharpness but are in focus.

    On the real archive this exact confusion rejected two sunsets.
    """
    fog = analyze(low_contrast_scene())
    assert fog.blur_ratio >= tf.MIN_BLUR_RATIO
    assert fog.passed


def test_blur_ratio_collapses_when_the_frame_is_actually_defocused():
    defocused = analyze(low_contrast_scene().filter(ImageFilter.GaussianBlur(6)))
    assert defocused.blur_ratio < tf.MIN_BLUR_RATIO
    assert not defocused.passed


def test_blur_ratio_separates_focus_regardless_of_scene_contrast():
    """The property absolute sharpness lacks: contrast cancels out."""
    contrasty = analyze(photo_like())
    faded = analyze(low_contrast_scene())

    # Wildly different absolute detail...
    assert contrasty.sharpness_tile > faded.sharpness_tile * 5
    # ...yet both read as in focus, and both clear the threshold.
    assert contrasty.blur_ratio >= tf.MIN_BLUR_RATIO
    assert faded.blur_ratio >= tf.MIN_BLUR_RATIO


def test_a_flat_frame_does_not_divide_by_zero():
    report = analyze(Image.new("RGB", (400, 300), (128, 128, 128)))
    assert report.blur_ratio >= 0.0


def test_absolute_sharpness_is_reported_but_never_rejects():
    """looks_soft is a diagnostic; only blur ratio and clipping decide."""
    fog = analyze(low_contrast_scene())
    assert fog.looks_soft or fog.sharpness_tile < 500
    assert fog.passed


# --- clipping ---------------------------------------------------------------


def test_a_black_frame_reads_as_crushed_shadows():
    report = analyze(Image.new("RGB", (400, 300), (0, 0, 0)))
    assert report.clipped_shadows == pytest.approx(1.0)
    assert report.clipped_highlights == pytest.approx(0.0)
    assert any("shadows" in r for r in report.rejected_for)


def test_a_white_frame_reads_as_blown_highlights():
    report = analyze(Image.new("RGB", (400, 300), (255, 255, 255)))
    assert report.clipped_highlights == pytest.approx(1.0)
    assert any("highlights" in r for r in report.rejected_for)


def test_a_well_exposed_frame_clips_nothing():
    report = analyze(detailed())
    assert report.clipped_shadows < 0.01
    assert report.clipped_highlights < 0.01


def test_partial_clipping_is_measured_as_a_fraction():
    img = Image.new("RGB", (400, 300), (128, 128, 128))
    ImageDraw.Draw(img).rectangle([0, 0, 399, 74], fill=(255, 255, 255))  # top quarter
    assert analyze(img).clipped_highlights == pytest.approx(0.25, abs=0.03)


# --- perceptual hash --------------------------------------------------------


def test_identical_frames_hash_identically():
    assert analyze(detailed(seed=1)).phash == analyze(detailed(seed=1)).phash


def test_different_scenes_hash_differently():
    assert analyze(detailed(seed=1)).phash != analyze(detailed(seed=2)).phash


def photo_like(width=900, height=700):
    """A frame with the low-frequency structure pHash actually keys on.

    Neither noise nor a plain gradient works here: the first has no perceptual
    structure and the second has almost none, so both hash unstably under a
    small shift. Real photographs have large tonal regions and distinct
    subjects, which is what this builds.
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        draw.line([(0, y), (width, y)], fill=(30 + y * 90 // height, 60 + y * 120 // height, 140))
    draw.ellipse([width * 0.15, height * 0.20, width * 0.45, height * 0.55], fill=(235, 190, 90))
    draw.rectangle([width * 0.55, height * 0.35, width * 0.85, height * 0.75], fill=(40, 45, 60))
    draw.polygon(
        [(width * 0.10, height * 0.90), (width * 0.35, height * 0.65), (width * 0.60, height * 0.92)],
        fill=(200, 90, 70),
    )
    return img


def test_a_slight_recompose_still_hashes_close():
    """A burst sibling is the same scene framed a few pixels over.

    The shift is a crop from a larger canvas, not an AFFINE transform -- the
    latter fills the vacated edge with black, which is itself a big perceptual
    change and would be testing the wrong thing.
    """
    canvas = photo_like(940, 740)
    base = canvas.crop((0, 0, 900, 700))
    nudged = canvas.crop((8, 8, 908, 708))
    assert tf._hamming(analyze(base).phash, analyze(nudged).phash) <= tf.PHASH_DISTANCE


def test_unrelated_frames_are_not_burst_siblings():
    """Guards the threshold from the other side -- unrelated frames stay apart."""
    a, b = analyze(detailed(seed=11)), analyze(detailed(seed=12))
    assert tf._hamming(a.phash, b.phash) > tf.PHASH_DISTANCE


@pytest.mark.parametrize(("a", "b", "expected"), [
    ("ffffffffffffffff", "ffffffffffffffff", 0),
    ("0000000000000000", "0000000000000001", 1),
    ("0000000000000000", "000000000000000f", 4),
    (None, "0000000000000000", 999),
    ("short", "0000000000000000", 999),
    ("zzzz", "0000", 999),
])
def test_hamming(a, b, expected):
    assert tf._hamming(a, b) == expected


# --- burst grouping ---------------------------------------------------------


def frame(name, when, phash, tile=100.0):
    return {"filename": name, "date_shot": when, "phash": phash, "sharpness_tile": tile}


def test_a_burst_collapses_into_one_group():
    burst = [
        frame("a.RW2", "2026-03-16T15:44:16", "ffffffffffffffff"),
        frame("b.RW2", "2026-03-16T15:44:17", "ffffffffffffffff"),
        frame("c.RW2", "2026-03-16T15:44:18", "fffffffffffffffe"),
    ]
    groups = group_bursts(burst)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_similar_frames_far_apart_in_time_are_not_a_burst():
    groups = group_bursts([
        frame("a.RW2", "2026-03-16T15:44:16", "ffffffffffffffff"),
        frame("b.RW2", "2026-03-16T16:20:00", "ffffffffffffffff"),
    ])
    assert len(groups) == 2


def test_consecutive_but_different_scenes_are_not_a_burst():
    groups = group_bursts([
        frame("a.RW2", "2026-03-16T15:44:16", "ffffffffffffffff"),
        frame("b.RW2", "2026-03-16T15:44:17", "0000000000000000"),
    ])
    assert len(groups) == 2


def test_frames_without_a_timestamp_are_never_grouped():
    """Without a time there is no way to tell a burst from two similar photos."""
    groups = group_bursts([
        frame("a.RW2", None, "ffffffffffffffff"),
        frame("b.RW2", None, "ffffffffffffffff"),
    ])
    assert len(groups) == 2


def test_grouping_is_independent_of_input_order():
    frames = [
        frame("c.RW2", "2026-03-16T15:44:18", "ffffffffffffffff"),
        frame("a.RW2", "2026-03-16T15:44:16", "ffffffffffffffff"),
        frame("b.RW2", "2026-03-16T15:44:17", "ffffffffffffffff"),
    ]
    assert len(group_bursts(frames)) == 1
    assert len(group_bursts(list(reversed(frames)))) == 1


def test_every_frame_survives_grouping():
    frames = [
        frame("a.RW2", "2026-03-16T15:44:16", "ffffffffffffffff"),
        frame("b.RW2", "2026-03-16T15:44:17", "ffffffffffffffff"),
        frame("c.RW2", "2026-03-16T18:00:00", "0000000000000000"),
        frame("d.RW2", None, "1234567812345678"),
    ]
    assert sum(len(g) for g in group_bursts(frames)) == len(frames)


def test_the_sharpest_frame_wins_the_burst():
    group = [
        frame("soft.RW2", "2026-03-16T15:44:16", "ffffffffffffffff", tile=120.0),
        frame("sharp.RW2", "2026-03-16T15:44:17", "ffffffffffffffff", tile=980.0),
        frame("softer.RW2", "2026-03-16T15:44:18", "ffffffffffffffff", tile=90.0),
    ]
    assert pick_from_burst(group)["filename"] == "sharp.RW2"


def test_burst_pick_tolerates_a_missing_sharpness():
    group = [
        {"filename": "unmeasured.RW2"},
        frame("measured.RW2", "2026-03-16T15:44:17", "ffffffffffffffff", tile=10.0),
    ]
    assert pick_from_burst(group)["filename"] == "measured.RW2"


# --- decoding ---------------------------------------------------------------


def test_a_jpeg_can_be_measured_end_to_end():
    report = analyze(load_for_analysis(WITH_EXIF, "JPEG"))
    assert report.sharpness_tile > 0
    assert len(report.phash) == 16


def test_measuring_does_not_modify_the_source_image():
    """analyze() thumbnails internally; the caller's image must be untouched."""
    img = detailed()
    before = img.size
    analyze(img)
    assert img.size == before
