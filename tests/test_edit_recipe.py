"""The preview search, and the two cases the whole design exists to separate.

A dark frame and an out-of-focus frame both score badly now. One of them is
worth keeping. Every test here is ultimately about that distinction.
"""

import numpy as np
import pytest
from synthetic import blown, blurred, dark_but_recoverable, flat, near_black, photo_like

from photoai import edit_recipe


def array(image):
    return np.asarray(image.convert("RGB"), dtype=np.float64)


# --- the quality function ---------------------------------------------------


def test_a_normal_frame_scores_in_the_working_range():
    """Not near 100. A scale whose typical value is 90 cannot discriminate."""
    score = edit_recipe.frame_quality(array(photo_like()))
    assert 30 < score < 90


def test_darkness_costs_quality():
    assert edit_recipe.frame_quality(array(dark_but_recoverable())) < edit_recipe.frame_quality(
        array(photo_like())
    )


def test_blur_costs_quality():
    assert edit_recipe.frame_quality(array(blurred())) < edit_recipe.frame_quality(array(photo_like()))


def test_flatness_costs_quality():
    assert edit_recipe.frame_quality(array(flat())) < edit_recipe.frame_quality(array(photo_like()))


def test_clipping_multiplies_rather_than_merely_subtracting():
    """As an additive term it was a flat bonus on every frame."""
    assert edit_recipe.frame_quality(array(blown())) < edit_recipe.frame_quality(array(photo_like()))


def test_an_empty_array_is_zero_not_an_exception():
    assert edit_recipe.frame_quality(np.zeros((0, 0, 3))) == 0.0


def test_a_black_frame_scores_near_the_bottom():
    assert edit_recipe.frame_quality(array(near_black())) < 15


# --- exposure ---------------------------------------------------------------


def test_a_dark_frame_asks_to_be_raised():
    assert edit_recipe.suggest_ev(array(dark_but_recoverable())) > 0.5


def test_a_correctly_exposed_frame_asks_for_almost_nothing():
    assert abs(edit_recipe.suggest_ev(array(photo_like()))) < 1.0


def test_the_exposure_suggestion_is_bounded():
    """Beyond two stops it is no longer 'a normal edit'."""
    assert abs(edit_recipe.suggest_ev(array(near_black()))) <= edit_recipe.MAX_EV


def test_applying_the_suggested_exposure_improves_the_frame():
    base = array(dark_but_recoverable())
    lifted = edit_recipe.apply_exposure(base, edit_recipe.suggest_ev(base))
    assert edit_recipe.frame_quality(lifted) > edit_recipe.frame_quality(base)


def test_exposure_stays_in_range():
    out = edit_recipe.apply_exposure(array(photo_like()), 2.0)
    assert out.min() >= 0 and out.max() <= 255


# --- the two cases that matter ----------------------------------------------


def test_a_dark_but_recoverable_frame_has_real_uplift():
    """The headline case: unedited is not the same as unusable."""
    recipe = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    assert recipe.uplift > 10
    assert recipe.best_score > recipe.current_score
    assert any("exposure" in step.lower() for step in recipe.human_readable())


def test_a_blurred_frame_gains_far_less_than_a_dark_one():
    """Exposure can be fixed; focus cannot. The search must reflect that."""
    dark = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    soft = edit_recipe.search(blurred(), is_raw=True)
    assert soft.uplift < dark.uplift


def test_uplift_is_never_negative():
    assert edit_recipe.search(photo_like(), is_raw=True).uplift >= 0


def test_a_jpeg_is_credited_with_less_recovery_than_a_raw():
    """The data highlight recovery would use has already been discarded."""
    raw = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    jpeg = edit_recipe.search(dark_but_recoverable(), is_raw=False)
    assert jpeg.uplift < raw.uplift


def test_the_recipe_is_instructions_not_pixels():
    recipe = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    assert all(isinstance(step, str) for step in recipe.human_readable())
    assert recipe.human_readable()


def test_nothing_generative_is_ever_proposed():
    """Marketplace eligibility depends on this staying true."""
    recipe = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    assert not recipe.uses_generative
    text = " ".join(recipe.human_readable()).lower()
    assert "generative" not in text and "generate" not in text


def test_an_already_good_frame_needs_no_edit():
    recipe = edit_recipe.search(photo_like(), is_raw=True)
    assert recipe.uplift < 12


def test_the_recipe_warns_against_aggressive_sharpening():
    recipe = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    assert any("sharpen" in step.lower() for step in recipe.human_readable())


# --- costs ------------------------------------------------------------------


def test_a_heavy_crop_is_charged_for_the_resolution_it_costs():
    cheap = edit_recipe._penalty(keep_fraction=1.0, ev_used=0.0, denoised=False)
    dear = edit_recipe._penalty(keep_fraction=0.5, ev_used=0.0, denoised=False)
    assert dear > cheap


def test_a_large_exposure_push_is_charged_for():
    assert edit_recipe._penalty(1.0, 2.0, False) > edit_recipe._penalty(1.0, 0.3, False)


def test_a_small_exposure_correction_is_free():
    assert edit_recipe._penalty(1.0, 0.4, False) == 0.0


def test_denoising_is_charged_for():
    assert edit_recipe._penalty(1.0, 0.0, True) > edit_recipe._penalty(1.0, 0.0, False)


def test_penalties_come_off_the_uplift():
    recipe = edit_recipe.SearchResult(raw_uplift=20.0, penalties=8.0)
    assert recipe.uplift == 12.0


def test_penalties_cannot_make_uplift_negative():
    assert edit_recipe.SearchResult(raw_uplift=2.0, penalties=50.0).uplift == 0.0


# --- geometry and crops -----------------------------------------------------


# Seed 3 generates a scene with plenty of straight structure -- buildings and a
# horizon. Seeds 0, 4 and 5 generate scenes that are mostly sky and texture,
# which the estimator declines to measure; that case is asserted separately.
STRUCTURED_SEED = 3
UNSTRUCTURED_SEED = 5


def rotated_without_fill(degrees, size=(800, 600), seed=STRUCTURED_SEED):
    """Rotate, then crop the fill away -- a real tilted photo has no gray corners."""
    turned = photo_like(*size, seed=seed).rotate(degrees, expand=False, fillcolor=(120, 120, 120))
    return turned.crop((120, 90, size[0] - 120, size[1] - 90))


def test_a_level_frame_reports_no_tilt():
    assert abs(edit_recipe.estimate_tilt(array(rotated_without_fill(0)))) < 1.0


@pytest.mark.parametrize("degrees", [2.0, 4.0, 6.0, -4.0])
def test_a_rotated_frame_reports_the_angle_it_was_rotated_by(degrees):
    """Presence is not enough: an earlier version returned 0.5 for every input."""
    measured = edit_recipe.estimate_tilt(array(rotated_without_fill(degrees)))
    assert abs(measured) == pytest.approx(abs(degrees), abs=1.5)


def test_the_reported_tilt_opposes_the_rotation_so_the_fix_undoes_it():
    clockwise = edit_recipe.estimate_tilt(array(rotated_without_fill(4.0)))
    anticlockwise = edit_recipe.estimate_tilt(array(rotated_without_fill(-4.0)))
    assert clockwise * anticlockwise < 0


def test_a_scene_without_straight_lines_declines_rather_than_guessing():
    """Foliage and open water have a dominant direction only by accident.

    Before the coherence gate, exactly these scenes returned a confident and
    wrong angle -- which would have put a spurious rotation in the recipe.
    """
    tilted = rotated_without_fill(4.0, seed=UNSTRUCTURED_SEED)
    assert edit_recipe.estimate_tilt(array(tilted)) == 0.0


def test_a_featureless_frame_reports_no_tilt():
    assert edit_recipe.estimate_tilt(np.full((200, 200, 3), 128.0)) == 0.0


def test_a_declined_tilt_puts_no_rotation_in_the_recipe(monkeypatch):
    """A refusal must reach the recipe as silence, not as 'rotate 0 degrees'."""
    monkeypatch.setattr(edit_recipe, "estimate_tilt", lambda _: 0.0)
    recipe = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    assert not any("Straighten" in step for step in recipe.human_readable())


def test_a_measured_tilt_does_reach_the_recipe(monkeypatch):
    monkeypatch.setattr(edit_recipe, "estimate_tilt", lambda _: -3.4)
    recipe = edit_recipe.search(dark_but_recoverable(), is_raw=True)
    steps = recipe.human_readable()
    assert any("Straighten" in step and "3.4" in step for step in steps)
    assert any("clockwise" in step for step in steps)


def test_an_absurd_tilt_is_not_proposed():
    """Beyond a few degrees it is a creative choice, not a mistake."""
    assert abs(edit_recipe.estimate_tilt(array(photo_like()))) <= edit_recipe.MAX_STRAIGHTEN_DEG


def test_a_tiny_image_does_not_crash_tilt_estimation():
    assert edit_recipe.estimate_tilt(np.zeros((4, 4, 3))) == 0.0


def test_crops_never_throw_away_more_than_the_limit():
    base = array(photo_like())
    total = base.shape[0] * base.shape[1]
    for _, (left, top, right, bottom) in edit_recipe.saliency_crops(base):
        assert ((bottom - top) * (right - left)) / total >= edit_recipe.MIN_CROP_KEEP


def test_crops_stay_inside_the_frame():
    base = array(photo_like())
    height, width = base.shape[:2]
    for _, (left, top, right, bottom) in edit_recipe.saliency_crops(base):
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height


def test_a_featureless_frame_proposes_no_saliency_crop():
    assert edit_recipe.saliency_crops(np.zeros((100, 100, 3))) == []


# --- noise ------------------------------------------------------------------


def test_noise_is_measured_where_the_frame_is_flat():
    """Measuring everywhere measures edges, and every photograph has edges."""
    clean = array(photo_like(seed=1))
    rng = np.random.default_rng(3)
    noisy = np.clip(clean + rng.normal(0, 14, clean.shape), 0, 255)
    assert edit_recipe.estimate_noise(noisy) > edit_recipe.estimate_noise(clean)


def test_a_tiny_image_has_no_measurable_noise():
    assert edit_recipe.estimate_noise(np.zeros((8, 8, 3))) == 0.0


@pytest.mark.parametrize("builder", [photo_like, dark_but_recoverable, blurred, flat, blown])
def test_the_search_survives_every_defect_without_raising(builder):
    recipe = edit_recipe.search(builder(), is_raw=True, noisy=True)
    assert 0 <= recipe.best_score <= 100
    assert recipe.uplift >= 0


def test_the_two_recipe_types_are_named_apart():
    """One is the output of a search, the other a declarative edit.

    They were both called EditRecipe, in modules one import apart, and the
    only way to tell which you had was to look at its fields.
    """
    from photoai import edit_schema

    assert edit_recipe.SearchResult is not edit_schema.EditRecipe
    assert not hasattr(edit_recipe, "EditRecipe")
    assert hasattr(edit_recipe.SearchResult(), "raw_uplift")
    assert hasattr(edit_schema.EditRecipe(), "global_adjustments")
