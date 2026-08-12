"""The fixable/partial/unrecoverable split the whole scoring model rests on."""

import pytest

from issues import (
    FIXABILITY,
    Fixability,
    IssueCode,
    IssueSet,
    detect_photo_issues,
    summarise,
)


class Report:
    """Stands in for a technical_filter.TechnicalReport."""

    def __init__(self, blur_ratio=40.0, clipped_highlights=0.0, clipped_shadows=0.0):
        self.blur_ratio = blur_ratio
        self.clipped_highlights = clipped_highlights
        self.clipped_shadows = clipped_shadows


def found_for(report=None, **kwargs):
    defaults = {
        "megapixels": 24.0,
        "mean_luma": 118.0,
        "stddev_luma": 55.0,
        "channel_means": (120.0, 118.0, 116.0),
        "noise_estimate": 1.0,
        "is_raw": True,
    }
    return detect_photo_issues(report or Report(), **{**defaults, **kwargs})


# --- the taxonomy -----------------------------------------------------------


def test_every_issue_code_has_a_fixability():
    """A code with no classification would silently score as recoverable."""
    assert set(FIXABILITY) == set(IssueCode)


@pytest.mark.parametrize(
    "code",
    [IssueCode.UNDEREXPOSED, IssueCode.COLOR_CAST, IssueCode.TILTED_HORIZON, IssueCode.WEAK_CROP],
)
def test_slider_problems_are_fixable(code):
    assert FIXABILITY[code] is Fixability.FIXABLE


@pytest.mark.parametrize(
    "code",
    [IssueCode.MISSED_FOCUS, IssueCode.SEVERE_MOTION_BLUR, IssueCode.CORRUPT_FILE, IssueCode.NO_USABLE_SEGMENT],
)
def test_information_that_is_gone_is_unrecoverable(code):
    assert FIXABILITY[code] is Fixability.UNRECOVERABLE


@pytest.mark.parametrize(
    "code", [IssueCode.MODERATE_SHAKE, IssueCode.HEAVY_NOISE, IssueCode.SOFT_FOCUS]
)
def test_problems_with_a_cost_are_partial(code):
    assert FIXABILITY[code] is Fixability.PARTIAL


def test_only_unrecoverable_issues_block():
    assert not IssueSet([]).has_blocker
    blocked = IssueSet()
    blocked.add(IssueCode.MISSED_FOCUS)
    assert blocked.has_blocker


def test_a_set_of_only_fixable_issues_does_not_block():
    routine = IssueSet()
    routine.add(IssueCode.UNDEREXPOSED)
    routine.add(IssueCode.COLOR_CAST)
    assert not routine.has_blocker


# --- detection --------------------------------------------------------------


def test_a_clean_frame_has_no_issues():
    assert len(found_for()) == 0


def test_missed_focus_is_detected_from_the_blur_ratio():
    found = found_for(Report(blur_ratio=1.2))
    assert IssueCode.MISSED_FOCUS in found.codes()
    assert found.has_blocker


def test_slightly_soft_is_partial_rather_than_fatal():
    found = found_for(Report(blur_ratio=4.0))
    assert IssueCode.SOFT_FOCUS in found.codes()
    assert not found.has_blocker


def test_a_foggy_scene_is_not_called_out_of_focus():
    """An earlier threshold rejected fog and two hazy sunsets from a real archive."""
    found = found_for(Report(blur_ratio=7.0), stddev_luma=30.0)
    assert IssueCode.MISSED_FOCUS not in found.codes()


def test_underexposure_is_fixable():
    found = found_for(mean_luma=44.0)
    assert IssueCode.UNDEREXPOSED in found.codes()
    assert not found.has_blocker


def test_overexposure_is_detected():
    assert IssueCode.OVEREXPOSED in found_for(mean_luma=215.0).codes()


def test_flat_contrast_is_fixable():
    found = found_for(stddev_luma=18.0)
    assert IssueCode.FLAT_CONTRAST in found.codes()
    assert found.fixable


def test_a_colour_cast_is_detected_but_not_certain():
    found = found_for(channel_means=(180.0, 120.0, 110.0))
    cast = [i for i in found if i.code is IssueCode.COLOR_CAST]
    assert cast and cast[0].certainty < 1.0


def test_mild_and_heavy_noise_are_graded_differently():
    assert IssueCode.MILD_NOISE in found_for(noise_estimate=4.0).codes()
    assert IssueCode.HEAVY_NOISE in found_for(noise_estimate=12.0).codes()


def test_a_little_clipping_is_partial_not_fatal():
    found = found_for(Report(clipped_highlights=0.08))
    assert IssueCode.SOME_CLIPPED_HIGHLIGHTS in found.codes()
    assert not found.has_blocker


def test_a_mostly_white_frame_is_unrecoverable():
    found = found_for(Report(clipped_highlights=0.85), is_raw=False)
    assert IssueCode.BLOWN_HIGHLIGHTS in found.codes()
    assert found.has_blocker


def test_raw_is_given_more_highlight_latitude_than_jpeg():
    """The rendered preview cannot show the stop or two the RAW still holds."""
    report = Report(clipped_highlights=0.40)
    assert IssueCode.BLOWN_HIGHLIGHTS in found_for(report, is_raw=False).codes()
    assert IssueCode.BLOWN_HIGHLIGHTS not in found_for(report, is_raw=True).codes()


def test_a_night_frame_is_not_called_crushed():
    """Most of a night photograph is legitimately black."""
    found = found_for(Report(clipped_shadows=0.5), mean_luma=45.0)
    assert IssueCode.CRUSHED_SHADOWS not in found.codes()


def test_a_frame_that_is_almost_entirely_black_is_unrecoverable():
    assert IssueCode.CRUSHED_SHADOWS in found_for(Report(clipped_shadows=0.85)).codes()


def test_a_truly_unusable_resolution_is_unrecoverable():
    found = found_for(megapixels=0.4)
    assert IssueCode.INSUFFICIENT_RESOLUTION in found.codes()
    assert found.has_blocker


def test_being_below_one_marketplace_floor_is_not_a_defect():
    """3 MP is fine for editorial, for print at size, and for the portfolio."""
    assert IssueCode.INSUFFICIENT_RESOLUTION not in found_for(megapixels=3.0).codes()


# --- reporting --------------------------------------------------------------


def test_issues_group_into_the_three_lists_the_ui_prints():
    found = IssueSet()
    found.add(IssueCode.UNDEREXPOSED, "mean luma 44")
    found.add(IssueCode.HEAVY_NOISE, "sigma 9")
    found.add(IssueCode.MISSED_FOCUS, "blur ratio 1.1")
    grouped = summarise(found)
    assert len(grouped["fixable"]) == 1
    assert len(grouped["partially_fixable"]) == 1
    assert len(grouped["unrecoverable"]) == 1


def test_an_issue_describes_itself_with_its_measurement():
    found = IssueSet()
    found.add(IssueCode.UNDEREXPOSED, "mean luma 44")
    assert found.issues[0].describe() == "underexposed: mean luma 44"


def test_an_issue_without_detail_still_describes_itself():
    found = IssueSet()
    found.add(IssueCode.MISSED_FOCUS)
    assert found.issues[0].describe() == "missed_focus"


def test_an_empty_set_reports_three_empty_lists():
    assert summarise(IssueSet()) == {
        "fixable": [],
        "partially_fixable": [],
        "unrecoverable": [],
    }
