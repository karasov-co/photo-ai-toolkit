"""Parsing one Stage 2 reply, strictly.

These moved from `test_routing.py` when the rest of that module was deleted:
`assign_destinations`, `Destination`, `RoutingConfig`, `_pick_flagship`,
`blocks_commercial_stock` and `summarise` had no caller anywhere in the
pipeline, and their tests were the only thing keeping them alive.

The parser survived because it is the one part that runs.
"""

import pytest

from assessment_parser import (
    AssessmentParseError,
    Genre,
    Recover,
    parse_assessment,
)

VALID = {
    "genre": "street",
    "axis_a": 72,
    "axis_b": 40,
    "axis_c": 55,
    "recover": "easy",
    "faces": False,
    "brand_mark": False,
    "note": "lift shadows, crop left edge",
}


def test_parses_a_well_formed_payload():
    a = parse_assessment(VALID, "P1.RW2")
    assert a.genre is Genre.STREET
    assert (a.axis_a, a.axis_b, a.axis_c) == (72, 40, 55)
    assert a.recover is Recover.EASY
    assert a.faces is False


@pytest.mark.parametrize("key", sorted({"genre", "axis_a", "axis_b", "axis_c", "recover", "faces"}))
def test_a_missing_required_key_is_an_error(key):
    payload = {k: v for k, v in VALID.items() if k != key}
    with pytest.raises(AssessmentParseError, match=key):
        parse_assessment(payload, "P1.RW2")


def test_an_unknown_genre_falls_back_rather_than_failing_the_frame():
    a = parse_assessment({**VALID, "genre": "macro-ish"}, "P1.RW2")
    assert a.genre is Genre.OTHER


@pytest.mark.parametrize(
    ("raw", "expected"), [(150, 100), (-20, 0), ("83", 83), (72.6, 73), (None, 0), ("x", 0)]
)
def test_axis_values_are_clamped_into_range(raw, expected):
    assert parse_assessment({**VALID, "axis_a": raw}, "P1.RW2").axis_a == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("true", True), ("false", False), ("no", False), (None, True)],
)
def test_release_flags_fail_safe_when_unclear(raw, expected):
    """Guessing 'no face' on an uncertain frame is the expensive way to be wrong."""
    assert parse_assessment({**VALID, "faces": raw}, "P1.RW2").faces is expected


def test_an_unparseable_face_flag_stays_pessimistic():
    """A face nobody could confirm is treated as present: the cheap direction."""
    assert parse_assessment({**VALID, "axis_a": 100, "faces": None}, "P1.RW2").faces is True


def test_an_unparseable_brand_flag_does_not_invent_a_brand():
    """The opposite default, for the opposite reason.

    Guessing "brand" on every unanswered frame removed street photography from
    commercial use, which is worse than the error it was avoiding.
    """
    assert parse_assessment({**VALID, "brand_mark": None}, "P1.RW2").brand_mark is False


def test_signage_is_absent_unless_the_model_says_otherwise():
    assert parse_assessment(VALID, "P1.RW2").signage_text is False
    assert parse_assessment({**VALID, "signage_text": True}, "P1.RW2").signage_text is True


def test_an_older_reply_carrying_logos_is_read_as_a_brand_mark():
    """Cached replies from before the split conflated the two; assume the worse."""
    payload = {k: v for k, v in VALID.items() if k != "brand_mark"}
    assert parse_assessment({**payload, "logos": True}, "P1.RW2").brand_mark is True


def test_the_note_is_trimmed_to_twelve_words():
    long_note = " ".join(f"word{i}" for i in range(30))
    assert len(parse_assessment({**VALID, "note": long_note}, "P1.RW2").note.split()) == 12


def test_a_missing_note_is_empty_not_an_error():
    payload = {k: v for k, v in VALID.items() if k != "note"}
    assert parse_assessment(payload, "P1.RW2").note == ""


def test_stage_0_rejections_ride_along_into_the_assessment():
    a = parse_assessment(VALID, "P1.RW2", technically_rejected_for=["blown highlights (80%)"])
    assert a.technically_rejected
    assert a.technically_rejected_for == ["blown highlights (80%)"]
