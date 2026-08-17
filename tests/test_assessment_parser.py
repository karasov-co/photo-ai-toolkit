"""Parsing one Stage 2 reply, strictly.

These moved from `test_routing.py` when the rest of that module was deleted:
`assign_destinations`, `Destination`, `RoutingConfig`, `_pick_flagship`,
`blocks_commercial_stock` and `summarise` had no caller anywhere in the
pipeline, and their tests were the only thing keeping them alive.

The parser survived because it is the one part that runs.
"""

import pytest

from photoai.assessment_parser import (
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
    "note": "lift shadows, crop left edge",
}


def test_parses_a_well_formed_payload():
    a = parse_assessment(VALID, "P1.RW2")
    assert a.genre is Genre.STREET
    assert (a.axis_a, a.axis_b, a.axis_c) == (72, 40, 55)
    assert a.recover is Recover.EASY


@pytest.mark.parametrize("key", sorted({"genre", "axis_b", "axis_c", "recover"}))
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


def test_the_parser_no_longer_knows_about_faces_or_brands():
    """Five tests lived here, about how cautiously to read a face flag.

    The prompt stopped asking. A vision model guessing at whether a release
    would be needed is answering a legal question it cannot answer, and the
    answer decided which pile a photograph landed in.
    """
    a = parse_assessment(VALID, "P1.RW2")
    for gone in ("faces", "brand_mark", "signage_text"):
        assert not hasattr(a, gone)


def test_a_reply_cached_before_the_change_still_parses():
    """The expensive part of a run is the API call; none of them are re-made."""
    old = {**VALID, "faces": True, "brand_mark": True, "logos": True, "signage_text": True}
    assert parse_assessment(old, "P1.RW2").genre is Genre.STREET


def test_a_reply_from_before_axis_a_came_back_gets_the_neutral_default():
    payload = {k: v for k, v in VALID.items() if k != "axis_a"}
    assert parse_assessment(payload, "P1.RW2").axis_a == 50
