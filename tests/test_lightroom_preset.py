"""The export path, checked as a document rather than as a good intention.

A sidecar was the only thing this tool wrote, and a sidecar is develop settings
with no identity. Dropped into Lightroom's Import Presets it appeared in the
panel as `<x:xmpmet` -- the root XML tag, because Lightroom found no `crs:Name`
-- with an Amount slider that did nothing from 0 to 100, because there was
nothing to apply. Beside a JPEG it was ignored outright.

These tests pin the four fields that make a document a preset, the namespace
mistake that would make Lightroom drop the file entirely, and the white balance
that used to be written into a namespace only this project reads.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from photoai.edit_schema import EditRecipe
from photoai.exporters import adobe_xmp


def recipe(**overrides):
    r = EditRecipe(asset_id="abc123", source_checksum="deadbeef", variant="faithful")
    for key, value in overrides.items():
        setattr(r.global_adjustments, key, value)
    return r


# --- it has to be a preset ----------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["crs:PresetType", "crs:UUID", "crs:SupportsAmount", "crs:HasSettings"],
)
def test_the_preset_carries_the_fields_that_make_it_one(field):
    assert field in adobe_xmp.to_lightroom_preset(recipe(), stem="P1019606")


def test_the_preset_has_a_name_lightroom_can_read():
    """Without this it is listed as `<x:xmpmet`, which is the root tag."""
    document = adobe_xmp.to_lightroom_preset(recipe(), stem="P1019606")
    assert "<crs:Name>" in document
    assert "P1019606" in document
    assert "x-default" in document


def test_the_preset_is_well_formed_xml():
    """It emitted `pat:temperatureDeltaK` without declaring the prefix, and a
    malformed preset is one Lightroom drops rather than one it applies badly."""
    document = adobe_xmp.to_lightroom_preset(recipe(temperature_delta_k=400), stem="a")
    ET.fromstring(document)


def test_the_preset_uses_no_namespace_it_does_not_declare():
    document = adobe_xmp.to_lightroom_preset(recipe(temperature_delta_k=400), stem="a")
    assert "pat:" not in document


def test_the_uuid_is_stable_so_a_re_export_replaces_rather_than_stacks():
    assert adobe_xmp.preset_uuid(recipe()) == adobe_xmp.preset_uuid(recipe())
    other = EditRecipe(asset_id="abc123", source_checksum="deadbeef", variant="expressive")
    assert adobe_xmp.preset_uuid(other) != adobe_xmp.preset_uuid(recipe())


# --- white balance has to survive the trip ------------------------------------


def test_a_rendered_file_gets_a_relative_temperature_adobe_reads():
    """Camera Raw's Temperature is -100..+100 for a JPEG, which is what a
    measured delta already is. This is the case that used to vanish entirely."""
    settings, lost = adobe_xmp.temperature_settings(
        recipe(temperature_delta_k=-3000), is_raw=False
    )
    assert settings["crs:Temperature"] == "-36"
    assert not lost


def test_a_raw_with_a_known_as_shot_temperature_gets_an_absolute_one():
    settings, lost = adobe_xmp.temperature_settings(
        recipe(temperature_delta_k=500), is_raw=True, as_shot_temperature_k=5200
    )
    assert settings["crs:Temperature"] == "5700"
    assert not lost


def test_a_raw_without_one_says_so_instead_of_writing_a_delta_as_kelvin():
    """Writing 500 into an absolute field turns a 5200K frame into a 500K one."""
    settings, lost = adobe_xmp.temperature_settings(
        recipe(temperature_delta_k=500), is_raw=True, as_shot_temperature_k=None
    )
    assert "crs:Temperature" not in settings
    assert "+500K" in lost and "by hand" in lost


def test_the_lost_white_balance_reaches_the_sidecar_warnings():
    document = adobe_xmp.to_adobe_xmp(recipe(temperature_delta_k=500), is_raw=True)
    assert "not written" in document


def test_the_measured_delta_is_still_recorded_in_the_sidecar():
    """The correction Adobe cannot take is still a measurement somebody may want."""
    document = adobe_xmp.to_adobe_xmp(recipe(temperature_delta_k=500), is_raw=True)
    assert 'pat:temperatureDeltaK="500"' in document


def test_an_extreme_measurement_cannot_write_a_slider_off_its_scale():
    settings, _ = adobe_xmp.temperature_settings(
        recipe(temperature_delta_k=-40000), is_raw=False
    )
    assert settings["crs:Temperature"] == "-100"


# --- and it has to be written -------------------------------------------------


def test_export_writes_a_preset_beside_the_sidecar(tmp_path, monkeypatch):
    from photoai import recipe_export

    written = recipe_export.PRESETS_DIRNAME
    assert written == "presets", "presets live apart so a bulk import cannot grab sidecars"


# --- the decision has to reach the catalogue ----------------------------------


def rated(category, score=80):
    return type("R", (), {"category": category, "final_score": score,
                          "filename": "a.jpg", "status": "ok"})()


def test_the_pile_becomes_stars_and_a_colour():
    document = adobe_xmp.to_rating_sidecar(rated("TOP", 91))
    assert 'xmp:Rating="5"' in document
    assert 'xmp:Label="Yellow"' in document
    ET.fromstring(document)


def test_nothing_is_ever_rated_one_star_or_zero():
    """Zero means unrated in every catalogue, and one star is what many
    photographers use for reject -- a decision this tool has not earned."""
    for category in adobe_xmp.STARS:
        assert adobe_xmp.STARS[category] >= 2


def test_every_pile_has_a_star_and_a_colour():
    from photoai.curation import PhotoCategory

    for category in PhotoCategory:
        assert category.name in adobe_xmp.STARS, category.name
        assert category.name in adobe_xmp.LABELS, category.name


def test_an_unknown_category_does_not_invent_a_rating():
    document = adobe_xmp.to_rating_sidecar(rated("SOMETHING_NEW"))
    assert 'xmp:Rating="0"' in document


def test_ratings_are_written_for_the_whole_shoot(tmp_path):
    """A two-star frame is a decision too, and the pile is only useful as a sort
    if every photograph carries one."""
    from photoai import recipe_export

    records = [rated("TOP"), rated("WEAK"), rated("GOOD_PERSONAL")]
    for i, record in enumerate(records):
        record.filename = f"p{i}.jpg"
    written = recipe_export.write_ratings(records, tmp_path / "ratings")
    assert written == 3
    assert len(list((tmp_path / "ratings").glob("*.xmp"))) == 3


def test_a_failed_photograph_gets_no_rating(tmp_path):
    from photoai import recipe_export

    broken = rated("WEAK")
    broken.status = "error"
    assert recipe_export.write_ratings([broken], tmp_path / "r") == 0


# --- a rating must never eat the photographer's own sidecar -------------------


def test_a_merge_keeps_everything_except_the_two_attributes(tmp_path):
    """`<stem>.xmp` is where their develop settings and keywords already live.
    plan_apply exists for exactly this collision on the develop side."""
    theirs = (
        '<x:xmpmeta><rdf:RDF><rdf:Description rdf:about=""'
        ' xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"'
        ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
        ' crs:Exposure2012="+1.50" crs:HasCrop="True"'
        ' xmp:Rating="5" xmp:Label="Green"/></rdf:RDF></x:xmpmeta>'
    )
    merged = adobe_xmp.merge_rating(theirs, rated("WEAK"))
    assert 'crs:Exposure2012="+1.50"' in merged
    assert 'crs:HasCrop="True"' in merged
    assert 'xmp:Rating="2"' in merged


def test_a_sidecar_without_a_rating_gains_one_and_its_namespace():
    theirs = (
        '<x:xmpmeta><rdf:RDF><rdf:Description rdf:about=""'
        ' xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"'
        ' crs:Exposure2012="+1.50"/></rdf:RDF></x:xmpmeta>'
    )
    merged = adobe_xmp.merge_rating(theirs, rated("TOP"))
    assert "xmlns:xmp=" in merged
    assert 'xmp:Rating="5"' in merged
    ET.fromstring(merged.replace("x:xmpmeta", "xmpmeta").replace("rdf:", "").replace("x:", ""))


def test_planning_a_rating_never_writes(tmp_path):
    target = tmp_path / "P1.xmp"
    target.write_text('<x><rdf:Description crs:Exposure2012="+1.00"/></x>')
    before = target.read_text(encoding="utf-8")
    plan = adobe_xmp.plan_rating(rated("TOP"), target)
    assert plan.exists
    assert target.read_text(encoding="utf-8") == before
    assert plan.diff


def test_writing_a_rating_keeps_a_copy_of_what_was_there(tmp_path):
    target = tmp_path / "P1.xmp"
    target.write_text('<x><rdf:Description crs:Exposure2012="+1.00"/></x>')
    adobe_xmp.plan_rating(rated("TOP"), target).write()
    spare = target.with_suffix(".xmp.before-photoai")
    assert spare.is_file()
    assert "crs:Exposure2012" in spare.read_text(encoding="utf-8")


def test_the_colours_are_not_hardcoded(monkeypatch):
    """Red already means reject, or to-print, or client-selected in plenty of
    working catalogues. Overwriting that meaning is worse than not labelling."""
    monkeypatch.setenv("PHOTO_AI_LABELS", "TOP=Orange,WEAK=")
    assert adobe_xmp.LABELS.get("TOP") == "Orange"
    assert adobe_xmp.LABELS.get("WEAK") == ""
    assert 'xmp:Label="Orange"' in adobe_xmp.to_rating_sidecar(rated("TOP"))


# --- the photographer has already done the labelling --------------------------


def test_stars_in_a_catalogue_become_human_piles(tmp_path):
    from photoai import bench_quality

    (tmp_path / "a.xmp").write_text('<x><rdf:Description xmp:Rating="5"/></x>')
    (tmp_path / "b.xmp").write_text('<x><rdf:Description xmp:Rating="2"/></x>')
    labels = bench_quality.read_catalog_labels(tmp_path)
    assert labels["a"]["pile"] == "top"
    assert labels["b"]["pile"] == "good"


def test_an_unrated_photograph_is_not_read_as_a_rejection(tmp_path):
    """Zero means nobody looked, in every catalogue there is."""
    from photoai import bench_quality

    (tmp_path / "a.xmp").write_text('<x><rdf:Description xmp:Rating="0"/></x>')
    (tmp_path / "b.xmp").write_text("<x><rdf:Description/></x>")
    assert bench_quality.read_catalog_labels(tmp_path) == {}
