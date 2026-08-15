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

from edit_schema import EditRecipe
from exporters import adobe_xmp


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
    import recipe_export

    written = recipe_export.PRESETS_DIRNAME
    assert written == "presets", "presets live apart so a bulk import cannot grab sidecars"
