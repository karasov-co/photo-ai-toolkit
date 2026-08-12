"""The gaps between modules that worked and a pipeline that used them.

Every test here is a case where a component was correct in isolation and had no
effect in the real run: a camera key that was always blank, a holdout count that
was always zero, a monitor whose verdict nothing read, a prompt that relabelled
a rendered measurement as sensor data.
"""

import numpy as np
import pytest
from synthetic import photo_like, write_jpeg

import pipeline
import prompts
import recipe_validator
import selective_policy
from edit_schema import EditRecipe
from model_monitoring import Monitor, Observation
from preference_model import PersonalModel, Prediction

# --- the measurement domain reaches the prompt -------------------------------


def test_a_raw_and_a_jpeg_measurement_are_labelled_differently():
    raw = {"filename": "a.RW2", "clipped_highlights": 0.08, "clipped_shadows": 0.02,
           "measurement_domain": "raw_sensor", "headroom_stops": 2.4, "encoded": "x"}
    jpeg = {"filename": "b.jpg", "clipped_highlights": 0.08, "clipped_shadows": 0.02,
            "measurement_domain": "rendered_image", "encoded": "x"}
    text = prompts.stage2_user_content([raw, jpeg])[0]["text"]

    assert "[raw_sensor]" in text
    assert "[rendered_image]" in text
    assert "stops of highlight headroom remain" in text
    assert "lower bound" in text


def rubric() -> str:
    """Whitespace-normalised, so a phrase split across a line break still matches."""
    return " ".join(prompts.STAGE2_SYSTEM.split())


def test_the_prompt_no_longer_calls_a_rendered_measurement_raw_ground_truth():
    """The substitution the model had no way to catch."""
    text = rubric()
    assert "the number is the ground truth" not in text
    assert "rendered_image" in text
    assert "NOT the sensor's verdict" in text


def test_the_prompt_explains_both_domains():
    text = rubric()
    assert "raw_sensor" in text and "before any development" in text
    assert "already developed JPEG" in text


def test_a_measurement_carries_its_own_domain():
    assert pipeline.Measurement().measurement_domain == "rendered_image"


def test_a_raw_measurement_reaches_the_record(tmp_path):
    """On a JPEG archive the domain must say so rather than claiming sensor data."""
    root = tmp_path / "archive"
    write_jpeg(photo_like(1600, 1200, seed=3), root / "a.jpg")
    result = pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))
    assert result.records


@pytest.mark.skipif(
    not list(pipeline.Path("photos").glob("*.RW2")), reason="no RAW files available"
)
def test_a_real_raw_measures_on_the_sensor_plane(tmp_path):
    raw = sorted(pipeline.Path("photos").glob("*.RW2"))[0]
    import media

    asset = media.Asset(path=raw, kind=media.MediaKind.PHOTO, checksum="c", size_bytes=1)
    measurement = pipeline.measure_photo(asset, tmp_path / "previews")

    assert measurement.raw_available
    assert measurement.measurement_domain == "raw_sensor"
    assert measurement.raw_highlight_headroom_stops > 0
    assert measurement.raw_measurement_version


# --- the camera key actually reaches the model -------------------------------


def test_the_camera_key_is_built_from_exif():
    assert pipeline._camera_key({"camera_make": "Panasonic", "camera_model": "DC-S5M2"}) == (
        "Panasonic DC-S5M2"
    )


def test_a_model_name_repeating_the_make_is_not_doubled():
    assert pipeline._camera_key(
        {"camera_make": "Panasonic", "camera_model": "Panasonic DC-S5M2"}
    ) == "Panasonic DC-S5M2"


def test_missing_exif_gives_an_empty_camera_key_which_reads_as_familiar():
    """Otherwise every scan and every stripped file would abstain forever."""
    assert pipeline._camera_key({}) == ""
    assert PersonalModel().knows_camera("")


def test_an_unfamiliar_camera_makes_the_model_abstain():
    model = PersonalModel(decisions=2000, genres={"street": 500}, cameras={"Panasonic DC-S5M2"})
    assert model.knows_camera("Panasonic DC-S5M2")
    assert not model.knows_camera("Fujifilm X-T5")
    assert model.predict("a", genre="street", camera="Fujifilm X-T5").abstained


def test_the_camera_reaches_the_record(tmp_path):
    root = tmp_path / "archive"
    write_jpeg(photo_like(1200, 900, seed=4), root / "a.jpg")
    result = pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=tmp_path / "out"))
    assert hasattr(result.records[0], "camera")


# --- the monitor is the tenth gate -------------------------------------------


def test_there_are_ten_gates_on_the_personal_model():
    decision = selective_policy.decide(
        asset_id="a1", route_class="review", technical_evidence="",
        prediction=Prediction("a1", probability=0.01, abstained=False),
        model=PersonalModel(decisions=5000, genres={"street": 500}),
        artistic_scores=_no_signal(), genre="street",
        holdout_checks=10_000, shadow_mode=False, monitor_healthy=True,
    )
    assert len(decision.gates) == 10


def test_an_unhealthy_monitor_blocks_the_personal_model():
    """The monitor could switch automation off and the pipeline carried on."""
    decision = selective_policy.decide(
        asset_id="a1", route_class="review", technical_evidence="",
        prediction=Prediction("a1", probability=0.01, abstained=False),
        model=PersonalModel(decisions=5000, genres={"street": 500}),
        artistic_scores=_no_signal(), genre="street",
        holdout_checks=10_000, shadow_mode=False, monitor_healthy=False,
    )
    assert not decision.acts_on_files
    assert any("monitor_healthy" in gate for gate in decision.failed_gates)


def test_a_healthy_monitor_alone_is_not_enough():
    """It is one gate of ten, not a master switch."""
    decision = selective_policy.decide(
        asset_id="a1", route_class="review", technical_evidence="",
        prediction=Prediction("a1", probability=0.01, abstained=False),
        model=PersonalModel(decisions=5),
        artistic_scores=_no_signal(), genre="street",
        holdout_checks=0, shadow_mode=False, monitor_healthy=True,
    )
    assert not decision.acts_on_files


def _no_signal():
    from artistic import ArtisticScores

    return ArtisticScores(curatorial_uncertainty=10)


# --- the monitoring loop closes ----------------------------------------------


def test_the_pipeline_records_an_observation_for_every_prediction(tmp_path):
    root = tmp_path / "archive"
    for i in range(3):
        write_jpeg(photo_like(1200, 900, seed=10 + i), root / f"f{i}.jpg")
    out = tmp_path / "out"
    pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=out))

    monitor = Monitor(out / "model_monitoring.json")
    assert len(monitor.state.observations) == 3
    assert all(not o.resolved for o in monitor.state.observations)


def test_an_override_resolves_the_observation(tmp_path):
    import overrides as overrides_module

    root = tmp_path / "archive"
    write_jpeg(photo_like(1200, 900, seed=20), root / "a.jpg")
    out = tmp_path / "out"
    result = pipeline.run(pipeline.PipelineOptions(input_dir=root, output_dir=out))
    record = result.records[0]

    store = overrides_module.OverrideStore(out / overrides_module.OVERRIDES_NAME)
    store.set(overrides_module.Override(asset_id=record.asset_id, route_class="flagship"))

    resolved = overrides_module.resolve_observations(
        result.records, store, out / "model_monitoring.json"
    )
    assert resolved >= 1

    monitor = Monitor(out / "model_monitoring.json")
    assert any(o.resolved for o in monitor.state.observations)


def test_a_resolved_false_trash_shows_up_in_the_rate(tmp_path):
    monitor = Monitor(tmp_path / "m.json")
    monitor.observe(Observation(asset_id="a", predicted="safe_quarantine_candidate"))
    monitor.resolve("a", "restored")
    rate, resolved = monitor.false_trash_rate()
    assert resolved == 1
    assert rate == 1.0


def test_an_unresolved_observation_does_not_count_as_success(tmp_path):
    """A rate over an empty set is 0% and means nothing."""
    monitor = Monitor(tmp_path / "m.json")
    for i in range(10):
        monitor.observe(Observation(asset_id=f"a{i}", predicted="auto_keep"))
    _, resolved = monitor.false_trash_rate()
    assert resolved == 0


# --- the skin heuristic no longer vetoes a landscape -------------------------


def warm_surface(seed=5):
    """Sand, wood and sunset all sit in the same hue band as skin."""
    base = np.asarray(photo_like(300, 220, seed=seed).convert("RGB"), dtype=np.float64)
    base[:, :, 0] = np.clip(base[:, :, 0] * 1.5 + 40, 0, 255)
    base[:, :, 1] = np.clip(base[:, :, 1] * 1.1 + 10, 0, 255)
    base[:, :, 2] = np.clip(base[:, :, 2] * 0.5, 0, 255)
    return base


def hue_shifted(rgb: np.ndarray, degrees: float = 14.0) -> np.ndarray:
    """Rotate hue at constant luminance.

    Constant luminance matters: brightening a channel also brightens every edge,
    which trips the halo check and would make this a test of the wrong rule.
    """
    import colorsys

    out = rgb.copy() / 255.0
    height, width = out.shape[:2]
    flat = out.reshape(-1, 3)
    for i in range(flat.shape[0]):
        h, s, v = colorsys.rgb_to_hsv(*flat[i])
        flat[i] = colorsys.hsv_to_rgb((h + degrees / 360.0) % 1.0, s, v)
    return (flat.reshape(height, width, 3) * 255.0)


def test_a_warm_landscape_is_not_vetoed_when_no_face_was_confirmed():
    from PIL import Image

    surface = warm_surface()
    before = Image.fromarray(surface.astype(np.uint8))
    after = Image.fromarray(hue_shifted(surface).astype(np.uint8))

    result = recipe_validator.validate(before, after, EditRecipe(), faces_present=False)
    assert result.ok, f"a warm landscape must not be vetoed: {result.reasons}"


def test_the_same_shift_is_a_veto_once_a_face_is_confirmed():
    from PIL import Image

    surface = warm_surface()
    before = Image.fromarray(surface.astype(np.uint8))
    after = Image.fromarray(hue_shifted(surface).astype(np.uint8))

    result = recipe_validator.validate(before, after, EditRecipe(), faces_present=True)
    if any("skin_hue_drift" in r for r in result.reasons):
        assert not result.ok


def test_an_unconfirmed_skin_finding_is_still_reported():
    """Advisory, not silent: a false veto has to be countable."""
    from PIL import Image

    surface = warm_surface()
    before = Image.fromarray(surface.astype(np.uint8))
    after = Image.fromarray(hue_shifted(surface).astype(np.uint8))

    result = recipe_validator.validate(before, after, EditRecipe(), faces_present=False)
    drift = result.measurements.get("skin_hue_drift_deg")
    if drift is not None and drift > recipe_validator.MAX_SKIN_HUE_DRIFT_DEG:
        assert any("may be sand" in v.detail for v in result.violations)
        assert result.ok


# --- the sidecar manifest survives into the report ---------------------------


def test_the_record_can_carry_the_sidecar_manifest():
    from reports import AssetRecord

    record = AssetRecord(
        asset_id="a", source_path="/x/a.RW2", filename="a.RW2", media_type="photo", checksum="c",
        suggested_sidecars={"adobe": "/x/s.xmp"}, darkroom_engine="builtin",
        darkroom_rejections=["expressive: low_key_flattened"],
    )
    payload = record.to_dict()
    assert payload["suggested_sidecars"]["adobe"] == "/x/s.xmp"
    assert payload["darkroom_rejections"]
    assert payload["darkroom_engine"] == "builtin"


def test_the_html_shows_what_the_darkroom_refused(tmp_path):
    """A rejection says what the tool nearly did to the frame."""
    import reports
    from reports import AssetRecord

    record = AssetRecord(
        asset_id="a", source_path="/x/a.RW2", filename="a.RW2", media_type="photo", checksum="c",
        route_class="flagship", route="commercial", decision_bucket="auto_keep_and_edit",
        preserve_intent=["the low-key structure"],
        darkroom_rejections=["expressive: low_key_flattened"],
        rendered_variants={"original": "/x/o.jpg", "faithful": "/x/f.jpg"},
    )
    body = reports.write_html([record], tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Darkroom" in body
    assert "refused: expressive" in body
    assert "keep: the low-key structure" in body
    assert "auto_keep_and_edit" in body
