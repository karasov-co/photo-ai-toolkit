"""The report has to survive being moved, and never show an empty tile.

Both come from the same reported failure: every card rendered as a black box.
The cause was not an absolute path, which is what it looked like. The page was
written inside `.internal/staging/<run>/`, the image paths were computed
relative to *that*, and the transactional publish then moved the file two
levels up -- so `../../previews/x.jpg` pointed outside the run.

A better relative path would have fixed that one bug. Owning the images fixes
the class of them, and makes the folder something you can send to somebody.
"""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest
from synthetic import photo_like, write_jpeg

import report_assets
import simple_report
from reports import AssetRecord


def record(name="a.jpg", source=None, **overrides):
    payload = {
        "asset_id": name, "source_path": str(source or f"/gone/{name}"),
        "filename": name, "media_type": "photo", "checksum": name, "asset_key": name,
        "category": "GOOD_STOCK", "final_score": 72, "status": "ok",
        "scores": {"current_quality": 50, "post_edit_potential": 72},
        "category_reasons": ["a photograph worth keeping"],
    }
    payload.update(overrides)
    return AssetRecord(**payload)


@pytest.fixture
def photos(tmp_path):
    root = tmp_path / "photos"
    for i in range(3):
        write_jpeg(photo_like(900, 600, seed=i + 1), root / f"p{i}.jpg")
    return root


@pytest.fixture
def built(tmp_path, photos):
    records = [
        record(f"p{i}.jpg", source=photos / f"p{i}.jpg", asset_key=f"p{i}.jpg")
        for i in range(3)
    ]
    report_dir = tmp_path / "run" / "report"
    assets = simple_report.write_folder(
        records, report_dir, cache_dir=tmp_path / "run" / ".internal" / "thumbs"
    )
    return report_dir, assets, records


def sources_in(page: str) -> list[str]:
    return re.findall(r'src="([^"]+)"', page)


# --- A1: nothing absolute, ever ----------------------------------------------


def test_the_page_contains_no_absolute_path(built):
    report_dir, _, _ = built
    page = (report_dir / "index.html").read_text()

    for src in sources_in(page):
        assert not src.startswith("/"), src
        assert not src.startswith("file://"), src
        assert "/Users/" not in src, src
    assert "/Users/" not in page


def test_every_src_lives_under_assets(built):
    report_dir, _, _ = built
    for src in sources_in((report_dir / "index.html").read_text()):
        assert src.startswith("assets/"), src


def test_the_derivatives_are_the_sizes_asked_for(built):
    from PIL import Image

    report_dir, _, _ = built
    thumb = next((report_dir / "assets" / "thumbs").glob("*.jpg"))
    full = next((report_dir / "assets" / "full").glob("*.jpg"))
    assert max(Image.open(thumb).size) == report_assets.THUMB_PX
    assert max(Image.open(full).size) <= report_assets.FULL_PX


# --- A6: it still works somewhere else ---------------------------------------


def test_every_asset_resolves_after_the_folder_is_moved(built, tmp_path):
    report_dir, _, _ = built
    moved = tmp_path / "somewhere" / "else" / "report"
    moved.parent.mkdir(parents=True)
    shutil.copytree(report_dir, moved)

    page = (moved / "index.html").read_text()
    referenced = sources_in(page)
    assert referenced
    for src in referenced:
        assert (moved / src).is_file(), f"{src} does not resolve after the move"


def test_it_opens_from_a_different_working_directory(built, tmp_path):
    """Opened by path from elsewhere, every asset still resolves."""
    report_dir, _, _ = built
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()

    script = (
        "import re,sys;from pathlib import Path;"
        "p=Path(sys.argv[1]);"
        "srcs=re.findall(r'src=\"([^\"]+)\"',p.read_text());"
        "missing=[s for s in srcs if not (p.parent/s).is_file()];"
        "print('MISSING' if missing else 'OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, str(report_dir / "index.html")],
        cwd=elsewhere, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "OK"


def test_nothing_is_fetched_from_the_network(built):
    report_dir, _, _ = built
    page = (report_dir / "index.html").read_text()
    for marker in ("http://", "https://", "//cdn", "<script src", "<link rel=\"stylesheet\""):
        assert marker not in page, marker


# --- A2: the single file ------------------------------------------------------


def test_standalone_is_one_file_with_everything_inside(tmp_path, photos):
    records = [record(f"p{i}.jpg", source=photos / f"p{i}.jpg", asset_key=f"p{i}.jpg")
               for i in range(3)]
    path = tmp_path / "run" / "report_standalone.html"
    simple_report.write_standalone(records, path)

    page = path.read_text()
    assert sources_in(page), "no images at all"
    assert all(src.startswith("data:image/jpeg;base64,") for src in sources_in(page))
    assert "http://" not in page and "https://" not in page


def test_standalone_still_opens_after_being_moved(tmp_path, photos):
    records = [record("p0.jpg", source=photos / "p0.jpg", asset_key="p0.jpg")]
    path = tmp_path / "run" / "report_standalone.html"
    simple_report.write_standalone(records, path)

    moved = tmp_path / "far" / "away.html"
    moved.parent.mkdir(parents=True)
    shutil.copyfile(path, moved)
    page = moved.read_text()

    assert "data:image/jpeg;base64," in page
    assert not [s for s in sources_in(page) if not s.startswith("data:")]


def test_the_lightbox_reuses_the_inlined_image(tmp_path, photos):
    """A second 1600px copy per photograph would double an already large file."""
    records = [record("p0.jpg", source=photos / "p0.jpg", asset_key="p0.jpg")]
    path = tmp_path / "standalone.html"
    simple_report.write_standalone(records, path)
    page = path.read_text()

    srcs = re.findall(r'src="(data:[^"]+)"', page)
    fulls = re.findall(r'data-full="(data:[^"]+)"', page)
    assert srcs and fulls
    assert srcs[0] == fulls[0], "the full-size copy was inlined a second time"


def test_embed_width_and_quality_are_honoured(tmp_path, photos):
    small = report_assets.inline(
        [record("p0.jpg", source=photos / "p0.jpg", asset_key="p0.jpg")], width=120, quality=40
    )
    large = report_assets.inline(
        [record("p0.jpg", source=photos / "p0.jpg", asset_key="p0.jpg")], width=480, quality=90
    )
    assert len(small.derivatives["p0.jpg"].thumb) < len(large.derivatives["p0.jpg"].thumb)


# --- A3: orientation, metadata, video ----------------------------------------


def test_exif_orientation_is_applied_not_carried(tmp_path):
    """A phone portrait must not arrive on its side."""
    from PIL import Image

    source = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90
    Image.new("RGB", (400, 200), (90, 120, 160)).save(source, exif=exif)
    loaded = report_assets.load_image(source)
    assert loaded.size == (200, 400), "the orientation tag was ignored"


def test_derivatives_carry_no_exif_or_gps(tmp_path):
    from PIL import Image

    source = tmp_path / "located.jpg"
    exif = Image.Exif()
    exif[0x010F] = "Panasonic"          # Make
    exif[0x0110] = "DC-S5M2"            # Model
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (41.0, 23.0, 0.0)  # latitude
    Image.new("RGB", (800, 600), (10, 20, 30)).save(source, exif=exif)

    assert dict(Image.open(source).getexif()), "the fixture carries no metadata to strip"

    encoded = report_assets.encode(report_assets.load_image(source), 480, 78)
    written = tmp_path / "thumb.jpg"
    written.write_bytes(encoded)
    survived = Image.open(written).getexif()
    assert not dict(survived), "metadata survived into the derivative"
    assert not dict(survived.get_ifd(0x8825)), "GPS survived into the derivative"


def test_video_is_skipped_with_a_reason(tmp_path):
    result = report_assets.build(
        [record("clip.MOV", source=tmp_path / "clip.MOV")], tmp_path / "r", tmp_path / "c"
    )
    assert result.counts == {"with_image": 0, "without_image": 1}
    assert "video" in result.missing[0].reason
    # The card reads the reason from `derivatives`, so it must be there too --
    # otherwise the tile says one thing and the summary line says another.
    assert "video" in result.derivatives["clip.MOV"].reason
    assert not result.derivatives["clip.MOV"].ok


def test_the_cache_is_keyed_by_content_and_shape(tmp_path, photos):
    a = report_assets.cache_key(photos / "p0.jpg", 480, 78)
    assert a == report_assets.cache_key(photos / "p0.jpg", 480, 78)
    assert a != report_assets.cache_key(photos / "p0.jpg", 480, 90)
    assert a != report_assets.cache_key(photos / "p0.jpg", 1600, 78)
    assert a != report_assets.cache_key(photos / "p1.jpg", 480, 78)


def test_a_second_build_reuses_the_cached_derivative(tmp_path, photos):
    records = [record("p0.jpg", source=photos / "p0.jpg", asset_key="p0.jpg")]
    cache = tmp_path / "cache"
    report_assets.build(records, tmp_path / "r1", cache)
    written = {p.name for p in cache.iterdir()}

    report_assets.build(records, tmp_path / "r2", cache)
    assert {p.name for p in cache.iterdir()} == written, "the cache grew on a repeat build"


# --- A4: never a silent black box --------------------------------------------


def test_a_frame_with_no_file_gets_a_visible_placeholder(tmp_path):
    missing = record("gone.jpg", source=tmp_path / "not-here.jpg")
    result = report_assets.build([missing], tmp_path / "r", tmp_path / "c")
    simple_report.write([missing], tmp_path / "r" / "index.html", assets=result)

    page = (tmp_path / "r" / "index.html").read_text()
    assert "no longer where it was analysed" in page
    assert "<img" not in page.split("<details")[0]


def test_the_page_counts_the_cards_it_could_not_show(tmp_path, photos):
    records = [
        record("p0.jpg", source=photos / "p0.jpg", asset_key="p0.jpg"),
        record("gone.jpg", source=tmp_path / "not-here.jpg", asset_key="gone.jpg"),
    ]
    result = report_assets.build(records, tmp_path / "r", tmp_path / "c")
    simple_report.write(records, tmp_path / "r" / "index.html", assets=result)

    page = (tmp_path / "r" / "index.html").read_text()
    assert "1 photograph(s) could not be shown" in page
    assert result.counts == {"with_image": 1, "without_image": 1}


def test_a_report_built_with_no_assets_shows_placeholders_not_blanks(tmp_path):
    simple_report.write([record()], tmp_path / "index.html")
    page = (tmp_path / "index.html").read_text()
    assert "noimg" in page


# --- A5: readable and self-contained -----------------------------------------


def test_images_are_lazy_and_the_grid_is_responsive(built):
    report_dir, _, _ = built
    page = (report_dir / "index.html").read_text()
    assert 'loading="lazy"' in page
    assert "@media (max-width:640px)" in page
    assert "grid-template-columns" in page


def test_the_filter_bar_is_sticky_and_counts_each_bucket(built):
    report_dir, _, _ = built
    page = (report_dir / "index.html").read_text()
    assert "position:sticky" in page
    for bucket, _ in simple_report.SECTIONS:
        assert f'data-bucket="{bucket}"' in page


def test_the_script_is_inline_vanilla_javascript(built):
    report_dir, _, _ = built
    page = (report_dir / "index.html").read_text()
    assert "<script>" in page
    assert "<script src" not in page
    for library in ("jquery", "react", "lodash", "bootstrap"):
        assert library not in page.lower()


def test_the_header_explains_the_scale_and_the_thresholds(built):
    report_dir, _, _ = built
    page = (report_dir / "index.html").read_text()
    assert "after a normal edit" in page
    assert str(simple_report.TOP_THRESHOLD) in page


# --- the three numbers on a card ---------------------------------------------


def rendered(*records):
    """One card's own markup, so CSS in the same page cannot satisfy a match."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "r.html"
        simple_report.write(list(records), path)
        text = path.read_text()
    return "\n".join(re.findall(r'<div class="now">.*?</div>', text)) + "\n" + "\n".join(
        re.findall(r'<span class="score".*?</span>', text, re.S)
    )


def header(*records):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "r.html"
        simple_report.write(list(records), path)
        return re.search(r'<p class="scale">(.*?)</p>', path.read_text()).group(1)


def test_the_score_caption_says_what_the_number_is():
    page = rendered(
        record("a.jpg", final_score=71, scores={"current_quality": 64, "content": 80})
    )
    assert "final score after editing" in page
    assert "technical quality: 64" in page
    assert "content: 80" in page
    assert "technical quality now" not in page


def test_content_is_absent_rather_than_zero_when_nothing_looked():
    """An offline run has no content read. A 0 would print as a verdict."""
    page = rendered(record("a.jpg", scores={"current_quality": 64}))
    assert "technical quality: 64" in page
    assert "content:" not in page


def test_the_header_names_both_bucket_thresholds():
    line = header(record("a.jpg"))
    assert f"Top starts at {simple_report.TOP_THRESHOLD}" in line
    assert f"under {simple_report.WEAK_THRESHOLD} is weak" in line
    assert "stock" in line and "personal" in line
