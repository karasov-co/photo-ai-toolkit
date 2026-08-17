"""Typing, identity and grouping -- the three things everything downstream trusts."""

import pytest
from PIL import Image
from synthetic import photo_like, write_jpeg

from photoai import media

# --- type detection ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("a.jpg", media.MediaKind.PHOTO),
        ("a.JPG", media.MediaKind.PHOTO),
        ("a.rw2", media.MediaKind.PHOTO),
        ("a.CR3", media.MediaKind.PHOTO),
        ("a.heic", media.MediaKind.PHOTO),
        ("a.tiff", media.MediaKind.PHOTO),
        ("a.mov", media.MediaKind.VIDEO),
        ("a.MP4", media.MediaKind.VIDEO),
        ("a.mts", media.MediaKind.VIDEO),
        ("a.txt", media.MediaKind.UNSUPPORTED),
        ("a.xmp", media.MediaKind.UNSUPPORTED),
        ("a", media.MediaKind.UNSUPPORTED),
    ],
)
def test_photos_videos_and_everything_else_are_told_apart(tmp_path, name, kind):
    assert media.classify(tmp_path / name) is kind


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("a.rw2", media.PhotoFormat.RAW),
        ("a.jpg", media.PhotoFormat.JPEG),
        ("a.tif", media.PhotoFormat.TIFF),
        ("a.heic", media.PhotoFormat.HEIC),
        ("a.png", media.PhotoFormat.OTHER),
    ],
)
def test_photo_formats_are_distinguished(tmp_path, name, fmt):
    assert media.photo_format(tmp_path / name) is fmt


def test_raw_and_tiff_carry_recovery_latitude_and_jpeg_does_not(tmp_path):
    """Recoverability scoring leans on this, so it is asserted directly."""
    assert media.has_raw_data(tmp_path / "a.rw2")
    assert media.has_raw_data(tmp_path / "a.tif")
    assert not media.has_raw_data(tmp_path / "a.jpg")


# --- unreadable and hostile files -------------------------------------------


def test_a_file_that_is_not_an_image_raises_rather_than_crashing(tmp_path):
    fake = tmp_path / "not_really.jpg"
    fake.write_bytes(b"this is not a JPEG")
    with pytest.raises(media.UnreadableMedia):
        media.open_photo(fake)


def test_a_truncated_image_raises_unreadable(tmp_path):
    good = write_jpeg(photo_like(), tmp_path / "good.jpg")
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(good.read_bytes()[:200])
    with pytest.raises(media.UnreadableMedia):
        media.open_photo(truncated)


def test_a_decompression_bomb_is_refused_before_the_buffer_is_allocated(tmp_path, monkeypatch):
    """The header is checked first; a 10 GB allocation is not catchable."""
    path = write_jpeg(photo_like(64, 64), tmp_path / "bomb.jpg")
    monkeypatch.setattr(media, "MAX_PIXELS", 100)
    with pytest.raises(media.UnreadableMedia, match="pixel limit"):
        media.open_photo(path)


def test_a_normal_photograph_opens_as_rgb(tmp_path):
    path = write_jpeg(photo_like(120, 90), tmp_path / "ok.jpg")
    image = media.open_photo(path)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (120, 90)


# --- identity ---------------------------------------------------------------


def test_the_same_bytes_give_the_same_checksum(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"identical" * 100)
    b.write_bytes(b"identical" * 100)
    assert media.checksum_file(a) == media.checksum_file(b)


def test_different_bytes_give_different_checksums(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert media.checksum_file(a) != media.checksum_file(b)


def test_a_rename_does_not_change_the_asset_id(tmp_path):
    """This is what makes the cache and the override store survive tidying up."""
    original = tmp_path / "P1042675.RW2"
    original.write_bytes(b"\x00" * 5000)
    before = media.checksum_file(original)
    renamed = tmp_path / "temple_light.RW2"
    original.rename(renamed)
    assert media.checksum_file(renamed) == before


def test_the_fast_checksum_still_separates_files_that_differ_in_the_middle(tmp_path):
    """Head+tail+size could collide; check the guard on size actually holds."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 1001)
    assert media.checksum_file(a) != media.checksum_file(b)


def test_full_and_fast_checksums_agree_on_a_small_file(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(b"small enough to read whole")
    assert media.checksum_file(path) == media.checksum_file(path, full=True)


# --- sidecar grouping -------------------------------------------------------


def test_raw_jpeg_and_xmp_of_one_photograph_are_one_asset(tmp_path):
    for name in ("P1042675.RW2", "P1042675.JPG", "P1042675.xmp"):
        (tmp_path / name).write_bytes(b"data")
    assets = media.discover(tmp_path)
    assert len(assets) == 1
    assert assets[0].path.name == "P1042675.RW2"
    assert [p.name for p in assets[0].siblings] == ["P1042675.JPG"]
    assert [p.name for p in assets[0].sidecars] == ["P1042675.xmp"]


def test_the_raw_is_preferred_over_its_jpeg_twin(tmp_path):
    """The RAW is the file with the latitude; scoring both bills twice."""
    (tmp_path / "shot.JPG").write_bytes(b"data")
    (tmp_path / "shot.RW2").write_bytes(b"data")
    assets = media.discover(tmp_path)
    assert len(assets) == 1
    assert assets[0].is_raw


def test_every_file_of_an_asset_is_listed_for_moving_together(tmp_path):
    for name in ("clip.MOV", "clip.xmp"):
        (tmp_path / name).write_bytes(b"data")
    asset = media.discover(tmp_path)[0]
    assert [p.name for p in asset.all_files] == ["clip.MOV", "clip.xmp"]


def test_an_orphan_sidecar_produces_no_asset(tmp_path):
    (tmp_path / "lonely.xmp").write_bytes(b"data")
    assert media.discover(tmp_path) == []


def test_unsupported_files_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"data")
    (tmp_path / "shot.jpg").write_bytes(b"data")
    assert [a.filename for a in media.discover(tmp_path)] == ["shot.jpg"]


def test_the_same_stem_in_different_folders_stays_two_assets(tmp_path):
    """Cameras restart numbering; two cards can both hold P1000001.RW2."""
    for folder in ("card_a", "card_b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "P1000001.RW2").write_bytes(b"data")
    assert len(media.discover(tmp_path)) == 2


def test_symlinks_are_not_followed_by_default(tmp_path):
    """The tool's own output is a symlink farm; following it walks in circles."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "shot.jpg").write_bytes(b"data")
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    (link_dir / "shot.jpg").symlink_to(real / "shot.jpg")

    assert len(media.discover(tmp_path)) == 1


def test_an_asset_id_is_stable_and_short():
    asset = media.Asset(
        path=media.Path("x.jpg"), kind=media.MediaKind.PHOTO, checksum="a" * 64, size_bytes=1
    )
    assert asset.asset_id == "a" * 16


def test_megapixels_are_rounded_for_display():
    assert media.megapixels(6000, 4000) == 24.0
    assert media.megapixels(1920, 1080) == 2.07
