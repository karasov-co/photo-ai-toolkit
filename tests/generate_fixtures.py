"""Regenerate the binary fixtures in tests/fixtures/.

The fixtures are committed, so this only needs to run when a fixture changes.
Everything here is synthetic — no real photographs are stored in the repo.

    python tests/generate_fixtures.py
"""

import sys
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPS, IFD, Base
from PIL.TiffImagePlugin import IFDRational

FIXTURES = Path(__file__).parent / "fixtures"

# Panasonic RW2 magic. LibRaw recognises the signature, then fails on the
# truncated body -- which is the RAW failure path we want to exercise.
RW2_MAGIC = b"IIU\x00\x18\x00\x00\x00"

# Enough of a real RW2 to carry the EXIF IFD, and stopping short of the
# embedded JPEG preview that begins at offset 6144.
RAW_HEADER_BYTES = 5120


def _gradient(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = (40 + x * 160 // width, 70 + y * 120 // height, 150)
    return img


def make_with_exif() -> None:
    """A JPEG carrying every field exif_reader knows how to read."""
    exif = Image.Exif()
    exif[Base.Make.value] = "Panasonic"
    exif[Base.Model.value] = "DC-S5M2"
    exif[Base.DateTime.value] = "2026:03:14 09:26:53"

    exif_ifd = exif.get_ifd(IFD.Exif)
    exif_ifd[Base.DateTimeOriginal.value] = "2026:03:14 09:26:53"
    exif_ifd[Base.LensModel.value] = "LUMIX S 35mm F1.8"
    exif_ifd[Base.ISOSpeedRatings.value] = 400
    exif_ifd[Base.FNumber.value] = IFDRational(28, 10)
    exif_ifd[Base.FocalLength.value] = IFDRational(35, 1)
    exif_ifd[Base.ExposureTime.value] = IFDRational(1, 250)

    # 41deg 23' 18.06" N, 2deg 10' 26.34" E
    gps = exif.get_ifd(IFD.GPSInfo)
    gps[GPS.GPSLatitudeRef.value] = "N"
    gps[GPS.GPSLatitude.value] = (IFDRational(41, 1), IFDRational(23, 1), IFDRational(1806, 100))
    gps[GPS.GPSLongitudeRef.value] = "E"
    gps[GPS.GPSLongitude.value] = (IFDRational(2, 1), IFDRational(10, 1), IFDRational(2634, 100))

    _gradient(900, 600).save(FIXTURES / "sample_with_exif.jpg", "JPEG", quality=70, exif=exif)


def make_without_exif() -> None:
    """A valid JPEG with no EXIF block at all. Portrait, to test orientation."""
    _gradient(400, 700).save(FIXTURES / "sample_no_exif.jpg", "JPEG", quality=70)


def make_corrupt_exif() -> None:
    """Valid JPEG framing, but the bytes inside the APP1/EXIF segment are garbage."""
    data = bytearray((FIXTURES / "sample_with_exif.jpg").read_bytes())
    start = data.find(b"Exif\x00\x00")
    if start == -1:  # pragma: no cover - only if Pillow changes its writer
        raise RuntimeError("no EXIF segment found in sample_with_exif.jpg")
    for i in range(start + 6, min(start + 200, len(data))):
        data[i] = 0xFF
    (FIXTURES / "corrupt_exif.jpg").write_bytes(bytes(data))


def make_truncated_raw() -> None:
    """RW2 signature followed by zero padding -- a RAW file LibRaw cannot decode."""
    (FIXTURES / "truncated.rw2").write_bytes(RW2_MAGIC + b"\x00" * 2040)


def make_raw_header(source: Path) -> None:
    """The EXIF header of a real RW2, with nothing else in it.

    This is the one fixture that cannot be synthesised: exifread parses a real
    Panasonic IFD, so the test needs real bytes. Only the first 5 KB is kept --
    the embedded JPEG preview starts at offset 6144, so no image data is
    included -- and the camera serial number is overwritten with zeroes.

    The source RAW is not in the repository (they run ~34 MB). Pass one in to
    regenerate:  python tests/generate_fixtures.py path/to/photo.RW2
    """
    data = bytearray(source.read_bytes()[:RAW_HEADER_BYTES])

    serial_start = data.find(b"WJ4JB")
    if serial_start != -1:
        end = data.find(b"\x00", serial_start)
        data[serial_start:end] = b"0" * (end - serial_start)

    if b"\xff\xd8\xff" in data:  # pragma: no cover - guard, not expected to fire
        raise RuntimeError("refusing to write a fixture containing JPEG image data")

    (FIXTURES / "raw_header.rw2").write_bytes(bytes(data))


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_with_exif()
    make_without_exif()
    make_corrupt_exif()
    make_truncated_raw()

    if len(sys.argv) > 1:
        make_raw_header(Path(sys.argv[1]))
    else:
        print("(skipping raw_header.rw2 -- pass a source .RW2 path to regenerate it)\n")

    for f in sorted(FIXTURES.iterdir()):
        print(f"{f.name:24s} {f.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()
