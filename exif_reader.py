import logging
from datetime import datetime
from pathlib import Path

import exifread
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)

EXIF_EMPTY = {
    "camera_make": None,
    "camera_model": None,
    "lens": None,
    "iso": None,
    "shutter_speed": None,
    "aperture": None,
    "focal_length": None,
    "date_shot": None,
    "gps_lat": None,
    "gps_lon": None,
}


def extract_exif(path: Path, file_type: str) -> dict:
    try:
        if file_type == "RAW":
            return _extract_raw(path)
        return _extract_pillow(path)
    except Exception as e:
        logger.warning("Unexpected EXIF error for %s: %s", path.name, e)
        return dict(EXIF_EMPTY)


def _extract_pillow(path: Path) -> dict:
    result = dict(EXIF_EMPTY)
    try:
        with Image.open(path) as img:
            raw_exif = img._getexif() if hasattr(img, "_getexif") else None
            if raw_exif is None:
                try:
                    exif_data = img.getexif()
                    raw_exif = dict(exif_data) if exif_data else None
                except Exception:
                    pass
            if not raw_exif:
                return result

            named = {TAGS.get(tag, tag): value for tag, value in raw_exif.items()}

            result["camera_make"] = _str_or_none(named.get("Make"))
            result["camera_model"] = _str_or_none(named.get("Model"))
            result["lens"] = _str_or_none(
                named.get("LensModel") or named.get("LensSpecification")
            )
            result["iso"] = _int_or_none(named.get("ISOSpeedRatings"))
            result["aperture"] = _parse_rational(named.get("FNumber"))
            result["focal_length"] = _parse_rational(named.get("FocalLength"))
            result["shutter_speed"] = _parse_shutter(named.get("ExposureTime"))
            result["date_shot"] = _parse_date(
                _str_or_none(named.get("DateTimeOriginal") or named.get("DateTime"))
            )

            gps_ifd = named.get("GPSInfo")
            if gps_ifd and isinstance(gps_ifd, dict):
                named_gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                lat, lon = _parse_gps(named_gps)
                result["gps_lat"] = lat
                result["gps_lon"] = lon
    except Exception as e:
        logger.warning("EXIF extraction failed for %s: %s", path.name, e)
    return result


def _extract_raw(path: Path) -> dict:
    """Read EXIF from a RAW file.

    Pillow has no decoder for .RW2/.ARW/.CR3/.NEF, so this reads the EXIF
    directory directly with exifread, then fills any gaps from LibRaw. exifread
    covers the TIFF-based formats and is the only one of the two that reports
    make, model and GPS; LibRaw covers containers exifread cannot parse (CR3).
    """
    result = dict(EXIF_EMPTY)

    try:
        result.update(_read_raw_exifread(path))
    except Exception as e:
        logger.warning("exifread failed for %s: %s", path.name, e)

    if any(result[key] is None for key in _RAWPY_FILLABLE):
        try:
            _fill_from_rawpy(path, result)
        except Exception as e:
            logger.warning("LibRaw metadata unavailable for %s: %s", path.name, e)

    return result


_RAWPY_FILLABLE = ("lens", "iso", "shutter_speed", "aperture", "focal_length", "date_shot")


def _read_raw_exifread(path: Path) -> dict:
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    if not tags:
        return {}

    result = {
        "camera_make": _str_or_none(_tag(tags, "Image Make")),
        "camera_model": _str_or_none(_tag(tags, "Image Model")),
        "lens": _str_or_none(_tag(tags, "EXIF LensModel", "MakerNote LensType")),
        "iso": _int_or_none(_tag(tags, "EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")),
        "aperture": _parse_rational(_tag(tags, "EXIF FNumber")),
        "focal_length": _parse_rational(_tag(tags, "EXIF FocalLength")),
        "shutter_speed": _parse_shutter(_tag(tags, "EXIF ExposureTime")),
        "date_shot": _parse_date(_str_or_none(_tag(tags, "EXIF DateTimeOriginal", "Image DateTime"))),
    }

    named_gps = {
        key.removeprefix("GPS "): _tag(tags, key) for key in tags if key.startswith("GPS GPS")
    }
    if named_gps:
        result["gps_lat"], result["gps_lon"] = _parse_gps(named_gps)

    return result


def _tag(tags: dict, *names: str):
    """First present tag among `names`, unwrapped to a scalar or tuple."""
    for name in names:
        tag = tags.get(name)
        if tag is None:
            continue
        values = getattr(tag, "values", tag)
        if isinstance(values, (list, tuple)):
            if not values:
                continue
            return tuple(values) if len(values) > 1 else values[0]
        return values
    return None


def _fill_from_rawpy(path: Path, result: dict) -> None:
    """Fill still-missing fields from LibRaw. Imported lazily — this is a fallback."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        other = raw.other
        candidates = {
            "iso": _int_or_none(other.iso_speed),
            "shutter_speed": _parse_shutter(other.shutter_speed),
            "aperture": _parse_rational(other.aperture),
            "focal_length": _parse_rational(other.focal_length),
            "date_shot": other.timestamp.isoformat() if other.timestamp else None,
            "lens": _str_or_none(getattr(raw.lens, "model", None)),
        }

    for key, value in candidates.items():
        if result.get(key) is None and value is not None:
            result[key] = value


def _parse_rational(value) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            if value.denominator == 0:
                return None
            return float(value.numerator) / float(value.denominator)
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            if den == 0:
                return None
            return float(num) / float(den)
        return float(value)
    except Exception:
        return None


def _parse_shutter(value) -> str | None:
    seconds = _parse_rational(value)
    if seconds is None:
        return None
    if seconds <= 0:
        return None
    if seconds < 1:
        denom = round(1.0 / seconds)
        return f"1/{denom}"
    return f"{seconds:.1f}s"


def _parse_gps(named_gps: dict) -> tuple[float | None, float | None]:
    try:
        lat_dms = named_gps.get("GPSLatitude")
        lat_ref = named_gps.get("GPSLatitudeRef")
        lon_dms = named_gps.get("GPSLongitude")
        lon_ref = named_gps.get("GPSLongitudeRef")

        if not all([lat_dms, lat_ref, lon_dms, lon_ref]):
            return None, None

        lat = _dms_to_decimal(lat_dms)
        lon = _dms_to_decimal(lon_dms)

        if lat is None or lon is None:
            return None, None

        if str(lat_ref).upper() == "S":
            lat = -lat
        if str(lon_ref).upper() == "W":
            lon = -lon

        return round(lat, 6), round(lon, 6)
    except Exception as e:
        logger.warning("GPS parse failed: %s", e)
        return None, None


def _dms_to_decimal(dms) -> float | None:
    try:
        deg = _parse_rational(dms[0])
        minutes = _parse_rational(dms[1])
        seconds = _parse_rational(dms[2])
        if any(v is None for v in [deg, minutes, seconds]):
            return None
        return deg + minutes / 60.0 + seconds / 3600.0
    except Exception:
        return None


def _parse_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    formats = ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)
    except Exception:
        return None
