import logging
from datetime import datetime
from pathlib import Path

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
                    raw_exif = {k: v for k, v in exif_data.items()} if exif_data else None
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
    try:
        return _extract_pillow(path)
    except Exception as e:
        logger.warning("RAW EXIF extraction failed for %s: %s", path.name, e)
        return dict(EXIF_EMPTY)


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
