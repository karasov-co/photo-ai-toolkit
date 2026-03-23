import logging
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

logger = logging.getLogger(__name__)

PREVIEW_MAX_PX = 512
PREVIEW_QUALITY = 85


class PreviewGenerationError(Exception):
    pass


def generate_preview(path: Path, file_type: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (path.stem + ".jpg")

    try:
        if file_type == "RAW":
            return _preview_raw(path, out_path)
        return _preview_standard(path, out_path)
    except PreviewGenerationError:
        raise
    except Exception as e:
        raise PreviewGenerationError(f"Failed to generate preview for {path.name}: {e}") from e


def _preview_standard(path: Path, out_path: Path) -> Path:
    try:
        with Image.open(path) as img:
            img = _ensure_rgb(img)
            img.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX), Image.LANCZOS)
            img.save(out_path, "JPEG", quality=PREVIEW_QUALITY)
        return out_path
    except Exception as e:
        raise PreviewGenerationError(f"Pillow preview failed for {path.name}: {e}") from e


def _preview_raw(path: Path, out_path: Path) -> Path:
    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        img = Image.fromarray(rgb)
        img.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=PREVIEW_QUALITY)
        return out_path
    except rawpy.LibRawError as e:
        raise PreviewGenerationError(f"rawpy failed for {path.name}: {e}") from e
    except Exception as e:
        raise PreviewGenerationError(f"RAW preview failed for {path.name}: {e}") from e


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img
