"""darktable and RawTherapee adapters. Optional, and honest when absent.

Both are better raw converters than anything in this repository, and a
photographer who uses one wants the preview to match what their own converter
will do. Neither is installed on most machines, so both are wired as adapters
that report `is_available() is False` and are simply skipped -- never a crash,
and never a silent substitution of a different engine's output under the same
name.

The recipes are written to each program's own sidecar format rather than to a
shared one. Adobe's XMP and darktable's XMP are different schemas that happen to
share a file extension; a single file claiming to satisfy both is read correctly
by neither.

**Neither adapter is verified against a real installation.** They are written
from the documented command-line interfaces and are structurally correct, but
nothing on this machine could execute them. They are marked as unverified in
`version()` so that a report never implies a check that did not happen.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from photoai.renderers.base import EngineUnavailable, Renderer, RenderError, register

logger = logging.getLogger(__name__)

RENDER_TIMEOUT = 300


class _ExternalRenderer(Renderer):
    binary: str = ""
    _version_cache: str | None = None

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def version(self) -> str:
        if not self.is_available():
            return "unavailable"
        if self._version_cache is None:
            try:
                result = subprocess.run(
                    [self.binary, "--version"], capture_output=True, timeout=30, check=False
                )
                first = result.stdout.decode("utf-8", "replace").splitlines()
                self._version_cache = (first[0] if first else self.binary).strip() + " [unverified]"
            except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
                self._version_cache = f"{self.binary} [unverified]"
        return self._version_cache


class DarktableRenderer(_ExternalRenderer):
    """`darktable-cli <in> <sidecar> <out>` with a generated XMP."""

    name = "darktable"
    binary = "darktable-cli"

    def render(self, path: Path, recipe, *, max_px: int = 1024) -> Image.Image:
        if not self.is_available():
            raise EngineUnavailable("darktable-cli is not installed")

        from photoai.exporters.darktable_xmp import to_darktable_xmp

        with tempfile.TemporaryDirectory(prefix="pat_dt_") as scratch:
            sidecar = Path(scratch) / f"{path.stem}.xmp"
            sidecar.write_text(to_darktable_xmp(recipe), encoding="utf-8")
            out_path = Path(scratch) / "render.jpg"
            cmd = [
                self.binary, str(path), str(sidecar), str(out_path),
                "--width", str(max_px), "--height", str(max_px),
                "--hq", "true", "--core", "--disable-opencl",
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, timeout=RENDER_TIMEOUT, check=False
                )
            except subprocess.TimeoutExpired as e:
                raise RenderError(f"darktable-cli timed out on {path.name}") from e
            if not out_path.exists():
                stderr = result.stderr.decode("utf-8", "replace")[:300]
                raise RenderError(f"darktable-cli produced nothing: {stderr}")
            with Image.open(out_path) as img:
                return img.convert("RGB").copy()


class RawTherapeeRenderer(_ExternalRenderer):
    """`rawtherapee-cli -o <out> -p <pp3> -c <in>`."""

    name = "rawtherapee"
    binary = "rawtherapee-cli"

    def render(self, path: Path, recipe, *, max_px: int = 1024) -> Image.Image:
        if not self.is_available():
            raise EngineUnavailable("rawtherapee-cli is not installed")

        from photoai.exporters.rawtherapee_pp3 import to_pp3

        with tempfile.TemporaryDirectory(prefix="pat_rt_") as scratch:
            profile = Path(scratch) / "recipe.pp3"
            profile.write_text(to_pp3(recipe), encoding="utf-8")
            out_path = Path(scratch) / "render.jpg"
            cmd = [
                self.binary, "-o", str(out_path), "-p", str(profile),
                "-j90", "-s", "-c", str(path),
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=RENDER_TIMEOUT, check=False)
            except subprocess.TimeoutExpired as e:
                raise RenderError(f"rawtherapee-cli timed out on {path.name}") from e
            if not out_path.exists():
                raise RenderError("rawtherapee-cli produced nothing")
            with Image.open(out_path) as img:
                rendered = img.convert("RGB")
                rendered.thumbnail((max_px, max_px), Image.LANCZOS)
                return rendered.copy()


register(DarktableRenderer())
register(RawTherapeeRenderer())
