"""The renderer contract, and a registry of what this machine can actually do.

A recipe is only worth anything if something can turn it back into pixels the
same way twice. That is what a renderer is for, and why the interface insists on
two properties:

- **Availability is a question, not an assumption.** `is_available()` is checked
  before a renderer is used, because darktable and RawTherapee are optional
  installs and a missing binary must degrade the run rather than crash it.
- **Rendering is deterministic.** The same recipe and the same file produce the
  same pixels. Without that, comparing two candidate edits measures the
  renderer's noise as well as the edit's effect, and the search picks winners at
  random.

The built-in renderer is preferred by default precisely because it has no
external dependency: it reads the sensor data through LibRaw, which is already a
hard dependency, so every machine that can run the analysis can also render the
proposals. The external adapters exist because a photographer who has darktable
wants the preview to match what darktable will do.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """The renderer's binary or library is not installed on this machine."""


class RenderError(RuntimeError):
    pass


class Renderer(ABC):
    """Turn (RAW file, recipe) into pixels, reproducibly."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def render(self, path: Path, recipe, *, max_px: int = 1024) -> Image.Image: ...

    def describe(self) -> str:
        return f"{self.name} {self.version()}" if self.is_available() else f"{self.name} (unavailable)"


_REGISTRY: dict[str, Renderer] = {}


def register(renderer: Renderer) -> Renderer:
    _REGISTRY[renderer.name] = renderer
    return renderer


def available() -> list[Renderer]:
    return [r for r in _REGISTRY.values() if r.is_available()]


def get(name: str | None = None) -> Renderer:
    """Pick a renderer: the named one, or the first that works.

    Raises rather than returning a broken renderer, because a caller that
    silently gets no preview will report an edit it never verified.
    """
    if name:
        renderer = _REGISTRY.get(name)
        if renderer is None:
            raise EngineUnavailable(f"unknown renderer {name!r}; have {sorted(_REGISTRY)}")
        if not renderer.is_available():
            raise EngineUnavailable(f"{name} is not installed on this machine")
        return renderer

    for renderer in _REGISTRY.values():
        if renderer.is_available():
            return renderer
    raise EngineUnavailable("no rendering engine is available")


def registry() -> dict[str, Renderer]:
    return dict(_REGISTRY)
