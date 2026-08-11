"""Synthetic frames with known defects, built in memory.

No private photographs enter the repository. Every image the suite needs is
generated here from a seed, which also means each defect is *exactly* the
defect named: `blurred()` differs from `photo_like()` in focus and in nothing
else, so a test that says "a blurred frame must not be promoted" is testing
blur rather than whatever else happened to differ between two real files.

`photo_like` matters more than it looks. Random noise has enormous Laplacian
variance and reads as tack sharp; a flat gradient has none and reads as out of
focus. Neither behaves like a photograph, and tests built on them pass or fail
for reasons unrelated to the code. This builds something with real structure --
edges, texture, tonal range -- so the measurements land in the same range real
frames produce.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def photo_like(width: int = 640, height: int = 480, seed: int = 0) -> Image.Image:
    """A frame with edges, texture and tonal range, like a real photograph.

    The **composition** varies with the seed, not just the grain. An earlier
    version varied only the noise, so two frames with different seeds were
    structurally identical, produced the same perceptual hash, and were
    correctly clustered as near-duplicates by code the test was trying to prove
    kept them apart. Two different photographs have to actually differ.
    """
    rng = np.random.default_rng(seed)

    horizon = 0.30 + 0.35 * rng.random()
    ys = np.mgrid[0:height, 0:width][0]
    sky = 60 + 140 * (1.0 - ys / height) * (0.7 + 0.6 * rng.random())
    base = np.dstack([sky * (0.8 + 0.2 * rng.random()), sky * 0.92, sky]).astype(np.float64)

    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Hard edges give the Laplacian something real to measure. Each box is
    # skipped rather than drawn inverted when the canvas is too small for it,
    # so the same builder works at thumbnail sizes.
    for _ in range(int(rng.integers(4, 9))):
        block = int(width * (0.05 + 0.09 * rng.random()))
        x = int(width * rng.random() * 0.92)
        top = int(height * (horizon + 0.3 * rng.random()))
        shade = int(rng.integers(25, 110))
        if x + block >= width or top >= height or block < 4:
            continue
        draw.rectangle([x, top, x + block, height], fill=(shade, shade + 8, shade + 14))
        if block > 10 and top + 22 < height:
            draw.rectangle([x + 4, top + 8, x + block - 4, top + 22], fill=(200, 195, 170))

    sun_x = 0.1 + 0.8 * rng.random()
    sun_y = 0.05 + 0.2 * rng.random()
    radius = max(4, int(width * (0.04 + 0.05 * rng.random())))
    draw.ellipse(
        [int(width * sun_x), int(height * sun_y), int(width * sun_x) + radius, int(height * sun_y) + radius],
        fill=(245, 232, 200),
    )

    stride = int(rng.integers(7, 14))
    foreground = int(height * (0.80 + 0.12 * rng.random()))
    for i in range(0, width, stride):
        draw.line([(i, foreground), (i + 5, min(height - 1, foreground + 5))], fill=(70, 66, 58), width=2)

    grain = rng.normal(0, 3.0, (height, width, 3))
    return Image.fromarray(
        np.clip(np.asarray(img, dtype=np.float64) + grain, 0, 255).astype(np.uint8)
    )


def dark_but_recoverable(seed: int = 0, size: tuple[int, int] = (640, 480)) -> Image.Image:
    """Two stops under, no clipping at either end: the archetypal fixable frame."""
    array = np.asarray(photo_like(*size, seed=seed), dtype=np.float64)
    linear = (array / 255.0) ** 2.2
    darkened = (np.clip(linear * 0.22, 0, 1) ** (1 / 2.2)) * 255.0
    # Lifted off pure black so nothing is crushed -- the information is all there.
    return Image.fromarray(np.clip(darkened + 6, 0, 255).astype(np.uint8))


def blurred(seed: int = 0, radius: float = 9.0, size: tuple[int, int] = (640, 480)) -> Image.Image:
    """Correctly exposed and completely out of focus."""
    return photo_like(*size, seed=seed).filter(ImageFilter.GaussianBlur(radius))


def flat(seed: int = 0) -> Image.Image:
    """Correct brightness, almost no tonal range."""
    array = np.asarray(photo_like(seed=seed), dtype=np.float64)
    return Image.fromarray(np.clip(118 + (array - array.mean()) * 0.18, 0, 255).astype(np.uint8))


def blown(seed: int = 0) -> Image.Image:
    """Most of the frame is pure white with nothing behind it."""
    array = np.asarray(photo_like(seed=seed), dtype=np.float64)
    return Image.fromarray(np.clip(array * 2.6 + 90, 0, 255).astype(np.uint8))


def near_black(width: int = 320, height: int = 240) -> Image.Image:
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def shifted(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """Translate by cropping and re-pasting, so no black edge is introduced.

    An affine transform fills the vacated edge with black, and that black band
    is itself a strong feature -- phase correlation locks onto it and reports
    the wrong shift.
    """
    array = np.asarray(image)
    return Image.fromarray(np.roll(np.roll(array, dy, axis=0), dx, axis=1))


def write_jpeg(image: Image.Image, path, quality: int = 92):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "JPEG", quality=quality)
    return path


def gray_frames(count: int = 6, *, jitter: float = 0.0, pan: float = 0.0, seed: int = 0):
    """A run of consecutive greyscale frames with known camera motion.

    `pan` moves the frame consistently in one direction; `jitter` moves it
    randomly. That is precisely the distinction `analyse_motion` has to make,
    so the fixture makes it controllable rather than incidental.
    """
    rng = np.random.default_rng(seed)
    base = photo_like(256, 192, seed=seed).convert("L")
    frames = []
    offset = 0.0
    for i in range(count):
        offset += pan
        dx = int(round(offset + rng.normal(0, jitter)))
        dy = int(round(rng.normal(0, jitter)))
        frames.append(np.asarray(shifted(base, dx, dy), dtype=np.float64))
        del i
    return frames
