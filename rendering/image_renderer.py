"""Image compositing primitives shared by all three stages.

The renderer is owned by the environment.  An agent asks for a *placement*; the
environment decides what pixels that produces.  (prompt.md Requirement 3.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = [
    "load_rgb",
    "save_rgb",
    "to_pil",
    "resize_rgb",
    "make_patch_texture",
    "paste_patch",
    "load_sprite",
]


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(path)


def to_pil(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))


def resize_rgb(image_rgb: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    with to_pil(image_rgb) as im:
        return np.asarray(im.resize(size, Image.BILINEAR), dtype=np.uint8)


def make_patch_texture(size: int = 128, seed: int = 0, cells: int = 8) -> Image.Image:
    """A procedurally generated high-contrast RGBA patch.

    Stands in for a printed adversarial board.  It is fixed for the whole
    project (the agent optimises *placement*, not pixels), so a single seed
    keeps every experiment comparable.
    """
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(cells, cells, 3), dtype=np.uint8)
    tex = np.asarray(
        Image.fromarray(blocks).resize((size, size), Image.NEAREST), dtype=np.uint8
    )
    # A little high-frequency noise on top of the blocks.
    noise = rng.integers(-30, 31, size=tex.shape)
    tex = np.clip(tex.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    rgba = np.dstack([tex, np.full((size, size, 1), 255, dtype=np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


def load_sprite(
    path: str | Path | None,
    fallback_color: tuple[int, int, int] = (0, 200, 0),
    fallback_size: tuple[int, int] = (120, 260),
) -> Image.Image:
    """An RGBA cutout of one real object, or a flat placeholder if unavailable.

    Used for the target and for obstacles: each becomes exactly one
    :class:`~environments.world.WorldObject`, with this as its fixed sprite.
    """
    if path and Path(path).exists():
        with Image.open(path) as im:
            return im.convert("RGBA")
    return Image.new("RGBA", fallback_size, (*fallback_color, 255))


def paste_patch(
    base_rgb: np.ndarray,
    patch_rgba: Image.Image,
    center_x: float,
    center_y: float,
    size: float,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Composite ``patch_rgba`` onto ``base_rgb`` at a pixel-space placement.

    Returns a new array; the base image is never mutated (the environment keeps
    a pristine copy of the clean scene for every reset).
    """
    side = max(1, int(round(size)))
    patch = patch_rgba.resize((side, side), Image.BILINEAR)
    if rotation_deg:
        patch = patch.rotate(float(rotation_deg), resample=Image.BILINEAR, expand=True)

    canvas = to_pil(base_rgb).convert("RGBA")
    top_left = (int(round(center_x - patch.width / 2)), int(round(center_y - patch.height / 2)))
    canvas.paste(patch, top_left, patch)  # PIL clips placements that hang off-canvas
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)
