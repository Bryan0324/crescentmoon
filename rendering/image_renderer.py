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
    "load_sprite",
    "silhouette_min_span",
    "render_patch_from_params",
    "texture_param_count",
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


def texture_param_count(cells: int) -> int:
    """Length of the flat texture action vector for a ``cells x cells`` grid."""
    return cells * cells * 3


def render_patch_from_params(params: np.ndarray, cells: int, size: int) -> Image.Image:
    """Build an opaque RGBA patch from a flat, agent-chosen colour grid.

    ``params`` is ``texture_param_count(cells)`` values in [0, 1] -- one RGB
    triple per coarse cell -- upscaled with nearest-neighbour so cell
    boundaries stay sharp (a printed board, not a blur). This is what lets an
    agent search over the patch's *pattern*, not just its placement, while
    staying entirely within the Environment API: the agent picks numbers, the
    environment renders them and reports back a reward. It never sees the
    victim's gradient or features (Rule 1) -- this is black-box search over a
    small parameter grid, not gradient-based adversarial optimisation.
    """
    grid = np.clip(np.asarray(params, dtype=np.float64), 0.0, 1.0).reshape(cells, cells, 3)
    blocks = (grid * 255.0).astype(np.uint8)
    tex = np.asarray(
        Image.fromarray(blocks, mode="RGB").resize((size, size), Image.NEAREST), dtype=np.uint8
    )
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


def silhouette_min_span(sprite: Image.Image, edge_trim: float = 0.03) -> float:
    """The narrowest horizontal extent of ``sprite``'s own alpha silhouette,
    in pixels.

    A person (or any non-rectangular cutout) is not the same width at every
    height -- narrower at the waist than at outstretched-elbow shoulders, for
    instance. Bounding a patch by the sprite's overall *bounding-box* width
    lets it be wider than the object actually is wherever the silhouette
    happens to be narrower, so it visibly pokes out past the object's own
    edges there. Using the narrowest row instead is the only bound that holds
    at every height, not just the widest one -- the same reasoning as the
    rotation safety margin in ``ImageEnvironment`` (worst case, not average
    case). The top/bottom ``edge_trim`` fraction of rows is excluded: a
    single antialiased pixel at a hairline or a shoe tip would otherwise make
    the bound collapse to almost nothing.
    """
    alpha = np.asarray(sprite.getchannel("A")) > 127
    height = alpha.shape[0]
    lo, hi = int(height * edge_trim), int(height * (1.0 - edge_trim))
    spans = []
    for row in alpha[lo:hi]:
        cols = np.nonzero(row)[0]
        if len(cols):
            spans.append(int(cols[-1] - cols[0] + 1))
    return float(min(spans)) if spans else float(sprite.width)
