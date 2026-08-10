"""Camera renderer for the 2D world.

World coordinates are the camera's pixel coordinates (x right, y down). The
renderer owns nothing about the scene's content -- not the background, not
"the target", not "the obstacles" -- it just paints whatever
:class:`~environments.world.World` it is given, one object at a time, each at
its own position with its own fixed sprite (optionally rotated). Painting an
object never touches another object's pixels; the only cross-object effect is
occlusion from paint order (see ``World._PAINT_ORDER``). Used by both
``ImageEnvironment`` (Stage 1) and ``Physics2DEnvironment`` (Stage 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # avoid a runtime rendering -> environments -> rendering cycle
    from environments.world import World

__all__ = ["Renderer2DConfig", "Renderer2D", "make_background"]


def make_background(width: int, height: int, seed: int = 0) -> Image.Image:
    """A low-texture outdoor-ish backdrop: sky gradient over flat ground.

    Deliberately boring so that the only thing a detector can latch onto is the
    target sprite -- otherwise reward would be polluted by background objects.
    """
    rng = np.random.default_rng(seed)
    horizon = int(height * 0.62)
    img = np.zeros((height, width, 3), dtype=np.float64)

    sky_top = np.array([150.0, 185.0, 220.0])
    sky_bottom = np.array([215.0, 228.0, 238.0])
    for row in range(horizon):
        t = row / max(1, horizon - 1)
        img[row, :, :] = sky_top * (1 - t) + sky_bottom * t

    ground_top = np.array([135.0, 132.0, 125.0])
    ground_bottom = np.array([95.0, 93.0, 90.0])
    for row in range(horizon, height):
        t = (row - horizon) / max(1, height - horizon - 1)
        img[row, :, :] = ground_top * (1 - t) + ground_bottom * t

    img += rng.normal(0.0, 2.5, size=img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


@dataclass
class Renderer2DConfig:
    width: int = 512
    height: int = 512


class Renderer2D:
    def __init__(self, config: Renderer2DConfig) -> None:
        self.config = config

    def render(self, world: World) -> np.ndarray:
        canvas = Image.new("RGBA", (self.config.width, self.config.height), (0, 0, 0, 255))
        for obj in world.objects:
            if obj.sprite is None:
                continue  # bookkeeping-only object -- nothing of its own to paint
            sprite = obj.sprite
            if obj.rotation_deg:
                sprite = sprite.rotate(obj.rotation_deg, resample=Image.BILINEAR, expand=True)
            topleft = (
                int(round(obj.position[0] - sprite.width / 2)),
                int(round(obj.position[1] - sprite.height / 2)),
            )
            canvas.paste(sprite, topleft, sprite)  # sprite's own alpha only
        return np.asarray(canvas.convert("RGB"), dtype=np.uint8)
