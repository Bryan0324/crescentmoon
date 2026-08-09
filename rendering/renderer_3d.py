"""Camera renderer for the 3D world.

A pinhole camera at the origin looking down +Z (x right, y up). Like
:mod:`renderer_2d`, this renderer owns only the background; every other object
comes from a :class:`~environments.world.World` and is drawn as a depth-sorted
billboard at its own projected position and its own fixed sprite. Depth
sorting is what makes Stage 3 genuinely three-dimensional: the attacker only
occludes the target if its object is actually *nearer the camera*, and an
object behind the target has no visual effect at all -- exactly what a real,
opaque, physical object would do.

No 3D engine is pulled in -- prompt.md section 24 explicitly rules that out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from environments.world import World
from .renderer_2d import make_background

__all__ = ["Renderer3DConfig", "Renderer3D"]


@dataclass
class Renderer3DConfig:
    width: int = 512
    height: int = 512
    fov_deg: float = 55.0
    near: float = 1.0
    background_seed: int = 0


class Renderer3D:
    def __init__(self, config: Renderer3DConfig) -> None:
        self.config = config
        self.focal = (config.width / 2.0) / math.tan(math.radians(config.fov_deg) / 2.0)
        self._cx = config.width / 2.0
        self._cy = config.height / 2.0
        self._background = make_background(
            config.width, config.height, config.background_seed
        ).convert("RGBA")

    def project(self, point: np.ndarray) -> tuple[float, float]:
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        z = max(z, self.config.near)
        return self._cx + self.focal * x / z, self._cy - self.focal * y / z

    def _pixel_size(self, world_size: float, z: float) -> float:
        return self.focal * world_size / max(z, self.config.near)

    def render(self, world: World) -> np.ndarray:
        layers: list[tuple[float, Image.Image, tuple[int, int]]] = []
        for obj in world.objects:
            z = float(obj.position[2])
            w_px = max(1, int(self._pixel_size(2 * obj.half_extents[0], z)))
            h_px = max(1, int(self._pixel_size(2 * obj.half_extents[1], z)))
            sprite = obj.sprite.resize((w_px, h_px), Image.BILINEAR)
            u, v = self.project(obj.position)
            topleft = (int(round(u - sprite.width / 2)), int(round(v - sprite.height / 2)))
            layers.append((z, sprite, topleft))

        frame = self._background.copy()
        for _, sprite, topleft in sorted(layers, key=lambda item: -item[0]):
            frame.paste(sprite, topleft, sprite)  # sprite's own alpha only
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    def screen_bbox(self, obj) -> tuple[float, float, float, float]:
        """Where ``obj`` projects to on screen -- for privileged bbox bookkeeping only."""
        z = float(obj.position[2])
        w_px = self._pixel_size(2 * obj.half_extents[0], z)
        h_px = self._pixel_size(2 * obj.half_extents[1], z)
        u, v = self.project(obj.position)
        return (u - w_px / 2, v - h_px / 2, u + w_px / 2, v + h_px / 2)
