"""The World: an explicit, object-based scene graph for the physics stages.

A real physical attacker cannot repaint pixels belonging to a different
object -- occlusion (putting a separate, opaque thing in front of the camera)
is the only cross-object effect it has. Baking "the target", "the obstacles"
and "the attacker" into ad hoc renderer/physics parameters hides that fact;
modelling the scene as a list of independent, opaque objects makes it
structural instead of incidental. Physics only ever moves the object flagged
``movable``; the renderer only ever paints each object's own fixed sprite at
its own position. There is no operation anywhere that lets one object's pixels
bleed into another's -- the World simply has no such method.

Used by every 2D stage -- ``ImageEnvironment`` (Stage 1), ``Physics2DEnvironment``
(Stage 2) -- and by ``Physics3DEnvironment`` (Stage 3). The target is a real,
alpha-matted cutout of one photographed object (see
``scripts/prepare_assets.py``), not a bounding-box rectangle: its own sprite
*is* its own silhouette, so "the attacker's object" and "the target's object"
never share a pixel unless the attacker is genuinely drawn on top of it --
exactly the one effect a physical occluder is allowed to have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

__all__ = ["ObjectKind", "WorldObject", "World"]

ObjectKind = Literal["background", "target", "obstacle", "attacker"]

# Paint order when there is no real depth to sort by (the 2D stages): later
# kinds are drawn on top of earlier ones, so the attacker -- the only object
# that can occlude anything -- is always nearest the lens. The 3D stage
# ignores this and depth-sorts every frame instead (see Renderer3D.render).
_PAINT_ORDER: dict[ObjectKind, int] = {"background": 0, "target": 1, "obstacle": 2, "attacker": 3}


@dataclass
class WorldObject:
    """One physical thing in the scene: its own position, footprint and look.

    Nothing outside this object can change ``sprite`` -- an attacker can only
    ever construct actions that move *its own* object.
    """

    id: str
    kind: ObjectKind
    position: np.ndarray       # n-dim: pixel space (2D) or world-space metres (3D)
    half_extents: np.ndarray   # n-dim: collision footprint / billboard half-size
    sprite: Image.Image        # RGBA -- this object's own, fixed appearance
    movable: bool = False
    #: 2D-only: rotation applied to the sprite before it is pasted. 3D objects
    #: leave this at 0 -- depth and projection are handled by Renderer3D instead.
    rotation_deg: float = 0.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.half_extents = np.asarray(self.half_extents, dtype=np.float64)

    @property
    def paint_order(self) -> int:
        return _PAINT_ORDER[self.kind]


class World:
    """A flat set of objects. There is deliberately no "recolour object X" call."""

    def __init__(self, objects: list[WorldObject] | None = None) -> None:
        self._objects: dict[str, WorldObject] = {obj.id: obj for obj in (objects or [])}

    def add(self, obj: WorldObject) -> WorldObject:
        self._objects[obj.id] = obj
        return obj

    def get(self, object_id: str) -> WorldObject:
        return self._objects[object_id]

    def of_kind(self, kind: ObjectKind) -> list[WorldObject]:
        return [obj for obj in self._objects.values() if obj.kind == kind]

    def excluding(self, *kinds: ObjectKind) -> "World":
        """A copy containing every object except the given kinds (for a 'clean
        scene' render that leaves the attacker out)."""
        return World([obj for obj in self._objects.values() if obj.kind not in kinds])

    @property
    def objects(self) -> list[WorldObject]:
        """Every object, in paint order (2D) / arbitrary order (3D depth-sorts itself)."""
        return sorted(self._objects.values(), key=lambda obj: obj.paint_order)

    def background(self) -> WorldObject:
        return self.of_kind("background")[0]

    def target(self) -> WorldObject:
        return self.of_kind("target")[0]

    def obstacles(self) -> list[WorldObject]:
        return self.of_kind("obstacle")

    def attacker(self) -> WorldObject:
        return self.of_kind("attacker")[0]

    def __len__(self) -> int:
        return len(self._objects)
