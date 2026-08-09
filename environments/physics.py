"""A deliberately small physics core, shared by the 2D and 3D environments.

Everything is dimension-agnostic (n = 2 or n = 3), which is what lets Stage 3
reuse Stage 2 unchanged: only the number of axes grows.  Scope, per prompt.md
section 8: position, velocity, collision, boundary, movement limit.  Nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["AABB", "Body", "StepOutcome", "integrate"]


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box, ``lo``/``hi`` per axis."""

    lo: np.ndarray
    hi: np.ndarray

    @staticmethod
    def from_center(center, half_extents) -> "AABB":
        c = np.asarray(center, dtype=np.float64)
        h = np.asarray(half_extents, dtype=np.float64)
        return AABB(c - h, c + h)

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    @property
    def half_extents(self) -> np.ndarray:
        return (self.hi - self.lo) / 2.0

    def overlaps(self, other: "AABB") -> bool:
        return bool(np.all(self.lo < other.hi) and np.all(self.hi > other.lo))


@dataclass
class Body:
    """A kinematic box that the agent pushes around."""

    position: np.ndarray
    half_extents: np.ndarray
    velocity: np.ndarray = field(default=None)  # type: ignore[assignment]
    max_speed: float = 30.0
    max_step: float = 20.0
    damping: float = 0.85

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.half_extents = np.asarray(self.half_extents, dtype=np.float64)
        if self.velocity is None:
            self.velocity = np.zeros_like(self.position)
        else:
            self.velocity = np.asarray(self.velocity, dtype=np.float64)

    @property
    def box(self) -> AABB:
        return AABB.from_center(self.position, self.half_extents)


@dataclass(frozen=True)
class StepOutcome:
    displacement: np.ndarray
    collided: bool
    hit_boundary: bool

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.displacement))


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > max_norm > 0:
        return vec * (max_norm / norm)
    return vec


def integrate(
    body: Body,
    acceleration: np.ndarray,
    obstacles: list[AABB],
    bounds: AABB,
    dt: float = 1.0,
) -> StepOutcome:
    """Advance ``body`` by one tick, honouring speed / step / collision / boundary limits.

    Movement is resolved one axis at a time: an axis that would end inside an
    obstacle is rolled back and its velocity zeroed, so a body can slide along a
    wall but never through it.  This is the mechanism that makes "just move to
    the best spot" impossible for the agent once obstacles exist.
    """
    accel = np.asarray(acceleration, dtype=np.float64)
    start = body.position.copy()

    body.velocity = _clip_norm(body.velocity * body.damping + accel * dt, body.max_speed)
    delta = _clip_norm(body.velocity * dt, body.max_step)

    collided = False
    hit_boundary = False

    for axis in range(body.position.shape[0]):
        if delta[axis] == 0.0:
            continue
        candidate = body.position.copy()
        candidate[axis] += delta[axis]

        # Boundary: clamp so the body's box stays fully inside the world.
        lo = bounds.lo[axis] + body.half_extents[axis]
        hi = bounds.hi[axis] - body.half_extents[axis]
        clamped = float(np.clip(candidate[axis], lo, hi))
        if clamped != candidate[axis]:
            hit_boundary = True
            candidate[axis] = clamped
            body.velocity[axis] = 0.0

        probe = AABB.from_center(candidate, body.half_extents)
        if any(probe.overlaps(obs) for obs in obstacles):
            collided = True
            body.velocity[axis] = 0.0
            continue  # axis blocked: keep the old coordinate

        body.position = candidate

    return StepOutcome(
        displacement=body.position - start,
        collided=collided,
        hit_boundary=hit_boundary,
    )
