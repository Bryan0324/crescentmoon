"""Stage 3 -- Physics3DEnvironment.

Structurally identical to Stage 2.  The action grows from (dx, dy) to
(dx, dy, dz), the physics core is called with 3-vectors instead of 2-vectors,
and the renderer projects billboards through a pinhole camera.  Agents, reward
and the Environment API are untouched -- that is the whole point of Stage 3.

Like Stage 2, the scene is a :class:`~environments.world.World` composed of
several real, background-removed objects (target, obstacle(s), attacker),
and the agent's action only ever moves the ``attacker`` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.victim import VictimModel
from rendering.image_renderer import load_sprite, make_patch_texture, resize_rgb
from rendering.renderer_3d import Renderer3D, Renderer3DConfig
from reward.attack_reward import AttackReward, RewardConfig

from .base import BaseEnvironment, Observation, StepResult
from .physics import AABB, Body, integrate
from .spaces import BoxSpace, DictSpace, ImageSpace
from .world import World, WorldObject

__all__ = ["ObstacleSpec3D", "Physics3DEnvConfig", "Physics3DEnvironment"]


@dataclass(frozen=True)
class ObstacleSpec3D:
    """One more real, background-removed object composed into the scene --
    placed exactly like the target (its own world-space center + display
    height), never a placeholder block."""

    sprite_path: str
    center: tuple[float, float, float]
    height: float


@dataclass
class Physics3DEnvConfig:
    render_size: int = 512
    obs_size: int = 64
    target_class: str | None = "person"
    target_center: tuple[float, float, float] = (0.0, 0.0, 11.0)
    target_world_height: float = 3.4
    target_sprite_path: str | None = None

    #: Patch side (world units), as a fraction of the *target's own* narrower
    #: dimension -- a physical board cannot be bigger than the object it
    #: attacks. Must stay <= 1.0 (enforced in __init__): at that bound the
    #: patch can never extend past the target's own edges.
    patch_world_frac: float = 0.8
    patch_seed: int = 0
    background_seed: int = 0
    fov_deg: float = 55.0

    max_steps: int = 50
    accel: float = 0.22        # world units / tick^2 at |action| = 1
    max_speed: float = 0.55
    max_step: float = 0.40
    damping: float = 0.85

    world_lo: tuple[float, float, float] = (-5.0, -3.2, 3.0)
    world_hi: tuple[float, float, float] = (5.0, 3.2, 13.5)
    spawn_center: tuple[float, float, float] = (-3.2, -1.4, 6.0)
    spawn_jitter: float = 0.7

    obstacles: list[ObstacleSpec3D] = field(default_factory=list)
    match_iou: float = 0.2

    #: See Physics2DEnvConfig.terminate_on_success -- fixed-length episodes make
    #: sustained suppression optimal instead of hovering at the threshold.
    terminate_on_success: bool = False


def _build_world(config: Physics3DEnvConfig) -> World:
    if not 0.0 < config.patch_world_frac <= 1.0:
        raise ValueError(
            "patch_world_frac must be in (0, 1]: a physical patch cannot be bigger than the "
            f"object it attacks, got {config.patch_world_frac}"
        )
    world = World()

    target_sprite = load_sprite(config.target_sprite_path)
    aspect = target_sprite.width / target_sprite.height
    half_h = config.target_world_height / 2.0
    world.add(
        WorldObject(
            id="target",
            kind="target",
            position=np.asarray(config.target_center, dtype=float),
            half_extents=np.array([half_h * aspect, half_h, 0.15]),
            sprite=target_sprite,
        )
    )

    for i, spec in enumerate(config.obstacles):
        obstacle_sprite = load_sprite(spec.sprite_path)
        obstacle_aspect = obstacle_sprite.width / obstacle_sprite.height
        obstacle_half_h = spec.height / 2.0
        world.add(
            WorldObject(
                id=f"obstacle_{i}",
                kind="obstacle",
                position=np.asarray(spec.center, dtype=float),
                half_extents=np.array([obstacle_half_h * obstacle_aspect, obstacle_half_h, 0.15]),
                sprite=obstacle_sprite,
            )
        )

    target_min_dim = 2.0 * min(half_h * aspect, half_h)
    patch_world_size = config.patch_world_frac * target_min_dim
    patch = make_patch_texture(size=256, seed=config.patch_seed)
    s = patch_world_size / 2.0
    world.add(
        WorldObject(
            id="attacker",
            kind="attacker",
            position=np.zeros(3),
            half_extents=np.array([s, s, 0.15]),
            sprite=patch,
            movable=True,
        )
    )
    return world


class Physics3DEnvironment(BaseEnvironment):
    def __init__(
        self,
        config: Physics3DEnvConfig,
        victim: VictimModel,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._config = config
        self._victim = victim
        self._reward_fn = AttackReward(reward_config)

        self._renderer = Renderer3D(
            Renderer3DConfig(
                width=config.render_size,
                height=config.render_size,
                fov_deg=config.fov_deg,
                background_seed=config.background_seed,
            )
        )
        self._world = _build_world(config)
        self._obstacle_boxes = [
            AABB.from_center(obj.position, obj.half_extents) for obj in self._world.obstacles()
        ]
        self._bounds = AABB(
            np.asarray(config.world_lo, dtype=float), np.asarray(config.world_hi, dtype=float)
        )

        self._action_space = BoxSpace(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            names=("dx", "dy", "dz"),
        )
        self._observation_space = DictSpace(
            {
                "image": ImageSpace(config.obs_size, config.obs_size),
                "vector": BoxSpace(
                    low=np.array([0.0, 0.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                    names=(
                        "step_progress",
                        "last_action_success",
                        "last_dx",
                        "last_dy",
                        "last_dz",
                    ),
                ),
            }
        )

        self._rng = np.random.default_rng(0)
        self._body: Body | None = None
        self._frame = self._renderer.render(self._world.excluding("attacker"))
        self._steps = 0
        self._last_success = 1.0
        self._last_action = np.zeros(3)
        self._baseline_conf = 0.0
        self._baseline_bbox = (0.0, 0.0, 0.0, 0.0)
        self._baseline_class = config.target_class or ""
        self._telemetry: dict[str, Any] = {}
        self.reset()

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    def action_space(self) -> BoxSpace:
        return self._action_space

    def observation_space(self) -> DictSpace:
        return self._observation_space

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self._config
        attacker = self._world.attacker()

        self._body = Body(
            position=self._sample_spawn(),
            half_extents=attacker.half_extents,
            max_speed=cfg.max_speed,
            max_step=cfg.max_step,
            damping=cfg.damping,
        )
        attacker.position = self._body.position
        self._steps = 0
        self._last_success = 1.0
        self._last_action = np.zeros(3)

        clean = self._renderer.render(self._world.excluding("attacker"))
        detections = self._victim.detect(clean)
        target = detections.best(cfg.target_class) or detections.best()
        if target is None:
            raise RuntimeError(
                f"victim '{self._victim.name}' does not see the target in the clean 3D scene; "
                "check the target sprite, its depth, or the confidence threshold"
            )
        self._baseline_conf = target.confidence
        self._baseline_bbox = target.bbox
        self._baseline_class = target.cls_name

        self._frame = self._renderer.render(self._world)
        self._telemetry = {
            "stage": "physics3d",
            "baseline_confidence": self._baseline_conf,
            "baseline_bbox": list(self._baseline_bbox),
            "baseline_class": self._baseline_class,
            "n_obstacles": len(self._obstacle_boxes),
            "spawn": self._body.position.tolist(),
            "steps": [],
        }
        return self.observe()

    def observe(self) -> Observation:
        vector = np.array(
            [
                self._steps / max(1, self._config.max_steps),
                self._last_success,
                *self._last_action,
            ],
            dtype=np.float32,
        )
        return {
            "image": resize_rgb(self._frame, self._config.obs_size),
            "vector": vector,
        }

    def step(self, action) -> StepResult:
        assert self._body is not None
        cfg = self._config
        requested = np.asarray(action, dtype=np.float64).reshape(-1)
        if requested.shape != self._action_space.shape:
            raise ValueError(
                f"action must have shape {self._action_space.shape}, got {requested.shape}"
            )

        applied = self._action_space.clip(requested).astype(np.float64)
        out_of_range = not self._action_space.contains(requested)

        outcome = integrate(
            self._body,
            acceleration=applied * cfg.accel,
            obstacles=self._obstacle_boxes,
            bounds=self._bounds,
            dt=1.0,
        )
        # Same rule as Stage 2: obstacles are charged for, the world edge is not.
        invalid = bool(out_of_range or outcome.collided)
        blocked = bool(invalid or outcome.hit_boundary)

        self._world.attacker().position = self._body.position

        self._frame = self._renderer.render(self._world)
        detections = self._victim.detect(self._frame)
        match = detections.best_matching(
            self._baseline_bbox, self._baseline_class, cfg.match_iou
        )
        current_conf = match.confidence if match is not None else 0.0

        action_cost = float(outcome.distance / max(1e-6, cfg.max_step))
        success = self._reward_fn.is_success(current_conf)
        breakdown = self._reward_fn(
            baseline_confidence=self._baseline_conf,
            current_confidence=current_conf,
            action_cost=action_cost,
            invalid=invalid,
            success=success,
        )

        self._steps += 1
        self._last_success = 0.0 if blocked else 1.0
        self._last_action = applied
        terminated = bool(success and cfg.terminate_on_success)
        truncated = bool(not terminated and self._steps >= cfg.max_steps)

        self._telemetry["steps"].append(
            {
                "step": self._steps,
                "action": applied.tolist(),
                "position": self._body.position.tolist(),
                "collided": bool(outcome.collided),
                "hit_boundary": bool(outcome.hit_boundary),
                "action_valid": not invalid,
                "action_cost": action_cost,
                "detected": match is not None,
                "attack_success": bool(success),
                **breakdown.to_dict(),
            }
        )

        info = {
            "step": self._steps,
            "action_valid": not invalid,
            "last_action_success": not blocked,
        }
        return StepResult(self.observe(), breakdown.total, terminated, truncated, info)

    # ------------------------------------------------------------------
    # experimenter-facing (sealed off from agents)
    # ------------------------------------------------------------------
    def render_human(self) -> np.ndarray:
        return self._frame.copy()

    def clean_image(self) -> np.ndarray:
        return self._renderer.render(self._world.excluding("attacker"))

    def victim_report(self, image_rgb: np.ndarray):
        return self._victim.detect(image_rgb)

    def pop_telemetry(self) -> dict[str, Any]:
        data = self._telemetry
        self._telemetry = {**data, "steps": []}
        return data

    # ------------------------------------------------------------------
    def _sample_spawn(self) -> np.ndarray:
        cfg = self._config
        half = self._world.attacker().half_extents
        lo = self._bounds.lo + half
        hi = self._bounds.hi - half
        center = np.asarray(cfg.spawn_center, dtype=float)
        for _ in range(64):
            pos = np.clip(
                center + self._rng.uniform(-cfg.spawn_jitter, cfg.spawn_jitter, size=3), lo, hi
            )
            probe = AABB.from_center(pos, half)
            if not any(probe.overlaps(o) for o in self._obstacle_boxes):
                return pos
        return np.clip(center, lo, hi)
