"""Stage 2 -- Physics2DEnvironment.

Same Environment API, same reward meaning, same agents.  What changes is that
the agent no longer *places* the attacking object: it can only push it, and the
environment decides where it actually ends up.

    action (dx, dy) -> constraint check -> physics -> render -> YOLO -> reward

The scene is a :class:`~environments.world.World` of independent objects
(target, obstacles, attacker).  The agent's action only ever moves the
``attacker`` object's ``Body``; nothing in this module ever writes to the
target's or an obstacle's position or sprite once the world is built --
occlusion by paint order is the only cross-object effect the attacker has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from models.victim import VictimModel
from rendering.image_renderer import load_sprite, make_patch_texture, resize_rgb
from rendering.renderer_2d import Renderer2D, Renderer2DConfig, make_background
from reward.attack_reward import AttackReward, RewardConfig

from .base import BaseEnvironment, Observation, StepResult
from .physics import AABB, Body, integrate
from .spaces import BoxSpace, DictSpace, ImageSpace
from .world import World, WorldObject

__all__ = ["Physics2DEnvConfig", "Physics2DEnvironment"]


@dataclass
class Physics2DEnvConfig:
    render_size: int = 512
    obs_size: int = 64
    target_class: str | None = "person"
    target_center: tuple[float, float] = (256.0, 290.0)
    target_height: int = 250
    target_sprite_path: str | None = None

    #: Patch side, as a fraction of the *target's own* narrower dimension --
    #: a physical board cannot be bigger than the object it attacks. Must stay
    #: <= 1.0 (enforced in __init__): at that bound the patch can never extend
    #: past the target's own edges.
    patch_frac: float = 0.8
    patch_seed: int = 0
    background_seed: int = 0

    max_steps: int = 40
    accel: float = 10.0        # world units / tick^2 at |action| = 1
    max_speed: float = 26.0
    max_step: float = 18.0     # hard cap on displacement per tick
    damping: float = 0.85

    spawn_center: tuple[float, float] = (90.0, 430.0)
    spawn_jitter: float = 45.0
    obstacles: list[tuple[float, float, float, float]] = field(default_factory=list)
    match_iou: float = 0.2

    #: End the episode the moment the attack succeeds?  Off by default: with a
    #: per-step reward, stopping early would pay *less* than hovering just above
    #: the success threshold, and the agent would learn to do exactly that.
    #: Fixed-length episodes make sustained suppression the optimal behaviour.
    terminate_on_success: bool = False


def _build_world(config: Physics2DEnvConfig) -> World:
    if not 0.0 < config.patch_frac <= 1.0:
        raise ValueError(
            "patch_frac must be in (0, 1]: a physical patch cannot be bigger than the "
            f"object it attacks, got {config.patch_frac}"
        )
    world = World()

    size = (config.render_size, config.render_size)
    background = make_background(*size, config.background_seed).convert("RGBA")
    world.add(
        WorldObject(
            id="background",
            kind="background",
            position=np.array([config.render_size / 2.0, config.render_size / 2.0]),
            half_extents=np.array([config.render_size / 2.0, config.render_size / 2.0]),
            sprite=background,
        )
    )

    target_sprite = load_sprite(config.target_sprite_path)
    scale = config.target_height / target_sprite.height
    target_sprite = target_sprite.resize(
        (max(1, int(target_sprite.width * scale)), config.target_height), Image.BILINEAR
    )
    world.add(
        WorldObject(
            id="target",
            kind="target",
            position=np.asarray(config.target_center, dtype=float),
            half_extents=np.array([target_sprite.width / 2.0, target_sprite.height / 2.0]),
            sprite=target_sprite,
        )
    )

    for i, (x1, y1, x2, y2) in enumerate(config.obstacles):
        w, h = max(1, int(x2 - x1)), max(1, int(y2 - y1))
        block = Image.new("RGBA", (w, h), (70, 72, 78, 255))
        world.add(
            WorldObject(
                id=f"obstacle_{i}",
                kind="obstacle",
                position=np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]),
                half_extents=np.array([w / 2.0, h / 2.0]),
                sprite=block,
            )
        )

    target_min_dim = min(target_sprite.width, target_sprite.height)
    patch_side = max(1, int(round(config.patch_frac * target_min_dim)))
    patch = make_patch_texture(size=256, seed=config.patch_seed).resize(
        (patch_side, patch_side), Image.BILINEAR
    )
    world.add(
        WorldObject(
            id="attacker",
            kind="attacker",
            position=np.zeros(2),
            half_extents=np.array([patch_side / 2.0, patch_side / 2.0]),
            sprite=patch,
            movable=True,
        )
    )
    return world


class Physics2DEnvironment(BaseEnvironment):
    def __init__(
        self,
        config: Physics2DEnvConfig,
        victim: VictimModel,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._config = config
        self._victim = victim
        self._reward_fn = AttackReward(reward_config)

        self._renderer = Renderer2D(
            Renderer2DConfig(width=config.render_size, height=config.render_size)
        )
        self._world = _build_world(config)
        self._obstacle_boxes = [
            AABB.from_center(obj.position, obj.half_extents) for obj in self._world.obstacles()
        ]
        self._bounds = AABB(
            np.zeros(2), np.array([config.render_size, config.render_size], dtype=float)
        )

        self._action_space = BoxSpace(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), names=("dx", "dy")
        )
        self._observation_space = DictSpace(
            {
                "image": ImageSpace(config.obs_size, config.obs_size),
                "vector": BoxSpace(
                    low=np.array([0.0, 0.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    names=("step_progress", "last_action_success", "last_dx", "last_dy"),
                ),
            }
        )

        self._rng = np.random.default_rng(0)
        self._body: Body | None = None
        self._frame = self._renderer.render(self._world.excluding("attacker"))
        self._steps = 0
        self._last_success = 1.0
        self._last_action = np.zeros(2)
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
        self._last_action = np.zeros(2)

        # Baseline: what the victim sees before the attacker is in frame.
        clean = self._renderer.render(self._world.excluding("attacker"))
        detections = self._victim.detect(clean)
        target = detections.best(cfg.target_class) or detections.best()
        if target is None:
            raise RuntimeError(
                f"victim '{self._victim.name}' does not see the target in the clean scene; "
                "check assets/target sprite or lower the confidence threshold"
            )
        self._baseline_conf = target.confidence
        self._baseline_bbox = target.bbox
        self._baseline_class = target.cls_name

        self._frame = self._renderer.render(self._world)
        self._telemetry = {
            "stage": "physics2d",
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

        # --- physics: the environment, not the agent, moves the object ----
        outcome = integrate(
            self._body,
            acceleration=applied * cfg.accel,
            obstacles=self._obstacle_boxes,
            bounds=self._bounds,
            dt=1.0,
        )
        # Penalised: an out-of-range request, or a push into an obstacle.
        # *Not* penalised: touching the world edge -- that is the world being the
        # world, and charging 0.25/step for it swamps the attack signal entirely.
        invalid = bool(out_of_range or outcome.collided)
        blocked = bool(invalid or outcome.hit_boundary)

        # The attacker's own object is the only thing this step touches.
        self._world.attacker().position = self._body.position

        # --- render -> victim -> reward -----------------------------------
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
        # The observation still tells the agent that its push failed, even when
        # the reward does not charge for it.
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
                center + self._rng.uniform(-cfg.spawn_jitter, cfg.spawn_jitter, size=2), lo, hi
            )
            probe = AABB.from_center(pos, half)
            if not any(probe.overlaps(o) for o in self._obstacle_boxes):
                return pos
        return np.clip(center, lo, hi)
