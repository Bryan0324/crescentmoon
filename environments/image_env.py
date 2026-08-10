"""Stage 1 -- ImageEnvironment.

A real, alpha-matted cutout of one object (see ``scripts/prepare_assets.py``)
sitting on a plain backdrop, wrapped in the Environment API. No physics, no
RL: the point is to prove that an agent restricted to
``observe() / step() / action_space()`` can still find placements that hurt
the victim model.

    agent -> action -> validate -> render attacked scene -> YOLO -> reward -> agent

The scene is a :class:`~environments.world.World` of independent objects --
background, target, attacker -- exactly like Stage 2, just without physics
driving the attacker: the agent's action *is* the attacker's placement for
that step, instead of a push integrated by ``environments.physics``. Nothing
in this module ever writes to the target's or the background's position or
sprite; the only thing ``step()`` ever changes is the attacker's own object,
and the only cross-object effect is occlusion by paint order (see
``rendering/renderer_2d.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from models.victim import VictimModel
from rendering.image_renderer import load_sprite, make_patch_texture, resize_rgb
from rendering.renderer_2d import Renderer2D, Renderer2DConfig, make_background
from reward.attack_reward import AttackReward, RewardConfig

from .base import BaseEnvironment, Observation, StepResult
from .spaces import BoxSpace, DictSpace, ImageSpace
from .world import World, WorldObject

__all__ = ["ImageEnvConfig", "ImageEnvironment"]


@dataclass
class ImageEnvConfig:
    target_sprite_path: str | Path | None  # a real, background-removed cutout (RGBA)
    render_size: int = 512          # resolution the victim model sees
    obs_size: int = 64              # resolution the agent sees (a coarse camera)
    target_class: str | None = "person"
    target_center: tuple[float, float] = (256.0, 290.0)
    target_height: int = 250
    background_seed: int = 0

    patch_seed: int = 0
    patch_min_frac: float = 0.10    # patch side, as a fraction of render_size
    patch_max_frac: float = 0.32
    max_rotation_deg: float = 90.0
    max_steps: int = 1              # one placement per episode by default
    match_iou: float = 0.2


def _build_world(config: ImageEnvConfig) -> World:
    world = World()

    background = make_background(config.render_size, config.render_size, config.background_seed)
    world.add(
        WorldObject(
            id="background",
            kind="background",
            position=np.array([config.render_size / 2.0, config.render_size / 2.0]),
            half_extents=np.array([config.render_size / 2.0, config.render_size / 2.0]),
            sprite=background.convert("RGBA"),
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

    world.add(
        WorldObject(
            id="attacker",
            kind="attacker",
            position=np.zeros(2),
            half_extents=np.array([1.0, 1.0]),
            sprite=make_patch_texture(size=8, seed=config.patch_seed),  # replaced every step
            movable=True,
        )
    )
    return world


class ImageEnvironment(BaseEnvironment):
    """Place a physical-looking patch in front of a target object; the
    environment renders the scene and does the rest."""

    def __init__(
        self,
        config: ImageEnvConfig,
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
        self._patch_texture = make_patch_texture(size=256, seed=config.patch_seed)

        side = float(config.render_size)
        self._action_space = BoxSpace(
            low=np.array([0.0, 0.0, config.patch_min_frac * side, -config.max_rotation_deg]),
            high=np.array([side, side, config.patch_max_frac * side, config.max_rotation_deg]),
            names=("x", "y", "size", "rotation"),
        )
        self._observation_space = DictSpace(
            {
                "image": ImageSpace(config.obs_size, config.obs_size),
                "vector": BoxSpace(
                    low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                    names=(
                        "step_progress",
                        "last_action_valid",
                        "last_x_norm",
                        "last_y_norm",
                        "last_size_norm",
                        "last_rotation_norm",
                    ),
                ),
            }
        )

        self._rng = np.random.default_rng(0)
        self._frame = self._renderer.render(self._world.excluding("attacker"))
        self._steps = 0
        self._last_valid = 1.0
        self._last_action_norm = np.zeros(4, dtype=np.float64)
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

        self._frame = self._renderer.render(self._world.excluding("attacker"))
        self._steps = 0
        self._last_valid = 1.0
        self._last_action_norm = np.zeros(4, dtype=np.float64)

        detections = self._victim.detect(self._frame)
        target = detections.best(self._config.target_class) or detections.best()
        if target is None:
            raise RuntimeError(
                f"victim '{self._victim.name}' found nothing in the clean scene; "
                f"check {self._config.target_sprite_path} or lower the confidence threshold"
            )
        self._baseline_conf = target.confidence
        self._baseline_bbox = target.bbox
        self._baseline_class = target.cls_name

        self._telemetry = {
            "stage": "image",
            "baseline_confidence": self._baseline_conf,
            "baseline_bbox": list(self._baseline_bbox),
            "baseline_class": self._baseline_class,
            "steps": [],
        }
        return self.observe()

    def observe(self) -> Observation:
        cfg = self._config
        vector = np.array(
            [
                self._steps / max(1, cfg.max_steps),
                self._last_valid,
                *self._last_action_norm,
            ],
            dtype=np.float32,
        )
        return {
            "image": resize_rgb(self._frame, cfg.obs_size),
            "vector": vector,
        }

    def step(self, action) -> StepResult:
        cfg = self._config
        requested = np.asarray(action, dtype=np.float64).reshape(-1)
        if requested.shape != self._action_space.shape:
            raise ValueError(
                f"action must have shape {self._action_space.shape}, got {requested.shape}"
            )

        # --- the environment validates; the agent cannot bypass this ------
        applied = self._action_space.clip(requested).astype(np.float64)
        invalid = not self._action_space.contains(requested)
        x, y, size, rotation = applied

        # --- world update: only the attacker's own object changes --------
        side = max(1, int(round(size)))
        attacker = self._world.attacker()
        attacker.sprite = self._patch_texture.resize((side, side), Image.BILINEAR)
        attacker.position = np.array([x, y])
        attacker.half_extents = np.array([side / 2.0, side / 2.0])
        attacker.rotation_deg = float(rotation)

        # --- render -> victim model ----------------------------------------
        self._frame = self._renderer.render(self._world)
        detections = self._victim.detect(self._frame)
        match = detections.best_matching(
            self._baseline_bbox, self._baseline_class, cfg.match_iou
        )
        current_conf = match.confidence if match is not None else 0.0

        # --- evaluation -> reward ----------------------------------------
        action_cost = float((size / cfg.render_size) ** 2)  # patch area fraction
        success = self._reward_fn.is_success(current_conf)
        breakdown = self._reward_fn(
            baseline_confidence=self._baseline_conf,
            current_confidence=current_conf,
            action_cost=action_cost,
            invalid=invalid,
            success=success,
        )

        self._steps += 1
        self._last_valid = 0.0 if invalid else 1.0
        lo, hi = self._action_space.low, self._action_space.high
        span = np.where(hi > lo, hi - lo, 1.0)
        self._last_action_norm = (applied - lo) / span
        self._last_action_norm[3] = 2 * self._last_action_norm[3] - 1  # rotation in [-1, 1]

        terminated = bool(success)
        truncated = bool(not terminated and self._steps >= cfg.max_steps)

        self._telemetry["steps"].append(
            {
                "step": self._steps,
                "action": applied.tolist(),
                "action_valid": not invalid,
                "action_cost": action_cost,
                "detected": match is not None,
                "attack_success": bool(success),
                **breakdown.to_dict(),
            }
        )

        info = {"step": self._steps, "action_valid": not invalid, "action_applied": applied}
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
