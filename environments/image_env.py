"""Stage 1 -- ImageEnvironment.

A photo wrapped in the Environment API.  No physics, no RL: the point is to
prove that an agent restricted to ``observe() / step() / action_space()`` can
still find placements that hurt the victim model.

    agent -> action -> validate -> render attacked image -> YOLO -> reward -> agent
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from models.victim import VictimModel
from rendering.image_renderer import load_rgb, make_patch_texture, paste_patch, resize_rgb
from reward.attack_reward import AttackReward, RewardConfig

from .base import BaseEnvironment, Observation, StepResult
from .spaces import BoxSpace, DictSpace, ImageSpace

__all__ = ["ImageEnvConfig", "ImageEnvironment"]


@dataclass
class ImageEnvConfig:
    photo_path: str | Path
    render_size: int = 512          # resolution the victim model sees
    obs_size: int = 64              # resolution the agent sees (a coarse camera)
    target_class: str | None = "person"
    patch_seed: int = 0
    patch_min_frac: float = 0.10    # patch side, as a fraction of render_size
    patch_max_frac: float = 0.32
    max_rotation_deg: float = 90.0
    max_steps: int = 1              # one placement per episode by default
    match_iou: float = 0.2


class ImageEnvironment(BaseEnvironment):
    """Place a physical-looking patch on a photo; the environment does the rest."""

    def __init__(
        self,
        config: ImageEnvConfig,
        victim: VictimModel,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._config = config
        self._victim = victim
        self._reward_fn = AttackReward(reward_config)

        photo = load_rgb(config.photo_path)
        self._clean = resize_rgb(photo, config.render_size)
        self._patch = make_patch_texture(size=256, seed=config.patch_seed)

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
        self._current: np.ndarray = self._clean.copy()
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

        self._current = self._clean.copy()
        self._steps = 0
        self._last_valid = 1.0
        self._last_action_norm = np.zeros(4, dtype=np.float64)

        detections = self._victim.detect(self._clean)
        target = detections.best(self._config.target_class) or detections.best()
        if target is None:
            raise RuntimeError(
                f"victim '{self._victim.name}' found nothing in {self._config.photo_path}; "
                "pick another photo or lower the victim's confidence threshold"
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
            "image": resize_rgb(self._current, cfg.obs_size),
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

        # --- world update + render ---------------------------------------
        self._current = paste_patch(self._clean, self._patch, x, y, size, rotation)

        # --- victim model -------------------------------------------------
        detections = self._victim.detect(self._current)
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
        return self._current.copy()

    def clean_image(self) -> np.ndarray:
        return self._clean.copy()

    def victim_report(self, image_rgb: np.ndarray):
        return self._victim.detect(image_rgb)

    def pop_telemetry(self) -> dict[str, Any]:
        data = self._telemetry
        self._telemetry = {**data, "steps": []}
        return data
