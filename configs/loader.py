"""Config loading and environment construction.

Notebooks and scripts share these builders, so an experiment run from a
notebook is the same experiment the CLI runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from environments.image_env import ImageEnvConfig, ImageEnvironment
from environments.physics2d_env import Physics2DEnvConfig, Physics2DEnvironment
from environments.physics3d_env import Physics3DEnvConfig, Physics3DEnvironment
from models.victim import VictimModel
from reward.attack_reward import RewardConfig

__all__ = [
    "PROJECT_ROOT",
    "load_config",
    "build_victim",
    "build_reward_config",
    "build_stage1_env",
    "build_stage2_env",
    "build_stage3_env",
    "resolve",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def resolve(path: str | Path) -> Path:
    """Make a config-relative path absolute against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ----------------------------------------------------------------------
def build_victim(cfg: dict[str, Any]) -> VictimModel:
    """Instantiate the frozen victim model described by the config."""
    vc = cfg["victim"]
    kind = vc.get("kind", "yolo").lower()
    if kind == "yolo":
        from models.yolo_victim import YOLOVictim

        weights = vc.get("weights", "yolov8n.pt")
        local = resolve(weights)
        return YOLOVictim(
            weights=str(local) if local.exists() else weights,
            conf_threshold=float(vc.get("conf_threshold", 0.05)),
            imgsz=int(vc.get("imgsz", 320)),
            device=vc.get("device", "cpu"),
        )
    if kind == "colorblob":
        from models.stub_victim import ColorBlobVictim

        return ColorBlobVictim()
    raise ValueError(f"unknown victim kind: {kind!r}")


def build_reward_config(cfg: dict[str, Any]) -> RewardConfig:
    return RewardConfig(**cfg["reward"])


def _target_class(cfg: dict[str, Any]) -> str | None:
    if cfg["victim"].get("kind", "yolo").lower() == "colorblob":
        return "target"
    return cfg["victim"].get("target_class")


# ----------------------------------------------------------------------
def build_stage1_env(cfg: dict[str, Any], victim: VictimModel) -> ImageEnvironment:
    s = cfg["stage1"]
    return ImageEnvironment(
        ImageEnvConfig(
            photo_path=resolve(cfg["assets"]["photo"]),
            render_size=int(s["render_size"]),
            obs_size=int(s["obs_size"]),
            target_class=_target_class(cfg),
            patch_min_frac=float(s["patch_min_frac"]),
            patch_max_frac=float(s["patch_max_frac"]),
            max_rotation_deg=float(s["max_rotation_deg"]),
            max_steps=int(s["max_steps"]),
        ),
        victim=victim,
        reward_config=build_reward_config(cfg),
    )


def build_stage2_env(
    cfg: dict[str, Any], victim: VictimModel, *, obstacles: bool = False
) -> Physics2DEnvironment:
    s = cfg["stage2"]
    sprite = resolve(cfg["assets"]["target_sprite"])
    return Physics2DEnvironment(
        Physics2DEnvConfig(
            render_size=int(s["render_size"]),
            obs_size=int(s["obs_size"]),
            target_class=_target_class(cfg),
            target_center=tuple(s["target_center"]),
            target_height=int(s["target_height"]),
            target_sprite_path=str(sprite) if sprite.exists() else None,
            patch_size=int(s["patch_size"]),
            max_steps=int(s["max_steps"]),
            accel=float(s["accel"]),
            max_speed=float(s["max_speed"]),
            max_step=float(s["max_step"]),
            damping=float(s["damping"]),
            spawn_center=tuple(s["spawn_center"]),
            spawn_jitter=float(s["spawn_jitter"]),
            obstacles=[tuple(o) for o in s["obstacles"]] if obstacles else [],
            terminate_on_success=bool(s.get("terminate_on_success", False)),
        ),
        victim=victim,
        reward_config=build_reward_config(cfg),
    )


def build_stage3_env(
    cfg: dict[str, Any], victim: VictimModel, *, obstacles: bool = False
) -> Physics3DEnvironment:
    s = cfg["stage3"]
    sprite = resolve(cfg["assets"]["target_sprite"])
    return Physics3DEnvironment(
        Physics3DEnvConfig(
            render_size=int(s["render_size"]),
            obs_size=int(s["obs_size"]),
            target_class=_target_class(cfg),
            target_center=tuple(s["target_center"]),
            target_world_height=float(s["target_world_height"]),
            target_sprite_path=str(sprite) if sprite.exists() else None,
            patch_world_size=float(s["patch_world_size"]),
            fov_deg=float(s["fov_deg"]),
            max_steps=int(s["max_steps"]),
            accel=float(s["accel"]),
            max_speed=float(s["max_speed"]),
            max_step=float(s["max_step"]),
            damping=float(s["damping"]),
            world_lo=tuple(s["world_lo"]),
            world_hi=tuple(s["world_hi"]),
            spawn_center=tuple(s["spawn_center"]),
            spawn_jitter=float(s["spawn_jitter"]),
            obstacles=[(tuple(c), tuple(h)) for c, h in s["obstacles"]] if obstacles else [],
            terminate_on_success=bool(s.get("terminate_on_success", False)),
        ),
        victim=victim,
        reward_config=build_reward_config(cfg),
    )
