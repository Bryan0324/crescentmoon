from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environments.image_env import ImageEnvConfig, ImageEnvironment  # noqa: E402
from environments.physics2d_env import ObstacleSpec, Physics2DEnvConfig, Physics2DEnvironment  # noqa: E402
from environments.physics3d_env import (  # noqa: E402
    ObstacleSpec3D,
    Physics3DEnvConfig,
    Physics3DEnvironment,
)
from models.stub_victim import ColorBlobVictim  # noqa: E402

# The whole test suite runs on the deterministic stub victim: no weights, no
# network, and confidence that decreases monotonically with occlusion.


def _make_cutout(path: Path, size: tuple[int, int], colour: tuple[int, int, int]) -> Path:
    """A real, non-rectangular alpha-matte cutout -- like scripts/prepare_assets.py
    produces via segmentation, just synthetic so tests need no network or model."""
    sprite = Image.new("RGBA", size, (*colour, 0))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse([4, 4, size[0] - 4, size[1] - 4], fill=255)
    sprite.putalpha(mask)
    sprite.save(path)
    return path


@pytest.fixture(scope="session")
def synthetic_cutout_sprite(tmp_path_factory) -> Path:
    return _make_cutout(tmp_path_factory.mktemp("assets") / "target_sprite.png", (70, 140), (0, 200, 0))


@pytest.fixture(scope="session")
def synthetic_obstacle_sprite(tmp_path_factory) -> Path:
    """A second, differently-coloured cutout -- Stage 2/3's obstacle variant
    composes this real object into the scene alongside the target."""
    return _make_cutout(
        tmp_path_factory.mktemp("assets") / "obstacle_sprite.png", (50, 55), (90, 95, 105)
    )


@pytest.fixture
def victim() -> ColorBlobVictim:
    return ColorBlobVictim()


@pytest.fixture
def image_env(synthetic_cutout_sprite, victim) -> ImageEnvironment:
    return ImageEnvironment(
        ImageEnvConfig(
            target_sprite_path=synthetic_cutout_sprite,
            render_size=192,
            obs_size=32,
            target_class="target",
            target_center=(96.0, 105.0),
            target_height=95,
            patch_min_frac=0.2,
            patch_max_frac=0.6,
            max_steps=1,
        ),
        victim=victim,
    )


@pytest.fixture
def physics2d_env(victim) -> Physics2DEnvironment:
    return Physics2DEnvironment(
        Physics2DEnvConfig(
            render_size=192,
            obs_size=32,
            target_class="target",
            target_center=(96.0, 105.0),
            target_height=95,
            max_steps=8,
            # Movement is bounded to the target's own footprint (roughly
            # x:[72,120] y:[58,152] at this size) -- spawn well inside it.
            spawn_center=(96.0, 70.0),
            spawn_jitter=5.0,
        ),
        victim=victim,
    )


@pytest.fixture
def physics2d_env_obstacles(victim, synthetic_obstacle_sprite) -> Physics2DEnvironment:
    return Physics2DEnvironment(
        Physics2DEnvConfig(
            render_size=192,
            obs_size=32,
            target_class="target",
            target_center=(96.0, 105.0),
            target_height=95,
            max_steps=8,
            spawn_center=(96.0, 70.0),
            spawn_jitter=5.0,
            # Placed to overlap the lower half of the target's own bounding
            # box, so it actually sits inside the attacker's now target-bounded
            # reachable area instead of off to the side where it could never
            # be reached (and so never block anything).
            obstacles=[
                ObstacleSpec(sprite_path=str(synthetic_obstacle_sprite), center=(95.0, 147.5), height=55.0)
            ],
        ),
        victim=victim,
    )


@pytest.fixture
def physics3d_env(victim) -> Physics3DEnvironment:
    return Physics3DEnvironment(
        Physics3DEnvConfig(
            render_size=192,
            obs_size=32,
            target_class="target",
            max_steps=8,
        ),
        victim=victim,
    )


@pytest.fixture
def physics3d_env_obstacles(victim, synthetic_obstacle_sprite) -> Physics3DEnvironment:
    return Physics3DEnvironment(
        Physics3DEnvConfig(
            render_size=192,
            obs_size=32,
            target_class="target",
            max_steps=8,
            obstacles=[
                ObstacleSpec3D(
                    sprite_path=str(synthetic_obstacle_sprite), center=(-0.6, -0.4, 7.0), height=1.0
                )
            ],
        ),
        victim=victim,
    )


@pytest.fixture(params=["image", "physics2d", "physics3d"])
def any_env(request, image_env, physics2d_env, physics3d_env):
    return {
        "image": image_env,
        "physics2d": physics2d_env,
        "physics3d": physics3d_env,
    }[request.param]
