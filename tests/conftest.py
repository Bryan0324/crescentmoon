from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environments.image_env import ImageEnvConfig, ImageEnvironment  # noqa: E402
from environments.physics2d_env import Physics2DEnvConfig, Physics2DEnvironment  # noqa: E402
from environments.physics3d_env import Physics3DEnvConfig, Physics3DEnvironment  # noqa: E402
from models.stub_victim import ColorBlobVictim  # noqa: E402
from rendering.renderer_2d import make_background  # noqa: E402

# The whole test suite runs on the deterministic stub victim: no weights, no
# network, and confidence that decreases monotonically with occlusion.


@pytest.fixture(scope="session")
def synthetic_photo(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("assets") / "photo.jpg"
    frame = make_background(256, 256, seed=3).convert("RGB")
    frame.paste(Image.new("RGB", (70, 140), (0, 200, 0)), (93, 60))
    frame.save(path)
    return path


@pytest.fixture
def victim() -> ColorBlobVictim:
    return ColorBlobVictim()


@pytest.fixture
def image_env(synthetic_photo, victim) -> ImageEnvironment:
    return ImageEnvironment(
        ImageEnvConfig(
            photo_path=synthetic_photo,
            render_size=192,
            obs_size=32,
            target_class="target",
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
            patch_size=60,
            max_steps=8,
            spawn_center=(40.0, 160.0),
            spawn_jitter=5.0,
        ),
        victim=victim,
    )


@pytest.fixture
def physics2d_env_obstacles(victim) -> Physics2DEnvironment:
    return Physics2DEnvironment(
        Physics2DEnvConfig(
            render_size=192,
            obs_size=32,
            target_class="target",
            target_center=(96.0, 105.0),
            target_height=95,
            patch_size=60,
            max_steps=8,
            spawn_center=(40.0, 160.0),
            spawn_jitter=5.0,
            obstacles=[(70.0, 120.0, 120.0, 175.0)],
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


@pytest.fixture(params=["image", "physics2d", "physics3d"])
def any_env(request, image_env, physics2d_env, physics3d_env):
    return {
        "image": image_env,
        "physics2d": physics2d_env,
        "physics3d": physics3d_env,
    }[request.param]
