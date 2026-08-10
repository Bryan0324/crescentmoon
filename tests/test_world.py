"""The World is what turns "real attacks are per-object" into a structural
guarantee rather than a convention: the attacker's action can only move its
own WorldObject, and nothing in the Physics 2D/3D environments ever writes to
a target's or an obstacle's position or sprite after construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from environments.world import World, WorldObject


def _object(id_, kind, position, half=(5.0, 5.0), movable=False) -> WorldObject:
    sprite = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    return WorldObject(
        id=id_, kind=kind, position=np.array(position), half_extents=np.array(half),
        sprite=sprite, movable=movable,
    )


def test_world_accessors_find_objects_by_kind():
    world = World(
        [
            _object("target", "target", (0, 0)),
            _object("obstacle_0", "obstacle", (10, 10)),
            _object("obstacle_1", "obstacle", (20, 20)),
            _object("attacker", "attacker", (30, 30), movable=True),
        ]
    )
    assert world.target().id == "target"
    assert {o.id for o in world.obstacles()} == {"obstacle_0", "obstacle_1"}
    assert world.attacker().id == "attacker"
    assert len(world) == 4


def test_excluding_drops_only_the_named_kind():
    world = World([_object("target", "target", (0, 0)), _object("attacker", "attacker", (1, 1))])
    clean = world.excluding("attacker")
    assert len(clean) == 1
    assert clean.of_kind("attacker") == []
    assert len(world) == 2, "excluding() must not mutate the original world"


def test_paint_order_puts_the_attacker_on_top():
    world = World(
        [
            _object("attacker", "attacker", (0, 0)),
            _object("obstacle_0", "obstacle", (0, 0)),
            _object("target", "target", (0, 0)),
            _object("background", "background", (0, 0)),
        ]
    )
    kinds_in_order = [o.kind for o in world.objects]
    assert kinds_in_order == ["background", "target", "obstacle", "attacker"]


ALL_ENVS = ["image_env", "physics2d_env", "physics2d_env_obstacles", "physics3d_env"]
#: Stage 3's backdrop is generated and owned by Renderer3D, not a WorldObject
#: (see DESIGN.md section 5) -- it is not something an attacker could ever
#: occupy or occlude, unlike the 2D stages' background.
ENVS_WITH_BACKGROUND_OBJECT = ["image_env", "physics2d_env", "physics2d_env_obstacles"]


@pytest.mark.parametrize("env_name", ALL_ENVS)
def test_the_attacker_is_the_only_object_that_moves(env_name, request):
    """The core claim behind the World refactor -- true for all three stages,
    not just the physics ones: an attack cannot touch pixels belonging to any
    object other than the one it owns."""
    env = request.getfixturevalue(env_name)
    env.reset(seed=0)
    world = env._world  # test-only: reach through the seal to check the invariant directly
    target_before = world.target().position.copy()
    obstacles_before = [o.position.copy() for o in world.obstacles()]
    background_before = (
        world.background().position.copy() if env_name in ENVS_WITH_BACKGROUND_OBJECT else None
    )

    rng = np.random.default_rng(1)
    for _ in range(6):
        env.step(env.action_space().sample(rng))

    assert np.array_equal(world.target().position, target_before)
    assert all(
        np.array_equal(o.position, before)
        for o, before in zip(world.obstacles(), obstacles_before)
    )
    if background_before is not None:
        assert np.array_equal(world.background().position, background_before)


@pytest.mark.parametrize("env_name", ALL_ENVS)
def test_only_the_attacker_object_is_movable(env_name, request):
    env = request.getfixturevalue(env_name)
    env.reset(seed=0)
    world = env._world
    assert world.attacker().movable is True
    assert all(not o.movable for o in world.obstacles())
    assert not world.target().movable
    if env_name in ENVS_WITH_BACKGROUND_OBJECT:
        assert not world.background().movable


def test_stage1_target_is_a_real_non_rectangular_cutout(image_env):
    """Stage 1's target must be a genuine alpha-matted object, not a bounding
    box: some pixels inside its own sprite's rectangle are fully transparent."""
    image_env.reset(seed=0)
    sprite = image_env._world.target().sprite
    alpha = np.asarray(sprite.getchannel("A"))
    assert alpha.min() == 0, "corners of a real cutout should be transparent"
    assert alpha.max() == 255, "the object itself should be opaque"


def test_stage1_attacker_object_reflects_the_last_action(image_env):
    image_env.reset(seed=0)
    size = float(image_env.action_space().high[2])  # the legal maximum
    n_texture = image_env.action_space().n - 4
    texture = np.full(n_texture, 0.2)  # a dark grey pattern, easy to check for
    image_env.step(np.concatenate([[120.0, 140.0, size, 15.0], texture]))
    attacker = image_env._world.attacker()
    assert np.allclose(attacker.position, [120.0, 140.0])
    assert attacker.rotation_deg == 15.0
    assert attacker.sprite.size == (round(size), round(size))
    # the chosen texture, not a fixed/random one, actually got rendered
    pixel = attacker.sprite.getpixel((attacker.sprite.width // 2, attacker.sprite.height // 2))
    assert pixel[:3] == (round(0.2 * 255),) * 3


# ---------------------------------------------------------------- boundary
# "the attack shouldn't overtake the object's boundary": a physical patch
# cannot be bigger than the object it attacks, in any of the three stages.


def test_stage1_action_space_caps_size_at_the_targets_own_silhouette(image_env):
    """Bounded by the narrowest row of the target's own silhouette, not its
    bounding-box width -- a non-rectangular object is not the same width at
    every height (see rendering.image_renderer.silhouette_min_span)."""
    from rendering.image_renderer import silhouette_min_span

    target_min_dim = silhouette_min_span(image_env._world.target().sprite)
    max_size = image_env.action_space().high[2]
    assert max_size <= target_min_dim + 1e-6


def test_stage1_max_size_survives_worst_case_rotation(image_env):
    """A rotated square's axis-aligned footprint grows by up to sqrt(2); the
    legal max size must already account for that, not just the side length at
    rotation=0, or a 45-degree patch could still spill past the target."""
    import math

    from rendering.image_renderer import silhouette_min_span

    target_min_dim = silhouette_min_span(image_env._world.target().sprite)
    max_size = float(image_env.action_space().high[2])
    worst_case_footprint = max_size * math.sqrt(2)
    assert worst_case_footprint <= target_min_dim + 1e-6


def test_stage1_rejects_a_patch_max_frac_above_one(synthetic_cutout_sprite, victim):
    from environments.image_env import ImageEnvConfig, ImageEnvironment

    with pytest.raises(ValueError):
        ImageEnvironment(
            ImageEnvConfig(
                target_sprite_path=synthetic_cutout_sprite,
                render_size=192,
                obs_size=32,
                target_class="target",
                patch_max_frac=1.5,  # bigger than the target -- must be rejected
            ),
            victim=victim,
        )


@pytest.mark.parametrize("env_name", ["physics2d_env", "physics3d_env"])
def test_physics_attacker_patch_never_exceeds_the_targets_own_silhouette(env_name, request):
    """Bounded by the narrowest row of the target's own silhouette, not its
    bounding-box width -- same reasoning as the Stage 1 test above."""
    from rendering.image_renderer import silhouette_min_span

    env = request.getfixturevalue(env_name)
    env.reset(seed=0)
    world = env._world
    target = world.target()
    min_span_px = silhouette_min_span(target.sprite)
    if env_name == "physics2d_env":
        target_min_dim = min_span_px  # Stage 2's world *is* pixel space
    else:
        world_per_px = (2.0 * target.half_extents[1]) / target.sprite.height
        target_min_dim = min_span_px * world_per_px
    attacker_size = 2.0 * min(world.attacker().half_extents[:2])
    assert attacker_size <= target_min_dim + 1e-6


def test_physics2d_rejects_a_patch_frac_above_one(victim):
    from environments.physics2d_env import Physics2DEnvConfig, Physics2DEnvironment

    with pytest.raises(ValueError):
        Physics2DEnvironment(
            Physics2DEnvConfig(render_size=192, obs_size=32, target_class="target", patch_frac=1.2),
            victim=victim,
        )


def test_physics3d_rejects_a_patch_world_frac_above_one(victim):
    from environments.physics3d_env import Physics3DEnvConfig, Physics3DEnvironment

    with pytest.raises(ValueError):
        Physics3DEnvironment(
            Physics3DEnvConfig(
                render_size=192, obs_size=32, target_class="target", patch_world_frac=1.2
            ),
            victim=victim,
        )
