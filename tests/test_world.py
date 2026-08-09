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
        ]
    )
    kinds_in_order = [o.kind for o in world.objects]
    assert kinds_in_order == ["target", "obstacle", "attacker"]


@pytest.mark.parametrize("env_name", ["physics2d_env", "physics2d_env_obstacles", "physics3d_env"])
def test_the_attacker_is_the_only_object_that_moves(env_name, request):
    """The core claim behind the World refactor: an attack cannot touch pixels
    belonging to any object other than the one it owns."""
    env = request.getfixturevalue(env_name)
    env.reset(seed=0)
    world = env._world  # test-only: reach through the seal to check the invariant directly
    target_before = world.target().position.copy()
    obstacles_before = [o.position.copy() for o in world.obstacles()]

    rng = np.random.default_rng(1)
    for _ in range(6):
        env.step(env.action_space().sample(rng))

    assert np.array_equal(world.target().position, target_before)
    assert all(
        np.array_equal(o.position, before)
        for o, before in zip(world.obstacles(), obstacles_before)
    )


def test_only_the_attacker_object_is_movable(physics2d_env):
    physics2d_env.reset(seed=0)
    world = physics2d_env._world
    assert world.attacker().movable is True
    assert all(not o.movable for o in world.obstacles())
    assert not world.target().movable
