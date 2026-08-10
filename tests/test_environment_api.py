"""Every environment must honour the same contract -- that is the whole project."""

from __future__ import annotations

import numpy as np
import pytest

from environments.base import BaseEnvironment, StepResult


def test_all_envs_are_base_environments(any_env):
    assert isinstance(any_env, BaseEnvironment)


def test_reset_returns_a_valid_observation(any_env):
    obs = any_env.reset(seed=1)
    assert any_env.observation_space().contains(obs)


def test_observe_matches_observation_space(any_env):
    any_env.reset(seed=1)
    assert any_env.observation_space().contains(any_env.observe())


def test_step_returns_the_five_tuple(any_env):
    any_env.reset(seed=1)
    action = any_env.action_space().sample(np.random.default_rng(0))
    result = any_env.step(action)

    assert isinstance(result, StepResult)
    obs, reward, terminated, truncated, info = result
    assert any_env.observation_space().contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_terminates_within_its_step_budget(any_env):
    any_env.reset(seed=1)
    rng = np.random.default_rng(0)
    for step in range(500):
        _, _, terminated, truncated, _ = any_env.step(any_env.action_space().sample(rng))
        if terminated or truncated:
            break
    else:  # pragma: no cover - would mean an unbounded episode
        pytest.fail("episode never ended")
    assert step < 500


def test_same_seed_gives_the_same_episode(any_env):
    rng_actions = [
        any_env.action_space().sample(np.random.default_rng(k)) for k in range(4)
    ]

    def rollout():
        any_env.reset(seed=7)
        out = []
        for action in rng_actions:
            _, reward, terminated, truncated, _ = any_env.step(action)
            out.append(reward)
            if terminated or truncated:
                break
        return out

    assert rollout() == pytest.approx(rollout())


def test_action_outside_the_space_is_clipped_not_obeyed(any_env):
    """Requirement 5: the environment enforces limits; the agent cannot bypass them."""
    any_env.reset(seed=1)
    space = any_env.action_space()
    absurd = space.high * 1000.0 + 1000.0

    _, _, _, _, info = any_env.step(absurd)
    assert info["action_valid"] is False

    applied = info.get("action_applied")
    if applied is not None:  # Stage 1 reports the clipped placement
        assert np.all(applied <= space.high + 1e-6)
        assert np.all(applied >= space.low - 1e-6)


def test_wrong_action_shape_is_rejected(any_env):
    any_env.reset(seed=1)
    with pytest.raises(ValueError):
        any_env.step(np.zeros(any_env.action_space().n + 3))


def test_observation_never_leaks_privileged_information(any_env):
    """Rule 1: no ground-truth boxes, no victim confidence in the observation."""
    obs = any_env.reset(seed=1)
    assert set(obs) == {"image", "vector"}

    forbidden = ("conf", "bbox", "target", "position", "yolo", "victim")
    names = any_env.observation_space()["vector"].names
    assert not any(bad in name.lower() for name in names for bad in forbidden)

    # The privileged data exists -- but only on the experimenter's handle.
    telemetry = any_env.pop_telemetry()
    assert "baseline_confidence" in telemetry and "baseline_bbox" in telemetry


def test_physics_episodes_are_fixed_length(physics2d_env):
    """Ending early on success would pay less than hovering at the threshold."""
    physics2d_env.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = physics2d_env.step(
            physics2d_env.action_space().sample(rng)
        )
        steps += 1
        assert not terminated, "the 2D env should not terminate early by default"
        if truncated:
            break
    assert steps == 8  # conftest sets max_steps=8


def test_success_is_read_from_telemetry_not_from_terminated(physics2d_env):
    physics2d_env.reset(seed=0)
    physics2d_env.step(np.zeros(physics2d_env.action_space().n))
    step = physics2d_env.pop_telemetry()["steps"][-1]
    assert "attack_success" in step and isinstance(step["attack_success"], bool)


PUBLIC_INFO_KEYS = {"step", "action_valid", "action_applied", "last_action_success"}


def test_info_dict_carries_no_victim_feedback(any_env):
    """Requirement 4: even ``info`` must not hand the agent the victim's output."""
    any_env.reset(seed=1)
    _, _, _, _, info = any_env.step(any_env.action_space().sample(np.random.default_rng(0)))
    assert set(info) <= PUBLIC_INFO_KEYS
