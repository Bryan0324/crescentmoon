"""Agents are environment-agnostic, and the attack actually does something."""

from __future__ import annotations

import numpy as np
import pytest

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from evaluation.metrics import summarize
from evaluation.runner import run_episodes


@pytest.mark.parametrize("agent_cls", [RandomAgent, GreedyAgent])
def test_the_same_agent_runs_in_every_environment(any_env, agent_cls):
    """Section 3: an agent is set up from the public spaces alone."""
    records, traces = run_episodes(any_env, agent_cls(seed=0), n_episodes=2, seed=0)
    assert len(records) == 2
    assert all(r.steps >= 1 for r in records)
    assert all(t.telemetry["steps"] for t in traces)


def test_agents_only_ever_receive_observation_and_reward(any_env):
    """The agent API has no ``info`` and no environment handle -- verify by signature."""
    import inspect

    from agents.base import AttackAgent

    act_params = set(inspect.signature(AttackAgent.act).parameters)
    feedback_params = set(inspect.signature(AttackAgent.observe_step).parameters)
    assert act_params == {"self", "observation"}
    assert feedback_params == {"self", "observation", "reward", "terminated", "truncated"}


def test_occluding_the_target_lowers_the_victims_confidence(image_env):
    """The core Stage 1 claim, on the deterministic stub victim."""
    image_env.reset(seed=0)
    telemetry = image_env.pop_telemetry()
    baseline = telemetry["baseline_confidence"]
    assert baseline > 0.5

    size = image_env.action_space().high[2]
    centre = image_env.action_space().high[0] / 2
    image_env.reset(seed=0)
    image_env.step(np.array([centre, centre, size, 0.0]))
    attacked = image_env.pop_telemetry()["steps"][-1]["current_confidence"]

    assert attacked < baseline


def test_search_finds_a_better_attack_than_its_first_guess(image_env):
    records, _ = run_episodes(image_env, RandomAgent(seed=1), n_episodes=25, seed=0)
    confidences = [r.best_confidence for r in records]
    assert min(confidences) < confidences[0] or min(confidences) < records[0].baseline_confidence

    summary = summarize(records)
    assert 0.0 <= summary["attack_success_rate"] <= 1.0
    assert summary["n_episodes"] == 25


def test_obstacles_restrict_where_the_attacker_can_go(
    physics2d_env, physics2d_env_obstacles
):
    """Q4 in miniature: the same push produces less progress once a wall is there."""
    def travel(env) -> float:
        env.reset(seed=0)
        start = None
        end = None
        for _ in range(8):
            _, _, terminated, truncated, _ = env.step(np.array([0.6, -0.8]))
            if terminated or truncated:
                break
        steps = env.pop_telemetry()["steps"]
        start, end = steps[0]["position"], steps[-1]["position"]
        return float(np.linalg.norm(np.asarray(end) - np.asarray(start)))

    assert travel(physics2d_env_obstacles) < travel(physics2d_env)
