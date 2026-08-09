"""Physics limits and the reward contract."""

from __future__ import annotations

import numpy as np
import pytest

from environments.physics import AABB, Body, integrate
from reward.attack_reward import AttackReward, RewardConfig


# ---------------------------------------------------------------- physics
def _bounds(dim: int, size: float = 100.0) -> AABB:
    return AABB(np.zeros(dim), np.full(dim, size))


@pytest.mark.parametrize("dim", [2, 3])
def test_body_never_leaves_the_world(dim):
    body = Body(position=np.full(dim, 50.0), half_extents=np.full(dim, 5.0), max_step=20.0)
    push = np.ones(dim) * 50.0
    for _ in range(40):
        integrate(body, push, obstacles=[], bounds=_bounds(dim))
    assert np.all(body.position <= 95.0 + 1e-6)
    assert np.all(body.position >= 5.0 - 1e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_movement_per_step_is_capped(dim):
    body = Body(
        position=np.full(dim, 50.0),
        half_extents=np.full(dim, 1.0),
        max_speed=1000.0,
        max_step=7.0,
    )
    outcome = integrate(body, np.ones(dim) * 1e6, obstacles=[], bounds=_bounds(dim, 1e6))
    assert outcome.distance <= 7.0 + 1e-6


def test_a_body_cannot_pass_through_a_wall():
    wall = AABB(np.array([40.0, 0.0]), np.array([60.0, 100.0]))
    body = Body(position=np.array([20.0, 50.0]), half_extents=np.array([5.0, 5.0]))

    collided_at_least_once = False
    for _ in range(60):
        outcome = integrate(body, np.array([50.0, 0.0]), obstacles=[wall], bounds=_bounds(2))
        collided_at_least_once |= outcome.collided

    assert collided_at_least_once
    assert body.position[0] + 5.0 <= 40.0 + 1e-6, "the body ended up inside/behind the wall"


def test_sliding_along_a_wall_is_still_possible():
    wall = AABB(np.array([40.0, 0.0]), np.array([60.0, 60.0]))
    body = Body(position=np.array([20.0, 80.0]), half_extents=np.array([5.0, 5.0]))
    for _ in range(30):
        integrate(body, np.array([50.0, 0.0]), obstacles=[wall], bounds=_bounds(2))
    assert body.position[0] > 60.0, "the body should have passed below the wall"


# ----------------------------------------------------------------- reward
def test_reward_grows_as_the_victim_gets_less_confident():
    fn = AttackReward(RewardConfig(w_move=0.0, w_invalid=0.0, success_bonus=0.0))
    weak = fn(baseline_confidence=0.9, current_confidence=0.8).total
    strong = fn(baseline_confidence=0.9, current_confidence=0.1).total
    assert strong > weak > 0


def test_costs_and_penalties_reduce_reward():
    cfg = RewardConfig(w_move=0.5, w_invalid=1.0, success_bonus=0.0)
    fn = AttackReward(cfg)
    free = fn(baseline_confidence=0.9, current_confidence=0.4).total
    costly = fn(baseline_confidence=0.9, current_confidence=0.4, action_cost=1.0).total
    illegal = fn(baseline_confidence=0.9, current_confidence=0.4, invalid=True).total
    assert costly == pytest.approx(free - 0.5)
    assert illegal == pytest.approx(free - 1.0)


def test_success_threshold_and_bonus():
    fn = AttackReward(RewardConfig(success_threshold=0.25, success_bonus=2.0))
    assert fn.is_success(0.2) and not fn.is_success(0.3)
    with_bonus = fn(baseline_confidence=0.9, current_confidence=0.2, success=True)
    without = fn(baseline_confidence=0.9, current_confidence=0.2, success=False)
    assert with_bonus.total == pytest.approx(without.total + 2.0)


def test_breakdown_reports_the_confidence_drop():
    fn = AttackReward()
    breakdown = fn(baseline_confidence=0.85, current_confidence=0.30)
    assert breakdown.confidence_drop == pytest.approx(0.55)
