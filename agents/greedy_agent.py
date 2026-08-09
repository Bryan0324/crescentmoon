"""Greedy local search (a (1+1) hill climber) driven only by the reward signal.

The same code is the Stage 1 and the Stage 2/3 baseline:

* Stage 1 has one step per episode, so "keep the best action, perturb it" is
  exactly random-restart hill climbing over placements.
* Stage 2/3 have many steps per episode, so the same rule becomes "repeat a
  move that helped, perturb after a move that did not".

It never looks at the image and never knows which environment it is in -- it
only needs ``action_space()`` and the scalar reward.
"""

from __future__ import annotations

import numpy as np

from .base import AttackAgent

__all__ = ["GreedyAgent"]


class GreedyAgent(AttackAgent):
    name = "greedy"

    def __init__(
        self,
        seed: int = 0,
        sigma: float = 0.35,
        sigma_min: float = 0.05,
        sigma_max: float = 0.6,
        shrink: float = 0.85,
        grow: float = 1.12,
        score_decay: float = 0.97,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.sigma0 = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.shrink = shrink
        self.grow = grow
        self.score_decay = score_decay

        self._sigma = sigma
        self._best: np.ndarray | None = None       # normalised to [0, 1]^n
        self._best_score = -np.inf
        self._proposal: np.ndarray | None = None
        self._repeat = False

    # ------------------------------------------------------------------
    def setup(self, action_space, observation_space) -> None:
        super().setup(action_space, observation_space)
        self._best = np.full(action_space.n, 0.5)
        self._best_score = -np.inf
        self._sigma = self.sigma0

    def _denormalise(self, unit: np.ndarray) -> np.ndarray:
        lo, hi = self.action_space.low, self.action_space.high
        return (lo + np.clip(unit, 0.0, 1.0) * (hi - lo)).astype(np.float32)

    # ------------------------------------------------------------------
    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        assert self._best is not None, "call setup() first"
        if self._repeat:
            self._proposal = self._best.copy()  # a move that just paid off
        else:
            noise = self._rng.normal(0.0, self._sigma, size=self._best.shape)
            self._proposal = np.clip(self._best + noise, 0.0, 1.0)
        return self._denormalise(self._proposal)

    def observe_step(self, observation, reward: float, terminated: bool, truncated: bool) -> None:
        if self._proposal is None:
            return
        if reward > self._best_score:
            self._best = self._proposal.copy()
            self._best_score = float(reward)
            self._sigma = max(self.sigma_min, self._sigma * self.shrink)
            self._repeat = True
        else:
            self._sigma = min(self.sigma_max, self._sigma * self.grow)
            self._repeat = False
        # Forget slowly: in a moving world yesterday's best is not a ceiling.
        self._best_score *= self.score_decay

    def start_episode(self) -> None:
        self._repeat = False
