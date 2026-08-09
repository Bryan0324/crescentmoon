"""Uniform random search over the legal action space -- the honest lower bound."""

from __future__ import annotations

import numpy as np

from .base import AttackAgent

__all__ = ["RandomAgent"]


class RandomAgent(AttackAgent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return self.action_space.sample(self._rng)
