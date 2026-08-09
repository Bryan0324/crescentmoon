"""Attack-agent interface -- identical for Stage 1, 2 and 3.

An agent sees three things and nothing else: an ``observation``, a scalar
``reward``, and the episode-end flags.  Note what is *absent* from these
signatures: no ``info`` dict, no environment handle, no victim model.  An agent
that needs more than this has broken the project's core rule.
"""

from __future__ import annotations

import abc

import numpy as np

from environments.spaces import BoxSpace, DictSpace

__all__ = ["AttackAgent"]


class AttackAgent(abc.ABC):
    name: str = "agent"

    def setup(self, action_space: BoxSpace, observation_space: DictSpace) -> None:
        """Learn what is legal to do -- via the public ``action_space()`` call only."""
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Choose an action from an observation."""

    def observe_step(
        self,
        observation: dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Feedback hook.  Deliberately has no ``info`` argument."""

    def start_episode(self) -> None:
        """Called before the first action of an episode."""

    def end_episode(self, episode_return: float) -> None:
        """Called once the episode is over."""

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
