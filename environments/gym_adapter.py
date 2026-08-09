"""Gymnasium adapter -- the *only* place the project depends on an RL framework.

Stable-Baselines3 needs a ``gymnasium.Env``; our Environment API is smaller and
framework-free.  This shim translates between them, and it works on a
:class:`~environments.sealed.SealedEnvironment`, which proves PPO needs nothing
beyond the five public calls.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces as gym_spaces

from .base import BaseEnvironment
from .sealed import SealedEnvironment, seal
from .spaces import BoxSpace, ImageSpace

__all__ = ["GymEnvAdapter", "make_gym_env"]


def _to_gym_space(space):
    if isinstance(space, ImageSpace):
        return gym_spaces.Box(low=0, high=255, shape=space.shape, dtype=np.uint8)
    if isinstance(space, BoxSpace):
        return gym_spaces.Box(low=space.low, high=space.high, dtype=np.float32)
    raise TypeError(f"unsupported space: {type(space).__name__}")


class GymEnvAdapter(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: BaseEnvironment | SealedEnvironment) -> None:
        super().__init__()
        self._env = seal(env)
        self.action_space = _to_gym_space(self._env.action_space())
        self.observation_space = gym_spaces.Dict(
            {key: _to_gym_space(sub) for key, sub in self._env.observation_space().items()}
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self._env.reset(seed=seed)
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)


def make_gym_env(env_factory: Callable[[], BaseEnvironment]) -> Callable[[], GymEnvAdapter]:
    """Build a thunk suitable for SB3's vectorised env constructors."""

    def _thunk() -> GymEnvAdapter:
        return GymEnvAdapter(env_factory())

    return _thunk
