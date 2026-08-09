"""Turn "the agent must not peek inside the environment" into a runtime guarantee.

``seal(env)`` returns a proxy that exposes *only* the five methods of the
Environment API.  Every other attribute access raises ``EnvironmentAccessError``,
so violations such as ``env.target.x`` or ``env._victim.model`` fail loudly
instead of silently producing an over-powered agent.
"""

from __future__ import annotations

from typing import Any

from .base import BaseEnvironment, Observation, StepResult
from .spaces import BoxSpace, DictSpace

__all__ = ["EnvironmentAccessError", "SealedEnvironment", "seal"]

#: The complete agent-facing surface.  Nothing else gets through.
PUBLIC_API = ("reset", "observe", "step", "action_space", "observation_space")


class EnvironmentAccessError(AttributeError):
    """Raised when something tries to reach past the Environment API."""


class SealedEnvironment:
    """A restricted handle on an environment: the Environment API and nothing else."""

    __slots__ = ("_env",)

    def __init__(self, env: BaseEnvironment) -> None:
        object.__setattr__(self, "_env", env)

    # -- the five allowed calls -------------------------------------------
    def reset(self, *, seed: int | None = None) -> Observation:
        return self._env.reset(seed=seed)

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action) -> StepResult:
        return self._env.step(action)

    def action_space(self) -> BoxSpace:
        return self._env.action_space()

    def observation_space(self) -> DictSpace:
        return self._env.observation_space()

    # -- everything else is a hard error ----------------------------------
    def __getattr__(self, name: str) -> Any:
        raise EnvironmentAccessError(
            f"'{name}' is not part of the Environment API. "
            f"An attack agent may only use {PUBLIC_API}."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise EnvironmentAccessError(
            f"cannot set '{name}': the environment owns its own state."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<SealedEnvironment action_dim={self.action_space().n}>"


def seal(env: BaseEnvironment) -> SealedEnvironment:
    """Wrap ``env`` so that only the Environment API is reachable."""
    if isinstance(env, SealedEnvironment):
        return env
    return SealedEnvironment(env)
