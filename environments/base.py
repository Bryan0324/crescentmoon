"""The single Environment API that every stage of the project shares.

Design rule (see prompt.md, Rule 1): an Attack Agent may only touch the five
public methods declared here.  Everything else -- world state, physics,
renderer, victim model, ground-truth boxes -- lives behind this boundary and is
named with a leading underscore.  ``environments.sealed.seal()`` turns that
convention into a hard runtime guarantee.
"""

from __future__ import annotations

import abc
from typing import Any, NamedTuple

import numpy as np

from .spaces import BoxSpace, DictSpace

__all__ = ["Observation", "StepResult", "BaseEnvironment"]

Observation = dict[str, np.ndarray]


class StepResult(NamedTuple):
    """The 5-tuple returned by :meth:`BaseEnvironment.step`."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class BaseEnvironment(abc.ABC):
    """Common interface for ImageEnvironment / Physics2DEnvironment / Physics3DEnvironment.

    Only these five methods are part of the agent-facing contract:

    ``reset()``            start a new episode, return the first observation
    ``observe()``          return what the agent is allowed to know, right now
    ``step(action)``       validate + execute an action, return the 5-tuple
    ``action_space()``     what the agent may do
    ``observation_space()`` what the agent will receive

    Subclasses may add *experimenter-facing* helpers (rendering at full
    resolution, privileged telemetry for plots), but those are unreachable from
    a sealed environment handle and must never be needed to run an agent.
    """

    metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # agent-facing API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def reset(self, *, seed: int | None = None) -> Observation:
        """Begin a new episode and return the first observation."""

    @abc.abstractmethod
    def observe(self) -> Observation:
        """Return the current observation (only publicly allowed information)."""

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Validate and execute ``action``; advance the world by one tick."""

    @abc.abstractmethod
    def action_space(self) -> BoxSpace:
        """Describe the actions the agent may request."""

    @abc.abstractmethod
    def observation_space(self) -> DictSpace:
        """Describe the structure of an observation."""

    # ------------------------------------------------------------------
    # experimenter-facing API (never used by an agent)
    # ------------------------------------------------------------------
    def render_human(self) -> np.ndarray:
        """Full-resolution RGB frame -- for notebooks and figures only.

        This is the very image the victim model sees, which is why it is *not*
        part of ``observe()``: the agent receives a downscaled camera view.
        """
        raise NotImplementedError

    def victim_report(self, image_rgb: np.ndarray):
        """Run the victim on an arbitrary image, for drawing boxes in figures.

        Experimenter-only, like :meth:`render_human`: reachable from a notebook,
        unreachable from an agent.
        """
        raise NotImplementedError

    def pop_telemetry(self) -> dict[str, Any]:
        """Privileged per-episode logs (true positions, victim confidences).

        Used by the experiment runner to build metrics and plots.  Blocked by
        :func:`environments.sealed.seal`, so no agent can read it.
        """
        return {}

    # ------------------------------------------------------------------
    def close(self) -> None:  # pragma: no cover - trivial
        return None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} action_dim={self.action_space().n}>"
