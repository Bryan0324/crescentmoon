"""The experiment loop.

This is the enforcement point for the project's core rule.  The agent is only
ever handed ``observation`` / ``reward`` / ``terminated`` / ``truncated``, and
it talks to the environment through a :class:`SealedEnvironment`.  Everything
used to build metrics and figures is read from the *unsealed* handle, which the
agent does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from agents.base import AttackAgent
from environments.base import BaseEnvironment
from environments.sealed import seal

from .metrics import EpisodeRecord

__all__ = ["EpisodeTrace", "run_episodes"]


@dataclass
class EpisodeTrace:
    """Per-step privileged detail for one episode (plots and figures only)."""

    record: EpisodeRecord
    telemetry: dict[str, Any]
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def confidences(self) -> list[float]:
        return [s["current_confidence"] for s in self.telemetry.get("steps", [])]

    def best_frame(self) -> np.ndarray | None:
        """The frame where the victim was least confident."""
        if not self.frames or not self.confidences:
            return None
        return self.frames[int(np.argmin(self.confidences))]


def run_episodes(
    env: BaseEnvironment,
    agent: AttackAgent,
    n_episodes: int,
    *,
    method: str | None = None,
    variant: str = "default",
    seed: int = 0,
    collect_frames: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[EpisodeRecord], list[EpisodeTrace]]:
    """Run ``n_episodes`` of ``agent`` against ``env`` and return records + traces."""
    api = seal(env)  # the agent-facing handle: five methods, nothing else
    agent.setup(api.action_space(), api.observation_space())
    method = method or agent.name

    records: list[EpisodeRecord] = []
    traces: list[EpisodeTrace] = []

    for episode in range(n_episodes):
        observation = api.reset(seed=seed + episode)
        agent.start_episode()

        total_reward = 0.0
        frames: list[np.ndarray] = []
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(observation)
            observation, reward, terminated, truncated, _info = api.step(action)
            agent.observe_step(observation, reward, terminated, truncated)
            total_reward += float(reward)
            if collect_frames:
                frames.append(env.render_human())

        agent.end_episode(total_reward)

        telemetry = env.pop_telemetry()
        steps = telemetry.get("steps", [])
        confidences = [s["current_confidence"] for s in steps] or [
            telemetry.get("baseline_confidence", 0.0)
        ]
        record = EpisodeRecord(
            method=method,
            stage=telemetry.get("stage", "unknown"),
            variant=variant,
            episode=episode,
            steps=len(steps),
            total_reward=total_reward,
            baseline_confidence=float(telemetry.get("baseline_confidence", 0.0)),
            final_confidence=float(confidences[-1]),
            best_confidence=float(min(confidences)),
            # "the attack worked at least once during the episode" -- read from
            # telemetry rather than from `terminated`, because the physics stages
            # deliberately run fixed-length episodes (see terminate_on_success).
            success=bool(terminated or any(s.get("attack_success", False) for s in steps)),
            movement_cost=float(sum(s.get("action_cost", 0.0) for s in steps)),
            invalid_steps=int(sum(not s.get("action_valid", True) for s in steps)),
            collisions=int(sum(bool(s.get("collided", False)) for s in steps)),
        )
        records.append(record)
        traces.append(EpisodeTrace(record=record, telemetry=telemetry, frames=frames))

        if progress is not None:
            progress(episode + 1, n_episodes)

    return records, traces
