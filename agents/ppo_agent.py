"""PPO attack agent (Stable-Baselines3), used from Stage 2 onwards.

PPO is trained *through the same sealed Environment API* as the random and
greedy baselines -- see :class:`environments.gym_adapter.GymEnvAdapter`.  No
gradient ever flows into the victim model; the only learning signal is the
scalar reward the environment hands back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import AttackAgent

__all__ = ["PPOAgent", "TrainingCurve"]


class TrainingCurve:
    """Collects episode returns during learning so we can plot a reward curve."""

    def __init__(self) -> None:
        self.timesteps: list[int] = []
        self.returns: list[float] = []
        self.lengths: list[int] = []

    def to_dict(self) -> dict[str, list]:
        return {
            "timesteps": self.timesteps,
            "returns": self.returns,
            "lengths": self.lengths,
        }


def _make_callback(curve: TrainingCurve):
    from stable_baselines3.common.callbacks import BaseCallback

    class _CurveCallback(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("episode")
                if episode is not None:
                    curve.timesteps.append(int(self.num_timesteps))
                    curve.returns.append(float(episode["r"]))
                    curve.lengths.append(int(episode["l"]))
            return True

    return _CurveCallback()


DEFAULT_PPO_KWARGS: dict[str, Any] = {
    "policy": "MultiInputPolicy",
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 6,
    "learning_rate": 3e-4,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "verbose": 0,
}


class PPOAgent(AttackAgent):
    name = "ppo"

    def __init__(self, model=None, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic
        self.curve = TrainingCurve()

    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        env_factory: Callable[[], Any],
        total_timesteps: int = 12_000,
        seed: int = 0,
        deterministic: bool = True,
        progress: bool = False,
        **ppo_kwargs: Any,
    ) -> "PPOAgent":
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        from environments.gym_adapter import GymEnvAdapter

        kwargs = {**DEFAULT_PPO_KWARGS, **ppo_kwargs}
        env = Monitor(GymEnvAdapter(env_factory()))

        agent = cls(deterministic=deterministic)
        agent.model = PPO(env=env, seed=seed, **kwargs)
        agent.model.learn(
            total_timesteps=total_timesteps,
            callback=_make_callback(agent.curve),
            progress_bar=progress,
        )
        return agent

    # ------------------------------------------------------------------
    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PPOAgent has no policy: call train() or load() first")
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> "PPOAgent":
        from stable_baselines3 import PPO

        return cls(model=PPO.load(str(path)), deterministic=deterministic)
