"""Attack agents.  All of them share one interface and none of them know which
environment they are attacking."""

from .base import AttackAgent
from .greedy_agent import GreedyAgent
from .random_agent import RandomAgent

__all__ = ["AttackAgent", "RandomAgent", "GreedyAgent", "PPOAgent"]


def __getattr__(name: str):
    # PPOAgent pulls in Stable-Baselines3; keep the import lazy so Stage 1 runs
    # without an RL framework installed.
    if name == "PPOAgent":
        from .ppo_agent import PPOAgent

        return PPOAgent
    raise AttributeError(name)
