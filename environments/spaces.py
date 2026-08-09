"""Minimal space definitions for the Environment API.

These are intentionally *not* Gymnasium spaces: the core Environment API must be
usable without any RL framework installed.  ``environments/gym_adapter.py``
converts them to Gymnasium spaces when Stable-Baselines3 is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BoxSpace", "ImageSpace", "DictSpace"]


@dataclass(frozen=True)
class BoxSpace:
    """A bounded continuous vector space.

    ``names`` documents the semantic meaning of each component so an agent can
    print what it is allowed to do without knowing the environment class.
    """

    low: np.ndarray
    high: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float32)
        high = np.asarray(self.high, dtype=np.float32)
        if low.shape != high.shape:
            raise ValueError(f"low/high shape mismatch: {low.shape} vs {high.shape}")
        if len(self.names) != low.shape[0]:
            raise ValueError("names must have one entry per dimension")
        if np.any(high < low):
            raise ValueError("high must be >= low")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.low.shape)

    @property
    def n(self) -> int:
        return int(self.low.shape[0])

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high).astype(np.float32)

    def clip(self, value) -> np.ndarray:
        return np.clip(np.asarray(value, dtype=np.float32), self.low, self.high)

    def contains(self, value) -> bool:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != self.shape:
            return False
        return bool(np.all(arr >= self.low - 1e-6) and np.all(arr <= self.high + 1e-6))

    def describe(self) -> str:
        rows = [
            f"    {name:<16} [{lo:+.2f}, {hi:+.2f}]"
            for name, lo, hi in zip(self.names, self.low, self.high)
        ]
        return "BoxSpace(\n" + "\n".join(rows) + "\n)"


@dataclass(frozen=True)
class ImageSpace:
    """An RGB uint8 image observation of fixed size."""

    height: int
    width: int
    channels: int = 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def contains(self, value) -> bool:
        arr = np.asarray(value)
        return arr.shape == self.shape and arr.dtype == np.uint8

    def describe(self) -> str:
        return f"ImageSpace(shape={self.shape}, dtype=uint8, range=[0, 255])"


@dataclass(frozen=True)
class DictSpace:
    """A dictionary of named sub-spaces -- the observation space of every env."""

    spaces: dict[str, BoxSpace | ImageSpace] = field(default_factory=dict)

    def keys(self):
        return self.spaces.keys()

    def items(self):
        return self.spaces.items()

    def __getitem__(self, key: str):
        return self.spaces[key]

    def contains(self, value) -> bool:
        if not isinstance(value, dict) or set(value) != set(self.spaces):
            return False
        return all(space.contains(value[key]) for key, space in self.spaces.items())

    def describe(self) -> str:
        rows = [f"  {key}: {space.describe()}" for key, space in self.spaces.items()]
        return "DictSpace(\n" + "\n".join(rows) + "\n)"
