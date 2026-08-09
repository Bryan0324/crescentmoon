"""Episode records and the metrics every stage reports.

The metric set is fixed across Stage 1/2/3 (prompt.md sections 7 and 13) so the
three stages can be put in one table:

    attack success rate, confidence drop, average reward, movement cost,
    episode length.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = ["EpisodeRecord", "summarize", "to_dataframe", "save_records", "load_records"]


@dataclass
class EpisodeRecord:
    method: str
    stage: str
    variant: str
    episode: int
    steps: int
    total_reward: float
    baseline_confidence: float
    final_confidence: float
    best_confidence: float      # lowest confidence reached during the episode
    success: bool
    movement_cost: float
    invalid_steps: int
    collisions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_drop(self) -> float:
        return self.baseline_confidence - self.best_confidence

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence_drop"] = self.confidence_drop
        return d


def summarize(records: Iterable[EpisodeRecord]) -> dict[str, Any]:
    records = list(records)
    if not records:
        return {"n_episodes": 0}

    drops = np.array([r.confidence_drop for r in records], dtype=float)
    rewards = np.array([r.total_reward for r in records], dtype=float)
    steps = np.array([r.steps for r in records], dtype=float)
    costs = np.array([r.movement_cost for r in records], dtype=float)
    best = np.array([r.best_confidence for r in records], dtype=float)
    base = np.array([r.baseline_confidence for r in records], dtype=float)
    success = np.array([r.success for r in records], dtype=float)
    invalid = np.array([r.invalid_steps for r in records], dtype=float)

    return {
        "method": records[0].method,
        "stage": records[0].stage,
        "variant": records[0].variant,
        "n_episodes": len(records),
        "attack_success_rate": float(success.mean()),
        # For a *search* (stage 1) the deliverable is the single best attack found
        # inside the query budget, not the average over every query tried.
        "best_confidence_found": float(best.min()),
        "best_confidence_drop": float((base - best).max()),
        "mean_baseline_confidence": float(base.mean()),
        "mean_best_confidence": float(best.mean()),
        "mean_confidence_drop": float(drops.mean()),
        "std_confidence_drop": float(drops.std()),
        "relative_confidence_drop": float(drops.mean() / max(1e-9, base.mean())),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_episode_length": float(steps.mean()),
        "mean_movement_cost": float(costs.mean()),
        "mean_invalid_steps": float(invalid.mean()),
    }


def to_dataframe(records: Iterable[EpisodeRecord]):
    import pandas as pd

    return pd.DataFrame([r.to_dict() for r in records])


def save_records(path: str | Path, records: Iterable[EpisodeRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in records]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_records(path: str | Path) -> list[EpisodeRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for row in payload:
        row = dict(row)
        row.pop("confidence_drop", None)
        out.append(EpisodeRecord(**row))
    return out
