"""The one reward definition shared by every stage.

Meaning is fixed across Stage 1 / 2 / 3 (prompt.md section 12): *the better the
attack, the higher the reward*.  Only the weights change per stage, never the
shape.  The agent never computes any of this -- the environment does.

    reward = w_conf * confidence_drop
             - w_move * action_cost
             - w_invalid * invalid_action
             + success_bonus (once, on termination by success)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["RewardConfig", "RewardBreakdown", "AttackReward"]


@dataclass(frozen=True)
class RewardConfig:
    w_conf: float = 1.0
    w_move: float = 0.05
    w_invalid: float = 0.25
    success_threshold: float = 0.25
    success_bonus: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-step accounting, kept for plots -- privileged, not sent to the agent."""

    total: float
    attack_term: float
    move_cost: float
    invalid_penalty: float
    bonus: float
    baseline_confidence: float
    current_confidence: float

    @property
    def confidence_drop(self) -> float:
        return self.baseline_confidence - self.current_confidence

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence_drop"] = self.confidence_drop
        return d


class AttackReward:
    """Turn a victim-model outcome into a scalar reward."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def is_success(self, current_confidence: float) -> bool:
        """The attack succeeded once the victim's confidence falls below threshold."""
        return current_confidence < self.config.success_threshold

    def __call__(
        self,
        *,
        baseline_confidence: float,
        current_confidence: float,
        action_cost: float = 0.0,
        invalid: bool = False,
        success: bool = False,
    ) -> RewardBreakdown:
        cfg = self.config
        attack_term = cfg.w_conf * (baseline_confidence - current_confidence)
        move_cost = cfg.w_move * float(action_cost)
        invalid_penalty = cfg.w_invalid * float(bool(invalid))
        bonus = cfg.success_bonus if success else 0.0
        total = attack_term - move_cost - invalid_penalty + bonus
        return RewardBreakdown(
            total=float(total),
            attack_term=float(attack_term),
            move_cost=float(move_cost),
            invalid_penalty=float(invalid_penalty),
            bonus=float(bonus),
            baseline_confidence=float(baseline_confidence),
            current_confidence=float(current_confidence),
        )
