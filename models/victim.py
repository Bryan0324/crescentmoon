"""Victim model interface.

The victim is *owned by the environment*.  An agent never sees this module: it
gets a scalar reward computed from the detections, nothing else -- no logits, no
features, no gradients, no parameters.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

__all__ = ["Detection", "Detections", "VictimModel", "iou"]


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class Detections(list):
    """A list of :class:`Detection` with a couple of query helpers."""

    def of_class(self, cls_name: str) -> "Detections":
        return Detections(d for d in self if d.cls_name == cls_name)

    def best(self, cls_name: str | None = None) -> Detection | None:
        pool = self if cls_name is None else self.of_class(cls_name)
        return max(pool, key=lambda d: d.confidence, default=None)

    def best_matching(
        self,
        bbox: tuple[float, float, float, float],
        cls_name: str | None = None,
        iou_threshold: float = 0.2,
    ) -> Detection | None:
        """Highest-confidence detection overlapping ``bbox``.

        Used to follow *one specific* target across an episode instead of
        accidentally rewarding the agent for suppressing some other object.
        """
        pool = self if cls_name is None else self.of_class(cls_name)
        matches = [d for d in pool if iou(d.bbox, bbox) >= iou_threshold]
        return max(matches, key=lambda d: d.confidence, default=None)


class VictimModel(abc.ABC):
    """A frozen, pretrained detector called by the environment only."""

    name: str = "victim"

    @abc.abstractmethod
    def detect(self, image_rgb: np.ndarray) -> Detections:
        """Run detection on an HxWx3 uint8 RGB image."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
