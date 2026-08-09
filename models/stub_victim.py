"""A tiny deterministic victim used for tests and offline smoke-runs.

It is *not* a research result: it detects one signature colour and reports a
confidence proportional to how much of that colour is still visible.  It exists
so the full Environment -> Victim -> Reward pipeline can be exercised (unit
tests, CI, machines with no YOLO weights) with zero downloads, and because it
occludes monotonically it makes the reward contract easy to assert.
"""

from __future__ import annotations

import numpy as np

from .victim import Detection, Detections, VictimModel

__all__ = ["ColorBlobVictim"]


class ColorBlobVictim(VictimModel):
    def __init__(
        self,
        target_rgb: tuple[int, int, int] = (0, 200, 0),
        tolerance: int = 45,
        cls_name: str = "target",
        reference_area: float | None = None,
    ) -> None:
        self.target_rgb = np.asarray(target_rgb, dtype=np.int16)
        self.tolerance = int(tolerance)
        self.cls_name = cls_name
        self._reference_area = reference_area
        self.name = f"colorblob:{cls_name}"

    def detect(self, image_rgb: np.ndarray) -> Detections:
        img = np.asarray(image_rgb, dtype=np.int16)
        mask = np.all(np.abs(img - self.target_rgb) <= self.tolerance, axis=-1)
        area = float(mask.sum())
        if area < 4:
            return Detections()

        if self._reference_area is None:
            # Calibrate on the first (clean) frame we ever see.
            self._reference_area = max(area, 1.0)

        ys, xs = np.nonzero(mask)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        confidence = float(np.clip(area / self._reference_area, 0.0, 1.0))
        return Detections(
            [Detection(cls_id=0, cls_name=self.cls_name, confidence=confidence, bbox=bbox)]
        )
