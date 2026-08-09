"""YOLO victim model (Ultralytics), pretrained and frozen.

Never trained, never differentiated through, never handed to an agent.  The
environment holds the only reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .victim import Detection, Detections, VictimModel

__all__ = ["YOLOVictim"]

_DEFAULT_WEIGHTS = "yolov8n.pt"


class YOLOVictim(VictimModel):
    """Frozen Ultralytics YOLO detector.

    Parameters
    ----------
    weights:
        Ultralytics checkpoint name or path (default ``yolov8n.pt``).
    conf_threshold:
        Minimum confidence kept in the returned detections.  Keep it low
        (0.05) so the environment can observe *gradual* confidence decay
        rather than a step function at the default 0.25.
    imgsz:
        Inference resolution.  320 keeps CPU rollouts fast enough for PPO.
    """

    def __init__(
        self,
        weights: str | Path = _DEFAULT_WEIGHTS,
        conf_threshold: float = 0.05,
        imgsz: int = 320,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        # Keep Ultralytics quiet and offline-friendly inside notebooks.
        os.environ.setdefault("YOLO_VERBOSE", "false")

        self._model = YOLO(str(weights))
        self._model.to(device)
        # Frozen: eval mode + no gradients.  The RL loop must never update it.
        self._model.model.eval()
        for param in self._model.model.parameters():
            param.requires_grad_(False)

        self._names: dict[int, str] = dict(self._model.names)
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.device = device
        self.name = f"yolo:{Path(str(weights)).stem}"

    @property
    def class_names(self) -> list[str]:
        return list(self._names.values())

    def detect(self, image_rgb: np.ndarray) -> Detections:
        import torch

        with torch.inference_mode():
            results = self._model.predict(
                image_rgb[:, :, ::-1],  # ultralytics expects BGR for ndarray input
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )

        out = Detections()
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, conf, cls):
                out.append(
                    Detection(
                        cls_id=int(cls_id),
                        cls_name=self._names.get(int(cls_id), str(cls_id)),
                        confidence=float(score),
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    )
                )
        return out
