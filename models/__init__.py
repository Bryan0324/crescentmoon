"""Victim models.  Owned by the environment; never handed to an agent."""

from .stub_victim import ColorBlobVictim
from .victim import Detection, Detections, VictimModel, iou

__all__ = ["VictimModel", "Detection", "Detections", "iou", "ColorBlobVictim", "YOLOVictim"]


def __getattr__(name: str):
    if name == "YOLOVictim":  # lazy: ultralytics + torch are heavy
        from .yolo_victim import YOLOVictim

        return YOLOVictim
    raise AttributeError(name)
