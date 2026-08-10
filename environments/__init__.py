"""Environments: the only thing an attack agent is allowed to talk to."""

from .base import BaseEnvironment, Observation, StepResult
from .image_env import ImageEnvConfig, ImageEnvironment
from .physics2d_env import ObstacleSpec, Physics2DEnvConfig, Physics2DEnvironment
from .physics3d_env import ObstacleSpec3D, Physics3DEnvConfig, Physics3DEnvironment
from .sealed import EnvironmentAccessError, SealedEnvironment, seal
from .spaces import BoxSpace, DictSpace, ImageSpace
from .world import ObjectKind, World, WorldObject

__all__ = [
    "BaseEnvironment",
    "Observation",
    "StepResult",
    "BoxSpace",
    "DictSpace",
    "ImageSpace",
    "seal",
    "SealedEnvironment",
    "EnvironmentAccessError",
    "World",
    "WorldObject",
    "ObjectKind",
    "ImageEnvironment",
    "ImageEnvConfig",
    "Physics2DEnvironment",
    "Physics2DEnvConfig",
    "ObstacleSpec",
    "Physics3DEnvironment",
    "Physics3DEnvConfig",
    "ObstacleSpec3D",
]
