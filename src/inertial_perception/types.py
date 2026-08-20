from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class ImuSample:
    timestamp: float
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray

@dataclass
class CameraFeature:
    feature_id: int
    u: float
    v: float
    confidence: float = 1.0
    world_position: np.ndarray | None = None

@dataclass
class CameraFrame:
    timestamp: float
    features: list[CameraFeature]
    image_rgb: np.ndarray | None = None
    render_width: int = 0
    render_height: int = 0

@dataclass
class RayMeasurement:
    direction: np.ndarray
    distance: float
    confidence: float = 1.0

@dataclass
class RangeFrame:
    timestamp: float
    rays: list[RayMeasurement]
    rows: int
    cols: int

@dataclass
class State:
    timestamp: float
    orientation: Rotation
    gyro_bias: np.ndarray

@dataclass
class GroundTruthState:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation: Rotation
    angular_velocity: np.ndarray
