from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def blend_rotation(a: Rotation, b: Rotation, weight: float) -> Rotation:
    weight = float(np.clip(weight, 0.0, 1.0))
    return Slerp([0.0, 1.0], Rotation.concatenate([a, b]))([weight])[0]

def rotation_error_deg(est: Rotation, true: Rotation) -> float:
    return float(np.degrees((true * est.inv()).magnitude()))

def align_vector_correction(current_world: np.ndarray, desired_world: np.ndarray) -> Rotation:
    a=np.asarray(current_world,float); b=np.asarray(desired_world,float); a/=np.linalg.norm(a); b/=np.linalg.norm(b)
    cross=np.cross(a,b); s=np.linalg.norm(cross); c=float(np.clip(np.dot(a,b),-1.0,1.0))
    if s<1e-12:
        if c>0: return Rotation.identity()
        axis=np.cross(a,[1.,0.,0.])
        if np.linalg.norm(axis)<1e-6: axis=np.cross(a,[0.,1.,0.])
        axis/=np.linalg.norm(axis); return Rotation.from_rotvec(axis*np.pi)
    return Rotation.from_rotvec((cross/s)*np.arctan2(s,c))
