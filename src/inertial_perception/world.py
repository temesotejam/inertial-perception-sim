from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .types import GroundTruthState

LANDMARKS=np.array([[4,-1.8,.4],[4,-1,1.4],[4,-.2,.8],[4,.5,1.8],[4,1.2,.5],[4,1.8,1.3],[3,-1.3,2],[3.5,.9,2.2],[5,0,1.1]],float)

def euler_profile(t: float, scenario: str):
    if scenario=="static": e=np.zeros(3)
    elif scenario=="roll": e=np.radians([20*np.sin(2*np.pi*.35*t),0,0])
    elif scenario=="pitch": e=np.radians([0,18*np.sin(2*np.pi*.28*t),0])
    elif scenario=="yaw": e=np.radians([0,0,35*np.sin(2*np.pi*.18*t)])
    else: e=np.radians([16*np.sin(2*np.pi*.31*t),12*np.sin(2*np.pi*.23*t+.4),28*np.sin(2*np.pi*.17*t+.8)])
    return e

def truth_at(t: float, scenario: str, eps: float=1e-4) -> GroundTruthState:
    r=Rotation.from_euler("xyz",euler_profile(t,scenario)); r2=Rotation.from_euler("xyz",euler_profile(t+eps,scenario))
    omega=(r.inv()*r2).as_rotvec()/eps
    return GroundTruthState(t,np.array([0.,0.,1.]),np.zeros(3),np.zeros(3),r,omega)
