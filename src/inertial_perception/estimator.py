from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .types import State
from .math3d import blend_rotation, align_vector_correction

class AttitudeEstimator:
    def __init__(self,initial_orientation=None,camera_gain=.16,range_gain=.12,accel_gain=.008):
        self.orientation=initial_orientation or Rotation.identity(); self.gyro_bias=np.zeros(3); self.last_t=None; self.camera_gain=camera_gain; self.range_gain=range_gain; self.accel_gain=accel_gain; self.last_event={"kind":"init","residual_deg":0.,"correction_deg":0.}
    def propagate(self,imu):
        if self.last_t is not None:
            dt=imu.timestamp-self.last_t; self.orientation=self.orientation*Rotation.from_rotvec((imu.angular_velocity-self.gyro_bias)*dt)
            a=imu.linear_acceleration
            if np.linalg.norm(a)>1e-6:
                up=a/np.linalg.norm(a); corr=align_vector_correction(self.orientation.apply(up),np.array([0.,0.,1.])); self.orientation=blend_rotation(self.orientation,corr*self.orientation,self.accel_gain)
        self.last_t=imu.timestamp
    def update_camera(self,obs):
        residual=(obs*self.orientation.inv()).magnitude(); before=self.orientation; self.orientation=blend_rotation(self.orientation,obs,self.camera_gain); correction=(self.orientation*before.inv()).magnitude(); self.last_event={"kind":"camera","residual_deg":float(np.degrees(residual)),"correction_deg":float(np.degrees(correction))}
    def update_range_floor(self,normal_body):
        corr=align_vector_correction(self.orientation.apply(normal_body),np.array([0.,0.,1.])); residual=corr.magnitude(); before=self.orientation; self.orientation=blend_rotation(self.orientation,corr*self.orientation,self.range_gain); correction=(self.orientation*before.inv()).magnitude(); self.last_event={"kind":"range","residual_deg":float(np.degrees(residual)),"correction_deg":float(np.degrees(correction))}
    def state(self,t): return State(t,self.orientation,self.gyro_bias.copy())
