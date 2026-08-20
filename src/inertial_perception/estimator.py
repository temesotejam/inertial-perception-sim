from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .types import State
from .math3d import blend_rotation, align_vector_correction

class AttitudeEstimator:
    def __init__(self,initial_orientation=None,camera_gain=.22,range_gain=.16,accel_gain=.008):
        self.orientation=initial_orientation or Rotation.identity(); self.gyro_bias=np.zeros(3); self.last_t=None; self.camera_gain=camera_gain; self.range_gain=range_gain; self.accel_gain=accel_gain; self.last_event={"kind":"init","residual_deg":0.,"correction_deg":0.}
    def propagate(self,imu):
        if self.last_t is not None:
            dt=imu.timestamp-self.last_t; self.orientation=self.orientation*Rotation.from_rotvec((imu.angular_velocity-self.gyro_bias)*dt)
            a=imu.linear_acceleration
            if np.linalg.norm(a)>1e-6:
                up=a/np.linalg.norm(a);corr=align_vector_correction(self.orientation.apply(up),np.array([0.,0.,1.]));self.orientation=blend_rotation(self.orientation,corr*self.orientation,self.accel_gain)
        self.last_t=imu.timestamp
    def update_camera_relative(self,relative_rotation,reference_orientation,tracks=0,track_rms_deg=float('nan')):
        target=reference_orientation*relative_rotation;imu_delta=reference_orientation.inv()*self.orientation
        residual=(target*self.orientation.inv()).magnitude();before=self.orientation
        count_quality=max(0.0,min(1.0,(tracks-3)/6.0));rms_quality=max(0.0,min(1.0,1.0-track_rms_deg/1.2)) if np.isfinite(track_rms_deg) else 0.0
        quality=count_quality*rms_quality;gain=self.camera_gain*quality
        self.orientation=blend_rotation(self.orientation,target,gain);correction=(self.orientation*before.inv()).magnitude()
        self.last_event={"kind":"camera_relative","residual_deg":float(np.degrees(residual)),"correction_deg":float(np.degrees(correction)),"tracks":int(tracks),"track_rms_deg":float(track_rms_deg),"visual_rotation_deg":float(np.degrees(relative_rotation.magnitude())),"imu_rotation_deg":float(np.degrees(imu_delta.magnitude())),"visual_quality":float(quality),"camera_gain_applied":float(gain)}
    def update_range_relative(self,relative_rotation,reference_orientation,pairs=0,range_rms_m=float('nan'),translation_m=float('nan')):
        target=reference_orientation*relative_rotation;imu_delta=reference_orientation.inv()*self.orientation
        residual=(target*self.orientation.inv()).magnitude();before=self.orientation
        count_quality=max(0.0,min(1.0,(pairs-7)/16.0));rms_quality=max(0.0,min(1.0,1.0-range_rms_m/.10)) if np.isfinite(range_rms_m) else 0.0
        translation_quality=max(0.0,min(1.0,1.0-translation_m/.18)) if np.isfinite(translation_m) else 0.0
        quality=count_quality*rms_quality*translation_quality;gain=self.range_gain*quality
        self.orientation=blend_rotation(self.orientation,target,gain);correction=(self.orientation*before.inv()).magnitude()
        self.last_event={"kind":"range_relative","residual_deg":float(np.degrees(residual)),"correction_deg":float(np.degrees(correction)),"pairs":int(pairs),"range_rms_m":float(range_rms_m),"range_translation_m":float(translation_m),"range_rotation_deg":float(np.degrees(relative_rotation.magnitude())),"imu_rotation_deg":float(np.degrees(imu_delta.magnitude())),"range_quality":float(quality),"range_gain_applied":float(gain)}
    def update_range_floor(self,normal_body):
        """Legacy Version 1 absolute-floor update, kept only for compatibility."""
        corr=align_vector_correction(self.orientation.apply(normal_body),np.array([0.,0.,1.]));residual=corr.magnitude();before=self.orientation;self.orientation=blend_rotation(self.orientation,corr*self.orientation,self.range_gain);correction=(self.orientation*before.inv()).magnitude();self.last_event={"kind":"range","residual_deg":float(np.degrees(residual)),"correction_deg":float(np.degrees(correction))}
    def state(self,t): return State(t,self.orientation,self.gyro_bias.copy())
