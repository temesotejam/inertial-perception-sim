from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .types import State
from .math3d import align_vector_correction


def _skew(v):
    x,y,z=np.asarray(v,float);return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)


class AttitudeESKF:
    """Minimal 6-state error-state Kalman filter for attitude and gyro bias.

    Nominal state: orientation q and gyro bias b_g.
    Error state: [dtheta, db_g]. Camera supplies a full 3-D relative attitude
    residual; Range supplies only the observable tilt residual from local
    surface-normal consistency. The filter never consumes renderer/world GT.
    """
    def __init__(self,initial_orientation=None,gyro_noise_std=.0015,bias_rw_std=2.5e-4,
                 accel_noise_deg=2.5,camera_noise_deg=.75,range_noise_deg=1.8):
        self.orientation=initial_orientation or Rotation.identity();self.gyro_bias=np.zeros(3);self.last_t=None
        self.P=np.diag(np.radians([2.,2.,4.,.35,.35,.6])**2)
        self.gyro_noise_std=float(gyro_noise_std);self.bias_rw_std=float(bias_rw_std)
        self.accel_noise=np.radians(accel_noise_deg);self.camera_noise=np.radians(camera_noise_deg);self.range_noise=np.radians(range_noise_deg)
        self.last_event={"kind":"init","residual_deg":0.,"correction_deg":0.,"filter":"eskf"}

    def _inject(self,dx):
        self.orientation=Rotation.from_rotvec(dx[:3])*self.orientation
        self.gyro_bias=self.gyro_bias+dx[3:]

    def _update_rotvec(self,residual,R_cov,kind,extra=None,H=None):
        residual=np.asarray(residual,float)
        if H is None:H=np.hstack((np.eye(3),np.zeros((3,3))))
        S=H@self.P@H.T+R_cov;K=self.P@H.T@np.linalg.inv(S);dx=K@residual
        before=self.orientation;self._inject(dx)
        I=np.eye(6);A=I-K@H;self.P=A@self.P@A.T+K@R_cov@K.T
        self.P=(self.P+self.P.T)*.5
        correction=(self.orientation*before.inv()).magnitude()
        ev={"kind":kind,"residual_deg":float(np.degrees(np.linalg.norm(residual))),"correction_deg":float(np.degrees(correction)),"filter":"eskf","bias_norm_dps":float(np.degrees(np.linalg.norm(self.gyro_bias))),"sigma_att_deg":float(np.degrees(np.sqrt(np.mean(np.diag(self.P)[:3]))))}
        if extra:ev.update(extra)
        self.last_event=ev

    def propagate(self,imu):
        if self.last_t is not None:
            dt=max(0.,float(imu.timestamp-self.last_t));omega=np.asarray(imu.angular_velocity,float)-self.gyro_bias
            self.orientation=self.orientation*Rotation.from_rotvec(omega*dt)
            F=np.eye(6);F[:3,:3]-=_skew(omega)*dt;F[:3,3:]=-np.eye(3)*dt
            Q=np.zeros((6,6));Q[:3,:3]=np.eye(3)*(self.gyro_noise_std**2)*dt;Q[3:,3:]=np.eye(3)*(self.bias_rw_std**2)*dt
            self.P=F@self.P@F.T+Q;self.P=(self.P+self.P.T)*.5
        a=np.asarray(imu.linear_acceleration,float)
        if np.linalg.norm(a)>1e-6:
            up=a/np.linalg.norm(a);corr=align_vector_correction(self.orientation.apply(up),np.array([0.,0.,1.]))
            rv=corr.as_rotvec()
            if np.linalg.norm(rv)<np.radians(12):
                axis_world=self.orientation.apply(up);Htheta=np.eye(3)-np.outer(axis_world,axis_world);H=np.hstack((Htheta,np.zeros((3,3))))
                self._update_rotvec(rv,np.eye(3)*self.accel_noise**2,"accel",H=H)
        self.last_t=imu.timestamp

    def update_camera_relative(self,relative_rotation,reference_orientation,tracks=0,track_rms_deg=float('nan')):
        target=reference_orientation*relative_rotation;residual=(target*self.orientation.inv()).as_rotvec()
        count_quality=max(0.05,min(1.0,(tracks-3)/6.0));rms_quality=max(.05,min(1.0,1.0-track_rms_deg/1.5)) if np.isfinite(track_rms_deg) else .05
        quality=count_quality*rms_quality;sigma=self.camera_noise/max(np.sqrt(quality),.15)
        imu_delta=reference_orientation.inv()*self.orientation
        extra={"tracks":int(tracks),"track_rms_deg":float(track_rms_deg),"visual_rotation_deg":float(np.degrees(relative_rotation.magnitude())),"imu_rotation_deg":float(np.degrees(imu_delta.magnitude())),"visual_quality":float(quality),"measurement_sigma_deg":float(np.degrees(sigma))}
        self._update_rotvec(residual,np.eye(3)*sigma**2,"camera_relative",extra)

    def update_range_relative(self,previous_normal,current_normal,reference_orientation,pairs=0,range_rms_m=float('nan'),translation_m=float('nan'),range_rotation_deg=float('nan')):
        desired_world=reference_orientation.apply(np.asarray(previous_normal,float));current_world=self.orientation.apply(np.asarray(current_normal,float))
        corr=align_vector_correction(current_world,desired_world);residual=corr.as_rotvec()
        count_quality=max(.05,min(1.0,(pairs-7)/16.0));rms_quality=max(.05,min(1.0,1.0-range_rms_m/.12)) if np.isfinite(range_rms_m) else .05
        translation_quality=max(.05,min(1.0,1.0-translation_m/.22)) if np.isfinite(translation_m) else .05
        quality=count_quality*rms_quality*translation_quality;sigma=self.range_noise/max(np.sqrt(quality),.18)
        n=current_world/np.linalg.norm(current_world);Htheta=np.eye(3)-np.outer(n,n);H=np.hstack((Htheta,np.zeros((3,3))))
        imu_delta=reference_orientation.inv()*self.orientation
        extra={"pairs":int(pairs),"range_rms_m":float(range_rms_m),"range_translation_m":float(translation_m),"range_rotation_deg":float(range_rotation_deg),"imu_rotation_deg":float(np.degrees(imu_delta.magnitude())),"range_quality":float(quality),"measurement_sigma_deg":float(np.degrees(sigma))}
        self._update_rotvec(residual,np.eye(3)*sigma**2,"range_relative",extra,H=H)

    def state(self,t):return State(t,self.orientation,self.gyro_bias.copy())
