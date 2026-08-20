from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .types import State
from .math3d import align_vector_correction

G_WORLD=np.array([0.,0.,-9.80665])

def _skew(v):
    x,y,z=np.asarray(v,float);return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)

class InertialESKF:
    """15-state inertial navigation ESKF.

    Nominal state: p, v, q, b_g, b_a.
    Error state: [dp, dv, dtheta, db_g, db_a].
    Camera/Range still constrain attitude only in Version 5. Position and
    velocity are propagated from IMU and deliberately left uncorrected until a
    later translational frontend is added.
    """
    def __init__(self,initial_position=None,initial_velocity=None,initial_orientation=None,
                 gyro_noise_std=.0015,accel_noise_std=.025,bias_rw_std=2.5e-4,
                 accel_bias_rw_std=2.0e-3,accel_gravity_noise_deg=3.0,
                 camera_noise_deg=.75,range_noise_deg=1.8,gravity_update=False):
        self.position=np.zeros(3) if initial_position is None else np.asarray(initial_position,float).copy()
        self.velocity=np.zeros(3) if initial_velocity is None else np.asarray(initial_velocity,float).copy()
        self.orientation=initial_orientation or Rotation.identity();self.gyro_bias=np.zeros(3);self.accel_bias=np.zeros(3);self.last_t=None
        sig=np.array([.05,.05,.05,.10,.10,.10,*np.radians([2.,2.,4.]),*np.radians([.35,.35,.6]),.08,.08,.08]);self.P=np.diag(sig**2)
        self.gyro_noise_std=float(gyro_noise_std);self.accel_noise_std=float(accel_noise_std);self.bias_rw_std=float(bias_rw_std);self.accel_bias_rw_std=float(accel_bias_rw_std)
        self.accel_gravity_noise=np.radians(accel_gravity_noise_deg);self.camera_noise=np.radians(camera_noise_deg);self.range_noise=np.radians(range_noise_deg);self.gravity_update=bool(gravity_update)
        self.last_event={"kind":"init","residual_deg":0.,"correction_deg":0.,"filter":"ins_eskf"}

    def _inject(self,dx):
        self.position+=dx[:3];self.velocity+=dx[3:6];self.orientation=Rotation.from_rotvec(dx[6:9])*self.orientation;self.gyro_bias+=dx[9:12];self.accel_bias+=dx[12:15]

    def _update(self,residual,H,R_cov,kind,extra=None):
        residual=np.asarray(residual,float);S=H@self.P@H.T+R_cov;K=self.P@H.T@np.linalg.inv(S);dx=K@residual;before=self.orientation;self._inject(dx)
        I=np.eye(15);A=I-K@H;self.P=A@self.P@A.T+K@R_cov@K.T;self.P=(self.P+self.P.T)*.5;correction=(self.orientation*before.inv()).magnitude()
        ev={"kind":kind,"residual_deg":float(np.degrees(np.linalg.norm(residual))),"correction_deg":float(np.degrees(correction)),"filter":"ins_eskf","bias_norm_dps":float(np.degrees(np.linalg.norm(self.gyro_bias))),"accel_bias_norm":float(np.linalg.norm(self.accel_bias)),"sigma_att_deg":float(np.degrees(np.sqrt(np.mean(np.diag(self.P)[6:9])))),"sigma_pos_m":float(np.sqrt(np.mean(np.diag(self.P)[:3]))),"sigma_vel_mps":float(np.sqrt(np.mean(np.diag(self.P)[3:6])))}
        if extra:ev.update(extra)
        self.last_event=ev

    def propagate(self,imu):
        if self.last_t is not None:
            dt=max(0.,float(imu.timestamp-self.last_t));omega=np.asarray(imu.angular_velocity,float)-self.gyro_bias;specific=np.asarray(imu.linear_acceleration,float)-self.accel_bias;Rwb=self.orientation.as_matrix();a_world=Rwb@specific+G_WORLD
            self.position=self.position+self.velocity*dt+.5*a_world*dt*dt;self.velocity=self.velocity+a_world*dt;self.orientation=self.orientation*Rotation.from_rotvec(omega*dt)
            Fc=np.zeros((15,15));Fc[:3,3:6]=np.eye(3);Fc[3:6,6:9]=-Rwb@_skew(specific);Fc[3:6,12:15]=-Rwb;Fc[6:9,6:9]=-_skew(omega);Fc[6:9,9:12]=-np.eye(3);F=np.eye(15)+Fc*dt
            Q=np.zeros((15,15));Q[3:6,3:6]=np.eye(3)*(self.accel_noise_std**2)*dt;Q[6:9,6:9]=np.eye(3)*(self.gyro_noise_std**2)*dt;Q[9:12,9:12]=np.eye(3)*(self.bias_rw_std**2)*dt;Q[12:15,12:15]=np.eye(3)*(self.accel_bias_rw_std**2)*dt
            self.P=F@self.P@F.T+Q;self.P=(self.P+self.P.T)*.5
        if self.gravity_update:
            a=np.asarray(imu.linear_acceleration,float)-self.accel_bias;norm=float(np.linalg.norm(a))
            if norm>1e-6 and abs(norm-9.80665)<.40:
                up=a/norm;corr=align_vector_correction(self.orientation.apply(up),np.array([0.,0.,1.]));rv=corr.as_rotvec()
                if np.linalg.norm(rv)<np.radians(12):
                    axis_world=self.orientation.apply(up);H=np.zeros((3,15));H[:,6:9]=np.eye(3)-np.outer(axis_world,axis_world);self._update(rv,H,np.eye(3)*self.accel_gravity_noise**2,"accel")
        self.last_t=imu.timestamp

    def update_camera_relative(self,relative_rotation,reference_orientation,tracks=0,track_rms_deg=float('nan')):
        target=reference_orientation*relative_rotation;residual=(target*self.orientation.inv()).as_rotvec();count_quality=max(.05,min(1.,(tracks-3)/6.));rms_quality=max(.05,min(1.,1.-track_rms_deg/1.5)) if np.isfinite(track_rms_deg) else .05;quality=count_quality*rms_quality;sigma=self.camera_noise/max(np.sqrt(quality),.15);H=np.zeros((3,15));H[:,6:9]=np.eye(3);imu_delta=reference_orientation.inv()*self.orientation
        extra={"tracks":int(tracks),"track_rms_deg":float(track_rms_deg),"visual_rotation_deg":float(np.degrees(relative_rotation.magnitude())),"imu_rotation_deg":float(np.degrees(imu_delta.magnitude())),"visual_quality":float(quality),"measurement_sigma_deg":float(np.degrees(sigma))};self._update(residual,H,np.eye(3)*sigma**2,"camera_relative",extra)

    def update_range_relative(self,previous_normal,current_normal,reference_orientation,pairs=0,range_rms_m=float('nan'),translation_m=float('nan'),range_rotation_deg=float('nan')):
        desired_world=reference_orientation.apply(np.asarray(previous_normal,float));current_world=self.orientation.apply(np.asarray(current_normal,float));corr=align_vector_correction(current_world,desired_world);residual=corr.as_rotvec();count_quality=max(.05,min(1.,(pairs-7)/16.));rms_quality=max(.05,min(1.,1.-range_rms_m/.12)) if np.isfinite(range_rms_m) else .05;translation_quality=max(.05,min(1.,1.-translation_m/.22)) if np.isfinite(translation_m) else .05;quality=count_quality*rms_quality*translation_quality;sigma=self.range_noise/max(np.sqrt(quality),.18);n=current_world/np.linalg.norm(current_world);H=np.zeros((3,15));H[:,6:9]=np.eye(3)-np.outer(n,n);imu_delta=reference_orientation.inv()*self.orientation
        extra={"pairs":int(pairs),"range_rms_m":float(range_rms_m),"range_translation_m":float(translation_m),"range_rotation_deg":float(range_rotation_deg),"imu_rotation_deg":float(np.degrees(imu_delta.magnitude())),"range_quality":float(quality),"measurement_sigma_deg":float(np.degrees(sigma))};self._update(residual,H,np.eye(3)*sigma**2,"range_relative",extra)

    def state(self,t):return State(t,self.orientation,self.gyro_bias.copy(),self.position.copy(),self.velocity.copy(),self.accel_bias.copy())
