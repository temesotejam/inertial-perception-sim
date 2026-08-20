from __future__ import annotations
import numpy as np
from .types import ImuSample, CameraFrame, CameraFeature, RangeFrame, RayMeasurement, GroundTruthState
from .world import LANDMARKS, ray_scene_distance

class ImuSimulator:
    def __init__(self,rng,gyro_bias=(.002,-.0015,.006),gyro_noise_std=.0015,accel_noise_std=.025):
        self.rng=rng; self.bias=np.asarray(gyro_bias,float); self.gyro_noise_std=gyro_noise_std; self.accel_noise_std=accel_noise_std
    def sample(self,gt:GroundTruthState):
        gyro=gt.angular_velocity+self.bias+self.rng.normal(0,self.gyro_noise_std,3)
        g=np.array([0.,0.,-9.80665]); acc=gt.orientation.inv().apply(gt.acceleration-g)+self.rng.normal(0,self.accel_noise_std,3)
        return ImuSample(gt.timestamp,gyro,acc)

class CameraSimulator:
    """Generic monocular pinhole camera. Body +X is optical forward, +Y left, +Z up."""
    def __init__(self,rng,width=640,height=480,fov_deg=70.,pixel_noise_std=1.2,dropout=.03):
        self.rng=rng; self.width=width; self.height=height; self.fx=width/(2*np.tan(np.radians(fov_deg)/2)); self.fy=self.fx; self.cx=width/2; self.cy=height/2; self.pixel_noise_std=pixel_noise_std; self.dropout=dropout
    def sample(self,gt):
        feats=[]
        for i,pw in enumerate(LANDMARKS):
            x,y,z=gt.orientation.inv().apply(pw-gt.position)
            if x<=.1: continue
            u=self.cx-self.fx*y/x; v=self.cy-self.fy*z/x
            if 0<=u<self.width and 0<=v<self.height and self.rng.random()>=self.dropout:
                feats.append(CameraFeature(i,float(u+self.rng.normal(0,self.pixel_noise_std)),float(v+self.rng.normal(0,self.pixel_noise_std)),1.0))
        return CameraFrame(gt.timestamp,feats)
    def bearing_from_pixel(self,u,v):
        b=np.array([1.,-(u-self.cx)/self.fx,-(v-self.cy)/self.fy]); return b/np.linalg.norm(b)

class GridRangeSimulator:
    """Generic grid range sensor. Default optical axis points body -Z into the shared 3D scene."""
    def __init__(self,rng,rows=8,cols=8,fov_x_deg=45,fov_y_deg=45,noise_std=.008,max_range=4.,dropout=.02):
        self.rng=rng; self.rows=rows; self.cols=cols; self.noise_std=noise_std; self.max_range=max_range; self.dropout=dropout; self.dirs=[]
        for py in np.linspace(-np.radians(fov_y_deg)/2,np.radians(fov_y_deg)/2,rows):
            for px in np.linspace(-np.radians(fov_x_deg)/2,np.radians(fov_x_deg)/2,cols):
                d=np.array([np.tan(py),np.tan(px),-1.]); d/=np.linalg.norm(d); self.dirs.append(d)
    def sample(self,gt):
        rays=[]; origin=gt.position
        for db in self.dirs:
            if self.rng.random()<self.dropout:
                rays.append(RayMeasurement(db.copy(),float('nan'),0.)); continue
            dw=gt.orientation.apply(db)
            dist,_=ray_scene_distance(origin,dw,self.max_range)
            if dist is None:
                rays.append(RayMeasurement(db.copy(),float('nan'),0.)); continue
            measured=max(0,float(dist)+self.rng.normal(0,self.noise_std))
            rays.append(RayMeasurement(db.copy(),measured,1.))
        return RangeFrame(gt.timestamp,rays,self.rows,self.cols)
