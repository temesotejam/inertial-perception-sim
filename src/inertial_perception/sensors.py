from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .types import ImuSample, CameraFrame, CameraFeature, RangeFrame, RayMeasurement, GroundTruthState
from .world import ray_scene_distance
from .camera_render import render_camera, detect_harris_features

class ImuSimulator:
    def __init__(self,rng,gyro_bias=(.002,-.0015,.006),gyro_noise_std=.0015,accel_noise_std=.025):
        self.rng=rng; self.bias=np.asarray(gyro_bias,float); self.gyro_noise_std=gyro_noise_std; self.accel_noise_std=accel_noise_std
    def sample(self,gt:GroundTruthState):
        gyro=gt.angular_velocity+self.bias+self.rng.normal(0,self.gyro_noise_std,3)
        g=np.array([0.,0.,-9.80665]); acc=gt.orientation.inv().apply(gt.acceleration-g)+self.rng.normal(0,self.accel_noise_std,3)
        return ImuSample(gt.timestamp,gyro,acc)

class CameraSimulator:
    """Generic monocular camera with a real synthetic-image frontend."""
    def __init__(self,rng,width=640,height=480,fov_deg=70.,pixel_noise_std=.25,dropout=.03,render_width=80,render_height=60,max_features=32):
        self.rng=rng; self.width=width; self.height=height; self.fov_deg=fov_deg
        self.fx=width/(2*np.tan(np.radians(fov_deg)/2)); self.fy=self.fx; self.cx=width/2; self.cy=height/2
        self.pixel_noise_std=pixel_noise_std; self.dropout=dropout; self.render_width=render_width; self.render_height=render_height; self.max_features=max_features
        self._next_id=0; self._previous=[]
    def _assign_id(self,world_position):
        best=None; best_d=.22
        for f in self._previous:
            if f.world_position is None: continue
            d=float(np.linalg.norm(f.world_position-world_position))
            if d<best_d:best=f.feature_id;best_d=d
        if best is None:best=self._next_id;self._next_id+=1
        return int(best)
    def sample(self,gt):
        rgb,depth,world=render_camera(gt,self.render_width,self.render_height,self.fov_deg)
        detected=detect_harris_features(rgb,depth,world,self.max_features);feats=[];sx=self.width/self.render_width;sy=self.height/self.render_height
        for x,y,score,pw in detected:
            if self.rng.random()<self.dropout:continue
            u=(x+.5)*sx+self.rng.normal(0,self.pixel_noise_std);v=(y+.5)*sy+self.rng.normal(0,self.pixel_noise_std)
            feats.append(CameraFeature(self._assign_id(pw),float(u),float(v),float(max(score,0.)),pw))
        self._previous=feats
        return CameraFrame(gt.timestamp,feats,rgb,self.render_width,self.render_height)
    def bearing_from_pixel(self,u,v):
        b=np.array([1.,-(u-self.cx)/self.fx,-(v-self.cy)/self.fy]);return b/np.linalg.norm(b)

class GridRangeSimulator:
    def __init__(self,rng,rows=8,cols=8,fov_x_deg=45,fov_y_deg=45,tilt_down_deg=35.,noise_std=.008,max_range=4.,dropout=.02):
        self.rng=rng; self.rows=rows; self.cols=cols; self.noise_std=noise_std; self.max_range=max_range; self.dropout=dropout; self.tilt_down_deg=tilt_down_deg; self.dirs=[]
        mount=Rotation.from_euler('y',np.radians(tilt_down_deg))
        for py in np.linspace(-np.radians(fov_y_deg)/2,np.radians(fov_y_deg)/2,rows):
            for px in np.linspace(-np.radians(fov_x_deg)/2,np.radians(fov_x_deg)/2,cols):
                d=np.array([1.,np.tan(px),np.tan(py)]);d/=np.linalg.norm(d);d=mount.apply(d);d/=np.linalg.norm(d);self.dirs.append(d)
    def sample(self,gt):
        rays=[];origin=gt.position
        for db in self.dirs:
            if self.rng.random()<self.dropout:rays.append(RayMeasurement(db.copy(),float('nan'),0.));continue
            dw=gt.orientation.apply(db);dist,_=ray_scene_distance(origin,dw,self.max_range)
            if dist is None:rays.append(RayMeasurement(db.copy(),float('nan'),0.));continue
            rays.append(RayMeasurement(db.copy(),max(0,float(dist)+self.rng.normal(0,self.noise_std)),1.))
        return RangeFrame(gt.timestamp,rays,self.rows,self.cols)
