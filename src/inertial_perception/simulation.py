from __future__ import annotations
import base64
import json
from pathlib import Path
import numpy as np
from .world import truth_at,SCENE
from .sensors import ImuSimulator,CameraSimulator,GridRangeSimulator
from .frontend import visual_relative_rotation,range_floor_normal
from .estimator import AttitudeEstimator
from .math3d import rotation_error_deg


def _euler_deg(r): return r.as_euler('xyz',degrees=True).tolist()


def _range_to_camera_overlay(frame,cam,range_gt=None,camera_gt=None):
    out=[]
    for idx,r in enumerate(frame.rays):
        if not np.isfinite(r.distance) or r.confidence<=0:continue
        if range_gt is None or camera_gt is None:
            x,y,z=np.asarray(r.direction,float)*float(r.distance)
        else:
            pw=range_gt.position+range_gt.orientation.apply(np.asarray(r.direction,float)*float(r.distance))
            x,y,z=camera_gt.orientation.inv().apply(pw-camera_gt.position)
        if x<=.1:continue
        u=cam.cx-cam.fx*y/x;v=cam.cy-cam.fy*z/x
        if 0<=u<cam.width and 0<=v<cam.height:out.append([float(u),float(v),float(r.distance),idx])
    return out


def run_simulation(scenario='combined',duration=10.,seed=42,camera_enabled=True,range_enabled=True,imu_hz=200,camera_hz=15,range_hz=15):
    rng=np.random.default_rng(seed);imu=ImuSimulator(rng);cam=CameraSimulator(rng);rngs=GridRangeSimulator(rng)
    est=AttitudeEstimator(initial_orientation=truth_at(0,scenario).orientation);dt=1/imu_hz;next_cam=next_range=0.;records=[]
    last_cam=last_range=last_cam_gt=last_range_gt=None;prev_cam=None;prev_cam_reference=None;last_visual=None
    for k in range(int(round(duration*imu_hz))+1):
        t=k*dt;gt=truth_at(t,scenario);est.propagate(imu.sample(gt));event={"kind":"imu","residual_deg":0.,"correction_deg":0.}
        if camera_enabled and t+1e-9>=next_cam:
            last_cam_gt=gt;last_cam=cam.sample(gt);vis=visual_relative_rotation(prev_cam,last_cam,cam)
            if vis is not None and prev_cam_reference is not None:
                est.update_camera_relative(vis["rotation"],prev_cam_reference,vis["tracks"],vis["track_rms_deg"]);event=est.last_event.copy();last_visual=event.copy()
            prev_cam=last_cam;prev_cam_reference=est.orientation
            next_cam+=1/camera_hz
        if range_enabled and t+1e-9>=next_range:
            last_range_gt=gt;last_range=rngs.sample(gt);normal=range_floor_normal(last_range)
            if normal is not None:est.update_range_floor(normal);event=est.last_event.copy()
            next_range+=1/range_hz
        s=est.state(t);rec={"t":round(t,6),"true_euler":_euler_deg(gt.orientation),"est_euler":_euler_deg(s.orientation),"error_deg":rotation_error_deg(s.orientation,gt.orientation),"event":event}
        if last_visual is not None:rec["visual_frontend"]=last_visual.copy()
        if k%max(1,int(imu_hz/20))==0:
            if last_cam is not None:
                rec["camera_timestamp"]=float(last_cam.timestamp);rec["camera_pose_euler"]=_euler_deg(last_cam_gt.orientation);rec["camera_render_size"]=[last_cam.render_width,last_cam.render_height]
                rec["camera_rgb_b64"]=base64.b64encode(last_cam.image_rgb.tobytes()).decode('ascii');rec["camera_features"]=[[f.u,f.v,f.feature_id,f.confidence] for f in last_cam.features]
            if last_range is not None:
                rec["range_timestamp"]=float(last_range.timestamp);rec["range_distances"]=[None if not np.isfinite(r.distance) else r.distance for r in last_range.rays]
                rec["range_world_points"]=[(last_range_gt.position+last_range_gt.orientation.apply(r.direction*r.distance)).tolist() for r in last_range.rays if np.isfinite(r.distance)]
                if last_cam is not None:
                    rec["range_camera_overlay"]=_range_to_camera_overlay(last_range,cam,last_range_gt,last_cam_gt);rec["range_camera_dt_ms"]=float((last_range.timestamp-last_cam.timestamp)*1000.)
        records.append(rec)
    errors=np.array([r['error_deg'] for r in records]);metrics={"scenario":scenario,"duration_s":duration,"seed":seed,"camera_enabled":camera_enabled,"range_enabled":range_enabled,"orientation_rmse_deg":float(np.sqrt(np.mean(errors**2))),"orientation_mae_deg":float(np.mean(np.abs(errors))),"orientation_max_deg":float(np.max(errors))}
    return {"meta":{"imu_hz":imu_hz,"camera_hz":camera_hz,"range_hz":range_hz,"camera":{"width":cam.width,"height":cam.height,"fov_deg":cam.fov_deg,"render_width":cam.render_width,"render_height":cam.render_height,"feature_detector":"Harris","constraint":"relative_rotation_from_image_tracks"},"range":{"rows":rngs.rows,"cols":rngs.cols,"tilt_down_deg":rngs.tilt_down_deg},"scene":SCENE},"metrics":metrics,"records":records}


def compare_modes(scenario='combined',duration=10.,seed=42):
    modes={"imu_only":(False,False),"imu_camera":(True,False),"imu_range":(False,True),"all":(True,True)}
    return {name:run_simulation(scenario,duration,seed,c,r) for name,(c,r) in modes.items()}


def export_demo(path,scenario='combined',duration=10.,seed=42):
    out=compare_modes(scenario,duration,seed);Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(out,separators=(',',':')),encoding='utf-8');return out
