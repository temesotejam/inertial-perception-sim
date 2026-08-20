from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .world import truth_at,SCENE
from .sensors import ImuSimulator,CameraSimulator,GridRangeSimulator
from .frontend import visual_orientation,range_floor_normal
from .estimator import AttitudeEstimator
from .math3d import rotation_error_deg

def _euler_deg(r): return r.as_euler('xyz',degrees=True).tolist()

def run_simulation(scenario='combined',duration=10.,seed=42,camera_enabled=True,range_enabled=True,imu_hz=200,camera_hz=30,range_hz=15):
    rng=np.random.default_rng(seed); imu=ImuSimulator(rng); cam=CameraSimulator(rng); rngs=GridRangeSimulator(rng); est=AttitudeEstimator(initial_orientation=truth_at(0,scenario).orientation); dt=1/imu_hz; next_cam=next_range=0.; records=[]; last_cam=last_range=None
    for k in range(int(round(duration*imu_hz))+1):
        t=k*dt; gt=truth_at(t,scenario); est.propagate(imu.sample(gt)); event={"kind":"imu","residual_deg":0.,"correction_deg":0.}
        if camera_enabled and t+1e-9>=next_cam:
            last_cam=cam.sample(gt); obs=visual_orientation(last_cam,cam,gt.position)
            if obs is not None: est.update_camera(obs); event=est.last_event.copy()
            next_cam+=1/camera_hz
        if range_enabled and t+1e-9>=next_range:
            last_range=rngs.sample(gt); normal=range_floor_normal(last_range)
            if normal is not None: est.update_range_floor(normal); event=est.last_event.copy()
            next_range+=1/range_hz
        s=est.state(t); rec={"t":round(t,6),"true_euler":_euler_deg(gt.orientation),"est_euler":_euler_deg(s.orientation),"error_deg":rotation_error_deg(s.orientation,gt.orientation),"event":event}
        if k%max(1,int(imu_hz/20))==0:
            if last_cam is not None: rec["camera_features"]=[[f.u,f.v,f.feature_id] for f in last_cam.features]
            if last_range is not None:
                rec["range_distances"]=[None if not np.isfinite(r.distance) else r.distance for r in last_range.rays]; pts=[]
                for r in last_range.rays:
                    if np.isfinite(r.distance): pts.append((s.orientation.apply(r.direction*r.distance)+gt.position).tolist())
                rec["range_world_points"]=pts
        records.append(rec)
    errors=np.array([r['error_deg'] for r in records]); metrics={"scenario":scenario,"duration_s":duration,"seed":seed,"camera_enabled":camera_enabled,"range_enabled":range_enabled,"orientation_rmse_deg":float(np.sqrt(np.mean(errors**2))),"orientation_mae_deg":float(np.mean(np.abs(errors))),"orientation_max_deg":float(np.max(errors))}
    return {"meta":{"imu_hz":imu_hz,"camera_hz":camera_hz,"range_hz":range_hz,"camera":{"width":cam.width,"height":cam.height,"fov_deg":70.0},"range":{"rows":rngs.rows,"cols":rngs.cols},"scene":SCENE},"metrics":metrics,"records":records}

def compare_modes(scenario='combined',duration=10.,seed=42):
    modes={"imu_only":(False,False),"imu_camera":(True,False),"imu_range":(False,True),"all":(True,True)}
    return {name:run_simulation(scenario,duration,seed,c,r) for name,(c,r) in modes.items()}

def export_demo(path,scenario='combined',duration=10.,seed=42):
    out=compare_modes(scenario,duration,seed); Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(out,separators=(',',':')),encoding='utf-8'); return out
