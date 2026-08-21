from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from inertial_perception.world import truth_at
from inertial_perception.sensors import CameraSimulator
from inertial_perception.frontend import visual_relative_rotation
from inertial_perception.math3d import rotation_error_deg


def evaluate(duration=4.0,seed=21,camera_hz=15):
    rng=np.random.default_rng(seed);cam=CameraSimulator(rng);prev=None;prev_gt=None;rows=[]
    for t in np.arange(0,duration+1e-9,1/camera_hz):
        gt=truth_at(float(t),'translation');frame=cam.sample(gt)
        if prev is not None:
            vis=visual_relative_rotation(prev,frame,cam)
            if vis is not None:
                expected=prev_gt.orientation.inv()*gt.orientation
                err=rotation_error_deg(vis['rotation'],expected)
                rv=(vis['rotation']*expected.inv()).as_rotvec()
                rows.append({'t':float(t),'rotation_error_deg':float(err),'error_rotvec_deg':np.degrees(rv).tolist(),'tracks':int(vis['tracks']),'track_rms_deg':float(vis['track_rms_deg'])})
        prev=frame;prev_gt=gt
    e=np.array([r['rotation_error_deg'] for r in rows]);rv=np.array([r['error_rotvec_deg'] for r in rows]) if rows else np.zeros((0,3))
    summary={'samples':len(rows),'rotation_rmse_deg':float(np.sqrt(np.mean(e**2))) if len(e) else float('nan'),'rotation_p95_deg':float(np.percentile(e,95)) if len(e) else float('nan'),'axis_rmse_deg':np.sqrt(np.mean(rv**2,axis=0)).tolist() if len(rv) else [float('nan')]*3,'mean_tracks':float(np.mean([r['tracks'] for r in rows])) if rows else 0.0,'mean_track_rms_deg':float(np.mean([r['track_rms_deg'] for r in rows])) if rows else float('nan')}
    return {'summary':summary,'rows':rows}

if __name__=='__main__':
    out=evaluate();Path('web/data/camera_translation_eval.json').write_text(json.dumps(out,separators=(',',':')),encoding='utf-8')
    s=out['summary'];print(f"Camera translation visual ΔR: RMSE={s['rotation_rmse_deg']:.4f} deg p95={s['rotation_p95_deg']:.4f} deg axis_RMSE={s['axis_rmse_deg']} tracks={s['mean_tracks']:.2f} track_RMS={s['mean_track_rms_deg']:.4f} deg")
