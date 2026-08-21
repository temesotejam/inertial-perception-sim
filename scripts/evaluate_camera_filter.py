from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inertial_perception.simulation import run_simulation


def wrap_deg(x):return (np.asarray(x,float)+180.)%360.-180.

def summarize(camera,range_enabled=False):
    out=run_simulation('translation',4.,21,camera,range_enabled,estimator_kind='ins_eskf')
    rec=out['records'];true=np.array([r['true_euler'] for r in rec]);est=np.array([r['est_euler'] for r in rec]);de=wrap_deg(est-true)
    axis=np.sqrt(np.mean(de**2,axis=0));vis=[r.get('visual_frontend') for r in rec if r.get('visual_frontend')]
    prior=np.array([v.get('prior_residual_rms_deg',np.nan) for v in vis],float);sig=np.array([v.get('measurement_sigma_deg',np.nan) for v in vis],float)
    return {'metrics':out['metrics'],'euler_axis_rmse_deg':axis.tolist(),'mean_prior_residual_deg':float(np.nanmean(prior)) if len(prior) else float('nan'),'mean_camera_sigma_deg':float(np.nanmean(sig)) if len(sig) else float('nan'),'final_gyro_bias_dps':out['metrics']['final_gyro_bias_dps'],'final_accel_bias':out['metrics']['final_accel_bias']}

if __name__=='__main__':
    result={'imu_only':summarize(False,False),'imu_camera':summarize(True,False),'all':summarize(True,True)}
    Path('web/data/camera_filter_eval.json').write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
    for name,d in result.items():
        m=d['metrics'];print(f"{name:10s} pos={m['position_rmse_m']:.4f}m vel={m['velocity_rmse_mps']:.4f}m/s ori={m['orientation_rmse_deg']:.4f}deg axis={d['euler_axis_rmse_deg']} bg={d['final_gyro_bias_dps']} prior={d['mean_prior_residual_deg']:.4f}deg sigma={d['mean_camera_sigma_deg']:.4f}deg")
