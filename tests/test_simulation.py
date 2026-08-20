import numpy as np
from inertial_perception.simulation import compare_modes,run_simulation

def test_all_modes_are_finite():
    out=compare_modes('combined',duration=3.,seed=7)
    for data in out.values(): assert np.isfinite(data['metrics']['orientation_rmse_deg']) and data['metrics']['orientation_rmse_deg']<20

def test_camera_and_range_improve_combined_motion():
    out=compare_modes('combined',duration=8.,seed=42); imu=out['imu_only']['metrics']['orientation_rmse_deg']; allm=out['all']['metrics']['orientation_rmse_deg']; assert allm<imu*.60 and allm<1.5

def test_range_improves_roll_motion():
    imu=run_simulation('roll',6.,3,False,False)['metrics']['orientation_rmse_deg']; rng=run_simulation('roll',6.,3,False,True)['metrics']['orientation_rmse_deg']; assert rng<imu

def test_camera_improves_yaw_motion():
    imu=run_simulation('yaw',6.,9,False,False)['metrics']['orientation_rmse_deg']; cam=run_simulation('yaw',6.,9,True,False)['metrics']['orientation_rmse_deg']; assert cam<imu*.5
