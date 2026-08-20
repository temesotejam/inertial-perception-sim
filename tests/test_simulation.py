import numpy as np
from inertial_perception.simulation import compare_modes,run_simulation


def test_all_modes_are_finite():
    out=compare_modes('combined',duration=3.,seed=7)
    for data in out.values():assert np.isfinite(data['metrics']['orientation_rmse_deg']) and data['metrics']['orientation_rmse_deg']<20


def test_relative_camera_reduces_yaw_drift():
    imu=run_simulation('yaw',6.,9,False,False)['metrics']['orientation_rmse_deg'];cam=run_simulation('yaw',6.,9,True,False)['metrics']['orientation_rmse_deg']
    assert cam<imu*.98


def test_range_improves_roll_motion():
    imu=run_simulation('roll',6.,3,False,False)['metrics']['orientation_rmse_deg'];rng=run_simulation('roll',6.,3,False,True)['metrics']['orientation_rmse_deg'];assert rng<imu


def test_combined_relative_fusion_remains_bounded_without_world_map():
    out=compare_modes('combined',duration=8.,seed=42);imu=out['imu_only']['metrics']['orientation_rmse_deg'];allm=out['all']['metrics']['orientation_rmse_deg']
    # Version 2 no longer receives absolute world landmark orientation. Relative
    # Camera constraints should remain stable rather than reproducing the much
    # stronger Version 1 mapped-landmark correction.
    assert allm<imu*1.15 and allm<2.0
