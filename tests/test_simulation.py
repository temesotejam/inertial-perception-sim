import numpy as np
from inertial_perception.simulation import compare_modes,run_simulation,compare_estimators,_sensor_rngs
from inertial_perception.sensors import ImuSimulator,CameraSimulator,GridRangeSimulator
from inertial_perception.world import truth_at


def test_all_modes_are_finite_with_eskf():
    out=compare_modes('combined',duration=3.,seed=7,estimator_kind='eskf')
    for data in out.values():
        assert data['metrics']['estimator_kind']=='eskf'
        assert np.isfinite(data['metrics']['orientation_rmse_deg']) and data['metrics']['orientation_rmse_deg']<20
        assert np.all(np.isfinite(data['metrics']['final_gyro_bias_dps']))


def test_relative_camera_reduces_yaw_drift_with_eskf():
    imu=run_simulation('yaw',6.,9,False,False,estimator_kind='eskf')['metrics']['orientation_rmse_deg']
    cam=run_simulation('yaw',6.,9,True,False,estimator_kind='eskf')['metrics']['orientation_rmse_deg']
    assert cam<imu


def test_range_improves_roll_motion_with_eskf():
    imu=run_simulation('roll',6.,3,False,False,estimator_kind='eskf')['metrics']['orientation_rmse_deg']
    rng=run_simulation('roll',6.,3,False,True,estimator_kind='eskf')['metrics']['orientation_rmse_deg']
    assert rng<imu


def test_combined_eskf_remains_bounded_without_world_map():
    out=compare_modes('combined',duration=8.,seed=42,estimator_kind='eskf')
    imu=out['imu_only']['metrics']['orientation_rmse_deg'];allm=out['all']['metrics']['orientation_rmse_deg']
    assert allm<3.0 and allm<imu*1.5


def test_legacy_estimators_remain_available_for_regression_comparison():
    out=compare_estimators('combined',duration=2.,seed=11)
    assert set(out)=={'blend','attitude_eskf','ins_eskf'}
    for kind in out:assert np.isfinite(out[kind]['all']['metrics']['orientation_rmse_deg'])


def test_sensor_rng_streams_are_independent_of_camera_workload():
    a_imu,a_cam,a_range=_sensor_rngs(17);b_imu,b_cam,b_range=_sensor_rngs(17)
    imu_a,imu_b=ImuSimulator(a_imu),ImuSimulator(b_imu);cam_a=CameraSimulator(a_cam);range_a,range_b=GridRangeSimulator(a_range),GridRangeSimulator(b_range)
    gt=truth_at(.7,'translation')
    # Consume a large amount of Camera randomness in only one copy.
    for _ in range(8):cam_a.sample(gt)
    ia,ib=imu_a.sample(gt),imu_b.sample(gt)
    assert np.allclose(ia.angular_velocity,ib.angular_velocity)
    assert np.allclose(ia.linear_acceleration,ib.linear_acceleration)
    ra,rb=range_a.sample(gt),range_b.sample(gt)
    da=np.array([r.distance for r in ra.rays]);db=np.array([r.distance for r in rb.rays])
    assert np.array_equal(np.isnan(da),np.isnan(db))
    assert np.allclose(da[~np.isnan(da)],db[~np.isnan(db)])
