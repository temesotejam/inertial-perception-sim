import numpy as np
from inertial_perception.inertial_eskf import InertialESKF,G_WORLD
from inertial_perception.types import ImuSample
from inertial_perception.world import truth_at
from inertial_perception.math3d import rotation_error_deg
from inertial_perception.simulation import run_simulation


def ideal_imu(gt):
    specific=gt.orientation.inv().apply(gt.acceleration-G_WORLD)
    return ImuSample(gt.timestamp,gt.angular_velocity.copy(),specific)


def test_full_inertial_eskf_has_15_state_covariance():
    gt=truth_at(0.,'translation');f=InertialESKF(gt.position,gt.velocity,gt.orientation)
    assert f.P.shape==(15,15)
    assert np.all(np.isfinite(f.P))
    assert np.allclose(f.P,f.P.T)


def test_ideal_imu_propagates_translation_state():
    gt0=truth_at(0.,'translation');f=InertialESKF(gt0.position,gt0.velocity,gt0.orientation,gyro_noise_std=0,accel_noise_std=0,bias_rw_std=0,accel_bias_rw_std=0)
    for t in np.arange(0.,2.0001,.005):f.propagate(ideal_imu(truth_at(float(t),'translation')))
    gt=truth_at(2.,'translation')
    assert np.linalg.norm(f.position-gt.position)<.02
    assert np.linalg.norm(f.velocity-gt.velocity)<.02
    assert rotation_error_deg(f.orientation,gt.orientation)<.15


def test_translation_simulation_exports_position_velocity_and_uncertainty():
    out=run_simulation('translation',2.,5,False,False,estimator_kind='ins_eskf')
    m=out['metrics'];r=out['records'][-1]
    assert np.isfinite(m['position_rmse_m']) and np.isfinite(m['velocity_rmse_mps'])
    assert len(r['est_position'])==3 and len(r['est_velocity'])==3 and len(r['accel_bias'])==3
    assert len(r['eskf_sigma_pos_m'])==3 and len(r['eskf_sigma_vel_mps'])==3 and len(r['eskf_sigma_accel_bias'])==3
    assert max(r['eskf_sigma_pos_m'])>.05


def test_external_attitude_constraints_do_not_directly_reset_position():
    out=run_simulation('translation',1.5,8,True,True,estimator_kind='ins_eskf')
    assert out['meta']['state']=='p,v,q,b_g,b_a'
    assert np.isfinite(out['metrics']['final_position_error_m'])
