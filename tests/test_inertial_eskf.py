import numpy as np
from scipy.spatial.transform import Rotation
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


def test_legacy_relative_update_without_clone_injects_attitude_only():
    gt0=truth_at(0.,'translation');f=InertialESKF(gt0.position,gt0.velocity,gt0.orientation)
    for t in np.arange(0.,.5,.005):f.propagate(ideal_imu(truth_at(float(t),'translation')))
    p=f.position.copy();v=f.velocity.copy();bg=f.gyro_bias.copy();ba=f.accel_bias.copy();q=f.orientation
    f.update_camera_relative(Rotation.from_euler('xyz',[.3,-.2,.4],degrees=True),q,tracks=10,track_rms_deg=.15)
    assert np.allclose(f.position,p,atol=1e-12)
    assert np.allclose(f.velocity,v,atol=1e-12)
    assert np.allclose(f.gyro_bias,bg,atol=1e-12)
    assert np.allclose(f.accel_bias,ba,atol=1e-12)
    assert f.last_event['camera_state_coupling']=='attitude_only'


def test_camera_clone_augments_and_propagates_covariance():
    f=InertialESKF(initial_orientation=Rotation.identity());f.propagate(ImuSample(0.,np.zeros(3),np.array([0.,0.,9.80665])))
    f.set_camera_clone();assert f.P.shape==(18,18) and f.has_camera_clone
    cross0=f.P[:15,15:18].copy()
    for k in range(1,21):f.propagate(ImuSample(k*.005,np.array([.002,-.003,.004]),np.array([0.,0.,9.80665])))
    assert f.P.shape==(18,18);assert np.all(np.isfinite(f.P));assert np.allclose(f.P,f.P.T,atol=1e-10)
    assert not np.allclose(f.P[:15,15:18],cross0)


def test_cloned_camera_relative_update_can_correct_gyro_bias():
    f=InertialESKF(initial_orientation=Rotation.identity(),gyro_noise_std=0.,bias_rw_std=0.)
    f.propagate(ImuSample(0.,np.zeros(3),np.array([0.,0.,9.80665])));f.set_camera_clone();ref=f.camera_clone_orientation
    biased_rate=np.radians([.20,-.10,.35])
    for k in range(1,41):f.propagate(ImuSample(k*.005,biased_rate,np.array([0.,0.,9.80665])))
    before=f.gyro_bias.copy();f.update_camera_relative(Rotation.identity(),ref,tracks=12,track_rms_deg=.1)
    assert f.last_event['camera_clone_active'] is True
    assert f.last_event['camera_state_coupling']=='cloned_pose_relative'
    assert np.linalg.norm(f.gyro_bias-before)>1e-7
    assert np.dot(f.gyro_bias,biased_rate)>0
    assert np.all(np.isfinite(f.P)) and np.allclose(f.P,f.P.T,atol=1e-10)


def test_parallax_guard_rejects_camera_tilt_but_keeps_heading_component_without_range_support():
    f=InertialESKF(initial_orientation=Rotation.identity());f.set_camera_clone();ref=f.camera_clone_orientation
    f.update_camera_relative(Rotation.from_euler('xyz',[2.0,-1.5,1.0],degrees=True),ref,tracks=10,track_rms_deg=.15,parallax_detected=True,range_tilt_supported=False)
    e=f.orientation.as_euler('xyz',degrees=True)
    assert abs(e[0])<.05 and abs(e[1])<.05
    assert abs(e[2])>.05
    assert f.last_event['camera_state_coupling']=='cloned_pose_relative_yaw'


def test_correlated_range_relative_update_injects_attitude_only():
    gt0=truth_at(0.,'translation');f=InertialESKF(gt0.position,gt0.velocity,gt0.orientation)
    for t in np.arange(0.,.5,.005):f.propagate(ideal_imu(truth_at(float(t),'translation')))
    p=f.position.copy();v=f.velocity.copy();bg=f.gyro_bias.copy();ba=f.accel_bias.copy();q=f.orientation
    n0=np.array([0.,0.,1.]);n1=Rotation.from_euler('x',2,degrees=True).inv().apply(n0)
    f.update_range_relative(n0,n1,q,pairs=20,range_rms_m=.01,translation_m=.01,range_rotation_deg=2.)
    assert np.allclose(f.position,p,atol=1e-12)
    assert np.allclose(f.velocity,v,atol=1e-12)
    assert np.allclose(f.gyro_bias,bg,atol=1e-12)
    assert np.allclose(f.accel_bias,ba,atol=1e-12)
    assert f.last_event['range_state_coupling']=='attitude_only'
