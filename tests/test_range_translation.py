import numpy as np
from scipy.spatial.transform import Rotation
from inertial_perception.world import truth_at
from inertial_perception.sensors import GridRangeSimulator
from inertial_perception.frontend import range_normal_translation
from inertial_perception.inertial_eskf import InertialESKF
from inertial_perception.simulation import run_simulation


def test_range_plane_offset_recovers_normal_displacement():
    sensor=GridRangeSimulator(np.random.default_rng(12),noise_std=0,dropout=0)
    gt0=truth_at(.20,'translation');gt1=truth_at(.32,'translation')
    f0=sensor.sample(gt0);f1=sensor.sample(gt1)
    relative=gt0.orientation.inv()*gt1.orientation
    obs=range_normal_translation(f0,f1,relative)
    assert obs is not None
    n_world=gt0.orientation.apply(obs['normal_previous'])
    expected=float(np.dot(n_world,gt1.position-gt0.position))
    assert abs(obs['displacement_m']-expected)<.025
    assert obs['observable']=='normal_translation_only'


def test_range_translation_eskf_updates_only_one_measured_position_component():
    f=InertialESKF(initial_position=np.array([0.,0.,1.]),initial_velocity=np.zeros(3),initial_orientation=Rotation.identity())
    f.position=np.array([.4,-.3,1.12])
    before=f.position.copy();ref=np.array([0.,0.,1.])
    f.update_range_translation(np.array([0.,0.,1.]),.02,ref,Rotation.identity(),quality=1.0,plane_rms_m=.005,normal_angle_deg=.2)
    assert abs(f.position[2]-(ref[2]+.02)) < abs(before[2]-(ref[2]+.02))
    assert abs(f.position[0]-before[0])<.02
    assert abs(f.position[1]-before[1])<.02
    assert f.last_event['kind']=='range_translation'
    assert f.last_event['position_correction_m']>0


def test_range_translation_reduces_position_uncertainty_along_observed_axis():
    f=InertialESKF(initial_position=np.array([0.,0.,1.]),initial_velocity=np.zeros(3),initial_orientation=Rotation.identity())
    before=float(f.P[2,2])
    f.update_range_translation(np.array([0.,0.,1.]),0.,np.array([0.,0.,1.]),Rotation.identity(),quality=1.0)
    assert f.P[2,2] < before
    assert np.isfinite(f.P).all()


def test_range_translation_reduces_vertical_position_drift_in_translation_scenario():
    imu=run_simulation('translation',4.,21,False,False,estimator_kind='ins_eskf')['metrics']
    rng=run_simulation('translation',4.,21,False,True,estimator_kind='ins_eskf')['metrics']
    assert rng['vertical_position_rmse_m'] < imu['vertical_position_rmse_m']
    assert np.isfinite(rng['final_vertical_position_error_m'])
