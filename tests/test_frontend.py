import numpy as np
from inertial_perception.world import truth_at
from inertial_perception.sensors import CameraSimulator,GridRangeSimulator
from inertial_perception.frontend import visual_relative_rotation,range_floor_normal
from inertial_perception.math3d import rotation_error_deg


def test_visual_relative_rotation_recovers_motion_without_world_map():
    rng=np.random.default_rng(1);cam=CameraSimulator(rng,pixel_noise_std=0,dropout=0)
    gt0=truth_at(1.15,'combined');gt1=truth_at(1.20,'combined');f0=cam.sample(gt0);f1=cam.sample(gt1)
    vis=visual_relative_rotation(f0,f1,cam)
    assert vis is not None and vis['tracks']>=5
    expected=gt0.orientation.inv()*gt1.orientation
    assert rotation_error_deg(vis['rotation'],expected)<1.0


def test_feature_ids_do_not_require_world_positions():
    rng=np.random.default_rng(4);cam=CameraSimulator(rng,pixel_noise_std=0,dropout=0)
    f0=cam.sample(truth_at(.40,'combined'));f1=cam.sample(truth_at(.43,'combined'))
    # Erase simulation-only 3-D associations: the relative frontend must still work.
    for f in f0.features+f1.features:f.world_position=None
    vis=visual_relative_rotation(f0,f1,cam)
    assert vis is not None and vis['tracks']>=5


def test_range_plane_normal_exists():
    rng=np.random.default_rng(2);sensor=GridRangeSimulator(rng,noise_std=0,dropout=0);gt=truth_at(.7,'roll');n=range_floor_normal(sensor.sample(gt));assert n is not None;expected=gt.orientation.inv().apply([0,0,1]);assert abs(float(np.dot(n,expected)))>.999
