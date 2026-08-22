import numpy as np
from scipy.spatial.transform import Rotation
from inertial_perception.world import truth_at
from inertial_perception.sensors import CameraSimulator,GridRangeSimulator
from inertial_perception.frontend import visual_relative_rotation,range_relative_rotation,range_floor_normal,_epipolar_refine
from inertial_perception.math3d import rotation_error_deg


def test_visual_relative_rotation_recovers_motion_without_world_map():
    rng=np.random.default_rng(1);cam=CameraSimulator(rng,pixel_noise_std=0,dropout=0)
    gt0=truth_at(1.15,'combined');gt1=truth_at(1.20,'combined');f0=cam.sample(gt0);f1=cam.sample(gt1)
    vis=visual_relative_rotation(f0,f1,cam)
    assert vis is not None and vis['tracks']>=4
    expected=gt0.orientation.inv()*gt1.orientation
    assert rotation_error_deg(vis['rotation'],expected)<1.0
    assert vis['track_rms_deg']<1.0


def test_feature_ids_do_not_require_world_positions():
    rng=np.random.default_rng(4);cam=CameraSimulator(rng,pixel_noise_std=0,dropout=0)
    f0=cam.sample(truth_at(.40,'combined'));f1=cam.sample(truth_at(.43,'combined'))
    for f in f0.features+f1.features:f.world_position=None
    vis=visual_relative_rotation(f0,f1,cam)
    assert vis is not None and vis['tracks']>=5


def test_epipolar_refinement_separates_translation_from_rotation():
    rng=np.random.default_rng(8);true_r=Rotation.from_euler('xyz',[1.1,-.7,1.8],degrees=True);t=np.array([.18,-.08,.05])
    p0=np.c_[rng.uniform(2.0,5.0,20),rng.uniform(-1.4,1.4,20),rng.uniform(-.9,.9,20)]
    p1=true_r.inv().apply(p0-t);b0=p0/np.linalg.norm(p0,axis=1)[:,None];b1=p1/np.linalg.norm(p1,axis=1)[:,None]
    prior=true_r*Rotation.from_euler('xyz',[.35,-.25,.20],degrees=True)
    pure,_=Rotation.align_vectors(b0,b1);epi=_epipolar_refine(b0,b1,prior,min_tracks=5)
    assert epi is not None
    assert rotation_error_deg(epi['rotation'],true_r)<rotation_error_deg(pure,true_r)
    assert rotation_error_deg(epi['rotation'],true_r)<.45


def test_relative_range_recovers_roll_without_horizontal_floor_prior():
    rng=np.random.default_rng(22);sensor=GridRangeSimulator(rng,noise_std=0,dropout=0)
    gt0=truth_at(.60,'roll');gt1=truth_at(.65,'roll');r0=sensor.sample(gt0);r1=sensor.sample(gt1)
    rel=range_relative_rotation(r0,r1)
    assert rel is not None and rel['pairs']>=8
    expected=gt0.orientation.inv()*gt1.orientation
    assert rotation_error_deg(rel['rotation'],expected)<2.0
    assert rel['range_rms_m']<.10


def test_range_plane_normal_exists_for_legacy_debug_only():
    rng=np.random.default_rng(2);sensor=GridRangeSimulator(rng,noise_std=0,dropout=0);gt=truth_at(.7,'roll');n=range_floor_normal(sensor.sample(gt));assert n is not None;expected=gt.orientation.inv().apply([0,0,1]);assert abs(float(np.dot(n,expected)))>.999
