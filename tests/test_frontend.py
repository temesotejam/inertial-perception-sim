import numpy as np
from inertial_perception.world import truth_at
from inertial_perception.sensors import CameraSimulator,GridRangeSimulator
from inertial_perception.frontend import visual_orientation,range_floor_normal
from inertial_perception.math3d import rotation_error_deg

def test_visual_orientation_recovers_truth():
    rng=np.random.default_rng(1); cam=CameraSimulator(rng,pixel_noise_std=0,dropout=0); gt=truth_at(1.2,'combined'); obs=visual_orientation(cam.sample(gt),cam,gt.position); assert obs is not None; assert rotation_error_deg(obs,gt.orientation)<.05

def test_range_plane_normal_exists():
    rng=np.random.default_rng(2); sensor=GridRangeSimulator(rng,noise_std=0,dropout=0); gt=truth_at(.7,'roll'); n=range_floor_normal(sensor.sample(gt)); assert n is not None; expected=gt.orientation.inv().apply([0,0,1]); assert abs(float(np.dot(n,expected)))>.999
