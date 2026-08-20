import numpy as np
from inertial_perception.world import truth_at
from inertial_perception.sensors import CameraSimulator, GridRangeSimulator
from inertial_perception.simulation import _range_to_camera_overlay
from inertial_perception.frontend import range_floor_normal


def test_tilted_range_sensor_overlaps_camera_fov():
    rng=np.random.default_rng(12)
    cam=CameraSimulator(rng,pixel_noise_std=0,dropout=0)
    sensor=GridRangeSimulator(rng,noise_std=0,dropout=0,tilt_down_deg=35)
    gt=truth_at(0.0,'static')
    frame=sensor.sample(gt)
    overlay=_range_to_camera_overlay(frame,cam)
    assert len(overlay)>=8
    assert all(0<=p[0]<cam.width and 0<=p[1]<cam.height for p in overlay)
    assert all(0<p[2]<=sensor.max_range for p in overlay)


def test_tilted_range_still_recovers_floor_normal():
    rng=np.random.default_rng(13)
    sensor=GridRangeSimulator(rng,noise_std=0,dropout=0,tilt_down_deg=35)
    gt=truth_at(0.4,'roll')
    normal=range_floor_normal(sensor.sample(gt))
    assert normal is not None
    expected=gt.orientation.inv().apply([0,0,1])
    assert abs(float(np.dot(normal,expected)))>.995
