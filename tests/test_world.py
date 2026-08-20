import numpy as np
from inertial_perception.world import SCENE,terrain_height,ray_scene_distance


def test_calibration_pad_is_flat_but_outer_terrain_is_not():
    assert abs(terrain_height(0.0,0.0)) < 1e-12
    samples=[terrain_height(2.0,-0.7),terrain_height(3.0,1.2),terrain_height(-1.8,2.0)]
    assert max(abs(x) for x in samples) > 0.04


def test_downward_ray_hits_shared_terrain_near_one_meter():
    distance,object_id=ray_scene_distance(np.array([0.0,0.0,1.0]),np.array([0.0,0.0,-1.0]),4.0)
    assert object_id == "terrain"
    assert distance is not None
    assert abs(distance-1.0) < 1e-3


def test_scene_contains_colored_geometry_and_light():
    assert len(SCENE["objects"]) >= 4
    assert all("color" in obj for obj in SCENE["objects"])
    assert len(SCENE["light"]["direction"]) == 3
