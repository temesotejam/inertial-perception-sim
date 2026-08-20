import numpy as np
from inertial_perception.world import truth_at
from inertial_perception.sensors import CameraSimulator
from inertial_perception.simulation import run_simulation


def _gray(rgb):
    return .299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2]


def test_features_are_detected_from_the_same_rendered_image():
    cam=CameraSimulator(np.random.default_rng(5),pixel_noise_std=0,dropout=0)
    frame=cam.sample(truth_at(.9,'combined'))
    assert frame.image_rgb.shape==(cam.render_height,cam.render_width,3)
    assert len(frame.features)>=3
    gray=_gray(frame.image_rgb)
    sx=cam.render_width/cam.width; sy=cam.render_height/cam.height
    strong=0
    for f in frame.features:
        assert f.world_position is not None
        x=int(np.clip(f.u*sx,1,cam.render_width-2)); y=int(np.clip(f.v*sy,1,cam.render_height-2))
        gx=abs(float(gray[y,x+1])-float(gray[y,x-1])); gy=abs(float(gray[y+1,x])-float(gray[y-1,x]))
        strong += (gx+gy)>8
    assert strong>=max(2,len(frame.features)//3)


def test_feature_ids_track_between_nearby_camera_frames():
    cam=CameraSimulator(np.random.default_rng(6),pixel_noise_std=0,dropout=0)
    a=cam.sample(truth_at(1.0,'combined')); b=cam.sample(truth_at(1.0+1/15,'combined'))
    ids_a={f.feature_id for f in a.features}; ids_b={f.feature_id for f in b.features}
    assert len(ids_a & ids_b)>=2


def test_exported_camera_image_and_features_share_timestamp():
    out=run_simulation('combined',duration=.4,seed=8,camera_enabled=True,range_enabled=True)
    records=[r for r in out['records'] if 'camera_rgb_b64' in r and 'camera_features' in r]
    assert records
    for r in records[:5]:
        assert r['camera_timestamp']<=r['t']+1e-9
        assert len(r['camera_pose_euler'])==3
        assert r['camera_render_size']==[80,60]
        assert abs(r.get('range_camera_dt_ms',0))<80
