import numpy as np
from scipy.spatial.transform import Rotation
from inertial_perception.eskf import AttitudeESKF
from inertial_perception.types import ImuSample


def test_eskf_covariance_stays_finite_and_symmetric():
    f=AttitudeESKF()
    for k in range(1,401):
        t=k/200;f.propagate(ImuSample(t,np.array([.01,-.015,.02]),np.array([0.,0.,9.80665])))
    assert np.all(np.isfinite(f.P));assert np.allclose(f.P,f.P.T,atol=1e-10)
    assert np.min(np.linalg.eigvalsh(f.P))>-1e-10


def test_camera_update_reduces_attitude_uncertainty():
    f=AttitudeESKF(initial_orientation=Rotation.identity());before=float(np.trace(f.P[:3,:3]))
    rel=Rotation.from_euler('z',2,degrees=True)
    f.update_camera_relative(rel,Rotation.identity(),tracks=10,track_rms_deg=.15)
    after=float(np.trace(f.P[:3,:3]))
    assert after<before;assert f.last_event['kind']=='camera_relative'


def test_range_update_preserves_unobserved_normal_axis_information():
    f=AttitudeESKF(initial_orientation=Rotation.from_euler('z',15,degrees=True));before_yaw=f.orientation.as_euler('xyz',degrees=True)[2]
    n0=np.array([0.,0.,1.]);n1=Rotation.from_euler('x',3,degrees=True).inv().apply(n0)
    f.update_range_relative(n0,n1,Rotation.from_euler('z',15,degrees=True),pairs=20,range_rms_m=.01,translation_m=.01,range_rotation_deg=3.)
    after_yaw=f.orientation.as_euler('xyz',degrees=True)[2]
    assert abs(after_yaw-before_yaw)<.25
    assert f.last_event['kind']=='range_relative'
