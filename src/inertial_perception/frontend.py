from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .world import LANDMARKS

def visual_orientation(frame,camera,position=np.array([0.,0.,1.])):
    if len(frame.features)<3: return None
    world=[]; body=[]
    for f in frame.features:
        w=LANDMARKS[f.feature_id]-position; world.append(w/np.linalg.norm(w)); body.append(camera.bearing_from_pixel(f.u,f.v))
    rot,_=Rotation.align_vectors(np.asarray(world),np.asarray(body)); return rot

def _fit_plane_normal(points):
    pts=np.asarray(points,float); c=pts.mean(axis=0); _,_,vh=np.linalg.svd(pts-c,full_matrices=False); n=vh[-1]; n/=np.linalg.norm(n)
    if n[2]<0:n=-n
    return n,c

def range_floor_normal(frame,local_fraction=.45):
    """Estimate a local support-plane normal from generic range rays.

    A wide-FOV range sensor may simultaneously see nearby support ground,
    distant terrain relief, and objects. Using every hit as a single 'floor'
    biases attitude correction. We therefore use the nearest local subset for
    the inertial constraint, while all rays remain available to perception and
    RGB-D rendering.
    """
    valid=[(float(r.distance),r.direction*r.distance) for r in frame.rays if np.isfinite(r.distance) and r.confidence>0]
    if len(valid)<6:return None
    valid.sort(key=lambda x:x[0]); keep=max(8,int(np.ceil(len(valid)*local_fraction))); pts=np.asarray([p for _,p in valid[:keep]])
    n,c=_fit_plane_normal(pts)
    # One robust trimming pass suppresses object edges among the local rays.
    residual=np.abs((pts-c)@n)
    if len(pts)>=10:
        order=np.argsort(residual); trimmed=pts[order[:max(8,int(np.ceil(len(pts)*.8)))]]; n,_=_fit_plane_normal(trimmed)
    return n
