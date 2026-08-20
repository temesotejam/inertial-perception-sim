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

def range_floor_normal(frame):
    pts=[r.direction*r.distance for r in frame.rays if np.isfinite(r.distance) and r.confidence>0]
    if len(pts)<6: return None
    pts=np.asarray(pts); c=pts.mean(axis=0); _,_,vh=np.linalg.svd(pts-c,full_matrices=False); n=vh[-1]; n/=np.linalg.norm(n)
    if n[2]<0:n=-n
    return n
