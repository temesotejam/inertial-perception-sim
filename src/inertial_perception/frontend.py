from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def visual_orientation(frame,camera,position=np.array([0.,0.,1.])):
    """Estimate absolute camera orientation from image-detected mapped features.

    Feature locations now come from Harris corners on the rendered image. The
    simulation renderer attaches the corresponding static world hit to each
    corner, standing in for the landmark map that a real VIO/SLAM system would
    build over time.
    """
    usable=[f for f in frame.features if f.world_position is not None]
    if len(usable)<3:return None
    world=[]; body=[]
    for f in usable:
        w=np.asarray(f.world_position,float)-position
        if np.linalg.norm(w)<1e-8:continue
        world.append(w/np.linalg.norm(w)); body.append(camera.bearing_from_pixel(f.u,f.v))
    if len(world)<3:return None
    rot,_=Rotation.align_vectors(np.asarray(world),np.asarray(body)); return rot


def _fit_plane_normal(points):
    pts=np.asarray(points,float); c=pts.mean(axis=0); _,_,vh=np.linalg.svd(pts-c,full_matrices=False); n=vh[-1]; n/=np.linalg.norm(n)
    if n[2]<0:n=-n
    return n,c


def range_floor_normal(frame,local_fraction=.45):
    """Estimate a local support-plane normal from generic range rays."""
    valid=[(float(r.distance),r.direction*r.distance) for r in frame.rays if np.isfinite(r.distance) and r.confidence>0]
    if len(valid)<6:return None
    valid.sort(key=lambda x:x[0]); keep=max(8,int(np.ceil(len(valid)*local_fraction))); pts=np.asarray([p for _,p in valid[:keep]])
    n,c=_fit_plane_normal(pts); residual=np.abs((pts-c)@n)
    if len(pts)>=10:
        order=np.argsort(residual); trimmed=pts[order[:max(8,int(np.ceil(len(pts)*.8)))]]; n,_=_fit_plane_normal(trimmed)
    return n
