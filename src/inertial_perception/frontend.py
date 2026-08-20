from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def visual_relative_rotation(previous_frame,current_frame,camera,min_tracks=5):
    """Estimate relative camera rotation from matched image bearings only.

    No world position, depth, or renderer ground truth is used. Matching is by
    persistent feature_id assigned by the image-space tracker. The returned
    rotation is R_prev^-1 * R_cur for the body-to-world convention used here.
    """
    if previous_frame is None or current_frame is None:return None
    prev={f.feature_id:f for f in previous_frame.features};pairs=[]
    for f in current_frame.features:
        p=prev.get(f.feature_id)
        if p is None:continue
        pairs.append((camera.bearing_from_pixel(p.u,p.v),camera.bearing_from_pixel(f.u,f.v)))
    if len(pairs)<min_tracks:return None
    prev_b=np.asarray([a for a,_ in pairs]);cur_b=np.asarray([b for _,b in pairs])
    rot,_=Rotation.align_vectors(prev_b,cur_b)
    # One robust trimming pass rejects mismatched 2-D nearest-neighbour tracks.
    pred=rot.apply(cur_b);dots=np.clip(np.sum(pred*prev_b,axis=1),-1,1);err=np.arccos(dots)
    if len(pairs)>=8:
        keep=np.argsort(err)[:max(min_tracks,int(np.ceil(len(pairs)*.75)))]
        rot,_=Rotation.align_vectors(prev_b[keep],cur_b[keep]);err=err[keep]
    rms=float(np.degrees(np.sqrt(np.mean(err**2)))) if len(err) else float('nan')
    return {"rotation":rot,"tracks":int(len(err)),"track_rms_deg":rms}


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
