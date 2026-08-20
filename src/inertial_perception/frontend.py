from __future__ import annotations
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def visual_relative_rotation(previous_frame,current_frame,camera,min_tracks=4):
    """Estimate frame-to-frame camera rotation from matched image bearings only.

    Matching is by image-tracked feature_id. A deterministic small-sample
    RANSAC step keeps only correspondences that can be explained by one rigid
    rotation, rejecting appearance matches that are geometrically inconsistent.
    No world position or renderer ground truth is used.
    """
    if previous_frame is None or current_frame is None:return None
    prev={f.feature_id:f for f in previous_frame.features};pairs=[]
    for f in current_frame.features:
        p=prev.get(f.feature_id)
        if p is not None:pairs.append((camera.bearing_from_pixel(p.u,p.v),camera.bearing_from_pixel(f.u,f.v)))
    if len(pairs)<min_tracks:return None
    prev_b=np.asarray([a for a,_ in pairs]);cur_b=np.asarray([b for _,b in pairs]);n=len(pairs)
    best=None;best_in=None;thr=np.radians(.65)
    combos=list(itertools.combinations(range(n),3))[:160]
    for ids in combos:
        try:rot,_=Rotation.align_vectors(prev_b[list(ids)],cur_b[list(ids)])
        except Exception:continue
        pred=rot.apply(cur_b);err=np.arccos(np.clip(np.sum(pred*prev_b,axis=1),-1,1));inn=np.where(err<thr)[0]
        if len(inn)<min_tracks:continue
        score=(len(inn),-float(np.mean(err[inn])))
        if best is None or score>best:best=score;best_in=inn
    if best_in is None:
        rot,_=Rotation.align_vectors(prev_b,cur_b);pred=rot.apply(cur_b);err=np.arccos(np.clip(np.sum(pred*prev_b,axis=1),-1,1));best_in=np.argsort(err)[:max(min_tracks,int(np.ceil(n*.6)))]
    rot,_=Rotation.align_vectors(prev_b[best_in],cur_b[best_in]);pred=rot.apply(cur_b[best_in]);err=np.arccos(np.clip(np.sum(pred*prev_b[best_in],axis=1),-1,1));rms=float(np.degrees(np.sqrt(np.mean(err**2))))
    return {"rotation":rot,"tracks":int(len(best_in)),"candidate_tracks":int(n),"track_rms_deg":rms}


def _fit_plane_normal(points):
    pts=np.asarray(points,float);c=pts.mean(axis=0);_,_,vh=np.linalg.svd(pts-c,full_matrices=False);n=vh[-1];n/=np.linalg.norm(n)
    if n[2]<0:n=-n
    return n,c


def range_floor_normal(frame,local_fraction=.45):
    valid=[(float(r.distance),r.direction*r.distance) for r in frame.rays if np.isfinite(r.distance) and r.confidence>0]
    if len(valid)<6:return None
    valid.sort(key=lambda x:x[0]);keep=max(8,int(np.ceil(len(valid)*local_fraction)));pts=np.asarray([p for _,p in valid[:keep]])
    n,c=_fit_plane_normal(pts);residual=np.abs((pts-c)@n)
    if len(pts)>=10:
        order=np.argsort(residual);trimmed=pts[order[:max(8,int(np.ceil(len(pts)*.8)))]];n,_=_fit_plane_normal(trimmed)
    return n
