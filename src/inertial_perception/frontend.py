from __future__ import annotations
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def visual_relative_rotation(previous_frame,current_frame,camera,min_tracks=4):
    """Estimate frame-to-frame camera rotation from matched image bearings only."""
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


def _range_points(frame):
    return np.asarray([np.asarray(r.direction,float)*float(r.distance) for r in frame.rays if np.isfinite(r.distance) and r.confidence>0],float)


def range_relative_rotation(previous_frame,current_frame,min_pairs=8,max_pair_distance=.28,iterations=5):
    """Estimate relative rotation from consecutive range point clouds.

    The frontend knows only each sensor-frame ray and metric distance. It does
    not assume a horizontal floor or any world-frame surface normal. A small
    deterministic point-to-point ICP aligns the current cloud to the previous
    cloud; robust distance trimming rejects points that changed visibility.
    The returned rotation has the same convention as the visual frontend:
    previous_orientation.inv() * current_orientation.
    """
    if previous_frame is None or current_frame is None:return None
    prev=_range_points(previous_frame);cur=_range_points(current_frame)
    if len(prev)<min_pairs or len(cur)<min_pairs:return None
    rot=Rotation.identity();trans=np.zeros(3);pairs_used=0;rms=float('inf')
    for _ in range(iterations):
        moved=rot.apply(cur)+trans
        dist=np.linalg.norm(moved[:,None,:]-prev[None,:,:],axis=2)
        j=np.argmin(dist,axis=1);d=dist[np.arange(len(cur)),j]
        # Mutual nearest neighbors reduce edge/object swaps.
        i_back=np.argmin(dist,axis=0)
        mutual=np.array([i_back[jj]==ii for ii,jj in enumerate(j)],bool)
        valid=mutual&(d<max_pair_distance)
        ids=np.where(valid)[0]
        if len(ids)<min_pairs:
            ids=np.argsort(d)[:min(min_pairs,len(cur))]
            ids=ids[d[ids]<max_pair_distance*1.5]
        if len(ids)<min_pairs:return None
        a=prev[j[ids]];b=cur[ids]
        # Kabsch rigid alignment; translation is estimated but only rotation is
        # fed to the attitude estimator.
        ca=a.mean(axis=0);cb=b.mean(axis=0)
        try:new_rot,_=Rotation.align_vectors(a-ca,b-cb)
        except Exception:return None
        new_trans=ca-new_rot.apply(cb)
        residual=np.linalg.norm(new_rot.apply(b)+new_trans-a,axis=1)
        # One robust trim pass on each ICP iteration.
        keep=np.argsort(residual)[:max(min_pairs,int(np.ceil(len(residual)*.8)))]
        a2=a[keep];b2=b[keep];ca=a2.mean(axis=0);cb=b2.mean(axis=0)
        new_rot,_=Rotation.align_vectors(a2-ca,b2-cb);new_trans=ca-new_rot.apply(cb)
        residual=np.linalg.norm(new_rot.apply(b2)+new_trans-a2,axis=1)
        rot,trans=new_rot,new_trans;pairs_used=len(keep);rms=float(np.sqrt(np.mean(residual**2)))
    if pairs_used<min_pairs:return None
    return {"rotation":rot,"pairs":int(pairs_used),"range_rms_m":rms,"translation_m":float(np.linalg.norm(trans)),"prev_points":int(len(prev)),"current_points":int(len(cur))}


def _fit_plane_normal(points):
    pts=np.asarray(points,float);c=pts.mean(axis=0);_,_,vh=np.linalg.svd(pts-c,full_matrices=False);n=vh[-1];n/=np.linalg.norm(n)
    if n[2]<0:n=-n
    return n,c


def range_floor_normal(frame,local_fraction=.45):
    """Legacy/debug helper. Not used by the Version 3 estimator path."""
    valid=[(float(r.distance),r.direction*r.distance) for r in frame.rays if np.isfinite(r.distance) and r.confidence>0]
    if len(valid)<6:return None
    valid.sort(key=lambda x:x[0]);keep=max(8,int(np.ceil(len(valid)*local_fraction)));pts=np.asarray([p for _,p in valid[:keep]])
    n,c=_fit_plane_normal(pts);residual=np.abs((pts-c)@n)
    if len(pts)>=10:
        order=np.argsort(residual);trimmed=pts[order[:max(8,int(np.ceil(len(pts)*.8)))]];n,_=_fit_plane_normal(trimmed)
    return n
