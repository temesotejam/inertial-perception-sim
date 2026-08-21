from __future__ import annotations
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def visual_relative_rotation(previous_frame,current_frame,camera,min_tracks=4,rotation_prior=None):
    """Estimate frame-to-frame camera rotation from matched image bearings.

    When an inertial relative-rotation prior is available it is used only to
    reject/weight tracks whose motion is inconsistent with rigid rotation.
    This prevents translational parallax from being absorbed as false camera
    attitude while keeping the final visual constraint image-derived.
    """
    if previous_frame is None or current_frame is None:return None
    prev={f.feature_id:f for f in previous_frame.features};pairs=[]
    for f in current_frame.features:
        p=prev.get(f.feature_id)
        if p is not None:pairs.append((camera.bearing_from_pixel(p.u,p.v),camera.bearing_from_pixel(f.u,f.v)))
    if len(pairs)<min_tracks:return None
    prev_b=np.asarray([a for a,_ in pairs]);cur_b=np.asarray([b for _,b in pairs]);n=len(pairs);candidate_tracks=n
    prior_rms=float('nan');prior_kept=n;weights=None
    if rotation_prior is not None:
        pred=np.asarray(rotation_prior.apply(cur_b));pe=np.arccos(np.clip(np.sum(pred*prev_b,axis=1),-1,1));prior_rms=float(np.degrees(np.sqrt(np.mean(pe**2))))
        order=np.argsort(pe);keep_n=max(min_tracks,int(np.ceil(n*.60)));gate=min(np.radians(1.2),max(np.radians(.30),float(np.percentile(pe,65))))
        ids=np.where(pe<=gate)[0]
        if len(ids)<keep_n:ids=order[:keep_n]
        elif len(ids)>keep_n:ids=ids[np.argsort(pe[ids])[:keep_n]]
        prev_b=prev_b[ids];cur_b=cur_b[ids];pe=pe[ids];n=len(ids);prior_kept=n
        floor=np.radians(.08);weights=1.0/np.maximum(pe,floor)**2;weights=weights/np.mean(weights)
    best=None;best_in=None;thr=np.radians(.65)
    combos=list(itertools.combinations(range(n),3))[:160]
    for ids in combos:
        try:
            w=None if weights is None else weights[list(ids)]
            rot,_=Rotation.align_vectors(prev_b[list(ids)],cur_b[list(ids)],weights=w)
        except Exception:continue
        pred=rot.apply(cur_b);err=np.arccos(np.clip(np.sum(pred*prev_b,axis=1),-1,1));inn=np.where(err<thr)[0]
        if len(inn)<min_tracks:continue
        score=(len(inn),-float(np.mean(err[inn])))
        if best is None or score>best:best=score;best_in=inn
    if best_in is None:
        try:
            rot,_=Rotation.align_vectors(prev_b,cur_b,weights=weights)
        except Exception:return None
        pred=rot.apply(cur_b);err=np.arccos(np.clip(np.sum(pred*prev_b,axis=1),-1,1));best_in=np.argsort(err)[:max(min_tracks,int(np.ceil(n*.6)))]
    w=None if weights is None else weights[best_in]
    rot,_=Rotation.align_vectors(prev_b[best_in],cur_b[best_in],weights=w);pred=rot.apply(cur_b[best_in]);err=np.arccos(np.clip(np.sum(pred*prev_b[best_in],axis=1),-1,1));rms=float(np.degrees(np.sqrt(np.mean(err**2))))
    return {"rotation":rot,"tracks":int(len(best_in)),"candidate_tracks":int(candidate_tracks),"track_rms_deg":rms,"prior_used":bool(rotation_prior is not None),"prior_kept_tracks":int(prior_kept),"prior_residual_rms_deg":prior_rms}


def _range_points(frame):
    return np.asarray([np.asarray(r.direction,float)*float(r.distance) for r in frame.rays if np.isfinite(r.distance) and r.confidence>0],float)


def _fit_plane_normal(points):
    pts=np.asarray(points,float);c=pts.mean(axis=0);_,_,vh=np.linalg.svd(pts-c,full_matrices=False);n=vh[-1];n/=np.linalg.norm(n)
    if n[2]<0:n=-n
    return n,c


def _local_range_plane(frame,local_fraction=.45):
    valid=[(float(r.distance),np.asarray(r.direction,float)*float(r.distance)) for r in frame.rays if np.isfinite(r.distance) and r.confidence>0]
    if len(valid)<8:return None
    valid.sort(key=lambda x:x[0]);keep=max(8,int(np.ceil(len(valid)*local_fraction)));pts=np.asarray([p for _,p in valid[:keep]])
    n,c=_fit_plane_normal(pts);residual=np.abs((pts-c)@n)
    if len(pts)>=10:
        order=np.argsort(residual);pts=pts[order[:max(8,int(np.ceil(len(pts)*.8)))]];n,c=_fit_plane_normal(pts);residual=np.abs((pts-c)@n)
    return {"normal":n,"center":c,"points":pts,"plane_rms_m":float(np.sqrt(np.mean(residual**2)))}


def _minimal_rotation(source,target):
    a=np.asarray(source,float);b=np.asarray(target,float);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    cross=np.cross(a,b);s=np.linalg.norm(cross);c=float(np.clip(np.dot(a,b),-1,1))
    if s<1e-12:
        if c>0:return Rotation.identity()
        axis=np.cross(a,[1.,0.,0.])
        if np.linalg.norm(axis)<1e-6:axis=np.cross(a,[0.,1.,0.])
        axis/=np.linalg.norm(axis);return Rotation.from_rotvec(axis*np.pi)
    return Rotation.from_rotvec((cross/s)*np.arctan2(s,c))


def _range_icp_diagnostics(prev,cur,min_pairs=8,max_pair_distance=.28):
    if len(prev)<min_pairs or len(cur)<min_pairs:return None
    dist=np.linalg.norm(cur[:,None,:]-prev[None,:,:],axis=2);j=np.argmin(dist,axis=1);d=dist[np.arange(len(cur)),j]
    i_back=np.argmin(dist,axis=0);mutual=np.array([i_back[jj]==ii for ii,jj in enumerate(j)],bool)
    ids=np.where(mutual&(d<max_pair_distance))[0]
    if len(ids)<min_pairs:
        ids=np.argsort(d)[:min(min_pairs,len(cur))];ids=ids[d[ids]<max_pair_distance*1.5]
    if len(ids)<min_pairs:return None
    a=prev[j[ids]];b=cur[ids];ca=a.mean(axis=0);cb=b.mean(axis=0)
    try:rot,_=Rotation.align_vectors(a-ca,b-cb)
    except Exception:return None
    trans=ca-rot.apply(cb);res=np.linalg.norm(rot.apply(b)+trans-a,axis=1)
    keep=np.argsort(res)[:max(min_pairs,int(np.ceil(len(res)*.8)))];a=a[keep];b=b[keep];ca=a.mean(axis=0);cb=b.mean(axis=0)
    rot,_=Rotation.align_vectors(a-ca,b-cb);trans=ca-rot.apply(cb);res=np.linalg.norm(rot.apply(b)+trans-a,axis=1)
    return {"pairs":int(len(keep)),"range_rms_m":float(np.sqrt(np.mean(res**2))),"translation_m":float(np.linalg.norm(trans))}


def range_relative_rotation(previous_frame,current_frame,min_pairs=8):
    """Estimate only the Range-observable relative tilt between frames."""
    if previous_frame is None or current_frame is None:return None
    p0=_local_range_plane(previous_frame);p1=_local_range_plane(current_frame)
    if p0 is None or p1 is None:return None
    rot=_minimal_rotation(p1["normal"],p0["normal"])
    prev=_range_points(previous_frame);cur=_range_points(current_frame);diag=_range_icp_diagnostics(prev,cur,min_pairs=min_pairs)
    if diag is None:return None
    plane_rms=max(p0["plane_rms_m"],p1["plane_rms_m"])
    return {"rotation":rot,"previous_normal":p0["normal"].copy(),"current_normal":p1["normal"].copy(),"pairs":diag["pairs"],"range_rms_m":max(diag["range_rms_m"],plane_rms),"translation_m":diag["translation_m"],"prev_points":int(len(prev)),"current_points":int(len(cur)),"observable":"tilt_only","plane_rms_m":float(plane_rms)}


def range_normal_translation(previous_frame,current_frame,relative_rotation):
    """Measure only the translation component observable along a local plane normal."""
    if previous_frame is None or current_frame is None or relative_rotation is None:return None
    p0=_local_range_plane(previous_frame);p1=_local_range_plane(current_frame)
    if p0 is None or p1 is None:return None
    n0=p0["normal"]/np.linalg.norm(p0["normal"]);n1r=relative_rotation.apply(p1["normal"]);n1r/=np.linalg.norm(n1r)
    if float(np.dot(n0,n1r))<0:n1r=-n1r
    normal_angle=float(np.degrees(np.arccos(np.clip(np.dot(n0,n1r),-1,1))))
    n=n0+n1r;nn=np.linalg.norm(n)
    if nn<1e-8:return None
    n/=nn
    c1r=relative_rotation.apply(p1["center"])
    displacement=float(np.dot(n,p0["center"]-c1r))
    plane_rms=max(float(p0["plane_rms_m"]),float(p1["plane_rms_m"]))
    quality=float(np.clip((1.0-normal_angle/8.0)*(1.0-plane_rms/.05),0.0,1.0))
    if normal_angle>12.0 or plane_rms>.08:return None
    return {"normal_previous":n.copy(),"displacement_m":displacement,"normal_angle_deg":normal_angle,"plane_rms_m":plane_rms,"quality":quality,"observable":"normal_translation_only"}


def range_floor_normal(frame,local_fraction=.45):
    """Legacy/debug helper. Not used by the active estimator path."""
    p=_local_range_plane(frame,local_fraction)
    return None if p is None else p["normal"]
