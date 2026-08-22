from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, sobel
from .world import SCENE


def _hex_rgb(h):
    h=h.lstrip('#'); return np.array([int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)],float)


def _terrain_height_array(x,y):
    t=SCENE['terrain']; r=np.hypot(x,y); a=t['flat_radius']; b=t['blend_radius']
    blend=np.zeros_like(r,float); blend[r>=b]=1.0; m=(r>a)&(r<b)
    u=(r[m]-a)/(b-a); blend[m]=u*u*(3-2*u)
    wave=t['amplitude']*(0.58*np.sin(t['wave_x']*x)*np.cos(t['wave_y']*y)+0.27*np.sin(1.65*y+0.35*x))
    r2=(x-t['bump_x'])**2+(y-t['bump_y'])**2
    bump=t['bump']*np.exp(-r2/(2*t['bump_sigma']**2))
    return blend*(wave+bump)


def _terrain_hits(origin,directions,max_range):
    n=len(directions); lo=np.zeros(n); hi=np.zeros(n); hit=np.zeros(n,bool)
    prev_t=0.0; prev_p=origin[None,:]+directions*prev_t
    prev_f=prev_p[:,2]-_terrain_height_array(prev_p[:,0],prev_p[:,1])
    for t in np.linspace(max_range/64,max_range,64):
        p=origin[None,:]+directions*t; f=p[:,2]-_terrain_height_array(p[:,0],p[:,1])
        cross=(~hit)&(f<=0)&(prev_f>0)
        lo[cross]=prev_t; hi[cross]=t; hit[cross]=True
        prev_t=float(t); prev_f=f
    for _ in range(8):
        mid=(lo+hi)/2; p=origin[None,:]+directions*mid[:,None]
        f=p[:,2]-_terrain_height_array(p[:,0],p[:,1])
        high=hit&(f<=0); low=hit&~high; hi[high]=mid[high]; lo[low]=mid[low]
    out=np.full(n,np.inf); out[hit]=(lo[hit]+hi[hit])/2
    return out


def _box_hits(origin,directions,obj,max_range):
    c=np.asarray(obj['center'],float); s=np.asarray(obj['size'],float); lower=c-s/2; upper=c+s/2
    eps=1e-12; inv=np.where(np.abs(directions)>eps,1.0/directions,np.inf)
    t0=(lower-origin)[None,:]*inv; t1=(upper-origin)[None,:]*inv
    tmin=np.max(np.minimum(t0,t1),axis=1); tmax=np.min(np.maximum(t0,t1),axis=1)
    valid=(tmax>=np.maximum(tmin,0))&(tmin<=max_range)
    out=np.full(len(directions),np.inf); out[valid]=np.maximum(tmin[valid],0)
    return out


def _pillar_hits(origin,directions,obj,max_range):
    c=np.asarray(obj['center'],float); r=float(obj['radius']); h=float(obj['height']); z0=c[2]-h/2; z1=c[2]+h/2
    ox,oy=origin[0]-c[0],origin[1]-c[1]; dx,dy=directions[:,0],directions[:,1]
    a=dx*dx+dy*dy; b=2*(ox*dx+oy*dy); cc=ox*ox+oy*oy-r*r; disc=b*b-4*a*cc
    out=np.full(len(directions),np.inf); valid=(a>1e-12)&(disc>=0)
    sqrt=np.sqrt(np.maximum(disc,0)); roots=np.stack(((-b-sqrt)/(2*np.where(a>1e-12,a,1)),(-b+sqrt)/(2*np.where(a>1e-12,a,1))),axis=1)
    roots.sort(axis=1)
    for j in range(2):
        t=roots[:,j]; z=origin[2]+t*directions[:,2]
        ok=valid&(t>=0)&(t<=max_range)&(z>=z0)&(z<=z1)&~np.isfinite(out)
        out[ok]=t[ok]
    return out


def render_camera(gt,width=96,height=72,fov_deg=70.,max_range=7.0):
    fx=width/(2*np.tan(np.radians(fov_deg)/2)); fy=fx; cx=width/2; cy=height/2
    yy,xx=np.mgrid[0:height,0:width]
    dirs_body=np.stack((np.ones_like(xx,float),-(xx+.5-cx)/fx,-(yy+.5-cy)/fy),axis=-1).reshape(-1,3)
    dirs_body/=np.linalg.norm(dirs_body,axis=1,keepdims=True)
    dirs_world=gt.orientation.apply(dirs_body); origin=np.asarray(gt.position,float)
    best=_terrain_hits(origin,dirs_world,max_range); hit_id=np.zeros(len(best),int)
    for j,obj in enumerate(SCENE['objects'],start=1):
        d=_pillar_hits(origin,dirs_world,obj,max_range) if obj['type']=='pillar' else _box_hits(origin,dirs_world,obj,max_range)
        m=d<best; best[m]=d[m]; hit_id[m]=j
    finite=np.isfinite(best); points=np.full((len(best),3),np.nan,float); points[finite]=origin+dirs_world[finite]*best[finite,None]
    rgb=np.zeros((len(best),3),float);sky_t=np.clip((yy.reshape(-1)+.5)/height,0,1)[:,None]
    rgb[:]=np.array([126,184,220])*(1-sky_t)+np.array([226,235,226])*sky_t
    terrain=finite&(hit_id==0)
    if np.any(terrain):
        p=points[terrain]; base=_hex_rgb(SCENE['terrain']['base_color']);tex=.82+.11*np.sin(4.2*p[:,0])*.5+.08*np.cos(5.1*p[:,1])+.05*np.sin(8*(p[:,0]+p[:,1]));rgb[terrain]=base[None,:]*tex[:,None]
    for j,obj in enumerate(SCENE['objects'],start=1):
        m=finite&(hit_id==j)
        if not np.any(m):continue
        p=points[m];base=_hex_rgb(obj['color']);tex=.82+.10*np.sin(8*p[:,0])+.08*np.cos(7*p[:,2]);rgb[m]=base[None,:]*tex[:,None]
    rgb=np.clip(rgb,0,255).astype(np.uint8).reshape(height,width,3)
    return rgb,best.reshape(height,width),points.reshape(height,width,3)


def detect_harris_features(rgb,depth,world_points,max_features=64,min_distance=3,percentile=86):
    """Detect image corners with a configurable density threshold."""
    gray=(.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2]).astype(float)/255.0
    ix=sobel(gray,axis=1,mode='nearest'); iy=sobel(gray,axis=0,mode='nearest')
    sxx=gaussian_filter(ix*ix,1); syy=gaussian_filter(iy*iy,1); sxy=gaussian_filter(ix*iy,1)
    response=(sxx*syy-sxy*sxy)-.045*(sxx+syy)**2
    valid=np.isfinite(depth)&(depth>0);vals=response[valid]
    if vals.size==0:return []
    threshold=max(float(np.percentile(vals,percentile)),1e-8)
    local=response==maximum_filter(response,size=2*min_distance+1,mode='nearest')
    ys,xs=np.nonzero(valid&local&(response>=threshold));order=np.argsort(response[ys,xs])[::-1];out=[]
    for idx in order:
        y=int(ys[idx]);x=int(xs[idx]);score=float(response[y,x])
        if any((x-p[0])**2+(y-p[1])**2<min_distance**2 for p in out):continue
        out.append((x,y,score,world_points[y,x].copy()))
        if len(out)>=max_features:break
    return out
