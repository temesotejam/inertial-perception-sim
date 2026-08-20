from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .types import GroundTruthState

# Fixed visual landmarks used by the generic monocular frontend.
LANDMARKS=np.array([[4,-1.8,.4],[4,-1,1.4],[4,-.2,.8],[4,.5,1.8],[4,1.2,.5],[4,1.8,1.3],[3,-1.3,2],[3.5,.9,2.2],[5,0,1.1]],float)

# The scene description is deliberately product-agnostic and JSON-friendly so
# the Python simulator and browser viewer can render the same world.
SCENE={
    "terrain":{"x_min":-2.0,"x_max":6.0,"y_min":-3.0,"y_max":3.0,"step":0.35,
               "amplitude":0.055,"wave_x":0.72,"wave_y":1.08,"bump":0.075,
               "bump_x":1.7,"bump_y":-0.55,"bump_sigma":0.75,
               "base_color":"#9aa77b"},
    "light":{"direction":[-0.55,-0.35,-1.0],"ambient":0.38,"diffuse":0.72},
    "objects":[
        {"type":"box","id":"red_box","center":[3.0,-1.25,0.48],"size":[0.85,0.75,0.9],"color":"#b95f50"},
        {"type":"box","id":"blue_crate","center":[4.35,1.15,0.36],"size":[0.95,0.75,0.65],"color":"#537fa5"},
        {"type":"pillar","id":"yellow_pillar","center":[3.65,0.15,0.72],"radius":0.28,"height":1.4,"color":"#c5a64a"},
        {"type":"box","id":"green_step","center":[1.85,1.55,0.22],"size":[1.15,0.72,0.38],"color":"#6f9270"},
    ],
}

def terrain_height(x: float, y: float) -> float:
    t=SCENE["terrain"]
    wave=t["amplitude"]*(0.58*np.sin(t["wave_x"]*x)*np.cos(t["wave_y"]*y)+0.27*np.sin(1.65*y+0.35*x))
    r2=(x-t["bump_x"])**2+(y-t["bump_y"])**2
    bump=t["bump"]*np.exp(-r2/(2*t["bump_sigma"]**2))
    return float(wave+bump)

def terrain_normal(x: float, y: float, eps: float=1e-3) -> np.ndarray:
    dzdx=(terrain_height(x+eps,y)-terrain_height(x-eps,y))/(2*eps)
    dzdy=(terrain_height(x,y+eps)-terrain_height(x,y-eps))/(2*eps)
    n=np.array([-dzdx,-dzdy,1.0]); return n/np.linalg.norm(n)

def _ray_box(origin,direction,obj,max_range):
    c=np.asarray(obj["center"],float); s=np.asarray(obj["size"],float); lo=c-s/2; hi=c+s/2
    inv=np.where(np.abs(direction)>1e-10,1.0/direction,np.inf)
    t0=(lo-origin)*inv; t1=(hi-origin)*inv
    tmin=float(np.max(np.minimum(t0,t1))); tmax=float(np.min(np.maximum(t0,t1)))
    if tmax<max(tmin,0.0) or tmin>max_range:return None
    return max(tmin,0.0)

def _ray_pillar(origin,direction,obj,max_range):
    c=np.asarray(obj["center"],float); r=float(obj["radius"]); h=float(obj["height"]); z0=c[2]-h/2; z1=c[2]+h/2
    ox,oy=origin[0]-c[0],origin[1]-c[1]; dx,dy=direction[0],direction[1]
    a=dx*dx+dy*dy; b=2*(ox*dx+oy*dy); cc=ox*ox+oy*oy-r*r
    if a<1e-12:return None
    disc=b*b-4*a*cc
    if disc<0:return None
    for t in sorted(((-b-np.sqrt(disc))/(2*a),(-b+np.sqrt(disc))/(2*a))):
        if 0<=t<=max_range:
            z=origin[2]+t*direction[2]
            if z0<=z<=z1:return float(t)
    return None

def _ray_terrain(origin,direction,max_range):
    # Height-field ray marching followed by bisection. This is intentionally
    # simple and deterministic; it is sufficient for the small generic ToF grid.
    prev_t=0.0; prev_f=origin[2]-terrain_height(origin[0],origin[1])
    for t in np.linspace(max_range/120,max_range,120):
        p=origin+t*direction; f=p[2]-terrain_height(p[0],p[1])
        if f<=0<prev_f:
            lo,hi=prev_t,float(t)
            for _ in range(12):
                mid=(lo+hi)/2; q=origin+mid*direction
                if q[2]-terrain_height(q[0],q[1])>0:lo=mid
                else:hi=mid
            return (lo+hi)/2
        prev_t,prev_f=float(t),float(f)
    return None

def ray_scene_distance(origin,direction,max_range=4.0):
    origin=np.asarray(origin,float); direction=np.asarray(direction,float); direction=direction/np.linalg.norm(direction)
    hits=[]
    t=_ray_terrain(origin,direction,max_range)
    if t is not None:hits.append((t,"terrain"))
    for obj in SCENE["objects"]:
        hit=_ray_box(origin,direction,obj,max_range) if obj["type"]=="box" else _ray_pillar(origin,direction,obj,max_range)
        if hit is not None:hits.append((hit,obj["id"]))
    return min(hits,key=lambda x:x[0]) if hits else (None,None)

def euler_profile(t: float, scenario: str):
    if scenario=="static": e=np.zeros(3)
    elif scenario=="roll": e=np.radians([20*np.sin(2*np.pi*.35*t),0,0])
    elif scenario=="pitch": e=np.radians([0,18*np.sin(2*np.pi*.28*t),0])
    elif scenario=="yaw": e=np.radians([0,0,35*np.sin(2*np.pi*.18*t)])
    else: e=np.radians([16*np.sin(2*np.pi*.31*t),12*np.sin(2*np.pi*.23*t+.4),28*np.sin(2*np.pi*.17*t+.8)])
    return e

def truth_at(t: float, scenario: str, eps: float=1e-4) -> GroundTruthState:
    r=Rotation.from_euler("xyz",euler_profile(t,scenario)); r2=Rotation.from_euler("xyz",euler_profile(t+eps,scenario))
    omega=(r.inv()*r2).as_rotvec()/eps
    return GroundTruthState(t,np.array([0.,0.,1.]),np.zeros(3),np.zeros(3),r,omega)
