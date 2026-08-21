from __future__ import annotations
import base64,json
from pathlib import Path
import numpy as np
from .world import truth_at,SCENE
from .sensors import ImuSimulator,CameraSimulator,GridRangeSimulator
from .frontend import visual_relative_rotation,range_relative_rotation,range_normal_translation
from .estimator import AttitudeEstimator
from .eskf import AttitudeESKF
from .inertial_eskf import InertialESKF
from .math3d import rotation_error_deg

def _euler_deg(r):return r.as_euler('xyz',degrees=True).tolist()

def _range_to_camera_overlay(frame,cam,range_gt=None,camera_gt=None):
 out=[]
 for idx,r in enumerate(frame.rays):
  if not np.isfinite(r.distance) or r.confidence<=0:continue
  if range_gt is None or camera_gt is None:x,y,z=np.asarray(r.direction,float)*float(r.distance)
  else:
   pw=range_gt.position+range_gt.orientation.apply(np.asarray(r.direction,float)*float(r.distance));x,y,z=camera_gt.orientation.inv().apply(pw-camera_gt.position)
  if x<=.1:continue
  u=cam.cx-cam.fx*y/x;v=cam.cy-cam.fy*z/x
  if 0<=u<cam.width and 0<=v<cam.height:out.append([float(u),float(v),float(r.distance),idx])
 return out

def _make_estimator(kind,gt0):
 if kind=='blend':return AttitudeEstimator(initial_orientation=gt0.orientation)
 if kind in ('attitude_eskf','eskf6'):return AttitudeESKF(initial_orientation=gt0.orientation)
 if kind in ('ins_eskf','eskf'):return InertialESKF(gt0.position,gt0.velocity,gt0.orientation)
 raise ValueError(f'unknown estimator_kind: {kind}')

def _diag(est):
 if not hasattr(est,'P'):return {}
 d=np.maximum(np.diag(est.P),0)
 if len(d)==15:return {'eskf_sigma_pos_m':np.sqrt(d[:3]).tolist(),'eskf_sigma_vel_mps':np.sqrt(d[3:6]).tolist(),'eskf_sigma_att_deg':np.degrees(np.sqrt(d[6:9])).tolist(),'eskf_sigma_bias_dps':np.degrees(np.sqrt(d[9:12])).tolist(),'eskf_sigma_accel_bias':np.sqrt(d[12:15]).tolist()}
 return {'eskf_sigma_att_deg':np.degrees(np.sqrt(d[:3])).tolist(),'eskf_sigma_bias_dps':np.degrees(np.sqrt(d[3:6])).tolist()}

def run_simulation(scenario='combined',duration=10.,seed=42,camera_enabled=True,range_enabled=True,imu_hz=200,camera_hz=15,range_hz=15,estimator_kind='ins_eskf'):
 rng=np.random.default_rng(seed);imu=ImuSimulator(rng);cam=CameraSimulator(rng);rngs=GridRangeSimulator(rng);gt0=truth_at(0,scenario);est=_make_estimator(estimator_kind,gt0);dt=1/imu_hz;next_cam=next_range=0.;records=[]
 last_cam=last_range=last_cam_gt=last_range_gt=None;prev_cam=None;prev_cam_reference=None;last_visual=None;prev_range=None;prev_range_reference=None;prev_range_position=None;last_range_frontend=None
 for k in range(int(round(duration*imu_hz))+1):
  t=k*dt;gt=truth_at(t,scenario);est.propagate(imu.sample(gt));event={'kind':'imu','residual_deg':0.,'correction_deg':0.,'filter':estimator_kind};camera_sampled=range_sampled=False
  if camera_enabled and t+1e-9>=next_cam:
   last_cam_gt=gt;last_cam=cam.sample(gt);visual_prior=prev_cam_reference.inv()*est.orientation if prev_cam_reference is not None else None;vis=visual_relative_rotation(prev_cam,last_cam,cam,rotation_prior=visual_prior)
   if vis is not None and prev_cam_reference is not None:
    est.update_camera_relative(vis['rotation'],prev_cam_reference,vis['tracks'],vis['track_rms_deg']);event=est.last_event.copy();event['candidate_tracks']=vis.get('candidate_tracks',vis['tracks']);event['prior_used']=vis.get('prior_used',False);event['prior_kept_tracks']=vis.get('prior_kept_tracks',vis['tracks']);event['prior_residual_rms_deg']=vis.get('prior_residual_rms_deg',float('nan'));last_visual=event.copy()
   prev_cam=last_cam;camera_sampled=True;next_cam+=1/camera_hz
  if range_enabled and t+1e-9>=next_range:
   last_range_gt=gt;last_range=rngs.sample(gt);rr=range_relative_rotation(prev_range,last_range);rt=None;range_diag={}
   if prev_range is not None and prev_range_reference is not None and hasattr(est,'position'):
    rotation_hint=prev_range_reference.inv()*est.orientation;rt=range_normal_translation(prev_range,last_range,rotation_hint)
   if rr is not None and prev_range_reference is not None:
    est.update_range_relative(rr['previous_normal'],rr['current_normal'],prev_range_reference,rr['pairs'],rr['range_rms_m'],rr['translation_m'],float(np.degrees(rr['rotation'].magnitude())));event=est.last_event.copy();range_diag=event.copy();range_diag['range_prev_points']=rr['prev_points'];range_diag['range_current_points']=rr['current_points'];range_diag['range_observable']=rr['observable']
   if rt is not None and prev_range_position is not None and prev_range_reference is not None and hasattr(est,'update_range_translation'):
    est.update_range_translation(rt['normal_previous'],rt['displacement_m'],prev_range_position,prev_range_reference,rt['quality'],rt['plane_rms_m'],rt['normal_angle_deg']);event=est.last_event.copy();range_diag.update(event);range_diag['range_translation_observable']=rt['observable'];range_diag['range_translation_displacement_m']=rt['displacement_m']
   if range_diag:last_range_frontend=range_diag.copy()
   prev_range=last_range;range_sampled=True;next_range+=1/range_hz
  if camera_sampled:prev_cam_reference=est.orientation
  if range_sampled:
   prev_range_reference=est.orientation
   if hasattr(est,'position'):prev_range_position=est.position.copy()
  s=est.state(t);rec={'t':round(t,6),'true_euler':_euler_deg(gt.orientation),'est_euler':_euler_deg(s.orientation),'error_deg':rotation_error_deg(s.orientation,gt.orientation),'event':event,'estimator_kind':estimator_kind,'gyro_bias_dps':np.degrees(s.gyro_bias).tolist()}
  if hasattr(est,'position'):
   rec.update({'true_position':gt.position.tolist(),'est_position':s.position.tolist(),'position_error_m':float(np.linalg.norm(s.position-gt.position)),'vertical_position_error_m':float(s.position[2]-gt.position[2]),'true_velocity':gt.velocity.tolist(),'est_velocity':s.velocity.tolist(),'velocity_error_mps':float(np.linalg.norm(s.velocity-gt.velocity)),'accel_bias':s.accel_bias.tolist()})
  rec.update(_diag(est))
  if last_visual is not None:rec['visual_frontend']=last_visual.copy()
  if last_range_frontend is not None:rec['range_frontend']=last_range_frontend.copy()
  if k%max(1,int(imu_hz/20))==0:
   if last_cam is not None:
    rec['camera_timestamp']=float(last_cam.timestamp);rec['camera_pose_euler']=_euler_deg(last_cam_gt.orientation);rec['camera_render_size']=[last_cam.render_width,last_cam.render_height];rec['camera_rgb_b64']=base64.b64encode(last_cam.image_rgb.tobytes()).decode('ascii');rec['camera_features']=[[f.u,f.v,f.feature_id,f.confidence] for f in last_cam.features]
   if last_range is not None:
    rec['range_timestamp']=float(last_range.timestamp);rec['range_distances']=[None if not np.isfinite(r.distance) else r.distance for r in last_range.rays];rec['range_world_points']=[(last_range_gt.position+last_range_gt.orientation.apply(r.direction*r.distance)).tolist() for r in last_range.rays if np.isfinite(r.distance)]
    if last_cam is not None:rec['range_camera_overlay']=_range_to_camera_overlay(last_range,cam,last_range_gt,last_cam_gt);rec['range_camera_dt_ms']=float((last_range.timestamp-last_cam.timestamp)*1000.)
  records.append(rec)
 errors=np.array([r['error_deg'] for r in records]);metrics={'scenario':scenario,'duration_s':duration,'seed':seed,'camera_enabled':camera_enabled,'range_enabled':range_enabled,'estimator_kind':estimator_kind,'orientation_rmse_deg':float(np.sqrt(np.mean(errors**2))),'orientation_mae_deg':float(np.mean(np.abs(errors))),'orientation_max_deg':float(np.max(errors)),'final_gyro_bias_dps':np.degrees(est.gyro_bias).tolist()}
 if hasattr(est,'position'):
  pe=np.array([r['position_error_m'] for r in records]);ze=np.array([r['vertical_position_error_m'] for r in records]);ve=np.array([r['velocity_error_mps'] for r in records]);metrics.update({'position_rmse_m':float(np.sqrt(np.mean(pe**2))),'vertical_position_rmse_m':float(np.sqrt(np.mean(ze**2))),'velocity_rmse_mps':float(np.sqrt(np.mean(ve**2))),'final_position_error_m':float(pe[-1]),'final_vertical_position_error_m':float(ze[-1]),'final_velocity_error_mps':float(ve[-1]),'final_accel_bias':est.accel_bias.tolist()})
 return {'meta':{'imu_hz':imu_hz,'camera_hz':camera_hz,'range_hz':range_hz,'estimator_kind':estimator_kind,'state':'p,v,q,b_g,b_a' if hasattr(est,'position') else 'q,b_g','camera':{'width':cam.width,'height':cam.height,'fov_deg':cam.fov_deg,'render_width':cam.render_width,'render_height':cam.render_height,'feature_detector':'Harris','tracker':'2D patch + inertial-prior gating + rotation RANSAC','constraint':'relative_rotation_from_image_tracks'},'range':{'rows':rngs.rows,'cols':rngs.cols,'tilt_down_deg':rngs.tilt_down_deg,'tracker':'local-plane temporal normal + ICP health check','constraint':'relative_tilt + normal_translation','translation_observability':'local_plane_normal_only','absolute_floor_prior':False},'scene':SCENE},'metrics':metrics,'records':records}

def compare_modes(scenario='combined',duration=10.,seed=42,estimator_kind='ins_eskf'):
 modes={'imu_only':(False,False),'imu_camera':(True,False),'imu_range':(False,True),'all':(True,True)};return {name:run_simulation(scenario,duration,seed,c,r,estimator_kind=estimator_kind) for name,(c,r) in modes.items()}

def compare_estimators(scenario='combined',duration=8.,seed=42):return {kind:compare_modes(scenario,duration,seed,kind) for kind in ('blend','attitude_eskf','ins_eskf')}

def export_demo(path,scenario='combined',duration=10.,seed=42,estimator_kind='ins_eskf'):
 out=compare_modes(scenario,duration,seed,estimator_kind);Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(out,separators=(',',':')),encoding='utf-8');return out
