// Diagnostics for the relative visual frontend.
(() => {
  const baseDrawCamera=drawCamera;
  drawCamera=function(r){
    baseDrawCamera(r);const v=r.visual_frontend;if(!v)return;const w=camera.clientWidth;
    const eskf=String(r.estimator_kind||'').includes('eskf');const weight=eskf?`obs σ ${(v.measurement_sigma_deg??0).toFixed(2)}°`:`gain ${(v.camera_gain_applied??0).toFixed(3)}`;
    cc.save();cc.fillStyle='rgba(7,21,38,.82)';cc.fillRect(w-236,42,224,92);cc.fillStyle='white';cc.font='11px system-ui';
    cc.fillText(`Tracks ${v.tracks??0} / ${v.candidate_tracks??v.tracks??0}`,w-226,59);cc.fillText(`Visual ΔR ${(v.visual_rotation_deg??0).toFixed(3)}°`,w-226,76);cc.fillText(`IMU ΔR ${(v.imu_rotation_deg??0).toFixed(3)}°`,w-226,93);cc.fillText(`track RMS ${(v.track_rms_deg??0).toFixed(3)}°`,w-226,110);cc.fillText(`quality ${(100*(v.visual_quality??0)).toFixed(0)}% · ${weight}`,w-226,127);cc.restore();
  };
  const baseDrawState=drawState;
  drawState=function(r){
    baseDrawState(r);const v=r.visual_frontend;if(!v)return;const el=document.querySelector('#event');const eskf=String(r.estimator_kind||'').includes('eskf');const weight=eskf?`Obs σ: ${(v.measurement_sigma_deg??0).toFixed(2)}°`:`Gain: ${(v.camera_gain_applied??0).toFixed(3)}`;
    el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Relative Visual Constraint</strong><br>Tracks: ${v.tracks??0}/${v.candidate_tracks??v.tracks??0} · Visual ΔR: ${(v.visual_rotation_deg??0).toFixed(3)}° · IMU ΔR: ${(v.imu_rotation_deg??0).toFixed(3)}°<br>Residual: ${(v.residual_deg??0).toFixed(3)}° · Track RMS: ${(v.track_rms_deg??0).toFixed(3)}° · Quality: ${(100*(v.visual_quality??0)).toFixed(0)}% · ${weight}`;
  };
})();
