// Diagnostics for the Version 2 relative visual frontend.
(() => {
  const baseDrawCamera=drawCamera;
  drawCamera=function(r){
    baseDrawCamera(r);
    const v=r.visual_frontend;if(!v)return;
    const w=camera.clientWidth;
    cc.save();cc.fillStyle='rgba(7,21,38,.82)';cc.fillRect(w-222,42,210,72);cc.fillStyle='white';cc.font='11px system-ui';
    cc.fillText(`Visual tracks ${v.tracks ?? 0}`,w-212,59);
    cc.fillText(`Visual ΔR ${(v.visual_rotation_deg ?? 0).toFixed(3)}°`,w-212,76);
    cc.fillText(`IMU ΔR ${(v.imu_rotation_deg ?? 0).toFixed(3)}°`,w-212,93);
    cc.fillText(`track RMS ${(v.track_rms_deg ?? 0).toFixed(3)}°`,w-212,110);cc.restore();
  };

  const baseDrawState=drawState;
  drawState=function(r){
    baseDrawState(r);
    const v=r.visual_frontend;if(!v)return;
    const el=document.querySelector('#event');
    el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Relative Visual Constraint</strong><br>`+
      `Tracks: ${v.tracks ?? 0} · Visual ΔR: ${(v.visual_rotation_deg ?? 0).toFixed(3)}° · IMU ΔR: ${(v.imu_rotation_deg ?? 0).toFixed(3)}°<br>`+
      `Visual/IMU residual: ${(v.residual_deg ?? 0).toFixed(3)}° · Track RMS: ${(v.track_rms_deg ?? 0).toFixed(3)}°`;
  };
})();
