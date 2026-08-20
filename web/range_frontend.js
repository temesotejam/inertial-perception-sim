// Diagnostics for the Version 3 relative range frontend.
(() => {
  const baseDrawState=drawState;
  drawState=function(r){
    baseDrawState(r);
    const q=r.range_frontend;if(!q)return;
    const el=document.querySelector('#event');
    el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Relative Range Constraint</strong><br>`+
      `Pairs: ${q.pairs ?? 0} · Range ΔR: ${(q.range_rotation_deg ?? 0).toFixed(3)}° · IMU ΔR: ${(q.imu_rotation_deg ?? 0).toFixed(3)}°<br>`+
      `ICP RMS: ${(q.range_rms_m ?? 0).toFixed(3)} m · translation: ${(q.range_translation_m ?? 0).toFixed(3)} m · quality: ${(q.range_quality ?? 0).toFixed(2)} · gain: ${(q.range_gain_applied ?? 0).toFixed(3)}`;
  };

  const baseDrawDepth=drawDepth;
  drawDepth=function(r){
    baseDrawDepth(r);
    const q=r.range_frontend;if(!q)return;
    const host=document.querySelector('#depth')?.parentElement;
    if(!host)return;
    let box=host.querySelector('.range-front-diag');
    if(!box){box=document.createElement('p');box.className='eval-note range-front-diag';host.appendChild(box)}
    box.textContent=`Relative Range: ${q.pairs ?? 0} pairs · ΔR ${(q.range_rotation_deg ?? 0).toFixed(3)}° · ICP RMS ${(q.range_rms_m ?? 0).toFixed(3)} m · quality ${(q.range_quality ?? 0).toFixed(2)}`;
  };
})();
