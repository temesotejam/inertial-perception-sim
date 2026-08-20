// Diagnostics for the relative range frontend and active estimator.
(() => {
  const baseDrawState=drawState;
  drawState=function(r){
    baseDrawState(r);
    const el=document.querySelector('#event');if(!el)return;
    const q=r.range_frontend;
    if(q){
      const weight=r.estimator_kind==='eskf'?`obs σ ${(q.measurement_sigma_deg ?? 0).toFixed(2)}°`:`gain ${(q.range_gain_applied ?? 0).toFixed(3)}`;
      el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Relative Range Constraint</strong><br>`+
        `Pairs: ${q.pairs ?? 0} · Range ΔR: ${(q.range_rotation_deg ?? 0).toFixed(3)}° · IMU ΔR: ${(q.imu_rotation_deg ?? 0).toFixed(3)}°<br>`+
        `ICP RMS: ${(q.range_rms_m ?? 0).toFixed(3)} m · translation: ${(q.range_translation_m ?? 0).toFixed(3)} m · quality: ${(q.range_quality ?? 0).toFixed(2)} · ${weight}`;
    }
    if(r.estimator_kind==='eskf'){
      const sa=r.eskf_sigma_att_deg||[],sb=r.eskf_sigma_bias_dps||[],b=r.gyro_bias_dps||[];
      const fmt=(a,n)=>a.length?a.map(x=>Number(x).toFixed(n)).join(' / '):'—';
      el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Attitude ESKF</strong><br>`+
        `Attitude 1σ: ${fmt(sa,2)}°<br>Gyro bias: ${fmt(b,3)} °/s · bias 1σ: ${fmt(sb,3)} °/s`;
    }
  };

  const baseDrawDepth=drawDepth;
  drawDepth=function(r){
    baseDrawDepth(r);
    const q=r.range_frontend;if(!q)return;
    const host=document.querySelector('#depth')?.parentElement;if(!host)return;
    let box=host.querySelector('.range-front-diag');if(!box){box=document.createElement('p');box.className='eval-note range-front-diag';host.appendChild(box)}
    box.textContent=`Relative Range: ${q.pairs ?? 0} pairs · tilt ΔR ${(q.range_rotation_deg ?? 0).toFixed(3)}° · ICP RMS ${(q.range_rms_m ?? 0).toFixed(3)} m · quality ${(q.range_quality ?? 0).toFixed(2)}`;
  };
})();
