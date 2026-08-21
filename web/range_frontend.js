// Diagnostics for the relative Range frontend and full inertial estimator.
(() => {
  const baseDrawState=drawState;
  drawState=function(r){
    baseDrawState(r);const el=document.querySelector('#event');if(!el)return;const q=r.range_frontend;const eskf=String(r.estimator_kind||'').includes('eskf');
    if(q){
      const weight=eskf?`attitude obs σ ${(q.measurement_sigma_deg??0).toFixed(2)}°`:`gain ${(q.range_gain_applied??0).toFixed(3)}`;
      el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Relative Range Constraint</strong><br>`+
        `Tilt: ${(q.range_rotation_deg??0).toFixed(3)}° · ICP RMS ${(q.range_rms_m??0).toFixed(3)} m · ${weight}`;
      if(Number.isFinite(q.range_translation_displacement_m)){
        el.innerHTML+=`<br>Normal displacement: ${q.range_translation_displacement_m.toFixed(4)} m`+
          ` · residual ${(q.range_translation_residual_m??0).toFixed(4)} m · σ ${(q.range_translation_sigma_m??0).toFixed(4)} m`+
          ` · quality ${(q.range_translation_quality??0).toFixed(2)}`+
          `<br>Position correction ${(q.position_correction_m??0).toFixed(4)} m · velocity correction ${(q.velocity_correction_mps??0).toFixed(4)} m/s`;
      }
    }
    if(eskf){
      const sa=r.eskf_sigma_att_deg||[],sb=r.eskf_sigma_bias_dps||[],b=r.gyro_bias_dps||[],sp=r.eskf_sigma_pos_m||[],sv=r.eskf_sigma_vel_mps||[],p=r.est_position||[],v=r.est_velocity||[],ba=r.accel_bias||[];
      const fmt=(a,n)=>a.length?a.map(x=>Number(x).toFixed(n)).join(' / '):'—';const full=String(r.estimator_kind||'').includes('ins_')||p.length;
      el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>${full?'Full INS ESKF':'Attitude ESKF'}</strong><br>`+
        `Attitude 1σ: ${fmt(sa,2)}°<br>Gyro bias: ${fmt(b,3)} °/s · bias 1σ: ${fmt(sb,3)} °/s`;
      if(full)el.innerHTML+=`<br>Position: ${fmt(p,3)} m · 1σ ${fmt(sp,3)} m<br>Velocity: ${fmt(v,3)} m/s · 1σ ${fmt(sv,3)} m/s<br>Accel bias: ${fmt(ba,4)} m/s²`;
    }
  };
  const baseDrawDepth=drawDepth;
  drawDepth=function(r){
    baseDrawDepth(r);const q=r.range_frontend;if(!q)return;const host=document.querySelector('#depth')?.parentElement;if(!host)return;
    let box=host.querySelector('.range-front-diag');if(!box){box=document.createElement('p');box.className='eval-note range-front-diag';host.appendChild(box)}
    const tr=Number.isFinite(q.range_translation_displacement_m)?` · normal Δp ${q.range_translation_displacement_m.toFixed(4)} m · σ ${(q.range_translation_sigma_m??0).toFixed(4)} m`:'';
    box.textContent=`Relative Range: tilt ΔR ${(q.range_rotation_deg??0).toFixed(3)}° · ICP RMS ${(q.range_rms_m??0).toFixed(3)} m${tr}`;
  };
})();
