// Attitude ESKF diagnostics.
(() => {
  const baseDrawState=drawState;
  drawState=function(r){
    baseDrawState(r);
    const el=document.querySelector('#event');if(!el)return;
    const kind=r.estimator_kind || data?.[mode]?.meta?.estimator_kind || 'blend';
    if(kind!=='eskf'){
      el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Estimator</strong>: legacy blend`;
      return;
    }
    const sa=r.eskf_sigma_att_deg||[],sb=r.eskf_sigma_bias_dps||[],b=r.gyro_bias_dps||[],e=r.event||{};
    const fmt=(a,n=3)=>a.length? a.map(x=>Number(x).toFixed(n)).join(' / '):'—';
    const meas=Number.isFinite(e.measurement_sigma_deg)?` · obs σ ${e.measurement_sigma_deg.toFixed(2)}°`:'';
    el.innerHTML+=`<hr style="border:0;border-top:1px solid #dbe3ee;margin:9px 0"><strong>Attitude ESKF</strong>${meas}<br>`+
      `Attitude 1σ: ${fmt(sa,2)}°<br>`+
      `Gyro bias: ${fmt(b,3)} °/s · bias 1σ: ${fmt(sb,3)} °/s`;
  };
})();
