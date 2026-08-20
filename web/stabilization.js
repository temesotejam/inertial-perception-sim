// Viewer stabilization layer: keep the last valid visual data visible and
// distinguish LIVE / STALE / DROPOUT / DISABLED instead of blanking panels.
(() => {
  const cache = {mode:null,camera:null,range:null};
  const badges = new Map();

  const style=document.createElement('style');
  style.textContent=`
    .panel-status{position:absolute;top:10px;right:10px;z-index:5;padding:4px 8px;border-radius:999px;font:700 10px/1.2 system-ui;letter-spacing:.04em;box-shadow:0 1px 3px rgba(0,0,0,.12);pointer-events:none}
    .panel-status.live{background:#dcfce7;color:#166534;border:1px solid #86efac}
    .panel-status.stale{background:#fef3c7;color:#92400e;border:1px solid #fcd34d}
    .panel-status.dropout{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5}
    .panel-status.disabled{background:#e5e7eb;color:#475569;border:1px solid #cbd5e1}
    .card.stable-panel{position:relative;overflow:hidden}
  `;
  document.head.appendChild(style);

  function addBadge(targetId,key){
    const el=document.querySelector(targetId); if(!el) return;
    const card=el.closest('.card'); if(!card) return;
    card.classList.add('stable-panel');
    const b=document.createElement('div'); b.className='panel-status disabled'; b.textContent='DISABLED'; card.appendChild(b); badges.set(key,b);
  }
  addBadge('#camera','camera'); addBadge('#denseDepth','dense'); addBadge('#depthError','error'); addBadge('#depth','range');

  function setBadge(key,state,ageMs=0){
    const b=badges.get(key); if(!b) return;
    b.className='panel-status '+state;
    if(state==='live') b.textContent='LIVE';
    else if(state==='dropout') b.textContent='DROPOUT · HOLD';
    else if(state==='disabled') b.textContent='DISABLED';
    else b.textContent=`STALE · ${Math.max(0,ageMs).toFixed(0)} ms`;
  }

  function modeFlags(){
    const m=data?.[mode]?.metrics||{};
    return {camera:!!m.camera_enabled,range:!!m.range_enabled};
  }
  function resetForMode(){cache.mode=mode;cache.camera=null;cache.range=null;}
  function cloneRecord(r){return {...r};}

  function stabilize(r){
    if(cache.mode!==mode) resetForMode();
    const flags=modeFlags(), out=cloneRecord(r);

    const camValid=flags.camera && !r.camera_dropped && r.camera_rgb_b64 && r.camera_timestamp!=null;
    if(camValid){
      cache.camera={
        timestamp:r.camera_timestamp, rgb:r.camera_rgb_b64, size:r.camera_render_size,
        features:r.camera_features, pose:r.camera_pose_euler
      };
      setBadge('camera','live');
    }else if(!flags.camera){
      cache.camera=null; setBadge('camera','disabled');
    }else if(cache.camera){
      const age=(r.t-cache.camera.timestamp)*1000;
      setBadge('camera',r.camera_dropped?'dropout':'stale',age);
      out.camera_rgb_b64=cache.camera.rgb; out.camera_render_size=cache.camera.size;
      out.camera_timestamp=cache.camera.timestamp; out.camera_features=cache.camera.features;
      out.camera_pose_euler=cache.camera.pose; out.camera_dropped=false;
    }else setBadge('camera',r.camera_dropped?'dropout':'stale',0);

    const rangeValid=flags.range && !r.range_dropped && Array.isArray(r.range_distances) && r.range_timestamp!=null;
    if(rangeValid){
      cache.range={
        timestamp:r.range_timestamp,dist:r.range_distances,overlay:r.range_camera_overlay,
        points:r.range_world_points,dt:r.range_camera_dt_ms
      };
      setBadge('range','live'); setBadge('dense','live'); setBadge('error','live');
    }else if(!flags.range){
      cache.range=null; setBadge('range','disabled'); setBadge('dense','disabled'); setBadge('error','disabled');
    }else if(cache.range){
      const age=(r.t-cache.range.timestamp)*1000;
      const state=r.range_dropped?'dropout':'stale';
      setBadge('range',state,age); setBadge('dense',state,age); setBadge('error',state,age);
      out.range_distances=cache.range.dist; out.range_camera_overlay=cache.range.overlay;
      out.range_world_points=cache.range.points; out.range_timestamp=cache.range.timestamp;
      out.range_camera_dt_ms=cache.range.dt; out.range_dropped=false;
    }else{
      const state=r.range_dropped?'dropout':'stale'; setBadge('range',state,0);setBadge('dense',state,0);setBadge('error',state,0);
    }
    return out;
  }

  const originalDrawCamera=drawCamera;
  drawCamera=function(r){originalDrawCamera(stabilize(r));};

  const originalDrawDepth=drawDepth;
  drawDepth=function(r){originalDrawDepth(stabilize(r));};

  // World range points should also hold rather than blink.
  const originalDrawWorld=drawWorld;
  drawWorld=function(r){originalDrawWorld(stabilize(r));};

  // Keep status labels correct even when a mode changes before fresh data arrives.
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{
    cache.mode=null;
  }));
})();
