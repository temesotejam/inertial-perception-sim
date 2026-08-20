// Edge-aware dense depth interpolation for the browser viewer.
// Uses sparse ToF→camera projections as metric depth anchors and the synthetic
// camera image as guidance. This is intentionally a transparent classical
// interpolation, not a learned monocular-depth model.
(() => {
  const dense = document.querySelector('#denseDepth');
  if (!dense) return;
  const dc = dense.getContext('2d');
  const guide = document.createElement('canvas');
  guide.width = 80; guide.height = 60;
  const gctx = guide.getContext('2d', {willReadFrequently:true});
  let lastDense = null;

  function resizeDense(){
    const r=dense.getBoundingClientRect(), d=devicePixelRatio||1;
    dense.width=r.width*d; dense.height=r.height*d;
    dc.setTransform(d,0,0,d,0,0);
  }
  function sampleGuide(){
    gctx.clearRect(0,0,80,60);
    gctx.drawImage(camera,0,0,80,60);
    return gctx.getImageData(0,0,80,60).data;
  }
  function rgbAt(img,x,y){
    x=Math.max(0,Math.min(79,Math.round(x))); y=Math.max(0,Math.min(59,Math.round(y)));
    const k=(y*80+x)*4; return [img[k],img[k+1],img[k+2]];
  }
  function colorDist(a,b){
    const dr=a[0]-b[0],dg=a[1]-b[1],db=a[2]-b[2]; return Math.sqrt(dr*dr+dg*dg+db*db);
  }
  function interpolate(r){
    if(r.range_dropped || !r.range_camera_overlay || !r.range_camera_overlay.length) return null;
    const img=sampleGuide();
    const anchors=r.range_camera_overlay.map(([u,v,d,idx])=>({
      x:u/640*80, y:v/480*60, d, idx, c:rgbAt(img,u/640*80,v/480*60)
    }));
    const out=new Float32Array(80*60); out.fill(NaN);
    const conf=new Float32Array(80*60);
    const sigmaS=11.0, sigmaC=42.0, maxRadius=24.0;
    let filled=0;
    for(let y=0;y<60;y++) for(let x=0;x<80;x++){
      const c=rgbAt(img,x,y); let ws=0, ds=0, nearest=1e9;
      for(const a of anchors){
        const dx=x-a.x,dy=y-a.y,rr=Math.hypot(dx,dy); if(rr>maxRadius) continue;
        nearest=Math.min(nearest,rr);
        const cd=colorDist(c,a.c);
        const w=Math.exp(-(rr*rr)/(2*sigmaS*sigmaS))*Math.exp(-(cd*cd)/(2*sigmaC*sigmaC));
        ws+=w; ds+=w*a.d;
      }
      const k=y*80+x;
      if(ws>0.08 && nearest<maxRadius){out[k]=ds/ws; conf[k]=Math.min(1,ws); filled++;}
    }
    return {depth:out,confidence:conf,coverage:filled/(80*60),anchors:anchors.length};
  }
  function depthRgb(d){
    const q=Math.max(0,Math.min(1,(d-.5)/3.5));
    const h=12+q*205, s=.86, l=.52;
    const a=s*Math.min(l,1-l), f=n=>{const k=(n+h/30)%12;return l-a*Math.max(-1,Math.min(k-3,9-k,1));};
    return [255*f(0),255*f(8),255*f(4)];
  }
  function paintDense(result){
    resizeDense(); const w=dense.clientWidth,h=dense.clientHeight;
    dc.clearRect(0,0,w,h);
    if(!result){dc.fillStyle='#eef2f7';dc.fillRect(0,0,w,h);dc.fillStyle='#64748b';dc.font='14px system-ui';dc.fillText('No range data',16,28);return;}
    const cellW=w/80,cellH=h/60;
    for(let y=0;y<60;y++) for(let x=0;x<80;x++){
      const k=y*80+x,d=result.depth[k]; if(!Number.isFinite(d)) continue;
      const [r,g,b]=depthRgb(d),a=.25+.75*result.confidence[k];
      dc.fillStyle=`rgba(${r|0},${g|0},${b|0},${a.toFixed(3)})`;
      dc.fillRect(x*cellW,y*cellH,cellW+1,cellH+1);
    }
    dc.fillStyle='rgba(15,23,42,.78)';dc.fillRect(10,10,235,38);
    dc.fillStyle='white';dc.font='12px system-ui';
    dc.fillText(`Dense depth 80×60 | anchors ${result.anchors}`,18,25);
    dc.fillText(`coverage ${(result.coverage*100).toFixed(1)}%`,18,41);
  }
  function overlayDense(result){
    const toggle=document.querySelector('#denseOverlayToggle');
    if(!toggle || !toggle.checked || !result) return;
    const w=camera.clientWidth,h=camera.clientHeight,cellW=w/80,cellH=h/60;
    cc.save();cc.globalAlpha=.20;
    for(let y=0;y<60;y++) for(let x=0;x<80;x++){
      const k=y*80+x,d=result.depth[k]; if(!Number.isFinite(d)) continue;
      const [r,g,b]=depthRgb(d);cc.fillStyle=`rgb(${r|0},${g|0},${b|0})`;cc.fillRect(x*cellW,y*cellH,cellW+1,cellH+1);
    }
    cc.restore();
  }

  // Capture the rendered RGB scene immediately before sparse RGB-D markers.
  const sparseOverlay=drawRgbdOverlay;
  drawRgbdOverlay=function(r){
    lastDense=interpolate(r);
    paintDense(lastDense);
    overlayDense(lastDense);
    sparseOverlay(r);
  };

  window.addEventListener('resize',()=>lastDense&&paintDense(lastDense));
})();
