// Interactive experiment model for the browser viewer.
// It deliberately operates on the exported reference trajectory instead of
// pretending to rerun the Python estimator in JavaScript. Fault controls alter
// a transparent display-layer experiment so users can see the qualitative
// effect while preserving the Python simulation as the source of truth.

window.ExperimentModel = (() => {
  const defaults = {
    gyroBiasDps: 0,
    gyroNoiseDps: 0,
    cameraDropout: 0,
    rangeDropout: 0,
    cameraDelayMs: 0,
    rangeDelayMs: 0,
  };

  function seededNoise(i, scale) {
    if (!scale) return 0;
    const x = Math.sin((i + 1) * 12.9898 + 78.233) * 43758.5453;
    return (2 * (x - Math.floor(x)) - 1) * scale;
  }

  function apply(record, index, settings, sourceRecords) {
    const s = {...defaults, ...settings};
    const out = structuredClone(record);
    const t = record.t;
    const drift = s.gyroBiasDps * t;
    const n = seededNoise(index, s.gyroNoiseDps);
    out.display_est_euler = record.est_euler.slice();
    out.display_est_euler[2] += drift + n;

    const camShift = Math.max(0, Math.round(s.cameraDelayMs / 1000 * 200));
    const rangeShift = Math.max(0, Math.round(s.rangeDelayMs / 1000 * 200));
    if (record.event?.kind === 'camera' && camShift && sourceRecords[index - camShift]) {
      out.event = {...record.event, delayed_from_t: sourceRecords[index - camShift].t};
    }
    if (record.event?.kind === 'range' && rangeShift && sourceRecords[index - rangeShift]) {
      out.event = {...record.event, delayed_from_t: sourceRecords[index - rangeShift].t};
    }

    const camDrop = s.cameraDropout > 0 && ((index * 37) % 100) < s.cameraDropout;
    const rangeDrop = s.rangeDropout > 0 && ((index * 53) % 100) < s.rangeDropout;
    out.camera_dropped = camDrop;
    out.range_dropped = rangeDrop;
    return out;
  }

  return {defaults, apply};
})();
