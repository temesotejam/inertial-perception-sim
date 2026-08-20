# inertial-perception-sim

**An IMU-centric, sensor-agnostic perception simulator that makes sensor fusion visible.**

The core idea is a loop:

```text
IMU predicts motion
      ↓
Estimated pose corrects how external sensor data is interpreted
      ↓
Camera / range sensor observe the world
      ↓
External observations constrain the inertial estimate
      ↺
```

Version 1 uses three generic sensor classes: a raw 6-axis IMU, a monocular camera, and a grid range sensor (8×8 by default). No estimator code knows a product or vendor name.

## What Version 1 demonstrates

- deterministic 3-axis ground-truth motion
- noisy IMU propagation with gyro bias
- monocular landmark observations and visual orientation correction
- generic 8×8 ray-based range sensing and floor-plane correction
- IMU-only, IMU+Camera, IMU+Range, and all-sensors comparison
- Ground Truth vs Estimate metrics
- a browser viewer intended for GitHub Pages
- repeatable tests and regression thresholds in GitHub Actions

## Run locally

```bash
python -m pip install -e .[dev]
python scripts/export_demo.py
python -m http.server 8000 -d web
```

Then open `http://localhost:8000`.

Run tests with:

```bash
pytest -q
```

## Human-readable viewer

The viewer shows true and estimated orientation, live Roll/Pitch/Yaw, orientation error, camera feature locations, an 8×8 range depth map, world-frame range points, and the latest estimator residual/correction. Four fusion modes can be compared on identical simulated motion.

## Design rules

1. **IMU is the time-axis backbone.** External observations correct drift.
2. **No product names above the sensor adapter boundary.**
3. **Simulation and future hardware use the same observation types and estimator.**
4. **Ground truth is inaccessible to the estimator.** It is only used by simulated sensors, evaluation, and visualization.
5. **All measurements have timestamps.** Simulation time is independent from wall-clock time.
6. **Simple methods stay available.** Future ESKF/factor-graph implementations are compared rather than replacing the educational baseline.

See [`docs/architecture.md`](docs/architecture.md) for the frame and fusion model.
