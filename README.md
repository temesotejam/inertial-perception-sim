# inertial-perception-sim

**An IMU-centric, sensor-agnostic inertial perception simulator that makes estimation, observability, and drift correction visible.**

The current architecture is:

```text
                         Camera RGB
                            ↓
                     Harris + 2D tracks
                            ↓
                      relative rotation
                            │
                            ▼
IMU ───────────────→ 15-state INS ESKF ←──────── Range 8×8
200 Hz               p, v, q, bg, ba             │
  │                         ▲                     ├─ relative tilt
  └─ propagate all state    │                     └─ observable normal translation
                            │
                         covariance
```

The active full INS state is

```text
x = [p, v, q, b_g, b_a]
```

with 15-dimensional error state

```text
δx = [δp, δv, δθ, δb_g, δb_a].
```

## What the simulator currently demonstrates

- deterministic 3-D rotational and translational ground-truth motion
- raw 6-axis IMU simulation with gyro bias, accelerometer bias, and noise
- full inertial propagation of position, velocity, attitude, gyro bias, and accelerometer bias
- 15×15 ESKF covariance propagation
- synthetic RGB rendering from the shared 3-D scene
- Harris features detected from the exact displayed Camera image
- image-space temporal feature tracking and RANSAC-based relative Camera rotation
- generic 8×8 ray-based Range sensing
- relative Range tilt from temporal local-plane normals without a horizontal-floor prior
- metric Range translation only along the locally observable surface-normal direction
- IMU-only, IMU+Camera, IMU+Range, and all-sensors comparisons
- Ground Truth vs Estimate metrics and uncertainty diagnostics
- browser visualization for GitHub Pages
- deterministic regression tests in GitHub Actions

## Observability rules

The simulator deliberately avoids inventing constraints that a sensor cannot provide.

- **IMU** propagates all states but drifts.
- **Monocular Camera** currently supplies relative 3-D rotation; monocular translation/scale is not yet fused.
- **Range local plane** supplies relative tilt but does not claim Yaw about the observed plane normal.
- **Range translation** supplies only displacement along the observed local-plane normal. On a flat floor this mostly constrains height; motion along the floor remains unobserved until richer geometry or another frontend provides that information.

Renderer/world ground truth is never passed directly to the estimator. It is used only inside simulated sensors, visualization, and evaluation.

## Estimator options

The default is:

```python
estimator_kind="ins_eskf"
```

Regression baselines remain available:

```python
estimator_kind="attitude_eskf"
estimator_kind="blend"
```

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

`scripts/export_demo.py` generates:

- `web/data/demo.json` — the standard combined-motion viewer dataset
- `web/data/translation_eval.json` — translation/position-drift evaluation across fusion modes

## Viewer

The browser viewer exposes not only the final estimate, but also why corrections happened. Depending on the active frontend it shows:

- Ground Truth and estimated Roll/Pitch/Yaw
- estimated position and velocity
- ESKF 1σ for position, velocity, attitude, gyro bias, and accelerometer bias
- Camera image, tracked Harris features, Visual ΔR, residual, and track quality
- 8×8 Range data, Range tilt, ICP health metrics
- observable Range normal displacement, translational residual, measurement σ, and resulting position/velocity correction
- Dense Depth interpolation and Ground Truth depth error
- fault injection and event trace

## Design rules

1. **IMU is the time-axis backbone.** External observations correct inertial drift.
2. **Respect observability.** Do not update a state component just because an algorithm can numerically produce a value.
3. **No product names above the sensor-adapter boundary.**
4. **Ground truth is inaccessible to the estimator.**
5. **All measurements have physical timestamps.**
6. **Simple methods remain available as regression/educational baselines.**
7. **Simulation and future hardware should share the same generic measurement and estimator interfaces.**

See `docs/` for the architecture and frontend notes.
