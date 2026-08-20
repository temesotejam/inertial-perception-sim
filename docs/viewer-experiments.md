# Viewer experiments

The GitHub Pages viewer has two clearly separated roles:

1. **Python reference simulation** — the authoritative estimator result used for metrics and CI.
2. **Browser experiment layer** — a human-readable sensitivity/fault visualization applied to the exported reference trajectory.

The browser layer intentionally does **not** claim to rerun the Python estimator. This keeps the demonstration transparent while the estimator remains single-source-of-truth in Python.

## Interactive controls

- Gyro bias: shows accumulated yaw-like drift over time.
- Gyro noise: adds deterministic visible perturbation for repeatable demonstrations.
- Camera dropout: hides selected camera observations.
- Range dropout: hides selected range frames.
- Camera timestamp delay: marks camera updates as stale and shows the source timestamp.
- Range timestamp delay: marks range updates as stale and shows the source timestamp.

## Event trace

The viewer keeps a rolling trace of the current playback events:

`time | event kind | residual | applied correction | dropout/delay annotations`

The intent is to let a user pause and advance one event at a time while relating the estimator's `predict → observe → correct` cycle to the sensor views.

## Source of truth

Any quantitative claim about estimator performance must come from the Python simulation and GitHub Actions tests. Browser fault controls are educational sensitivity overlays until a future milestone ports the estimator core to a shared runtime (for example WebAssembly) or reruns simulations server-side.
