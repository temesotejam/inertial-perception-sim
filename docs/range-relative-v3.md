# Version 3 — Relative Range Constraint

The estimator no longer assumes that the nearest range surface is a horizontal world floor.

Pipeline:

1. Convert each valid 8×8 range ray to a sensor-frame metric 3-D point.
2. Match consecutive point clouds with deterministic nearest-neighbor ICP.
3. Keep mutual nearest pairs and robustly trim large residuals.
4. Estimate the rigid transform with Kabsch alignment.
5. Feed only the relative rotation into the attitude estimator.
6. Reduce range gain when pair count is low, ICP RMS is high, or inferred translation is large.

The old `range_floor_normal()` helper remains only for legacy/debug tests and is not used by the Version 3 estimator path.

This deliberately preserves observability limits: a featureless flat plane cannot determine rotation about its normal, while terrain relief and objects provide additional geometric constraints.
