# Version 3 — Relative Range Constraint

The estimator no longer assumes that the nearest range surface is a horizontal world floor.

Pipeline:

1. Convert each valid 8×8 range ray to sensor-frame metric 3-D points.
2. Select the nearest local support surface independently in consecutive frames.
3. Fit a robust local plane normal in each frame.
4. Use the change between those two measured normals as a relative **tilt-only** constraint.
5. Apply only the minimum correction needed to make the two normals agree in the world frame, preserving rotation about the observed normal.
6. Run nearest-neighbor ICP only as a health check; pair count, RMS and inferred translation reduce Range gain when geometry is unreliable.

The old `range_floor_normal()` helper remains only for legacy/debug tests and is not used by the Version 3 estimator path.

This deliberately preserves observability limits. A featureless flat plane provides Roll/Pitch information but does not determine Yaw about its normal, so Range never freezes or invents that unobservable component. Camera remains responsible for visual Yaw information when available.
