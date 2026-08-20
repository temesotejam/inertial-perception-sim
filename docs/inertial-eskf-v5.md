# Version 5 — Full Inertial Navigation ESKF

Version 5 expands the fusion state from attitude-only to the standard inertial navigation nominal state:

`x = [p, v, q, b_g, b_a]`

with 15-dimensional error state:

`dx = [dp, dv, dtheta, db_g, db_a]`.

## Propagation

IMU specific force is bias-corrected, rotated into the world frame, gravity is added, and position/velocity are integrated. Orientation is propagated from bias-corrected gyro rate. Covariance propagates position/velocity/attitude and both sensor biases.

## External constraints

Version 5 deliberately keeps the existing relative frontends unchanged:

- Camera constrains relative 3-D orientation from tracked image bearings.
- Range constrains only relative tilt from temporal local-surface normals.
- Neither frontend directly observes position or velocity yet.

Therefore position and velocity drift under noisy IMU data. This is intentional and establishes the baseline for the next stage, where visual/range translation constraints can be added and compared.

## Accelerometer gravity pseudo-update

The full INS ESKF does not use `accelerometer direction = gravity direction` by default. During translational acceleration that assumption is invalid. A future stationary detector can selectively enable gravity/ZUPT-style updates.

## New translation scenario

`translation` adds smooth 3-D position, velocity, acceleration, and modest attitude motion. It is used to verify that an ideal IMU reproduces the trajectory while noisy IMU integration exhibits physically expected drift.
