# Version 4 — Attitude ESKF

Version 4 replaces the default hand-tuned blend fusion with a minimal error-state Kalman filter while keeping the Version 2/3 relative Camera and Range frontends unchanged.

## Nominal and error state

Nominal state:

- orientation `q`
- gyro bias `b_g`

Error state:

`dx = [dtheta, db_g]`

with a 6×6 covariance matrix `P`.

## Propagation

The nominal orientation is propagated from the bias-corrected gyro measurement. The linearized error dynamics couple attitude error and gyro-bias error. Gyro white noise and gyro-bias random walk increase covariance over time.

Accelerometer gravity is used only as a weak tilt observation; it does not directly constrain yaw.

## Camera update

The existing relative visual frontend provides the frame-to-frame 3-D rotation from image tracks. That relative rotation is composed with the previous fused orientation to form an attitude observation. Track count and track RMS set the observation covariance rather than a hand-tuned blend gain.

## Range update

The existing relative Range frontend provides only the observable local-surface tilt relation. The ESKF update projects the measurement Jacobian onto the plane perpendicular to the observed surface normal, deliberately leaving rotation about that normal unconstrained.

## Diagnostics

The simulator exports:

- estimated gyro bias in deg/s
- attitude 1σ in degrees
- gyro-bias 1σ in deg/s
- observation sigma for Camera/Range updates

The legacy `blend` estimator remains selectable for regression comparison with `estimator_kind='blend'`; the default is now `estimator_kind='eskf'`.
