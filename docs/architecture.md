# Architecture

The simulator is deliberately sensor-agnostic above the driver/simulator boundary.

```text
Ground truth -> simulated sensors -> common observations -> frontends -> constraints -> estimator
                       ^                                              |
                       |-------------- estimated pose ----------------|
```

## Frames

- World: right-handed, +X forward at initial pose, +Y left, +Z up.
- Body: colocated with the IMU.
- Camera and range sensors are modeled as replaceable observation sources. Version 1 keeps their simulated extrinsics aligned with Body to make the fusion mathematics visible before adding calibration complexity.

## Version 1 estimator

Version 1 estimates orientation using quaternion/rotation composition. IMU gyro measurements propagate the state at high rate. Accelerometer gravity provides a weak roll/pitch correction. The visual frontend converts known-landmark image observations into an orientation constraint. The range frontend fits a floor plane and provides a gravity-aligned normal constraint.

This intentionally simple estimator remains when ESKF and optimization estimators are added later, so each method can be compared on identical sensor data.
