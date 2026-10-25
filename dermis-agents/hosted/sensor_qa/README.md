# Sensor Health & Calibration Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for **the Dermis sensors themselves: saturation, noise, dropped frames and calibration state**.

Focus: separating sensor faults from real grip failures, noise vs spec, calibration status.

Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip vibration, sensor saturation and noise, judges them against this application's limits, charts the result and recommends what to change.

Keywords: sensor, health, calibration, calibrate, noise, rail, saturation, imu, magnetometer, drift, quality, unreliable

Example questions:
- Are any readings unreliable in episode_0030?
- Sensor health check across the fleet

