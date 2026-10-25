# Industrial Manipulator Touch Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for **industrial arms and grippers on production lines, where yield and repeatability matter**.

Focus: first-pass yield, repeatability across runs, headroom before failure, cycle time to stable hold.

Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip vibration, sensor saturation and noise, judges them against this application's limits, charts the result and recommends what to change.

Keywords: industrial, factory, production, line, assembly, manufacturing, cobot, gripper, yield

Example questions:
- What's our yield on cans and bottles with SafeGrip?
- Industrial review of episode_0010

