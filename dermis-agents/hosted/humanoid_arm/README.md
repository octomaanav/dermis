# Humanoid Arm Touch Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for **dexterous humanoid hands and arms doing general household manipulation**.

Focus: grasp stability, fragile-object safety, contact timing across fingers.

Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip vibration, sensor saturation and noise, judges them against this application's limits, charts the result and recommends what to change.

Keywords: humanoid, arm, hand, dexterous, household, manipulation, optimus, figure

Example questions:
- Analyze episode_0040 for a humanoid arm
- How stable is the humanoid grasp on the egg?

