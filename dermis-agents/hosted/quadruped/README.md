# Quadruped Foot-Contact Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for **legged robots (robot dogs) sensing foot contact, foot slip and touchdown impact**.

Focus: contact onset timing, slip detection from vibration, impact shocks, sensor saturation on touchdown.

Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip vibration, sensor saturation and noise, judges them against this application's limits, charts the result and recommends what to change.

Keywords: quadruped, dog, legged, foot, feet, paw, spot, unitree, gait, terrain

Example questions:
- Check slip detection for the robot dog
- Quadruped view of episode_0105

Note: No quadruped recordings yet, so this reads hand-glove recordings with the four pods mapped to four feet. Slip, impact and contact-timing signals carry over; force levels don't.

