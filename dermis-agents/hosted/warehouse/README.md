# Warehouse Pick & Place Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for **logistics picking of mixed items, from light balls to a 520 g bottle**.

Focus: no drops, slip on heavy items, place without impact.

Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip vibration, sensor saturation and noise, judges them against this application's limits, charts the result and recommends what to change.

Keywords: warehouse, logistics, pick, picking, place, sku, bin, fulfillment, amazon, package

Example questions:
- Which items do we drop most?
- Warehouse review of the bottle episodes

