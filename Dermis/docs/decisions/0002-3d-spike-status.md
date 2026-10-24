# ADR 0002: 3D renderer spike status

**Status:** Proposed; measurements blocked  
**Date:** 2026-10-22

## Context

The brief requires screenshot and performance comparison of a RealityKit rigged mesh and a custom Metal raymarched SDF before selecting the production renderer.

## Current evidence

The local Xcode 26.6 installation is missing a required `libclang.dylib`, so neither prototype can be compiled, launched, captured, or measured. Selecting a winner without those measurements would defeat the purpose of the spike.

## Provisional direction

RealityKit is the integration baseline because it provides native USD/USDZ assets, animation, materials, camera control, picking, and SwiftUI `RealityView`. A Metal SDF spike remains mandatory before this ADR is accepted.

## Required measurements

- Retina screenshots against the visual bar in the brief.
- FPS, GPU time, CPU use, picking accuracy, callout anchor projection, posing complexity, and source/binary size.
- 30 fps behavior for inactive windows and zero rendering while hidden.

