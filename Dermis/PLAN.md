# Dermis Studio implementation plan

This plan follows the phases and acceptance criteria in the build brief. It is updated as work lands.

## Build order

### Phase 0 — Foundations (in progress)

1. Repair/verify the local Xcode toolchain and install XcodeGen.
2. De-risk the 3D viewport with RealityKit and Metal SDF spikes; record screenshots, frame/GPU timing, picking, pose complexity, and code size in an ADR.
3. Establish the XcodeGen app and local Swift packages.
4. Implement design tokens and native macOS navigation/window structure.
5. Implement shared data models, validity pipeline, and simulator.
6. Implement legacy CSV parsing and serial transport/hot-plug.
7. Specify Dermis Wire Protocol v1 and add codec tests.
8. Connect simulator and legacy hardware to the live UI.
9. Measure launch, drops, latency, CPU, memory, and rendering performance.

### Phase 1 — Live and record

- Production 3D glove, force overlays, callouts, track lanes, inspector, and status bar.
- Pod identification, rest check/tare, and visible validity flags.
- Crash-safe session packages, library, replay, CSV/NPZ export.
- Wire Protocol v1 in firmware and app with clock synchronization.

### Phase 2 — Calibrate and export

- Calibration suite, versioned profiles, reprocessing, contact/slip/friction features.
- Load-cell integration, MCAP/PDF exports, synchronized camera/audio.
- Python reader and LeRobot/RLDS exporters.

### Phase 3 — Product features

- Rig, QC, Twin, pose inputs, grasp recognition, presentation/demo composer.
- Streaming, dataset builder, updates, signing, notarization, and DMG.

### Phase 4 — New hardware

- IMUs, Earth-field correction, pose-dependent crosstalk, expanded pod layouts, and multi-taxel tiles.

## Package boundaries

- `DermisCore`: units, samples, validity, calibration-independent signal processing.
- `DermisDevice`: serial transport, protocol codecs, legacy CSV, simulator, replay, clock sync.
- `DermisStorage`: session packages, recovery, SQLite index.
- `DermisExport`: CSV/NPZ/MCAP/PDF/Python bridge.
- `DermisRender`: RealityKit hand, Metal plots, colormaps.
- `DermisStream`: Network.framework outputs and pose input.
- `App`: SwiftUI composition and macOS commands.

## Principal risks

1. 3D quality and rigging without a restricted-license hand model.
2. Current local Xcode installation is incomplete and cannot compile Swift.
3. Reliable serial hot-plug and no-reset behavior across ESP32-S3 reconnects.
4. Sustained recording and bounded memory at the measured 357.7 Hz array rate.
5. Accurate callout projection and live plots within CPU/GPU budgets.
6. Calibration quality depends on future load-cell/rig hardware and representative data.

## Current milestone

Create a compilable Phase 0 vertical slice: simulator → processing → four live pod readouts and instrument-grade shell, then validate it against real legacy CSV hardware.

