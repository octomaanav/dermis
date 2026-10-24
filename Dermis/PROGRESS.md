# Dermis Studio progress

## Phase 0

### Working

- Existing ESP32-S3 firmware is preserved under `firmware/`.
- Four MLX90393 sensors have passed detection, configuration, isolation, and sustained acquisition at 1 MHz.
- Production app architecture and implementation plan are established.
- XcodeGen 2.46.0 is installed and `DermisStudio.xcodeproj` generates successfully from `project.yml`.
- Phase 0 vertical-slice source is present: design tokens, native app shell, simulator, four pod readouts, inspector, lane views, a RealityKit hand/pod spike, validity evaluation, and the legacy CSV parser.
- Unit tests cover analog limits, identical-axis junk frames, valid legacy rows, metadata, and truncation.

### Blocked verification

- Xcode 26.6 reports a missing `libclang.dylib`; `swift` and `xcodebuild` cannot run.
- The separate Command Line Tools install is also internally inconsistent/outdated and cannot link SwiftPM manifests.
- Consequently, Swift compilation/tests, launch, screenshots, UI review, and measured app performance are pending Apple toolchain repair.

### Hardware baseline

| Metric | Measured |
|---|---:|
| Full-array rate | 357.7 frames/s |
| Errors in final 10 s run | 0 |
| Saturation events | 0 |
| Reliable SPI clock | 1 MHz |

### Performance budget status

Not yet measurable. Target budgets remain: <30 ms latency, <12% live CPU, <2% idle CPU, <300 MB after one hour, <1 s cold launch, and <30 MB app bundle.

