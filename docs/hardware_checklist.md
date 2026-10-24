# Hardware integration checklist

How to move from the development sources (SYNTHETIC, REPLAY) to the assembled glove, and what still has to be measured before the glove can drive the task reliably. Each item says how to verify it with commands that exist in this repo.

## 1. Get frames into the shared pipeline

Every source feeds the same `TelemetryPipeline` (`src/dermis/telemetry/pipeline.py`). Swapping sources changes only the `--source` flag.

| Hub output | Command | Status |
|---|---|---|
| Decoded JSON, one frame per line, over USB serial | `python -m dermis stream --source serial --serial-port /dev/cu.usbmodemXXXX` | Unverified: no hardware. Uses `SerialSource` in `telemetry/sources.py`. |
| Dermis Wire v1 binary over USB serial | `python -m dermis stream --source serial --serial-port /dev/cu.usbmodemXXXX --wire` | Unverified on hardware. The decoder is verified against `capture_2s.bin` and `frame.bin`. The existing README notes the fake-hub PTY delivers no frames on macOS 27. |
| A bridge process that prints decoded JSONL | `bridge \| python -m dermis stream --source stdin` | Tested with SYNTHETIC lines. |
| A bridge sending decoded JSONL over TCP | `python -m dermis stream --source tcp --port 8765` | Tested with SYNTHETIC lines. The armed TCP teleop path is separate: see `docs/tcp_teleop.md`, owned by the TCP teleop work. |
| Recorded binary capture | `python -m dermis stream --source wire-file --path capture.bin` | Tested on `firmware/v1/vectors/capture_2s.bin`. |

If the final stream does not match `firmware/v1/schemas/decoded_frame.schema.json`, write a new adapter that returns `TelemetryFrame` (see `telemetry/adapter.py`) and pass it as `TelemetryPipeline(adapter=...)`. Nothing downstream changes.

- [ ] **Firmware conformance.** Run `firmware/v1/firmware/test_vectors.c` (or the same calls on the ESP32) and compare the output with `python firmware/v1/host/python/dermis_wire.py check-vectors`.
- [ ] **Pod map.** Check that DEVICE_INFO `pods[]` matches the physical mounting: tap each pad and watch which lane moves in `python -m dermis dashboard --source serial ...`.
- [ ] **Rates and gaps.** Stream for 60 s and read the final stats line: `lost_frames`, `duplicates`, `out_of_order`, `restarts`, `config_changes`. The specification expects about 350 frames/s.
- [ ] **Flags seen live.** Confirm that STALE, TEMP_STALE, BAD_LENGTH and RAIL show up as amber or red in the dashboard lanes, and not as zeros.
- [ ] **Clock sync.** `t_host_ns` is null until the host runs SYNC_PING fits (SPEC section 7). This repo does **not** implement clock sync yet. Watchdogs use host receive time only.
- [ ] **Record raw.** Use `--record raw.jsonl` on `stream`, or the dashboard's Record button. It writes the decoded frames as received, with flags and nulls kept.

## 2. Calibration that exists now (magnetic change, not force)

- [ ] **Unloaded baseline.** Run `python -m dermis calibrate --source serial --serial-port ... --frames 600 --out data/calibration/glove_v1.json` with nothing touching the pads. It uses fresh samples only and rejects pods with fewer than 50 samples or more than 50 µT noise. The result is tied to `device_id` and `config_id`. Any config change invalidates it, and the dashboard says so.
- [ ] **Mount transforms.** Fill `mount` in DEVICE_INFO (SET_CONFIG). `calibrate --source wire-file` picks the mounts up automatically. The live serial path does not yet pass DEVICE_INFO to calibrate.
- [ ] **Normalization scale.** `--scale-ut` (default 500 µT) is a placeholder. Choose it from a recorded firm pinch so that normalized deformation is about 1 at a firm grip.
- [ ] **Drift.** Re-capture the baseline after 10 minutes of wear and compare: temperature drift, glove slippage, and nearby ferromagnetic objects all move the baseline.

## 3. Measurements still needed before glove control is reliable

None of these exist yet. Until they do, the dashboard shows |ΔB| in µT and unitless deformation, never newtons.

1. **Force calibration per pod.** A load cell against each pad, with known loads (for example 0–10 N) in normal and shear directions, gives the ΔB → force map with hysteresis and repeatability. The protocol reserves FORCE_REF (0x12) for this.
2. **Crosstalk.** ΔB on neighbouring pods when one pad is pressed, and ΔB caused by finger flexion with no contact. Bending moves the magnet relative to the sensor, so pose-dependent offsets are likely.
3. **Earth field and pose.** The tactile magnetometers see Earth's field (about 50 µT) rotate with the hand. Measure ΔB during free hand rotation with no contact. If it is comparable to light touch, subtract it using the IMU orientation (SPEC section 8 recommends co-mounted IMUs).
4. **Temperature coefficient.** ΔB per °C over a warm-up from room temperature to skin temperature.
5. **Latency.** Glove event to sim command, measured end to end (for example a pad tap against a marker-button press). Needed to choose the watchdog timeout and the hold length.
6. **Hand position.** IMU acceleration is not integrated into position. XYZ hand motion needs an external tracker (camera or mocap) through `teleop.mapping.PoseProvider`. Until then, z (and optionally xy) comes from the keyboard.
7. **Mapping parameters.** The gains, deadbands and grip threshold of `ExperimentalTiltGripMapping` are guesses. Fit them on recorded sessions, and enable the mapping only with `--experimental-glove-teleop`.

## 4. Glove-as-expert demonstrations

The intended path is: a human teleoperates the simulated gripper with the glove, and the system records robot observations and the issued robot actions. The learned policy then needs only robot observations at run time (no glove).

- [ ] `python -m dermis demo --source glove --glove-input serial --serial-port ... --calibration data/calibration/glove_v1.json --experimental-glove-teleop` saves episodes with `source=glove`, the calibration ID, the telemetry source and config ID, and a per-tick status tag (ACTIVE or PAUSED_*).
- [ ] Before training, check what fraction of ticks were paused in each episode, using the `tags` array in the `.npz`.
- [ ] Train with `sources: [glove]` in the BC config. Glove episodes have no `expert_actions`, so the issued actions are the labels.
- [ ] Direct transfer from *natural* human manipulation, with no robot in the loop, is **not** supported. It needs hand/object tracking, retargeting from human hand to gripper, and alignment of human tactile signals with simulated contact. Raw glove telemetry alone does not provide these.
