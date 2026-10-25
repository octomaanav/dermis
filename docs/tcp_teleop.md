# TCP telemetry → simulated gripper (TELEOPERATION test)

This is an **input-integration test**. Glove-shaped telemetry arrives over TCP, and an
**EXPERIMENTAL** mapping turns it into commands for the simulated Franka Panda parallel-jaw gripper.
It is not neural-network training and not autonomous control. Every motion comes from received
telemetry and only happens while the operator has armed the controller.

```
sender (separate process)  --TCP, NDJSON-->  TelemetryServer (asyncio thread)
                                               bytes -> NDJSONFramer -> UTF-8/JSON -> JSON Schema
                                               -> adapter -> TelemetryPipeline -> LatestSlot (1 item)
                                                                                   |
PyBullet main thread, 30 Hz:  LatestSlot.get() -> TeleopController (mapping, watchdog, limits) -> env.step()
```

| Piece | File |
| --- | --- |
| TCP receiver, latest-frame handoff, counters | `src/dermis/server/tcp_server.py` |
| NDJSON framing (partial, combined, UTF-8, size limit) | `src/dermis/server/ndjson.py` |
| Schema validation (`firmware/v1/schemas/decoded_frame.schema.json`) | `src/dermis/server/validation.py` |
| Fixture/schema lookup (no hardcoded paths; `DERMIS_KIT_DIR` overrides) | `src/dermis/telemetry/kit.py` |
| Adapter, ordering, STALE/hold logic (shared with the other telemetry paths) | `src/dermis/telemetry/{adapter,pipeline}.py` |
| EXPERIMENTAL mapping (replaceable) | `src/dermis/server/teleop/mapping.py` (arm), `hand.py` (3-claw hand) |
| Draft-schema normalization | `src/dermis/server/compat.py` |
| Robot adapters (hand / arm) | `src/dermis/server/teleop/robots.py` |
| Operator calibration | `src/dermis/server/teleop/calibration.py` |
| Arm/disarm, watchdog, smoothing and limits | `src/dermis/server/teleop/controller.py` |
| Simulation loop, GUI, keys | `src/dermis/server/teleop/session.py` |
| Development sender, fault injection | `src/dermis/server/dev_sender.py` |
| Headless integration check | `src/dermis/server/teleop/pipeline_check.py` |

## Robots

`--robot hand` (the default) is a **human-like 3-finger hand** on a forearm: palm down, fingers pointing forward.

| Part | Driven by | Notes |
| --- | --- | --- |
| thumb (blue) | `thumb_tip` magnetic change | opposes toward the index; stops where it touches the fingers |
| index finger (orange) | `index_tip` magnetic change | 3 phalanges curl together; joints limited to 0–80°, 0–95° and 0–60°; never bends backwards |
| middle finger (purple) | `middle_tip` magnetic change | same |
| hand tilt (roll, pitch about the wrist) | wrist IMU gravity direction, falling back to `index_tip` | quasi-static only; no yaw |
| hand movement (e.g. a punch) | **EXPERIMENTAL** IMU motion bursts (`server/teleop/motion.py`) | see below |
| fingertip pads | that pod's magnetometer state | green fresh, yellow held/STALE, red unusable |
| lamp sphere behind the hand | controller state | green ARMED, orange PAUSED, red DISARMED |

- If one pod goes bad, only its own finger holds; the others keep moving.
- The fingers are posed kinematically. Joint angles come straight from the smoothed command, so no contact
  force can push a joint past its limit. The thumb is checked against the finger geometry every tick.
- `--robot arm` is the Franka Panda gripper described below.

### Moving the hand with the glove (EXPERIMENTAL)

The glove has accelerometers and gyros, which measure acceleration and rotation, never position.
Integrating acceleration twice drifts by metres within seconds, so this is **not position tracking**.
It reproduces short, quick motions like a punch and then returns the hand home:

- **Gravity removal:** orientation comes from the gyro and is re-levelled from gravity whenever the hand is
  still. Gravity is then subtracted from the accelerometer.
- **Integration:** the remaining acceleration is integrated into a velocity, which is zeroed whenever the hand
  is still. The velocity is integrated into an offset.
- **Return and limits:** once the hand is still, it springs back home. The offset is clamped to x −5…+30 cm,
  y ±12 cm and z −10…+12 cm around home.
- **Slow tilts never move the hand.** Readings within 5 % of 1 g count as no linear motion, and after a
  dropout the orientation is re-levelled at once.
- **Every frame counts.** The estimator runs on every received frame, not just the 30 Hz the sim samples.
  Send at **≥ 100 Hz** for motion; at 30 Hz a 250 ms punch is only about 7 samples.
- **Measured on synthetic data:** a 30 cm punch shows as about 28 cm at 100 Hz and about 26 cm at 30 Hz. It
  returns home within about 1 s, with up to 5 cm of recoil.
- **Real glove: assumptions to verify:**
  - The IMU chip's +x points along the fingers and +z out of the back of the hand.
  - The wrist IMU is preferred over the index IMU, because a fingertip IMU also rotates when the finger curls.
- `--no-motion` keeps the wrist in place (tilt only).

```sh
python -m dermis teleop --source tcp --gui --auto
python -m dermis send-test --mode synthetic --profile punch --rate 100     # fist + 30 cm punch every 2.5 s
```

## Run it

From the repository root, with the venv active (`source .venv/bin/activate`; setup is in the README).

Terminal 1 (receiver + simulator):

```sh
python -m dermis teleop --source tcp --host 127.0.0.1 --port 8765 --gui
python -m dermis teleop --source tcp --gui --tilt-xy      # also map index tilt to XY velocity
```

Terminal 2 (SYNTHETIC sender):

```sh
python -m dermis send-test --host 127.0.0.1 --port 8765 --mode synthetic --rate 30
```

Automated check (no window, ~20 s each):

```sh
python -m dermis test-tcp-pipeline --headless                      # 3-claw hand, v1 frames
python -m dermis test-tcp-pipeline --headless --format draft       # hand, proposed draft frames
python -m dermis test-tcp-pipeline --headless --robot arm          # Panda gripper
python -m pytest tests/test_tcp_*.py                      # unit tests + the same integration check
```

The same programs are also available as `python -m dermis.server.dev_sender ...`.

### WebSocket source bridge

The receiver above speaks TCP NDJSON. If the producer instead exposes a WebSocket stream, run the
bridge beside the receiver. Each WebSocket **text** message must be one decoded-frame JSON object;
the bridge immediately writes it as one NDJSON record to the local receiver (no frame batching).

```sh
# Terminal 1: existing Dermis receiver
python -m dermis teleop --host 127.0.0.1 --port 8765 --gui --auto

# Terminal 2: use the mock until the real URL is available
python -m dermis mock-ws-server --host 127.0.0.1 --port 8766 --path /frames --rate 100

# Terminal 3: replace --url with the real ws:// or wss:// endpoint later
python -m dermis websocket-bridge --url ws://127.0.0.1:8766/frames --tcp-host 127.0.0.1 --tcp-port 8765
```

For a real service, add `--subscribe '{"type":"subscribe","topic":"frames"}'` if it requires a
post-connect subscription message, and repeat `--header 'Authorization: Bearer TOKEN'` for required
handshake credentials. The bridge reconnects automatically after failures. Keep the TCP receiver on
`127.0.0.1`: it is only the local bridge that needs to reach it.

### Hands-free mode (`--auto`) and arm down/up (`--tilt-z`)

```sh
# Terminal 1: calibrates and arms by itself; pitch of the index pod moves the arm down/up
python -m dermis teleop --source tcp --gui --auto --tilt-z
# Terminal 2: SYNTHETIC up/down stream (gripper stays open, index pod pitches +-0.45 rad every 6 s)
python -m dermis send-test --mode synthetic --profile updown --rate 30
```

- **`--auto` is a development convenience.** It presses C and R for you:
  - It calibrates after 0.5 s of frames on a new connection. The sender must be still and unloaded
    then; the synthetic stream always is for its first 3 s.
  - It arms as soon as the input is usable.
  - After a disconnect it re-arms when a sender reconnects, and recalibrates first if the device or
    config changed.
  - Pressing **X** disarms and pauses auto mode until you press **R**.
  - With a real glove, hold the hand still when you connect.
- **`--tilt-z`** maps index-pod pitch to a bounded Z velocity and roll to Y.
  - It gives about ±0.04 m/s, with the arm staying inside z 0.10–0.35 m.
  - Expect the arm to sink about 8 cm and come back up every 6 s.
  - Stopping the stream freezes it where it is.
- **To save the up/down stream as a dataset and replay it:**

  ```sh
  python -m dermis send-test --profile updown --duration 30 --write data/synthetic_updown.jsonl
  python -m dermis send-test --mode replay --path data/synthetic_updown.jsonl
  ```

### Operator steps

The simulator starts **DISARMED**. Click the PyBullet window to give it the keyboard, or type the
letter and press Enter in Terminal 1.

1. Start the sender. The synthetic stream begins with 3 s of stationary, unloaded data.
2. **C**: calibrate. This captures 1 s of baseline. It needs fresh, usable index and thumb magnetometer
   readings (and the index IMU with `--tilt-xy`) and a steady signal. The result is saved to
   `runs/teleop/calibration-*.json`. Calibration is refused while armed, and it never re-runs on its own.
3. **R**: arm. This needs a calibration, a connected controller and usable input.
4. **X**: disarm. **E / Q** move Z up and down while armed; Z is otherwise fixed.
5. To stop, close the window or press Ctrl+C in Terminal 1, and press Ctrl+C in Terminal 2
   (`--duration` also ends the sender).

### What you should see

- **The lamp sphere** above the workspace shows the state. **Red** means DISARMED, **green** means
  ARMED and **orange** means PAUSED. The orange box outline is the teleop workspace limit.
- **The terminal** prints each state change immediately (with its reason), then one status line per
  second. The line shows connection state, source label (`SYNTHETIC`), receive rate, frames received,
  accepted and rejected, duplicates and out-of-order frames, the age of the latest usable control frame,
  `control_valid` and the measured opening/end effector. `--gui-text` also draws the state in the 3D
  view, at a frame-rate cost of about 15–25 ms per step.
- **While armed**, the fingers close and open smoothly every 4 s, following the synthetic magnetic
  change. With `--tilt-xy` the gripper also drifts slowly in X/Y inside the box.
- **Stopping the sender** (`--faults pause@10:3`, or Ctrl+Z in Terminal 2):
- Within 500 ms (the default for the measured 10 Hz Wi-Fi stream) the lamp turns orange,
  Cartesian motion stops and the opening holds. Set `--watchdog-ms` explicitly for another source.
  - The controller resumes when frames return.
  - Disconnecting turns the lamp red, and the controller needs **R** to re-arm.

## Wire format: newline-delimited JSON (NDJSON)

- Each message is one decoded-frame object in the shape of
  `firmware/v1/schemas/decoded_frame.schema.json`, as **compact UTF-8 JSON on a single line**
  terminated by `\n` (`\r\n` is accepted).
- **This is not the binary Dermis Wire protocol.** The current firmware speaks Dermis Wire v1
  (COBS/CRC) over USB, and nothing assumes it can send TCP JSON. An ESP32 TCP client, or a bridge,
  must produce this decoded form.
- **Each line is limited** to 16384 bytes by default (`--max-message-bytes`, env
  `DERMIS_TCP_MAX_MESSAGE_BYTES`). A compact four-pod frame is about 1.5 KB. A longer line is
  discarded up to its newline and counted, and the connection stays open.
- **Bad lines** are counted and skipped without closing the connection. These are malformed JSON,
  invalid UTF-8, a top-level value that isn't an object, schema violations and adapter errors.
  `NaN`/`Infinity`/overflowing numbers are accepted at parse time, counted (`non_finite_messages`),
  and make only the affected channel unusable.
- **Blank lines** are ignored and can be used as keep-alives.
- **Server to client:** on connect, the server sends one line, `{"type":"hello",...}`. A second
  client gets `{"type":"error","error":"controller_busy"}` and is closed. Only one controlling client
  is allowed at a time. Clients may ignore anything the server sends.
- **Timestamps:** `t_device_us` and the per-sensor `t_us` are the device clock, and `t_host_ns` must
  be `null` unless the sender has a real clock-sync fit. The receiver never invents `t_host_ns`.
  Stream age and the watchdog use only the host's monotonic receive time.

### If the frame schema changes

- **Contract:** `firmware/v1/schemas/decoded_frame.schema.json` is the contract, and it is never edited here.
- **Where shape knowledge lives:** only three places:
  - `server/compat.py`, which normalizes known draft shapes;
  - `telemetry/adapter.py`, which maps v1 to internal types;
  - the schema file.

  The pipeline, mappings, controller and simulator only see normalized telemetry.
- **The proposed draft shape is handled by `compat.py`:**
  - a pod without an `imu` key gets `imu: null`;
  - a bare `"wrist_imu": {...}` record becomes pod `wrist`;
  - a missing `flags` becomes `[]`;
  - a missing `valid` means **invalid**. It is never invented; pass `--assume-valid` to accept finite values instead.
  - Every frame that needed normalizing is counted (`normalized` in the status line), and the first one's notes are logged.
- **Strict mode:** `--strict-schema` turns normalization off and rejects anything that is not exactly v1.
- **Recording:** `--record` still stores the raw line exactly as received.
- **Test it:** `python -m dermis send-test --format draft` streams the draft shape, and
  `python -m dermis test-tcp-pipeline --format draft` runs the full check on it.
- **Cleaner for the firmware:** the v1 schema already allows the draft's content, with no changes on our side:
  - `"wrist": {"pod_id": "P-0005", "mag": null, "imu": {..., "flags": [], "valid": true}}`
  - `"middle_tip": {..., "imu": null}`

### How frames are judged

| Situation | Handling |
| --- | --- |
| `valid:false`, `null` values, `BAD_LENGTH`/`CHIP_ERROR`/`COMM_FAIL` | channel unusable (shown as missing, never interpolated) |
| `STALE` with `valid:true` | value is usable but old. It is held for at most 50 ms of device time, after which the channel is unusable and control pauses. |
| `TEMP_STALE` | affects only the temperature, never `b_ut`/`acc`/`gyr` freshness |
| Required pod missing / channel unusable | control pauses with the reason (e.g. `thumb_tip mag unusable (STALE too long)`) |
| Non-required pod invalid (e.g. the fixture's `thenar_apb` mag) | ignored by the mapping |
| Duplicate frame number | dropped, counted |
| Late frame (up to 64 frames back) | dropped, counted, never re-applied |
| Larger backward jump / `t_device_us` going backwards | treated as a stream restart or clock reset; held values are discarded |
| New TCP connection | ordering and holds reset. The controller **disarms** and needs a re-arm. |
| `config_id` change, `CONFIG_CHANGED`, other `device_id` or `pod_id` | calibration invalidated, controller disarms, recalibration required |

## Development sender

```sh
python -m dermis send-test --mode fixture                 # example frame, content unchanged, once
python -m dermis send-test --mode replay --path rec.jsonl --speed 1.0
python -m dermis send-test --mode replay --path firmware/v1/vectors/capture_2s.bin   # wire capture, decoded first
python -m dermis send-test --mode synthetic --rate 30 --duration 60    # add --no-tilt for a fixed gravity vector
python -m dermis send-test --mode synthetic --profile updown          # pitch only: arm down/up with teleop --tilt-z
python -m dermis send-test --profile updown --duration 30 --write data/synthetic_updown.jsonl   # save, don't send
python -m dermis send-test --mode synthetic --faults pause@10:3,malformed@5,burst@6:1
python -m dermis send-test --mode synthetic --faults ci   # the integration-test fault schedule
```

- **Fixture mode** sends `firmware/v1/examples/decoded_frame.json` with its content unchanged,
  re-serialized onto one line. Sending it again repeats frame 1234, which the receiver drops as a
  duplicate. Its constant values could not demonstrate motion anyway, so it only verifies schema and
  transport. The original fixtures are never modified.
- **Replay mode** sends recorded frames unchanged. It paces them by `t_device_us`, scaled by
  `--speed`, or by `--rate` if the recording has no usable timestamps.
  - There is **no real glove recording in the repository yet**.
  - `firmware/v1/vectors/capture_2s.bin` is output of the reference *simulated* hub.
  - To make a real recording, run `teleop --record runs/teleop/session.jsonl` while the glove or bridge
    streams. This writes the received lines exactly as sent, plus a `.recv.jsonl` sidecar with host
    receive times. The recording replays directly.
- **Synthetic mode** is clearly labelled **SYNTHETIC** development data built on the example frame's
  structure:
  - Pods, pod IDs, units and per-sensor timestamp offsets are kept.
  - Frame numbers increase monotonically, and the device timestamps are consistent with them.
  - `t_host_ns` is `null`.
  - The stream starts with a stationary calibration interval. After that, the index/thumb `b_ut.z`
    rises by up to 8000/7000 µT (pinch) on a 4 s cycle, and the accelerometer follows a coherent
    gravity vector under slow tilt.
  - `device_id` is `synthetic-dermis-dev`, so the receiver labels the source SYNTHETIC without any
    non-schema field.

  Synthetic data proves software integration only.
- **Faults** (`--faults name@t[:dur]`, in seconds from the start):

  | Fault | Effect |
  | --- | --- |
  | `malformed` | a truncated-JSON line and an invalid-UTF-8 line |
  | `fragment` | the next frame is written in three pieces |
  | `combine` | the next three frames go out in one write |
  | `duplicate` | the next frame is sent twice |
  | `reorder` | frames n and n+1 are swapped |
  | `oversize` | one line larger than the size limit |
  | `nonfinite` | the index accelerometer x value is `NaN` |
  | `burst` | 4× the rate for the window |
  | `null_index` | the index magnetometer is null/invalid for the window |
  | `stale_thumb` | the thumb magnetometer is repeated with `STALE` for the window |
  | `pause` | nothing is sent for the window; the device keeps counting frames |
  | `disconnect` | the connection closes and reconnects |
  | `reboot` | like `disconnect`, but the frame counter and device clock restart from 0 |

## Provisional mapping (EXPERIMENTAL)

- **Grip.** The mapping averages the baseline-subtracted change in `b_ut` along one chip axis over
  the selected pods (`--grip-pods`, `--grip-axis`, default z). That change is multiplied by
  `--grip-sign` and `--grip-gain-per-ut` (default 1/6000), with a 150 µT deadband and saturation at
  `--grip-saturation-ut`. The result is a closure from 0 to 1, which maps to a finger opening from
  80 mm down to 0.
- **Tilt** (`--tilt-xy`, optional). The index accelerometer's gravity direction, relative to the
  calibrated rest direction, becomes an XY *velocity*. The tilt angle is capped at 0.45 rad, which
  limits the velocity before the global limit applies. Samples whose |a| is more than ±15 % away from
  1 g give zero velocity.
- **Z** stays fixed, except while E/Q are held when armed.
- **Limits:** 0.05 m/s per axis (`--max-speed`) and 0.5 m/s² acceleration. The opening is
  rate-limited to 0.10 m/s and smoothed with low-pass filters. The workspace is x 0.40–0.65, y ±0.15
  and z 0.10–0.35 m, enforced on the integrated position target. The watchdog defaults to 500 ms
  (`--watchdog-ms`, env `DERMIS_WATCHDOG_MS`).
- **Limitations:**
  - The magnetic change is **not a calibrated force**.
  - Acceleration is **never integrated** into position.
  - The tactile magnetometers are **not a heading reference**.
  - The axis, sign and gain values are guesses until they are measured on the real glove.
  - The tilt mapping assumes the pod's chip axes and only works while the hand moves slowly.
  - The mapping is replaceable: implement `evaluate(snapshot, calibration) -> MappingOutput` and
    `required_channels()` (see `server/teleop/mapping.py`) without changing the server, pipeline or simulator.

## Listening on the LAN (ESP32 / another machine)

```sh
python -m dermis teleop --source tcp --host 0.0.0.0 --port 8765 --gui     # or DERMIS_TCP_HOST=0.0.0.0
```

`0.0.0.0` means "listen on every interface". It is **not** an address clients can connect to:
clients use the Mac's actual LAN IP. To find that IP without assuming an interface name:

```sh
ipconfig getifaddr "$(route -n get default | awk '/interface:/{print $2}')"   # IP of the default-route interface
ifconfig | awk '/inet /&&$2!="127.0.0.1"{print $2}'                           # every non-loopback IPv4 address
```

You can also open System Settings → Wi-Fi (or Network) → Details. The glove and the Mac must be on
the same network. Some guest or isolated Wi-Fi networks block client-to-client traffic.

### ESP32 client sketch (outline)

- Join the Wi-Fi and open a TCP client socket to `<Mac LAN IP>:8765`.
- Optionally read and ignore the server's hello line.
- For each frame, build the decoded JSON:
  - Use SI units, and set invalid values to `null` with `valid:false` and the flag names from
    SPEC §4.4.
  - Set `t_host_ns:null`.
  - Use the same `device_id`, `config_id` and pod IDs as `DEVICE_INFO`.
  - Write the JSON compactly, followed by `\n`.
- Send at the rate you want to control at; around 30–100 Hz is plenty. The receiver keeps only the
  newest frame, so sending faster than 30 Hz adds load without reducing latency.
- If writes start blocking, drop frames rather than queueing them on the ESP32.
- After a disconnect, reconnect with backoff. The operator must re-arm.
- If `controller_busy` comes back, another client is connected.

### Fallback: USB/serial → TCP bridge

If the firmware or network can't send TCP JSON, run a small bridge on the Mac that reads the USB
stream, converts it to decoded frames and connects to the receiver as the single TCP client.
- **Dermis Wire binary:** decode with the reference decoder in
  `firmware/v1/host/python/dermis_wire.py` (`StreamDecoder`, `DeviceInfo`, `decode_frame`).
- **Serial port already printing one JSON object per line:**
  `socat -u FILE:/dev/cu.usbmodemXXXX,b115200,raw TCP:127.0.0.1:8765` is enough, provided the lines
  match the schema.

## Troubleshooting

- **`cannot listen on 127.0.0.1:8765: Address already in use`.** Another receiver is running. Find it
  with `lsof -nP -iTCP:8765 -sTCP:LISTEN`, stop it, or use `--port 8766` on both sides.
- **Sender says `cannot connect`.** Start the receiver first. The sender retries for
  `--connect-timeout` seconds (15 s by default). Check that host and port match.
- **Sender says `controller_busy`.** Only one controlling client is allowed. Stop the other sender
  or bridge.
- **macOS firewall or local network access.** Loopback (`127.0.0.1`) is never blocked. When you
  listen on `0.0.0.0`, macOS may ask whether Python may accept incoming connections: allow it. The
  setting is under System Settings → Network → Firewall → Options. This tool never changes firewall
  settings. On recent macOS versions, apps that *connect out* to LAN devices may also need Privacy &
  Security → Local Network permission for the terminal.
- **Arm refused.** The terminal says why. Common reasons: not calibrated, the calibration was
  invalidated by a device or config change, no connection, or the input is not usable (STALE or
  invalid required channels, or the watchdog has fired).
- **Calibration failed: not steady.** Keep the hand still and unloaded. For noisy hardware, raise
  `ControllerConfig.calib_max_mag_std_ut` (80 µT by default).
- **The GUI is sluggish.** Leave `--gui-text` off. Try `--substeps 4`.
