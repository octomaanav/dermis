# Dermis → Mana: simulator-state stream

Files in this folder:

| File | What it is |
| --- | --- |
| `dermis_sim_stream.schema.json` | JSON Schema (draft 2020-12) for every message Dermis sends |
| `example_hand_checks.jsonl` | A real recording from the Dermis hand simulator: 1 `hello` + 420 `frame` messages at 30 Hz, running the scripted checks below |

The example was produced by `dermis.server.mana.export_check_stream()` and is checked by `tests/test_mana_stream.py`.

## What Mana needs to change

### 1. Connection and messages

- **Transport.** One JSON object per WebSocket text message. Recordings use the same objects, one per line (`.jsonl`).
- **First message.** The first message on every connection, and the first line of every recording, is
  `{"type": "hello", ...}` with the model. Rebuild the scene on every `hello`, including after a reconnect.
- **Following messages.** Every message after that is `{"type": "frame", ...}`, one per simulator
  control tick (30 Hz, about 2.2 KB each).
- **Version check.** Check `schema == "dermis.sim_stream/1"`. Ignore fields you don't know; later
  versions only add fields.
- **Default URL:** `ws://127.0.0.1:8766/dermis`. Port 8765 is already used by Dermis for glove input.
- **Bridge status.** The live Dermis bridge is not built yet. Until then, load
  `example_hand_checks.jsonl` as a recording.

### 2. Draw the Dermis hand, not a built-in one

- **Geometry.** Build it from `hello.model.links[].visuals`:
  - box: `size.x/y/z` are full extents;
  - sphere: `size.radius`;
  - cylinder: `size.length` + `size.radius`;
  - capsule: `size.length` + `size.radius`, where length is the straight segment between the two
    hemisphere centres, so the total length is length + 2 × radius;
  - cylinders and capsules lie along their local **+z** axis;
  - each visual's `pos` / `quat` / `rgba` are relative to its link frame.
- **Links.** The hand has a base (`index: -1`: palm + forearm) and 3 digits (thumb, index, middle).
  Each digit has 3 revolute joints and a fixed fingertip pad. Joint names are `thumb_CMC`, `thumb_MCP`,
  `thumb_IP`, `index_MCP`, `index_PIP`, `index_DIP`, `middle_MCP`, `middle_PIP` and `middle_DIP`.
- **Fewer digits than Mana's hand.** There are no ring or little fingers, so don't draw them.

### 3. Place each link from the frame; never compute or smooth motion

- **Poses come straight from the frame.** Set the base from `frame.base`, and each link from
  `frame.links[i]` (world pose of that link's frame). Mana needs no forward kinematics.
- **No interpolation, smoothing or prediction.** Render the latest frame exactly. If frames arrive late,
  show them late.
- **Charts use the frame's numbers:**
  - `joints[].q` is measured and `joints[].q_cmd` is commanded, both in radians;
  - `joints[].lower` / `upper` from `hello.model.joints` give the joint limits.
- **Measured below commanded is expected for the thumb.** It stops when it touches the other fingers,
  so its measured angle can stay below the command. That's correct, not an error.

### 4. Coordinates and units

- **Units:** metres, radians, newtons, seconds.
- **World frame:** right-handed, **+z up**, +x forward (the hand points along +x).
- **Converting for Three.js.** Three.js is +y up by default. Either set `camera.up = (0, 0, 1)`, or put
  everything under a root group rotated −90° about X.
- **Quaternions are `[x, y, z, w]`** (PyBullet order). Three.js `Quaternion(x, y, z, w)` takes the same
  order; don't treat element 0 as w.

### 5. Two different "stale" states

| Situation | What Dermis sends | What Mana should show |
| --- | --- | --- |
| Telemetry stopped or invalid (watchdog) | Frames keep arriving with `stale: true`; the pose is deliberately held; `source.telemetry_age_ms` grows | **STALE (telemetry)**, with the scene frozen as sent |
| Simulator or connection gone | No frames; the socket may close | **DISCONNECTED** after about 250 ms without a frame (7 frame periods), then automatic reconnect |

Mana's current STALE logic only covers the second row.

### 6. Labels, timestamps and drops

- **Source label.** Show `frame.source.label` exactly: SYNTHETIC, SCRIPTED, REPLAY, GLOVE,
  TCP-UNVERIFIED or NO SOURCE. Only GLOVE means the real glove. Replace Mana's fixed
  "SYNTHETIC PREVIEW" banner with this field.
- **Control mode.** Show `frame.control.mode` and `state`: DISARMED, TELEOPERATION, AUTONOMOUS or
  SCRIPTED_TEST.
- **Timeline.** Use `t_sim_s` (simulation time) for scrubbing and for comparing against simulator logs.
  It's the "source timestamp". Don't mix in browser time.
- **Dropped frames.** `seq` increases by 1 per frame on a connection. A gap means dropped frames: count
  them, don't fill them in.

### 7. Objects and contact may be absent

- **The hand simulator has no object.** `objects` and `contacts` are empty arrays, and
  `hello.model.objects` is `[]`.
- **Say so instead of drawing zero contact.** Show "no object in this simulation", not a zero-contact
  chart.
- **Grasp checks need an object.** The "grasp / lift / drop" checks only apply to streams that include
  an object (e.g. the assistive-hand demo, later).
- **Contact forces are simulated.** `contacts[].normal_force_n` is a simulated force, never measured on a
  person.

## Checks with `example_hand_checks.jsonl`

| `t_sim_s` | What the simulator did | Mana should show |
| --- | --- | --- |
| 0–1 | rest | still hand |
| 1–3 | palm +10 cm along x, then back | whole hand moves along x only |
| 3–5 | palm +8 cm along y, then back | whole hand moves along y only |
| 5–7 | palm +8 cm along z, then back | whole hand moves along z only |
| 7–9 | wrist roll +0.4 rad about the hand's x axis, then back | all fingers rotate with the palm |
| 9–11 | index finger flex 0 → 1 → 0 | only index links move; q tracks q_cmd |
| 11–13 | all fingers make a fist, then open | thumb's q may stay below q_cmd (stopped by fingers) |
| 13–14 | telemetry stale | frames keep coming, nothing moves, `stale: true` → STALE banner |

These are verified against the simulator by `tests/test_mana_stream.py`. For example, during 1–3 s the
palm moves 0.10 m along x and 0.000 m along y and z.
