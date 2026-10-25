# Build prompt: Dermis Rehab Simulator (single-file HTML)

You are a senior front-end and real-time graphics engineer. Build **one self-contained HTML file** called `dermis-rehab-sim.html`. It connects to a WebSocket that streams data from **Dermis**, a sensing glove with magnetic-elastomer pressure pads. It renders the user's hand-rehab activity in real time as a high-definition pressure visualization, and it counts and scores rehab exercises.

Read the whole spec before writing code. Accuracy to the data format and honesty about what is measured matter more than flourish.

---

## 1. Deliverable and constraints

- **One `.html` file.** Inline CSS and JS. No build step and no framework.
- **External scripts:** load them only from `cdnjs.cloudflare.com`. Three.js r160+ is allowed; nothing else is required. **Fonts:** load them only from Google Fonts. **Optional pieces:** if Three.js fails to load, the 2D views must still work.
- **Target browsers:** current Chrome and Safari on a 1440×900 laptop, and also when mirrored to a projector.
- **Canvases:** size every canvas by `devicePixelRatio`, so lines and gradients stay crisp on Retina screens.
- **Rendering:** use `requestAnimationFrame` at 60 fps, decoupled from the data rate. Data arrives anywhere from 10 Hz to 400 Hz.
- **Robustness:** the page must never throw an uncaught error. Malformed messages are counted and logged, never fatal.
- **Code structure:** keep it readable, with clearly separated modules inside one `<script>`: `Transport`, `Parser`, `Calibration`, `SignalPipeline`, `ExerciseEngine`, `Renderer2D`, `Renderer3D`, `Timeline`, `UI`, `Exporter`.

---

## 2. What the hardware is (context)

- **Pressure pods:** the glove has 4 pressure pods. Each pod is an MLX90393 3-axis magnetometer under a magnetized silicone pad. Pressing the pad moves the magnet, the field vector `b_ut` changes, and the firmware turns that change into force.
- **IMU:** there is 1 BMI160 IMU (accelerometer + gyroscope).
- **Pod locations in this capture:**
  | Channel | Location key | Finger | Pad area (cm²) |
  |---|---|---|---|
  | 0 | `palm_center` | palm | 2.2 |
  | 1 | `middle_tip` | middle fingertip | 1.0 |
  | 2 | `thumb_tip` | thumb tip | 1.1 |
  | 3 | `index_tip` | index fingertip | 1.0 |
- **Do not hard-code this mapping.** Read pods from each frame by location key and from the calibration file. Other location keys that may appear: `thenar_apb`, `ring_tip`, `little_tip`, `hypothenar`, `hand_dorsum`, `wrist`, `unassigned`. Any pod count from 1 to 12 must work.

**Honesty rule ("high definition" without inventing data).** There are only 4 measured points. "High definition" here means:
- smooth, physically plausible interpolation
- crisp rendering
- 60 fps motion
- rich derived signals: shear, slip, contact patch size, steadiness

It does not mean fake taxels. Show the sensed regions at full intensity. Draw spread or blur outside a pad as a visibly softer "estimated" layer, and include a toggle that shows measured points only. Regions with no sensor, such as the ring and little fingers, render as neutral and are labelled "not sensed".

---

## 3. Data input

### 3.1 WebSocket (primary)

- **Connection controls (top bar):**
  - a **text input for the WebSocket URL**, default `ws://localhost:8765`, saved to `localStorage` inside try/catch
  - a **Connect / Disconnect** button
  - a status pill with these states: Disconnected / Connecting / Live / Reconnecting / Error
- **Reconnects:** retry automatically with exponential backoff (0.5 s → 8 s max) until the user presses Disconnect.
- **Show live:** messages/s, frames/s, last-frame age in ms, dropped frames (from gaps in `frame`), and parse errors.
- **Accepted message payloads** (detect the shape automatically):
  1. one decoded frame JSON object (schema in §4)
  2. a JSON array of frames
  3. newline-delimited JSON, with several frames per message
  4. a derived-row object whose keys match the CSV header in §4.3 (`palm_force_n`, …)
  5. control objects: `{"kind":"dermis-calibration", ...}` (the same shape as the calibration file), `{"type":"marker"}`, `{"type":"device_info", ...}`. Apply calibration objects immediately.
  6. binary messages: ignore them, count them, and log `binary frames not supported in v1`.

### 3.2 Replay (essential for demos; the live hardware may fail)

- **File loading:** drag and drop (or use a file picker for) a raw `.jsonl` capture, a derived `.csv`, or a calibration `.json`.
- **Transport bar:** play/pause, speed (0.25×, 0.5×, 1×, 2×, 4×), loop, and a scrubber with a timecode.
- **Timing:** always come from `t_device_us`, never from message arrival time. Replay keeps the original timing, gaps included.
- **One code path:** replay feeds the exact same pipeline as the WebSocket.

### 3.3 Built-in demo generator

Add a "Demo" source. It synthesizes realistic frames at 10 Hz from scripted rehab sets (pinch reps, tripod reps, grip holds, an isolated middle-finger press, and a slip event), using the field layout from §4. Use it so the app can be tested with no files and no device.

---

## 4. Exact data formats

### 4.1 Raw decoded frame (one per WebSocket message or per JSONL line)

A real frame from the capture, unchanged:

```json
{"schema_version":1,"device_id":"DERMIS-74DFAEB1E590","frame":2515,"t_device_us":1025216402,"t_host_ns":null,
 "config_id":"dermis-4mlx-1imu-force-zero-span-v3","flags":[],
 "pods":{
  "palm_center":{"pod_id":"P-0001","mag":{"ch":0,"b_ut":[963.603,-2191.536,10043.97],"temp_c":23.5177,"load_signal_ut":-38.58268,"force_n":0,"force_raw_n":-0.01226,"force_reference_n":5,"force_valid":true,"t_us":1025218705,"status":35,"flags":["FORCE_CAL_READY"],"valid":true}},
  "middle_tip":{"pod_id":"P-0002","mag":{"ch":1,"b_ut":[-519.048,-648.8101,-5475.008],"temp_c":25.73009,"load_signal_ut":498.3502,"force_n":0.301508,"force_raw_n":0.301508,"force_reference_n":5,"force_valid":true,"t_us":1025218841,"status":3,"flags":["FORCE_CAL_READY"],"valid":true}},
  "thumb_tip":{"pod_id":"P-0003","mag":{"ch":2,"b_ut":[2357.343,-372.465,8928.832],"temp_c":23.20796,"load_signal_ut":17.73918,"force_n":0,"force_raw_n":0.002484,"force_reference_n":5,"force_valid":true,"t_us":1025218966,"status":3,"flags":["FORCE_CAL_READY"],"valid":true}},
  "index_tip":{"pod_id":"P-0004","mag":{"ch":3,"b_ut":[-790.587,-216.27,18093.86],"temp_c":24.64602,"load_signal_ut":11343.32,"force_n":3.886708,"force_raw_n":3.886708,"force_reference_n":5,"force_valid":true,"t_us":1025219090,"status":3,"flags":["FORCE_CAL_READY"],"valid":true}}},
 "imu":{"imu_id":"IMU-0001","ch":0,"acc_mps2":[7.327454,5.911284,1.893215],"gyr_rads":[-0.140478,0.080881,0.056404],"temp_c":24.71094,"t_us":1025219224,"sensortime":9281697,"flags":[],"valid":true}}
```

**Field meanings:**

| Field | Meaning / handling |
|---|---|
| `frame` | Device frame counter. A gap greater than 1 means frames were lost; count them. |
| `t_device_us` | Device clock in µs. **This is the only timebase.** |
| `t_host_ns` | Currently always `null`; ignore it. |
| `pods.<loc>.mag.b_ut` | Field vector in µT, chip axes |
| `load_signal_ut` | The field change projected onto the pod's calibrated press axis (µT) |
| `force_n` | Calibrated normal force in newtons, after the deadband and clamp. **`null` while zeroing.** |
| `force_raw_n` | Force before the deadband and clamp. It can be negative. |
| `force_valid` | `false` while zeroing |
| `flags` (mag) | `FORCE_CAL_READY` (normal), `FORCE_ZEROING` (re-tare in progress, force null), `FORCE_CLAMPED_HIGH` (saturated at 9.80665 N = 1 kgf), `FORCE_CLAMPED_LOW` (raw force well below zero, explained in §5) |
| `valid` | `false` → treat this pod as missing for this frame. Never interpolate silently. |
| `temp_c` | Pad temperature. Small rises (~0.5–1 °C) happen with sustained skin contact. |
| `imu.acc_mps2` | Specific force, m/s². Reads about +9.81 on the upward axis when still. |
| `imu.gyr_rads` | Angular rate, rad/s |
| `imu.sensortime` | **Do not use for timing.** It resets at random in this capture. |

The format can also arrive with `imu` inside each pod (`pods.<loc>.imu`). Support both a top-level `imu` and per-pod IMUs.

### 4.2 Calibration file (`dermis-calibration-*.json`)

The complete real file:

```json
{"kind":"dermis-calibration","version":1,"device_id":"DERMIS-74DFAEB1E590",
 "assignments":{"0":"palm","1":"middle","2":"thumb","3":"index"},
 "areas":{"palm":2.2,"middle":1,"thumb":1.1,"index":1},
 "references":{"0":5,"1":5,"2":5,"3":5},
 "baseline":{
  "0":{"b":[965.236,-2200.404,10070.901],"noiseUt":33.39,"temp":23.54,"fieldRateFloorUtS":368.47,"forceRateFloorNS":0.2},
  "1":{"b":[-608.937,-506.902,-4979.587],"noiseUt":23.56,"temp":25.59,"fieldRateFloorUtS":220.08,"forceRateFloorNS":0.2},
  "2":{"b":[2362.592,-398.941,8897.496],"noiseUt":24.94,"temp":23.10,"fieldRateFloorUtS":148.73,"forceRateFloorNS":0.2},
  "3":{"b":[-203.305,-361.186,6651.805],"noiseUt":14.66,"temp":24.53,"fieldRateFloorUtS":152.42,"forceRateFloorNS":0.2},
  "imu":{"acc":[9.226056,-1.096544,3.192668],"gyr":[-0.034587,0.003725,-0.022881],"roll":-0.33083,"pitch":-1.22004}},
 "firmware_spans":[
  {"ch":0,"unit":[-0.094715,-0.625167,0.774723],"gain_n_per_ut":0.000318,"reference_n":5},
  {"ch":1,"unit":[-0.133947,-0.175527,-0.97532],"gain_n_per_ut":0.000605,"reference_n":5},
  {"ch":2,"unit":[0.6471,-0.095134,0.756446],"gain_n_per_ut":0.00014,"reference_n":5},
  {"ch":3,"unit":[-0.066627,0.150002,0.986438],"gain_n_per_ut":0.000343,"reference_n":5}]}
```

**Embedding rules:**
- Embed this as the **default calibration**, so the app works with no file loaded.
- Match the calibration's `assignments` (by channel) to the frame's pods using `mag.ch`.
- If the calibration's `device_id` doesn't match the stream, show a warning.

### 4.3 Derived CSV (`dermis-derived-*.csv`, host-software output)

**Header (42 columns):**
`frame,t_device_us,` then, for each of `palm, middle, thumb, index`: `<f>_source,<f>_force_n,<f>_pressure_kpa,<f>_temp_c,<f>_temp_delta_c,<f>_slip_score,<f>_contact,<f>_field_delta_ut,<f>_load_signal_ut`, then `roll_deg,pitch_deg,yaw_deg,motion_score`.

**Value meanings:**
- `slip_score` is 0–100.
- `contact` is 0/1.
- `source` is `S1`…`S4`.
- Empty force cells mean zeroing (null).
- **Derived rows already contain processed values.** Use them directly, and skip the recomputation steps they already cover.

---

## 5. Facts from the real capture (use them for defaults and edge cases)

The capture has 628 frames, about 63 s, frames 2401–3028.

**Timing and zeroing**
- **Rate:** about 10 Hz. Median dt is 100.0 ms; jitter goes from 6.7 ms to 302 ms (one 300 ms gap at frame 2781).
- **Design for 10–400 Hz.** The firmware spec allows about 350 Hz.
- **Zeroing:** frames 2402–2430 (~3 s at the start) carry `FORCE_ZEROING`, with `force_n` / `force_raw_n` / `load_signal_ut` = `null`. Show a "Calibrating — keep hand relaxed" state with a progress shimmer. Don't count reps.

**Force behaviour**
- **Deadband:** `force_n` is 0 while `force_raw_n` is below about **0.08 N** (for example, raw 0.077 → 0, raw 0.081 → 0.081). Use 0.08 N as the default noise floor.
- **High saturation:** the middle fingertip hits `FORCE_CLAMPED_HIGH` at 9.80665 N (frame 2502; the raw value was 10.13 N). Render saturation distinctly, and never report it as "9.8 N exactly"; report it as "≥ 9.8 N (saturated)".
- **`FORCE_CLAMPED_LOW` is not zero pressure.** On the palm (frames 2580–2599 and others), raw force goes to −1.75 N while `|b − b0|` is several thousand µT. The magnet is moving sideways or against the press axis: lateral shear, a pad being rolled or peeled, or the glove shifting. Show this as a **shear/lateral-load indicator** (arrow + amber ring), not as pressure. The thumb shows the same pattern (frames 2790–2792).
- **Creep / residual offset:** after a heavy press, the middle pad held a 0.2–0.5 N residual for about 30 s, and the index pad held about 0.4 N. Add:
  - a manual **Re-tare** button (a software offset per pod, taken when the hand is relaxed)
  - optional auto-tare: when IMU motion is low **and** a pod's force has stayed steady below 0.6 N for ≥ 3 s with no other pod loaded, slowly bleed that pod's offset (time constant ~5 s, off by default, with the state shown).

**Peaks in the capture** (use them as acceptance checks, §11)

| Pod | Peak | Frame | t from start |
|---|---|---|---|
| middle | ≥ 9.81 N (saturated) | 2502 | 10.1 s |
| index | 3.89 N → 38.9 kPa | 2515 | 11.4 s |
| palm | 7.99 N → 36.3 kPa | 2551 | 15.0 s |
| thumb | 3.30 N → 30.0 kPa | 2608 | 20.7 s |

**Derived CSV maxima:**
- pressure: middle 92.9 kPa, index 36.1, palm 34.1, thumb 29.4
- slip scores reach 94–99 on every pod
- contact frames: middle 341, index 211, thumb 119, palm 100
- roll ranges from −129° to +158°, pitch from −13° to +48°

**IMU:** the accelerometer mostly reads about 9 m/s² on chip X, so the hand was largely palm-sideways. `sensortime` resets at frames 2476, 2536 and 2705.

---

## 6. Signal pipeline (implement exactly)

Run these steps for each frame, for each pod:

1. **Validity:** if `valid === false` or the force is null → state `missing` or `zeroing`. Keep the slot in the ring buffer as NaN.
2. **Force:** use the device's `force_n`, minus the software tare. **Recompute mode** (toggle, for debugging): `Δb = b_ut − baseline.b`, `load = Δb · unit`, `F = gain_n_per_ut × load`. This matches the device's `load_signal_ut` and `force_n` to within 0.1% on the real data (index frame 2515: load 11 348 µT vs device 11 343 µT; F 3.89 N). Show the difference in the debug panel.
3. **Pressure (kPa):** `P = F / (area_cm2 × 1e-4) / 1000 = F × 10 / area_cm2`.
4. **Shear (µT, relative):** `s = Δb − (Δb·unit)·unit` and `|s|`. Normalize by `noiseUt` for display. Raise a shear flag when `|s| > 8 × noiseUt` or when `FORCE_CLAMPED_LOW` is set.
5. **Field rate:** `|Δb_t − Δb_{t−1}| / dt` (µT/s). **Slip event:** the pod is in contact **and** the field rate exceeds `2 × fieldRateFloorUtS` **and** the shear direction changes by more than 25° or force drops faster than `forceRateFloorNS × 10` N/s. Debounce for 300 ms. If a derived `slip_score` exists, use it instead (event when it rises above 70).
6. **Contact with hysteresis:** on above 0.15 N, off below 0.08 N (both configurable).
7. **Contact patch radius (rendering only):** `r = r0 × (F / Fref)^(1/3)` (Hertz-like), clamped to the pad's physical size, where `r0` is derived from the pad area.
8. **Display smoothing:** a One-Euro filter (min cutoff 1.0 Hz, beta 0.02) on force, shear and orientation. Use it **for display only**; metrics use unsmoothed values.
9. **Interpolation to 60 fps:** render one frame behind real time (≈100 ms at 10 Hz) and lerp between the last two samples by `t_device_us`. **Don't interpolate across gaps over 250 ms or across invalid samples.** Show a brief "signal gap" tick instead.

**Orientation (from the IMU):**
- `roll = atan2(ay, az)`
- `pitch = atan2(−ax, √(ay² + az²))`
- Check: the calibration's `acc` [9.226, −1.097, 3.193] gives roll −0.331 rad and pitch −1.220 rad, the same as the file.
- Complementary filter with the gyro (α = 0.98).
- Yaw: integrate gyro z, mark it "relative, drifts", and offer a reset.
- Show all angles relative to the calibration pose, with an absolute-angles toggle.
- `motion_score` = rolling RMS of `|gyr|` over 0.5 s, scaled 0–100.

**Buffers:** typed-array ring buffers sized for 15 min at 400 Hz per channel, allocated lazily. No unbounded arrays.

---

## 7. High-definition visualization

**Visual direction:**
- **Mood:** a calm clinical instrument, not a gaming HUD.
- **Palette:** dark graphite/navy background, one cool accent for UI, warm colors only for pressure.
- **Type:** Inter (UI) and IBM Plex Mono (numbers), with large tabular numerals.
- **Contrast:** WCAG AA everywhere, and readable from 3 m on a projector.

**Layout (desktop):**
- **Top bar:** source (WS / Replay / Demo), URL input, Connect, status pill, device id, fps, data Hz, latency, drops, Re-tare, settings
- **Center (≈60% width):** the main **Hand Pressure Map**
- **Right column:** the **Exercise Coach** panel
- **Bottom:** the **Session Timeline**, full width
- **Collapsible drawer:** the debug panel (raw frame, flags, recompute diff, parse log)

### 7.1 Hand Pressure Map (hero view)

- **Hand drawing:** an anatomically proportioned **palmar view of a right hand**, drawn as a smooth SVG path (or a procedural spline outline), with subtle crease lines (thenar crease, distal palmar crease, finger joints).
  - Mirror it for `hand: left` when known.
  - Place the sensor sites on it: thumb pad, index pad, middle pad, palm center (or the thenar eminence when the location is `thenar_apb`).
- **Pressure field:** render it with a **WebGL fragment shader**, falling back to Canvas2D. Use an offscreen field texture of at least 512×512, clipped to the hand silhouette.
  - **Kernel:** each pod contributes an anisotropic Gaussian, elongated along the finger axis. Its σ comes from the contact-patch radius (§6.7), and its amplitude is pressure in kPa.
  - **Combination:** combine kernels with a soft max, not a sum, so the field never exceeds the real measured peak.
  - **Colormap:** a perceptually uniform one (an inferno-like ramp from deep indigo → magenta → orange → near-white), embedded as a 256-entry LUT. **Avoid rainbow / jet / turbo.**
  - **Colorblind check:** confirm the ramp has a monotonic lightness ramp.
  - **Legend:** in kPa. Scale modes: fixed 0–100 kPa (default, so sessions are comparable), auto, or per-pod.
  - **Isobars:** thin contour lines every 10 kPa, toggleable.
- **Measured vs. estimated:** inside each pad's physical footprint, full opacity. Outside it, the "estimated spread" layer at ≤ 40% opacity with a faint dither. A "Measured points only" toggle shows just the pads as crisp disks.
- **Per-pad overlays:**
  - live N and kPa label with leader line
  - contact ring (thin white ring when in contact)
  - **saturation:** a white-hot core plus a pulsing "SAT ≥9.8 N" badge
  - **shear:** an amber arrow whose length is `|s|` (normalized) and whose direction is `s` projected into the pad plane. Pod axes are not mapped yet, so label the direction "approximate" and allow a per-pod rotation offset in settings.
  - **slip:** an expanding ripple ring, plus a short haptic-style shake of the badge
  - **zeroing:** a grey shimmer with a "calibrating" label
  - **missing/invalid:** a hatched grey disk labelled "no data"
- **Unsensed areas:** fingers without a pod render in neutral tone with a tiny "not sensed" label.
- **Temperature:** optionally tint the pad outline by `temp_delta_c` (a skin-contact cue), off by default.

### 7.2 3D view (Three.js, toggle in a tab or picture-in-picture)

- **Pressure terrain:** a 256×256 plane displaced by the same field and colored with the same LUT. Use soft directional and hemisphere lighting, orbit controls (implement them by hand or load from cdnjs), and an orthographic/perspective toggle.
- **Hand pose avatar:** a stylized low-poly hand (capsules and rounded boxes) rotated by the IMU orientation.
  - Finger curl is **illustrative**: driven by the current exercise state (pinch → thumb and index curl toward each other; grip → all fingers curl). Label it "pose illustrative; orientation measured".
  - Color fingertip spheres by pressure.
  - Draw a small XYZ gizmo with roll/pitch/yaw readouts.

### 7.3 Per-pod strips (under the map or in the right column)

For each pod: name, live force (N, 2 decimals), pressure (kPa), a 10 s sparkline with the target band shaded, session peak, a contact dot, and slip / shear / saturation icons.

### 7.4 Session Timeline (bottom)

- **Traces:** four force traces, one color per pod. Use a categorical palette that is distinct from the pressure LUT and colorblind-safe.
- **Markers:** rep spans (shaded), holds, slip events (▲), saturation (red tick), zeroing (grey band), signal gaps (dashed), device markers.
- **Navigation:** zoom with the wheel and pan by dragging. In replay, click to seek. Hovering shows a crosshair with all values at that time.

---

## 8. Rehab layer: Exercise Coach

**Default program framing:** "Home hand rehab after stroke". It must be editable, and its labels must stay generic, not diagnostic.

### 8.1 Exercise library and detection

Detection runs on unsmoothed force, with a 150 ms debounce.

| Exercise | Active pods (above onset) | Must stay quiet (< 0.2 N) | Notes |
|---|---|---|---|
| Tip pinch | thumb + index | middle, palm | Shows the thumb/index balance |
| Tripod pinch | thumb + index + middle | palm | |
| Power grip | palm + ≥ 2 fingertips | — | Total grip = sum of pods |
| Isolated finger press | exactly one fingertip | all other pods | Finger individuation |
| Relax / release | none for ≥ 1 s | all | Counts toward relaxation quality |

**Modes:**
- **Guided:** the therapist picks the exercise, target force, hold time and reps; the coach prompts each rep.
- **Free:** the app classifies what it sees and logs it, with a confidence value.
- **Default targets:** pinch 2.0 N, tripod 2.5 N, grip 5.0 N, isolated press 1.5 N, hold 3 s, 10 reps. All are editable.

### 8.2 Rep state machine

`IDLE → LOADING` when the target-pod force exceeds `max(0.3 N, 25% of target)`, then `→ HOLD` within ±15% of target, then `→ RELEASING` below 50% of peak, then `→ IDLE` below 0.15 N. A rep counts if it reached ≥ 80% of target. Holds count from the HOLD time.

**Per-rep metrics:**
- peak force (N) and peak pressure (kPa)
- time to peak (s)
- **rate of force development** (N/s, 10–90% rise)
- hold time within the target band (s)
- **steadiness**: coefficient of variation of force during the hold (lower is better), plus 4–12 Hz tremor power from an FFT when the data rate is ≥ 50 Hz, otherwise "N/A at 10 Hz"
- release time (s)
- slip count
- **compensation flag**: non-target pods above 0.3 N during the rep (for example, palm loading during a pinch)
- **individuation index**: `1 − Σ(off-target force) / Σ(all force)` (for isolated presses)
- **balance**: thumb ÷ index force ratio (for pinches)

### 8.3 Coach panel UI

- **Header:** current exercise name and the rep counter as big type ("Rep 4 / 10").
- **Gauge:** a large semicircular force gauge with the target band, a live needle and a ghost peak marker.
- **Hold ring:** a countdown ring that fills during HOLD.
- **Cue line:** one short line of plain-language coaching, at most one cue at a time, rate-limited to one every 2 s. Examples: "Squeeze a little more", "Hold it… 2 s", "Relax your palm — try fingertips only", "Nice steady hold".
- **Last-rep card:** peak, hold, steadiness, flags.
- **Session summary:**
  - reps done, reps that hit target
  - % of time in the target band
  - mean and best peak
  - **fatigue trend**: slope of peak force across reps, as % change from first to last third
  - mean steadiness
  - slip and compensation counts
  - total active time vs. rest time

### 8.4 Measurable impact (export)

- **Export session JSON:** device id, calibration summary, program, per-rep arrays, session summary, data quality (drops, gaps, zeroing time, saturation count) and app version.
- **Export CSV:** per-rep table and a downsampled time series.
- **Progress view:** keep previous sessions' summaries in `localStorage` (wrapped in try/catch; the page must work without it). Show trend sparklines across sessions: peak pinch force, steadiness, individuation and time-in-band.

### 8.5 Claude session note (hook; no API key in the page)

- **Button:** "Generate session note".
- **Request:** POST the session JSON to a **configurable endpoint URL** set in settings. This is the user's own backend, which calls Claude.
- **Response:** render the returned `{patient_note, clinician_note}`. The patient note is a plain-language summary plus one encouragement; the clinician note is SOAP-style with objective metrics.
- **No endpoint set:** show the exact JSON payload with a Copy button.
- **Never embed API keys.**

### 8.6 Disclaimer

Add a small footer line: "Research prototype · not a medical device · metrics are indicative."

---

## 9. Settings drawer

Include:
- the contact on/off thresholds and the noise floor
- the pressure scale mode
- the colormap intensity gamma
- the estimated-spread opacity
- a per-pod rotation offset for shear arrows
- a per-pod enable toggle
- hand side (L/R)
- the auto-tare toggle
- smoothing on/off
- recompute-from-`b_ut` mode
- target and program editing
- the Claude endpoint URL

Save all settings to `localStorage` (wrapped in try/catch).

---

## 10. Performance, robustness, accessibility

- **Rendering budget:** under 4 ms per frame at 60 fps on an M1 laptop. Upload the pressure field as a small uniform array of pod parameters; the GPU evaluates the kernels. Don't recompute the field per pixel on the CPU.
- **Parsing:** must keep up with 400 msgs/s. Batch DOM updates to at most 10 Hz for text; canvases update every rAF.
- **Unknown input:** unknown fields are ignored, unknown pods appear in an "Other pods" list, and a malformed message increments a counter and is logged (keep the last 50 errors in the debug drawer).
- **Keyboard shortcuts:** Space play/pause, ←/→ step frame (replay), T re-tare, M add marker, D debug drawer. All controls are reachable by tab with visible focus rings.
- **Motion:** respect `prefers-reduced-motion` by disabling ripples and shakes.
- **Layout:** responsive down to 1024 px wide. Below that, stack the panels vertically.

---

## 11. Acceptance tests (must pass with the real capture replayed)

1. **Zeroing:** loading `dermis-raw-*.jsonl` shows "Calibrating" for about the first 3 s (frames 2402–2430) and counts no reps there.
2. **Saturation:** at about 10.1 s (frame 2502), the middle pad shows **SAT ≥ 9.8 N** with the white-hot core; the timeline shows a red tick.
3. **Index peak:** at about 11.4 s (frame 2515), the index pad shows 3.89 N / 38.9 kPa.
4. **Palm peak:** at about 15.0 s (frame 2551), the palm shows 7.99 N / 36.3 kPa (area 2.2 cm²).
5. **Thumb peak:** at about 20.7 s (frame 2608), the thumb shows 3.30 N / 30.0 kPa.
6. **Shear, not pressure:** palm frames 2580–2599 show the amber **shear** indicator and **no** pressure, even though `force_n` = 0.
7. **Recompute mode:** forces from `b_ut` match the device's `force_n` to within 2% on every non-clamped, non-zeroing frame. The debug panel shows the max error.
8. **Gap:** the 302 ms gap at frame 2781 renders as a gap; there is no interpolated bridge.
9. **Derived CSV:** loading it plays the same session using its own `slip_score`, `contact` and `roll/pitch/yaw` columns.
10. **WebSocket:** with the replay server below running, Connect goes Live, shows about 10 Hz data, survives a server restart (auto-reconnect), and shows a flawless pressure map at 60 fps.
11. **Demo source:** the Demo source runs with no files or network and exercises every overlay state.

---

## 12. Include a test server (as a comment at the end of the HTML)

Put a ready-to-run Python replay server in an HTML comment at the bottom of the file:

```python
# pip install websockets
# python replay_ws.py dermis-raw-1792909170615.jsonl --port 8765 --speed 1.0 --loop
```

The server must:
- read the JSONL file
- send each line as one text message, keeping the original `t_device_us` spacing divided by `--speed`
- loop forever with `--loop`
- serve multiple clients

Keep it about 30 lines, using `asyncio` and `websockets`.

---

## 13. Out of scope for v1

- binary COBS/CRC frame decoding in the browser (the host bridge sends decoded JSON)
- clock sync
- multi-user state
- auth
- any real medical scoring claims

**Build the complete file now. Don't leave TODOs for core features in §3–§8.**
