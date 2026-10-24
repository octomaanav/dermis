# Dermis console

The console makes the pipeline visible end to end: **glove telemetry → signal processing → recorded
demonstrations → training → autonomous evaluation**.

- Every number comes from the running system or from files on disk.
- Nothing is simulated for display, and empty panels say why they are empty.
- It is local only: standard-library Python HTTP on `127.0.0.1`, plus plain HTML, CSS and JS with no
  external libraries and no network access.

## Launch

There are two ways to run it. The UI is the same; what it can show depends on the process.

| Command | What it shows |
| --- | --- |
| `python -m dermis teleop --source tcp --gui --auto --dashboard-port 8000` | Live robot (hand by default): TCP ingestion, controller, arming, calibration, recording, demonstrations, plus datasets, training and evaluation from disk |
| `python -m dermis dashboard --source synthetic --rate 100 --http-port 8000` | Monitor: telemetry from any source (`synthetic`, `replay`, `tcp`, `serial`, `wire-file`), plus datasets, training and evaluation. No robot control |
| `python -m dermis dashboard … --cup-controller grip_load_expert,fixed_14N` | Adds the live fragile-cup panel in AUTONOMOUS mode (SCRIPTED baseline controllers; use `--cup-checkpoint DIR` for trained ones) |

Open **http://127.0.0.1:8000**. Stop with Ctrl+C in the terminal.

### Synthetic demonstration (two terminals)

```sh
# 1: robot + console. --auto calibrates and arms during the sender's still start.
python -m dermis teleop --source tcp --gui --auto --dashboard-port 8000
# 2: SYNTHETIC glove at 100 Hz: fist + punch every 2.5 s
python -m dermis send-test --mode synthetic --profile punch --rate 100
```

Add `--faults ci` to the sender to watch the fault counters and the event timeline react. It injects
malformed lines, duplicates, reordering, oversized lines, a burst, null and stale channels, a
pause and a reboot.

## Judge walkthrough (about 3 minutes)

1. **Presentation view**, the default:
   - **Status bar.** Source **SYNTHETIC** and control mode **TELEOPERATION** are separate. Connection,
     device and config IDs, recording, calibration and the host-side frame age are always visible.
   - **Simulation.** A schematic drawn from joint angles read back from PyBullet. The native PyBullet
     window shows the 3D scene beside it; see *Simulation display* below for why it isn't a camera feed.
   - **Glove pods.** Each pod sits at its anatomical location. ✓, !, ✕ and ? mark healthy, degraded,
     invalid and missing, so state never depends on colour alone. Each pod shows its |ΔB| from the
     unloaded baseline, IMU availability and sample age. The teal arc is the *estimated* finger flex
     from the EXPERIMENTAL mapping. Click a pod to inspect it.
   - **Four key numbers.** Accepted frames/s, usable channels %, newest frame age and missing frame IDs.
     Hover a label for its definition.
   - **Active mode.** Explains which stages run right now and the current outcome.
2. Stop the sender (Ctrl+Z):
   - Within 250 ms the controller pauses on the watchdog.
   - The frame age turns red, and the event shows on the timeline (Engineering view).
   - Type `fg` to resume.
3. **Engineering view**, using the toggle at the top:
   - **Live ingestion.** Received vs accepted rates, rejected by reason, duplicates, out-of-order, missing
     IDs (connection and session), usable channels with their denominator, receive interval, jitter,
     host processing time and control-loop timing.
   - **Sensor plots for the selected pod:**
     - absolute B and baseline-subtracted ΔB on separate charts, so a ~27 000 µT baseline doesn't hide
       a 500 µT change;
     - acceleration and angular velocity;
     - the mapped command.
     - Hovering any chart moves a shared cursor across all of them. Gaps are invalid samples; amber dots
       are STALE values.
     - Pause freezes the display only. Ingestion continues.
   - **Calibration.** Capture refuses unless the required channels are fresh and new frames are arriving,
     and asks for confirmation before replacing a calibration. It reports per-channel *stationary
     variation*.
   - **Raw frame inspector.** The latest JSON line with its verdict, rejection reason and timing.
     Changed fields are highlighted, and it can copy the frame.
   - **Recording.** Raw telemetry, with task label and outcome, is kept separate from teleop
     demonstrations (robot state + issued commands + markers).
   - **Episodes and episode viewer.** Observations vs issued commands vs simulated contact, with
     annotations: contact onset, release, simulated-threshold crossings, env events and markers.
   - **Training.** Loss curves, epochs, data snapshot and split, training time, and inference time
     measured on demand.
   - **Evaluation.** Results grouped by task and scenario suite. Run-level spread is shown across
     training seeds.
4. **Fragile cup**, when the dashboard runs with `--cup-controller`:
   - Pour, Reset (same scenario), Next and controller selection, plus a paired comparison.
   - Shows the simulated finger loads against the simulated threshold.
   - In AUTONOMOUS mode the glove stream stays visible, but no human motor command reaches the robot.

**What each view proves:**
- **Presentation** shows real glove-shaped telemetry being validated, turned into a usable signal and
  driving a robot in real time, with its safety state visible.
- **Engineering** shows the evidence: ingestion quality, raw frames, calibration, recorded data, and
  how training relates to autonomous task success.

## Source labels

| Label | When |
| --- | --- |
| SYNTHETIC | Generated data: `--source synthetic`, or a network stream whose `device_id` starts with `synthetic` (the dev sender) |
| REPLAY | Recorded data played back (`--source replay` / `wire-file`) |
| TCP-UNVERIFIED | Any other TCP or stdin stream: there is no proof it is the real glove (e.g. the supplied fixture replayed over TCP) |
| GLOVE | A USB serial device, or a TCP stream an operator vouches for with `teleop --source-label GLOVE` |

The same rule applies in the teleop and monitor consoles, and in recording file names and metadata.

## Metric definitions

They are shown as tooltips in the UI; the source of truth is `src/dermis/dashboard/metrics.py`.

| Metric | Definition | Scope |
| --- | --- | --- |
| Received / accepted frames/s | Lines received (any verdict) and frames accepted after schema, duplicate and order checks | rolling 5 s window, host clock |
| Rejected | Malformed JSON or UTF-8, oversized, schema violations, adapter errors | connection and session |
| Missing frame IDs | Unique IDs never received, inferred from forward gaps in the device frame counter. A late frame that fills a gap is taken back out. Restarts on reconnect, device change or counter reset. **Not network packet loss**: gaps can arise upstream (sensor bus, firmware buffer, bridge). | connection and session |
| Duplicates / out of order | Same number as the last accepted frame / older within the 64-frame reorder window; dropped | connection and session |
| Usable channels | Average over the window of usable ÷ (configured pods × 2 sensors). Usable = valid, finite, fresh or STALE for at most 50 ms. TEMP_STALE does not count against it. | window |
| Newest frame age | Host time since the newest accepted frame arrived. **Not latency**: device and host clocks are not synchronized, so `t_host_ns` stays null. | instant |
| Receive interval / jitter | Mean and standard deviation of host receive intervals between consecutive accepted frames | window |
| Host processing | Parse + compat + schema + pipeline time per line, median / p95 (TCP receiver only) | window |
| Control tick | Simulation step time, median / p95, and overruns of the 33 ms budget (teleop only) | last 300 ticks |

## Architecture

```
TCP / source thread ──> TelemetryPipeline ──> LatestSlot ──> controller + PyBullet (sim thread, 30 Hz)
        │ every line / frame                                        │ once per tick: service()
        ▼                                                           ▼
     LiveHub (bounded deques, metrics) <──────────────────── issued commands, queued console commands
        │ 10 Hz, decimated (≤ 60 samples per update)
        ▼
/api/stream (SSE, one per tab, max 6) ──> browser (charts redraw on animation frames,
                                          DOM panels rebuilt at most 4×/s and only when changed)
```

- **Robot control never waits on the browser.** Console buttons queue a command that the simulation
  thread runs at its next tick (bounded queue, 5 s timeout).
- **Recording runs at full rate in the TCP receiver.** Charts are decimated.
- **Browser buffers are capped** at 30 s and 20 000 samples, 300 events and 4 000 commands.
- **Browser reconnects are automatic** (EventSource). Changing pods closes the old stream before opening
  a new one, so there are never duplicate subscriptions.
- **Access is local only.** Non-loopback clients get 403, and POSTs to `/api/control/*` need an
  `X-Dermis: 1` header, which blocks cross-site form posts.

## Simulation display

Rendering cost was measured on an M2 Mac:

| Renderer | Cost per frame | Effect |
| --- | --- | --- |
| Software (`ER_TINY_RENDERER`), 512×320 | ~950 ms | Holds the Python GIL the whole time; it stalled ingestion and control in testing |
| GPU (`ER_BULLET_HARDWARE_OPENGL`, GUI window only), 320×200 | ~36 ms | More than one 33 ms control tick |
| GPU, 512×320 | ~59 ms | Two control ticks |

So the camera feed is **off by default**:
- `teleop --dashboard-feed-hz 2` enables it, GUI mode only. Frames are rendered on the simulation thread
  and the console shows the measured render time.
- Without it, the presentation view draws a schematic from measured simulator state, and the native
  PyBullet window is meant to sit beside the browser.
- The fragile-cup worker never renders, because its environment is headless.

## Measured overhead

Synthetic punch stream at 100 Hz → `teleop --headless --auto --dashboard-port`, hand robot, M2 Mac (8 cores).

**Test conditions:** the machine was heavily loaded. The load average was 13–21 because of other user
applications and a concurrent evaluation job. Numbers are 20 s windows, measured back to back.

| Condition | Received / accepted fps | Control tick median / p95 | Overruns per 20 s |
| --- | --- | --- | --- |
| No browser | 99.0 / 99.0 | 7.0 / 27.2 ms | 17 |
| Presentation view open (headless Chrome) | 100.0 / 100.0 | 6.9 / 12.5 ms | 12 |
| Engineering view open (plots + raw inspector at 2 Hz) | 100.2 / 100.2 | 6.4 / 12.4 ms | 5 |
| No browser again | 100.0 / 100.0 | 6.6 / 11.0 ms | 2 |

**GUI mode, 20 s windows, load average 5–8.** This checks whether PyBullet holding the GIL starves the
TCP receiver thread:

| Sender rate | Received / accepted fps | Newest frame age | Control tick median / p95 | Overruns | Watchdog pauses |
| --- | --- | --- | --- | --- | --- |
| 100 Hz | 100.2 / 100.2 | 7.5 ms | 10.6 / 20.2 ms | 4 | 0 |
| 350 Hz (the glove's rate) | 349.4 / 349.4 | 5.1 ms | 5.8 / 15.9 ms | 9 | 0 |

- **The receiver keeps up** because the sim loop sleeps for most of each 33 ms tick.
- **If a heavier scene pushed the step time toward the budget,** that idle time would shrink. The next
  step would then be to move ingestion to a child process, like `glove/procsource.py`.

**Monitor mode (`dashboard --source synthetic --rate 100`, load average about 6).**

| Setup | Accepted frames/s |
| --- | --- |
| Monitor alone | 99.2 |
| With the fragile-cup worker running in the same process | 99.0 |

- An earlier synthetic source delivered only about 70 frames/s; it now paces to deadlines.
- In this mode the frame age is read from the reader's newest frame, so the console's 50 ms poller does
  not inflate it.

- **Ingestion is unaffected by the console.** Run-to-run variation in tick timing is dominated by the
  machine's load.
- **The first version of the client** rebuilt its DOM 10×/s, which roughly tripled overruns
  (9 → 62 per 20 s with the presentation view). It now rebuilds at most 4×/s and only on change.
- **Per frame, the console adds about 90 µs** in the ingestion thread.
- **Schema validation was the largest ingestion cost**, about 3 ms per frame with `jsonschema`. A fast
  validator now checks the same rules in ~66 µs. It is generated from the schema file and verified
  against `jsonschema` on more than 2 000 mutated frames (`tests/test_tcp_validation_fast.py`).

## Limitations and not verified

- **The mapping is EXPERIMENTAL.** Finger flex in the pod map is the mapping's output, not a measured
  finger pose. Magnetic values are µT changes, never force; contact forces are simulated.
- **Teleop demonstrations are not training data.** They are recorded with `bc_ready: false`, because
  they don't use the grasp-lift-place BC observation format.
- **Fragile-cup checkpoints:** none existed when this was written, so the live panel was verified with
  SCRIPTED controllers only. `--cup-checkpoint` uses the cup's own loader and is untested here.
- **Monitor-mode limits:** with sources other than TCP, rejection timestamps are poll times (±50 ms),
  and host processing time is not measured.
- **Visual checks used headless Chrome screenshots** (1600 px desktop width) of both views, the
  empty/no-source state, and the AUTONOMOUS cup panel. They were not checked on a projector or at
  other window sizes. Below 1100 px wide the layout switches to one column.
