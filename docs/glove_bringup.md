# Glove bring-up and live teaching

This page covers what to run when the assembled glove arrives, and how its grip control connects to the fragile-cup teaching workflow.

Everything here was tested **without hardware**: SYNTHETIC frames, the supplied fixtures, and scripted stand-in demonstrations. Those tests validate software integration only, not sensing accuracy or real-world transfer.

## 0. Pieces and where they live

| Purpose | Code |
|---|---|
| Ingestion (serial / TCP / stdin / replay), ordering, STALE handling, bounded hold | `telemetry/` (shared by every path) |
| Pod discovery (DEVICE_INFO + frames), source label | `glove/discovery.py` |
| Two-point grip calibration, configurable mapping, arming / watchdog / limits | `glove/grip.py`, `configs/glove/grip.yaml` |
| Terminal inspector, labelled trials, raw recording | `glove/session.py`, `glove/report.py` |
| Live plots (raw B, baseline-subtracted ΔB, IMU, flags, rates) | the console: `python -m dermis dashboard ...`, see `docs/dashboard.md` |
| Armed TCP teleop of the Panda / 3-finger hand | `server/teleop/`, see `docs/tcp_teleop.md` |
| Teaching: TEACH → snapshot → train → dev eval → activate → AUTONOMOUS | `cup/live.py`, `cup/teaching.py` |

**Source labels** (aligned with the console):

| Label | When |
|---|---|
| SYNTHETIC | device_id starts with `synthetic` |
| REPLAY | Files |
| REAL HARDWARE | A USB serial device, or `--confirm-hardware` given by the operator |
| TCP-UNVERIFIED | Any other network or stdin stream, which could be a bridge or a replay |

## 1. Inspect the glove (first thing to run)

```sh
# USB serial, JSON lines (or add --wire for Dermis Wire v1 binary)
python -m dermis glove inspect --source serial --serial-port /dev/cu.usbmodemXXXX --out data/glove/s1
# glove (or its bridge) streaming decoded-frame NDJSON over TCP to this Mac
python -m dermis glove inspect --source tcp --port 8765 --out data/glove/s1 [--confirm-hardware]
# live plots in the browser at the same time (separate process, separate port/source)
python -m dermis dashboard --source serial --serial-port /dev/cu.usbmodemXXXX --http-port 8000
```

**What it prints:**
- **Discovery:** every pod found in DEVICE_INFO and in frames, with whether it has a magnetometer and an IMU. Nothing is hardcoded. Required grip pods that are missing get a warning.
- **Each second, per pod:**
  - raw B xyz (µT), and ΔB xyz relative to the unloaded reference once captured;
  - |ΔB| raw and filtered;
  - the field's validity: fresh, STALE, INVALID or MISSING;
  - sample age (device clock) and host age (receive time);
  - acceleration (m/s²) and angular rate (rad/s);
  - the received frame rate, plus lost, duplicate and out-of-order counts.

**Typed commands** (letter, then Enter):

| Key | Action |
|---|---|
| `u` | Labelled trial: unloaded |
| `l` | Labelled trial: light press |
| `s` | Labelled trial: stronger press |
| `r` | Labelled trial: release |
| `w` | Labelled trial: sideways loading. Recorded as such; it is never labelled slip. |
| `0` | Capture the unloaded grip reference |
| `1` | Capture the comfortable-press reference |
| `c` | Build the grip calibration |
| `q` | Save and quit |

```sh
python -m dermis glove report --session data/glove/s1     # overlay.png (raw vs filtered |dB|), repeats.png, numbers.json
```

**Session contents** (raw sensor data, kept separate from simulator episodes):
- `frames.jsonl`: every decoded frame plus its host receive time.
- `markers.jsonl`: trial and reference markers.
- `layout.json`: discovered pods and channels, plus DEVICE_INFO if the transport provided it.
- `meta.json`: source label, mapping config, and clock anchors.
- `grip_calibration.json`: only if the calibration was accepted.

Device time and host receive time are both kept and are **not synchronised**: there is no clock-sync fit.

## 2. Grip control

`configs/glove/grip.yaml` (EXPERIMENTAL):

| Field | Default | Meaning |
|---|---|---|
| `required_pods` | `index_tip, thumb_tip` | If unusable (missing, null, invalid, or STALE beyond 50 ms), grip goes PAUSED and the output is held. |
| `optional_pods` | `middle_tip` | Used when usable; never pauses control. |
| `axis` | `auto` | The calibrated unloaded→press direction. Alternatives: `x`, `y`, `z` (one chip axis), or `norm` (\|ΔB\|). |
| `sign` | `1` | Flip if pressing reduces the signal. |
| `gain` | `1` | Applied to closure. |
| `deadband` | `0.05` | In closure units. |
| `filter_tau_s` | `0.08` | Exponential smoothing time constant. |
| `spike_ut` | `4000` | A sample jumping more than this is rejected, not averaged. |
| `out_min`, `out_max` | 2 N, 20 N | Output limits: fragile-cup grip effort. |
| `max_rate` | 40 N/s | Slew limit. |
| `combine` | `mean` | Combine usable pods by mean (or max). |

**Calibration** is two-point, recorded manually: `0` (unloaded), then `1` (comfortable press), then `c`. It is **rejected** if any of these hold:
- A required pod has too few fresh samples (STALE, null and invalid samples are never averaged in).
- A reference is not stationary: standard deviation above 60 µT.
- The press is not separated from unloaded: |press − unloaded| < 5 × the larger stationary standard deviation.
- The device or config changed during or between the captures.

An accepted calibration is tied to the device, config and pod IDs. A mismatch disarms control.

**Control states:**

| State | Meaning |
|---|---|
| DISARMED | Default. The output is held. Leaving it requires explicit arming (A), an accepted, matching calibration, and live, usable required channels. |
| ARMED | Commands follow the calibrated mapping, bounded, filtered and slew-limited. |
| PAUSED | No frame for 0.25 s, or a required channel is unusable. The output is held, and control resumes automatically when the input is usable again. After 2 s without frames it goes to DISARMED (connection lost), and re-arming is required. |

Recalibration is refused while armed.

**Not done:**
- No force units.
- No slip detection.
- No independent finger-joint estimates.
- No hand position: acceleration is never double-integrated here.
- IMU tilt-to-velocity exists only in the separate, clearly labelled TCP teleop path (`teleop --tilt-xy / --tilt-z`).

## 3. Teaching workflow (fragile cup)

```sh
# 1-3. TEACH: identical valid grasp on every reset; keyboard grip; recording saves sim observations + issued efforts
python -m dermis cup teach                          # SPACE record on, P pour, Up/Down effort, R reset (same scenario)
# (glove instead of keyboard: same action interface, raw sensor stream saved separately and linked by host time)
python -m dermis cup teach --input glove --glove-input tcp --glove-port 8765 --experimental-glove-teleop \
    --grip-calibration data/glove/s1/grip_calibration.json          # A arms, X disarms
# 4. freeze the chosen episodes into an immutable snapshot
python -m dermis cup snapshot --from data/episodes/cup_teach --name demo1 [--last 5] [--source-filter keyboard]
# 5. train in the background (optionally fine-tune from an identified pretrained checkpoint)
python -m dermis cup train-candidate --snapshot data/datasets/cup/demo1 --name cand1 \
    [--init-from runs/cup/benchmark/runs/cup_bc_contact]
python -m dermis cup candidates                     # state, training time, dev results, which is ACTIVE
# 6. development evaluation (never the held-out suite)
python -m dermis cup evaluate-candidate --name cand1
# 7. explicit activation (refused without a dev evaluation)
python -m dermis cup activate --name cand1
# 8. AUTONOMOUS: policy acts on simulator observations; human grip input is not connected
python -m dermis cup judge
# inspection only: REPLAY plays recorded efforts open loop, with an orange REPLAY banner
python -m dermis cup replay --episode data/episodes/cup_teach/<id>.json --gui
```

**What the workflow guarantees:**
- **Separation from older checkpoints:** a checkpoint names its snapshot (episode IDs, SHA-256s, sources). Recording more episodes never changes an existing snapshot, checkpoint or the active pointer, and the judge banner names the checkpoint and its data.
- **Disclosed fine-tuning:** `pretrained_from` (checkpoint, its sources and episode count) is stored in the new checkpoint. Its normalizer and feature layout are reused.
- **What the network sees:** the autonomous network sees simulator observations only. Glove readings are never policy inputs. In AUTONOMOUS mode no human input object exists, and key presses only trigger judge controls (tested).
- **Paused glove ticks:** these are recorded with tags `GLOVE_PAUSED` / `GLOVE_DISARMED`, and the snapshot reports each episode's non-active tick fraction.

## 4. Is live teaching practical? (measured, scripted stand-ins)

`python -m dermis cup teaching-study`. Results are in §6. In short: fresh demonstrations alone are not enough, and fine-tuning a working policy on a few demonstrations made it worse. Adding the new demonstrations to the existing dataset and retraining works mechanically in about 1 minute.

## 5. First assembled-glove session: checklist

- [ ] **Connection.** `glove inspect` shows the expected pods, and the label reads REAL HARDWARE (serial) or TCP-UNVERIFIED (confirm with `--confirm-hardware` only if you know it's the glove). Lost, duplicate and out-of-order counts stay near zero for 60 s.
- [ ] **Channels.** Each pod's magnetometer shows `fresh`, not STALE or INVALID. The IMUs show about 9.8 m/s² at rest. Note any pod that is MISSING or declared but silent.
- [ ] **Pod mapping.** Press each pad in turn: only the expected pod's |ΔB| rises. Fix DEVICE_INFO `pods[]`, or `required_pods`, if not.
- [ ] **Trials.** Record `u`, `l`, `s`, `r` and `w`, 3 repeats each. Then run `glove report`:
  - unloaded stationary std is small next to the light press;
  - press plateaus repeat (CV ≤ ~15%);
  - the residual after release returns near zero;
  - sideways ΔB direction is reported, not labelled slip.
- [ ] **Orientation.** Rotate the unloaded hand slowly. If |ΔB| changes about as much as a light press does, orientation/Earth-field effects need handling before grip control.
- [ ] **Grip calibration.** `0` (unloaded, still), `1` (comfortable press, held still), `c`.
  - If REJECTED, read the reasons: separation, stationarity or sample count. Don't lower `min_separation` to force it through.
  - Try `axis: auto` first, then a single axis if one dominates.
- [ ] **Bounded control in TEACH.** `cup teach --input glove ...`:
  - A arms. The effort follows the pinch within `out_min`..`out_max`.
  - Stop the stream: within 0.25 s it shows PAUSED with the effort held; after 2 s, DISARMED. Reconnecting does **not** resume until you press A again.
  - Cover or unplug one required pod: grip pauses. An optional pod doesn't affect it.
- [ ] **Teaching rehearsal.** Record a few glove TEACH episodes, then snapshot → train-candidate → evaluate-candidate → activate → judge. Compare with the keyboard rehearsal numbers in §6, rather than expecting a single demonstration to suffice.

## 6. Measured timings and results

These come from `python -m dermis cup teaching-study`, plus the follow-ups noted below. Raw data is in `docs/results/teaching/teaching_study.json`.

**Setup:**
- **Stand-in demonstrations:** the slip-reflex controller (SCRIPTED, seeds 400000+), executed with 1 N effort noise. Scripted demonstrations are more consistent than keyboard ones, so treat these results as an **optimistic bound** for fresh human demonstrations.
- **Evaluation:** the development suite (40 scenarios, `configs/cup/dev.yaml`, sustained-damage task), never the held-out suite.
- **Machine:** an 8-core M2 with unrelated background load.

**Recording time:** each demonstration episode lasts 2.2–7.4 s of simulated time. TEACH runs in real time, so add the pre-pour wait the operator chooses, plus the reset.

| New demos | How trained | Dev success | Train wall (trainer) | Eval wall | Terminations |
|---|---|---|---|---|---|
| 0 | Pretrained checkpoint only (`cup_bc_contact`: 160 SCRIPTED train episodes) | 33/40 | – | 44.0 s | drop 3, excess_slip 3, sustained 1 |
| 1 | From scratch | 0/40 | 3.9 s (0.11 s) | 19.2 s | sustained 40 |
| 3 | From scratch | 1/40 | 3.8 s (0.08 s) | 19.3 s | drop 15, sustained 24 |
| 10 | From scratch | 0/40 | 3.8 s (0.15 s) | 15.7 s | sustained 40 |
| 1 | Fine-tune, 40 epochs | 4/40 | 3.8 s | 28.5 s | drop 3, sustained 33 |
| 3 | Fine-tune, 40 epochs | 10/40 | 3.8 s | 32.3 s | sustained 30 |
| 10 | Fine-tune, 40 epochs | 30/40 | 3.9 s | 42.0 s | drop 3, sustained 5, excess_slip 2 |
| 1 / 3 / 10 | Fine-tune, 3 epochs (follow-up) | 11 / 4 / 12 of 40 | about 3.8 s | 31–37 s | mostly damage_break / drop / sustained |
| 10 | **Retrain on 200 existing SCRIPTED + 10 new** (follow-up) | **37/40** | 9.0 s (4.7 s) | 42.4 s | sustained 3 |

**What this shows:**
- **Fresh demonstrations alone are insufficient.** 1–10 demonstrations from scratch reach 0–1 of 40, which is not a working policy.
- **Fine-tuning a working policy on a few demonstrations makes it worse.** The pretrained 33/40 drops to 4–30/40 with 40 epochs, and to 4–12/40 with 3 epochs. With this small network, a handful of episodes overwrites behaviour the pretrained policy had learned.
- **Adding new demonstrations to the existing data and retraining is the workable path.** The whole loop took **about 1 minute**: 0.4 s to snapshot, 9 s to train, 42 s to evaluate on 40 scenarios. The 37/40 vs 33/40 comparison is a single run against a single baseline (trained on a different split), and the new demonstrations came from the same demonstrator type, so it is **not** evidence that the new demonstrations helped.
- **Timing:** training is dominated by process start-up; the training loop itself is 0.1–5 s. Evaluation is the slow step, at 16–44 s for 40 scenarios with 6 workers. Use `--suite` with a smaller dev suite during a live demo if needed.
- **Live-demo recommendation:** show the existing active policy, record new demonstrations, snapshot them *together with* the existing dataset (`cup snapshot --from runs/cup/benchmark/episodes --from data/episodes/cup_teach --name ...`), retrain, evaluate on the development suite, then activate explicitly. Never claim that one demonstration taught the task.

**Ingestion under the simulator:**
- An in-process telemetry thread got only about 3–7 of 73 frames/s while the cup GUI loop ran, because PyBullet holds Python's interpreter lock. Glove TEACH therefore ingests in a child process (`glove/procsource.py`): 62–69 accepted frames/s under the GUI loop, with the newest frame 9.6 ms old at the median and 18 ms at worst.
- With synthetic glove input, the cup GUI loop ran at 0.90–0.97× real time.

## 7. Known limitations

- **No hardware yet.** All glove results are PENDING HARDWARE. The synthetic profiles exercise the code paths: step (held press) and periodic (continuous press).
- **Untuned defaults.** The default `min_separation` (5), stationarity limit (60 µT), deadband, gain and output range are guesses to revisit after the first real session. Don't loosen them just to get a calibration accepted.
- **No clock sync.** Device and host clocks are not synchronised. Raw recordings and episodes are linked by host monotonic receive time only.
- **Source label is an assumption.** REAL HARDWARE is inferred from the serial transport or the operator's confirmation, not verified by a device handshake.
- **Practicality results.** The measurements use scripted stand-ins; real keyboard or glove demonstrations are noisier. Keyboard TEACH was exercised by automated tests and window launches, not by a person at the keyboard.
- **Glove in the cup task.** Glove control in the cup task drives grip effort only. The arm is fixed in that task by design.
