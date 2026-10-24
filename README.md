# Dermis

A tactile glove (ESP32-S3, BMI160 IMUs, MLX90393 magnetometers under deformable magnetic fingertips) to capture demonstrations, and behavioral cloning to learn robotic grasp-lift-place from them.

This repository holds the **hardware-independent core**. It works today without the glove:

**collect demonstrations → save episodes → train a policy → AUTONOMOUS rollouts → held-out evaluation → inspect results**

It also includes a glove ingestion pipeline. The final glove stream plugs into it without changes to the learning code.

> Simulation only. Contact forces are simulated, and magnetic readings are reported as change from baseline, never as force. Nothing here makes clinical or biological claims.

Measured results are in [docs/results.md](docs/results.md). The hardware integration path and the measurements still needed are in [docs/hardware_checklist.md](docs/hardware_checklist.md).

## Setup (macOS or Linux; CPU is enough)

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serial]"        # numpy pandas pybullet torch pyyaml matplotlib jsonschema (+pytest, pyserial)
```

On macOS 27 / Python 3.12, PyBullet 3.2.7 installed from a prebuilt wheel. If pip has to compile it and fails (`'cstring' file not found`, or a zlib `fdopen` error), point the compiler at the SDK's C++ headers:

```sh
SDK=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX2*.sdk | tail -1)
ARCHFLAGS="-arch arm64" CFLAGS="-Dfdopen=fdopen" CXXFLAGS="-isystem $SDK/usr/include/c++/v1" \
  pip install -e ".[dev,serial]"
```

 The GPU is optional: set `device: auto` in a training config to use CUDA or MPS when available. The defaults run on CPU.

## Layout

| Path | Contents |
| --- | --- |
| `src/dermis/sim/` | PyBullet grasp-lift-place env (Franka Panda gripper), scripted expert, rollout loop |
| `src/dermis/bc/`, `data/`, `evaluate.py`, `report.py`, `study.py` | behavioral cloning, episode storage, evaluation and reports |
| `src/dermis/telemetry/` | decoded-frame adapter, ingestion pipeline, sources (replay, synthetic, serial, ...), magnetic baseline calibration |
| `src/dermis/teleop/` | keyboard / glove demonstrations (`demo`) |
| `src/dermis/server/` | TCP telemetry server and the armed TCP teleop path ([docs/tcp_teleop.md](docs/tcp_teleop.md)) |
| `src/dermis/dashboard/` | local web dashboard |
| `src/dermis/cup/` | separate **fragile_cup** experiment: hold a cup under a ramping load proxy; TEACH / AUTONOMOUS live modes ([docs/fragile_cup.md](docs/fragile_cup.md)) |
| `src/dermis/wipe/` | wiping feasibility study (wipe_v1): physics checks, controller comparison, glove pad-trial tools; decision NO-GO ([docs/wipe_feasibility.md](docs/wipe_feasibility.md)) |
| `src/dermis/glove/` | glove bring-up: pod discovery, inspector with labelled trials, two-point grip calibration, armed grip control, process-based ingestion; fragile_cup teaching workflow ([docs/glove_bringup.md](docs/glove_bringup.md)) |
| `configs/` | training, benchmark and suite configs |
| `firmware/` | ESP32-S3 firmware (v0), Dermis Wire v1 spec, schemas, example fixtures, reference decoder |
| `tests/` | pytest suite |
| `docs/` | hardware checklist, TCP teleop guide, results; `docs/studio/` holds design notes from the retired Swift app |
| `runs/`, `data/`, `results/` | generated outputs (git-ignored) |

## Commands

All of these were run while building this repo. Results are in [docs/results.md](docs/results.md).

```sh
python -m dermis smoke-test                                   # ~15 s: 4 scripted demos -> tiny model -> reload -> 2 AUTONOMOUS episodes
python -m dermis collect --source scripted --episodes 200 --out data/episodes/scripted
python -m dermis train --config configs/bc_contact.yaml       # or configs/bc_no_contact.yaml
python -m dermis evaluate --checkpoint runs/bc_contact --suite configs/suites/heldout_v1.json
python -m dermis evaluate --scripted --suite configs/suites/heldout_v1.json   # expert sanity check
python -m dermis compare results/*.json --out results/compare
python -m dermis benchmark --config configs/benchmark.yaml   # collect 200 -> train both -> expert + both on held-out -> table/chart (~5 min)
python -m dermis study --config configs/study.yaml           # data scaling x 3 training seeds, both variants (~15 min)

python -m dermis demo --source keyboard                       # GUI, KEYBOARD demos -> data/episodes/human
python -m dermis replay --episode data/episodes/scripted/scripted-s0000005.json --plot ep5.png [--gui]
python -m dermis rollout --checkpoint runs/bc_contact [--headless]   # watch AUTONOMOUS episodes

python -m dermis stream --source replay --path firmware/v1/examples/decoded_frame.json
python -m dermis stream --source wire-file --path firmware/v1/vectors/capture_2s.bin
python -m dermis stream --source synthetic --faults drop,duplicate,reorder,stale,invalid,missing_pod,config_change,gap
python -m dermis stream --source stdin  < frames.jsonl        # live decoded JSONL
python -m dermis stream --source tcp --port 8765              # live decoded JSONL over TCP
python -m dermis calibrate --source synthetic --frames 300 --out data/calibration/dev.json
python -m dermis dashboard --source synthetic --faults stale,invalid   # http://127.0.0.1:8000
python -m pytest -q                                           # 84 tests (incl. the TCP teleop tests); -m "not slow" skips 2 sim-heavy ones
python -m pytest -q firmware/v1/host/python/test_dermis_wire.py   # 23 existing wire-decoder tests

python -m dermis teleop --source tcp --host 127.0.0.1 --port 8765 --gui     # armed TCP teleop (docs/tcp_teleop.md)
python -m dermis send-test --host 127.0.0.1 --port 8765 --mode synthetic --rate 30
python -m dermis test-tcp-pipeline --headless
```

Every command takes `-h`. Each `collect`, `evaluate`, `benchmark` and `study` run spreads its episodes over worker processes (`--workers`, default half the CPU cores).

## The task

A Franka Panda with its parallel-jaw gripper picks up a 4 cm, 0.2 kg cube, carries it to a target disc, puts it down, opens, and backs away. The episode succeeds after the cube has rested inside the target radius (3.5 cm), with no finger contact, for 15 consecutive ticks. Grasping uses real contact physics: the cube is held only by finger friction and is never attached.

Each episode varies the cube position (x 0.45–0.62 m, y −0.18 to −0.02 m), the target position (x 0.45–0.62 m, y 0.06–0.18 m) and the friction coefficient (0.4–1.0). Ranges are set in `configs/task.yaml` / `TaskConfig`.

**Frames, units, timing.** World frame: z up, origin on the table under the robot base; units are m, s, N, kg. The gripper points straight down. Policy and control run at **30 Hz** (one `step()`); physics runs at **240 Hz**, so each tick is 8 substeps. Contact forces are averaged over a tick's substeps.

**Action** (5 floats, clipped to bounds every tick):

| | Unit | Range | Meaning |
|---|---|---|---|
| vx, vy, vz | m/s | ±0.20 | Desired gripper velocity. Integrated into a position target, which is clamped to the workspace and to 3 cm ahead of the measured gripper, then tracked by IK. |
| opening | m | 0–0.08 | Desired total finger opening |
| effort | N | 2–40 | Finger motor force limit. When closed on the cube, the squeeze is roughly this value. At 2 N the cube cannot be lifted; at 38 N it crosses the force threshold (both verified). |

**Observation** (26 floats, in this order; `OBS_NAMES` in `src/dermis/sim/env.py`):

| Group | Features |
|---|---|
| proprio | gripper xyz (m), gripper velocity (m/s), opening (m), opening velocity (m/s) |
| object | cube position relative to the gripper (m), cube velocity (m/s) |
| target | target xy relative to the gripper, target xy relative to the cube (m) |
| contact | left/right finger normal force on the cube (N, simulated), left/right contact flags |

The observation at tick t is read before the action at tick t. Nothing from later ticks is used. Velocities are finite differences over the previous tick.

**Metrics per episode:** success, completion time, drops (cube released from more than 3 cm up or outside the target), ticks over the force threshold, peak per-finger force, peak gripper–table force, and the termination reason (`success`, `timeout` after 15 s, `object_out_of_bounds`, or `aborted`). `EXCESS_FORCE_N = 30 N` per finger is a **simulated benchmark threshold**, not a biological or clinical measure.

## Data

**Sources and labels.** Demonstrations come from three sources: `scripted` (the expert; it reads privileged sim state and is never used to evaluate learned policies), `keyboard`, and `glove`. Displayed labels are SCRIPTED, KEYBOARD, GLOVE, REPLAY (recorded data or playback), AUTONOMOUS (a learned policy acting) and SYNTHETIC (generated telemetry, for software tests only).

**Episode files.** Each episode is `<id>.npz` plus `<id>.json` (`src/dermis/data/episodes.py`):

- Arrays:
  - `obs[t]` and `actions[t]`: the action actually issued, after clipping.
  - `expert_actions[t]`: scripted episodes only. These are the clean labels; when execution noise is added for data diversity, the issued action differs from them.
  - `t`
  - Contact log: `force_l`, `force_r`, `table_force`, `held`.
  - `obj_pos`: privileged, for inspection only.
  - Optional `tags[t]`: teleop status per tick.
- Metadata: ID, source, dt, full task config, scenario and seed, outcome and termination, feature and action names, calibration and telemetry config IDs.
- Persistence: `EpisodeStore` handles saving and loading. Swap it to change where episodes are stored.

**Scripted expert.** Every phase change depends only on quantities the policy can also observe. An earlier version used hidden tick counters, and cloned policies failed (5/20 on the dev suite) because identical observations needed different actions. Data collection adds Gaussian velocity noise (σ = 0.03 m/s) to the executed actions, while the clean expert labels are recorded alongside (DART-style). This gives the policy examples of recovering from drift.

**Replay.** REPLAY re-executes an episode's issued actions open-loop from its saved scenario, for inspection. The simulator is deterministic, so replay reproduces the recording exactly (measured deviation 0). REPLAY is not autonomy. AUTONOMOUS means a checkpoint chooses each action from the current observation (`rollout`, `evaluate`).

## Behavioral cloning

Code is in `src/dermis/bc/`. The policy is an MLP (2×256, ReLU) with a tanh output. Actions are scaled to [−1, 1] from their bounds, and the loss is MSE.

- **History.** Configurable short history: `history: 2` stacks t−1 and t. At the start of an episode the first observation is repeated, in training and at run time alike.
- **Splits.** Train/val split by whole episode, with a seeded split. No history window crosses an episode boundary.
- **Normalization.** Fitted on training episodes only.
- **Reproducibility.** Seeded and deterministic: retraining gives identical predictions (tested).
- **Checkpoint.** The checkpoint directory holds `model.pt`, `metadata.json` (feature names and order, normalizer, action bounds, full training config, provenance: data directory, data digest, train/val episode IDs and seeds, git commit, torch version) and `train_log.csv` (training log and validation metrics).

At run time the policy uses only robot/simulator observations, not the glove. The plan is glove-as-expert-controller: a person teleoperates the simulated robot with the glove, and the robot's observations and actions are recorded as demonstrations. Learning from natural human manipulation without the robot in the loop would need hand/object tracking, retargeting and contact alignment. It is not implemented.

**Contact experiment.** `bc_no_contact` and `bc_contact` use the same episodes, split, architecture, training budget and evaluation scenarios. They differ only in `feature_groups`. The experiment tests whether *simulated* contact observations help, and says nothing about real glove data. Evaluation suites use seeds of 1,000,000 and above (dev) and 2,000,000 and above (held-out). Training uses seeds from 0. Evaluation refuses to run if the seed ranges overlap.

## Glove ingestion

Code is in `src/dermis/telemetry/`. Every source passes through the same steps:

```
source -> RawMessage -> adapter (decoded_frame.schema.json -> TelemetryFrame) -> TelemetryPipeline -> Snapshot
```

**Sources:**

| Source | Label | What it reads |
|---|---|---|
| `replay` | REPLAY | Saved decoded JSON or JSONL frames |
| `wire-file` | REPLAY | Dermis Wire v1 binary capture, read with the existing reference decoder `firmware/v1/host/python/dermis_wire.py` |
| `synthetic` | SYNTHETIC | Generated frames with deterministic fault injection |
| `stdin`, `tcp` | GLOVE | Live decoded JSONL |
| `serial` | GLOVE | Live USB serial (JSON lines, or Dermis Wire with `--wire`). Unverified without hardware. |

**Pipeline behaviour:**

- **Validity.** Invalid or null readings become NaN with a false mask. A real zero stays zero.
- **Staleness.** STALE readings are usable but never fresh, even when `valid` is true. TEMP_STALE affects only temperature, never the magnetic or IMU values.
- **Ordering.** Duplicates and out-of-order frames are dropped and counted. Frame-counter gaps are counted, and the u32 counter wrap is handled. Large backward jumps and device changes reset the stream. Config changes discard held values.
- **Bounded hold.** Each channel carries an age (device clock) and a hold of the last fresh value that lasts at most 50 ms. This hold is the only imputation.
- **Watchdog.** If no data arrives for 250 ms of host time, the source is marked stale.
- **Raw data kept.** The raw decoded frame is kept for recording.
- **Other formats.** A stream in another format needs a new adapter only.

**Calibration** (`telemetry/calibration.py`):

- Unloaded baseline capture, using fresh samples only, with sample-count and noise checks.
- Per-channel baseline subtraction.
- Mount-string rotation into the pod frame, following SPEC §8.
- Configurable normalization scale.
- Versioned JSON with an ID, tied to `(device_id, config_id)`.
- Outputs are |ΔB| in µT and unitless normalized deformation, never newtons.

**Teleop** (`teleop/mapping.py`):

- `TeleopMapping` neutralizes the action (zero velocity, grip held) when there is no data, when a required channel is stale beyond the hold, or when the calibration doesn't match the stream's config. Paused ticks are recorded with their status.
- The provisional `ExperimentalTiltGripMapping` (hand tilt → xy velocity; magnetic change at the pads → grip) refuses to start without `--experimental-glove-teleop`.
- Position is never computed by double-integrating IMU acceleration. `PoseProvider` is the hook for an external tracker.
- The armed TCP teleop path (NDJSON server, controller with arm/disarm and pause states) lives in `server/` (`tcp_server.py`, `teleop/{session,controller,mapping,calibration}.py`). It is documented in `docs/tcp_teleop.md`.

**Swapping to the final glove.** Keep the command and change only the source flags. For example, `python -m dermis dashboard --source synthetic` becomes `python -m dermis dashboard --source serial --serial-port /dev/cu.usbmodemXXXX [--wire]`, or use `--source tcp` / `--source stdin` with a bridge. If the stream's JSON shape differs from `decoded_frame.schema.json`, add an adapter. The other steps are in [docs/hardware_checklist.md](docs/hardware_checklist.md).

## Dashboard

`python -m dermis dashboard --source …` serves a single local page on 127.0.0.1. It shows:

- The current source and mode labels.
- Per-pod magnetic lanes, with stale samples shaded amber and invalid or missing samples shaded red, and IMU status.
- Pipeline counters and events.
- Raw telemetry recording to JSONL, and baseline capture.
- The episode list with REPLAY inspection plots.
- Training runs with loss curves, plus a small job runner for scripted collection and the smoke test. `JobManager` is the extension point for orchestration.
- Evaluation results and comparisons.

It shows only data that exists on disk or in the stream: no hardcoded metrics and no confidence scores.

For launch options, the judge walkthrough, metric definitions and measured overhead of the extended console, see [docs/dashboard.md](docs/dashboard.md).

## fragile_cup (separate experiment)

The question: do simulated finger–object contact observations help a learned grip adapt when a held cup's load ramps up? The arm is fixed and only the grip effort is controlled. The cup is rigid, its "contents" are an external-force load proxy, and "damage" is a simulated contact-load threshold. Its metrics are reported separately from the grasp-lift-place benchmark and never merged with it. Full details, including the audit of the original pipeline, are in [docs/fragile_cup.md](docs/fragile_cup.md).

```sh
python -m dermis cup check-task                                      # fixed efforts vs feedback controllers (dev suite)
python -m dermis cup benchmark --config configs/cup/benchmark.yaml   # collect -> train both -> held-out eval, 3 seeds
python -m dermis cup smoke-test                                      # ~25 s pipeline check
python -m dermis cup teach                                           # GUI: human grip (keyboard), recorded
python -m dermis cup judge                                           # GUI: trained checkpoints; Pour / Reset / paired comparison
```

Held-out results (60 scenarios, 3 training seeds):
- **Primary arm** (slip-reflex demonstrator, which uses no contact): BC with contact averaged 0.944 success, BC without contact 0.922. The per-seed results overlap, so there is no reliable difference.
- **Secondary arm** (hybrid demonstrator, which uses contact): BC with contact 1.000, BC without contact 0.978 (4 sustained-damage failures out of 180).
- **Fixed grip:** a constant 8 N grip reaches 58/60 on the same scenarios, so this task leaves little room for adaptation.

## Limitations and unverified items

- No glove hardware was available. Serial ingestion, live calibration, glove teleoperation quality and every real-sensor claim are unverified. All telemetry tests use the supplied fixtures or SYNTHETIC data.
- Keyboard demos and the GUI were implemented but not driven by hand in this session. The keyboard-to-action mapping and the recording path are exercised by code shared with the tested glove and scripted paths.
- Clock sync (SYNC_PING → `t_host_ns`) is not implemented. Watchdogs use host receive time.
- The task is deliberately narrow and fully observable: the object pose is given directly. With 200 demonstrations both BC variants reach 50/50. With 25–50 demonstrations, *simulated* contact observations clearly help (0.61 vs 0.33 and 0.83 vs 0.14 mean held-out success over 3 seeds). That is evidence about this simulator and task only; see [docs/results.md](docs/results.md).
- Contact forces come from PyBullet's rigid-body solver: plausible, not validated.
- The old top-level prototype scripts (`sim_env.py`, `train_model.py`, …) and the Swift app were removed; they remain in git history.
