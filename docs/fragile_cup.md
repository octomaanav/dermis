# fragile_cup: does simulated touch help a learned grip adapt to load?

**Simulation benchmark.** The cup is a rigid PyBullet cylinder that never deforms. "Contents" are an external downward force on the cup, a *load proxy*, not a fluid. "Damage" is a *simulated threshold* on fingertip contact load. Nothing here validates real glove measurements, real cup mechanics or prosthetic performance.

This experiment is separate from the original grasp-lift-place benchmark: separate code (`src/dermis/cup/`), configs (`configs/cup/`), outputs (`runs/cup/`, `results/cup/`), CLI namespace (`python -m dermis cup …`) and metrics. The original environment, checkpoints and results are unchanged.

## 1. Audit of the existing training pipeline (grasp-lift-place)

| Stage | Code path |
|---|---|
| Scripted expert | `src/dermis/sim/expert.py` `ScriptedExpert.act()`. It reads privileged state through `env.privileged_state()`. `src/dermis/sim/rollout.py` `scripted_controller()` adds Gaussian velocity noise to the *issued* action and keeps the clean label. `collect_scripted()` / `collect_scripted_parallel()` save episodes. |
| Recording | `rollout.run_episode()`: `obs = env.observe()`, then controller, then `clip_action`, then `EpisodeRecorder.record(obs, action, t, expert_label, tag)` (`src/dermis/data/episodes.py`), then `env.step(action)`, then `log_step(info)`. The observation at t is recorded before the action at t is applied. |
| Contact extraction | `src/dermis/sim/env.py` `_contact_forces()`, called once per physics substep inside `step()` and averaged over the 8 substeps. |
| Training | `src/dermis/bc/train.py` `train()`: whole-episode split (`bc/dataset.py split_episodes`), normalizer fitted on training rows only, history windows inside one episode (`history_windows`), MLP with tanh output (`bc/model.py MLPPolicy`), MSE on actions scaled to [−1, 1], best-validation checkpoint (`save_checkpoint`). |
| Autonomous inference | `bc/model.py` `BCPolicy.act(obs)` keeps its own history buffer. `src/dermis/evaluate.py` `_run_chunk` calls it as `lambda obs, _env: (policy.act(obs), None)`, so the policy gets the observation only. |

**Contact features in the original task** (`OBS_GROUPS["contact"]`):

| Feature | Unit | What it measures |
|---|---|---|
| `force_l`, `force_r` | N | Sum of PyBullet's scalar *normal* force over the contact points between that finger link (joint index 9 or 10) and the **object**. Averaged over substeps. Finger–object only. No friction (shear) component, and no projection: it is the solver's normal-force magnitude. |
| `contact_l`, `contact_r` | 0/1 | That finger's normal force on the object > 0.05 N. Finger–object only. |

Robot–table contact (`table_force`, any robot link against the plane) is logged in episodes but is **not** an observation.

**Do the contact features indirectly reveal task phase or success?** They reveal phase.
- Both flags on means grasping, carrying or lowering. Both off means approaching, or released.
- While gripping, `force_l` / `force_r` ≈ the effort limit the expert itself commanded (10 N open, 15 N closed). So they partly echo the expert's own previous command, and that command is not otherwise in the observation.

They don't encode success directly. Success is "released, at rest, inside the target", and its position part is visible to both policies through the object and target features. The contact advantage measured in that study (`docs/results.md`, 25–50 demos) may therefore partly come from a clean "holding / not holding" phase signal rather than from tactile information in a richer sense. That is a plausible explanation, not a verified one.

**Synthetic glove streams vs. training data.** Every behavioral-cloning dataset in this repo, for both tasks, is a set of **SCRIPTED simulator episodes** (`source: scripted`): robot/simulator observations and the scripted controller's commands. Glove-shaped telemetry comes from SYNTHETIC frames (`telemetry/sources.py SyntheticSource`), the supplied fixture frames, or TCP/serial streams. It feeds only the telemetry pipeline, calibration, dashboard and the experimental teleoperation paths. No glove or synthetic-glove episode is in any training set. The only glove-sourced episodes ever created were in temporary test directories.

## 2. Task

Code: `src/dermis/cup/env.py`, `schedule.py`.

**Setup.** The Panda arm is fixed; only the grip is controlled. The cup is a rigid cylinder (radius 3 cm, height 9 cm). Its mass (0.08–0.15 kg) and friction (0.35–0.9; finger pads are 1.0, so this is the effective coefficient) are sampled per scenario.

**Identical setup for every policy** (`CupHoldEnv.reset`):
1. The open gripper straddles the cup standing on the table, 1 cm above its centre of mass.
2. The fingers close at 8 N for 0.5 s.
3. The arm lifts 10 cm over 1 s, then settles for 0.5 s.
4. The grasp is checked: both finger normals > 1 N and the cup within 3 cm of the grasp point. Otherwise the episode ends as `setup_failed`, which never happened in any run.
5. The arm joints are then locked, and the policy's first command replaces the 8 N setup effort.

The setup is scripted, not recorded and not policy-controlled. It uses no constraints, so the cup is held only by friction (tested).

**Action.** One float: finger effort limit, N, in [0.5, 40]. The fingers always close, so the squeeze ≈ the effort limit (tested within 0.3 N).

**Load proxy** (`schedule.py`): an external downward force on the cup's centre of mass. Each scenario has 2–4 ramp segments (0.6–2 N/s) separated by 0.5–1.5 s pauses, up to a peak of 2–8 N. Up to 3 bounded disturbances (±1.5 N, 0.15–0.4 s) are added on top. Evaluation pours at t = 1 s; in the live modes, pouring starts when the judge presses Pour. Episodes end 2 s after the schedule finishes.

**Contact forces** (`CupHoldEnv._finger_forces`):
- For each finger link, the force on the cup is summed over that finger's contact points, F = −(fₙ·n + f₁·d₁ + f₂·d₂), using PyBullet's normal on the cup and both friction directions.
- Each substep is queried once and grouped by link, so nothing is counted twice. That is tested: the per-finger vectors sum to the total robot→cup force, and every contact point is on a finger link.
- F is split against the finger's measured closing axis (from the finger link positions) into **normal** (squeeze) and **shear** (tangential), plus shear's world-z component.
- Palm contacts are logged separately and are never finger features.
- Verified: during pauses, shear_z(L) + shear_z(R) = m·g + applied load to within 0.05 N (`tests/test_cup_env.py`).

**Observations** (`OBS_NAMES`):

| Group | Features | Units | Contact? |
|---|---|---|---|
| gripper | opening, opening velocity, previous effort command | m, m/s, N | no |
| object | cup position relative to the grasp point (x, y, z); cup velocity relative to the gripper | m, m/s | no |
| contact | normal L/R, shear_z L/R, shear magnitude L/R, contact flags L/R | N, N, N, 0/1 | yes: finger–cup only |

**Not observed** by any policy: applied load, load schedule, future disturbances, friction, mass, damage accumulator, phase, expert state.
- The contact policy can infer the *current* load from shear_z. That is the physical quantity a tactile sensor would measure, not a privileged leak.
- The previous effort command is given to both policies equally.

**Outcomes** (`_update_outcome`, checked every tick in this order):

| Termination | Definition |
|---|---|
| `damage_break` | Either finger's normal force > 25 N (simulated threshold) |
| `damage_sustained` | (v2 task only) ∫ max(0, normal − 8 N) dt > 15 N·s for the more loaded finger (simulated sustained-squeeze limit) |
| `drop` | Cup slid > 3 cm down relative to the gripper, or no finger contact for 3 ticks |
| `success` | Schedule plus 2 s final hold completed, with total slip ≤ 1.5 cm |
| `excess_slip` | Schedule completed, but slip > 1.5 cm |
| `timeout` | Live mode only: no Pour within 120 s |
| `setup_failed` / `aborted` | Setup grasp invalid / operator abort |

Every result also records the phase in which a failure happened (pre-pour, ramp, pause, final hold).

## 3. Is the task meaningful? (checked before any training, dev suite only)

Command: `python -m dermis cup check-task`, run on `configs/cup/dev.yaml` (40 scenarios, seeds 1,000,000+).

**v1 (instantaneous 25 N break threshold only).** A fixed grip solves it, so v1 is not a meaningful adaptation test:

| Fixed effort (N) | 2 | 4 | 6 | 8 | 10 | 12 | 14–24 |
|---|---|---|---|---|---|---|---|
| Success /40 | 0 | 12 | 23 | 33 | 38 | 39 | **40** |

The feedback controllers also succeed: slip reflex 40/40, grip-load expert 40/40, oracle 40/40.

This follows from the physics. With an effort-limited gripper the squeeze equals the commanded effort whatever the load. So any fixed effort above the worst-case requirement, about (m·g + load) / (2μ) ≈ 14 N, and below the break threshold works everywhere.

**Recommended physically motivated adjustment (v2, adopted):** add a sustained-squeeze limit. Thin-walled cups crush or creep under prolonged grip, so the cup tolerates 8 N per finger indefinitely, and squeeze above that uses up a 15 N·s budget. These numbers were chosen as plausible defaults before running the check; they are simulated, not measured. Results under v2:

| Controller | Success /40 (dev) |
|---|---|
| fixed 2 / 4 / 6 / 8 / 10 N | 0 / 12 / 23 / **33** / 9 |
| fixed 12–24 N | 0 (all sustained damage) |
| slip reflex (no contact) | 39 |
| hybrid expert (contact + slip) | 39 |
| grip-load expert (contact, assumes worst-case μ = 0.35) | 21 |
| oracle (privileged load and friction) | 38 |

On dev, no fixed effort solves every v2 scenario. **On the held-out suite, however, a fixed 8 N grip reaches 58/60.** 8 N is exactly the sustained-safe level, so v2 is only marginally harder than a well-chosen constant. That is a real limitation of this benchmark. A stronger variant would widen the load range or lower the safe level relative to the worst-case requirement. I did not do that, to avoid tuning the task until the learned policies look good.

**Demonstrator choice.** This was decided on dev before any held-out run, with the rule fixed in advance: highest dev success, ties to the simpler controller.
- The slip reflex and hybrid expert tied at 39/40, so the **primary demonstrator is the slip reflex**.
- Its command depends only on non-contact observations, so its demonstrations contain no tactile-specific behaviour to copy.
- A clearly labelled **secondary arm** clones the hybrid expert, which uses contact shear.

## 4. Results (held-out suite: 60 scenarios, seeds 2,000,000+, v2 task)

Commands: `python -m dermis cup benchmark --config configs/cup/benchmark.yaml` and `--config configs/cup/benchmark_hybrid.yaml`.
- Each arm collects 200 SCRIPTED episodes (seeds 0–199), executed with 1 N Gaussian effort noise and clean labels recorded.
- Both variants use the same whole-episode split (160 train / 40 val), MLP 128×128, history 4, 40 epochs and the same scenarios. Training seeds are 0, 1 and 2.
- Scenario hashes were identical across every row of both reports.
- Training took 5–14 s per policy on CPU (24.9k–29.7k rows).

**Reference controllers** (same 60 scenarios):

| Controller | Success | Terminations |
|---|---|---|
| fixed 8 N | 58/60 | drop 2 |
| fixed 10 N | 15/60 | sustained damage 45 |
| fixed 14 N | 0/60 | sustained damage 60 |
| slip reflex (primary demonstrator) | 60/60 | – |
| hybrid expert (secondary demonstrator) | 60/60 | – |
| grip-load expert | 38/60 | sustained damage 22 |
| oracle (privileged) | 59/60 | sustained damage 1 |

**Learned policies, AUTONOMOUS, across training seeds:**

| Arm | Policy | Success per seed | Mean | All terminations (3 × 60) |
|---|---|---|---|---|
| Primary (slip-reflex demos) | BC, no contact | 54, 56, 56 | 0.922 | success 166, excess_slip 8, drop 4, damage_sustained 2 |
| Primary | BC, with contact | 58, 59, 53 | 0.944 | success 170, excess_slip 8, drop 2 |
| Secondary (hybrid demos) | BC, no contact | 58, 59, 59 | 0.978 | success 176, damage_sustained 4 |
| Secondary | BC, with contact | 60, 60, 60 | 1.000 | success 180 |

Every learned-policy failure happened in the **final hold** (see the failure-state distributions in `docs/results/cup/benchmark_*.md`).

**Interpretation:**
- **Primary arm:** no reliable difference. The per-seed ranges overlap (53–59 vs 54–56). This is expected: the demonstrator's commands are a function of non-contact observations only.
- **Secondary arm:** the contact-aware policy was perfect on all 3 seeds, and the no-contact policy failed 1–2 of 60 per seed. Each of those 4 failures was sustained damage during the final hold, consistent with over-gripping when the load is not directly sensed. The effect is small (4/180) and comes from one task and one demonstrator.
- **Fixed-grip baseline:** a fixed 8 N grip (58/60) matches or beats most learned policies here. Simulated contact sensing helped only marginally, and only when the demonstrator itself used contact.
- None of this says anything about real glove data or real cups.

## 5. Live modes (GUI)

Physics always runs headless, exactly as in evaluation. The window (`src/dermis/cup/viewer.py`) is a separate, display-only PyBullet client that copies the pose each display frame, so the GUI cannot change what a policy experiences.

On this Mac each window redraw takes about 80 ms: OpenGL runs through a Metal translation layer, and the faster `--opengl2` renderer segfaulted. The loop therefore runs physics and the policy at 30 Hz to keep up with the wall clock, and refreshes the window as often as it can. Measured: **control 0.81–0.89× real time, window about 3 fps**, under heavy unrelated system load. The terminal prints this measured speed every 5 s.

```sh
python -m dermis cup teach                      # TEACH: keyboard grip, records to data/episodes/cup_teach
python -m dermis cup judge                      # AUTONOMOUS: default checkpoints runs/cup/benchmark/runs/cup_bc_{no_contact,contact}
python -m dermis cup judge --checkpoint runs/cup/benchmark_hybrid/runs/cup_bc_contact --checkpoint runs/cup/benchmark_hybrid/runs/cup_bc_no_contact
python -m dermis cup evaluate --gui --checkpoint runs/cup/benchmark_hybrid/runs/cup_bc_contact --suite configs/cup/dev.yaml
```

**TEACH controls** (click the window first):

| Key or button | Action |
|---|---|
| Up/Down (W/S) | Raise/lower effort while held |
| 1–9 | Effort preset of 4–36 N |
| P / "Pour" | Start the load ramp |
| R / "Reset" | Restart the same scenario |
| N / "Next scenario" | Move to a new seed |
| SPACE / "Record on/off" | Toggle recording |

The banner shows the input source and recording status. Recorded episodes have source KEYBOARD, the issued effort at every tick, and the full scenario. The future glove input uses the same recorder: `--input glove --experimental-glove-teleop --calibration …`. It is unverified without hardware.

**AUTONOMOUS controls:**

| Key or button | Action |
|---|---|
| P / "Pour" | Start the load ramp |
| R / "Reset" | Restart the same scenario |
| N / "Next scenario" | Move to a new seed |
| T / "Switch policy" | Change the active checkpoint |
| C / "Paired comparison" | Run every loaded checkpoint on the current scenario with the pour at t = 1 s, one after another, then show and print both outcomes |

- No human motor-input object exists in this mode. Effort keys are counted and dropped (tested).
- The banner names the checkpoint, its feature groups and the number and source of its training episodes. It also states that TEACH recordings are not in the checkpoint until you retrain.

## 6. Manual test procedure

1. `python -m dermis cup judge`. Expect a red AUTONOMOUS banner reading "human grip input: DISCONNECTED", and the cup held at about 8 N.
2. Hold Up/W and press 9. The effort must not change: those keys never reach the robot in this mode. The terminal's 5-second status line still shows the policy's effort.
3. Press P (or "Pour"). The red arrow beside the cup grows with the applied load. The numbers line shows the load, fingertip normal and shear_z (shear_z rising with the load), and the slip.
4. Watch the outcome line turn green (`success`) or red, with the reason.
5. Press R. The same cup, friction and schedule restart identically. Press P at roughly the same moment to repeat it.
6. Press C. Both checkpoints run on the current scenario with the same pour time. The terminal prints a `[PAIRED]` line with both terminations, and the window shows "last comparison".
7. Press T to switch the active policy, then repeat steps 3–5.
8. Quit, then run `python -m dermis cup teach`. Press SPACE (recording ON), then P, and control the grip with Up/Down or the number keys until the episode ends. The terminal reports the saved episode in `data/episodes/cup_teach`.
9. Optional: train on your own episodes with `python -m dermis cup train --config configs/cup/bc_contact.yaml --data data/episodes/cup_teach --out runs/cup/my_policy`, then `python -m dermis cup judge --checkpoint runs/cup/my_policy`.

## 7. Tests and smoke run

- `tests/test_cup_env.py`:
  - holding without attachment (no constraints, free body)
  - friction dependence (0.5 N drops the cup)
  - force balance (shear_z sum = m·g + load)
  - no double counting
  - load changes the outcome (3 N holds unloaded, drops at 6 N load)
  - instant and sustained damage detection
  - identical resets
  - no privileged observations
  - bounded, deterministic schedules
- `tests/test_cup_live.py`:
  - policy output drives the finger motors (measured squeeze and joint motor force follow it)
  - human keys cannot change effort in AUTONOMOUS but do in TEACH
  - TEACH records the issued commands with source KEYBOARD
  - paired comparison and reset reproduce identical conditions
  - evaluation scenario hashes are identical across controllers
  - suites reject training seeds
  - scripted collection and training round trip
- `python -m dermis cup smoke-test` (latest run): 22.9 s total. 6 slip-reflex SCRIPTED demos, both variants trained for 3 epochs (1.44 s and 0.06 s wall, mostly startup), 3 AUTONOMOUS episodes each on identical scenarios (0/3 for both, all sustained damage, which is expected after 3 epochs). This is a pipeline check, not a performance claim.
