# Autonomous fragile-cup demo (trained on SCRIPTED SIMULATOR demonstrations)

**Checkpoint:** `runs/cup/autonomous_demo/cup_autonomous_demo_v1`. A copy of its metadata is in `docs/results/cup_autonomous_demo/checkpoint_metadata.json`.

This is a simulation of the fragile_cup task (Panda gripper, rigid cup, external-force load proxy, simulated damage thresholds). The model was trained **only on SCRIPTED simulator demonstrations**, with no glove or human data. Existing benchmark checkpoints and results are unchanged.

## 1. Existing compatible checkpoints

All results are on the held-out suite `cup_heldout_v1` (60 scenarios), from the original benchmark runs:

| Checkpoint | Demonstrator (SCRIPTED) | Inputs | Held-out |
|---|---|---|---|
| `runs/cup/benchmark_hybrid/runs/cup_bc_contact` (+ `_s1`, `_s2`) | hybrid_expert | gripper + object + contact | 60/60, 60/60, 60/60 |
| `runs/cup/benchmark_hybrid/runs/cup_bc_no_contact` (+ `_s1`, `_s2`) | hybrid_expert | gripper + object | 58, 59, 59 /60 |
| `runs/cup/benchmark/runs/cup_bc_contact` (+ `_s1`, `_s2`) | slip_reflex | gripper + object + contact | 58, 59, 53 /60 |
| `runs/cup/benchmark/runs/cup_bc_no_contact` (+ `_s1`, `_s2`) | slip_reflex | gripper + object | 54, 56, 56 /60 |
| `runs/cup/smoke/*`, `runs/cup/teaching_study/*` | slip_reflex | – | pipeline checks / teaching study only |

None of these recorded an explicit task version, so the demo checkpoint is a deterministic retrain of the best configuration with complete metadata.

## 2–5. Data, inspection, training, checkpoint

- **Demonstrations:**
  - 200 episodes, `runs/cup/benchmark_hybrid/episodes`, generated with the existing recorder by the scripted `hybrid_expert` (grip-load coupling plus slip reflex) in the actual environment, task v2 (`instant+sustained`).
  - Each episode was executed with 1 N Gaussian effort noise; the clean expert labels are stored separately (DART-style).
  - As executed, only 38/200 succeeded (144 sustained-damage, 11 drop, 4 excess-slip, 3 instant-damage), because of the injected noise. The policy learns the clean labels.
- **Inspection** (`python -m dermis cup inspect-episodes --data runs/cup/benchmark_hybrid/episodes --n 6`; results in `episode_inspection.json`):
  - The first observation shows the 8 N setup effort in all episodes, so observations are read before any action.
  - The effort in observation t+1 equals the action issued at t exactly (maximum error 0.0 N).
  - Logged contact equals the next observation's contact exactly.
  - Labels differ from issued efforts with about 1 N standard deviation, as the injected noise intends.
  - Force balance in quasi-static pauses: the vertical friction of both fingers equals m·g + load to within 0.015 N (p95) in 4 of 5 episodes. The fifth was **slipping** (it ended with 25 mm of excess slip), at a p95 error of 0.157 N.
  - The tool's strict 0.15 N pass line therefore reports `pass: false`. That is shown as measured, not loosened: it reflects kinetic, not static, friction during the slip.
- **Training:**
  - `configs/cup/bc_contact.yaml`: MLP 128×128, history 4, 40 epochs, seed 0.
  - Whole-episode split: 160 train / 40 validation.
  - Normalizer fitted on training episodes only.
  - 6.8 s on CPU. Best validation effort MAE 0.09 N.
- **Checkpoint contents** (`model.pt` + `metadata.json` + `train_log.csv`):
  - **Features:** 68 names in order, oldest frame to newest (`opening@t-3` … `contact_r@t`), with history 4.
  - **Normalizer:** mean and standard deviation, fitted on training data only.
  - **Action:** `effort`, bounds [0.5, 40] N, with written definitions of every action and observation.
  - **Task:** `task_version = fragile_cup/instant+sustained/cfg-bf13a0b7c200`, plus the full `task_config`.
  - **Provenance:** data directory, data digest `e308b4403ef135bb`, train/val episode IDs and seeds, sources `["scripted"]`, controller `hybrid_expert`, git commit, torch version.

## 6. Held-out evaluation (actual outcomes)

```sh
python -m dermis cup evaluate --checkpoint runs/cup/autonomous_demo/cup_autonomous_demo_v1 \
    --suite configs/cup/heldout.yaml --out runs/cup/autonomous_demo/heldout_eval.json --workers 6
```

**60/60 success** with no failures. Peak per-finger normal force: median 8.0 N, max 11.9 N. Final slip median 0.0 mm. Wall time 84 s.

The per-episode outcomes are identical to the original seed-0 checkpoint, so the retraining is reproducible. As the fragile_cup docs report, a fixed 8 N grip also reaches 58/60 on this suite: the task leaves little room for a learned controller to show more than imitation.

## 7. Watch it, or run the same checkpoint headless

```sh
# GUI: AUTONOMOUS; P / "Add load (P)" starts the load ramp, R resets the same scenario, N next, C paired comparison
python -m dermis cup judge --checkpoint runs/cup/autonomous_demo/cup_autonomous_demo_v1
# headless, same checkpoint, one scenario, per-tick physics trace + plot
python -m dermis cup worker --checkpoint runs/cup/autonomous_demo/cup_autonomous_demo_v1 --seed 2000000 \
    --out runs/cup/autonomous_demo/trace.jsonl --plot runs/cup/autonomous_demo/trace.png
# headless, whole suite (as above)
python -m dermis cup evaluate --checkpoint runs/cup/autonomous_demo/cup_autonomous_demo_v1 --suite configs/cup/heldout.yaml
```

**GUI speed:** the window is a display-only client. Physics and outcomes are identical to headless runs. On this Mac, under an unrelated system load average of 15–19, the GUI loop ran at 0.55–0.71× real time with the window at about 2 fps, because each PyBullet GUI text redraw costs 80–130 ms. Use the headless worker for anything quantitative.

## 8. Human commands are disconnected in AUTONOMOUS

Verified by `tests/test_cup_autonomous_demo.py::test_real_checkpoint_ignores_human_commands_and_exposes_state`:
- With the real checkpoint, 60 ticks of mashed grip keys ("up", "w", "9") were all counted as ignored.
- The applied efforts equal an independent copy of the checkpoint's outputs to within 1e-5 N.
- No human input object exists in AUTONOMOUS mode; Add load, Reset and Next only change the environment.

## 9. Headless worker state for visualization

`cup worker` writes one JSON line per tick:
- **Robot:** joint angles (arm j0–j6, fingers j9–j10) and gripper position.
- **Cup:** position and quaternion.
- **Contact:** per finger (`normal_l/r`, `shear_z_l/r`, `shear_l/r`) and `normal_total` = L + R, all SIMULATED.
- **Load:** applied load and remaining ramp time; schedule ID; phase; effort command.
- **Status:** the HoldStatus fields.
- **Events:** `load_start` and `termination`.

A header line holds the task config, the full scenario (including the load schedule) and its `schedule_id`. `LiveSession.snapshot()` also adds `robot_joints`, `cup_pose`, `finger_normal_total_n`, `hold_status`, `schedule_id`, `ramp_remaining_s` and `checkpoint`; existing keys are unchanged. Evaluation results now store `schedule_id` per episode.

## Visualization (GUI window)

- **Framing:** gripper, whole cup and the ground, with an orange catch disc on the ground plane.
- **Load fill:** a translucent blue fill inside the cup, attached to the cup's pose. Its level is applied load / 9.5 N (the maximum configured peak plus disturbance). It is labelled **Simulated load: X N** with the remaining ramp time. It is a picture of the backend's force, not a fluid.
- **Overlay:**
  - controller mode and checkpoint;
  - fingertip normal force **per finger** (L, R) and **total** (L + R);
  - cup displacement down relative to the hand, in cm;
  - movement status, load status and the hold timer against the required duration, then the outcome;
  - a fixed line stating the thresholds are SIMULATED and the cup is rigid.
- **Charts:** separate panels for applied load, per-finger contact, displacement and status, from `cup worker --plot`.

**Status definitions** (`src/dermis/cup/status.py`, computed every control tick from measured state):

| Status | Rule |
|---|---|
| HOLDING | Within the grasp region and not slipping |
| SLIPPING | Downward speed relative to the gripper > 3 mm/s for 5 ticks; clears after 10 ticks < 1 mm/s (hysteresis) |
| DROPPED | The benchmark drop rule: slip > 3 cm, or no finger contact for 3 ticks |
| EXCESSIVE LOAD | Per-finger normal force > 25 N, or the sustained 15 N·s budget exceeded. Shown **separately** from movement. A warning appears above half the budget. |
| SUCCESS | The benchmark's success termination |

**Explained discrepancy:** `excess_slip` (total slip > 1.5 cm, decided at the end) can follow a slow creep that stays below the SLIPPING speed threshold. The cm readout makes that visible.

## Diagnostic runs (identical scenario: seed 2,000,000, schedule `eb249327a36b`; not selected results)

| Controller | Outcome | Statuses seen | Peak normal per finger |
|---|---|---|---|
| fixed 2 N (insufficient) | **drop** at 3.07 s, during the load ramp | HOLDING → DROPPED (fell too fast for SLIPPING) | 2.0 N |
| fixed 3 N (insufficient) | **drop** at about 5.0 s | HOLDING → SLIPPING (≈4.9 s) → DROPPED | 3.0 N |
| fixed 4 N | success (this scenario is light) | HOLDING | 4.0 N |
| hybrid_expert (scripted feedback) | success | HOLDING | 7.95 N |
| demo checkpoint | success (matches its held-out record for this seed) | HOLDING | 7.95 N |

Plots and summaries are in `docs/results/cup_autonomous_demo/`. `runs/cup/autonomous_demo/diagnostic/verify.py` recomputed, for every tick of every trace:
- whether DROPPED matches the measured slip and the drop rule;
- whether EXCESSIVE LOAD matches the measured contact;
- the applied load from the saved schedule, independently;
- the timer and the required hold duration;
- the final outcome against the environment's termination.

Result: **0 mismatches in all 5 traces**. No run attaches, teleports or resets the cup; the tests assert zero constraints.

## Remaining limitations

- **GUI pace:** depends on PyBullet's slow GUI text rendering on this Mac. Physics is identical regardless.
- **Training data:** scripted demonstrations only. Nothing here involves glove measurements.
