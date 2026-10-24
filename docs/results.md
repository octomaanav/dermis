# Results (simulation)

All numbers come from runs on 2026-10-23 on an Apple M2 (CPU only, 6 worker processes). The machine-readable files are in `docs/results/`. Every policy evaluation is AUTONOMOUS: the learned policy chooses each action from the current observation. Nothing is played back. The contact comparison uses **simulated** contact observations and says nothing about the value of real glove data.

## 1. Benchmark: 200 scripted demonstrations, held-out suite

```sh
python -m dermis benchmark --config configs/benchmark.yaml --workers 6
```

This collects 200 scripted episodes (seeds 0–199; all 200 succeeded), splits them by episode into 160 train and 40 val, trains both variants for 60 epochs (about 6 s each), and evaluates the expert and both policies on `configs/suites/heldout_v1.json` (50 scenarios, seeds 2,000,000–2,000,049, friction 0.4/0.6/0.8/1.0).

| Policy | Mode | Success | Drop eps | Excess-force eps | Peak force N (max / median) | Completion s (median) | Timeouts |
|---|---|---|---|---|---|---|---|
| scripted expert | SCRIPTED | 50/50 | 0 | 0 | 15.7 / 15.3 | 6.0 | 0 |
| bc_no_contact | AUTONOMOUS | 50/50 | 0 | 0 | 15.8 / 15.4 | 6.0 | 0 |
| bc_contact | AUTONOMOUS | 50/50 | 0 | 0 | 16.5 / 15.6 | 6.1 | 0 |

Excess-force threshold: 30 N per finger. This is a simulated benchmark threshold, not a clinical or biological measure.

With 200 demonstrations both learned variants reach the ceiling, so this benchmark **cannot** show any effect of contact observations. Earlier, on the dev suite (`dev_v1`, 30 scenarios), the result was the same: 30/30 for all three.

## 2. Data scaling × training seeds, same held-out suite

```sh
python -m dermis study --config configs/study.yaml --workers 6
```

Each cell trains on the first N episodes of the benchmark data. Both variants use the same episodes and split. Training seeds are 0, 1 and 2, and the architecture, epochs and evaluation scenarios are identical. Configs were frozen before any held-out evaluation was run. Nothing was tuned on the held-out suite.

| Demos (train+val) | Policy | Success per training seed | Mean | Min–max | Drop eps | Excess-force eps (summed over 3 seeds) |
|---|---|---|---|---|---|---|
| 10 | bc_no_contact | 2/50, 2/50, 6/50 | 0.07 | 0.04–0.12 | 0 | 9 |
| 10 | bc_contact | 9/50, 4/50, 3/50 | 0.11 | 0.06–0.18 | 0 | 6 |
| 25 | bc_no_contact | 16/50, 23/50, 10/50 | 0.33 | 0.20–0.46 | 0 | 12 |
| 25 | bc_contact | 29/50, 31/50, 32/50 | 0.61 | 0.58–0.64 | 0 | 7 |
| 50 | bc_no_contact | 4/50, 15/50, 2/50 | 0.14 | 0.04–0.30 | 0 | 20 |
| 50 | bc_contact | 42/50, 37/50, 45/50 | 0.83 | 0.74–0.90 | 0 | 16 |
| 200 | bc_no_contact | 50/50 ×3 | 1.00 | 1.00–1.00 | 0 | 0 |
| 200 | bc_contact | 50/50 ×3 | 1.00 | 1.00–1.00 | 0 | 0 |

![study](results/study.png)

**What this shows, and what it doesn't:**

- With 25 and 50 demonstrations, simulated contact observations raised held-out success substantially, and the ranges don't overlap across the three seeds. At 10 demos both variants mostly fail. At 200 demos both are perfect.
- Without contact, results are erratic. Success *drops* from 25 to 50 demos, with large spread between seeds. With three seeds per cell this is a small sample: treat the effect size as rough.
- Observed failure mode, diagnosed on the **dev** suite and not on held-out: the no-contact policies at n=50 lift the cube and carry it to within 0–2 cm of the target in xy, then hover with the cube's bottom 0.6–4.6 cm above the table, gripper closed, until timeout. The contact policies lower and release. All failures were timeouts. There were no drops and no out-of-bounds episodes.
- Why contact helps here is a hypothesis, not a verified mechanism. The force features give an easy-to-learn "holding the cube" signal, which may help a small-data policy separate "descend to grasp" from "lower to place". The opening width carries similar information, but the network appears to use it less reliably at small N.
- Excess-force episodes occur only in the low-data regimes, where poorly trained policies output high effort. All are under the simulated threshold definition, not a real measure.
- This is one narrow, fully observable task with rigid-body contact from PyBullet. It does not show that real glove tactile data would help a real robot.

## 3. Smoke test

```sh
python -m dermis smoke-test
```

The latest run took about 15 s. It collected 4 scripted demos (4/4 succeeded), trained a 32×32 MLP for 3 epochs, reloaded the checkpoint in a fresh evaluation path, ran 2 AUTONOMOUS episodes (0/2 succeeded, which is expected for 3 epochs), and wrote `runs/smoke/results.json`. The smoke test checks that the pipeline executes. It is not a performance claim.

## 4. Development history worth knowing

- **First expert, hidden timers.** The first scripted expert used internal tick counters for some phase changes. BC trained on 48 of its episodes reached 5/20 on the dev suite. The policies lifted the cube but either never released it or put it down off target. The expert was rewritten so that every phase change depends on observable quantities. At 200 demos, cloning then reached 30/30 on the dev suite. This was a data and task-design fix, made before any held-out evaluation.
- **Determinism.** PyBullet is deterministic here. Replaying an episode's issued actions reproduces its observations exactly (maximum deviation 0.0).
