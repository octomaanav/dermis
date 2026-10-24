# Wiping feasibility study (wipe_v1): decision **NO-GO** for a tactile-advantage claim

**SYNTHETIC MANIPULATION BENCHMARK.** It does not model real cleaning, tissue or material physics. The load band, overload threshold and dwell rule are benchmark definitions. The glove question is separate and remains **PENDING HARDWARE**.

This study is separate from the pick-and-place and fragile_cup experiments. It lives in `src/dermis/wipe/`, `configs/wipe/` and `runs/wipe/`, with copies of the results in `docs/results/wipe/`. Its outputs are under `python -m dermis wipe …`, and its metrics are never merged with the other experiments.

## 1. What was implemented

**Environment** (`wipe/env.py`): headless PyBullet.
- **Surface:** a static heightfield over a 30 × 22 cm patch, 5 mm grid.
- **Pad:** a 0.1 kg sphere, radius 2 cm, with *soft contact* (PyBullet `contactStiffness` / `contactDamping`). It is held at zero rotation, so it slides rather than rolls. Friction against the surface is 0.3.
- **Wrist:** a spring-damper pulls the pad toward the commanded position, stiff in x/y (4000 N/m). The vertical stiffness `k_z` (100–4000 N/m) is a controller parameter, i.e. impedance.
- **Commands:** controllers command velocity, bounded at |vx|, |vy| ≤ 0.12 m/s and |vz| ≤ 0.05 m/s, at 30 Hz. Physics runs at 240 Hz with 8 substeps per tick.

**Shared x/y path:** a time-based raster, identical for every controller. It holds 2.0 s at the start cell (approach), runs 8 lanes along x at 0.08 m/s (wiping, 25.0 s), then holds 0.6 s (retract).

**Contamination grid:** 12 × 8 cells of 2 cm, 96 cells. A cell is **wiped** only when all of the following hold:
- the pad–surface contact point lies inside it,
- the normal load is within the in-range band (2–6 N),
- the episode is in the wiping phase,
- and this has accumulated for at least 0.15 s.

Each cell counts once.

**Normal load:** the *net* pad–surface contact force (normal plus friction parts, summed over all contact points from one query per substep), projected onto the analytic surface normal at the contact location. See §2 for why it isn't the sum of per-point normal magnitudes.

**Exploit guards** (tested in `tests/test_wipe.py`):
- **Hovering:** no contact, no coverage (0.00).
- **Pressing through the surface:** load above the band does not count and is tracked as overload. Coverage 0.00, overload 26.5 s, peak 67.7 N.
- **Moving too fast:** at 0.6 m/s a cell is crossed in 33 ms, far under the 150 ms dwell, so coverage stays below 0.2.
- **Repeated passes:** cells are binary, so 8 s of contact in one cell counts once.
- **Workspace escape:** terminates the episode.
- **Penetration:** more than 8 mm beyond the resting depth is flagged, and the episode terminates beyond 16 mm.

**Scenario variation** (`WipeScenario`), physically motivated:

| Variable | Range | Reason |
|---|---|---|
| Surface tilt x / y | ±3° | Imperfect fixturing or furniture. Shifts height by up to ±6 mm across the patch. |
| Bump amplitude / wavelength / direction / phase | 0–5 mm / 6–15 cm / 0–180° / 0–2π | Smooth, gentle undulation (worktops, upholstered or soft surfaces) |
| Pad contact stiffness | 3·10³–3·10⁴ N/m (log-uniform) | Pad material, from soft foam to firm rubber |
| Geometry calibration offset | ±2 mm, constant | Typical depth-sensor or calibration error. Seen by every geometry-using controller, equally. |
| Initial pad clearance | 3–6 cm | Approach start |
| **Flat sanity case** | 0 for all of the above, nominal pad | First scenario of every suite |

**Suites:**
- **Development suite:** flat case plus seeds 1,000,001–1,000,015 (16 scenarios). Used for all design and tuning.
- **Evaluation suite:** `configs/wipe/eval_v1.json`, flat case plus seeds 2,000,001–2,000,040 (41 scenarios). It was **frozen with a SHA-256 before tuning started**, and loading checks the hash.
- The task was not changed after freezing, so it is still version v1.

**Glove feasibility tools:** `wipe/glove_trials.py`, described in §6.

## 2. Physics validation (`python -m dermis wipe physics-check`: all 8 pass)

| Check | Result |
|---|---|
| Heightfield vs analytic profile (117 ray tests) | Max error 3·10⁻¹⁰ m. PyBullet centres heightfields between min and max height; this is accounted for. |
| Force sign and frame; static equilibrium | The net contact force points from surface to pad (0.7° from +z on flat). Normal load 3.649 N = wrist spring force k(z − z_cmd) = 3.649 N. |
| **Per-point normals** | **75% of individual contact points have normals tilted more than 8°** (min n_z 0.969 stationary, 0.947 while wiping). This is a triangle-mesh contact artifact of the sphere on the heightfield. Summing per-point normal magnitudes would make "load" depend on triangulation, so load is defined from the *net* force (§1). |
| No double counting | One query per substep. The pad→surface net force equals −(surface→pad net) to within 0.003 N of 4.2 N (the reversed query differs by about 0.05%). The reported load equals the net force projected on the surface normal exactly. |
| More compression → more load | At k_z = 300: 0, 0.58, 1.16, 2.33, 3.49 N. At k_z = 3000: 0, 4.80, 9.79, 21.2, 27.7 N (depths 0, 2, 4, 8, 12 mm). |
| Stationary pad stability | Over 6 combinations of pad stiffness (3·10³–3·10⁴) and wrist stiffness (300, 3000): relative standard deviation ≤ 0.4%, drift 0.000 N over the last second. |
| Substeps 240 → 480 Hz | Same 3 controllers on 4 scenarios: coverage change 0.00, p95 change ≤ 0.30 N, 0 success flips. |
| Wiped-cell events = real contact | 96/96 events lie inside their cell with load in band. 0 violations. |

No unstable or solver-dependent behaviour was found within these ranges. The per-point normal tilt is the one solver artifact; the load definition sidesteps it.

## 3. Controllers and observation access

Every controller uses the same path, velocity limits, 30 Hz rate, start conditions, physics and scenarios. Access is *enforced*: `ObsFilter` passes only the keys a controller declares.

| Controller | What it chooses | Shared path | Own commanded pose | Measured pad pose | Observed geometry (with calibration offset) | Normal load (tactile) |
|---|---|---|---|---|---|---|
| A constant height | Fixed height, from a nominal flat surface | ✓ | ✓ | – | – | – |
| B geometry-following, stiff wrist (k_z ≥ 1500) | Follows h_obs | ✓ | ✓ | ✓ | ✓ | – |
| C geometry-following, compliant wrist (k_z ≤ 1000; impedance) | Same code as B | ✓ | ✓ | ✓ | ✓ | – |
| Cp proprioceptive load (no tactile sensor) | Geometry feed-forward + PI on F_est = k_z·(z_meas − z_cmd) | ✓ | ✓ | ✓ | ✓ | – |
| D contact feedback | Geometry feed-forward + PI on measured normal load | ✓ | ✓ | ✓ | ✓ | ✓ |

- **Never available to any controller:** true geometry, the calibration offset itself, pad stiffness, scenario parameters, cell/dwell/coverage state, success, or future disturbances.
- **Shared objective:** all controllers target the same load (4 N, the band centre). A, B and C convert it to a press depth using the *nominal* pad stiffness (10⁴ N/m). Their tuning grids also vary the target (3.5 / 4.5 N), so they can absorb the band's position.
- **Commanded vs measured pose:** these are separate keys.
- **Proprioception reveals contact:** in this simulation the wrist deflection (measured − commanded height) *does* reveal contact load (F = k_z·deflection). Controller Cp uses exactly that and needs no tactile sensor.

**Tuning:** an equal budget of **8 configurations per controller**, on the development suite only. The selection rule was fixed beforehand: most successes, then mean coverage, then least overload.

| Controller | Chosen | Dev successes /16 | All 8 candidates (successes) |
|---|---|---|---|
| A | k_z 300, 3.5 N | 10 | 10, 7, 1, 1, 1, 1, 1, 1 |
| B | k_z 1500, 3.5 N | 10 | 10, 9, 9, 7, 4, 4, 4, 2 |
| C | k_z 150, 3.5 N | 15 | 15, 15, 15, 15, 15, 14, 14, 14 |
| Cp | k_z 1500, kp 3·10⁻⁴, ki 5·10⁻³ | 15 | 14–15 for all 8 |
| D | k_z 600, kp 3·10⁻⁴, ki 5·10⁻³ | 15 | 15 for all 8 |

C does well across its whole grid; it is not a lucky pick.

## 4. Results on the frozen evaluation suite (41 scenarios, identical for all controllers)

**Metrics:**
- **Coverage** = wiped cells / 96. All 96 cells lie on the shared path, and the flat case reaches 96/96.
- **Attempted coverage** = cells with any wiping-phase contact / 96.
- **Contact loss:** wiping-phase time with load < 0.2 N.
- **Overload:** time with load > 10 N, in any phase.
- **p95:** over wiping-phase substeps in contact.
- **Success:** coverage ≥ 0.90, overload ≤ 0.25 s, no penetration flag, run completed.
- **Timing:** approach (2.0 s), wiping (25.0 s) and retract (0.6 s) are timed separately and are identical for all controllers, because the path is time-based.

| Condition | A | B | C | Cp | D |
|---|---|---|---|---|---|
| **Nominal**: success | 23/41 | 27/41 | **40/41** | 40/41 | 40/41 |
| Coverage mean (min) | 0.911 (0.73) | 0.734 (0.00) | 0.996 (0.89) | 0.996 (0.89) | 0.996 (0.89) |
| Overload s (mean) / peak N (max) | 0.000 / 9.6 | 0.001 / 10.6 | 0.000 / 7.8 | 0.008 / 11.6 | 0.000 / 7.5 |
| p95 N (median) | 5.07 | 4.41 | 4.15 | 4.52 | 4.47 |
| **Obs-noise stress**: success | 23/41 | 14/41 | 40/41 | 40/41 | 40/41 |
| **2-tick-delay stress**: success | 9/41 | 0/41 | **28/41** | 0/41 | 0/41 |
| Coverage mean | 0.795 | 0.085 | 0.925 | 0.027 | 0.675 |

**Stress conditions:**
- **Obs-noise:** measured pose σ 0.5 mm, geometry σ 1 mm per query, load σ 0.3 N.
- **Delay:** every action is applied 2 ticks (67 ms) late.
- Controllers were *not* retuned for either condition.
- A's obs-noise row equals nominal because A observes no noisy channel.

**Paired comparisons on identical scenarios (nominal):** coverage difference D − other. The bootstrap intervals are descriptive only; this is not a significance test.

| Pair | Mean | Min / max | Bootstrap 95% interval | D better / other better / equal | Success only D / only other |
|---|---|---|---|---|---|
| D − A | +0.086 | +0.00 / +0.23 | [+0.061, +0.110] | 28 / 0 / 13 | 17 / 0 |
| D − B | +0.262 | −0.01 / +1.00 | [+0.149, +0.383] | 17 / 1 / 23 | 13 / 0 |
| D − C | **−0.000** | −0.01 / +0.00 | [−0.001, +0.000] | 0 / 1 / 40 | 0 / 0 |
| D − Cp | −0.000 | −0.01 / +0.00 | [−0.001, +0.000] | 0 / 1 / 40 | 0 / 0 |

Under 2-tick delay, D − C = −0.249 (C better in 41/41) and D − Cp = +0.648.

**Failures:**
- **Nominal, C / Cp / D:** all three fail the same scenario (seed 2,000,010: 4 mm bumps over 7.4 cm plus a 2.9° tilt). Each wipes 85/96 cells (0.885), with every cell touched, no overload and p95 5.3 N. That is a shared geometry limit near the dwell rule, not a controller difference.
- **B fails** when the calibration offset makes it hover or press too hard, because the stiff wrist turns a ±2 mm error into ±3–6 N.
- **A fails** on tilted or bumpy surfaces.
- **Under delay,** both feedback loops (Cp, D) oscillate. Their gains were tuned without delay, so this is a robustness finding, not proof that tactile control can't work with delay.

![coverage](results/wipe/coverage.png)

## 5. Does a reasonable no-contact controller already solve it?

**Yes.**
- The compliant (impedance) geometry follower C matches the tactile controller in nominal conditions and under observation noise: identical on 40/41 paired scenarios, mean difference −0.000.
- It is clearly *more* robust to control delay (28/41 vs 0/41).
- Even without a tactile sensor, wrist deflection is a load estimate (Cp), so "contact information" is available proprioceptively here.

Tactile feedback only beat the *stiff* or *open-loop* baselines (A, B), and a properly compliant wrist removes that gap.

## 6. Glove feasibility: PENDING HARDWARE

There's no physical measurement yet. Synthetic data was used only to software-test the tools, and every report from synthetic input is stamped `PENDING HARDWARE`.

**Physical setup:**
- The instrumented fingertip pod presses a thin pad (e.g. 1–2 mm silicone or felt) against a stationary rigid surface. The pad must transmit load into the sensing region over the magnet.
- **With a scale:** place the surface on it and log `host_unix_s,grams` (for example by hand-timed readings or a USB scale), passed with `--scale-csv`. The scale measures **one (vertical) reference component** only; it does not validate three-axis force.
- **Without a scale:** film the trials with a visible clock and keep the qualitative notes the recorder asks for.

**Recording** uses the existing ingestion pipeline:
```sh
python -m dermis wipe glove-record --source serial --serial-port /dev/cu.usbmodemXXXX [--wire] \
    --out data/glove_trials/session1 --pod index_tip --mount-note "1.5 mm silicone pad, pod mount +X+Y+Z" \
    --repeats 3 --trial-s 5 [--scale-csv scale.csv]
python -m dermis wipe glove-report --session data/glove_trials/session1
```

**What is saved:**
- Every decoded frame as received, plus host receive time: raw magnetic XYZ, IMU, validity/STALE flags, device time.
- The unloaded baseline calibration, which is rejected with a warning if the pad was loaded or moving.
- Trial markers with host monotonic and Unix times.
- Mount and pad notes, and device and config IDs.

**Trial labels**, each repeated: unloaded, light press, strong press, press and release, constant-contact wipe, lift-off, sideways load without slip, and observed slip (optional, with a note).

**Report output:**
- `repeats.png`: repeats overlaid per label.
- `overview.png`: raw B, baseline-subtracted ΔB, acceleration, and freshness/STALE/invalid flags with trial spans.
- `numbers.json` with:
  - unloaded noise floor
  - plateau |ΔB| per repeat
  - between-repeat CV
  - strong-vs-light separation
  - release residuals
  - sideways-vs-press ΔB direction angle (a *tangential-response* indicator, **not** slip detection)
  - fraction of fresh samples per label
  - scale association (Pearson r only, no newton conversion)

**Continue / stop checklist** (proposed thresholds, to confirm on the first real session):
- [ ] Unloaded baseline accepted. Noise floor σ well below the light-press change: aim for light-press plateau ≥ 5× σ.
- [ ] Repeat consistency: between-repeat CV of the light and strong press plateaus ≤ 15%.
- [ ] Strong vs light separation ≥ 3 (difference of means / pooled SD).
- [ ] Release recovery: residual after release or lift-off ≤ 10% of the press peak, within 1 s.
- [ ] During constant-contact wiping, the signal stays clearly above unloaded for the whole trial, with no drop-outs (fresh fraction ≥ 0.95).
- [ ] Orientation and crosstalk: compare rotating the unloaded hand against the light press. If rotation alone produces changes comparable to a light press, Earth-field or orientation compensation is needed first.
- [ ] If a scale is used: monotonic association with the reference (r ≥ 0.9) on the vertical component. This is *not* a force calibration.
- [ ] No claims of newtons, three-axis force, slip detection or absolute hand position from these data.

## 7. Optional learning step: not run

Its precondition was "contact feedback shows a useful development advantage". It didn't (C = D on dev, 15/16 each), so no BC policy was trained. The existing `bc/` and `cup/train.py` infrastructure could be reused if a future task version shows an advantage.

## 8. Decision: **NO-GO** for building a wiping product flow around a tactile advantage

**Reasons:**
- The physics is stable, and the task works as a benchmark.
- A simple compliant no-contact controller already solves it in nominal and noisy conditions, and is more robust to delay.
- Any tactile advantage over the stiff or open-loop baselines depends on giving the baseline a stiff wrist, which would be an artificial handicap.

**What this does *not* say:**
- It says nothing about whether the real glove senses wiping contact; that question is pending hardware.
- Tactile sensing could still matter where compliance alone is not enough. Plausible cases, *untested here*:
  - a required load that changes during the task;
  - surfaces with unknown, large compliance changes beyond 3·10³–3·10⁴ N/m;
  - detecting slip or contamination from friction changes;
  - sharp steps that a soft wrist cannot follow.
- Any such follow-up must be a new task version, with a new frozen suite, not a retune of this one.

## 9. Reproduction

```sh
python -m dermis wipe physics-check                       # 8 checks -> runs/wipe/physics_checks.json
python -m dermis wipe freeze-suite                        # refuses to overwrite configs/wipe/eval_v1.json
python -m dermis wipe tune --workers 6                    # 5 x 8 configs x 16 dev scenarios -> runs/wipe/tuning.json (37 min here)
python -m dermis wipe evaluate --workers 6                # frozen suite, 3 conditions -> runs/wipe/eval/{report.md,evaluation.json,*.png}
python -m pytest -q tests/test_wipe.py                    # exploit guards, access, suite integrity, glove tooling
```

Wall times are from an 8-core M2 under heavy unrelated system load.

## 10. Smallest next step

Record **one real glove session** with the procedure in §6: light, strong and press-release trials, 3 repeats, with a kitchen scale if available. Then run `glove-report` against the checklist. That answers the separate sensing question at low cost.

Don't build more wiping simulation until a task variant that gives tactile sensing a non-artificial role is specified and frozen first.
