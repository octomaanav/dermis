# Real-time pick-up demo: the judge lifts a real cup, the simulated hand lifts a simulated cup

A judge wears the glove and picks up a real cup from a real table. The three-finger simulated hand (`src/dermis/glass/hand3.py`) picks up a simulated cup from a simulated table at the same time.

## Run it

```sh
# 0. Join the glove's Wi-Fi hotspot "Dermis-S3-XXXX" (password: AP_PASSWORD in firmware/v0/src/main.cpp),
#    then check what it is sending: expect ~10 frames/s and bad_json 0.
python -m dermis glass glove-check

# 1. Glove calibration export (dermis-calibration v1: pod map, rest field, noise, load axis per pad)
#    lives at runs/glass/glove_calibration.json; pass another with --glove-cal.
# 2. Once per venue/glove (~10 s): hand open; a normal hold of the real cup; then move the hand and fingers
#    in the air TOUCHING NOTHING. Prints which pods registered and each pod's activation level.
python -m dermis glass pickup-calibrate          # saves runs/glass/pickup_calibration.json

# 3. The demo window (uses both calibrations automatically). Keep the hand open for the first second.
python -m dermis glass pickup                   # keys: R reset, Q quit

# No hardware: SYNTHETIC grasp cycles (normal -> too light to grip -> crush), or a recorded session.
python -m dermis glass pickup --source synthetic
python -m dermis glass pickup --source replay --file my_session.jsonl

# Record a live session (raw frames + every simulation tick) for later replay or analysis.
python -m dermis glass pickup --record runs/glass/pickup_session.jsonl
```

## What is live, and what is not

**When a pod counts as pressed** (`src/dermis/glass/glove_cal.py`, from the glove calibration export):
- **Signal:** only movement of the field **along that pad's load axis** (the export's `unit`) counts, relative to the open-hand rest. Drift across the axis (temperature, the glove shifting on the hand) is ignored. The size of the movement is used, not its sign, because the firmware's force went negative during a real grip, so the axis sign may be reversed.
- **Activation:** a pod is active above the highest of:
  - 15× its exported noise;
  - 80 µT;
  - **1.5× the largest signal that free hand movement produced in calibration step 3.**

  Moving the glove and flexing the fingers shift the pods' fields by far more than sensor noise, so the free-movement step is what stops waving from counting as a grip. A pod whose normal hold doesn't clear its own activation level can't tell a grip from movement, so calibration reports it and ignores it. A pod stays active until its signal falls below half its activation level. Inactive pods report 0.
- **Grasp:** the strongest pod at 12% of your calibrated normal hold plus a second pod at 5%, or **one** pod alone at 50%. It must last **0.3 s** (three glove frames), so a brief swing can't start a grasp.
- **Release:** the strongest pod falls below 30% of that grasp's peak (capped at 0.8× the grasp level), so a glove that doesn't spring fully back still lets go.
- **Drift at rest:** while the hand is open, each quiet pod's rest follows slow drift (5 s time constant). Creep slower than about 8 µT/s is absorbed; a real press is far faster.

**How hard the simulated hand grips** (`grip_effort` in `src/dermis/glass/pickup.py`). The cup's measured range in the simulator: 0.02 N·m or less drops, 0.025–0.08 holds, 0.09 or more can crack. Your squeeze maps onto that with a curve, not a straight multiple:

| Your squeeze vs. calibrated normal hold | Simulated grip |
|---|---|
| any recognized grasp (≥ 12%) | at least 0.04 N·m: holds |
| 1× | 0.05 |
| 3.5× | 0.075: still safe |
| about 3.8× and above | crosses the crack level |

Tested holding from 0.15× to 3× normal. Before this change, cracking took only about 1.7× normal, and a light hold was never recognized.
- This replaces the earlier signal: the field's movement in *any* direction from a rest taken in the first second. It counted drift as grip, which is why the hand showed GRIPPED at rest.

**Grip: the whole hand, not each finger** (default `--grip-mode combined`).
- A loose glove cannot press all four pods onto a cup at once. On the first real test, thumb and palm read high, index read low and middle read almost nothing.
- The grip is therefore the **mean of the two strongest pods** (any of thumb, index, middle, palm), and all three simulated fingers close with that strength.
- A grasp is detected when that grip passes the threshold, and released when it falls below the release threshold.
- `--grip-mode per_finger` restores one pod per finger.
- The simulated cup is held **only by friction**: a timid grip lets it slip and drop, and a crushing grip passes the **simulated** crack threshold (16 N per finger contact). After a drop or crack the scene resets once the judge lets go.

**Lift: the IMU says when the hand moves** (default `--lift-mode imu`).
1. After the grip closes, the hand is GRIPPED with the cup still on the table. The first 0.5 s record the IMU at rest, so keep still for a moment.
2. Any motion then starts the lift: |acceleration| moves 0.8 m/s² away from its rest value, or rotation changes by 0.5 rad/s, for 0.15 s. The cup rises 10 cm and stays up while the hand moves or stays still.
3. A motion that **starts with a dip** below the rest |acceleration| is read as a set-down. The cup goes back on the table and stays gripped until the next motion; the set-down's own motion has to end first.
4. Letting go sets the cup down if it is up, then opens the hand.

- The IMU is never integrated: only "moving or still" and the sign of the first jolt are used. The height is fixed, not tracked.
- With no valid IMU, the lift starts 1 s after the grip settles.
- Tune with `--imu-acc` and `--imu-gyro`; `glove-check` prints the IMU noise while the hand is still.

**Grip signal** (the window says which is in use):
- The current firmware (`dermis-4mlx-1imu-force-zero-span-v3`) sends `force_n` only once the glove has been tared and spanned on the device. Until then it sends `"force_n": null` (flags `FORCE_ZEROING` / `FORCE_SPAN_REQUIRED`).
- When every finger pod has a valid `force_n` (provisional, clamps at 9.81 N), that is used.
- Otherwise the grip is an **UNCALIBRATED proxy**: how far each pod's magnetic field moves from its open-hand reading, scaled by the venue calibration.
- The window shows CALIBRATING for the first second while it records the open-hand reading; keep the hand open then.

**Palm pod name:** firmware v3 calls the palm pod `palm_center`; it is read as `thenar_apb`.

**Which grip signal is used** is decided by the venue calibration, which records both during the normal hold:
- the on-device `force_n`;
- the raw field change from the open hand.

`force_n` is used only if it reads at least 0.3 N during the hold. On the first real test (2026-10-24) it read 0 N on every finger (`FORCE_CLAMPED_LOW` on index and middle) while the cup was held: the device's own span calibration did not follow the grip. In that case calibration chooses the raw-field proxy and says so, and the live run then ignores `force_n`. If neither signal moves, calibration stops and prints both.

**Calibration** sets the scale so that the calibrating person's normal hold lands in the middle of the simulated hand's safe grip range. Whoever calibrates defines "normal": a gentler judge drops the cup, a much harder one cracks it.

**Glove dropouts:** if no frame arrives for 0.5 s, the window shows GLOVE STALE and the simulation keeps its last command. It neither drops nor invents anything.

**Physics:**
- The hand is carried by one force-limited constraint, standing in for an arm.
- The cup is a free rigid body on the table; nothing attaches it (tested).
- On this Mac the simulation runs at about 3× real-time headroom (≈36% of one core in a real-time run).

## Verified

`tests/test_glass_pickup.py`, using SYNTHETIC grips:
- **Normal grip:** lifted about 9.7 cm, held, set down and released.
- **Weak grip:** dropped.
- **Crushing grip:** cracked while closing.
- **Constraints:** only the hand carrier is constrained.
- **Stale stream:** the last command is kept.
- **`force_n` and proxy extraction:** both work, including on the device's example frame.
- **Calibration:** maps a normal hold to the middle of the range.

A 27 s real-time headless run of the synthetic cycle produced, in order: lift → set down, drop → reset, crack → reset.

Against the repo's mock WebSocket server at 10 Hz: `glove-check` reported 10 frames/s with no decode errors. Calibration recorded 20 open / 30 hold frames even with a 3 s pause at each prompt.

**Not verified:** with the real glove. The only recording in the repo (`firmware/v0/dermis_ble.jsonl`) is 2.5 Hz BLE data with no grasp in it. The first live session should be recorded (`--record`) and checked.
