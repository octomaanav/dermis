# Real-time pick-up demo: the judge lifts a real cup, the simulated hand lifts a simulated cup

A judge wears the glove and picks up a real cup from a real table. The three-finger simulated hand (`src/dermis/glass/hand3.py`) picks up a simulated cup from a simulated table at the same time.

## Run it

```sh
# 0. Join the glove's Wi-Fi hotspot "Dermis-S3-XXXX" (password: AP_PASSWORD in firmware/v0/src/main.cpp),
#    then check what it is sending: expect ~10 frames/s and bad_json 0.
python -m dermis glass glove-check

# 1. Once per venue/glove (5 s): hand open, then a normal hold of the real cup.
python -m dermis glass pickup-calibrate          # saves runs/glass/pickup_calibration.json

# 2. The demo window (uses the calibration automatically).
python -m dermis glass pickup                   # keys: R reset, Q quit

# No hardware: SYNTHETIC grasp cycles (normal -> weak -> crush), or a recorded session.
python -m dermis glass pickup --source synthetic
python -m dermis glass pickup --source replay --file my_session.jsonl

# Record a live session (raw frames + every simulation tick) for later replay or analysis.
python -m dermis glass pickup --record runs/glass/pickup_session.jsonl
```

## What is live, and what is not

**The lift is triggered by the grasp, not tracked.**
- The glove has no position sensor, and double-integrating the IMU drifts by centimetres within a second, so the simulated hand cannot follow the real hand's height.
- When the thumb and the index or middle finger press for 60 ms, the simulated hand closes, then raises the cup 10 cm.
- When the judge lets go, it sets the cup down and opens.
- The judge should put the real cup back on the table before letting go. The simulated hand keeps its last firm grip while it lowers.

**The grip strength is live, per finger, every frame (the firmware streams at 10 Hz).**
- `thumb_tip` drives the simulated thumb, `index_tip` the index and `middle_tip` the middle finger.
- The simulated cup is held **only by friction** from those efforts:
  - a timid grip lets it slip and drop;
  - a crushing grip passes the **simulated** crack threshold (16 N per finger contact).
- After a drop or crack, the scene resets once the judge lets go.
- `thenar_apb` is shown on the bars but drives nothing: the simulated thenar pad is fixed to the palm.

**Grip source** (the window says which is in use):
- The current firmware (`dermis-4mlx-1imu-force-zero-span-v3`) sends `force_n` only once the glove has been tared and spanned on the device. Until then it sends `"force_n": null` (flags `FORCE_ZEROING` / `FORCE_SPAN_REQUIRED`).
- When every finger pod has a valid `force_n` (provisional, clamps at 9.81 N), that is used.
- Otherwise the grip is an **UNCALIBRATED proxy**: how far each pod's magnetic field moves from its open-hand reading, scaled by the venue calibration.
- The window shows CALIBRATING for the first second while it records the open-hand reading; keep the hand open then.

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
