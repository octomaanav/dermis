# dermis

Sensor glove (ESP32-S3 + BMI160 pods) → PyBullet teleoperation → behavioral cloning for a
foreign-body extraction task.

## Setup (macOS)

Use Python 3.12: PyBullet and PyTorch both ship macOS wheels for it.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyBullet has no macOS wheel, so pip compiles it. If that fails (`'cstring' file not found`, or a
zlib `fdopen` error), the Command Line Tools are not finding the SDK's C++ headers. Build it with:

```sh
SDK=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX2*.sdk | tail -1)
ARCHFLAGS="-arch arm64" CFLAGS="-Dfdopen=fdopen" CXXFLAGS="-isystem $SDK/usr/include/c++/v1" \
  pip install -r requirements.txt
```

PyTorch picks up Apple Silicon automatically: use `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`.

## Scripts

| Script | What it does |
| --- | --- |
| `mock_stream.py` | Replays `decoded_frame.json` in a loop at 30 Hz and prints the bend angles |
| `glove_stream.py` | Shared parsing + bend-angle trig; reads live glove frames (serial JSON or binary wire) |
| `sim_env.py` | PyBullet scene: Franka Panda, patient block with a hole, red tumor. WASD moves the gripper; the glove (default: `mock_stream.py`) sets the pinch |
| `record_demo.py` | Same scene, logs state + action at 30 Hz to `demonstrations.csv` |
| `generate_demos.py` | Scripted expert: 50 successful extraction episodes on random layouts → `scripted_demos.csv` |
| `train_model.py` | Trains a state → action MLP (MSE, MPS/CPU) and saves `surgeon_model.pth` |
| `autonomous_demo.py` | Runs the trained policy with a safety stop; on a stop, Claude writes an Emergency Diagnostic Report |

```sh
python mock_stream.py                       # 30 Hz replay of decoded_frame.json
python sim_env.py                           # mock_stream drives the pinch; WASD + Shift/Ctrl move
python sim_env.py --synthetic               # a fake hand that pinches and releases
python record_demo.py --keyboard-grip       # record demos without the glove (hold SPACE to close)
python generate_demos.py                    # 50 scripted demos -> scripted_demos.csv (~1.5 min)
python train_model.py --csv scripted_demos.csv   # the model that extracts the tumor (~1.5 min)
python train_model.py                       # trains on demonstrations.csv (makes 100 dummy rows if missing)
python train_model.py --dummy               # trains on fresh random rows in dummy_demonstrations.csv
export ANTHROPIC_API_KEY="sk-ant-your-key-here"   # placeholder
python autonomous_demo.py --fault-at 5      # or press F in the window to inject the fault
```

`mock_stream.py` looks for `./decoded_frame.json` first, then `firmware/v1/examples/decoded_frame.json`.
That file holds one frame, so the pinch is constant (index 0.01, thumb 0.55 → grip 0.28). Give it a list
of frames, or use `--synthetic`, to see the gripper open and close.

### Autonomous demo and safety stop

Each tick the policy predicts `target_grip_angle` and `target_move_x/y/z`; the gripper is commanded to
its current position plus the move (clamped to 1 cm per tick). The grip is on/off: above 0.5 closes fully.
Trained on `scripted_demos.csv`, it extracted the tumor in 20 of 20 unseen random layouts (5–7 s each);
then press F to show the safety stop. Before the command runs, the monitor
checks both the measured and the commanded gripper z against `SAFETY_Z_MIN = 0.0` (the table plane).
On a violation the arm is frozen, the telemetry is sent to Claude (`claude-haiku-5-5`), and the report
is printed. Without an API key, or if the call fails, a local report is printed instead, so the demo
never stops on a network problem. The fault (`F` or `--fault-at`) is simulated: a growing downward
bias on the policy's z output.

### Glove sources (`sim_env.py` and `record_demo.py` take these flags; `glove_stream.py` too)

```sh
--mock firmware/v1/examples/decoded_frame.json   # replay a JSON file (one frame, a list, or one per line)
--port /dev/cu.usbmodem1101                      # hub prints one JSON frame per line
--port /dev/cu.usbmodem1101 --wire               # hub speaks the binary Dermis Wire v1 (firmware/v1/SPEC.md)
```

To test the `--wire` path without hardware, run the fake hub in a second terminal and use the port it prints:

```sh
python firmware/v1/host/python/dermis_wire.py serve-pty     # prints e.g. /dev/ttys012
python glove_stream.py --port /dev/ttys012 --wire
```

> Known issue (macOS 27): the fake hub ACKs commands but delivers no DEVICE_INFO or frames, even to
> `dermis_wire.py dump --port`. The decoder itself is fine; it reads every frame of `firmware/v1/vectors/capture_2s.bin`.

Bend angle is the angle between the pod's current gravity vector and its rest vector, divided by
`--full-bend-deg` (default 90°). The rest vector defaults to "pod flat, z up"; press `C` in the sim
(or pass `--calibrate` to `glove_stream.py`) with your hand open to make the current pose zero.

### Sim controls

Click the PyBullet window first.

| Key | Action |
| --- | --- |
| W / S (or Up / Down) | gripper ±x |
| A / D (or Left / Right) | gripper ±y |
| Shift / Ctrl (or E / Q) | gripper up / down |
| Space | hold to close the gripper (works in every mode, overrides the glove) |
| C | calibrate glove (current pose = open) |
| R | reset scene; in `record_demo.py` also starts a new episode |
| F | inject the fault (`autonomous_demo.py`) |

### demonstrations.csv

One row per 30 Hz tick, appended across runs:

`episode, step, t` · state: `ee_x/y/z` (gripper), `tumor_x/y/z`, `finger_l/r` (finger joint opening, m) ·
action: `target_x/y/z` (commanded gripper position), `grip_cmd` (0 open .. 1 closed).

`train_model.py` converts this to its training columns: `gripper_* = ee_*`, `finger_opening = mean(finger_l, finger_r)`,
`target_move_* = target_* - ee_*`, `target_grip_angle = grip_cmd`. `scripted_demos.csv` uses the same columns.
Every reset (R, or a new episode) moves the patient block (±5 cm x, ±8 cm y) and the tumor within the hole. The policy only learns what the demos show: record several episodes
that descend into the hole, close on the tumor, and lift.
