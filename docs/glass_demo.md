# Glass hold under a water pour: three-finger hand + thenar pod (SIMULATION, SYNTHETIC glove data)

Three lanes run on **one identical scenario** (same glass and same pour):

| Lane | Controller | What happens |
|---|---|---|
| 1. Too little force | SCRIPTED `too_weak`, fixed 0.02 N·m | Holds the empty glass. As water is added, the glass slides out and falls. |
| 2. Too much force | SCRIPTED `over_squeeze`, 0.035 + 1.0 N·m/kg × water poured | Squeezes harder for every gram poured until a pod passes the **simulated crack threshold**. |
| 3. Neural network | AUTONOMOUS `runs/glass/glass_policy_v1` | Reads **only** the SYNTHETIC glove frames (4 pods) and its own previous command. Holds the glass still through the pour. |

```sh
python -m dermis glass demo                     # simulate 3 lanes (seed 30000) -> runs/glass/demo/: mp4 + plot + traces + frames
python -m dermis glass demo --gui --no-video    # same, replayed side by side in a PyBullet window
```

## What is simulated, and what is not

- **Hand** (`src/dermis/glass/hand3.py`, new; the peer teleop hand is untouched):
  - Thumb, index and middle each have 3 phalanges, driven by force-limited position motors.
  - The thumb opposes the two fingers. The fingers wrap behind the glass and press it into a **thenar pad** on the palm, where the glove's `thenar_apb` pod sits.
  - The teleop `HandSim` thumb moves along the cup axis and cannot oppose the fingers, so it could not do this grasp.
  - A first version pinched the glass at its equator, and the glass walked out whenever the grip effort changed. The enveloping grasp fixed that.
- **Water** is **mass**: the glass body's mass increases during the pour, so gravity really asks more of the grip. There is no fluid simulation. The blue fill is drawn from the mass (1 kg/L).
- **Crack** is a **simulated threshold**: the episode ends when any pod's true contact normal force exceeds 16 N. The glass is rigid. It is tinted red, not shown breaking.
- **Drop** means the glass fell more than 3 cm below its height when the setup fixture was removed.
- **Setup:**
  - The glass is held by a temporary fixed constraint for 1.5 s while the fingers close at 0.04 N·m.
  - The constraint is then removed; nothing attaches, teleports or resets it afterwards.
  - Tests assert zero constraints. Physics is identical for every controller.
- **Force balance** (test): with the full glass held still, the vertical friction on the glass equals its weight within 0.05 N (p95).
- **All four pods carry load** in the hold (test): thumb, index, middle and thenar_apb.

## Glove data schema

`src/dermis/glass/schema.py` validates and parses the frame layout from the device (`config_id`, `dermis-4mlx-1imu-force-provisional-v1`):
- 4 pods: `index_tip`, `thumb_tip`, `middle_tip`, `thenar_apb`.
- Each pod has one `mag` block with `b_ut`, `force_n`, `force_valid` and the `FORCE_PROVISIONAL` / `FORCE_CLAMPED_LOW` / `FORCE_CLAMPED_HIGH` flags.
- There is one top-level `imu`.

The device frame example passes (`tests/test_glass.py`). The older decoded-frame v1 adapter (`dermis.telemetry.adapter`) is unchanged.

```sh
python -m dermis glass check-schema runs/glass/demos/ep_0000.frames.jsonl     # 281/281 frames valid
```

**SYNTHETIC frames** (`src/dermis/glass/sensors.py`):
- One frame per control tick (30 Hz) is generated from simulator contact.
- **Labels:** top-level flag `SYNTHETIC`, device id `SYNTHETIC-GLASS-SIM`, and `FORCE_PROVISIONAL` on every pod.
- **force_n:** simulated normal force plus noise, clamped to [0, 9.80665] N with the matching flags, like the firmware.
- **b_ut:** an **invented** linear model (baseline from the real frame + gains × [vertical shear, horizontal shear, normal] + noise). It is not a calibration of real pods.
- **Pod coverage:** thumb, index and middle each read their whole digit; `thenar_apb` reads the palm pad.
- **IMU:** the hand is fixed, so it reads gravity plus noise only and is not used by the network.

## Training data and network

**Demonstrations:** `python -m dermis glass collect` (200 episodes, `runs/glass/demos/`):
- Made by the SCRIPTED `load_expert`. It sets grip from the **true** weight and friction (privileged simulator state), with a floor, a cap, a rate limit and a slip reflex.
- The executed grip had 0.008 N·m Gaussian noise added (DART-style); the clean expert command is the label.
- 200/200 succeeded.
- Each episode stores:
  - `.frames.jsonl`, the SYNTHETIC glove frames in the schema above;
  - `.trace.jsonl`, the simulator state;
  - `.npz`, the features and labels.

**Network:** `python -m dermis glass train` (`runs/glass/glass_policy_v1`):
- **Inputs:** per pod, `b_ut` minus the unloaded baseline frame (3), `force_n` and FORCE_CLAMPED_HIGH, plus the previous grip command. History of 4 frames (84 inputs). **No simulator ground truth.**
- **Architecture:** MLP 128×128.
- **Split:** 160 training / 40 validation **episodes**, with the normalizer fitted on training episodes only.
- Best validation MAE 0.0012 N·m; 4.8 s on CPU.
- **Metadata:** feature names, normalizer, task config and provenance (data digest, episode IDs, git commit).

## Results: held-out suite (60 unseen scenarios, seeds 30000–30059)

Glass mass 120–180 g, radius 31–35 mm, friction 0.5–0.6, water 220–350 g poured over 4–7 s.

| Controller | Outcome | Peak pod normal force, median / max | Mean grip |
|---|---|---|---|
| Neural network (AUTONOMOUS) | **60/60 held** | 7.2 / 8.5 N | 0.047 N·m |
| `load_expert` (SCRIPTED demonstrator) | 60/60 held | 7.3 / 8.6 N | 0.048 N·m |
| `too_weak` | 60/60 **dropped** | 5.3 / 6.0 N | 0.020 N·m |
| `over_squeeze` | 60/60 **cracked** (simulated threshold) | 16.4 / 19.9 N | 0.089 N·m |
| fixed 0.06 N·m (baseline) | 60/60 held | 11.7 / 12.4 N | 0.060 N·m |

**What these results show:**
- The network matches its demonstrator. It holds every unseen scenario, light or heavy, by raising its grip as the pods report the extra weight.
- **Honest baseline:** a well-chosen fixed grip (0.06 N·m) also holds all 60. It does so by squeezing about 60% harder than the network for the whole pour, and above the 9.81 N point where the glove's `force_n` clamps.
- The network's advantage here is holding with **less force**, not holding where nothing else can. In this simulation the safe fixed-grip window is wide: 0.04–0.08 N·m held all 6 sweep scenarios, ≤0.03 dropped and ≥0.1 cracked.

Demo run (seed 30000, `docs/results/glass_demo/`):

| Lane | Outcome |
|---|---|
| too_weak | Dropped at 3.47 s, with 131 g of water in the glass |
| over_squeeze | Cracked at 4.30 s, at the thenar_apb pod (17.0 N), with 175 g of water |
| network | Held through the whole 271 g pour (peak pod force 7.2 N, at thenar_apb) |

## Limitations

- Everything is simulated. The pod readings are synthetic, so nothing here shows how real pods respond.
- The network was trained to imitate a scripted demonstrator that used privileged state. It has not seen real glove data, human demonstrations or real glasses.
- The hand floats (no arm), and the glass is a rigid cylinder.
