"""Live fragile_cup modes for demos and judging. Two separate modes, chosen at launch:

TEACH        a human controls the grip (keyboard now; EXPERIMENTAL glove later). Simulator
             observations and the effort actually issued are recorded (when recording is on)
             with source KEYBOARD / GLOVE into data/episodes/cup_teach.
AUTONOMOUS   a previously trained checkpoint controls the grip from observations only. No human
             motor-input object exists in this mode; human key events can only trigger judge
             controls (Pour, Reset, Next, Compare, Switch policy), never effort.

Judge controls (both modes): Pour starts the load ramp of the current scenario; Reset restarts
the *same* scenario (identical initial conditions and load schedule); Next moves to a new seed;
Paired comparison (AUTONOMOUS) runs every loaded checkpoint on the current scenario with the
pour at the same moment, one after another, and shows both outcomes.

LiveSession has no PyBullet GUI code so it can be tested headless; run_gui() adds a display-only
CupViewer window. Physics always runs headless, identical to evaluation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pybullet as p

from ..data.episodes import EpisodeStore
from .controllers import KeyboardGrip, PolicyController
from .env import CupConfig, CupHoldEnv, CupScenario, clip_effort
from .rollout import CupRecorder, run_episode
from .status import HoldStatus
from .worker import schedule_id

TEACH, AUTONOMOUS = "TEACH", "AUTONOMOUS"
JUDGE_KEYS = {"p": "pour", "r": "reset", "n": "next", "c": "compare", "t": "switch", " ": "record"}


class LiveSession:
    def __init__(self, env: CupHoldEnv, mode: str, seed: int, teach_input=None, policies: dict | None = None,
                 teach_dir: str = "data/episodes/cup_teach", viewer=None):
        if mode not in (TEACH, AUTONOMOUS):
            raise ValueError(mode)
        if mode == TEACH and teach_input is None:
            raise ValueError("TEACH needs a human input (keyboard or glove)")
        if mode == AUTONOMOUS and not policies:
            raise ValueError("AUTONOMOUS needs at least one checkpoint")
        self.env, self.mode, self.seed, self.viewer = env, mode, seed, viewer
        self.human = teach_input if mode == TEACH else None          # does not exist in AUTONOMOUS
        self.policies = dict(policies or {}) if mode == AUTONOMOUS else {}
        self.policy_names = list(self.policies)
        self.active = self.policy_names[0] if self.policy_names else None
        self.store = EpisodeStore(teach_dir)
        self.recording = False
        self.saved: list[str] = []
        self.ignored_human_events = 0
        self.last_compare: list[dict] = []
        self.hold = HoldStatus()
        self.hold_state: dict = {}
        self.reset()

    # -- motor command source -----------------------------------------------------------

    @property
    def controller(self):
        return self.human if self.mode == TEACH else self.policies[self.active]

    def human_input(self, down: set, pressed: set) -> list[str]:
        """Route key events. Effort keys reach the robot only in TEACH. Returns judge actions taken."""
        actions = [JUDGE_KEYS[k] for k in pressed if k in JUDGE_KEYS]
        if self.mode == TEACH and hasattr(self.human, "handle_keys"):
            self.human.handle_keys(down, pressed)
        elif down - set(JUDGE_KEYS) or pressed - set(JUDGE_KEYS):
            self.ignored_human_events += 1                         # AUTONOMOUS: motor keys are dropped here
        for a in actions:
            self.do(a)
        return actions

    def do(self, action: str):
        if action == "pour":
            return self.env.pour()
        if action == "reset":
            return self.reset()
        if action == "next":
            self.seed += 1
            return self.reset()
        if action == "record" and self.mode == TEACH:
            self.recording = not self.recording
            if self.recording and self.env.stats.ticks > 0:
                self.reset()                                       # start recording from a clean episode
            return self.recording
        if action == "switch" and self.mode == AUTONOMOUS:
            i = (self.policy_names.index(self.active) + 1) % len(self.policy_names)
            self.active = self.policy_names[i]
            return self.reset()
        if action == "compare" and self.mode == AUTONOMOUS:
            return self.paired_compare()
        return None

    # -- episode ------------------------------------------------------------------

    def reset(self):
        self.scenario = CupScenario.sample(self.seed, self.env.cfg)
        self.obs = self.env.reset(self.scenario, pour_at_s=None)
        self.controller.reset()
        self.hold.reset()
        self.hold_state = self.hold.update(self.env)
        src = getattr(self.human, "label", "KEYBOARD").lower() if self.mode == TEACH else None
        self.rec = (CupRecorder(src, self.env.cfg, self.scenario, self.human.name,
                                extra={"mode": TEACH,
                                       "calibration_id": getattr(getattr(self.human, "cal", None), "calibration_id", None),
                                       "raw_recording": getattr(self.human, "raw_recording", None),
                                       "clock_note": "host_ns per tick links to raw_recording _t_recv_ns; device clock not synced"})
                    if self.mode == TEACH else None)
        return self.obs

    def tick(self) -> dict | None:
        if self.env.done:
            return None
        effort, label, tag = self.controller.act(self.obs, self.env)
        effort = clip_effort(effort)
        if self.rec is not None:
            self.rec.record(self.obs, effort, self.env.t, label, tag)
        self.obs, done, info = self.env.step(effort)
        self.hold_state = self.hold.update(self.env)              # every control tick (hysteresis)
        if self.viewer is not None:
            self.viewer.external_status = self.hold_state
        if self.rec is not None:
            self.rec.log(info)
            if done and self.recording:
                ep = self.rec.finish(self.obs, {**self.env.stats.to_dict(), "events": None, "pour_t_s": self.env.pour_t0})
                self.store.save(ep)
                self.saved.append(ep.id)
        return info

    def paired_compare(self, realtime: bool = False, pour_at_s: float | None = None) -> list[dict]:
        pour = pour_at_s if pour_at_s is not None else self.env.cfg.pre_pour_s
        out = []
        v = self.viewer
        tick = (lambda e, *_: (v.sync(e), v.status(e))) if v else None
        for name in self.policy_names:
            if v:
                v.text("mode", f"PAIRED COMPARISON  AUTONOMOUS: {name}  seed {self.scenario.seed}", 0, (0.75, 0.1, 0.1))
            outcome, _ = run_episode(self.env, self.scenario, self.policies[name], pour_at_s=pour, realtime=realtime,
                                     on_tick=tick)
            out.append({"policy": name, "seed": self.scenario.seed, "termination": outcome["termination"],
                        "peak_normal_n": round(outcome["peak_normal_n"], 2), "final_slip_mm": round(1000 * outcome["final_slip_m"], 1),
                        "mean_effort_n": round(outcome["mean_effort_n"], 2)})
        self.last_compare = out
        self.reset()
        return out

    def snapshot(self) -> dict:
        """Stable, JSON-safe view of the live state for displays (dashboard). applied_load_n is the
        simulator's load proxy: shown to humans, never given to policies."""
        env, st = self.env, self.env.stats
        c = env._contact
        rel = env._cup_pos() - env._ee_pos()
        return {"task": "fragile_cup", "mode": self.mode,
                "input_source": getattr(self.human, "label", None) if self.mode == TEACH else None,
                "input_status": getattr(self.human, "status", "ACTIVE") if self.mode == TEACH else None,
                "recording": self.recording, "saved_episodes": list(self.saved),
                "active_policy": self.active, "policies": list(self.policy_names),
                "seed": self.scenario.seed, "phase": env.phase(), "pouring": env.pour_t0 is not None,
                "t_s": round(env.t, 3), "applied_load_n": round(float(env.load_n), 3), "effort_cmd_n": round(env.effort, 3),
                "finger_normal_n": [round(float(x), 3) for x in c["normal"]],
                "finger_shear_z_n": [round(float(x), 3) for x in c["shear_z"]],
                "cup_slip_mm": round(1000 * float(env._grasp_rel[2] - rel[2]), 2),
                "damage_model": env.cfg.damage_model, "damage_ns": round(st.damage_ns, 3),
                "break_force_n": env.cfg.break_force_n, "outcome": st.termination,
                "last_compare": self.last_compare,
                # added for visualization (new keys only)
                "robot_joints": {f"j{j}": round(p.getJointState(env.robot, j, physicsClientId=env.client)[0], 5)
                                 for j in list(range(7)) + [9, 10]},
                "cup_pose": {"pos": [round(v, 5) for v in env._cup_pos()],
                             "quat": [round(v, 5) for v in p.getBasePositionAndOrientation(env.cup, physicsClientId=env.client)[1]]},
                "finger_normal_total_n": round(float(c["normal"].sum()), 3),
                "contact_convention": "per finger [L, R] in N (SIMULATED); total = L + R",
                "hold_status": self.hold_state, "schedule_id": schedule_id(self.scenario),
                "ramp_remaining_s": None if env.schedule_t is None else max(0.0, round(self.scenario.schedule.duration - env.schedule_t, 2)),
                "checkpoint": self._controller_name() if self.mode == AUTONOMOUS else None}

    def _controller_name(self) -> str | None:
        """Checkpoint name for learned controllers; the scripted controller's name otherwise (dashboard uses those)."""
        c = self.policies.get(self.active)
        pol = getattr(c, "policy", None)
        return pol.meta.get("name") if pol is not None and hasattr(pol, "meta") else getattr(c, "name", None)

    def status_lines(self) -> list[str]:
        if self.mode == TEACH:
            src = getattr(self.human, "label", "KEYBOARD")
            status = getattr(self.human, "status", "ACTIVE")
            msg = getattr(self.human, "last_message", "")
            keys = ("A: ARM glove   X: disarm   " if src == "GLOVE" else "Up/Down or W/S: effort   1-9: presets   ")
            return [f"TEACH  input: {src} ({status})   RECORDING: {'ON' if self.recording else 'off'}   saved: {len(self.saved)}"
                    + (f"   [{msg}]" if msg else ""),
                    keys + "P: Pour   R: reset   N: next   SPACE: record on/off"]
        pol = getattr(self.policies[self.active], "policy", None)
        if pol is None:                                    # scripted controller in AUTONOMOUS (dashboard panel)
            return [f"AUTONOMOUS  controller: {self._controller_name()} (SCRIPTED)  human grip input: DISCONNECTED",
                    "P: Add load   R: reset same scenario   N: next   T: switch   C: paired comparison"]
        meta = pol.meta
        prov = meta["provenance"]
        return [f"AUTONOMOUS  checkpoint: {meta['name']}  ({'+'.join(meta['feature_groups'])}; trained on "
                f"{len(prov['train_episode_ids'])} {'/'.join(prov['sources']).upper()} episodes)  human grip input: DISCONNECTED",
                "P: Pour   R: reset same scenario   N: next   T: switch policy   C: paired comparison   "
                "(TEACH recordings are not in this checkpoint until retrained)"]


def run_gui(mode: str, seed: int, checkpoints: list[str], teach_dir: str, cfg: CupConfig, glove=None) -> int:
    from .viewer import CupViewer
    env = CupHoldEnv(cfg)
    view = CupViewer(env, controls=True)
    if mode == TEACH:
        human = glove or KeyboardGrip()
        sess = LiveSession(env, TEACH, seed, teach_input=human, teach_dir=teach_dir, viewer=view)
    else:
        from .train import CupPolicy
        pols = {Path(c).name: PolicyController(CupPolicy.load(c)) for c in checkpoints}
        sess = LiveSession(env, AUTONOMOUS, seed, policies=pols, viewer=view)
    names = {"pour": "Add load (P)", "reset": "Reset same scenario (R)", "next": "Next scenario (N)"}
    names.update({"record": "Record on/off"} if mode == TEACH else {"switch": "Switch policy", "compare": "Paired comparison"})
    buttons = {k: view.button(v) for k, v in names.items()}
    counts = {k: view.read(b) for k, b in buttons.items()}
    print(f"[{mode}] window open. " + " | ".join(sess.status_lines()), flush=True)
    # Game-loop timing: one display frame (all GUI calls back-to-back, so one redraw) then as many
    # headless 30 Hz control ticks as needed to keep simulated time level with the wall clock.
    # Physics and the policy therefore run in real time at 30 Hz; only the window refresh rate varies.
    dt = env.cfg.dt
    t_start = time.monotonic()
    ticks = 0
    rate_t, rate_frames, rate_ticks = t_start, 0, 0
    try:
        while view.connected():
            # -- display frame: input + drawing, batched ------------------------------
            for k, b in buttons.items():
                val = view.read(b)
                if val != counts[k]:
                    counts[k] = val
                    if k == "compare":
                        print("[PAIRED] " + json.dumps(sess.paired_compare(realtime=True)), flush=True)
                        t_start, ticks = time.monotonic(), 0
                    else:
                        sess.do(k)
            down, pressed = view.keys()
            if "c" in pressed and mode == AUTONOMOUS:
                print("[PAIRED] " + json.dumps(sess.paired_compare(realtime=True)), flush=True)
                t_start, ticks = time.monotonic(), 0
                pressed = pressed - {"c"}
            sess.human_input(down, pressed)
            view.sync()
            lines = sess.status_lines()
            view.text("mode", lines[0], 0, (0.75, 0.1, 0.1) if mode == AUTONOMOUS else (0.1, 0.35, 0.75))
            view.text("help", lines[1], 1, (0.35, 0.35, 0.35))
            view.status()
            if sess.last_compare:
                view.text("cmp", "last comparison: " + "   ".join(f"{r['policy']}: {r['termination']}"
                                                               for r in sess.last_compare), 4)
            rate_frames += 1
            # -- control ticks: catch up with the wall clock (headless, 30 Hz) -----------
            due = int((time.monotonic() - t_start) / dt)
            n = 0
            while ticks < due and n < 10:
                if sess.mode == TEACH and n:
                    sess.human.handle_keys(down, set())      # held keys keep acting between frames
                sess.tick()
                ticks += 1
                n += 1
            if ticks < due:                                  # machine too slow: drop backlog, run in slow motion
                t_start += (due - ticks) * dt
            rate_ticks += n
            now = time.monotonic()
            if now - rate_t >= 5.0:
                print(f"[{mode}] control {rate_ticks / (now - rate_t) / 30.0:.2f}x real time, window "
                      f"{rate_frames / (now - rate_t):.1f} fps | " + view.status_values(env), flush=True)
                rate_t, rate_frames, rate_ticks = now, 0, 0
            time.sleep(max(0.0, t_start + (ticks + 1) * dt - time.monotonic()))
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        env.close()
        view.close()
        runner = getattr(sess.human, "runner", None)
        if runner is not None:
            info = runner.stop_recording()
            print(f"[GLOVE] raw recording closed: {info['path']} ({info['frames']} frames)")
    if sess.saved:
        print(f"[TEACH] saved {len(sess.saved)} episode(s) to {teach_dir}")
    return 0
