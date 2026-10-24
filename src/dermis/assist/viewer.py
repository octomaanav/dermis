"""Display-only 3-D view in a SEPARATE PROCESS (native PyBullet window).

The control loop writes the newest state (joint angles read back from physics, mug pose, banner, hold
status) into a shared-memory slot that is simply overwritten (never a queue, so a stalled window can
never block control); the child redraws from the latest slot as fast as it can. The view
cannot affect physics, and a slow redraw can never slow control (measured: one GUI redraw costs
~80 ms on this Mac). The window reuses the teleop HandSim GUI (camera centred on hand and mug).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time


NQ = 9


def _child(shared, banner, hold, stop, fps_out, opts: str) -> None:
    import pybullet as p
    from ..server.teleop.hand import HandSim
    from .env import HANDLE, MUG
    h = HandSim(gui=True)
    P = dict(physicsClientId=h.client)
    for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
        p.configureDebugVisualizer(flag, 0, **P)
    p.setGravity(0, 0, 0, **P)
    p.setJointMotorControlArray(h.body, h.joints, p.VELOCITY_CONTROL, forces=[0.0] * len(h.joints), **P)
    qy = p.getQuaternionFromEuler((math.pi / 2, 0, 0))
    vis = p.createVisualShapeArray([p.GEOM_CYLINDER, p.GEOM_CYLINDER], radii=[HANDLE["radius"], MUG["radius"]],
                                   lengths=[HANDLE["length"], MUG["height"]],
                                   visualFramePositions=[(0, 0, 0), (0, MUG["dy"], MUG["dz"])],
                                   visualFrameOrientations=[qy, (0, 0, 0, 1)],
                                   rgbaColors=[(0.95, 0.95, 0.98, 1), (0.85, 0.88, 0.95, 0.95)], **P)
    mug = p.createMultiBody(0, -1, vis, (HANDLE["x"], 0, HANDLE["z"]), **P)
    p.resetDebugVisualizerCamera(0.42, 35, -20, (0.08, -0.03, 0.17), **P)
    texts, shown, arrow, arrow_key = {}, {}, None, None
    frames, t0 = 0, time.monotonic()

    def text(key, s, z, color):
        if shown.get(key) == s:
            return
        shown[key] = s
        kw = dict(textColorRGB=color, textSize=1.4, **P)
        if key in texts:
            kw["replaceItemUniqueId"] = texts[key]
        texts[key] = p.addUserDebugText(s or " ", (0.0, -0.18, z), **kw)

    last_seq = -1.0
    while p.isConnected(h.client) and not stop.is_set():
        with shared.get_lock():
            v = list(shared)
            b, hl = banner.value.decode(errors="replace"), hold.value.decode(errors="replace")
        if v[0] != last_seq:
            last_seq = v[0]
            q, pos, orn, load, alarm = v[1:1 + NQ], v[1 + NQ:4 + NQ], v[4 + NQ:8 + NQ], v[8 + NQ], v[9 + NQ] > 0.5
            for j, qq in zip(h.joints, q):
                p.resetJointState(h.body, j, qq, **P)
            p.resetBasePositionAndOrientation(mug, pos, orn, **P)
            text("banner", b, 0.36, (0.1, 0.25, 0.7))
            text("hold", hl, 0.33, (0.75, 0.1, 0.1) if alarm else (0.1, 0.5, 0.2))
            key = round(load, 1)
            if key != arrow_key:
                arrow_key = key
                start = (pos[0], pos[1] + MUG["dy"] - MUG["radius"] - 0.01, pos[2])
                kw = dict(lineColorRGB=(0.85, 0.2, 0.1), lineWidth=4, **P)
                if arrow is not None:
                    kw["replaceItemUniqueId"] = arrow
                arrow = p.addUserDebugLine(start, (start[0], start[1], start[2] - max(0.012 * load, 1e-4)), **kw)
        p.stepSimulation(**P)                          # redraw (nothing in this world is dynamic)
        frames += 1
        now = time.monotonic()
        if now - t0 >= 2.0:
            fps_out.value = frames / (now - t0)
            frames, t0 = 0, now
        time.sleep(0.005)
    h.close()


class ProcessViewer:
    def __init__(self, opts: str = ""):
        ctx = mp.get_context("spawn")
        self._shared = ctx.Array("d", 10 + NQ)            # seq, q[9], mug pos[3], orn[4], load, alarm
        self._banner, self._hold = ctx.Array("c", 200), ctx.Array("c", 80)
        self._stop = ctx.Event()
        self._fps = ctx.Value("d", -1.0)
        self.proc = ctx.Process(target=_child, args=(self._shared, self._banner, self._hold, self._stop, self._fps, opts),
                                daemon=True)
        self.proc.start()
        self._n = 0

    def send(self, env, banner: str, hold: str) -> None:
        """Overwrite the latest-state slot (15 Hz). Never blocks on the display."""
        self._n += 1
        if self._n % 2:
            return
        import pybullet as p
        q = env.joint_state()["q"]
        pos, orn = p.getBasePositionAndOrientation(env.mug, **env._p)
        alarm = 1.0 if ("DROPPED" in hold or "EXCESSIVE" in hold) else 0.0
        with self._shared.get_lock():
            self._shared[:] = [float(self._n)] + list(q[:NQ]) + list(pos) + list(orn) + [env.load_n, alarm]
            self._banner.value = banner.encode()[:199]
            self._hold.value = hold.encode()[:79]

    def fps(self):
        v = self._fps.value
        return None if v < 0 else round(v, 1)

    def close(self) -> None:
        self._stop.set()
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
