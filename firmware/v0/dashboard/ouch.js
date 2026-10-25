"use strict";
// OUCH mode: arm from the nav bar, then a firm handshake squeeze triggers a one-shot full-screen gag.
// Squeeze = at least MIN_PATCHES patches pressed past the sensitivity level for HOLD_S seconds.
// A patch is "pressed" by calibrated force (N) when reliable, otherwise by raw magnetic change (uT).
(function () {
  const UT_PER_N = 300;                                  // fallback scale when force is not calibrated (raw field change)
  const MIN_PATCHES = 1, HOLD_S = 0.35, ARM_GRACE_S = 1.5, SHOW_MS = 4200;
    const $ = (id) => document.getElementById(id);
  const st = {armed: false, armedAt: 0, released: false, since: null, busy: false, audio: null, };

  const level = () => {
    const n = Math.min(9, Math.max(1, Math.round(Number($("ouchForce").value) || 2)));   // whole newtons, 1-9
    return {n, ut: n * UT_PER_N};
  };
  function pressed(p, lv) {
    if (!p || !p.valid) return 0;
    if (p.forceReliable && Number.isFinite(p.force)) return p.force / lv.n;
    return Number.isFinite(p.loadSignal) ? p.loadSignal / lv.ut : 0;
  }
  // Real recorded "hurt" clips (CC0, see sounds/LICENSE.txt). <audio> is used because the CSP only allows fetch() to ws://.
  const CLIPS = [1, 2, 3, 4, 5].map((n) => { const a = new Audio(`sounds/cartoon${n}.mp3`); a.preload = "auto"; return a; });
  function ouchSound() {
    const a = CLIPS[Math.floor(Math.random() * CLIPS.length)];
    a.currentTime = 0; a.volume = 1;
    a.play().catch(() => squeak());                         // autoplay blocked or file missing: synthetic boing
  }
  function unlockAudio() { for (const a of CLIPS) { a.muted = true; a.play().then(() => { a.pause(); a.currentTime = 0; a.muted = false; }).catch(() => { a.muted = false; }); } }
  function squeak() {
    try {
      const ac = st.audio ||= new (window.AudioContext || window.webkitAudioContext)();
      const t = ac.currentTime, o = ac.createOscillator(), g = ac.createGain();
      o.type = "square"; o.frequency.setValueAtTime(900, t); o.frequency.exponentialRampToValueAtTime(220, t + .35);
      g.gain.setValueAtTime(.18, t); g.gain.exponentialRampToValueAtTime(.001, t + .4);
      o.connect(g).connect(ac.destination); o.start(t); o.stop(t + .42);
    } catch (e) { /* audio is optional */ }
  }
  function setArmed(on) {
    st.armed = on; st.since = null; st.released = false; st.armedAt = performance.now() / 1000;
    const b = $("ouchBtn"); b.classList.toggle("armed", on);
    b.setAttribute("aria-pressed", on);
    $("ouchState").textContent = on ? "ARMED" : "OFF";
    if (on) unlockAudio();                                   // the toggle click is the user gesture browsers require
  }
  function fire(info) {
    if (st.busy) return;
    st.busy = true; setArmed(false);                         // one-shot: toggle turns itself off
    const o = $("ouch"); $("ouchInfo").textContent = info;
    o.hidden = false; o.classList.remove("out"); void o.offsetWidth; o.classList.add("in");
    document.body.classList.add("ouch-shake");
    ouchSound();
    st.timer = setTimeout(dismiss, SHOW_MS);
  }
  function dismiss() {
    clearTimeout(st.timer);
    const o = $("ouch"); if (o.hidden) return;
    o.classList.add("out"); document.body.classList.remove("ouch-shake");
    setTimeout(() => { o.hidden = true; o.classList.remove("in", "out"); st.busy = false; }, 700);
  }
  function feed(snap) {
    if (!st.armed || st.busy) return;
    const now = performance.now() / 1000, lv = level();
    const scores = Object.entries(snap.patches).map(([n, p]) => [n, pressed(p, lv)]);
    const hit = scores.filter(([, s]) => s >= 1);
    if (!scores.some(([, s]) => s >= .5)) st.released = true;          // must see an open hand first
    if (!st.released || now - st.armedAt < ARM_GRACE_S) { st.since = null; return; }
    if (hit.length < MIN_PATCHES) { st.since = null; return; }
    st.since ??= now;
    if (now - st.since >= HOLD_S) {
      const f = Object.values(snap.patches).map((p) => p.force).filter(Number.isFinite);
      fire(`${hit.length} patch${hit.length > 1 ? "es" : ""} squeezed${f.length ? ` · peak ${Math.max(...f).toFixed(1)} N` : ""} · ${hit.map(([n]) => n).join(" + ")}`);
    }
  }
  try { $("ouchForce").value = localStorage.getItem("dermis.ouchN") || $("ouchForce").value; } catch (e) {}
  $("ouchForce").onchange = () => { const n = level().n; $("ouchForce").value = n; try { localStorage.setItem("dermis.ouchN", n); } catch (e) {} };
  $("ouchBtn").onclick = () => setArmed(!st.armed);
  $("ouchTest").onclick = () => { fire("TEST · no hands were harmed"); };
  $("ouch").onclick = dismiss;
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") dismiss(); });
  window.Ouch = {feed, fire, setArmed};
})();
