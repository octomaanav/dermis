"use strict";
/* Dermis console. Data: /api/stream (SSE, 10 Hz, one stream per tab) + REST for files.
   Every number comes from the backend; nothing is invented. Browser buffers are bounded. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 1) => (v === null || v === undefined || !isFinite(v)) ? "—" : Number(v).toFixed(d);
const api = async (p) => { const r = await fetch(p); const j = await r.json(); if (!r.ok) throw new Error(j.error || r.status); return j; };
const post = async (p, body) => {
  const r = await fetch(p, {method: "POST", headers: {"Content-Type": "application/json", "X-Dermis": "1"}, body: JSON.stringify(body || {})});
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.status); return j;
};
const store = {get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } },
               set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} }};
function toast(msg, kind = "") { const t = $("toast"); t.textContent = msg; t.className = kind; t.style.display = "block";
  clearTimeout(toast.h); toast.h = setTimeout(() => t.style.display = "none", 4200); }

const S = {status: null, metrics: null, pods: [], podsOrder: [], pod: store.get("pod", null), defs: {}, kind: null,
           buf: [], cmds: [], events: [], cup: null, hasBase: {}, podHist: {}, paused: false, cursorT: null, tNow: 0, seen: new Set()};
const MAXBUF = 20000, MAXT = 30;
const COL = {x: "#22d3ee", y: "#f472b6", z: "#a3e635", w: "#fbbf24"};
const HEALTH = {healthy: ["var(--ok)", "✓"], degraded: ["var(--warn)", "!"], invalid: ["var(--bad)", "✕"], missing: ["var(--dim)", "?"]};

/* ---------------------------------------------------------------- views */
function setView(v) {
  $("pres").classList.toggle("on", v === "pres"); $("eng").classList.toggle("on", v === "eng");
  $("vPres").classList.toggle("on", v === "pres"); $("vEng").classList.toggle("on", v === "eng");
  store.set("view", v); scheduleDraw();
}
$("vPres").onclick = () => setView("pres"); $("vEng").onclick = () => setView("eng");

/* ---------------------------------------------------------------- live stream */
let es = null, lastMsg = 0;
function connect() {
  if (es) es.close();                                   // never two subscriptions from one tab
  es = new EventSource("/api/stream" + (S.pod ? "?pod=" + encodeURIComponent(S.pod) : ""));
  es.onmessage = (e) => { lastMsg = performance.now(); onPayload(JSON.parse(e.data)); };
  es.onerror = () => { $("bLink").textContent = "console link lost — reconnecting…"; $("bLink").className = "small warn"; };
}
setInterval(() => { if (performance.now() - lastMsg > 3000 && lastMsg) { $("bLink").textContent = "no console updates for >3 s"; $("bLink").className = "small warn"; } }, 1000);

function onPayload(d) {
  $("bLink").textContent = "live · 10 Hz"; $("bLink").className = "small dim";
  if (d.status) S.status = d.status;
  S.metrics = d.metrics; S.pods = d.pods; S.cup = d.cup; S.hasBase = d.has_baseline; S.tNow = d.t;
  if (JSON.stringify(S.podsOrder) !== JSON.stringify(d.pods_order)) { S.podsOrder = d.pods_order; buildPodSelect(); }
  if (S.pod !== d.pod) { S.pod = d.pod; }
  for (const s of d.samples) S.buf.push(s);
  for (const c of d.commands) S.cmds.push(c);
  for (const e of d.events) S.events.push(e);
  trim();
  for (const p of d.pods) if (p.health !== "missing") S.seen.add(p.pod);
  for (const p of d.pods) { const h = (S.podHist[p.pod] ||= []); h.push({t: d.t, v: p.db_ut ?? p.b_abs_ut, db: p.db_ut !== null && p.db_ut !== undefined, health: p.health}); while (h.length && h[0].t < d.t - 12) h.shift(); }
  S.dirty = true;
  scheduleDraw();
}
// DOM panels (status bar, pod map, tables) are rebuilt at most 4 times a second and only when the view
// is visible; charts redraw with requestAnimationFrame. Keeps the browser's CPU use, which competes
// with the control loop on a busy machine, low.
const setHTML = (el, html) => { if (el._h !== html) { el._h = html; el.innerHTML = html; return true; } return false; };
setInterval(() => { if (!S.dirty || document.hidden) return; S.dirty = false;
  renderStatus(); if ($("pres").classList.contains("on")) renderPres(); else renderEng(); }, 250);
function trim() {
  const lo = S.tNow - MAXT;
  while (S.buf.length && (S.buf[0].t < lo || S.buf.length > MAXBUF)) S.buf.shift();
  while (S.cmds.length && (S.cmds[0].t < lo || S.cmds.length > 4000)) S.cmds.shift();
  if (S.events.length > 300) S.events.splice(0, S.events.length - 300);
}

/* ---------------------------------------------------------------- status bar */
function renderStatus() {
  const st = S.status; if (!st) return;
  const src = st.source || "NONE";
  $("bSource").textContent = src; $("bSource").className = "tag " + (src === "NO SOURCE" ? "NONE" : src.split(" ")[0]);
  const mode = st.control_mode === "NO ROBOT CONTROL" ? "MONITOR" : st.control_mode;
  $("bMode").textContent = st.control_mode; $("bMode").className = "tag " + mode;
  $("bModeDetail").textContent = st.control_state && st.control_state !== st.control_mode ? st.control_state : "";
  const c = st.connection || {}; const good = ["connected", "receiving"].includes(c.state);
  $("bConnDot").className = "dot " + (good ? "ok" : c.state === "error" ? "bad" : "warn");
  $("bConn").textContent = (c.state || "—") + (c.peer ? " · " + c.peer : "") + (c.epoch ? " · #" + c.epoch : "");
  $("bDev").textContent = (st.device_id || "—") + " · cfg " + (st.config_id ?? "—");
  const r = st.recording || {}; const raw = r.raw || {}; const demo = r.demo;
  $("bRec").innerHTML = raw.active || demo ? `<span class="rec">● REC</span> ` + (raw.active ? `raw ${raw.frames} fr${raw.duration_s ? " · " + raw.duration_s + " s" : ""}` : "") +
    (demo ? ` demo ${demo.ticks} ticks` : "") : '<span class="muted">○ not recording</span>';
  const cal = st.calibration;
  $("bCal").innerHTML = st.calibrating ? '<span class="warn">◐ calibrating…</span>' : !cal || !cal.active ? '<span class="muted">✕ not calibrated</span>' :
    cal.compatible === false ? `<span class="bad">⚠ calibration incompatible</span>` : `<span class="ok">✓ calibrated</span> <span class="small muted">${cal.created_unix ? new Date(cal.created_unix * 1000).toLocaleTimeString() : esc(cal.id || "")}</span>`;
  const fa = st.frame_age_ms ?? S.metrics?.frame_age_ms; const ua = st.usable_age_ms;
  $("bAge").innerHTML = `<span class="${fa === null || fa === undefined ? "dim" : fa > 250 ? "bad" : fa > 100 ? "warn" : "ok"}">${fmt(fa, 0)} ms</span>` +
    (st.app === "teleop" ? ` <span class="dim">usable</span> <span class="${ua === null ? "dim" : ua > (st.watchdog_ms || 250) ? "bad" : "ok"}">${fmt(ua, 0)} ms</span>` : "");
}

/* ---------------------------------------------------------------- hand / pod map */
const POS = {thumb_tip: [70, 205], index_tip: [168, 92], middle_tip: [210, 70], ring_tip: [250, 88], little_tip: [287, 120],
             thenar_apb: [160, 318], hypothenar: [268, 322], palm_center: [212, 278], hand_dorsum: [212, 236], wrist: [212, 400]};
// label anchor offset [dx, dy, text-anchor] chosen so labels sit outside the hand and never overlap each other
const LAB = {thumb_tip: [-6, 34, "middle"], index_tip: [-30, -4, "end"], middle_tip: [0, -34, "middle"], ring_tip: [30, -14, "start"],
             little_tip: [26, 4, "start"], thenar_apb: [-28, 4, "end"], hypothenar: [26, 4, "start"], palm_center: [26, 4, "start"],
             hand_dorsum: [26, 4, "start"], wrist: [28, 4, "start"]};
function handSVG(big) {
  const flex = {}; const last = S.seen.size ? (S.cmds[S.cmds.length - 1] || {}) : {};      // no telemetry yet: no estimate to show
  for (const k in last) if (k.startsWith("flex:")) flex[k.slice(5)] = last[k];
  const all = S.pods.length ? S.pods : S.podsOrder.map((p) => ({pod: p, health: "missing"}));
  const pods = S.seen.size ? all.filter((p) => S.seen.has(p.pod)) : all;          // only pods actually present in the stream
  const unseen = S.seen.size ? all.filter((p) => !S.seen.has(p.pod)).map((p) => p.pod) : [];
  const extra = pods.filter((p) => !POS[p.pod]);
  let s = `<svg class="hand" viewBox="-30 -40 480 490" style="max-height:${big ? 480 : 340}px" role="img" aria-label="Glove pod map">
   <text x="-26" y="444" class="sub">palm view · right hand · positions schematic</text>
   <g class="skin"><rect x="128" y="180" width="170" height="185" rx="38"/>
    <rect x="152" y="70" width="34" height="140" rx="17"/><rect x="194" y="48" width="34" height="160" rx="17"/>
    <rect x="234" y="66" width="32" height="145" rx="16"/><rect x="272" y="100" width="30" height="115" rx="15"/>
    <path d="M140 300 C 105 280, 70 250, 58 205 C 52 185, 80 175, 92 192 C 110 225, 130 240, 150 250 Z"/>
    <rect x="168" y="360" width="90" height="70" rx="10"/></g>`;
  for (const p of pods) {
    const xy = POS[p.pod]; if (!xy) continue;
    const [col, sym] = HEALTH[p.health] || HEALTH.missing; const [x, y] = xy; const sel = p.pod === S.pod;
    const val = p.db_ut !== null && p.db_ut !== undefined ? `|ΔB| ${fmt(p.db_ut, 0)} µT` : p.b_abs_ut !== null && p.b_abs_ut !== undefined ? `|B| ${fmt(p.b_abs_ut / 1000, 1)} mT` : p.reason || p.mag || "";
    const imu = p.imu === "absent" ? "no IMU" : p.imu ? `IMU ${p.imu === "fresh" ? "✓" : p.imu === "stale" ? "stale" : "✕"}` : "";
    const f = flex[p.pod];
    const [dx, dy, anchor] = LAB[p.pod] || [24, 4, "start"]; const lx = x + dx, nl = f !== undefined ? 4 : 3;
    const ty = dy < 0 ? y + dy - 12 * (nl - 1) : y + dy;            // labels above a pod stack upwards
    s += `<g class="pod ${sel ? "sel" : ""}" data-pod="${esc(p.pod)}">
      <circle class="ring" cx="${x}" cy="${y}" r="24"/>
      ${f !== undefined ? `<circle cx="${x}" cy="${y}" r="29" fill="none" stroke="#14b8a6" stroke-width="4" stroke-dasharray="${(f * 182).toFixed(1)} 182" transform="rotate(-90 ${x} ${y})"><title>estimated flex ${f.toFixed(2)} (EXPERIMENTAL mapping output)</title></circle>` : ""}
      <circle cx="${x}" cy="${y}" r="17" fill="${col}" stroke="#05080b" stroke-width="2"><title>${esc(p.pod)}: ${esc(p.health)}</title></circle>
      <text x="${x}" y="${y + 5}" text-anchor="middle" style="font-weight:800;fill:#05080b">${sym}</text>
      <text x="${lx}" y="${ty}" text-anchor="${anchor}" style="font-weight:700;font-size:12px">${esc(p.pod.replace("_tip", "").replace("_", " "))}</text>
      <text x="${lx}" y="${ty + 12}" text-anchor="${anchor}" class="sub">${esc(val)}</text>
      <text x="${lx}" y="${ty + 24}" text-anchor="${anchor}" class="sub">${esc(imu)}${p.mag_age_ms !== null && p.mag_age_ms !== undefined && p.health !== "missing" ? " · " + fmt(p.mag_age_ms, 0) + " ms" : ""}</text>
      ${f !== undefined ? `<text x="${lx}" y="${ty + 36}" text-anchor="${anchor}" class="sub" style="fill:#2dd4bf">est. flex ${f.toFixed(2)}</text>` : ""}
    </g>`;
  }
  s += "</svg>";
  if (unseen.length) s += `<div class="small dim">Configured but not in the stream: ${unseen.map(esc).join(", ")}</div>`;
  if (extra.length) s += `<div class="small muted">Other pods: ${extra.map((p) => `<a href="#" data-pod="${esc(p.pod)}">${esc(p.pod)}</a> (${esc(p.health)})`).join(", ")}</div>`;
  if (Object.keys(flex).length) s += `<div class="small muted">Teal arc = estimated finger flex from the EXPERIMENTAL mapping (not a measured finger pose).</div>`;
  return s;
}
function bindPods(el) { el.querySelectorAll("[data-pod]").forEach((g) => g.onclick = (e) => { e.preventDefault(); selectPod(g.dataset.pod); }); }
function selectPod(p) { if (p === S.pod) return; S.pod = p; store.set("pod", p); S.buf = []; $("podSel").value = p; connect(); renderPres(); renderEng(); }
function buildPodSelect() { $("podSel").innerHTML = S.podsOrder.map((p) => `<option>${esc(p)}</option>`).join(""); if (S.pod) $("podSel").value = S.pod; }
$("podSel").onchange = (e) => selectPod(e.target.value);

/* ---------------------------------------------------------------- presentation */
function renderPres() {
  const m = S.metrics, st = S.status; if (!m || !st) return;
  $("kFps").innerHTML = `${fmt(m.accepted_fps, 0)}<span class="u">/s</span>`; $("kFpsS").textContent = `received ${fmt(m.received_fps, 0)}/s · ${m.window_s} s window`;
  $("kUsable").innerHTML = m.usable_pct === null ? "—" : `${fmt(m.usable_pct, 0)}<span class="u">%</span>`; $("kUsableS").textContent = m.usable_denominator || "";
  $("kAge").innerHTML = m.frame_age_ms === null ? "—" : `${fmt(m.frame_age_ms, 0)}<span class="u">ms</span>`;
  $("kAge").className = "v " + (m.frame_age_ms === null ? "" : m.frame_age_ms > 250 ? "bad" : m.frame_age_ms > 100 ? "warn" : "");
  $("kMissing").textContent = m.missing_ids.connection;
  const hp = $("handPres"); if (setHTML(hp, handSVG(true))) bindPods(hp);
  $("tlUnit").textContent = Object.values(S.hasBase).some((x) => x) ? "|ΔB| from unloaded baseline, µT · last 10 s" : "|B| magnitude, µT (no baseline yet) · last 10 s";
  explainMode();
}
function explainMode() {
  const st = S.status, cup = S.cup?.snapshot;
  const src = st.source, srcTxt = {SYNTHETIC: "SYNTHETIC development data (not a real glove)", REPLAY: "a REPLAY of recorded data", GLOVE: "the live glove"}[src] || src;
  const stages = ["Glove telemetry", "Signal processing", "Recording", "Training", "Evaluation"];
  let on = [0, 1], html;
  if (st.app === "teleop") {
    if (st.control_mode === "TELEOPERATION") {
      on = [0, 1];
      html = `<b>TELEOPERATION</b> · ${esc(srcTxt)} → validated → baseline-subtracted magnetic change → <b>EXPERIMENTAL mapping</b> → simulated ${esc(st.robot)}. ` +
        `${st.control_state === "PAUSED" ? `<span class="warn">Paused: ${esc(st.control_reason || "")}. Motion stopped, pose held.</span>` : "Telemetry is driving the robot now."} Not training, not autonomous.`;
    } else if (src === "NO SOURCE") {
      html = `<b>DISARMED</b> · waiting for a telemetry connection on <span class="mono">${esc(st.connection?.listen || "")}</span>. ` +
        `Start the glove bridge or <span class="mono">python -m dermis send-test --mode synthetic</span>. The robot holds still.`;
      on = [];
    } else {
      html = `<b>DISARMED</b> · ${esc(srcTxt)} is received and displayed, but the robot holds still. ${esc(st.control_reason || "")}`;
    }
    if (st.recording?.raw?.active || st.recording?.demo) on.push(2);
  } else if (cup) {
    on = [4];
    html = `<b>AUTONOMOUS</b> · the fragile-cup controller <b>${esc(cup.active_policy)}</b> (${esc(S.cup.kinds?.[cup.active_policy] || "")}) holds the cup in simulation. ` +
      `The telemetry stream (${esc(src)}) stays visible, but no human motor command reaches the robot in this mode.`;
  } else {
    html = `<b>MONITOR</b> · ${esc(srcTxt)} through the ingestion pipeline. No robot is controlled by this process; datasets, training and evaluation are shown from disk.`;
  }
  setHTML($("flow"), stages.map((s, i) => `<span class="${on.includes(i) ? "on" : ""}">${s}</span>`).join("<i>→</i>"));
  setHTML($("modeExplain"), html);
  if (cup) { $("outcome").innerHTML = `<span class="${cup.outcome === "success" ? "ok" : cup.outcome ? "bad" : "acc"}">${esc(cup.outcome ? cup.outcome.replace("_", " ") : cup.phase)}</span>`;
    $("outcomeS").textContent = `fragile cup · seed ${cup.seed} · t ${fmt(cup.t_s, 1)} s · ${cup.pouring ? "pouring" : "not pouring"}`; }
  else if (st.app === "teleop") { $("outcome").innerHTML = `<span class="${st.control_state === "ARMED" ? "ok" : st.control_state === "PAUSED" ? "warn" : "muted"}">${esc(st.control_state)}</span>`;
    $("outcomeS").textContent = st.control_reason || (st.control_valid ? "input usable" : ""); }
  else { $("outcome").innerHTML = `<span class="muted">monitoring</span>`; $("outcomeS").textContent = "no episode running in this process"; }
}

/* ---------------------------------------------------------------- simulation panel */
let simTimer = null;
function updateSim() {
  const st = S.status; if (!st) return;
  const feed = st.feed;
  const img = $("simImg"), cv = $("simSchem");
  if (feed && feed.hz > 0) {
    img.style.display = "block"; cv.style.display = "none";
    if (!simTimer) simTimer = setInterval(() => { img.src = "/api/sim.jpg?t=" + Date.now(); }, Math.max(200, 1000 / feed.hz));
    $("simLabel").textContent = `camera feed ${feed.hz} Hz · render ${fmt(feed.render_ms_median, 0)} ms median on the sim thread`;
    $("simNote").textContent = feed.note || "";
  } else {
    if (simTimer) { clearInterval(simTimer); simTimer = null; }
    img.style.display = "none"; cv.style.display = "block";
    $("simLabel").textContent = "schematic of measured simulator state";
    $("simNote").innerHTML = st.app === "teleop" ? `Not a camera render: drawn from joint angles / positions read back from PyBullet each tick. ` +
      `The native PyBullet window shows the 3D scene${feed?.note ? "; " + esc(feed.note) : ""}. A GPU camera feed is opt-in (<span class="mono">--dashboard-feed-hz 2</span>, GUI only) because it costs ~36 ms per frame on the control thread.`
      : S.cup ? "Schematic from the fragile-cup session's simulator state (no camera feed: software rendering would stall the controller thread)."
      : "No robot in this process. For live teleoperation run <span class=\"mono\">python -m dermis teleop --source tcp --gui --dashboard-port 8000</span>.";
  }
}
function drawSchematic() {
  const cv = $("simSchem"); if (cv.style.display === "none") return;
  const g = ctx(cv), W = cv.clientWidth, H = cv.clientHeight; const st = S.status; if (!st) return;
  g.fillStyle = css("--s2"); g.fillRect(0, 0, W, H);
  const rs = st.robot_state || {}; const cup = S.cup?.snapshot;
  g.font = "12px system-ui"; g.fillStyle = css("--muted");
  if (rs.robot === "hand" && rs.joints) {
    const pitch = rs.palm_rp ? rs.palm_rp[1] : 0, off = rs.offset || [0, 0, 0];
    const cx = W * 0.32 + off[0] * W * 1.2, cy = H * 0.42 - off[2] * H * 1.2;
    g.strokeStyle = css("--dim"); g.setLineDash([4, 4]); g.beginPath(); g.moveTo(W * 0.32, H * 0.12); g.lineTo(W * 0.32, H * 0.85); g.stroke(); g.setLineDash([]);
    g.fillText("home", W * 0.32 + 4, H * 0.1);
    g.save(); g.translate(cx, cy); g.rotate(pitch);
    g.fillStyle = "#3a4656"; g.fillRect(-W * 0.3, -14, W * 0.16, 28);              // forearm
    g.fillStyle = "#d9b99b"; g.fillRect(-W * 0.14, -16, W * 0.15, 32);              // palm (side view)
    const names = ["thumb", "index", "middle"], cols = ["#3b82f6", "#f59e0b", "#a855f7"], L = [0.11, 0.075, 0.06];
    for (let f = 0; f < 3; f++) {
      let x = W * 0.01, y = f === 0 ? 10 : (f === 1 ? -6 : 2), a = f === 0 ? 0.6 : 0;
      g.strokeStyle = cols[f]; g.lineWidth = f === 0 ? 7 : 6; g.lineCap = "round"; g.beginPath(); g.moveTo(x, y);
      for (let j = 0; j < 3; j++) { a += rs.joints[f * 3 + j] || 0; x += Math.cos(a) * L[j] * W; y += Math.sin(a) * L[j] * W; g.lineTo(x, y); }
      g.stroke(); g.fillStyle = cols[f]; g.fillText(names[f], x + 8, y + 4 + (f - 1) * 13);
    }
    g.restore();
    g.fillStyle = css("--muted");
    g.fillText(`wrist offset x ${fmt(off[0] * 100, 1)} cm · z ${fmt(off[2] * 100, 1)} cm · pitch ${fmt(pitch * 57.3, 0)}°`, 12, H - 12);
  } else if (rs.robot === "arm" && rs.ee) {
    const op = rs.opening ?? 0, cmd = rs.cmd_opening ?? 0, cx = W / 2, cy = H / 2, s = W * 1.8;
    g.fillStyle = "#3a4656"; g.fillRect(cx - 40, cy - 70, 80, 30);
    g.fillStyle = "#94a3b8"; g.fillRect(cx - op * s / 2 - 14, cy - 40, 14, 80); g.fillRect(cx + op * s / 2, cy - 40, 14, 80);
    g.strokeStyle = "#22d3ee"; g.setLineDash([4, 3]); g.strokeRect(cx - cmd * s / 2 - 14, cy - 40, cmd * s + 28, 80); g.setLineDash([]);
    g.fillStyle = css("--muted"); g.fillText(`measured opening ${fmt(op * 1000, 1)} mm · commanded ${fmt(cmd * 1000, 1)} mm (dashed)`, 12, H - 30);
    g.fillText(`end effector (${rs.ee.map((v) => fmt(v, 3)).join(", ")}) m`, 12, H - 12);
  } else if (cup) {
    const thr = cup.break_force_n, cx = W / 2, cy = H / 2;
    g.fillStyle = "#cbd5e1"; g.fillRect(cx - 26, cy - 55 + cup.cup_slip_mm, 52, 110);
    cup.finger_normal_n.forEach((fN, i) => { const x = i ? cx + 30 : cx - 44; const frac = Math.min(1.2, fN / thr);
      g.fillStyle = fN > thr ? css("--bad") : "#14b8a6"; g.fillRect(x, cy - 10, 14, 20);
      g.fillText(`${fmt(fN, 1)} N`, i ? x + 20 : x - 46, cy + 4); g.fillStyle = css("--s3"); g.fillRect(i ? W - 120 : 20, 20 + 0, 100, 8);
      g.fillStyle = fN > thr ? css("--bad") : "#14b8a6"; g.fillRect(i ? W - 120 : 20, 20, 100 * Math.min(1, frac), 8); g.fillStyle = css("--muted");
      g.fillText(`${i ? "right" : "left"} finger normal / threshold`, i ? W - 120 : 20, 44); });
    g.fillStyle = "#fbbf24"; g.beginPath(); g.moveTo(cx, cy - 90); g.lineTo(cx, cy - 60); g.stroke();
    g.fillStyle = css("--muted");
    g.fillText(`load proxy (simulated) ${fmt(cup.applied_load_n, 1)} N · slip ${fmt(cup.cup_slip_mm, 1)} mm · threshold ${thr} N per finger (simulated)`, 12, H - 12);
  } else {
    g.fillStyle = css("--dim"); g.font = "15px system-ui"; g.fillText(st.app === "teleop" ? "waiting for the first control tick…" : "no robot in this process", 20, H / 2);
  }
}

/* ---------------------------------------------------------------- canvas charts */
function ctx(cv) { const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); return g; }
function ema(pts, tau) { let y = null, tp = null; return pts.map((p) => { if (p.v === null) { y = null; tp = null; return {t: p.t, v: null}; }
  y = y === null ? p.v : y + (1 - Math.exp(-(p.t - tp) / tau)) * (p.v - y); tp = p.t; return {t: p.t, v: y}; }); }
function chart(cv, o) {
  const g = ctx(cv), W = cv.clientWidth, H = cv.clientHeight, L = o.compact ? 4 : 52, R = o.compact ? 70 : 8, T = o.compact ? 4 : 8, B = o.compact ? 4 : 18;
  g.font = "10.5px system-ui";
  const vals = []; o.series.forEach((s) => s.pts.forEach((p) => { if (p.v !== null && isFinite(p.v)) vals.push(p.v); }));
  if (o.hline !== undefined) vals.push(o.hline);
  if (!vals.length) { g.fillStyle = css("--dim"); g.font = "12px system-ui"; g.fillText(o.empty || "no usable samples in this window", L, H / 2); return null; }
  let lo = Math.min(...vals), hi = Math.max(...vals); if (hi - lo < (o.minSpan || 1e-6)) { const m = (hi + lo) / 2, h2 = (o.minSpan || 1e-6) / 2; lo = m - h2; hi = m + h2; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const X = (t) => L + (W - L - R) * (t - o.t0) / (o.t1 - o.t0), Y = (v) => H - B - (H - B - T) * (v - lo) / (hi - lo);
  g.strokeStyle = css("--line"); g.lineWidth = 1; g.fillStyle = css("--dim");
  const lab = (v) => Math.abs(v) >= 10000 ? v.toExponential(2) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
  if (o.compact) { const last = [...o.series[0].pts].reverse().find((p) => p.v !== null && isFinite(p.v));
    g.fillStyle = css("--ink"); g.font = "12px system-ui"; g.fillText(last ? lab(last.v) + " " + o.unit : "—", W - R + 6, H / 2 + 4); g.font = "10.5px system-ui"; }
  else { for (let k = 0; k <= 3; k++) { const v = lo + (hi - lo) * k / 3, y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.fillText(lab(v), 2, y + 3); }
    g.fillText(o.xlabel || `${(o.t1 - o.t0).toFixed(0)} s window · ${o.unit}`, L, H - 4); }
  if (o.hline !== undefined) { g.strokeStyle = css("--bad"); g.setLineDash([5, 4]); g.beginPath(); g.moveTo(L, Y(o.hline)); g.lineTo(W - R, Y(o.hline)); g.stroke(); g.setLineDash([]); }
  (o.vlines || []).forEach((v) => { if (v.t < o.t0 || v.t > o.t1) return; g.strokeStyle = v.color || "#fbbf24"; g.globalAlpha = .7; g.beginPath(); g.moveTo(X(v.t), T); g.lineTo(X(v.t), H - B); g.stroke(); g.globalAlpha = 1; });
  for (const s of o.series) {
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.5; g.setLineDash(s.dash ? [4, 3] : []); g.globalAlpha = s.alpha || 1; g.beginPath();
    let pen = false; for (const p of s.pts) { if (p.v === null || !isFinite(p.v) || p.t < o.t0) { pen = false; continue; } const x = X(p.t), y = Y(p.v); pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true; }
    g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    g.fillStyle = css("--warn"); for (const p of s.pts) if (p.st === "stale" && p.v !== null && p.t >= o.t0) { g.beginPath(); g.arc(X(p.t), Y(p.v), 2.4, 0, 6.3); g.fill(); }
  }
  if (S.cursorT !== null && S.cursorT >= o.t0 && S.cursorT <= o.t1) { g.strokeStyle = "#e6edf3"; g.globalAlpha = .55; g.beginPath(); g.moveTo(X(S.cursorT), T); g.lineTo(X(S.cursorT), H - B); g.stroke(); g.globalAlpha = 1; }
  cv._map = {L, R, W, t0: o.t0, t1: o.t1};
  return {at: (t) => o.series.map((s) => { let best = null; for (const p of s.pts) if (p.v !== null && (best === null || Math.abs(p.t - t) < Math.abs(best.t - t))) best = p; return best; })};
}
function bindCursor(cv, onT) {
  cv.onmousemove = (e) => { const m = cv._map; if (!m) return; const r = cv.getBoundingClientRect(); S.cursorT = m.t0 + (m.t1 - m.t0) * (e.clientX - r.left - m.L) / (m.W - m.L - m.R); (onT || scheduleDraw)(); };
  cv.onmouseleave = () => { S.cursorT = null; (onT || scheduleDraw)(); };
}

/* ---------------------------------------------------------------- live plots */
const PLOTS = [
  {id: "pB", title: "Magnetic field B (absolute)", unit: "µT", key: "b", st: "ms", names: ["Bx", "By", "Bz"]},
  {id: "pD", title: "Baseline-subtracted change ΔB", unit: "µT (change, not force)", key: "db", st: "ms", names: ["ΔBx", "ΔBy", "ΔBz"],
   empty: "No baseline yet: capture an unloaded baseline to see the change from rest."},
  {id: "pA", title: "Acceleration (specific force)", unit: "m/s²", key: "a", st: "is", names: ["ax", "ay", "az"]},
  {id: "pG", title: "Angular velocity", unit: "rad/s", key: "g", st: "is", names: ["gx", "gy", "gz"]},
  {id: "pC", title: "Mapped command", unit: "", key: "cmd"},
];
function buildPlots() {
  $("plots").innerHTML = PLOTS.map((p) => `<div class="chart" ${p.id === "pC" ? 'style="grid-column:1/-1"' : ""}><div class="cap"><b>${p.title}</b>
    <span class="muted" id="${p.id}u"></span><span class="val" id="${p.id}v"></span></div><canvas id="${p.id}"></canvas></div>`).join("");
  PLOTS.forEach((p) => bindCursor($(p.id)));
}
function drawPlots() {
  if (!$("eng").classList.contains("on")) return;
  const win = +$("winSel").value, t1 = S.frozenT ?? S.tNow, t0 = t1 - win, filt = $("filt").checked;
  const buf = S.paused ? S.frozenBuf : S.buf;
  $("plotPod").textContent = S.pod || "—";
  for (const p of PLOTS) {
    const cv = $(p.id); let series = [], unit = p.unit, empty = p.empty;
    if (p.key === "cmd") {
      const cmds = (S.paused ? S.frozenCmds : S.cmds);
      const fk = "flex:" + S.pod;
      if (cmds.some((c) => c[fk] !== undefined)) { unit = "estimated flex 0..1 (EXPERIMENTAL mapping output)"; series = [{label: "flex", color: COL.x, pts: cmds.map((c) => ({t: c.t, v: c[fk] ?? null}))}]; }
      else if (cmds.some((c) => c.opening_cmd_m !== undefined)) { unit = "gripper opening, mm (commanded solid, measured dashed)";
        series = [{label: "commanded", color: COL.x, pts: cmds.map((c) => ({t: c.t, v: c.opening_cmd_m * 1000}))},
                  {label: "measured", color: COL.w, dash: true, pts: cmds.map((c) => ({t: c.t, v: c.opening_meas_m === undefined ? null : c.opening_meas_m * 1000}))}]; }
      else empty = S.status?.app === "teleop" ? `No mapped command uses ${S.pod}.` : "No robot in this process: no commands to show.";
    } else {
      series = [0, 1, 2].map((k) => ({label: p.names[k], color: [COL.x, COL.y, COL.z][k], pts: buf.map((s) => ({t: s.t, st: s[p.st],
        v: (s[p.key] && (s[p.st] === "fresh" || s[p.st] === "stale")) ? s[p.key][k] : null}))}));
      if (filt) series = series.concat(series.map((s) => ({...s, pts: ema(s.pts, 0.05), color: "#ffffff", alpha: .55, width: 1})));
    }
    $(p.id + "u").innerHTML = (unit ? unit + " · " : "") + series.filter((s) => !s.alpha).map((s) => `<span class="sw" style="background:${s.color}"></span>${s.label}`).join(" ");
    const c = chart(cv, {series, t0, t1, unit: unit.split(" ")[0] || "", empty, minSpan: p.key === "b" ? 20 : p.key === "db" ? 20 : p.key === "a" ? 0.5 : p.key === "g" ? 0.05 : 0.05});
    $(p.id + "v").textContent = c && S.cursorT !== null ? c.at(S.cursorT).map((q, k) => q ? `${series[k].label} ${q.v.toFixed(3)}` : "").filter(Boolean).slice(0, 3).join("  ") + ` @ ${S.cursorT.toFixed(2)} s` : "";
  }
}
$("pauseBtn").onclick = () => { S.paused = !S.paused; if (S.paused) { S.frozenBuf = S.buf.slice(); S.frozenCmds = S.cmds.slice(); S.frozenT = S.tNow; } else { S.frozenT = null; }
  $("pauseBtn").textContent = S.paused ? "Resume display" : "Pause display"; $("pauseBtn").classList.toggle("primary", S.paused); scheduleDraw(); };
$("winSel").onchange = scheduleDraw; $("filt").onchange = scheduleDraw;

function drawTimeline() {
  const el = $("timeline"); const pods = S.seen.size ? S.podsOrder.filter((p) => S.seen.has(p)) : S.podsOrder;
  if (el.dataset.k !== pods.join()) { el.dataset.k = pods.join(); el.innerHTML = pods.map((p) => `<div class="row" style="flex-wrap:nowrap;margin-bottom:6px"><div style="width:110px" class="small">${esc(p.replace("_", " "))}</div><canvas id="tl_${p}" style="flex:1;height:34px;background:var(--s2);border-radius:6px"></canvas></div>`).join(""); }
  for (const p of pods) { const cv = $("tl_" + p); if (!cv) continue; const h = S.podHist[p] || [];
    chart(cv, {series: [{color: COL.x, pts: h.map((x) => ({t: x.t, v: x.health === "invalid" || x.health === "missing" ? null : x.v, st: x.health === "degraded" ? "stale" : ""}))}], t0: S.tNow - 10, t1: S.tNow, unit: "µT", minSpan: 20, compact: true,
           empty: !S.seen.size ? "no frame received yet" : (S.pods.find((x) => x.pod === p) || {}).health === "missing" ? "missing (no pod in frames)" : "invalid / no usable value"}); }
}
function drawEvents() {
  const cv = $("evStrip"); const g = ctx(cv), W = cv.clientWidth, H = cv.clientHeight, t1 = S.tNow, t0 = t1 - 60;
  g.fillStyle = css("--dim"); g.font = "10px system-ui"; g.fillText("last 60 s", 6, H - 5);
  for (const e of S.events) { if (e.t < t0) continue; const x = 4 + (W - 8) * (e.t - t0) / 60;
    g.fillStyle = e.level === "warn" ? (e.kind.startsWith("rejected") ? css("--bad") : css("--warn")) : css("--accent"); g.fillRect(x - 1, 6, 3, H - 18); }
}

let drawQueued = false;
function scheduleDraw() { if (drawQueued) return; drawQueued = true; requestAnimationFrame(() => { drawQueued = false;
  if ($("pres").classList.contains("on")) { updateSim(); drawSchematic(); drawTimeline(); } else { drawPlots(); drawEvents(); } }); }
window.addEventListener("resize", scheduleDraw);

/* ---------------------------------------------------------------- engineering panels */
const METRICS = [
  ["received_fps", "Received", (m) => fmt(m.received_fps, 1), "/s", (m) => `window ${m.window_s} s`],
  ["accepted_fps", "Accepted", (m) => fmt(m.accepted_fps, 1), "/s", (m) => `window ${m.window_s} s`],
  ["rejected", "Rejected", (m) => m.connection.rejected, "", (m) => `connection · session ${m.session.rejected}`],
  ["missing_ids", "Missing frame IDs", (m) => m.missing_ids.connection, "", (m) => `connection · session ${m.missing_ids.session} · late-filled ${m.missing_ids.late_filled}`],
  ["duplicates", "Duplicates", (m) => m.connection.duplicates, "", (m) => `connection · session ${m.session.duplicates}`],
  ["out_of_order", "Out of order", (m) => m.connection.out_of_order, "", (m) => `connection · session ${m.session.out_of_order}`],
  ["usable_pct", "Usable channels", (m) => fmt(m.usable_pct, 1), "%", (m) => m.usable_denominator],
  ["frame_age_ms", "Newest frame age", (m) => fmt(m.frame_age_ms, 0), "ms", () => "host clock (not latency)"],
  ["interval_ms", "Receive interval", (m) => fmt(m.interval_ms, 2), "ms", (m) => `mean · window ${m.window_s} s`],
  ["jitter_ms", "Interval jitter (σ)", (m) => fmt(m.jitter_ms, 2), "ms", () => "std of host receive intervals"],
  ["processing_us", "Host processing", (m) => m.processing_us.median === null ? "n/a" : fmt(m.processing_us.median, 0), "µs", (m) => m.processing_us.median === null ? "not measured for this source" : `median · p95 ${fmt(m.processing_us.p95, 0)} µs`],
];
function renderEng() {
  const m = S.metrics, st = S.status; if (!m || !st) return;
  $("ingScope").textContent = `connection ${fmt(m.connection_age_s, 0)} s old · session totals in subtitles`;
  setHTML($("metrics"), METRICS.map(([k, l, v, u, s]) => `<div class="metric" title="${esc(S.defs[k] || "")}"><div class="l help">${l}</div><div class="v">${v(m)}<span class="small muted"> ${u}</span></div><div class="s">${esc(s(m))}</div></div>`).join("") +
    (st.control_loop ? `<div class="metric" title="Control loop timing measured in the teleop process (budget = one control period)."><div class="l help">Control tick</div><div class="v">${fmt(st.control_loop.tick_ms_median, 1)}<span class="small muted"> ms</span></div><div class="s">median · p95 ${fmt(st.control_loop.tick_ms_p95, 1)} · budget ${st.control_loop.budget_ms} ms · overruns ${st.control_loop.overruns}</div></div>` : ""));
  const rb = Object.entries(m.rejected_by).map(([k, v]) => `${k} ${v}`).join(" · ");
  $("rejNote").innerHTML = `Rejected by reason (session): ${rb || "none"}. Missing IDs are gaps in the device frame counter; they can arise anywhere upstream (sensor bus, firmware buffer, bridge), not only on the network.` +
    (m.stream_resets.length ? ` Stream resets: ${m.stream_resets.map((r) => esc(r.reason)).join(", ")}.` : "") +
    (st.compat_normalized ? ` ${st.compat_normalized} frames were normalized from the draft shape (compat).` : "");
  const he = $("handEng"); if (setHTML(he, handSVG(false))) bindPods(he);
  const ptChanged = setHTML($("podTable"), `<table><thead><tr><th>Pod</th><th>State</th><th>Mag</th><th>IMU</th><th class="num nt">|ΔB| (µT)</th><th class="num nt">age (ms)</th></tr></thead><tbody>` +
    S.pods.map((p) => `<tr class="click ${p.pod === S.pod ? "sel" : ""}" data-pod="${esc(p.pod)}"><td>${esc(p.pod)}</td><td style="color:${(HEALTH[p.health] || HEALTH.missing)[0]}">${(HEALTH[p.health] || HEALTH.missing)[1]} ${esc(p.health)}</td>
      <td title="${esc((p.mag_flags || []).join(", "))}">${esc(p.mag || "")}${p.mag_flags?.length ? " · " + esc(p.mag_flags.join(",")) : ""}${p.saturated ? ' <span class="bad">RAIL</span>' : ""}</td>
      <td>${esc(p.imu || "")}${p.imu_flags?.length ? " · " + esc(p.imu_flags.join(",")) : ""}</td><td class="num">${fmt(p.db_ut, 0)}</td><td class="num">${fmt(p.mag_age_ms, 0)}</td></tr>`).join("") + "</tbody></table>" +
    `<div class="small dim" style="margin-top:6px">age = device-clock time since that pod's last fresh magnetometer sample. TEMP_STALE affects only temperature.</div>`);
  if (ptChanged) $("podTable").querySelectorAll("tr[data-pod]").forEach((tr) => tr.onclick = () => selectPod(tr.dataset.pod));
  renderCal(); renderEvents(); renderRec(); renderCup();
}
function renderCal() {
  const st = S.status, cal = st.calibration;
  let h = "";
  if (st.app === "teleop") h += `<div class="small">Required channels: <span class="mono">${esc((cal?.required_channels || []).join(", "))}</span></div>`;
  if (cal?.progress) { const p = cal.progress; h += `<div style="margin:8px 0"><div class="bar"><i style="width:${Math.min(100, 100 * p.elapsed_s / p.duration_s)}%"></i></div>
    <div class="small muted">${fmt(p.elapsed_s, 1)} / ${p.duration_s} s · ${p.samples} usable frames (need ${p.min_samples})${Object.keys(p.skipped || {}).length ? " · skipped: " + esc(JSON.stringify(p.skipped)) : ""}</div></div>`; }
  if (cal?.active) {
    h += `<div class="small" style="margin-top:6px"><b>${esc(cal.id || "calibration")}</b> · ${cal.created_unix ? new Date(cal.created_unix * 1000).toLocaleString() : ""} · device ${esc(cal.device_id)} · config ${cal.config_id}
      · ${cal.compatible === false ? `<span class="bad">incompatible: ${esc(cal.mismatch || "")}</span>` : cal.compatible ? '<span class="ok">compatible with the stream</span>' : '<span class="muted">no frame to compare</span>'}</div>`;
    const sv = cal.stationary_variation;
    if (sv?.mag_std_ut) h += `<table style="margin-top:6px"><thead><tr><th>Pod</th><th class="num nt">σx (µT)</th><th class="num nt">σy (µT)</th><th class="num nt">σz (µT)</th><th class="num nt">gyro rms (rad/s)</th></tr></thead><tbody>` +
      Object.entries(sv.mag_std_ut).map(([p, s]) => `<tr><td>${esc(p)}</td>${s.map((x) => `<td class="num">${fmt(x, 1)}</td>`).join("")}<td class="num">${fmt(sv.gyro_rms_rads?.[p], 3)}</td></tr>`).join("") +
      `</tbody></table><div class="small dim">Stationary variation: observed while held still during capture (${cal.samples} frames, ${fmt(cal.duration_s, 1)} s). Not sensor accuracy.</div>`;
    else if (sv?.per_pod) h += `<div class="small dim">Stationary variation (std while still; not accuracy): ${esc(JSON.stringify(sv.per_pod))}</div>`;
  } else if (!cal?.progress) h += `<div class="empty">No calibration. Hold the glove still and unloaded, then capture.</div>`;
  setHTML($("calInfo"), h);
  $("calBtn").textContent = cal?.active ? "Replace baseline…" : "Capture unloaded baseline";
}
function renderEvents() {
  setHTML($("events"), S.events.slice(-120).reverse().map((e) => `<div><span class="dim">${new Date(e.wall * 1000).toLocaleTimeString()}</span>
    <span class="${e.level === "warn" ? (e.kind.startsWith("rejected") ? "bad" : "warn") : "acc"}">${e.level === "warn" ? "▲" : "●"} ${esc(e.kind)}</span> ${esc(e.text)}</div>`).join("") || '<div class="dim">no events yet</div>');
}
function renderRec() {
  const st = S.status, raw = st.recording?.raw || {}, demo = st.recording?.demo;
  $("recBtn").textContent = raw.active ? "Stop raw recording" : "Start raw telemetry recording"; $("recBtn").className = "btn " + (raw.active ? "danger" : "primary");
  const teleop = st.app === "teleop";
  $("demoBtn").disabled = !teleop; $("markBtn").disabled = !teleop || !demo;
  $("demoBtn").textContent = demo ? "Stop demonstration" : "Start demonstration"; $("demoBtn").className = "btn " + (demo ? "danger" : "primary");
  if (!teleop) $("demoRes").innerHTML = '<span class="muted">Demonstrations need a robot: run <span class="mono">python -m dermis teleop … --dashboard-port 8000</span>.</span>';
  else if (demo) $("demoRes").textContent = `recording ${demo.id}: ${demo.ticks} ticks · ${demo.markers} markers · task "${demo.task}"`;
}

/* ---------------------------------------------------------------- controls */
async function control(name, body, out) {
  try { const r = await post("/api/control/" + name, body); if (out) out.innerHTML = `<span class="ok">✓ ${esc(r.message || r.path || r.id || "done")}</span>`; toast(`${name}: ${r.message || "ok"}`, "ok"); return r; }
  catch (e) { if (out) out.innerHTML = `<span class="bad">✕ ${esc(e.message)}</span>`; toast(`${name} failed: ${e.message}`, "bad"); return null; }
}
$("calBtn").onclick = async () => {
  let r = await control("calibrate", {}, $("calRes"));
  if (r && r.needs_confirmation) { if (confirm("A calibration is active. Replace it with a new unloaded baseline? Keep the glove still and unloaded.")) r = await control("calibrate", {replace: true}, $("calRes")); else $("calRes").innerHTML = '<span class="muted">kept the existing calibration</span>'; }
};
$("recBtn").onclick = async () => { const active = S.status?.recording?.raw?.active;
  const r = await control(active ? "record_stop" : "record_start", active ? {outcome: $("recOutcome").value} : {task: $("recTask").value}, $("recRes"));
  if (r && active) $("recRes").innerHTML = `<span class="ok">✓ saved ${esc(r.path)} · ${r.frames} frames · ${r.duration_s ?? "?"} s</span>`; loadRecordings(); };
$("demoBtn").onclick = async () => { const demo = S.status?.recording?.demo;
  if (demo) { const outcome = prompt("Outcome annotation (success / failed / aborted / other):", "success"); if (outcome === null) return;
    const r = await control("demo_stop", {outcome}, $("demoRes")); if (r) { $("demoRes").innerHTML = `<span class="ok">✓ saved ${esc(r.id)} (${r.ticks} ticks)</span>`; loadEpisodes(); } }
  else await control("demo_start", {task: $("recTask").value}, $("demoRes")); };
$("markBtn").onclick = () => control("demo_mark", {text: $("markText").value || "marker"}, $("demoRes"));

/* ---------------------------------------------------------------- raw inspector */
let rawHover = false, rawPrev = {}, rawText = "";
$("rawJson").onmouseenter = () => rawHover = true; $("rawJson").onmouseleave = () => rawHover = false;
function flat(o, p = "", out = {}) { if (o && typeof o === "object") { for (const k in o) flat(o[k], p ? p + "." + k : k, out); } else out[p] = o; return out; }
function hl(o, p, prev, ind = "") {
  const pad = ind + "  ";
  if (Array.isArray(o)) { if (o.every((x) => x === null || typeof x !== "object")) return `[${o.map((x, i) => hl(x, p + "." + i, prev, pad)).join(", ")}]`;
    return "[\n" + o.map((x, i) => pad + hl(x, p + "." + i, prev, pad)).join(",\n") + "\n" + ind + "]"; }
  if (o && typeof o === "object") return "{\n" + Object.keys(o).map((k) => `${pad}<span class="k">"${esc(k)}"</span>: ${hl(o[k], p ? p + "." + k : k, prev, pad)}`).join(",\n") + "\n" + ind + "}";
  const chg = $("rawDiff").checked && p in prev && prev[p] !== o ? " chg" : "";
  if (o === null) return `<span class="z${chg}">null</span>`;
  if (typeof o === "string") return `<span class="s${chg}">"${esc(o)}"</span>`;
  if (typeof o === "boolean") return `<span class="z${chg}">${o}</span>`;
  return `<span class="n${chg}">${o}</span>`;
}
async function refreshRaw(force) {
  if (!force && (!$("rawAuto").checked || rawHover || document.hidden || !$("eng").classList.contains("on"))) return;
  try {
    const r = await api("/api/raw"); const ll = r.last_line;
    let obj = null;
    if ($("rawWhich").value === "pod") obj = (await api("/api/frame?pod=" + encodeURIComponent(S.pod || ""))).frame;
    else if (ll?.text) { try { obj = JSON.parse(ll.text); } catch (e) { obj = null; } }
    rawText = obj ? JSON.stringify(obj, null, 2) : (ll?.text || "");
    $("rawJson").innerHTML = obj ? hl(obj, "", rawPrev) : esc(ll?.text || "no frame yet");
    if (obj) rawPrev = flat(obj);
    const d = r.device || {};
    $("rawMeta").innerHTML = ll ? `verdict <b class="${ll.verdict === "accepted" ? "ok" : "bad"}">${esc(ll.verdict)}</b>${ll.reason ? " · " + esc(ll.reason) : ""} · frame ${ll.frame ?? "—"} · received ${fmt(r.age_ms, 0)} ms ago (host)` +
      (ll.processing_us !== null && ll.processing_us !== undefined ? ` · processed in ${fmt(ll.processing_us, 0)} µs` : "") +
      ` · newest accepted: frame ${d.frame ?? "—"}, t_device_us ${d.t_device_us ?? "—"}, t_host_ns ${d.t_host_ns ?? "null (no clock sync)"}` +
      (r.last_rejection ? ` · last rejection: ${esc(JSON.stringify(r.last_rejection))}` : "") + `<br><span class="dim">${esc(r.timing_note || "")}</span>` : "no line received yet";
  } catch (e) { $("rawMeta").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
}
setInterval(refreshRaw, 500);
$("rawRefresh").onclick = () => refreshRaw(true); $("rawWhich").onchange = () => { rawPrev = {}; refreshRaw(true); };
$("rawCopy").onclick = async () => { try { await navigator.clipboard.writeText(rawText); toast("frame copied", "ok"); } catch (e) {
  const ta = document.createElement("textarea"); ta.value = rawText; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); toast("frame copied", "ok"); } };

/* ---------------------------------------------------------------- recordings & episodes */
async function loadRecordings() {
  try { const rs = await api("/api/recordings");
    $("recList").innerHTML = rs.slice().reverse().map((r) => `<tr><td class="mono small" title="${esc(r.path)}">${esc(r.path.split("/").pop())}</td><td>${esc(r.task || "")}</td><td class="num">${r.frames ?? "—"}</td><td class="num">${r.duration_s ?? "—"}</td><td>${esc(r.outcome || "")}</td></tr>`).join("") || '<tr><td colspan="5" class="dim">no raw recordings yet</td></tr>';
  } catch (e) {}
}
let episodes = [], epTab = store.get("epTab", "grasp_lift_place");
const TASKS = {grasp_lift_place: "Pick & place (BC)", fragile_cup: "Fragile cup", teleop: "Teleop demos"};
const epTask = (e) => e.kind === "teleop_demonstration" ? "teleop" : (e.task || "grasp_lift_place");
async function loadEpisodes() { try { episodes = await api("/api/episodes"); renderEpisodes(); } catch (e) {} }
function renderEpisodes() {
  $("epTabs").innerHTML = Object.entries(TASKS).map(([k, l]) => `<button class="${k === epTab ? "on" : ""}" data-k="${k}">${l} (${episodes.filter((e) => epTask(e) === k).length})</button>`).join("");
  $("epTabs").querySelectorAll("button").forEach((b) => b.onclick = () => { epTab = b.dataset.k; store.set("epTab", epTab); renderEpisodes(); });
  const f = $("epFilter").value.toLowerCase();
  const rows = episodes.filter((e) => epTask(e) === epTab && (!f || (e.dataset + e.source + e.id + (e.termination || "")).toLowerCase().includes(f))).slice(-400).reverse();
  $("epCount").textContent = `${rows.length} shown`;
  $("epList").innerHTML = rows.map((e) => `<tr class="click" data-path="${esc(e.path)}"><td><span class="tag ${esc((e.label || "").toUpperCase())}">${esc(e.label || e.source)}</span></td>
    <td class="small">${esc(e.id)}</td><td class="num">${e.duration_s ?? "—"}</td><td class="${e.success ? "ok" : "bad"}">${esc(e.termination || "—")}</td>
    <td class="small ${e.bc_ready ? "ok" : "dim"}">${e.bc_ready ? "ready" : "no"}</td></tr>`).join("") || '<tr><td colspan="5" class="dim">no episodes in this category</td></tr>';
  $("epList").querySelectorAll("tr[data-path]").forEach((tr) => tr.onclick = () => showEpisode(tr.dataset.path, tr));
}
$("epFilter").oninput = renderEpisodes;
let EP = null;
async function showEpisode(path, tr) {
  document.querySelectorAll("#epList tr").forEach((x) => x.classList.remove("sel")); tr?.classList.add("sel");
  try { EP = await api("/api/episode?path=" + encodeURIComponent(path)); } catch (e) { $("epMeta").innerHTML = `<span class="bad">${esc(e.message)}</span>`; return; }
  const m = EP.meta;
  $("epLabel").innerHTML = `<span class="tag REPLAY">REPLAY</span> <span class="tag ${esc((m.label || "").toUpperCase())}">${esc(m.label || m.source)}</span>`;
  $("epMeta").innerHTML = `${esc(m.id)} · ${esc(m.task || "grasp_lift_place")}${m.seed !== undefined && m.seed !== null ? " · seed " + m.seed : ""} · ${m.length} ticks · ${esc(m.termination || "")}` +
    (m.robot ? ` · robot ${esc(m.robot)}` : "") + (m.notes ? ` · "${esc(m.notes)}"` : "") +
    ` · BC: ${EP.bc_ready ? '<span class="ok">usable for training</span>' : `<span class="warn">not training data</span>${m.bc_note ? " (" + esc(m.bc_note) + ")" : ""}`}` + (EP.decimation > 1 ? ` · shown every ${EP.decimation}th tick` : "");
  const obsN = Object.keys(EP.obs), actN = Object.keys(EP.actions);
  const pick = (names, want) => want.filter((w) => names.includes(w));
  const groups = [];
  const rel = pick(obsN, ["obj_rel_x", "obj_rel_y", "obj_rel_z"]);
  groups.push({title: rel.length ? "Object relative to gripper (observation)" : "Observations (measured)", unit: rel.length ? "m" : "",
    series: (rel.length ? rel : obsN.slice(0, 6)).map((n) => ({label: n, src: EP.obs[n]}))});
  if (actN.includes("opening") && obsN.includes("opening")) groups.push({title: "Gripper opening: issued command vs measured", unit: "m",
    series: [{label: "issued (action)", src: EP.actions.opening}, {label: "measured (obs)", src: EP.obs.opening, dash: true}]});
  else groups.push({title: "Issued commands (actions)", unit: "", series: actN.slice(0, 6).map((n) => ({label: n, src: EP.actions[n]}))});
  if (EP.force_l) groups.push({title: "Simulated finger normal force (contact)", unit: "N", hline: 30, series: [{label: "left", src: EP.force_l}, {label: "right", src: EP.force_r, dash: true}]});
  else if (Object.keys(EP.logs).length) groups.push({title: "Logged signals", unit: "", series: Object.keys(EP.logs).slice(0, 4).map((n) => ({label: n, src: EP.logs[n]}))});
  const vel = pick(actN, ["vx", "vy", "vz"]);
  if (vel.length) groups.push({title: "Issued velocity command", unit: "m/s", series: vel.map((n) => ({label: n, src: EP.actions[n]}))});
  $("epPlots").innerHTML = groups.map((g, i) => `<div class="chart"><div class="cap"><b>${esc(g.title)}</b><span class="muted">${esc(g.unit)}</span><span class="val" id="epv${i}"></span></div><canvas id="epc${i}"></canvas></div>`).join("");
  EP.groups = groups; groups.forEach((g, i) => bindCursor($("epc" + i), drawEpisode)); drawEpisode();
  $("epNotes").style.display = "block";
  $("epNotes").innerHTML = `Annotations: ${EP.annotations.map((a) => `${esc(a.kind)} @ ${a.t.toFixed(2)} s`).join(" · ") || "none"}. ${EP.annotation_notes.map(esc).join(" ")}`;
}
function drawEpisode() {
  if (!EP) return; const t = EP.t, t0 = t[0] || 0, t1 = t[t.length - 1] || 1, cols = [COL.x, COL.y, COL.z, COL.w, "#60a5fa", "#fb7185"];
  EP.groups.forEach((g, i) => { const series = g.series.map((s, k) => ({label: s.label, color: cols[k % 6], dash: s.dash, pts: t.map((tt, j) => ({t: tt, v: s.src[j]}))}));
    const c = chart($("epc" + i), {series, t0, t1: Math.max(t1, t0 + 1e-3), unit: g.unit, hline: g.hline, xlabel: `episode time 0–${(t1 - t0).toFixed(1)} s · ${g.unit}`, vlines: EP.annotations.map((a) => ({t: a.t, color: a.kind.startsWith("marker") ? "#22d3ee" : "#fbbf24"}))});
    $("epv" + i).innerHTML = series.map((s) => `<span class="sw" style="background:${s.color}"></span>${esc(s.label)}`).join(" ") +
      (c && S.cursorT !== null ? " · " + c.at(S.cursorT).map((q, k) => q ? `${series[k].label} ${q.v.toFixed(3)}` : "").filter(Boolean).slice(0, 3).join(" ") + ` @ ${S.cursorT.toFixed(2)} s` : ""); });
}

/* ---------------------------------------------------------------- training & jobs */
let runs = [], runTab = "grasp_lift_place", runSel = null;
async function loadRuns() {
  try { runs = await api("/api/runs"); } catch (e) { return; }
  $("runTabs").innerHTML = ["grasp_lift_place", "fragile_cup"].map((k) => `<button class="${k === runTab ? "on" : ""}" data-k="${k}">${TASKS[k]} (${runs.filter((r) => r.task === k).length})</button>`).join("");
  $("runTabs").querySelectorAll("button").forEach((b) => b.onclick = () => { runTab = b.dataset.k; loadRuns(); });
  const rs = runs.filter((r) => r.task === runTab);
  $("runList").innerHTML = rs.map((r) => `<tr class="click ${r.path === runSel ? "sel" : ""}" data-p="${esc(r.path)}"><td>${esc(r.name)}</td>
    <td class="${r.status.state === "done" ? "ok" : r.status.state === "failed" || r.error ? "bad" : "warn"}">${esc(r.status.state)} ${r.status.epoch ?? ""}/${r.status.epochs ?? ""}</td>
    <td class="small">${esc((r.features || []).join("+"))}${r.history ? " h" + r.history : ""}</td><td class="num">${r.n_train}+${r.n_val}</td>
    <td class="num">${r.train_seconds ? fmt(r.train_seconds, 0) + " s" : "—"}</td></tr>`).join("") || '<tr><td colspan="5" class="dim">no training runs for this task</td></tr>';
  $("runList").querySelectorAll("tr[data-p]").forEach((tr) => tr.onclick = () => { runSel = tr.dataset.p; showRun(); loadRuns(); });
  if (runSel) showRun();
  try { const jobs = await api("/api/jobs"); $("jobs").innerHTML = jobs.map((j) => `<div style="margin-top:6px"><b>${esc(j.args.join(" "))}</b>: <span class="${j.state === "done" ? "ok" : j.state === "running" ? "warn" : "bad"}">${esc(j.state)}</span> · ${j.elapsed_s} s<pre class="json" style="max-height:120px">${esc(j.log_tail.join("\n"))}</pre></div>`).join("") || '<span class="dim">no jobs started from this console</span>'; } catch (e) {}
}
function showRun() {
  const r = runs.find((x) => x.path === runSel); if (!r) return;
  const ep = r.log.map((x) => x.epoch);
  const c = chart($("lossCv"), {series: [{label: "train", color: COL.x, pts: r.log.map((x) => ({t: x.epoch, v: x.train_mse}))},
    {label: "val", color: COL.w, dash: true, pts: r.log.map((x) => ({t: x.epoch, v: x.val_mse ?? null}))}], t0: ep[0] || 0, t1: Math.max(ep[ep.length - 1] || 1, (ep[0] || 0) + 1), unit: "MSE", xlabel: `epochs ${ep[0] || 0}–${ep[ep.length - 1] || 0} · MSE`, empty: "no training log"});
  $("lossTitle").textContent = `${r.name}: loss on actions scaled to [-1, 1]`; $("lossVal").textContent = "train solid · validation dashed · x = epoch";
  $("runInfo").innerHTML = `status <b>${esc(r.status.state)}</b> · epoch ${r.status.epoch}/${r.status.epochs} · best val ${fmt(r.status.best_score, 5)} · training time ${r.train_seconds ? fmt(r.train_seconds, 0) + " s" : "unknown"}` +
    ` · data ${esc(r.provenance.data_dir)} (${esc(r.provenance.data_digest)}) · sources ${esc((r.provenance.sources || []).join(","))} · split ${r.n_train} train / ${r.n_val} val` +
    (r.error ? ` · <span class="bad">error: ${esc(r.error)}</span>` : "") + `<br><span class="dim">Low loss is not task success: see Evaluation.</span>`;
  $("infBtn").disabled = !(r.has_checkpoint && r.task === "grasp_lift_place"); $("infRes").textContent = "";
}
$("infBtn").onclick = async () => { $("infRes").textContent = "measuring…";
  try { const r = await post("/api/inference", {checkpoint: runSel}); $("infRes").innerHTML = `median <b>${r.median_ms} ms</b> · p95 ${r.p95_ms} ms over ${r.n} steps <span class="dim">(${esc(r.note)})</span>`; }
  catch (e) { $("infRes").innerHTML = `<span class="bad">${esc(e.message)}</span>`; } };
document.querySelectorAll("button[data-job]").forEach((b) => b.onclick = async () => {
  try { await fetch("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({args: JSON.parse(b.dataset.job)})}); toast("job started", "ok"); loadRuns(); }
  catch (e) { toast(e.message, "bad"); } });

/* ---------------------------------------------------------------- evaluation */
let evalData = null, evalTab = null;
async function loadEval() { try { evalData = await api("/api/evaluation"); renderEval(); } catch (e) {} }
function renderEval() {
  const keys = Object.keys(evalData.groups).sort(); if (!evalTab || !keys.includes(evalTab)) evalTab = keys[0];
  $("evalNotes").innerHTML = evalData.notes.map((n) => `<div class="note">${esc(n)}</div>`).join("");
  $("evalTabs").innerHTML = keys.map((k) => `<button class="${k === evalTab ? "on" : ""}" data-k="${esc(k)}">${esc(k)} (${evalData.groups[k].length})</button>`).join("") || '<span class="dim">no evaluation results on disk</span>';
  $("evalTabs").querySelectorAll("button").forEach((b) => b.onclick = () => { evalTab = b.dataset.k; renderEval(); });
  const rows = (evalData.groups[evalTab] || []).slice().sort((a, b) => (b.success_rate ?? 0) - (a.success_rate ?? 0));
  const cup = evalTab?.startsWith("fragile_cup");
  $("evalTable").innerHTML = rows.length ? `<div class="scroll" style="max-height:340px"><table><thead><tr><th>Controller / policy</th><th>Mode</th><th class="num">Success</th><th style="width:110px"></th>
    <th class="num">${cup ? "Drops" : "Drop eps"}</th><th class="num" title="${cup ? "damage terminations (simulated load threshold)" : "episodes above the simulated excess-force threshold"}">${cup ? "Excess load" : "Excess-force eps"}</th>
    <th class="num">Timeouts</th><th class="num">${cup ? "" : "Median s (successes)"}</th><th class="num">Peak N (median)</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td title="${esc(r.path)}">${esc(r.name)}${r.features ? ` <span class="dim small">${esc(r.features.join("+"))}</span>` : ""}${r.demos ? ` <span class="dim small">${r.demos} demos</span>` : ""}</td>
      <td><span class="tag ${esc(r.label)}">${esc(r.label)}</span></td><td class="num">${r.successes}/${r.attempts} · ${fmt(100 * r.success_rate, 0)}%</td>
      <td><div class="bar"><i style="width:${100 * (r.success_rate || 0)}%"></i></div></td><td class="num">${r.drops ?? "—"}</td><td class="num">${r.excess_load ?? "—"}</td>
      <td class="num">${r.timeouts ?? "—"}</td><td class="num">${cup ? "" : fmt(r.completion_s_median, 2)}</td><td class="num">${fmt(r.peak_force_n_median, 1)}</td></tr>`).join("") + "</tbody></table></div>" +
    `<div class="small dim">Same task and suite in this table${cup ? "; fragile-cup forces and thresholds are simulated, the cup is rigid (no bending)" : ""}. ${rows[0]?.threshold_n ? `Excess-force threshold ${rows[0].threshold_n} N (simulated benchmark threshold).` : ""}</div>` : "";
  const v = evalData.variation.filter((x) => x.group === evalTab);
  $("variation").innerHTML = v.length ? `<table><thead><tr><th>Family</th><th class="num">Runs</th><th>Success per seed</th><th class="num">Min</th><th class="num">Mean</th><th class="num">Max</th></tr></thead><tbody>` +
    v.map((x) => `<tr><td>${esc(x.family)}</td><td class="num">${x.runs}</td><td class="mono small">${x.success_rates.map((s) => fmt(100 * s, 0) + "%").join(" · ")}</td><td class="num">${fmt(100 * x.min, 0)}%</td><td class="num">${fmt(100 * x.mean, 0)}%</td><td class="num">${fmt(100 * x.max, 0)}%</td></tr>`).join("") +
    `</tbody></table><div class="small dim">Spread across training seeds only; no confidence intervals or significance tests are computed.</div>` : '<div class="dim small">No seed replicates in this group.</div>';
}

/* ---------------------------------------------------------------- fragile cup */
function renderCup() {
  const c = S.cup;
  if (!c) { $("cupKind").textContent = "";
    $("cupLive").innerHTML = `<div class="dim">No live fragile-cup session in this process. Results (if any) are under Evaluation → fragile_cup.
      To run one: <span class="mono">python -m dermis dashboard --cup-controller grip_load_expert,fixed_14N</span> (SCRIPTED baselines) or <span class="mono">--cup-checkpoint runs/cup/…</span> (trained).</div>`; return; }
  if (!c.running) { $("cupLive").innerHTML = `<span class="${c.error ? "bad" : "muted"}">${esc(c.error || "starting…")}</span>`; return; }
  const s = c.snapshot; if (!s) return;
  $("cupKind").innerHTML = `<span class="tag AUTONOMOUS">AUTONOMOUS</span> ${esc(c.kinds[s.active_policy] || "")}`;
  const thr = s.break_force_n;
  const fb = (v, i) => `<div class="small">${i ? "right" : "left"} finger normal (simulated) <b>${fmt(v, 2)} N</b></div><div class="bar" style="margin-bottom:6px"><i style="width:${Math.min(100, 100 * v / (thr * 1.25))}%;background:${v > thr ? "var(--bad)" : "var(--teal)"}"></i><span class="thr" style="left:${100 / 1.25}%"></span></div>`;
  if (!$("cupCtl")) {
    $("cupLive").innerHTML = `<div class="grid g3" style="grid-template-columns:repeat(3,minmax(0,1fr))"><div id="cupVals"></div><div id="cupForces"></div><div>
      <div class="row" id="cupCtl"><button class="btn primary" data-a="pour">Pour</button><button class="btn" data-a="reset">Reset (same scenario)</button><button class="btn" data-a="next">Next scenario</button></div>
      <div class="row" style="margin-top:8px"><select id="cupPol"></select><button class="btn" id="cupSel">Use controller</button><button class="btn" id="cupCmp">Paired comparison</button></div>
      <div class="result" id="cupRes"></div><div id="cupCmpRes"></div></div></div>
      <div class="note">${esc(c.labels.applied_load_n)}. ${esc(c.labels.break_force_n)}. Dashboard controls change the environment only (Pour / Reset / Next / controller choice); in AUTONOMOUS mode no human motor command reaches the robot.</div>`;
    $("cupCtl").querySelectorAll("button").forEach((b) => b.onclick = () => control("cup_" + b.dataset.a, {}, $("cupRes")));
    $("cupSel").onclick = () => control("cup_select", {policy: $("cupPol").value}, $("cupRes"));
    $("cupCmp").onclick = async () => { $("cupCmpRes").textContent = "running every controller on this scenario…"; const r = await control("cup_compare", {}, $("cupRes"));
      if (r) $("cupCmpRes").innerHTML = `<table><thead><tr><th>Controller</th><th>Outcome</th><th class="num">Peak N</th><th class="num">Slip mm</th><th class="num">Mean effort N</th></tr></thead><tbody>${r.results.map((x) =>
        `<tr><td>${esc(x.policy)}</td><td class="${x.termination === "success" ? "ok" : "bad"}">${esc(x.termination)}</td><td class="num">${x.peak_normal_n}</td><td class="num">${x.final_slip_mm}</td><td class="num">${x.mean_effort_n}</td></tr>`).join("")}</tbody></table>`; };
  }
  const pol = $("cupPol"); if (pol.dataset.k !== s.policies.join()) { pol.dataset.k = s.policies.join(); pol.innerHTML = s.policies.map((p) => `<option>${esc(p)}</option>`).join(""); }
  $("cupVals").innerHTML = `<div class="small muted">Outcome</div><div class="outcome ${s.outcome === "success" ? "ok" : s.outcome ? "bad" : "acc"}">${esc(s.outcome ? s.outcome.replace("_", " ") : s.phase)}</div>
    <div class="small">controller <b>${esc(s.active_policy)}</b> · seed ${s.seed} · t ${fmt(s.t_s, 1)} s · ${s.pouring ? '<span class="warn">pouring</span>' : "not pouring"}</div>
    <div class="small">load proxy (simulated) <b>${fmt(s.applied_load_n, 2)} N</b> · commanded grip effort ${fmt(s.effort_cmd_n, 2)} N</div>
    <div class="small">cup displacement relative to hand <b>${fmt(s.cup_slip_mm, 1)} mm</b> · damage model ${esc(s.damage_model)}</div>`;
  $("cupForces").innerHTML = s.finger_normal_n.map(fb).join("") + `<div class="small dim">red line = simulated excessive-load threshold ${thr} N per finger</div>`;
}

/* ---------------------------------------------------------------- boot */
async function boot() {
  try { const c = await api("/api/console"); S.defs = c.definitions; S.kind = c.kind; S.status = c.status; S.podsOrder = c.pods; buildPodSelect(); renderStatus(); }
  catch (e) { $("bLink").textContent = "console API unreachable"; }
  document.querySelectorAll("[data-def]").forEach((el) => el.title = S.defs[el.dataset.def] || "");
  buildPlots(); setView(store.get("view", "pres")); connect();
  loadRecordings(); loadEpisodes(); loadRuns(); loadEval();
  setInterval(loadRuns, 5000); setInterval(loadRecordings, 6000); setInterval(() => { loadEpisodes(); loadEval(); }, 20000);
}
boot();
