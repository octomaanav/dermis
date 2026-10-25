"use strict";

const M = window.DermisMath;
const PATCHES = ["palm", "middle", "thumb", "index"];
const PATCH_LABEL = {palm: "Palm", middle: "Middle", thumb: "Thumb", index: "Index"};
const COLORS = {palm: "#f4a261", middle: "#a78bfa", thumb: "#36c5d7", index: "#60a5fa"};
const DEFAULT_ASSIGN = {0: "palm", 1: "middle", 2: "thumb", 3: "index"};
const DEFAULT_AREA = {palm: 2.2, middle: 1.0, thumb: 1.1, index: 1.0};
const DEFAULT_REFERENCE = {0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0};
const $ = (id) => document.getElementById(id);

function stored(name, fallback) {
  try { return {...fallback, ...JSON.parse(localStorage.getItem(name) || "{}")}; }
  catch (_) { return {...fallback}; }
}

const S = {
  socket: null, wanted: false, retry: null, retryDelay: 1000,
  assignments: stored("dermis.assignments.v1", DEFAULT_ASSIGN),
  areas: stored("dermis.areas.v1", DEFAULT_AREA),
  references: stored("dermis.references.v1", DEFAULT_REFERENCE),
  filters: {}, patchState: {}, imuFilters: null, imuState: null,
  baseline: {}, calibration: null, history: [], rawLog: [], derivedLog: [], recording: true,
  frames: 0, drops: 0, lastFrame: null, intervals: [], lastReceiveMs: null,
  firstDeviceUs: null, lastDeviceUs: null, t: 0, sourceNames: {}, alertKeys: {},
};

function fmt(v, digits = 2, fallback = "—") {
  return Number.isFinite(v) ? Number(v).toFixed(digits) : fallback;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
}
function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}
function saveConfig() {
  localStorage.setItem("dermis.assignments.v1", JSON.stringify(S.assignments));
  localStorage.setItem("dermis.areas.v1", JSON.stringify(S.areas));
  localStorage.setItem("dermis.references.v1", JSON.stringify(S.references));
}
function channelForPatch(patch) {
  const pair = Object.entries(S.assignments).find(([, p]) => p === patch);
  return pair ? Number(pair[0]) : null;
}
function sourceName(ch) { return `S${Number(ch) + 1}`; }

function makeFilters() {
  return {b: new M.MedianEma(5, .10), force: new M.MedianEma(5, .12), temp: new M.MedianEma(5, 1.5)};
}
function resetSignalState() {
  S.filters = {};
  S.patchState = Object.fromEntries(PATCHES.map((p) => [p, {contact: false, contactAge: 0,
    impulse: 0, slip: 0, slipEvidence: 0, slipActive: false, slipOnCount: 0, slipOffCount: 0}]));
  S.imuFilters = {acc: new M.MedianEma(3, .04), gyr: new M.MedianEma(3, .035)};
  S.imuState = {roll: 0, pitch: 0, yaw: 0, lastT: null};
  S.baseline = {};
  S.calibration = null;
}
resetSignalState();

function toast(text) {
  const el = $("toast"); el.textContent = text; el.style.display = "block";
  clearTimeout(el._timer); el._timer = setTimeout(() => el.style.display = "none", 3200);
}
function alert(kind, text, key = text) {
  const now = Date.now();
  if ((S.alertKeys[key] || 0) + 5000 > now) return;
  S.alertKeys[key] = now;
  const host = $("alerts");
  host.querySelector(".quiet")?.remove();
  const row = document.createElement("div"); row.className = kind; row.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  host.prepend(row); while (host.children.length > 30) host.lastChild.remove();
}

function setConnection(mode, text) {
  $("statusDot").className = mode;
  $("statusText").textContent = text;
  $("connectBtn").textContent = S.wanted ? "Disconnect" : "Connect";
}
function disconnect(manual = true) {
  if (manual) S.wanted = false;
  clearTimeout(S.retry); S.retry = null;
  if (S.socket) { const ws = S.socket; S.socket = null; ws.onclose = null; ws.close(); }
  setConnection("", manual ? "Offline" : "Reconnecting");
}
function connect() {
  if (S.wanted && S.socket) { disconnect(true); return; }
  S.wanted = true; clearTimeout(S.retry);
  const url = $("wsUrl").value.trim(); localStorage.setItem("dermis.wsUrl", url);
  setConnection("", "Connecting");
  let ws;
  try { ws = new WebSocket(url); } catch (e) { onSocketFailure(e.message); return; }
  S.socket = ws;
  ws.onopen = () => {
    if (S.socket !== ws) return;
    S.retryDelay = 1000; setConnection("live", "Live");
    resetSignalState(); beginCalibration(true); toast("Connected. Keep the glove still and every patch unloaded for 5 seconds.");
  };
  ws.onmessage = async (event) => {
    try {
      const text = typeof event.data === "string" ? event.data : await event.data.text();
      const msg = JSON.parse(text);
      if (msg && msg.type === "calibration") { onFirmwareCalibration(msg); return; }
      processFrame(msg, text);
    } catch (e) { alert("bad", `Rejected WebSocket message: ${e.message}`, "parse"); }
  };
  ws.onerror = () => {};
  ws.onclose = () => { if (S.socket === ws) S.socket = null; if (S.wanted) onSocketFailure("glove disconnected"); else setConnection("", "Offline"); };
}
function onSocketFailure(message) {
  setConnection("bad", "Retrying"); alert("warn", `${message}; reconnecting`, "socket");
  if (!S.wanted) return;
  const delay = S.retryDelay; S.retryDelay = Math.min(8000, S.retryDelay * 1.7);
  S.retry = setTimeout(connect, delay);
}

function beginCalibration(auto = false) {
  S.calibration = {startT: null, duration: 5.0, auto, samples: {}, imu: []};
  $("calState").textContent = "firmware tare + host zero · still";
  if (S.socket?.readyState === WebSocket.OPEN) S.socket.send(JSON.stringify({cmd: "tare"}));
}
function collectCalibration(ch, b, force, temp, flags, t) {
  const c = S.calibration; if (!c) return;
  const x = c.samples[ch] ||= {b: [], force: [], temp: [], fieldRates: [], forceRates: [],
    high: 0, total: 0, prevB: null, prevForce: null, prevT: null};
  if (b?.every(Number.isFinite)) x.b.push(b.slice());
  if (Number.isFinite(force)) x.force.push(force);
  if (Number.isFinite(temp)) x.temp.push(temp);
  x.total++; if ((flags || []).includes("FORCE_CLAMPED_HIGH")) x.high++;
  if (x.prevT !== null) {
    const dt = t - x.prevT;
    if (dt > .01 && dt < .5) {
      if (b?.every(Number.isFinite) && x.prevB) x.fieldRates.push(M.norm(M.sub(b, x.prevB)) / dt);
      if (Number.isFinite(force) && Number.isFinite(x.prevForce)) x.forceRates.push(Math.abs(force - x.prevForce) / dt);
    }
  }
  x.prevB = b?.slice() || null; x.prevForce = force; x.prevT = t;
}
function finishCalibration() {
  const c = S.calibration; if (!c) return;
  const base = {};
  for (const [ch, x] of Object.entries(c.samples)) {
    if (x.b.length < 12) continue;
    const b = M.vectorMedian(x.b);
    const force = M.median(x.force) || 0;
    const highFraction = x.total ? x.high / x.total : 1;
    const fieldRateCenter = M.median(x.fieldRates) || 0;
    const forceRateCenter = M.median(x.forceRates) || 0;
    base[ch] = {b, noiseUt: Math.max(1, M.vectorMadNorm(x.b, b)), force,
      temp: M.median(x.temp), highFraction,
      forceUsable: x.force.length >= 12 && highFraction < .2 && force < 9.7,
      fieldRateFloorUtS: Math.max(40, fieldRateCenter + 6 * (M.mad(x.fieldRates, fieldRateCenter) || 0)),
      forceRateFloorNS: Math.max(.2, forceRateCenter + 6 * (M.mad(x.forceRates, forceRateCenter) || 0))};
  }
  if (c.imu.length >= 12) {
    const acc = M.vectorMedian(c.imu.map((v) => v.acc));
    const gyr = M.vectorMedian(c.imu.map((v) => v.gyr));
    base.imu = {acc, gyr, roll: Math.atan2(acc[1], acc[2]), pitch: Math.atan2(-acc[0], Math.hypot(acc[1], acc[2]))};
  }
  S.baseline = base; S.calibration = null;
  const n = Object.keys(base).filter((x) => x !== "imu").length;
  const forceN = Object.entries(base).filter(([ch, x]) => ch !== "imu" && x.forceUsable).length;
  $("calState").textContent = n === 4 ? `zeroed · force ${forceN}/4 valid` : `partial calibration · ${n}/4`;
  if (n < 4) alert("bad", `Calibration captured only ${n}/4 valid MLX channels.`, "cal-partial");
  for (const [ch, b] of Object.entries(base)) {
    if (ch !== "imu" && !b.forceUsable) alert("warn", `${sourceName(ch)} needs a known-load span; force/contact stays disabled until it is calibrated.`, `force-base-${ch}`);
  }
  if (base.imu) Object.assign(S.imuState, {roll: 0, pitch: 0, yaw: 0, lastT: null});
  toast("Robust median/MAD zero and gyro bias captured.");
}

function frameTime(frame) {
  const us = Number(frame.t_device_us);
  if (!Number.isFinite(us)) return performance.now() / 1000;
  if (S.firstDeviceUs === null || (S.lastDeviceUs !== null && us < S.lastDeviceUs)) S.firstDeviceUs = us;
  S.lastDeviceUs = us;
  return (us - S.firstDeviceUs) / 1e6;
}
function findImu(frame) {
  if (frame.imu && typeof frame.imu === "object") return frame.imu;
  for (const pod of Object.values(frame.pods || {})) if (pod?.imu) return pod.imu;
  return null;
}
function updateImu(frame, t) {
  const imu = findImu(frame);
  if (!imu || imu.valid === false || !Array.isArray(imu.acc_mps2) || !Array.isArray(imu.gyr_rads)) return null;
  const acc = S.imuFilters.acc.update(imu.acc_mps2, t), gyrRaw = S.imuFilters.gyr.update(imu.gyr_rads, t);
  if (!acc || !gyrRaw) return null;
  if (S.calibration) S.calibration.imu.push({acc: imu.acc_mps2.slice(), gyr: imu.gyr_rads.slice()});
  const bias = S.baseline.imu?.gyr || [0, 0, 0];
  const gyr = gyrRaw.map((v, i) => v - bias[i]);
  const o = S.imuState, dt = o.lastT === null ? 0 : M.clamp(t - o.lastT, 0, .2); o.lastT = t;
  const ar = Math.atan2(acc[1], acc[2]), ap = Math.atan2(-acc[0], Math.hypot(acc[1], acc[2]));
  const br = S.baseline.imu?.roll || ar, bp = S.baseline.imu?.pitch || ap;
  if (dt > 0) {
    o.roll += gyr[0] * dt; o.pitch += gyr[1] * dt; o.yaw += gyr[2] * dt;
    const correction = 1 - Math.exp(-dt / .55);
    o.roll += correction * ((ar - br) - o.roll); o.pitch += correction * ((ap - bp) - o.pitch);
  }
  const accMag = M.norm(acc), gyroMag = M.norm(gyr), linear = Math.abs(accMag - 9.80665);
  return {acc, gyr, rpy: [o.roll, o.pitch, o.yaw].map((v) => v * 180 / Math.PI), accMag, gyroMag,
    linear, motion: M.clamp(100 * (linear / 5 + gyroMag / 4), 0, 100), flags: imu.flags || []};
}

function processFrame(frame, originalText = null) {
  if (!frame || typeof frame !== "object" || !frame.pods) throw new Error("frame has no pods object");
  const now = performance.now(), t = frameTime(frame); S.t = t; S.frames++;
  if (S.lastReceiveMs !== null) { S.intervals.push(now - S.lastReceiveMs); if (S.intervals.length > 120) S.intervals.shift(); }
  S.lastReceiveMs = now;
  if (Number.isInteger(frame.frame) && S.lastFrame !== null) {
    const delta = (frame.frame - S.lastFrame) >>> 0;
    if (delta > 1 && delta < 0x80000000) { S.drops += delta - 1; alert("warn", `Frame gap: ${delta - 1} ID(s) missing`, "gap"); }
  }
  if (Number.isInteger(frame.frame)) S.lastFrame = frame.frame;
  if (S.calibration && S.calibration.startT === null) S.calibration.startT = t;

  const samples = {};
  for (const [location, pod] of Object.entries(frame.pods)) {
    const mag = pod?.mag; if (!mag || !Number.isInteger(mag.ch)) continue;
    samples[mag.ch] = mag; S.sourceNames[mag.ch] = location;
  }
  const patches = {};
  for (const patch of PATCHES) {
    const ch = channelForPatch(patch), mag = samples[ch], prior = S.patchState[patch];
    if (!mag) { patches[patch] = {...prior, patch, ch, valid: false, missing: true}; continue; }
    const valid = mag.valid !== false && Array.isArray(mag.b_ut) && mag.b_ut.every(Number.isFinite);
    const forceValid = mag.force_valid !== false && Number.isFinite(mag.force_n);
    const flags = Array.isArray(mag.flags) ? mag.flags : [];
    if (!valid) { patches[patch] = {...prior, patch, ch, valid: false, missing: false, flags}; continue; }
    const filters = S.filters[ch] ||= makeFilters();
    const b = filters.b.update(mag.b_ut, t),
      temp = filters.temp.update(Number.isFinite(mag.temp_c) ? mag.temp_c : null, t);
    const forceRaw = forceValid ? filters.force.update(Number(mag.force_n), t) : null;
    collectCalibration(ch, b, forceRaw, temp, flags, t);
    const base = S.baseline[ch], db = base ? M.norm(M.sub(b, base.b)) : 0;
    const calibrated = flags.includes("FORCE_CAL_READY");
    const forceReliable = Boolean(base && forceValid && (calibrated || base.forceUsable));
    const force = forceReliable && Number.isFinite(forceRaw) ? Math.max(0, forceRaw - base.force) : null;
    const dt = Number.isFinite(prior.t) ? M.clamp(t - prior.t, .001, .5) : .1;
    const fieldRate = prior.b ? M.norm(M.sub(b, prior.b)) / dt : 0;
    const forceRate = Number.isFinite(force) && Number.isFinite(prior.force) ? (force - prior.force) / dt : 0;
    let contact = forceReliable ? prior.contact : false;
    if (!contact && force >= .25) contact = true;
    else if (contact && force < .12) contact = false;
    const contactAge = contact ? (prior.contact ? prior.contactAge + dt : 0) : 0;

    // Reject compression along the current field-change direction. Relative tangential motion is
    // more slip-like than a simple increase/decrease in normal patch compression.
    let transverseRate = 0;
    if (base && prior.b) {
      const displacement = M.sub(b, base.b), speed = b.map((v, i) => (v - prior.b[i]) / dt);
      const length = M.norm(displacement);
      if (length > Math.max(50, 3 * base.noiseUt)) {
        const axis = displacement.map((v) => v / length);
        const axial = speed.reduce((sum, v, i) => sum + v * axis[i], 0);
        transverseRate = M.norm(speed.map((v, i) => v - axial * axis[i]));
      }
    }
    const slipInstant = M.slipScore({contact: contact && contactAge >= .35,
      transverseRateUtS: transverseRate, forceRateNS: forceRate,
      fieldRateFloorUtS: base?.fieldRateFloorUtS || 40,
      forceRateFloorNS: base?.forceRateFloorNS || .2});
    const evidenceAlpha = 1 - Math.exp(-dt / .35);
    const slipEvidence = prior.slipEvidence + evidenceAlpha * (slipInstant - prior.slipEvidence);
    let slipOnCount = slipEvidence >= 42 ? prior.slipOnCount + 1 : 0;
    let slipOffCount = slipEvidence <= 18 ? prior.slipOffCount + 1 : 0;
    let slipActive = prior.slipActive;
    if (!contact) { slipActive = false; slipOnCount = 0; slipOffCount = 0; }
    else if (!slipActive && slipOnCount >= 3) slipActive = true;
    else if (slipActive && slipOffCount >= 5) slipActive = false;
    const slip = slipActive ? slipEvidence : 0;
    const impulse = prior.impulse + (contact && Number.isFinite(force) ? force * dt : 0);
    const pressure = Number.isFinite(force) ? force / Math.max(.1, S.areas[patch]) * 10 : null;
    patches[patch] = {patch, ch, valid: true, missing: false, flags, b, db, loadSignal: Number.isFinite(mag.load_signal_ut) ? mag.load_signal_ut : db, noiseUt: base?.noiseUt,
      forceRaw, force, forceReliable, forceRate, pressure, temp,
      tempDelta: base && Number.isFinite(temp) ? temp - base.temp : 0,
      fieldRate, transverseRate, contact, contactAge, slip, slipEvidence, slipActive, slipOnCount,
      slipOffCount, impulse, t, clampedHigh: flags.includes("FORCE_CLAMPED_HIGH"),
      clampedLow: flags.includes("FORCE_CLAMPED_LOW")};
    if (patches[patch].clampedHigh) alert("warn", `${sourceName(ch)} / ${PATCH_LABEL[patch]} force is clamped high; shade cannot show additional load.`, `clamp-hi-${ch}`);
  }
  S.patchState = patches;
  const imu = updateImu(frame, t); if (imu) S.imuState.latest = imu;
  if (S.calibration && t - S.calibration.startT >= S.calibration.duration) finishCalibration();
  const snap = {t, patches: Object.fromEntries(PATCHES.map((p) => [p, {...patches[p]}])), imu: imu || S.imuState.latest || null};
  S.history.push(snap); while (S.history.length && S.history[0].t < t - 45) S.history.shift();
  if (S.recording) {
    const raw = originalText || JSON.stringify(frame); S.rawLog.push(raw); if (S.rawLog.length > 100000) S.rawLog.shift();
    S.derivedLog.push(derivedRow(frame, snap)); if (S.derivedLog.length > 100000) S.derivedLog.shift();
  }
  render(frame, snap);
  window.Ouch?.feed(snap);
}

function derivedRow(frame, snap) {
  const row = {frame: frame.frame, t_device_us: frame.t_device_us};
  for (const p of PATCHES) {
    const x = snap.patches[p] || {};
    Object.assign(row, {[`${p}_source`]: x.ch === null ? "" : sourceName(x.ch), [`${p}_force_n`]: x.force,
      [`${p}_pressure_kpa`]: x.pressure, [`${p}_temp_c`]: x.temp, [`${p}_temp_delta_c`]: x.tempDelta,
      [`${p}_slip_score`]: x.slip, [`${p}_contact`]: x.contact ? 1 : 0, [`${p}_field_delta_ut`]: x.db,
      [`${p}_load_signal_ut`]: x.loadSignal});
  }
  const imu = snap.imu || {}; ["roll_deg", "pitch_deg", "yaw_deg"].forEach((k, i) => row[k] = imu.rpy?.[i]);
  row.motion_score = imu.motion; return row;
}

function render(frame, snap) {
  const valid = PATCHES.filter((p) => snap.patches[p]?.valid).length;
  const active = PATCHES.filter((p) => snap.patches[p]?.contact).length;
  const forces = PATCHES.map((p) => snap.patches[p]?.force).filter(Number.isFinite);
  const slips = PATCHES.map((p) => snap.patches[p]?.slip).filter(Number.isFinite);
  $("totalForce").textContent = forces.length ? `${fmt(forces.reduce((a, b) => a + b, 0), 2)}` : "—";
  $("contacts").textContent = `${active} / 4`; $("peakSlip").textContent = slips.length ? `${fmt(Math.max(...slips), 0)}%` : "—";
  $("quality").textContent = valid === 4 ? "GOOD" : valid ? "DEGRADED" : "NO DATA";
  $("quality").style.color = valid === 4 ? "var(--green)" : "var(--amber)";
  $("qualitySub").textContent = `${valid}/4 valid · ${S.drops} missing IDs`;
  if (!S.calibration) {
    const ready = PATCHES.filter((p) => snap.patches[p]?.flags?.includes("FORCE_CAL_READY")).length;
    const spanning = PATCHES.some((p) => snap.patches[p]?.flags?.includes("FORCE_SPAN_CALIBRATING"));
    $("calState").textContent = spanning ? "capturing loaded span · hold steady" : `zeroed · force ${ready}/4 calibrated`;
  }
  renderHand(snap.patches); renderPatchCards(snap.patches); renderImu(snap.imu); renderSession(); drawCharts();
}
function renderHand(patches) {
  for (const patch of PATCHES) {
    const x = patches[patch] || {}, g = document.querySelector(`.patch[data-patch="${patch}"]`),
      shownCh = Number.isInteger(x.ch) ? x.ch : channelForPatch(patch);
    const core = g.querySelector(".force-core"), halo = g.querySelector(".force-halo"), ring = g.querySelector(".temp-ring");
    const color = M.forceColor(x.force, 10); core.style.fill = color; core.style.stroke = x.contact ? "#ff7885" : "#738096";
    halo.style.opacity = x.contact ? String(.22 + .6 * M.clamp((x.force || 0) / 10, 0, 1)) : "0";
    ring.style.stroke = M.temperatureColor(x.tempDelta); g.querySelector(".reading").textContent = Number.isFinite(x.force) ? `${fmt(x.force, 2)} N` : "— N";
    g.querySelector(".source").textContent = Number.isInteger(shownCh) ? sourceName(shownCh) : "UNASSIGNED";
  }
}
function renderPatchCards(patches) {
  $("patchCards").innerHTML = PATCHES.map((p) => {
    const x = patches[p] || {}, warnings = [...(x.flags || [])],
      shownCh = Number.isInteger(x.ch) ? x.ch : channelForPatch(p);
    if (x.clampedHigh) warnings.unshift("FORCE CEILING");
    if (x.valid && !x.forceReliable && !warnings.includes("FORCE_SPAN_CALIBRATING")) warnings.unshift("FORCE CAL REQUIRED · CONTACT DISABLED");
    return `<div class="patch-card ${x.contact ? "active" : ""}"><div class="head"><b>${PATCH_LABEL[p]} · ${Number.isInteger(shownCh) ? sourceName(shownCh) : "—"}</b><i></i></div>
      <div class="metrics-row"><div><span>Force</span><b>${fmt(x.force,2)} N</b></div><div><span>Pressure*</span><b>${fmt(x.pressure,1)} kPa</b></div>
      <div><span>Temperature</span><b>${fmt(x.temp,1)}°C</b></div><div><span>Δ temperature</span><b>${fmt(x.tempDelta,1)}°C</b></div>
      <div><span>Slip risk*</span><b>${fmt(x.slip,0)}%</b></div><div><span>Loading rate</span><b>${fmt(x.forceRate,1)} N/s</b></div>
      <div><span>Load signal</span><b>${fmt(x.loadSignal,0)} µT</b></div><div><span>Impulse</span><b>${fmt(x.impulse,2)} N·s</b></div></div>
      <div class="flagline ${warnings.length ? "warn" : ""}">${x.valid ? `JSON force ${fmt(x.forceRaw,2)} N · ${warnings.length ? esc(warnings.join(" · ")) : "VALID · FILTERED"}` : (x.missing ? "MISSING" : "INVALID")}</div></div>`;
  }).join("");
}
function vectorText(v, n = 2) { return v?.every(Number.isFinite) ? v.map((x) => (x >= 0 ? "+" : "") + x.toFixed(n)).join(" / ") : "—"; }
function renderImu(imu) {
  $("imuQuality").textContent = imu ? "LIVE" : "NO DATA"; $("imuQuality").className = `badge ${imu ? "live" : ""}`;
  $("rpy").textContent = vectorText(imu?.rpy, 1); $("acc").textContent = vectorText(imu?.acc, 2); $("gyro").textContent = vectorText(imu?.gyr, 3);
  $("motion").textContent = imu ? `${fmt(imu.motion,0)}%` : "—";
}
function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : null; }
function std(a) { const m = mean(a); return m === null ? null : Math.sqrt(mean(a.map((x) => (x - m) ** 2))); }
function renderSession() {
  const avg = mean(S.intervals), jitter = std(S.intervals);
  $("rateText").textContent = avg ? `${(1000 / avg).toFixed(1)} Hz` : "0.0 Hz";
  $("frameCount").textContent = S.frames.toLocaleString(); $("dropCount").textContent = S.drops.toLocaleString();
  $("jitter").textContent = jitter === null ? "—" : `${jitter.toFixed(1)} ms`;
  $("logSize").textContent = bytes(S.rawLog.reduce((n, x) => n + x.length + 1, 0));
  for (let ch = 0; ch < 4; ch++) {
    const el = $(`source_${ch}`); if (el) el.textContent = `JSON name: ${S.sourceNames[ch] || "—"}`;
  }
  if (jitter > 35) alert("warn", `Receive jitter is high (${jitter.toFixed(1)} ms σ).`, "jitter-high");
}

function setupCards() {
  $("assigner").innerHTML = [0, 1, 2, 3].map((ch) => `<div class="assign"><b>${sourceName(ch)} <small class="muted">channel ${ch}</small></b>
    <label>Physical patch<select data-ch="${ch}">${PATCHES.map((p) => `<option value="${p}" ${S.assignments[ch] === p ? "selected" : ""}>${PATCH_LABEL[p]}</option>`).join("")}</select></label>
    <label>Patch area (cm²)<input data-area="${ch}" type="number" min="0.1" max="20" step="0.1" value="${S.areas[S.assignments[ch]] || 1}"></label>
    <label>Known calibration load (N)<input data-reference="${ch}" type="number" min="0.1" max="9.80665" step="0.1" value="${S.references[ch] || 5}"></label>
    <button data-span="${ch}">Capture loaded span · 3 s</button>
    <small id="source_${ch}">JSON name: —</small></div>`).join("");
  document.querySelectorAll("select[data-ch]").forEach((el) => el.onchange = () => {
    const ch = Number(el.dataset.ch), next = el.value, old = S.assignments[ch];
    const other = Object.keys(S.assignments).find((k) => Number(k) !== ch && S.assignments[k] === next);
    S.assignments[ch] = next; if (other !== undefined) S.assignments[other] = old;
    saveConfig(); resetSignalState(); setupCards(); beginCalibration(false); toast("Assignment changed; recalibrating zero.");
  });
  document.querySelectorAll("input[data-area]").forEach((el) => el.onchange = () => {
    const patch = S.assignments[Number(el.dataset.area)]; S.areas[patch] = M.clamp(Number(el.value) || 1, .1, 20); saveConfig();
  });
  document.querySelectorAll("input[data-reference]").forEach((el) => el.onchange = () => {
    const ch = Number(el.dataset.reference); S.references[ch] = M.clamp(Number(el.value) || 5, .1, 9.80665); el.value = S.references[ch]; saveConfig();
  });
  document.querySelectorAll("button[data-span]").forEach((el) => el.onclick = () => {
    const ch = Number(el.dataset.span), reference = M.clamp(Number(document.querySelector(`input[data-reference="${ch}"]`).value) || 5, .1, 9.80665);
    if (S.socket?.readyState !== WebSocket.OPEN) return toast("Connect to the glove before capturing a force span.");
    if (!S.baseline[ch]) return toast("Capture the unloaded zero first.");
    S.references[ch] = reference; saveConfig();
    S.filters[ch] && (S.filters[ch].force = new M.MedianEma(5, .12));
    S.socket.send(JSON.stringify({cmd: "span", ch, force_n: reference}));
    toast(`Keep only ${sourceName(ch)} at ${reference.toFixed(2)} N perfectly steady for 3 seconds.`);
  });
}

function chart(canvas, getter, unit, limit = null) {
  const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return; canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const t1 = S.t, t0 = Math.max(0, t1 - 30), left = 40, right = 8, top = 20, bottom = 22;
  const series = getter(); let vals = [];
  for (const s of series) for (const q of s.pts) if (q.t >= t0 && Number.isFinite(q.v)) vals.push(q.v);
  let ymin = limit?.[0], ymax = limit?.[1];
  if (!limit) { ymin = vals.length ? Math.min(...vals) : 0; ymax = vals.length ? Math.max(...vals) : 1; const pad = Math.max((ymax - ymin) * .12, .1); ymin -= pad; ymax += pad; }
  if (ymax <= ymin) ymax = ymin + 1;
  g.strokeStyle = "#1f2b38"; g.fillStyle = "#6e7c8d"; g.font = "10px system-ui"; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const y = top + (h - top - bottom) * k / 4; g.beginPath(); g.moveTo(left, y); g.lineTo(w - right, y); g.stroke(); const v = ymax - (ymax - ymin) * k / 4; g.fillText(v.toFixed(Math.abs(ymax-ymin)<10?1:0), 4, y + 3); }
  g.fillText(unit, 4, 12);
  for (const s of series) { g.strokeStyle = s.color; g.lineWidth = 1.8; g.beginPath(); let open = false;
    for (const q of s.pts) { if (q.t < t0 || !Number.isFinite(q.v)) { open = false; continue; } const x = left + (w-left-right)*(q.t-t0)/Math.max(.01,t1-t0), y = top+(h-top-bottom)*(ymax-q.v)/(ymax-ymin); if (!open) { g.moveTo(x,y); open=true; } else g.lineTo(x,y); } g.stroke(); }
  let lx = left; for (const s of series) { g.fillStyle=s.color; g.fillRect(lx,6,9,3); g.fillStyle="#9aa7b7"; g.fillText(s.label,lx+13,10); lx += 70; }
}
let drawPending = false;
function drawCharts() {
  if (drawPending) return; drawPending = true; requestAnimationFrame(() => { drawPending = false;
    const patchSeries = (key) => PATCHES.map((p) => ({label: PATCH_LABEL[p], color: COLORS[p], pts: S.history.map((h) => ({t:h.t,v:h.patches[p]?.[key]}))}));
    chart($("forceChart"), () => patchSeries("force"), "N", [0, 10]);
    chart($("slipChart"), () => patchSeries("slip"), "%", [0, 100]);
    chart($("imuChart"), () => [0,1,2].map((i) => ({label:["Roll","Pitch","Yaw"][i],color:["#35d0e5","#f4a261","#a78bfa"][i],pts:S.history.map((h)=>({t:h.t,v:h.imu?.rpy?.[i]}))})), "deg");
  });
}

function download(name, type, content) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([content], {type})); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function exportCsv() {
  if (!S.derivedLog.length) return toast("No derived samples to export.");
  const keys = Object.keys(S.derivedLog[0]);
  const cell = (v) => v === null || v === undefined ? "" : typeof v === "string" ? `"${v.replaceAll('"','""')}"` : String(v);
  download(`dermis-derived-${Date.now()}.csv`, "text/csv", [keys.join(","), ...S.derivedLog.map((r) => keys.map((k) => cell(r[k])).join(","))].join("\n"));
}
async function replayFile(file) {
  disconnect(true); resetSignalState(); S.firstDeviceUs = S.lastDeviceUs = null; beginCalibration(true);
  const lines = (await file.text()).split(/\r?\n/).filter(Boolean); let i = 0;
  const step = () => { for (let n=0; n<15 && i<lines.length; n++,i++) { try { processFrame(JSON.parse(lines[i]), lines[i]); } catch(e) { alert("bad",`Replay line ${i+1}: ${e.message}`,`replay-${i}`); } }
    if (i < lines.length) requestAnimationFrame(step); else { setConnection("", "Replay complete"); toast(`Replayed ${lines.length} JSONL frames.`); } };
  setConnection("live", "Replaying"); step();
}

// ---- calibration export / import -------------------------------------------------
// File = dashboard config (assignments, areas, reference loads), the host zero baseline, and the
// glove's force spans (read with get_cal, restored with set_cal; only the primary client may restore).
let calWaiter = null;
function onFirmwareCalibration(msg) { S.fwCal = msg; if (calWaiter) { calWaiter(msg); calWaiter = null; } }
function askFirmwareCalibration(timeoutMs = 1500) {
  if (S.socket?.readyState !== WebSocket.OPEN) return Promise.resolve(null);
  return new Promise((resolve) => {
    const timer = setTimeout(() => { calWaiter = null; resolve(null); }, timeoutMs);
    calWaiter = (m) => { clearTimeout(timer); resolve(m); };
    S.socket.send(JSON.stringify({cmd: "get_cal"}));
  });
}
async function exportCalibration() {
  const fw = await askFirmwareCalibration();
  const file = {kind: "dermis-calibration", version: 1, exported_at: new Date().toISOString(),
    device_id: fw?.device_id ?? null, assignments: S.assignments, areas: S.areas, references: S.references,
    baseline: S.baseline, firmware_spans: fw?.spans ?? null};
  download(`dermis-calibration-${Date.now()}.json`, "application/json", JSON.stringify(file, null, 2));
  toast(fw ? `Calibration exported (${fw.spans.filter((x) => x.ready).length}/4 force spans from glove).`
           : "Exported without glove force spans (not connected or glove did not answer).");
}
async function importCalibration(file) {
  let data;
  try { data = JSON.parse(await file.text()); } catch (e) { return toast(`Not valid JSON: ${e.message}`); }
  if (data?.kind !== "dermis-calibration" || data.version !== 1) return toast("Not a Dermis calibration file (kind/version mismatch).");
  const pick = (obj, keys, ok) => Object.fromEntries(keys.filter((k) => ok(obj?.[k])).map((k) => [k, obj[k]]));
  const chs = ["0", "1", "2", "3"];
  S.assignments = {...S.assignments, ...pick(data.assignments, chs, (v) => PATCHES.includes(v))};
  S.areas = {...S.areas, ...pick(data.areas, PATCHES, (v) => Number.isFinite(v) && v > 0)};
  S.references = {...S.references, ...pick(data.references, chs, (v) => Number.isFinite(v) && v > 0 && v <= 9.80665)};
  saveConfig(); setupCards();
  const base = data.baseline && typeof data.baseline === "object" ? data.baseline : {};
  const usable = Object.entries(base).filter(([k, b]) => (k === "imu" || chs.includes(k)) && b && typeof b === "object");
  if (usable.length) { S.baseline = Object.fromEntries(usable); S.calibration = null; $("calState").textContent = "imported zero · re-zero if the glove drifted"; }
  let pushed = "";
  const spans = Array.isArray(data.firmware_spans) ? data.firmware_spans.filter((x) => x?.ready) : [];
  if (spans.length) {
    if (S.socket?.readyState === WebSocket.OPEN) {
      const reply = askFirmwareCalibration(2000);   // the glove answers set_cal with its new state
      S.socket.send(JSON.stringify({cmd: "set_cal", spans}));
      const fw = await reply;
      const n = fw ? fw.spans.filter((x) => x.ready).length : null;
      pushed = n === null ? " Glove did not confirm (only the primary client can restore spans)."
        : n >= spans.length ? ` ${n}/4 force spans restored on glove.` : ` Glove has ${n}/4 spans; this client may be view-only.`;
    } else pushed = " Connect to the glove and import again to restore its force spans.";
  }
  toast(`Calibration imported.${pushed}`);
}
$("exportCal").onclick = exportCalibration;
$("importCal").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) importCalibration(f); };

const query = new URLSearchParams(location.search);
$("wsUrl").value = query.get("ws") || localStorage.getItem("dermis.wsUrl") || $("wsUrl").value;
$("connectBtn").onclick = connect;
$("calibrateBtn").onclick = () => { resetSignalState(); beginCalibration(false); toast("Firmware tare started. Keep every patch still and unloaded for 5 seconds."); };
$("recordBtn").onclick = () => { S.recording = !S.recording; $("recordBtn").classList.toggle("recording", S.recording); $("recordBtn").textContent = S.recording ? "● Recording" : "○ Recording paused"; };
$("resetAssign").onclick = () => { S.assignments={...DEFAULT_ASSIGN}; S.areas={...DEFAULT_AREA}; saveConfig(); setupCards(); resetSignalState(); beginCalibration(false); };
$("exportJson").onclick = () => S.rawLog.length ? download(`dermis-raw-${Date.now()}.jsonl`, "application/x-ndjson", S.rawLog.join("\n")+"\n") : toast("No raw frames to export.");
$("exportCsv").onclick = exportCsv;
$("clearLog").onclick = () => { S.rawLog=[]; S.derivedLog=[]; renderSession(); toast("Export log cleared."); };
$("replayFile").onchange = (e) => e.target.files[0] && replayFile(e.target.files[0]);
window.addEventListener("resize", drawCharts); setupCards();
render({pods:{}},{patches:S.patchState,imu:null});
if (query.get("autoconnect") === "1") setTimeout(connect, 0);
