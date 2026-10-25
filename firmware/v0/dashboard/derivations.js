(function (root) {
  "use strict";

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const finite = (v) => Number.isFinite(v);
  const norm = (v) => Math.hypot(...v);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const median = (values) => {
    const a = values.filter(finite).slice().sort((x, y) => x - y);
    if (!a.length) return null;
    const m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  };
  const vectorMedian = (values) => [0, 1, 2].map((i) => median(values.map((v) => v && v[i])));
  const mad = (values, center = median(values)) => {
    if (center === null) return null;
    return 1.4826 * median(values.filter(finite).map((v) => Math.abs(v - center)));
  };
  const vectorMadNorm = (values, center = vectorMedian(values)) =>
    norm([0, 1, 2].map((i) => mad(values.map((v) => v && v[i]), center[i]) || 0));

  class MedianEma {
    constructor(windowSize = 5, tauSeconds = 0.12) {
      this.windowSize = Math.max(1, windowSize);
      this.tau = Math.max(0, tauSeconds);
      this.history = [];
      this.value = null;
      this.lastSeconds = null;
    }
    update(input, seconds) {
      const vector = Array.isArray(input);
      const raw = vector ? input : [input];
      if (!raw.every(finite)) return null;
      const x = raw.map(Number);
      this.history.push(x);
      if (this.history.length > this.windowSize) this.history.shift();
      const center = vectorMedian(this.history);
      if (!this.value) this.value = center;
      else {
        const dt = this.lastSeconds === null ? 0 : clamp(seconds - this.lastSeconds, 0, 0.5);
        const a = this.tau === 0 ? 1 : 1 - Math.exp(-dt / this.tau);
        this.value = this.value.map((v, i) => v + a * (center[i] - v));
      }
      this.lastSeconds = seconds;
      return vector ? this.value.slice() : this.value[0];
    }
    reset() { this.history = []; this.value = null; this.lastSeconds = null; }
  }

  function temperatureColor(deltaC) {
    if (!finite(deltaC) || Math.abs(deltaC) <= 5) return "#667085";
    const x = clamp((Math.abs(deltaC) - 5) / 10, 0, 1);
    if (deltaC > 0) return `rgb(${Math.round(189 + 66 * x)},${Math.round(93 - 42 * x)},${Math.round(38 - 18 * x)})`;
    return `rgb(${Math.round(45 - 18 * x)},${Math.round(122 + 45 * x)},${Math.round(181 + 74 * x)})`;
  }

  function forceColor(forceN, scaleN = 10) {
    if (!finite(forceN) || forceN <= 0) return "#344054";
    const x = Math.sqrt(clamp(forceN / Math.max(0.1, scaleN), 0, 1));
    return `rgb(${Math.round(105 + 150 * x)},${Math.round(38 - 25 * x)},${Math.round(48 - 32 * x)})`;
  }

  function slipScore({contact, transverseRateUtS, forceRateNS,
    fieldRateFloorUtS = 40, forceRateFloorNS = 0.2}) {
    if (!contact) return 0;
    // Heuristic only: transverse field motion plus unloading can indicate relative patch motion.
    // Both terms use rate-domain noise floors measured during the unloaded calibration.
    const field = Math.max(0, transverseRateUtS - fieldRateFloorUtS) /
      Math.max(500, 3 * fieldRateFloorUtS);
    const unloading = Math.max(0, -forceRateNS - forceRateFloorNS) /
      Math.max(1, 4 * forceRateFloorNS);
    return clamp(100 * (1 - Math.exp(-(0.8 * field + unloading))), 0, 100);
  }

  const api = {clamp, finite, norm, sub, median, vectorMedian, mad, vectorMadNorm,
    MedianEma, temperatureColor, forceColor, slipScore};
  root.DermisMath = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
