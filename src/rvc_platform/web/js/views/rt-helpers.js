/* ============================================================
   rt-helpers.js — أدوات البث اللحظي المشتركة (بحث 2025):
   - WORKLET_CODE: مُقطّع الكتل (16kHz) بنقل Float32Array
   - RingHistory: مخزن حلقي محدود (منع تسريب الذاكرة)
   - JitterAdaptor: تقدير تكيّف المخزن + تصحيح الانزلاق (NetEQ مبسّط)
   - ConnQuality: EWMA + hysteresis لجودة الاتصال (أنماط WebRTC)
   - WavRecorder: تجميع PCM يدوي + ترويسة WAV صحيحة (بدل MediaRecorder)
   - Scope: مرئي Waveform/Spectrum من AnalyserNode (DPR-aware)
   - drawSparkline: رسم سلسلة زمنية LTR على canvas حلقي
   ============================================================ */

/* --------- مُقطّع الكتل: يجمع عينات الميكروفون في كتل blockMs --------- */
export const WORKLET_CODE = `
class IvsBlockProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const blockMs = (options && options.processorOptions && options.processorOptions.blockMs) || 160;
    this.blockSamples = Math.max(1, Math.round(sampleRate * blockMs / 1000));
    this.buf = new Float32Array(this.blockSamples);
    this.pos = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input.length && input[0]) {
      const ch = input[0];
      let i = 0;
      while (i < ch.length) {
        const take = Math.min(this.blockSamples - this.pos, ch.length - i);
        this.buf.set(ch.subarray(i, i + take), this.pos);
        this.pos += take;
        i += take;
        if (this.pos >= this.blockSamples) {
          const out = this.buf.slice(0);
          this.port.postMessage(out, [out.buffer]);
          this.pos = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('ivs-block', IvsBlockProcessor);
`;

/* ---------------------- مخزن حلقي محدود ---------------------- */
export class RingHistory {
  constructor(cap = 240) {
    this.cap = cap;
    this.buf = new Float32Array(cap);
    this.idx = 0;
    this.count = 0;
  }
  push(v) {
    if (!Number.isFinite(v)) return;
    this.buf[this.idx] = v;
    this.idx = (this.idx + 1) % this.cap;
    this.count = Math.min(this.count + 1, this.cap);
  }
  /** نسخة مرتبة قديم → جديد. */
  toArray() {
    const out = new Float32Array(this.count);
    for (let k = 0; k < this.count; k++) {
      out[k] = this.buf[(this.idx - this.count + k + 2 * this.cap) % this.cap];
    }
    return out;
  }
  stats() {
    const a = this.toArray();
    if (!a.length) return null;
    const sorted = Float32Array.from(a).sort();
    let sum = 0;
    for (const v of a) sum += v;
    return {
      n: a.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      avg: sum / a.length,
      p50: sorted[Math.floor(0.5 * (sorted.length - 1))],
      p95: sorted[Math.floor(0.95 * (sorted.length - 1))],
      jitter: sorted.length > 2 ? meanAbsDiff(a) : 0,
    };
  }
}

function meanAbsDiff(values) {
  let s = 0, n = 0;
  for (let i = 1; i < values.length; i++) { s += Math.abs(values[i] - values[i - 1]); n++; }
  return n ? s / n : 0;
}

/* ------------- مُكيّف مخزن التشغيل (NetEQ مبسّط للكتل المجدولة) -------------
   يراقب: (1) عمق المخزن المجدول أمام currentTime، (2) تذبذب وصول الكتل،
   (3) عدد underruns — ثم يقرر: تكبير pre-roll، أو «أكل» الفائض عبر
   playbackRate ≤ 1 (تصحيح انزلاق تدريجي محسوس أقل من 0.5%). */
export class JitterAdaptor {
  constructor({ blockSec, prerollBlocks = 2 }) {
    this.blockSec = blockSec;
    this.prerollBlocks = prerollBlocks;
    this.minPreroll = 2;
    this.maxPreroll = 6;
    this.aheadEwma = blockSec * prerollBlocks;
    this.arrivalJitterEwma = 0;
    this._lastArrival = 0;
    this.underruns = 0;
    this.underrunsWindow = [];
    this.driftCorr = 1.0;
  }
  noteArrival(nowMs) {
    if (this._lastArrival) {
      const dt = Math.abs(nowMs - this._lastArrival);
      this.arrivalJitterEwma = 0.9 * this.arrivalJitterEwma + 0.1 * dt;
    }
    this._lastArrival = nowMs;
  }
  noteScheduled(aheadSec) {
    this.aheadEwma = 0.9 * this.aheadEwma + 0.1 * Math.max(0, aheadSec);
    const target = this.prerollBlocks * this.blockSec;
    if (this.aheadEwma > target * 1.75) this.driftCorr = 0.995;      // فائض → أكل بهدوء
    else this.driftCorr = 1.0;
  }
  noteUnderrun(nowMs = performance.now()) {
    this.underruns += 1;
    this.underrunsWindow.push(nowMs);
    if (this.underrunsWindow.length > 8) this.underrunsWindow.shift();
    const recent = this.underrunsWindow.filter((t) => nowMs - t < 30000).length;
    if (recent >= 2 && this.prerollBlocks < this.maxPreroll) {
      this.prerollBlocks += 1;                        // وسّع امتصاص التذبذب
      this.underrunsWindow = [];
      return this.prerollBlocks;                       // أبلغ المستهلك بالتغيير
    }
    return null;
  }
  /** عتبة الاستئناف المريح بعد فجوة (ثوانٍ) — مثل upstream. */
  resumeAhead() {
    return Math.max(0.18, this.prerollBlocks * this.blockSec);
  }
}

/* ------------- جودة الاتصال: EWMA + hysteresis (أنماط getStats) ------------- */
export class ConnQuality {
  constructor() {
    this.rttEwma = null;
    this.level = 2; // 0 ممتاز 1 جيد 2 متوسط 3 ضعيف
    this._pend = null;
    this._misses = 0;
    this._lastPongAt = 0;
  }
  notePong(rttMs, nowMs = performance.now()) {
    this.rttEwma = this.rttEwma == null ? rttMs : 0.9 * this.rttEwma + 0.1 * rttMs;
    this._lastPongAt = nowMs;
    this._misses = 0;
  }
  notePongTimeout() { this._misses += 1; }
  get stale() { return this._lastPongAt === 0 || performance.now() - this._lastPongAt > 12000; }
  get dead() { return this._misses >= 2; }
  /** مستوى مع hysteresis: يلزم تأكيد انتقالين متتاليين قبل التبديل. */
  compute() {
    if (this.rttEwma == null) return this.level;
    const r = this.rttEwma;
    let want;
    if (r < 160) want = 0;
    else if (r < 320) want = 1;
    else if (r < 480) want = 2;
    else want = 3;
    if (want === this.level) { this._pend = null; return this.level; }
    if (this._pend === want) { this.level = want; this._pend = null; }
    else this._pend = want;
    return this.level;
  }
}

export const QUALITY_LABELS = [
  ["ممتاز — شبكة مستقرة", "ok"],
  ["جيد — مناسب للبث", "info"],
  ["متوسط — راقب الكمون", "warn"],
  ["ضعيف — قد تتقطع الجلسة", "error"],
];

/* ------------- مسجّل WAV: تجميع PCM يدوي + ترويسة قياسية ------------- */
export class WavRecorder {
  /** tracks: قاموس اسم → {sampleRate, chunks:[Float32Array]} — القنوات تُدمج عند الإنهاء. */
  constructor() { this.tracks = new Map(); this.startedAt = performance.now(); }
  addTrack(name, sampleRate) {
    if (!this.tracks.has(name)) this.tracks.set(name, { sampleRate, chunks: [] });
  }
  push(name, floats) {
    const t = this.tracks.get(name);
    if (t) t.chunks.push(Float32Array.from(floats));
  }
  get seconds() { return (performance.now() - this.startedAt) / 1000; }
  get empty() {
    for (const t of this.tracks.values()) if (t.chunks.length) return false;
    return true;
  }
  /** يُرجع Blob WAV 16-bit؛ القناتان «dry/wet» تخرج ستيريو إن توفرتا. */
  finish() {
    const names = [...this.tracks.keys()];
    if (!names.length) return null;
    // وحّد معدل القنوات إلى معدل القناة الأولى (تُحاكى مسبقًا في الجلسة).
    const sr = this.tracks.get(names[0]).sampleRate;
    const arrays = names.map((n) => {
      const t = this.tracks.get(n);
      const total = t.chunks.reduce((a, c) => a + c.length, 0);
      const merged = new Float32Array(total);
      let off = 0;
      for (const c of t.chunks) { merged.set(c, off); off += c.length; }
      return merged;
    });
    return encodeWav(arrays, sr);
  }
}

/** ترميز WAV PCM 16-bit: قناتان = ستيريو (L,R). */
export function encodeWav(channelArrays, sampleRate) {
  const channels = channelArrays.length;
  const len = Math.min(...channelArrays.map((a) => a.length));
  const dataBytes = len * channels * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  wstr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  wstr(8, "WAVE");
  wstr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);            // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  wstr(36, "data");
  view.setUint32(40, dataBytes, true);
  let off = 44;
  for (let i = 0; i < len; i++) {
    for (let ch = 0; ch < channels; ch++) {
      const v = Math.max(-1, Math.min(1, channelArrays[ch][i]));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      off += 2;
    }
  }
  return new Blob([buf], { type: "audio/wav" });
}

/* ------------- مرئي Scope: Waveform + Spectrum (AnalyserNode tap) ------------- */
export class Scope {
  constructor(canvas, { fftSize = 1024, color = "#34d399", fill = "rgba(52,211,153,0.12)" } = {}) {
    this.canvas = canvas;
    this.color = color;
    this.fill = fill;
    this.time = new Float32Array(fftSize);
    this.freq = new Uint8Array(fftSize / 2);
    this._rafId = null;
    this._alive = false;
  }
  attach(analyser) { this.analyser = analyser || null; }
  start() {
    if (this._alive) return;
    this._alive = true;
    const loop = () => {
      if (!this._alive) return;
      this.draw();
      this._rafId = requestAnimationFrame(loop);
    };
    this._rafId = requestAnimationFrame(loop);
  }
  stop() {
    this._alive = false;
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this._rafId = null;
  }
  draw() {
    const cv = this.canvas;
    if (!cv || !this.analyser) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 300, h = cv.clientHeight || 70;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const a = this.analyser;
    a.getFloatTimeDomainData(this.time);
    // خط الصفر
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    // الموجة — تتقدم LTR (محور زمن: لا يُعكس في RTL)
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const n = Math.min(a.fftSize, 2048);
    for (let i = 0; i < n; i += 2) {
      const x = (i / n) * w;
      const y = h / 2 - this.time[i] * (h / 2 - 2);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
  drawSpectrumInto(ctx, w, h, dpr) {
    // يُرسم من الخارج في لوحة الطيف المشتركة
    if (!this.analyser) return;
    this.analyser.getByteFrequencyData(this.freq);
    const bins = Math.min(48, this.freq.length);
    const bw = w / bins;
    ctx.fillStyle = this.color;
    for (let i = 0; i < bins; i++) {
      const v = this.freq[i] / 255;
      const bh = Math.max(2, v * (h - 4));
      ctx.globalAlpha = 0.75;
      ctx.fillRect(i * bw + 1, h - bh, Math.max(2, bw - 2), bh);
    }
    ctx.globalAlpha = 1;
  }
}

/* ------------- رسم Sparkline (LTR، DPR-aware، بلا محاور) ------------- */
export function drawSparkline(canvas, ring, { color = "#34d399", budget = 0, warnColor = "#f5b23c" } = {}) {
  const cv = canvas;
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 260, h = cv.clientHeight || 46;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const data = ring.toArray();
  if (!data.length) {
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "10px sans-serif";
    ctx.fillText("لا قياسات بعد", 6, h / 2);
    return;
  }
  const max = Math.max(...data, budget || 0, 1);
  // خط الميزانية (إن وُجد)
  if (budget > 0) {
    ctx.strokeStyle = "rgba(245,178,60,0.5)";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, h - (budget / max) * (h - 3) - 1);
    ctx.lineTo(w, h - (budget / max) * (h - 3) - 1);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const over = data[data.length - 1] > budget && budget > 0;
  ctx.strokeStyle = over ? warnColor : color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / Math.max(1, data.length - 1)) * (w - 2) + 1;
    const y = h - (data[i] / max) * (h - 4) - 2;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
}

/* ------------- تنسيقات عرض ------------- */
export function fmtKbps(bytesPerSec) {
  const kb = bytesPerSec / 1024;
  if (!Number.isFinite(kb)) return "—";
  if (kb >= 1024) return `${(kb / 1024).toFixed(1)} MB/s`;
  return `${kb.toFixed(1)} KB/s`;
}

export function fmtDur(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `0:${String(s).padStart(2, "0")}`;
}

export function nowClock() {
  const d = new Date();
  const p = (x) => String(x).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
