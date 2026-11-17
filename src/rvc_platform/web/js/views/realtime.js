/* ============================================================
   realtime.js — البث اللحظي: مايكروفون → AudioWorklet (16kHz int16)
   → WebSocket → خادم RVC → PCM → تشغيل مع مخزن امتصاص تذبذب،
   مع قياسات زمن الاستجابة (الخادم + العميل) وواجهة تحكم حية.
   ============================================================ */

import {
  get, openRealtimeWS, toast, el, clear, icon, clamp, fmtNum, fmtMs, f0MethodSelect,
} from "../api.js";

export default { mount };

const INPUT_RATE = 16000;
const PREQUEUE_BLOCKS = 2;          // jitter absorption: 2 blocks preroll
const RTT_KEEP = 100;                // keep last 100 client round-trips
const STATS_EVERY_MS = 2000;
const PING_EVERY_MS = 5000;

/* ------------------------------ mount ------------------------------ */

async function mount(root) {
  const state = {
    phase: "idle", // idle | connecting | streaming | error
    models: [],
    info: null,
    blockMs: 160,
    modelSr: null,
    settings: {
      pitch_shift: 0,
      index_rate: 0.0,
      rms_mix_rate: 0.0,
      f0_method: "rmvpe",
      silence_threshold_db: -40,
    },
    // live session objects
    ws: null,
    stream: null,
    inCtx: null,
    outCtx: null,
    workletNode: null,
    workletUrl: null,
    micSource: null,
    // stats
    pendingTs: [],
    rttHistory: [],
    blocksSent: 0,
    blocksReceived: 0,
    serverStats: null,
    serverBlocks: 0,
    nextTime: 0,
    statsTimer: null,
    pingTimer: null,
    lastError: null,
    sessionId: 0,
  };

  root.replaceChildren(el("div", { class: "stack" }, [
    buildIntroCard(),
    buildControlsCard(),
    buildStatusCard(),
    buildMetersCard(),
    buildLatencyCard(),
  ]));

  const ui = {
    model: root.querySelector("#rt-model"),
    btnStart: root.querySelector("#rt-start"),
    stateLine: root.querySelector("#rt-state"),
    statusBox: root.querySelector("#rt-status-box"),
    errorBox: root.querySelector("#rt-error"),
    pitch: root.querySelector("#rt-pitch"),
    pitchVal: root.querySelector("#rt-pitch-val"),
    index: root.querySelector("#rt-index"),
    indexVal: root.querySelector("#rt-index-val"),
    rms: root.querySelector("#rt-rms"),
    rmsVal: root.querySelector("#rt-rms-val"),
    f0: root.querySelector("#rt-f0"),
    infoBox: root.querySelector("#rt-info-box"),
    meterIn: root.querySelector("#rt-meter-in"),
    meterOut: root.querySelector("#rt-meter-out"),
    blocksSent: root.querySelector("#rt-blocks-sent"),
    blocksRecv: root.querySelector("#rt-blocks-recv"),
    srvBlocks: root.querySelector("#rt-srv-blocks"),
    srvTile: root.querySelector("#rt-srv-tiles"),
    cliTile: root.querySelector("#rt-cli-tiles"),
    verdict: root.querySelector("#rt-verdict"),
  };

  wireLiveControls(ui, state);
  ui.btnStart.addEventListener("click", () => {
    if (state.phase === "streaming" || state.phase === "connecting") stopSession(state, ui);
    else startSession(state, ui);
  });

  await Promise.all([loadModels(state, ui), loadInfo(state, ui)]);

  function cleanup() {
    teardownSession(state, ui, false);
  }
  return cleanup;
}

/* ------------------------------ static UI ------------------------------ */

function buildIntroCard() {
  return el("div", { class: "notice info" }, [
    el("span", { class: "n-icon" }, [icon("mic", 20)]),
    el("div", {}, [
      el("p", { style: "font-weight:700;margin:0 0 4px" }, ["هذا تحويل لحظي بالبث المباشر (وليس تحويلًا دفعيًا للملفات)"]),
      el("p", { class: "small", style: "margin:0" }, [
        "يُلتقط صوت الميكروفون على دفعات ",
        el("span", { class: "latin" }, ["int16 @ 16 kHz"]),
        " ويُحوَّل عبر WebSocket ثم يعاد بثه بمعدل عينة النموذج. الزمن اللحظي الممكن يعتمد فعليًا على قوة GPU وسرعة الشبكة — قد يزيد على الميزانية المتوقعة للأجهزة البطيئة.",
      ]),
    ]),
  ]);
}

function buildControlsCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["التحكم بالجلسة", el("span", { class: "sub latin" }, ["Controls"])]),
    el("div", { class: "rt-state", id: "rt-state" }, [
      el("span", { class: "badge idle dot" }, ["خامل"]),
      el("span", { class: "small dim" }, ["اضغط زر البدء بعد اختيار نموذج f0."]),
    ]),
    el("div", { class: "form-grid mt" }, [
      el("div", { class: "field" }, [
        el("label", { for: "rt-model" }, ["النموذج (فهرس f0 فقط)"]),
        el("select", { id: "rt-model" }, [
          el("option", { value: "" }, ["جارٍ جلب النماذج…"]),
        ]),
      ]),
      el("div", { class: "field" }, [
        el("label", {}, ["حالة الخادم اللحظي"]),
        el("div", { class: "small dim", id: "rt-info-box" }, ["…"]),
      ]),
    ]),
    el("div", { class: "slider-row mt" }, [
      el("div", { class: "slider-head" }, [
        el("label", { for: "rt-pitch" }, ["انزياح النغمة (Pitch)", el("span", { class: "opt-latin" }, ["semitones"])]),
        el("span", { class: "slider-val", id: "rt-pitch-val" }, ["0"]),
      ]),
      el("input", { type: "range", id: "rt-pitch", min: "-24", max: "24", step: "1", value: "0" }),
    ]),
    el("div", { class: "slider-row" }, [
      el("div", { class: "slider-head" }, [
        el("label", { for: "rt-index" }, ["نسبة الفهرس", el("span", { class: "opt-latin" }, ["Index Rate"])]),
        el("span", { class: "slider-val", id: "rt-index-val" }, ["0.00"]),
      ]),
      el("input", { type: "range", id: "rt-index", min: "0", max: "1", step: "0.01", value: "0" }),
    ]),
    el("div", { class: "slider-row" }, [
      el("div", { class: "slider-head" }, [
        el("label", { for: "rt-rms" }, ["مزج الشدة", el("span", { class: "opt-latin" }, ["RMS Mix"])]),
        el("span", { class: "slider-val", id: "rt-rms-val" }, ["0.00"]),
      ]),
      el("input", { type: "range", id: "rt-rms", min: "0", max: "1", step: "0.01", value: "0" }),
    ]),
    el("div", { class: "field mt" }, [
      el("label", { for: "rt-f0" }, ["طريقة استخراج الطبقة — ", el("span", { class: "latin" }, ["f0 method"])]),
      f0MethodSelect("rmvpe", "rt-f0"),
    ]),
    el("button", { class: "btn primary block mt", type: "button", id: "rt-start" }, [
      icon("play", 16), " بدء البث اللحظي",
    ]),
    el("div", { id: "rt-error", class: "mt hidden" }, []),
  ]);
}

function buildStatusCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["حالة الجلسة", el("span", { class: "sub latin" }, ["Session"])]),
    el("div", { id: "rt-status-box" }, [
      el("p", { class: "muted small" }, ["لا توجد جلسة نشطة."]),
    ]),
  ]);
}

function buildMetersCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["المستويات والعدادات", el("span", { class: "sub latin" }, ["Meters"])]),
    meterRow("rt-meter-in", "الدخل (مايكروفون)"),
    meterRow("rt-meter-out", "الخرج (الصوت المحوّل)"),
    el("div", { class: "latency-grid mt" }, [
      tileEl("rt-blocks-sent", "كتل مُرسلة"),
      tileEl("rt-blocks-recv", "كتل مستلمة"),
      tileEl("rt-srv-blocks", "كتل معالجة (خادم)"),
    ]),
  ]);
}

function meterRow(id, label) {
  return el("div", { class: "meter", id }, [
    el("span", { class: "meter-label" }, [label]),
    el("div", { class: "track" }, [
      el("div", { class: "fill", id: `${id}-fill` }),
      el("div", { class: "peak", id: `${id}-peak` }),
    ]),
    el("span", { class: "db", id: `${id}-db` }, ["-inf dB"]),
  ]);
}

function tileEl(id, label) {
  return el("div", { class: "stat-tile" }, [
    el("div", { class: "t-label" }, [label]),
    el("div", { class: "t-value", id }, ["—"]),
  ]);
}

function buildLatencyCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["لوحة زمن الاستجابة", el("span", { class: "sub latin" }, ["Latency"])]),
    el("div", { class: "small dim" }, [
      "الخادم: زمن استدلال المحرك لكل كتلة. العميل: زمن الذهاب-والإياب حتى جدولة الكتلة للتشغيل (آخر 100 قياس).",
    ]),
    el("div", { class: "latency-grid mt", id: "rt-srv-tiles" }, [
      tileEl("rt-srv-min", "خادم — أدنى"),
      tileEl("rt-srv-avg", "خادم — متوسط"),
      tileEl("rt-srv-p95", "خادم — p95"),
      tileEl("rt-srv-max", "خادم — أقصى"),
      tileEl("rt-srv-bridge", "جسر النقل"),
    ]),
    el("div", { class: "latency-grid mt", id: "rt-cli-tiles" }, [
      tileEl("rt-cli-min", "عميل — أدنى"),
      tileEl("rt-cli-avg", "عميل — متوسط"),
      tileEl("rt-cli-p95", "عميل — p95"),
    ]),
    el("div", { class: "verdict", id: "rt-verdict" }, ["لا توجد قياسات بعد."]),
  ]);
}

/* ------------------------------ data loads ------------------------------ */

async function loadModels(state, ui) {
  try {
    const models = await get("/models");
    state.models = (Array.isArray(models) ? models : []).filter((m) => m.f0);
    clear(ui.model);
    if (!state.models.length) {
      ui.model.append(el("option", { value: "" }, ["لا توجد نماذج f0 — تدريب أو رفع نموذج أولًا"]));
      ui.model.disabled = true;
      ui.btnStart.disabled = true;
      return;
    }
    for (const m of state.models) ui.model.append(el("option", { value: m.name }, [m.name]));
    if (state.info?.loaded_model?.name && state.models.some((m) => m.name === state.info.loaded_model.name)) {
      ui.model.value = state.info.loaded_model.name;
    } else {
      ui.model.value = state.models[0].name;
    }
  } catch (err) {
    clear(ui.model);
    ui.model.append(el("option", { value: "" }, ["تعذّر جلب النماذج"]));
    toast(err.message, "error");
  }
}

async function loadInfo(state, ui) {
  try {
    state.info = await get("/realtime/info");
    state.blockMs = state.info.block_ms || 160;
    state.settings.silence_threshold_db = state.info.silence_threshold_db ?? -40;
    if (state.info.f0_method) {
      state.settings.f0_method = state.info.f0_method;
      ui.f0.value = state.info.f0_method;
    }
    ui.infoBox.replaceChildren();
    ui.infoBox.append(el("span", { class: "latin" }, [
      `block ${state.info.block_ms} ms · crossfade ${state.info.crossfade_ms} ms · lookback ${state.info.lookback_ms} ms · f0 ${state.info.f0_method}`,
    ]));
    if (state.info.available) {
      ui.infoBox.append(el("span", { class: "badge ok" }, ["متاح الآن"]));
    } else {
      ui.infoBox.append(el("span", {}, [" — يتطلب نموذج f0 محمّلًا (سيُحمّل عند البدء)"]));
    }
    if (state.info.loaded_model?.name && state.models.some((m) => m.name === state.info.loaded_model.name)) {
      ui.model.value = state.info.loaded_model.name;
    }
  } catch (err) {
    ui.infoBox.textContent = err.message;
  }
}

/* ------------------------------ live controls ------------------------------ */

function wireLiveControls(ui, state) {
  ui.pitch.addEventListener("input", () => {
    ui.pitchVal.textContent = ui.pitch.value;
    state.settings.pitch_shift = Number(ui.pitch.value);
    pushSettings(state);
  });
  ui.index.addEventListener("input", () => {
    ui.indexVal.textContent = Number(ui.index.value).toFixed(2);
    state.settings.index_rate = Number(ui.index.value);
    pushSettings(state);
  });
  ui.rms.addEventListener("input", () => {
    ui.rmsVal.textContent = Number(ui.rms.value).toFixed(2);
    state.settings.rms_mix_rate = Number(ui.rms.value);
    pushSettings(state);
  });
  ui.f0.addEventListener("change", () => {
    state.settings.f0_method = ui.f0.value;
    pushSettings(state);
  });
}

function pushSettings(state) {
  if (state.phase === "streaming" && state.ws) {
    state.ws.sendText({ type: "settings", settings: settingsPayload(state) });
  }
}

function settingsPayload(state) {
  return {
    pitch_shift: state.settings.pitch_shift,
    index_rate: state.settings.index_rate,
    f0_method: state.settings.f0_method,
    rms_mix_rate: state.settings.rms_mix_rate,
    silence_threshold_db: state.settings.silence_threshold_db,
  };
}

/* ------------------------------ session lifecycle ------------------------------ */

async function startSession(state, ui) {
  if (state.phase === "connecting" || state.phase === "streaming") return;
  const modelName = ui.model.value;
  if (!modelName) {
    toast("اختر نموذجًا يدعم f0 أولًا.", "warn");
    return;
  }
  state.lastError = null;
  const sessionId = ++state.sessionId;
  const alive = () => state.sessionId === sessionId && (state.phase === "connecting" || state.phase === "streaming");
  setPhase(state, ui, "connecting", `جارٍ الاتصال والتحميل — ${modelName}`);

  // 1) microphone
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
  } catch (err) {
    fail(state, ui, `تعذّر الوصول إلى الميكروفون: ${err.name === "NotAllowedError" ? "الإذن مرفوض من المتصفح" : err.message || err.name}`);
    return;
  }
  if (!alive()) { for (const t of stream.getAudioTracks()) t.stop(); return; }
  state.stream = stream;

  // 2) input context @16k (fallback: native rate + resample)
  try {
    state.inCtx = new AudioContext({ sampleRate: INPUT_RATE });
  } catch {
    state.inCtx = new AudioContext();
  }
  if (state.inCtx.state === "suspended") { try { await state.inCtx.resume(); } catch { /* gesture-bound */ } }
  if (!alive()) { teardownSession(state, ui); return; }
  const inRate = state.inCtx.sampleRate;

  // 3) playback context (created under the click gesture, native rate)
  state.outCtx = new AudioContext();
  if (state.outCtx.state === "suspended") { try { await state.outCtx.resume(); } catch { /* retried on play */ } }
  if (!alive()) { teardownSession(state, ui); return; }
  state.nextTime = 0;

  // 4) worklet
  try {
    state.workletUrl = URL.createObjectURL(new Blob([WORKLET_CODE], { type: "application/javascript" }));
    await state.inCtx.audioWorklet.addModule(state.workletUrl);
    if (!alive()) { teardownSession(state, ui); return; }
    state.workletNode = new AudioWorkletNode(state.inCtx, "ivs-block", {
      processorOptions: { blockMs: state.blockMs },
      numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
    });
  } catch (err) {
    fail(state, ui, `تعذّر تشغيل معالج الصوت (AudioWorklet): ${err.message || err}`);
    return;
  }
  state.micSource = state.inCtx.createMediaStreamSource(stream);
  state.micSource.connect(state.workletNode);

  const track = stream.getAudioTracks()[0];
  if (track) track.addEventListener("ended", () => {
    if (state.phase === "streaming" || state.phase === "connecting") {
      fail(state, ui, "انقطع الميكروفون (توقف المسار) — تحقق من الجهاز ثم أعد المحاولة.");
    }
  });

  state.workletNode.port.onmessage = (ev) => {
    const block = ev.data; // Float32Array at input-ctx rate
    if (!(block instanceof Float32Array)) return;
    const rms = rmsOf(block);
    updateMeter(state, ui, "in", rms);
    let pcm = block;
    if (inRate !== INPUT_RATE) pcm = resampleLinear(block, inRate, INPUT_RATE);
    const int16 = floatToInt16(pcm);
    if (state.ws && state.ws.readyState() === WebSocket.OPEN) {
      state.pendingTs.push(performance.now());
      if (state.pendingTs.length > 8) state.pendingTs.shift();
      state.blocksSent += 1;
      ui.blocksSent.textContent = String(state.blocksSent);
      state.ws.sendBinary(int16.buffer);
    }
  };

  // 5) websocket
  const ws = openRealtimeWS();
  state.ws = ws;

  ws.onopen = () => {
    ws.sendText({ type: "start", model: modelName, settings: settingsPayload(state) });
  };

  ws.ontext = (msg) => {
    if (!msg || !msg.type) return;
    if (msg.type === "started") {
      if (!alive()) return;
      state.modelSr = msg.sample_rate;
      setPhase(state, ui, "streaming", `بث نشط — ${msg.model || modelName} · خرج ${fmtNum((msg.sample_rate || 0) / 1000, 1)} kHz · ميزانية ${msg.latency_budget_ms || state.blockMs} ms`);
      toast("بدأ البث اللحظي.", "success");
      ui.model.disabled = true;
      state.statsTimer = setInterval(() => ws.sendText({ type: "stats" }), STATS_EVERY_MS);
      state.pingTimer = setInterval(() => ws.sendText({ type: "ping" }), PING_EVERY_MS);
      ws.sendText({ type: "stats" });
    } else if (msg.type === "settings_applied") {
      toast("طُبِّقت الإعدادات الجديدة على البث.", "success", 2500);
    } else if (msg.type === "stats") {
      state.serverStats = msg;
      renderServerStats(state, ui, msg);
    } else if (msg.type === "pong") {
      state.serverBlocks = msg.blocks ?? 0;
      ui.srvBlocks.textContent = String(state.serverBlocks);
    } else if (msg.type === "error") {
      fail(state, ui, `خطأ من الخادم: ${msg.message || "غير معروف"}`);
    }
  };

  ws.onbinary = (buffer) => {
    if (!alive() || !state.outCtx) return;
    const int16 = new Int16Array(buffer);
    if (!int16.length) return;
    let floats = int16ToFloat(int16);
    const outRate = state.outCtx.sampleRate;
    if (state.modelSr && outRate !== state.modelSr) {
      floats = resampleLinear(floats, state.modelSr, outRate);
    }
    const rms = rmsOf(floats);
    updateMeter(state, ui, "out", rms);

    const ctx = state.outCtx;
    const audioBuffer = ctx.createBuffer(1, floats.length, outRate);
    audioBuffer.copyToChannel(floats, 0);
    const src = ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(ctx.destination);
    const blockSec = audioBuffer.duration;
    if (state.nextTime === 0) {
      state.nextTime = ctx.currentTime + (PREQUEUE_BLOCKS * blockSec);
    } else if (state.nextTime < ctx.currentTime + 0.004) {
      // underrun: rebuild the preroll cushion
      state.nextTime = ctx.currentTime + (PREQUEUE_BLOCKS * blockSec);
    }
    src.start(state.nextTime);
    state.nextTime += blockSec;

    // client round-trip: oldest pending send timestamp ↔ its scheduled playback
    if (state.pendingTs.length) {
      const sentAt = state.pendingTs.shift();
      const rtt = performance.now() - sentAt;
      state.rttHistory.push(rtt);
      if (state.rttHistory.length > RTT_KEEP) state.rttHistory.shift();
      renderClientStats(state, ui);
    }
    state.blocksReceived += 1;
    ui.blocksRecv.textContent = String(state.blocksReceived);
  };

  ws.onclose = (code, reason) => {
    if (state.phase === "streaming" || state.phase === "connecting") {
      if (code === 4401) {
        fail(state, ui, "الرمز غير صالح أو مفقود — أضف الرمز من تبويب الإعدادات ثم أعد المحاولة.");
      } else {
        fail(state, ui, `انقطع الاتصال بالخادم (code ${code}${reason ? ` — ${reason}` : ""}) — استخدم زر إعادة الاتصال.`);
      }
    }
  };
  ws.onerror = () => {
    if (state.phase === "connecting") {
      fail(state, ui, "خطأ في اتصال WebSocket — تأكد من الخادم ومن الرمز إن كان مطلوبًا.");
    }
  };
}

/** clean stop (user action or cleanup) — no error UI. */
function stopSession(state, ui) {
  teardownSession(state, ui, false);
  setPhase(state, ui, "idle", "أُوقفت الجلسة.");
}

/** stop + show Arabic error with reconnect button. */
function fail(state, ui, message) {
  teardownSession(state, ui, false);
  state.lastError = message;
  setPhase(state, ui, "error", message);
  toast(message, "error", 9000);
}

/** closes everything without touching the phase UI. */
function teardownSession(state, ui) {
  state.sessionId = (state.sessionId || 0) + 1; // invalidate in-flight start()
  if (state.statsTimer) { clearInterval(state.statsTimer); state.statsTimer = null; }
  if (state.pingTimer) { clearInterval(state.pingTimer); state.pingTimer = null; }
  if (state.ws) {
    const ws = state.ws;
    ws.onclose = null; ws.ontext = null; ws.onbinary = null; ws.onerror = null;
    ws.close();
    state.ws = null;
  }
  if (state.micSource) { try { state.micSource.disconnect(); } catch { /* already gone */ } state.micSource = null; }
  if (state.workletNode) { try { state.workletNode.disconnect(); } catch { /* already gone */ } state.workletNode = null; }
  if (state.stream) { for (const t of state.stream.getAudioTracks()) t.stop(); state.stream = null; }
  if (state.inCtx) { try { state.inCtx.close(); } catch { /* ignore */ } state.inCtx = null; }
  if (state.outCtx) { try { state.outCtx.close(); } catch { /* ignore */ } state.outCtx = null; }
  if (state.workletUrl) { URL.revokeObjectURL(state.workletUrl); state.workletUrl = null; }
  state.pendingTs = [];
  state.nextTime = 0;
  state.modelSr = null;
  ui.model.disabled = state.models.length === 0;
  resetMeters(ui);
}

/* ------------------------------ phase UI ------------------------------ */

const PHASE_BADGE = {
  idle: ["خامل", "idle"],
  connecting: ["جارٍ الاتصال", "loading"],
  streaming: ["بث نشط", "running"],
  error: ["خطأ", "error"],
};

function setPhase(state, ui, phase, message) {
  state.phase = phase;
  const [label, cls] = PHASE_BADGE[phase] || [phase, "idle"];
  clear(ui.stateLine);
  ui.stateLine.append(
    el("span", { class: `badge ${cls} dot` }, [label]),
    el("span", { class: "small dim" }, [message || ""]),
  );
  ui.btnStart.innerHTML = "";
  if (phase === "streaming" || phase === "connecting") {
    ui.btnStart.append(icon("stop", 16), document.createTextNode(" إيقاف البث"));
    ui.btnStart.classList.remove("primary");
    ui.btnStart.classList.add("danger");
  } else {
    ui.btnStart.append(icon("play", 16), document.createTextNode(" بدء البث اللحظي"));
    ui.btnStart.classList.add("primary");
    ui.btnStart.classList.remove("danger");
  }
  ui.btnStart.disabled = state.models.length === 0;
  renderStatusBox(state, ui);
}

function renderStatusBox(state, ui) {
  const box = ui.statusBox;
  clear(box);
  if (state.phase === "error") {
    const reconnect = el("button", { class: "btn primary", type: "button" }, ["إعادة الاتصال"]);
    reconnect.addEventListener("click", () => startSession(state, ui));
    box.append(
      el("div", { class: "notice danger" }, [
        el("span", { class: "n-icon" }, [icon("alert", 18)]),
        el("p", {}, [state.lastError || "خطأ غير معروف."]),
      ]),
      el("div", { class: "row mt" }, [
        reconnect,
        el("a", { class: "btn ghost", href: "#settings" }, ["الإعدادات والرمز"]),
      ]),
    );
    return;
  }
  if (state.phase === "streaming") {
    box.append(el("dl", { class: "kv" }, [
      el("dt", {}, ["معدل خرج النموذج"]),
      el("dd", { class: "latin" }, [`${fmtNum((state.modelSr || 0) / 1000, 1)} kHz`]),
      el("dt", {}, ["حجم الكتلة"]),
      el("dd", { class: "latin" }, [`${state.blockMs} ms`]),
      el("dt", {}, ["طريقة f0"]),
      el("dd", { class: "latin" }, [state.settings.f0_method]),
      el("dt", {}, ["حالة محرك الجلسة"]),
      el("dd", { class: "latin" }, [state.serverStats?.engine_status || "—"]),
    ]));
    return;
  }
  box.append(el("p", { class: "muted small" }, [
    state.phase === "connecting" ? "جارٍ تجهيز الجلسة…" : "لا توجد جلسة نشطة.",
  ]));
}

/* ------------------------------ meters ------------------------------ */

function updateMeter(state, ui, which, rms) {
  const base = which === "in" ? "rt-meter-in" : "rt-meter-out";
  const db = 20 * Math.log10(rms + 1e-9);
  const norm = clamp((db + 60) / 60, 0, 1);
  const fill = document.getElementById(`${base}-fill`);
  const dbEl = document.getElementById(`${base}-db`);
  const peak = document.getElementById(`${base}-peak`);
  if (fill) fill.style.width = `${(norm * 100).toFixed(1)}%`;
  if (dbEl) dbEl.textContent = db <= -60 ? "-inf dB" : `${fmtNum(db, 1)} dB`;
  if (peak) {
    peak.style.insetInlineStart = `${(norm * 100).toFixed(1)}%`;
    peak.style.opacity = norm > 0.02 ? "0.7" : "0";
  }
}

function resetMeters(ui) {
  for (const base of ["rt-meter-in", "rt-meter-out"]) {
    const fill = document.getElementById(`${base}-fill`);
    const peak = document.getElementById(`${base}-peak`);
    const db = document.getElementById(`${base}-db`);
    if (fill) fill.style.width = "0%";
    if (peak) { peak.style.opacity = "0"; peak.style.insetInlineStart = "0%"; }
    if (db) db.textContent = "-inf dB";
  }
}

/* ------------------------------ latency panels ------------------------------ */

function renderServerStats(state, ui, msg) {
  const srv = msg.server || {};
  const set = (id, v) => {
    const node = document.getElementById(id);
    if (node) node.textContent = v;
  };
  set("rt-srv-min", fmtMs(srv.min_ms));
  set("rt-srv-avg", fmtMs(srv.avg_ms));
  set("rt-srv-p95", fmtMs(srv.p95_ms));
  set("rt-srv-max", fmtMs(srv.max_ms));
  set("rt-srv-bridge", fmtMs(msg.bridge_overhead_ms));
  if (typeof msg.blocks === "number") {
    state.serverBlocks = msg.blocks;
    ui.srvBlocks.textContent = String(msg.blocks);
  }
  if (state.phase === "streaming") renderStatusBox(state, ui);
  renderVerdict(state, ui, srv);
}

function renderClientStats(state, ui) {
  const h = state.rttHistory.slice();
  if (!h.length) return;
  const stats = miniStats(h);
  const set = (id, v) => {
    const node = document.getElementById(id);
    if (node) node.textContent = v;
  };
  set("rt-cli-min", fmtMs(stats.min));
  set("rt-cli-avg", fmtMs(stats.avg));
  set("rt-cli-p95", fmtMs(stats.p95));
  renderVerdict(state, ui, state.serverStats?.server || null);
}

function renderVerdict(state, ui, srv) {
  const box = ui.verdict;
  if (!box) return;
  const budget = state.blockMs || 160;
  const clientAvg = state.rttHistory.length ? miniStats(state.rttHistory).avg : null;
  const serverAvg = Number.isFinite(Number(srv?.avg_ms)) ? Number(srv.avg_ms) : null;
  if (clientAvg === null && serverAvg === null) {
    box.className = "verdict";
    box.textContent = "لا توجد قياسات بعد.";
    return;
  }
  const totalAvg = clientAvg ?? serverAvg;
  const good = totalAvg < budget;
  box.className = `verdict ${good ? "good" : "bad"}`;
  const parts = [];
  if (clientAvg !== null) parts.push(`متوسط مسار العميل ${fmtMs(clientAvg)}`);
  if (serverAvg !== null) parts.push(`متوسط استدلال الخادم ${fmtMs(serverAvg)}`);
  box.textContent = `${parts.join(" · ")} — ${good
    ? `ضمن ميزانية الكتلة (${budget} ms)`
    : `يتجاوز ميزانية الكتلة (${budget} ms) — سيتراكم التأخير تدريجيًا`}`;
}

function miniStats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return {
    min: sorted[0],
    max: sorted[n - 1],
    avg: sorted.reduce((a, b) => a + b, 0) / n,
    p95: sorted[Math.min(n - 1, Math.floor(0.95 * (n - 1)))],
  };
}

/* ------------------------------ DSP helpers ------------------------------ */

function rmsOf(floats) {
  let sum = 0;
  for (let i = 0; i < floats.length; i++) sum += floats[i] * floats[i];
  return Math.sqrt(sum / Math.max(1, floats.length));
}

function int16ToFloat(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) out[i] = int16[i] / 32768;
  return out;
}

function floatToInt16(floats) {
  const out = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    const v = clamp(floats[i], -1, 1);
    out[i] = v < 0 ? v * 32768 : v * 32767;
  }
  return out;
}

function resampleLinear(floats, fromRate, toRate) {
  if (fromRate === toRate) return floats;
  const outLen = Math.max(1, Math.round(floats.length * toRate / fromRate));
  const out = new Float32Array(outLen);
  const step = fromRate / toRate;
  for (let i = 0; i < outLen; i++) {
    const pos = i * step;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = floats[idx] ?? 0;
    const b = floats[idx + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

/* ------------------------------ worklet source ------------------------------ */

const WORKLET_CODE = `
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
