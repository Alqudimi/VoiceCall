/* ============================================================
   realtime.js — البث اللحظي: مايكروفون → AudioWorklet (16kHz int16)
   → WebSocket → خادم RVC → PCM → تشغيل مع مخزن امتصاص تذبذب،
   مع قياسات زمن الاستجابة (الخادم + العميل).

   هذه الصفحة تعمل كمكالمة صوتية حيّة: كل ما تقوله في الميكروفون
   يُحوَّل فورًا بخامة الصوت العراقي ويُبَث عبر مكبر الصوت في نفس اللحظة.

   لوحة "معاملات النبرة والصوت" تضم كل ما يُمرَّر للمحرك:
   انزياح النغمة، نسبة الفهرس، مزج الشدة، بوابة الصمت، طريقة f0،
   حجم الكتلة (block_ms)، حجم الخرج، وسماع الدخل — وكلها تُطبَّق
   لحظيًا أثناء البث دون قطع الجلسة.
   ============================================================ */

import {
  get, post, openRealtimeWS, toast, el, clear, icon, clamp, fmtNum, fmtMs, f0MethodSelect,
} from "../api.js";

export default { mount };

const INPUT_RATE = 16000;
const PREQUEUE_BLOCKS = 2;          // jitter absorption: 2 blocks preroll
const RTT_KEEP = 100;                // keep last 100 client round-trips
const STATS_EVERY_MS = 2000;
const PING_EVERY_MS = 5000;
const PUSH_MIN_MS = 60;              // throttle for live settings frames
const HF_POLL_MS = 2000;             // HF model download progress poll
const BLOCK_MS_CHOICES = [60, 80, 100, 128, 160, 200, 250, 320, 400, 500];
const SPEAKER_PROFILE_KEY = "ivs_speaker_profile";

/* كل معاملات النبرة القابلة للتعديل لحظيًا (تُرسل عبر إطار settings) */
const TONE_PARAMS = [
  {
    key: "pitch_shift", id: "pitch",
    label: "انزياح النغمة", latin: "Pitch · semitones",
    min: -24, max: 24, step: 1, def: 0,
    fmt: (v) => String(Math.round(v)),
    hint: "رفع أو خفض طبقة الصوت بأنصاف النغمات. الموجب = صوت أعلى (أنثوي أكثر)، السالب = أعمق. جرّب ±12 لتبديل الانطباع الصوتي.",
  },
  {
    key: "index_rate", id: "index",
    label: "نسبة الفهرس", latin: "Index Rate",
    min: 0, max: 1, step: 0.01, def: 0.88,
    fmt: (v) => Number(v).toFixed(2),
    hint: "قوة مزج سمات الصوت العراقي من فهرس الاسترجاع. 0.88 هو الموصى به لنموذج iraqi_voice؛ الخفض يجعل الصوت أقرب لصوتك الأصلي.",
  },
  {
    key: "rms_mix_rate", id: "rms",
    label: "مزج الشدة", latin: "RMS Mix Rate",
    min: 0, max: 1, step: 0.01, def: 0.25,
    fmt: (v) => Number(v).toFixed(2),
    hint: "1 = الحفاظ على شدة صوتك الأصلية (نبرة إيقاعية كما تتكلم)، 0 = اتباع شدة النموذج المستهدفة.",
  },
  {
    key: "silence_threshold_db", id: "gate",
    label: "بوابة الصمت (عزل البعيد)", latin: "Silence Gate · dB",
    min: -70, max: -20, step: 1, def: -36,
    fmt: (v) => `${Math.round(v)} dB`,
    hint: "عزل ذكي مع ثبات واستمرار (220ms) لمنع تقطيع الحروف: الأصوات الأهدأ من هذا المستوى (كالضجيج والأصوات البعيدة) تُرفض قبل التحويل. ارفعه (-32..-28) لعزل أشد للمتحدث القريب فقط.",
  },
];

/* ------------------------------ mount ------------------------------ */

async function mount(root) {
  const state = {
    phase: "idle", // idle | connecting | streaming | error
    models: [],
    info: null,
    hf: null,
    blockMs: 160,
    modelSr: null,
    settings: {
      pitch_shift: 0,
      index_rate: 0.88,
      rms_mix_rate: 0.25,
      f0_method: "rmvpe",
      silence_threshold_db: -36,
      strict_isolation: true,
      speaker_profile: loadSavedProfile(),
    },
    // client-side live controls
    volume: 1.0,          // 0..2 output gain
    monitor: false,       // hear the raw microphone alongside
    aec: true,            // browser echo cancellation + noise suppression (blocks speaker feedback & far noise)
    speakerGuard: false,  // strict half-duplex: mute mic while speaker plays
    enrolling: false,
    // live session objects
    ws: null,
    stream: null,
    inCtx: null,
    outCtx: null,
    outGain: null,
    monGain: null,
    micFilter: null,
    micLowpass: null,
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
    nextTimeMon: 0,
    speakerBusyUntil: 0,
    statsTimer: null,
    pingTimer: null,
    hfTimer: null,
    lastError: null,
    sessionId: 0,
    userTouched: false,   // don't override sliders the user already moved
  };

  root.replaceChildren(el("div", { class: "stack" }, [
    buildIntroCard(),
    buildHfCard(),
    buildControlsCard(),
    buildParamsCard(),
    buildStatusCard(),
    buildMetersCard(),
    buildLatencyCard(),
  ]));

  const ui = {
    model: root.querySelector("#rt-model"),
    micSelect: root.querySelector("#rt-mic"),
    outSelect: root.querySelector("#rt-out"),
    blockMs: root.querySelector("#rt-blockms"),
    btnStart: root.querySelector("#rt-start"),
    stateLine: root.querySelector("#rt-state"),
    statusBox: root.querySelector("#rt-status-box"),
    errorBox: root.querySelector("#rt-error"),
    infoBox: root.querySelector("#rt-info-box"),
    hfCard: root.querySelector("#rt-hf-card"),
    hfBox: root.querySelector("#rt-hf-box"),
    applied: root.querySelector("#rt-applied"),
    volume: root.querySelector("#rt-volume"),
    volumeVal: root.querySelector("#rt-volume-val"),
    monitor: root.querySelector("#rt-monitor"),
    aec: root.querySelector("#rt-aec"),
    speakerGuard: root.querySelector("#rt-speaker-guard"),
    strictIsolation: root.querySelector("#rt-strict-isolation"),
    btnEnroll: root.querySelector("#rt-enroll"),
    btnEnrollClear: root.querySelector("#rt-enroll-clear"),
    spkBadge: root.querySelector("#rt-spk-badge"),
    f0: root.querySelector("#rt-f0"),
    blocksSent: root.querySelector("#rt-blocks-sent"),
    blocksRecv: root.querySelector("#rt-blocks-recv"),
    srvBlocks: root.querySelector("#rt-srv-blocks"),
    verdict: root.querySelector("#rt-verdict"),
    pitch: {}, index: {}, rms: {}, gate: {},
  };

  wireLiveControls(ui, state);
  renderSpeakerBadge(state, ui);
  const offDeviceChange = wireDeviceSelectors(ui);
  ui.outSelect?.addEventListener("change", () => {
    if (state.outCtx) attachOutputSink(state.outCtx, ui, state);
  });
  ui.btnStart.addEventListener("click", () => {
    if (state.phase === "streaming" || state.phase === "connecting") stopSession(state, ui);
    else startSession(state, ui);
  });
  const btnTest = document.getElementById("rt-test-speaker");
  if (btnTest) btnTest.addEventListener("click", () => testSpeaker(state, ui));

  await Promise.all([loadModels(state, ui), loadInfo(state, ui), loadHf(state, ui)]);

  function cleanup() {
    if (offDeviceChange) offDeviceChange();
    if (state.hfTimer) { clearInterval(state.hfTimer); state.hfTimer = null; }
    teardownSession(state, ui);
  }
  return cleanup;
}

/* ------------------------------ static UI ------------------------------ */

function buildIntroCard() {
  return el("div", { class: "notice info" }, [
    el("span", { class: "n-icon" }, [icon("mic", 20)]),
    el("div", {}, [
      el("p", { style: "font-weight:700;margin:0 0 4px" }, ["مكالمة صوتية حيّة بخامة الصوت العراقي"]),
      el("p", { class: "small", style: "margin:0" }, [
        "اضغط «بدء البث اللحظي» وتكلم في الميكروفون — كل كلمة تُسمع تُحوَّل فورًا بنموذج الاستنساخ وتُبث عبر مكبر الصوت في نفس اللحظة، تمامًا كأنك في مكالمة. عدّل أي معامل من لوحة المعاملات أدناه أثناء الكلام وسيُطبَّق فورًا دون قطع الصوت. ",
        el("span", { class: "dim" }, ["نصيحة: استخدم سماعات رأس لتجنب ارتداد الصوت المحوّل إلى الميكروفون."]),
      ]),
    ]),
  ]);
}

/* بطاقة نموذج HuggingFace: حالة/تنزيل/تقدم — تُرسم ديناميكيًا */
function buildHfCard() {
  return el("div", { class: "card hidden", id: "rt-hf-card" }, [
    el("h2", { class: "card-title" }, [
      "نموذج الصوت العراقي — HuggingFace",
      el("span", { class: "sub latin" }, ["eng7mi/Iraqi-v3"]),
    ]),
    el("div", { id: "rt-hf-box" }, []),
  ]);
}

function buildControlsCard() {
  const blockOpts = BLOCK_MS_CHOICES.map((v) =>
    el("option", { value: String(v), ...(v === 160 ? { selected: true } : {}) }, [`${v} ms`])
  );
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
        el("label", { for: "rt-mic" }, ["ميكروفون الإدخال — ", el("span", { class: "opt-latin" }, ["Input"])]),
        el("select", { id: "rt-mic" }, [
          el("option", { value: "" }, ["الافتراضي للنظام"]),
        ]),
      ]),
      el("div", { class: "field" }, [
        el("label", { for: "rt-out" }, ["جهاز الإخراج — ", el("span", { class: "opt-latin" }, ["Output"])]),
        el("select", { id: "rt-out" }, [
          el("option", { value: "" }, ["الافتراضي للنظام"]),
        ]),
      ]),
      el("div", { class: "field" }, [
        el("label", { for: "rt-blockms" }, ["حجم الكتلة — ", el("span", { class: "opt-latin" }, ["block ms"])]),
        el("select", { id: "rt-blockms" }, blockOpts),
        el("span", { class: "hint" }, ["أقل = استجابة أسرع (يتطلب GPU أقوى). يُطبَّق عند بدء الجلسة."]),
      ]),
      el("div", { class: "field" }, [
        el("label", {}, ["حالة الخادم اللحظي"]),
        el("div", { class: "small dim", id: "rt-info-box" }, ["…"]),
      ]),
    ]),
    el("div", { class: "row mt", style: "flex-wrap:wrap;gap:8px;align-items:center" }, [
      el("span", { class: "badge info dot", id: "rt-spk-badge" }, ["قفل تلقائي على المتحدث"]),
      el("button", { class: "btn sm", type: "button", id: "rt-enroll" }, [
        icon("mic", 14), " تسجيل بصمة صوتي (3 ثوانٍ)",
      ]),
      el("button", { class: "btn sm ghost", type: "button", id: "rt-enroll-clear" }, [
        "مسح البصمة",
      ]),
    ]),
    el("button", { class: "btn primary block mt", type: "button", id: "rt-start" }, [
      icon("play", 16), " بدء البث اللحظي",
    ]),
    el("div", { class: "row mt", style: "gap:8px;flex-wrap:wrap" }, [
      el("button", { class: "btn sm ghost", type: "button", id: "rt-test-speaker" }, [
        icon("speaker", 14), " اختبار المكبر (نغمة قصيرة)",
      ]),
    ]),
    el("div", { id: "rt-playback-alert", class: "mt hidden" }, []),
    el("div", { id: "rt-error", class: "mt hidden" }, []),
  ]);
}

/* لوحة معاملات النبرة والصوت — كل ما يُمرَّر للمحرك مع تطبيق لحظي */
function buildParamsCard() {
  const rows = TONE_PARAMS.map((p) => el("div", { class: "slider-row" }, [
    el("div", { class: "slider-head" }, [
      el("label", { for: `rt-${p.id}` }, [p.label, el("span", { class: "opt-latin" }, [` ${p.latin}`])]),
      el("span", { class: "slider-val", id: `rt-${p.id}-val` }, [p.fmt(p.def)]),
      el("input", {
        type: "number", class: "num-mini", id: `rt-${p.id}-num`,
        min: String(p.min), max: String(p.max), step: String(p.step), value: String(p.def),
      }),
    ]),
    el("input", { type: "range", id: `rt-${p.id}`, min: String(p.min), max: String(p.max), step: String(p.step), value: String(p.def) }),
    el("div", { class: "param-hint" }, [p.hint]),
  ]));

  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "معاملات النبرة والصوت",
      el("span", { class: "sub latin" }, ["Voice Parameters · live"]),
    ]),
    el("div", { class: "row" }, [
      el("span", { class: "small dim" }, ["كل تعديل هنا يُطبَّق على البث فورًا — تابع الكلام وجرّب بحرية."]),
      el("span", { class: "applied-flash", id: "rt-applied" }, ["طُبِّقت الإعدادات على البث"]),
    ]),
    el("div", { class: "preset-row" }, [
      el("button", { class: "btn sm", type: "button", id: "rt-preset-best" }, ["الإعداد الموصى به للعراقي"]),
      el("button", { class: "btn sm ghost", type: "button", id: "rt-preset-neutral" }, ["محايد"]),
    ]),
    ...rows,
    el("div", { class: "field mt" }, [
      el("label", { for: "rt-f0" }, ["طريقة استخراج الطبقة — ", el("span", { class: "latin" }, ["f0 method"])]),
      f0MethodSelect("rmvpe", "rt-f0"),
      el("span", { class: "hint" }, ["rmvpe الأدق للعربية، fcpe أخف وأسرع، pm للمعالجات الضعيفة. تُطبَّق فورًا على الكتلة التالية."]),
    ]),
    el("div", { class: "slider-row mt" }, [
      el("div", { class: "slider-head" }, [
        el("label", { for: "rt-volume" }, ["حجم الخرج", el("span", { class: "opt-latin" }, ["Output Gain"])]),
        el("span", { class: "slider-val", id: "rt-volume-val" }, ["100%"]),
      ]),
      el("input", { type: "range", id: "rt-volume", min: "0", max: "2", step: "0.05", value: "1" }),
      el("div", { class: "param-hint" }, ["مستوى صوت مكبر الصوت للصوت المحوّل — يُطبَّق لحظيًا داخل المتصفح دون أي تأخير إضافي."]),
    ]),
    el("label", { class: "check" }, [
      el("input", { type: "checkbox", id: "rt-strict-isolation", checked: true }),
      " تشديد عزل المتحدث فقط (يمنع استنساخ أي شخص آخر أو ضوضاء أو صدى تلقائيًا) ",
      el("span", { class: "opt-latin" }, ["Speaker Lock + Speech VAD"]),
    ]),
    el("label", { class: "check" }, [
      el("input", { type: "checkbox", id: "rt-aec", checked: true }),
      " عزل الصدى والأصوات البعيدة (مستحسن عند استخدام مكبر الصوت) ",
      el("span", { class: "opt-latin" }, ["AEC + Noise Suppression"]),
    ]),
    el("label", { class: "check" }, [
      el("input", { type: "checkbox", id: "rt-speaker-guard" }),
      " حماية قصوى ضد إعادة استنساخ صوت المكبر (يكتم الميكروفون أثناء نطق المكبر) ",
      el("span", { class: "opt-latin" }, ["Anti-Feedback"]),
    ]),
    el("label", { class: "check" }, [
      el("input", { type: "checkbox", id: "rt-monitor" }),
      " سماع صوتي الأصلي مع المحوّل ",
      el("span", { class: "opt-latin" }, ["Monitor Input"]),
    ]),
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
      ui.model.append(el("option", { value: "" }, ["لا توجد نماذج f0 — نزّل نموذج HuggingFace أعلاه أو ارفع نموذجًا"]));
      ui.model.disabled = true;
      ui.btnStart.disabled = true;
      return;
    }
    for (const m of state.models) ui.model.append(el("option", { value: m.name }, [m.name]));
    const preferred =
      (state.hf && state.hf.default_model) ||
      state.info?.default_model ||
      state.info?.loaded_model?.name;
    if (preferred && state.models.some((m) => m.name === preferred)) {
      ui.model.value = preferred;
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
    if (ui.gate?.slider && !state.userTouched) {
      const gateDef = TONE_PARAMS.find((p) => p.id === "gate");
      if (gateDef) {
        const v = clamp(Number(state.settings.silence_threshold_db), gateDef.min, gateDef.max);
        state.settings.silence_threshold_db = v;
        ui.gate.slider.value = String(v);
        ui.gate.num.value = String(v);
        ui.gate.val.textContent = gateDef.fmt(v);
      }
    }
    if (state.info.f0_method) {
      state.settings.f0_method = state.info.f0_method;
      ui.f0.value = state.info.f0_method;
    }
    if (state.info.recommended && !state.userTouched) {
      applyRecommended(state, ui, state.info.recommended, { quiet: true });
    }
    if (ui.blockMs) ui.blockMs.value = String(state.blockMs);
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

/* حالة نموذج HuggingFace + تقدم التنزيل */
async function loadHf(state, ui) {
  try {
    state.hf = await get("/models/hf/status");
  } catch {
    state.hf = null; // integration absent — hide the card entirely
  }
  renderHf(state, ui);
  scheduleHfPoll(state, ui);
  if (state.hf && state.hf.default_model && !ui.model.value) {
    await loadModels(state, ui);
  } else if (state.hf && state.hf.default_model && state.models.length && ui.model.value !== state.hf.default_model
             && state.models.some((m) => m.name === state.hf.default_model) && !state.userTouched) {
    ui.model.value = state.hf.default_model;
  }
}

function scheduleHfPoll(state, ui) {
  if (state.hfTimer) { clearInterval(state.hfTimer); state.hfTimer = null; }
  const active = state.hf && state.hf.download.state === "downloading";
  if (!active) return;
  state.hfTimer = setInterval(async () => {
    try {
      state.hf = await get("/models/hf/status");
      renderHf(state, ui);
      if (state.hf.download.state !== "downloading") {
        clearInterval(state.hfTimer);
        state.hfTimer = null;
        if (state.hf.download.state === "ready" || state.hf.model.present) {
          toast("اكتمل تنزيل نموذج الصوت العراقي وأصبح جاهزًا.", "success");
          await loadModels(state, ui);
          if (state.hf.default_model) ui.model.value = state.hf.default_model;
          ui.btnStart.disabled = state.models.length === 0;
        } else if (state.hf.download.state === "error") {
          toast(`فشل تنزيل النموذج: ${state.hf.download.error || "خطأ غير معروف"}`, "error", 10000);
        }
      }
    } catch { /* transient poll failure — keep trying */ }
  }, HF_POLL_MS);
}

function renderHf(state, ui) {
  const card = ui.hfCard;
  const box = ui.hfBox;
  if (!card || !box) return;
  if (!state.hf) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  clear(box);
  const hf = state.hf;

  // جاهز؟ بطاقة مختصرة
  if (hf.model.present && hf.download.state !== "downloading") {
    box.append(el("div", { class: "row" }, [
      el("span", { class: `badge ${hf.model.present ? "ok" : "warn"} dot` }, [
        hf.model.present ? `النموذج الافتراضي «${hf.local_name}» جاهز` : "النموذج غير منزّل",
      ]),
      hf.index.present ? el("span", { class: "badge ok" }, ["الفهرس مرفق"]) : null,
      hf.is_engine_default ? el("span", { class: "badge info" }, ["محمّل في المحرك"]) : null,
    ]));
    return;
  }

  // قيد التنزيل — أشرطة تقدم لكل ملف
  if (hf.download.state === "downloading") {
    box.append(el("div", { class: "row" }, [
      el("span", { class: "badge loading dot" }, ["جارٍ تنزيل النموذج من HuggingFace…"]),
    ]));
    for (const meta of Object.values(hf.download.files || {})) {
      const total = meta.total_bytes || 0;
      const done = meta.received_bytes || 0;
      const pct = total ? Math.min(100, (done / total) * 100) : 0;
      const statusLabel = {
        pending: "بانتظار", downloading: "ينزّل", done: "اكتمل",
        skipped: "موجود مسبقًا", failed: "فشل",
      }[meta.status] || meta.status;
      box.append(el("div", { class: "mt" }, [
        el("div", { class: "hf-file-row" }, [
          el("span", {}, [meta.label]),
          el("span", { class: "latin dim" }, [
            `${statusLabel} · ${fmtNum(done / (1024 * 1024), 1)} / ${fmtNum(total / (1024 * 1024), 1)} MB (${fmtNum(pct, 0)}%)`,
          ]),
        ]),
        el("div", { class: "progress" }, [el("div", { class: "bar", style: `width:${pct.toFixed(1)}%` })]),
      ]));
    }
    return;
  }

  // مفقود — زر تنزيل أو تحذير التوكن
  if (!hf.token_configured) {
    box.append(el("div", { class: "notice warn" }, [
      el("span", { class: "n-icon" }, [icon("alert", 18)]),
      el("div", {}, [
        el("p", { style: "margin:0 0 6px" }, [
          "النموذج الافتراضي (",
          el("span", { class: "latin" }, [hf.repo_id]),
          ") مستودع خاص — أضف توكن الوصول كمتغير بيئة ",
          el("span", { class: "latin" }, ["RVC_HF_TOKEN"]),
          " في ملف ",
          el("span", { class: "latin" }, [".env"]),
          " ثم أعد تشغيل الخادم.",
        ]),
        el("p", { class: "small dim", style: "margin:0" }, [
          "انسخ القالب: ",
          el("span", { class: "latin" }, ["cp .env.example .env"]),
        ]),
      ]),
    ]));
    return;
  }
  if (hf.download.state === "error") {
    box.append(el("div", { class: "notice danger" }, [
      el("span", { class: "n-icon" }, [icon("alert", 18)]),
      el("p", { style: "margin:0" }, [`فشل التنزيل: ${hf.download.error || "خطأ غير معروف"}`]),
    ]));
  }
  const btn = el("button", { class: "btn primary", type: "button" }, [
    icon("download", 15), " تنزيل نموذج الصوت العراقي الآن",
  ]);
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "جارٍ البدء…";
    try {
      const res = await post("/models/hf/download");
      if (res && res.started) {
        toast("بدأ التنزيل في الخلفية — سيعمل شريط التقدم هنا تلقائيًا.", "success");
        state.hf = await get("/models/hf/status");
        renderHf(state, ui);
        scheduleHfPoll(state, ui);
      } else {
        toast(res?.message || "التنزيل قيد التنفيذ بالفعل.", "warn");
      }
    } catch (err) {
      toast(err.message, "error");
      btn.disabled = false;
      btn.replaceChildren(icon("download", 15), document.createTextNode(" تنزيل نموذج الصوت العراقي الآن"));
    }
  });
  box.append(el("div", { class: "row mt" }, [btn]));
  box.append(el("p", { class: "small dim mt", style: "margin-bottom:0" }, [
    "المصدر: ",
    el("span", { class: "latin hf-url-box" }, [hf.model_url]),
  ]));
}

/* ------------------------------ live controls ------------------------------ */

function wireLiveControls(ui, state) {
  for (const p of TONE_PARAMS) {
    const slider = document.getElementById(`rt-${p.id}`);
    const num = document.getElementById(`rt-${p.id}-num`);
    const val = document.getElementById(`rt-${p.id}-val`);
    if (!slider || !num) continue;
    ui[p.id] = { slider, num, val };
    const apply = (raw, from) => {
      state.userTouched = true;
      let v = clamp(Number(raw), p.min, p.max);
      if (Number.isNaN(v)) v = p.def;
      v = Math.round(v / p.step) * p.step;
      v = Number(v.toFixed(4));
      state.settings[p.key] = v;
      if (from !== "slider") slider.value = String(v);
      if (from !== "num") num.value = String(v);
      val.textContent = p.fmt(v);
      pushSettings(state, ui);
    };
    slider.addEventListener("input", () => apply(slider.value, "slider"));
    num.addEventListener("change", () => apply(num.value, "num"));
  }

  ui.f0.addEventListener("change", () => {
    state.userTouched = true;
    state.settings.f0_method = ui.f0.value;
    pushSettings(state, ui);
  });

  // حجم الخرج — يُطبَّق داخل المتصفح فورًا (بدون دورة خادم)
  ui.volume.addEventListener("input", () => {
    state.volume = Number(ui.volume.value);
    ui.volumeVal.textContent = `${Math.round(state.volume * 100)}%`;
    if (state.outGain) state.outGain.gain.value = state.volume;
    if (state.volume > 0.001 && state.zeroVolWarned) {
      state.zeroVolWarned = false;          // volume restored — drop the alert
      const box = document.getElementById("rt-playback-alert");
      if (box && state.outCtx && state.outCtx.state === "running") {
        clear(box);
        box.classList.add("hidden");
      }
    }
  });

  // سماع الدخل — بوابة صوتية تنفتح لحظيًا
  ui.monitor.addEventListener("change", () => {
    state.monitor = ui.monitor.checked;
    if (state.monGain) state.monGain.gain.value = state.monitor ? 1.0 : 0.0;
    if (state.monitor) toast("سماع الدخل مفعّل — ستسمع صوتك الأصلي مع المحوّل. استخدم سماعات لتجنب الارتداد.", "info");
  });

  if (ui.aec) {
    ui.aec.addEventListener("change", () => {
      state.aec = ui.aec.checked;
      if (state.phase === "streaming") {
        toast("يُطبَّق تغيير عزل الصدى عند إعادة بدء الجلسة.", "info", 5000);
      }
    });
  }

  if (ui.speakerGuard) {
    ui.speakerGuard.addEventListener("change", () => {
      state.speakerGuard = ui.speakerGuard.checked;
    });
  }

  if (ui.strictIsolation) {
    ui.strictIsolation.addEventListener("change", () => {
      state.settings.strict_isolation = ui.strictIsolation.checked;
      pushSettings(state, ui);
      renderSpeakerBadge(state, ui);
    });
  }

  if (ui.btnEnroll) {
    ui.btnEnroll.addEventListener("click", () => enrollSpeakerVoice(state, ui));
  }
  if (ui.btnEnrollClear) {
    ui.btnEnrollClear.addEventListener("click", () => {
      saveSpeakerProfile(null);
      state.settings.speaker_profile = null;
      pushSettings(state, ui);
      renderSpeakerBadge(state, ui);
      toast("تم مسح بصمة الصوت — سيعمل القفل التلقائي على أول متحدث قريب.", "info", 4000);
    });
  }

  // حجم الكتلة — يُطبَّق عند بدء الجلسة القادمة
  ui.blockMs.addEventListener("change", () => {
    state.blockMs = Number(ui.blockMs.value) || 160;
    if (state.phase === "streaming") {
      toast(`حجم الكتلة الجديد (${state.blockMs} ms) سيُطبَّق عند إعادة بدء البث.`, "info", 6000);
    }
  });

  // الإعدادات الجاهزة
  document.getElementById("rt-preset-best").addEventListener("click", () => {
    const rec = state.hf?.recommended || state.info?.recommended || {
      pitch_shift: 0, index_rate: 0.88, rms_mix_rate: 0.25, f0_method: "rmvpe",
    };
    applyRecommended(state, ui, rec, { quiet: false });
  });
  document.getElementById("rt-preset-neutral").addEventListener("click", () => {
    applyRecommended(state, ui, {
      pitch_shift: 0, index_rate: 0, rms_mix_rate: 0,
      f0_method: state.settings.f0_method,
      silence_threshold_db: state.settings.silence_threshold_db,
    }, { quiet: false });
  });
}

function applyRecommended(state, ui, rec, { quiet = false } = {}) {
  if (!rec) return;
  for (const p of TONE_PARAMS) {
    if (rec[p.key] === undefined || rec[p.key] === null) continue;
    const v = clamp(Number(rec[p.key]), p.min, p.max);
    state.settings[p.key] = v;
    const refs = ui[p.id];
    if (refs) {
      refs.slider.value = String(v);
      refs.num.value = String(v);
      refs.val.textContent = p.fmt(v);
    }
  }
  if (rec.f0_method) {
    state.settings.f0_method = rec.f0_method;
    ui.f0.value = rec.f0_method;
  }
  pushSettings(state, ui);
  if (!quiet) toast("طُبِّق الإعداد الموصى به لنموذج الصوت العراقي.", "success", 2500);
}

/* إرسال إطار settings مع خنق خفيف كي لا يُغرق الخادم أثناء السحب المتواصل */
function pushSettings(state, ui) {
  if (state.phase === "streaming" && state.ws) {
    state._pushQueued = true;
    if (state._pushTimer) return;
    state._pushTimer = setTimeout(() => {
      state._pushTimer = null;
      if (state._pushQueued && state.phase === "streaming" && state.ws) {
        state._pushQueued = false;
        state.ws.sendText({ type: "settings", settings: settingsPayload(state) });
      }
    }, PUSH_MIN_MS);
  }
}

function settingsPayload(state) {
  return {
    pitch_shift: state.settings.pitch_shift,
    index_rate: state.settings.index_rate,
    f0_method: state.settings.f0_method,
    rms_mix_rate: state.settings.rms_mix_rate,
    silence_threshold_db: state.settings.silence_threshold_db,
    strict_isolation: state.settings.strict_isolation,
    speaker_profile: state.settings.speaker_profile || null,
  };
}

/* ------------------------------ device selection ------------------------------ */

function wireDeviceSelectors(ui) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return null;
  const refresh = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      fillSelect(ui.micSelect, devices.filter((d) => d.kind === "audioinput"), "ميكروفون");
      fillSelect(ui.outSelect, devices.filter((d) => d.kind === "audiooutput"), "مخرج");
    } catch { /* permission not granted yet — default device used */ }
  };
  refresh();
  navigator.mediaDevices.addEventListener?.("devicechange", refresh);
  return () => navigator.mediaDevices.removeEventListener?.("devicechange", refresh);
}

function fillSelect(select, devices, label) {
  if (!select) return;
  const current = select.value;
  const options = [el("option", { value: "" }, [`الافتراضي للنظام`])];
  for (const d of devices) {
    const name = d.label || `${label} ${d.deviceId.slice(0, 8)}`;
    options.push(el("option", { value: d.deviceId }, [name]));
  }
  clear(select);
  select.append(...options);
  select.value = options.some((o) => o.value === current) ? current : "";
}

async function attachOutputSink(outCtx, ui, state = null) {
  const deviceId = ui.outSelect ? ui.outSelect.value : "";
  try {
    if (typeof outCtx.setSinkId === "function") {
      await outCtx.setSinkId(deviceId || "");
      if (state) state.sinkFailed = false;
    } else {
      return; // context already created; sink selection unavailable
    }
  } catch (err) {
    // The selected output device is gone/invalid — fall back to the system
    // default so playback ALWAYS has a valid destination, and tell the user.
    if (state) state.sinkFailed = true;
    toast(`تعذّر اختيار جهاز الإخراج (${err.message}) — سيستخدم الافتراضي.`, "warn");
    try { await outCtx.setSinkId(""); } catch { /* default already active */ }
  }
}

/* ------------------------ playback health watchdog ------------------------ */

/** Play a short audible tone through the exact playback graph (outGain →
 *  destination). Diagnoses in one click: suspended context, wrong/missing
 *  output device, zero volume, or OS-level mute. */
async function testSpeaker(state, ui) {
  const ok = await resumeOutCtx(state, ui);
  const ctx = state.outCtx;
  if (!ctx || !ok || ctx.state !== "running") {
    state.outCtxSuspendedWarned = true;
    renderPlaybackAlert(state, ui);
    toast("فشل اختبار المكبر: سياق الصوت معلّق — اضغط زر «تفعيل الصوت» أو انقر بالصفحة أولًا.", "error", 9000);
    return;
  }
  const sr = ctx.sampleRate;
  const dur = 0.5;
  const n = Math.round(sr * dur);
  const buf = ctx.createBuffer(1, n, sr);
  const data = buf.getChannelData(0);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const env = Math.min(1, t / 0.03) * Math.max(0, 1 - Math.max(0, t - (dur - 0.05)) / 0.05);
    data[i] = 0.35 * Math.sin(2 * Math.PI * 440 * t) * env;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(state.outGain);
  src.start();
  toast("نغمة الاختبار تعمل الآن عبر نفس مسار الصوت — إن لم تسمعها فالمشكلة في جهاز/مستوى الإخراج وليست في البث.", "info", 8000);
}

/** Resume the playback context; verify it actually left "suspended".
 *  A silent resume() failure (autoplay policy lost after the permission
 *  dialog) is the #1 cause of "mic meter moves but speaker silent". */
async function resumeOutCtx(state, ui) {
  const ctx = state.outCtx;
  if (!ctx) return false;
  if (ctx.state === "running") return true;
  try { await ctx.resume(); } catch { /* gesture-bound */ }
  await new Promise((r) => setTimeout(r, 60));
  return ctx.state === "running";
}

/** One-time warning + retry when the playback context is stuck suspended. */
async function checkPlaybackHealth(state, ui) {
  const ctx = state.outCtx;
  if (!ctx || state.phase !== "streaming") return;
  if (ctx.state !== "suspended") return;
  const ok = await resumeOutCtx(state, ui);
  if (!ok && !state.outCtxSuspendedWarned) {
    state.outCtxSuspendedWarned = true;
    renderPlaybackAlert(state, ui);
  }
}

/** Visible alert card for playback problems (suspended ctx / zero volume / bad sink). */
function renderPlaybackAlert(state, ui) {
  const box = document.getElementById("rt-playback-alert");
  if (!box) return;
  clear(box);
  box.classList.remove("hidden");
  const ctx = state.outCtx;
  const issues = [];
  if (ctx && ctx.state === "suspended") issues.push("سياق الصوت في المتصفح معلّق (autoplay) — اضغط زر «تفعيل الصوت»");
  if (state.volume <= 0.001) issues.push("مستوى حجم الخرج = 0 — ارفعه من لوحة المعاملات");
  if (state.sinkFailed) issues.push("جهاز الإخراج المختار غير صالح — عاد الصوت للجهاز الافتراضي");
  if (!issues.length) { box.classList.add("hidden"); return; }
  box.append(el("div", { class: "notice warn" }, [
    el("span", { class: "n-icon" }, [icon("alert", 18)]),
    el("div", {}, [
      el("p", { style: "font-weight:700;margin:0 0 4px" }, ["⚠ الصوت المحوّل لا يُسمع الآن"]),
      ...issues.map((t) => el("p", { class: "small", style: "margin:2px 0" }, [t])),
      el("button", {
        class: "btn primary sm mt", type: "button",
        onclick: async () => {
          const ok = await resumeOutCtx(state, ui);
          if (ok) {
            state.outCtxSuspendedWarned = false;
            box.classList.add("hidden");
            toast("تفعّل الصوت — أكمل الكلام وستسمع الخامة فورًا.", "success");
          } else {
            toast("ما زال السياق معلّقًا — انقر في أي مكان بالصفحة ثم أعد المحاولة، أو استخدم متصفحًا آخر.", "error", 9000);
          }
        },
      }, ["تفعيل الصوت"]),
    ]),
  ]));
}

/* ------------------------------ session lifecycle ------------------------------ */

async function startSession(state, ui) {
  if (state.phase === "connecting" || state.phase === "streaming") return;
  const modelName = ui.model.value;
  if (!modelName) {
    toast("اختر نموذجًا يدعم f0 أولًا — أو نزّل النموذج الافتراضي من أعلى الصفحة.", "warn");
    return;
  }
  state.lastError = null;
  state.blocksSent = 0;
  state.blocksReceived = 0;
  state.serverBlocks = 0;
  state.rttHistory = [];
  state.serverStats = null;
  state.sinkFailed = false;
  state.zeroVolWarned = false;
  const pbAlert = document.getElementById("rt-playback-alert");
  if (pbAlert) { clear(pbAlert); pbAlert.classList.add("hidden"); }
  if (ui.blocksSent) ui.blocksSent.textContent = "0";
  if (ui.blocksRecv) ui.blocksRecv.textContent = "0";
  if (ui.srvBlocks) ui.srvBlocks.textContent = "0";
  const sessionId = ++state.sessionId;
  const alive = () => state.sessionId === sessionId && (state.phase === "connecting" || state.phase === "streaming");
  setPhase(state, ui, "connecting", `جارٍ الاتصال والتحميل — ${modelName}`);

  // Create playback context synchronously under the user click gesture so
  // autoplay policies never keep it suspended after async permission awaits.
  state.outCtx = new AudioContext();
  state.outCtxSuspendedWarned = false;   // one-time user warning when playback is blocked
  state.outCtx.setSinkId?.("")?.catch?.(() => {});
  // Aggressive initial resume + watchdog: some browsers drop the user-gesture
  // status after the async mic-permission dialog, leaving the context
  // suspended => mic meter moves but the speaker plays nothing.
  resumeOutCtx(state, ui);
  if (state.outCtx.state === "suspended") {
    state.outCtx.resume().then(() => checkPlaybackHealth(state, ui)).catch(() => {});
  }
  state.nextTime = 0;
  state.nextTimeMon = 0;
  state.speakerBusyUntil = 0;
  state.outGain = state.outCtx.createGain();
  state.outGain.gain.value = state.volume;
  state.outGain.connect(state.outCtx.destination);
  state.monGain = state.outCtx.createGain();
  state.monGain.gain.value = state.monitor ? 1.0 : 0.0;
  state.monGain.connect(state.outCtx.destination);

  // 1) microphone (selected device if any; AEC + NS block speaker feedback & room noise,
  //    while autoGainControl stays off so distant voices are NOT amplified)
  let stream;
  const micId = ui.micSelect ? ui.micSelect.value : "";
  const audioConstraints = {
    echoCancellation: Boolean(state.aec),
    noiseSuppression: Boolean(state.aec),
    autoGainControl: false,
    channelCount: 1,
  };
  if (micId) audioConstraints.deviceId = { exact: micId };
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
  } catch (err) {
    fail(state, ui, `تعذّر الوصول إلى الميكروفون: ${err.name === "NotAllowedError" ? "الإذن مرفوض من المتصفح" : err.message || err.name}`);
    return;
  }
  if (!alive()) { for (const t of stream.getAudioTracks()) t.stop(); teardownSession(state, ui); return; }
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

  // 3) ensure playback context resumed & output sink attached
  resumeOutCtx(state, ui);
  await attachOutputSink(state.outCtx, ui, state);
  if (!alive()) { teardownSession(state, ui); return; }

  // 4) worklet — بحجم الكتلة المختار (block_ms)
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
  if (typeof state.inCtx.createBiquadFilter === "function") {
    const hpf = state.inCtx.createBiquadFilter();
    hpf.type = "highpass";
    hpf.frequency.value = 100;
    hpf.Q.value = 0.707;
    const lpf = state.inCtx.createBiquadFilter();
    lpf.type = "lowpass";
    lpf.frequency.value = 7200;
    lpf.Q.value = 0.707;
    state.micFilter = hpf;
    state.micLowpass = lpf;
    state.micSource.connect(hpf);
    hpf.connect(lpf);
    lpf.connect(state.workletNode);
  } else {
    state.micFilter = null;
    state.micLowpass = null;
    state.micSource.connect(state.workletNode);
  }

  const track = stream.getAudioTracks()[0];
  if (track) track.addEventListener("ended", () => {
    if (state.phase === "streaming" || state.phase === "connecting") {
      fail(state, ui, "انقطع الميكروفون (توقف المسار) — تحقق من الجهاز ثم أعد المحاولة.");
    }
  });

  state.workletNode.port.onmessage = (ev) => {
    const block = ev.data; // Float32Array at input-ctx rate
    if (!(block instanceof Float32Array)) return;

    // Anti-feedback guard (when enabled): mute mic capture while speaker is playing
    const speakerActive = Boolean(
      state.speakerGuard && state.outCtx && state.outCtx.currentTime < state.speakerBusyUntil
    );
    const activeBlock = speakerActive ? new Float32Array(block.length) : block;

    const rms = rmsOf(activeBlock);
    updateMeter(state, ui, "in", rms);

    // سماع الدخل: بث الصوت الأصلي عبر مكبر الصوت في نفس اللحظة
    if (state.monitor && state.outCtx && state.monGain && !speakerActive) {
      const ctx = state.outCtx;
      if (ctx.state === "suspended") ctx.resume().catch(() => {});
      const outRate = ctx.sampleRate;
      let mon = activeBlock;
      if (inRate !== outRate) mon = resampleLinear(activeBlock, inRate, outRate);
      const buf = ctx.createBuffer(1, mon.length, outRate);
      buf.copyToChannel(mon, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(state.monGain);
      const blockSec = buf.duration;
      if (state.nextTimeMon === 0 || state.nextTimeMon < ctx.currentTime + 0.004) {
        state.nextTimeMon = ctx.currentTime + PREQUEUE_BLOCKS * blockSec;
      }
      src.start(state.nextTimeMon);
      state.nextTimeMon += blockSec;
    }

    let pcm = activeBlock;
    if (inRate !== INPUT_RATE) pcm = resampleLinear(activeBlock, inRate, INPUT_RATE);
    const int16 = floatToInt16(pcm);
    if (state.ws && state.ws.readyState() === WebSocket.OPEN) {
      state.pendingTs.push(performance.now());
      if (state.pendingTs.length > 8) state.pendingTs.shift();
      state.blocksSent += 1;
      if (ui.blocksSent) ui.blocksSent.textContent = String(state.blocksSent);
      state.ws.sendBinary(int16.buffer);
    }
  };

  // 5) websocket — مع تمرير حجم الكتلة المختار
  const ws = openRealtimeWS();
  state.ws = ws;

  ws.onopen = () => {
    ws.sendText({
      type: "start",
      model: modelName,
      settings: settingsPayload(state),
      block_ms: state.blockMs,
    });
  };

  ws.ontext = (msg) => {
    if (!msg || !msg.type) return;
    if (msg.type === "started") {
      if (!alive()) return;
      state.modelSr = msg.sample_rate;
      if (msg.block_ms) {
        state.blockMs = msg.block_ms;
        if (ui.blockMs) ui.blockMs.value = String(msg.block_ms);
      }
      setPhase(state, ui, "streaming", `بث نشط — ${msg.model || modelName} · خرج ${fmtNum((msg.sample_rate || 0) / 1000, 1)} kHz · ميزانية ${msg.latency_budget_ms || state.blockMs} ms`);
      toast("بدأ البث اللحظي — تكلم في الميكروفون وستسمع صوتك بالخامة العراقية فورًا.", "success");
      ui.model.disabled = true;
      state.statsTimer = setInterval(() => ws.sendText({ type: "stats" }), STATS_EVERY_MS);
      state.pingTimer = setInterval(() => ws.sendText({ type: "ping" }), PING_EVERY_MS);
      ws.sendText({ type: "stats" });
    } else if (msg.type === "settings_applied") {
      flashApplied(ui);
    } else if (msg.type === "stats") {
      state.serverStats = msg;
      renderServerStats(state, ui, msg);
      renderSpeakerBadge(state, ui, msg);
    } else if (msg.type === "pong") {
      state.serverBlocks = msg.blocks ?? 0;
      if (ui.srvBlocks) ui.srvBlocks.textContent = String(state.serverBlocks);
      renderSpeakerBadge(state, ui, msg);
    } else if (msg.type === "error") {
      fail(state, ui, `خطأ من الخادم: ${msg.message || "غير معروف"}`);
    }
  };

  ws.onbinary = (buffer) => {
    if (!alive() || !state.outCtx) return;
    const int16 = new Int16Array(buffer);
    if (!int16.length) return;
    let floats = int16ToFloat(int16);
    const ctx = state.outCtx;
    if (ctx.state === "suspended") {
      // a suspended context freezes currentTime: the block would be scheduled
      // into a future that never arrives — resume first, then re-check health
      ctx.resume().then(() => checkPlaybackHealth(state, ui)).catch(() => {});
    }
    const outRate = ctx.sampleRate;
    if (state.modelSr && outRate !== state.modelSr) {
      floats = resampleLinear(floats, state.modelSr, outRate);
    }
    const rms = rmsOf(floats);
    updateMeter(state, ui, "out", rms);

    const blockSec = floats.length / outRate;
    // Drop excess queued silent blocks when jitter buffer grows > 0.65s so latency self-heals
    if (state.nextTime > 0 && state.nextTime - ctx.currentTime > Math.max(0.65, 4 * blockSec) && rms < 0.004) {
      if (state.pendingTs.length) state.pendingTs.shift();
      state.blocksReceived += 1;
      if (ui.blocksRecv) ui.blocksRecv.textContent = String(state.blocksReceived);
      return;
    }

    let needsFadeIn = false;
    if (state.nextTime === 0) {
      state.nextTime = ctx.currentTime + Math.max(0.18, PREQUEUE_BLOCKS * blockSec);
      needsFadeIn = true;
    } else if (state.nextTime < ctx.currentTime + 0.004) {
      const gap = ctx.currentTime - state.nextTime;
      state.nextTime = ctx.currentTime + (gap > 0.35
        ? Math.max(0.18, PREQUEUE_BLOCKS * blockSec)
        : Math.max(0.06, 0.75 * blockSec));
      needsFadeIn = true;
    }
    if (needsFadeIn) {
      const fadeSamples = Math.min(floats.length, Math.round(outRate * 0.004));
      for (let i = 0; i < fadeSamples; i++) floats[i] *= i / fadeSamples;
    }

    const audioBuffer = ctx.createBuffer(1, floats.length, outRate);
    audioBuffer.copyToChannel(floats, 0);
    const src = ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(state.outGain);
    src.start(state.nextTime);
    if (rms > 0.012 && state.volume > 0.02) {
      state.speakerBusyUntil = Math.max(state.speakerBusyUntil, state.nextTime + blockSec + 0.14);
    }
    state.nextTime += blockSec;
    // if the context can never run, surface the problem instead of silence
    if (ctx.state === "suspended") checkPlaybackHealth(state, ui);
    if (state.volume <= 0.001 && !state.zeroVolWarned) {
      state.zeroVolWarned = true;
      renderPlaybackAlert(state, ui);
    }

    // client round-trip: oldest pending send timestamp ↔ its scheduled playback
    if (state.pendingTs.length) {
      const sentAt = state.pendingTs.shift();
      const rtt = performance.now() - sentAt;
      state.rttHistory.push(rtt);
      if (state.rttHistory.length > RTT_KEEP) state.rttHistory.shift();
      renderClientStats(state, ui);
    }
    state.blocksReceived += 1;
    if (ui.blocksRecv) ui.blocksRecv.textContent = String(state.blocksReceived);
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

/** وميض هادئ بدل التوست المتكرر عند كل تطبيق إعدادات أثناء السحب */
function flashApplied(ui) {
  const node = ui.applied;
  if (!node) return;
  node.classList.add("on");
  clearTimeout(node._t);
  node._t = setTimeout(() => node.classList.remove("on"), 900);
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
  if (state._pushTimer) { clearTimeout(state._pushTimer); state._pushTimer = null; }
  state._pushQueued = false;
  if (state.statsTimer) { clearInterval(state.statsTimer); state.statsTimer = null; }
  if (state.pingTimer) { clearInterval(state.pingTimer); state.pingTimer = null; }
  if (state.ws) {
    const ws = state.ws;
    ws.onclose = null; ws.ontext = null; ws.onbinary = null; ws.onerror = null;
    ws.close();
    state.ws = null;
  }
  if (state.micSource) { try { state.micSource.disconnect(); } catch { /* already gone */ } state.micSource = null; }
  if (state.micFilter) { try { state.micFilter.disconnect(); } catch { /* already gone */ } state.micFilter = null; }
  if (state.micLowpass) { try { state.micLowpass.disconnect(); } catch { /* already gone */ } state.micLowpass = null; }
  if (state.workletNode) { try { state.workletNode.disconnect(); } catch { /* already gone */ } state.workletNode = null; }
  if (state.stream) { for (const t of state.stream.getAudioTracks()) t.stop(); state.stream = null; }
  if (state.inCtx) { try { state.inCtx.close(); } catch { /* ignore */ } state.inCtx = null; }
  if (state.outCtx) { try { state.outCtx.close(); } catch { /* ignore */ } state.outCtx = null; }
  state.outGain = null;
  state.monGain = null;
  if (state.workletUrl) { URL.revokeObjectURL(state.workletUrl); state.workletUrl = null; }
  state.pendingTs = [];
  state.nextTime = 0;
  state.nextTimeMon = 0;
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
      el("dt", {}, ["نسبة الفهرس"]),
      el("dd", { class: "latin" }, [Number(state.settings.index_rate).toFixed(2)]),
      el("dt", {}, ["حجم الخرج"]),
      el("dd", { class: "latin" }, [`${Math.round(state.volume * 100)}%`]),
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
    if (ui.srvBlocks) ui.srvBlocks.textContent = String(msg.blocks);
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

/* ------------------------------ speaker voiceprint enrollment ------------------------------ */

function loadSavedProfile() {
  try {
    const raw = localStorage.getItem(SPEAKER_PROFILE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveSpeakerProfile(profile) {
  try {
    if (profile) localStorage.setItem(SPEAKER_PROFILE_KEY, JSON.stringify(profile));
    else localStorage.removeItem(SPEAKER_PROFILE_KEY);
  } catch { /* ignore blocked storage */ }
}

const GUARD_LABELS = {
  ok: ["يتحدث الآن — صوت المتحدث موثّق", "ok"],
  silence: ["صمت / بانتظار المتحدث", "idle"],
  noise: ["تم حجب ضوضاء غير كلامية", "warn"],
  far_voice: ["تم حجب صوت بعيد/خافت", "warn"],
  other_speaker: ["تم حجب صوت متحدث آخر", "error"],
  self_echo: ["تم حجب ارتداد/صدى المكبر", "warn"],
};

function renderSpeakerBadge(state, ui, liveMsg = null) {
  const badge = ui.spkBadge;
  if (!badge) return;
  if (state.enrolling) {
    badge.className = "badge loading dot";
    badge.textContent = "تكلّم بصوتك الطبيعي قرب الميكروفون (3 ثوانٍ)…";
    return;
  }
  if (!state.settings.strict_isolation) {
    badge.className = "badge warn dot";
    badge.textContent = "العزل الصارم معطّل";
    return;
  }
  if (state.phase === "streaming" && liveMsg && liveMsg.guard_status) {
    const [txt, cls] = GUARD_LABELS[liveMsg.guard_status] || ["نشط", "info"];
    badge.className = `badge ${cls} dot`;
    badge.textContent = txt;
    return;
  }
  if (state.settings.speaker_profile) {
    const f0 = state.settings.speaker_profile.f0_hz ? ` · ${Math.round(state.settings.speaker_profile.f0_hz)} Hz` : "";
    badge.className = "badge ok dot";
    badge.textContent = `بصمة صوتك محفوظة${f0} — مقفول عليك فقط`;
  } else {
    badge.className = "badge info dot";
    badge.textContent = "قفل تلقائي على أول متحدث قريب (أو سجّل بصمتك)";
  }
}

function int16ToBase64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

async function enrollSpeakerVoice(state, ui) {
  if (state.enrolling) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("المتصفح لا يدعم التقاط الميكروفون.", "error");
    return;
  }
  state.enrolling = true;
  if (ui.btnEnroll) ui.btnEnroll.disabled = true;
  renderSpeakerBadge(state, ui);
  let stream = null;
  let ctx = null;
  let worklet = null;
  let workletUrl = null;
  try {
    const micId = ui.micSelect ? ui.micSelect.value : "";
    const audioConstraints = {
      echoCancellation: Boolean(state.aec),
      noiseSuppression: Boolean(state.aec),
      autoGainControl: false,
      channelCount: 1,
    };
    if (micId) audioConstraints.deviceId = { exact: micId };
    stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    try { ctx = new AudioContext({ sampleRate: INPUT_RATE }); } catch { ctx = new AudioContext(); }
    if (ctx.state === "suspended") await ctx.resume();
    workletUrl = URL.createObjectURL(new Blob([WORKLET_CODE], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(workletUrl);
    worklet = new AudioWorkletNode(ctx, "ivs-block", {
      processorOptions: { blockMs: 160 },
      numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
    });
    const src = ctx.createMediaStreamSource(stream);
    if (typeof ctx.createBiquadFilter === "function") {
      const hpf = ctx.createBiquadFilter();
      hpf.type = "highpass";
      hpf.frequency.value = 100;
      const lpf = ctx.createBiquadFilter();
      lpf.type = "lowpass";
      lpf.frequency.value = 7200;
      src.connect(hpf);
      hpf.connect(lpf);
      lpf.connect(worklet);
    } else {
      src.connect(worklet);
    }
    const chunks = [];
    const inRate = ctx.sampleRate;
    await new Promise((resolve) => {
      const timer = setTimeout(resolve, 3200);
      worklet.port.onmessage = (ev) => {
        if (!(ev.data instanceof Float32Array)) return;
        let blk = ev.data;
        if (inRate !== INPUT_RATE) blk = resampleLinear(blk, inRate, INPUT_RATE);
        chunks.push(blk);
        updateMeter(state, ui, "in", rmsOf(blk));
        if (chunks.length >= 19) { clearTimeout(timer); resolve(); }
      };
    });
    const total = chunks.reduce((a, b) => a + b.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }
    const int16 = floatToInt16(merged);
    const res = await post("/realtime/enroll", { pcm_b64: int16ToBase64(int16) });
    saveSpeakerProfile(res.profile);
    state.settings.speaker_profile = res.profile;
    pushSettings(state, ui);
    toast(`تم حفظ بصمة صوتك بنجاح (${res.f0_hz ? `${res.f0_hz} Hz` : "جاهزة"}) — سيتم تجاهل أي صوت آخر.`, "success", 5000);
  } catch (err) {
    toast(err.message || "تعذّر تسجيل بصمة الصوت.", "error", 6000);
  } finally {
    state.enrolling = false;
    if (ui.btnEnroll) ui.btnEnroll.disabled = false;
    if (worklet) { try { worklet.disconnect(); } catch {} }
    if (stream) { for (const t of stream.getAudioTracks()) t.stop(); }
    if (ctx) { try { await ctx.close(); } catch {} }
    if (workletUrl) URL.revokeObjectURL(workletUrl);
    resetMeters(ui);
    renderSpeakerBadge(state, ui);
  }
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
