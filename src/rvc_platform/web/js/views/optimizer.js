/* ============================================================
   optimizer.js — ULTIMATE AI VOICE OPTIMIZER
   رابط قسم التحسين الذكي: بناء ملف المتحدث المرجعي، إعداد البحث،
   لوحة بث حية للتجارب، نتائج متعددة المعايير، مقارنة A/B،
   تطبيق أفضل الإعدادات وتصدير التقرير.
   ============================================================ */

import {
  get, post, postForm, getBlob, toast, el, clear, icon, setBusy,
  fmtNum, fmtDuration, fmtMB, confirmDialog,
} from "../api.js";

export default { mount };

const AUDIO_ACCEPT = ".wav,.mp3,.flac,.ogg,.m4a,.aac,.opus,.webm,audio/*";

const AXIS_LABELS = {
  speaker_similarity: "تشابه هوية المتحدث",
  timbre_similarity: "تشابه طابع الصوت",
  pitch_consistency: "اتساق طبقة الصوت",
  naturalness: "الطبيعية",
  clarity: "وضوح النطق",
  artifact_score: "خلوّه من التشوهات",
  prosody_preservation: "حفظ الأداء الصوتي",
  audio_quality: "جودة الصوت",
};

const AXIS_HINTS = {
  speaker_similarity: "تشابه كونيّ بين بصمات التحقق من المتحدث — ليس نسبة تطابق",
  timbre_similarity: "تقارب اللون الطيفي (MFCC/الفورمانت) مع المرجع",
  pitch_consistency: "قرب الطبقة والمدى النغمي من مرجع المتحدث",
  naturalness: "قرب الخصائص الصوتية (HNR/اهتزاز) من طبيعية المتحدث",
  clarity: "احتفاظ التفاصيل الطيفية والنطقية بالمدخل",
  artifact_score: "بوابات كشف التشوهات (معدني/روبوتي/طنين/تشويش)",
  prosody_preservation: "حفظ الإيقاع والوقفات والديناميكية مقابل المدخل",
  audio_quality: "نسبة الإشارة للتشويش والمدى الديناميكي",
};

const MODES = {
  quick: {
    label: "تحسين سريع",
    desc: "بحث مركّز على الإعدادات الأهم (~12 تجربة) — دقائق قليلة.",
    trials: 12,
  },
  advanced: {
    label: "تحسين متقدم",
    desc: "بحث واسع بخوارزمية TPE التكيفية + تحقق على مقاطع منفصلة (~24 تجربة).",
    trials: 24,
  },
  deep: {
    label: "تحسين عميق",
    desc: "بحث متعدد المراحل بنطاقات أوسع وتحقق موسّع (~48 تجربة).",
    trials: 48,
  },
};

const PHASE_LABELS = {
  idle: "خامل", baseline: "خط الأساس", exploration: "استكشاف",
  optimization: "تحسين موجَّه", fine_tuning: "ضبط دقيق",
  validation: "تحقق على مقاطع منفصلة", final_selection: "الاختيار النهائي",
  completed: "اكتمل", failed: "فشل", stopped: "أُوقف", paused: "متوقف مؤقتًا",
};

const AXIS_COLORS = {
  composite: "#5aa2e8", speaker: "#8fd48f", clarity: "#e0a13c",
  naturalness: "#c792ea", artifact: "#e05c5c",
};

let pollTimer = null;

/* ============================== mount ============================== */

async function mount(root) {
  const state = {
    models: [],
    loadedModel: null,
    profiles: [],
    samples: [],
    inputAudio: null,          // {id, file, path}
    settings: null,            // /convert/settings
    capabilities: null,
    mode: "advanced",
    weights: null,
    results: null,
    blindMode: false,
    lastSessionId: null,
    abUrls: [],                // object URLs to revoke
  };

  root.replaceChildren(el("div", { class: "stack" }, [
    buildHeaderCard(),
    buildEngineCard(),
    el("div", { class: "grid" }, [
      buildInputsCard(state),
      buildConfigCard(state),
    ]),
    buildLiveCard(state),
    buildResultsCard(state),
    buildABCard(state),
  ]));

  const ui = bindUI(root, state);
  wireConfigUI(ui, state);
  wireControlsUI(ui, state);

  await Promise.all([
    loadModels(ui, state),
    loadCapabilities(ui, state),
    refreshEngineCard(ui, state),
    refreshOptimizerStatus(ui, state),
    refreshProfiles(ui, state),
  ]);
  // show the latest persisted session's results immediately when no session
  // is running (returning to the tab after a completed/interrupted run)
  if (!state.results) {
    await loadResults(ui, state);
  }
  pollStatus(ui, state);

  function cleanup() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    for (const url of state.abUrls) URL.revokeObjectURL(url);
    state.abUrls = [];
  }
  return cleanup;
}

/* ============================ header ============================ */

function buildHeaderCard() {
  return el("div", { class: "card" }, [
    el("div", { class: "opt-header-band" }, [
      el("span", { class: "badge" }, ["RVC v2"]),
      el("div", { class: "grow" }, [
        el("div", { class: "opt-title" }, ["ULTIMATE AI VOICE OPTIMIZER — المحسّن الصوتي الذكي"]),
        el("div", { class: "opt-sub" }, [
          "تحليل هوية المتحدث المرجعي وضبط معاملات محرك RVC v2 تلقائيًا ",
          "للوصول لأقرب تشابه ممكن بالصوت الأصلي — دون التضحية بالطبيعية أو الوضوح.",
        ]),
      ]),
    ]),
    el("div", { class: "notice mt", style: "margin-bottom:0" }, [
      el("span", { class: "n-icon" }, [icon("info", 18)]),
      el("div", {}, [
        el("p", { style: "margin:0" }, [
          "كل الدرجات مقاييس نسبية داخل نطاق البحث الحالي وليست نسبة تطابق حقيقية. ",
          "تشابه الهوية هو تشابه كوني (cosine) بين بصمات تحقق من المتحدث — راجع «ملاحظات الموثوقية» في النتائج.",
        ]),
      ]),
    ]),
  ]);
}

function buildEngineCard() {
  return el("div", { class: "card", id: "opt-engine" }, [
    el("h2", { class: "card-title" }, ["حالة المحرك", el("span", { class: "sub latin" }, ["Engine"])]),
    el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
  ]);
}

/* ============================ inputs ============================ */

function buildInputsCard(state) {
  return el("div", { class: "card", id: "opt-inputs" }, [
    el("h2", { class: "card-title" }, [
      "المدخلات والملف المرجعي",
      el("span", { class: "sub latin" }, ["Inputs & Reference"]),
    ]),

    // ---- model ----
    el("div", { class: "field" }, [
      el("label", {}, ["نموذج RVC v2"]),
      el("div", { class: "row" }, [
        el("select", { id: "opt-model", class: "grow" }),
        el("button", { class: "btn", type: "button", id: "opt-load-model" }, [
          icon("cpu", 14), " تحميل",
        ]),
      ]),
      el("span", { class: "hint" }, ["يُحمَّل النموذج (PTH + فهرس INDEX) في المحرك نفسه المستخدم للتحويل."]),
    ]),

    // ---- reference profile ----
    el("div", { class: "field mt" }, [
      el("label", {}, ["الملف المرجعي للمتحدث — ", el("span", { class: "latin" }, ["Reference profile"])]),
      el("div", { class: "row" }, [
        el("select", { id: "opt-profile", class: "grow" }),
        el("input", {
          type: "text", id: "opt-profile-new", class: "grow",
          placeholder: "اسم ملف جديد (مثال: جلسة_نظيفة)",
          style: "max-width:170px",
        }),
      ]),
      el("span", { class: "hint" }, [
        "يمكن إنشاء أكثر من ملف مرجعي للنموذج نفسه — كل ملف يُحفظ مستقلًا ولا يُستخدم مع نموذج آخر.",
      ]),
    ]),

    // ---- reference samples ----
    el("div", { class: "field mt" }, [
      el("label", {}, ["عينات مرجعية نظيفة من صوت المتحدث الأصلي"]),
      el("button", { class: "btn block", type: "button", id: "opt-ref-upload" }, [
        icon("upload", 15), " إضافة عينات (يفضل 2-5 عينات متنوعة، ≥ 3 ثوانٍ لكل عينة)",
      ]),
      el("input", { type: "file", accept: AUDIO_ACCEPT, multiple: true, id: "opt-ref-files", class: "hidden" }),
      el("div", { class: "opt-samples-list mt", id: "opt-ref-list" }, []),
      el("div", { class: "row mt" }, [
        el("button", { class: "btn primary", type: "button", id: "opt-analyze" }, [
          icon("wave", 15), " تحليل العينات وبناء بصمة المتحدث",
        ]),
      ]),
      el("div", { id: "opt-profile-status", class: "mt" }, []),
    ]),

    // ---- input audio ----
    el("div", { class: "field mt" }, [
      el("label", {}, ["الصوت المراد تحويله"]),
      el("button", { class: "btn block", type: "button", id: "opt-input-upload" }, [
        icon("upload", 15), " اختيار ملف صوتي للتحسين",
      ]),
      el("input", { type: "file", accept: AUDIO_ACCEPT, id: "opt-input-file", class: "hidden" }),
      el("div", { id: "opt-input-info", class: "mt small muted" }, []),
    ]),

    // ---- current params ----
    el("div", { class: "mt", id: "opt-current-params" }, []),
  ]);
}

/* ============================ config ============================ */

function buildConfigCard(state) {
  const modeCards = el("div", { class: "opt-mode-grid" },
    Object.entries(MODES).map(([id, m]) =>
      el("div", { class: `opt-mode-card${id === state.mode ? " active" : ""}`,
                  dataset: { mode: id } }, [
        el("div", { class: "m-name" }, [m.label]),
        el("div", { class: "m-desc" }, [m.desc]),
      ])));

  return el("div", { class: "card", id: "opt-config" }, [
    el("h2", { class: "card-title" }, [
      "إعداد التحسين",
      el("span", { class: "sub latin" }, ["Optimization Setup"]),
    ]),

    el("div", { class: "field" }, [el("label", {}, ["نمط البحث"]), modeCards]),

    el("div", { class: "row mt" }, [
      el("div", { class: "field grow" }, [
        el("label", { for: "opt-trials" }, ["عدد التجارب (الميزانية)"]),
        el("input", { type: "number", id: "opt-trials", min: "4", max: "400",
                      step: "1", value: "24", class: "grow" }),
      ]),
      el("div", { class: "field grow" }, [
        el("label", { for: "opt-timebudget" }, ["حد زمني بالدقائق (0 = بلا حد)"]),
        el("input", { type: "number", id: "opt-timebudget", min: "0", max: "1440",
                      step: "1", value: "0", class: "grow" }),
      ]),
    ]),

    el("div", { class: "row" }, [
      el("div", { class: "field grow" }, [
        el("label", { for: "opt-strategy" }, ["خوارزمية البحث"]),
        el("select", { id: "opt-strategy" }, [
          el("option", { value: "auto", selected: true }, ["تلقائي — Optuna TPE عند توفره، وإلا البحث التكيفي المدمج"]),
          el("option", { value: "optuna" }, ["Optuna TPE / Bayesian"]),
          el("option", { value: "builtin" }, ["البحث التكيفي المدمج (coarse-to-fine)"]),
        ]),
      ]),
      el("div", { class: "field grow" }, [
        el("label", { for: "opt-embedder" }, ["نموذج بصمة المتحدث"]),
        el("select", { id: "opt-embedder" }, [
          el("option", { value: "auto", selected: true }, ["تلقائي — WavLM + ECAPA معًا عند الإمكان"]),
          el("option", { value: "wavlm" }, ["WavLM x-vector"]),
          el("option", { value: "ecapa" }, ["ECAPA-TDNN"]),
          el("option", { value: "both" }, ["كلاهما (أدق موثوقية)"]),
        ]),
      ]),
    ]),

    el("div", { class: "field" }, [
      el("label", { for: "opt-pitch" }, ["سياسة طبقة الصوت (Pitch)"]),
      el("select", { id: "opt-pitch" }, [
        el("option", { value: "preserve", selected: true }, ["المحافظة على طبقة الصوت الأصلية (بدون انزياح)"]),
        el("option", { value: "auto" }, ["تلقائي — محاذاة الطبقة لوسيط مرجع المتحدث"]),
        el("option", { value: "range" }, ["بحث ضمن نطاق آمن محدد"]),
      ]),
    ]),
    el("div", { class: "row", id: "opt-pitch-range-row" }, [
      el("div", { class: "field grow" }, [
        el("label", { for: "opt-pitch-min" }, ["أدنى نصف نغمة"]),
        el("input", { type: "number", id: "opt-pitch-min", min: "-24", max: "0",
                      value: "-6", class: "grow" }),
      ]),
      el("div", { class: "field grow" }, [
        el("label", { for: "opt-pitch-max" }, ["أقصى نصف نغمة"]),
        el("input", { type: "number", id: "opt-pitch-max", min: "0", max: "24",
                      value: "6", class: "grow" }),
      ]),
    ]),

    el("div", { class: "field" }, [
      el("label", {}, ["طرق استخراج الطبقة المشمولة بالبحث"]),
      el("div", { class: "row", id: "opt-f0-methods" }, []),
      el("span", { class: "hint" }, ["تُعرض فقط الطرق المتاحة فعليًا في هذا المحرك."]),
    ]),

    el("div", { class: "opt-weights mt" }, [
      el("div", { class: "row" }, [
        el("label", { style: "font-weight:700" }, ["أوزان معايير التقييم"]),
        el("button", { class: "btn ghost small", type: "button", id: "opt-weights-reset" }, ["استعادة الافتراضي"]),
      ]),
      el("div", { id: "opt-weights-list", class: "mt" }, []),
      el("span", { class: "hint" }, [
        "هوية المتحدث لها الوزن الأكبر افتراضيًا. الأوزان تُطبَّع تلقائيًا؛ صفر يعني تجاهل المعيار.",
      ]),
    ]),

    el("div", { class: "row mt" }, [
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", id: "opt-normalize", checked: true }),
        el("span", {}, ["توحيد مستوى الخرج"]),
      ]),
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", id: "opt-denoise" }),
        el("span", {}, ["إزالة ضجيج مسبقة (يفضّل إبقاؤها موحدة بين التجارب)"]),
      ]),
    ]),

    el("button", { class: "btn primary block mt", type: "button", id: "opt-start" }, [
      icon("play", 16), " بدء التحسين الذكي",
    ]),
  ]);
}

/* ============================ live ============================ */

function buildLiveCard(state) {
  return el("div", { class: "card hidden", id: "opt-live" }, [
    el("h2", { class: "card-title" }, [
      "لوحة البث الحية",
      el("span", { class: "sub latin" }, ["Live Optimization Dashboard"]),
    ]),
    el("div", { class: "opt-live-top" }, [
      el("span", { class: "badge dot opt-live-phase", id: "opt-phase" }, ["—"]),
      el("div", { class: "progress grow opt-progress-big" }, [
        el("div", { class: "bar", style: "width:0%" }),
        el("span", { class: "opt-progress-num", id: "opt-progress-num" }, ["0%"]),
      ]),
      el("span", { class: "dim small", id: "opt-elapsed" }, ["—"]),
    ]),
    el("p", { class: "small muted", id: "opt-message", style: "margin:8px 0" }, ["…"]),
    el("div", { class: "row" }, [
      el("div", { class: "grow" }, [
        el("div", { class: "small dim" }, ["التجربة الحالية"]),
        el("div", { class: "opt-trial-params", id: "opt-current-trial" }, ["—"]),
      ]),
    ]),
    el("div", { class: "opt-axes-grid mt", id: "opt-live-best" }, []),
    el("div", { class: "opt-resources mt", id: "opt-resources" }, []),
    el("div", { class: "opt-actions mt" }, [
      el("button", { class: "btn", type: "button", id: "opt-pause" }, [icon("stop", 14), " إيقاف مؤقت"]),
      el("button", { class: "btn", type: "button", id: "opt-resume" }, [icon("play", 14), " استكمال"]),
      el("button", { class: "btn danger", type: "button", id: "opt-cancel" }, [icon("trash", 14), " إلغاء"]),
      el("span", { class: "hint" }, [
        "الحالة تُحفظ بعد كل تجربة — الإيقاف أو إعادة التشغيل لا يضيّع النتائج.",
      ]),
    ]),
  ]);
}

/* ============================ results ============================ */

function buildResultsCard(state) {
  return el("div", { class: "card hidden", id: "opt-results" }, [
    el("h2", { class: "card-title" }, [
      "النتائج",
      el("span", { class: "sub latin" }, ["Results Dashboard"]),
    ]),
    el("div", { id: "opt-results-body" }, []),
  ]);
}

/* ============================ A/B ============================ */

function buildABCard(state) {
  return el("div", { class: "card hidden", id: "opt-ab" }, [
    el("h2", { class: "card-title" }, [
      "مقارنة استماعية A/B",
      el("span", { class: "sub latin" }, ["Audio A/B Comparison"]),
    ]),
    el("div", { class: "opt-ab-controls" }, [
      el("button", { class: "btn", type: "button", id: "opt-ab-play" }, [icon("play", 14), " تشغيل متزامن"]),
      el("button", { class: "btn", type: "button", id: "opt-ab-stop" }, [icon("stop", 14), " إيقاف الكل"]),
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", id: "opt-ab-loop" }),
        el("span", {}, ["تكرار تلقائي"]),
      ]),
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", id: "opt-ab-blind" }),
        el("span", {}, ["وضع Blind A/B (إخفاء التسميات)"]),
      ]),
      el("button", { class: "btn ghost", type: "button", id: "opt-ab-reveal", hidden: true }, ["كشف التسميات"]),
    ]),
    el("div", { class: "opt-ab-grid mt", id: "opt-ab-grid" }, []),
  ]);
}

/* ============================ data loading ============================ */

async function loadModels(ui, state) {
  try {
    state.models = await get("/models");
  } catch (err) {
    state.models = [];
    toast(err.message, "error");
  }
  const sel = ui.model;
  clear(sel);
  if (!state.models.length) {
    sel.append(el("option", { value: "" }, ["لا توجد نماذج — أضف نموذجًا من تبويب النماذج"]));
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  for (const m of state.models) {
    const opt = el("option", { value: m.name }, [
      `${m.name} · ${m.sample_rate / 1000}kHz · ${m.version}${m.index_path ? " · فهرس" : ""}`,
    ]);
    if (m.name === state.loadedModel) opt.selected = true;
    sel.append(opt);
  }
  refreshEngineCard(ui, state);
}

async function refreshEngineCard(ui, state) {
  try {
    const settings = await get("/convert/settings");
    state.settings = settings;
    const card = document.getElementById("opt-engine");
    if (card) {
      clear(card);
      const lm = settings.engine.loaded_model;
      const ready = settings.engine.status === "ready" && lm;
      card.append(
        el("div", { class: "row" }, [
          el("span", { class: `badge ${settings.engine.status} dot` },
            [ready ? "جاهز" : "غير محمّل"]),
          lm ? el("span", { class: "dim" }, [
            `النموذج المحمّل: ${lm.name} (${lm.sample_rate / 1000}kHz)`])
            : el("span", { class: "dim" }, ["لم يُحمَّل أي نموذج بعد"]),
        ]),
        el("div", { id: "opt-current-params", class: "mt" },
          [renderCurrentParams(settings)]),
      );
      state.loadedModel = lm ? lm.name : null;
      if (lm && ui.model.value !== lm.name && [...ui.model.options].some(o => o.value === lm.name)) {
        ui.model.value = lm.name;
      }
    }
    updateStartButton(ui, state);
  } catch (err) {
    toast(err.message, "error");
  }
}

function renderCurrentParams(settings) {
  const d = settings.defaults || {};
  const applied = settings.optimizer_applied || null;
  const chips = el("div", { class: "opt-chip-row" }, [
    el("span", { class: "badge" }, [`pitch: ${d.pitch_shift}`]),
    el("span", { class: "badge" }, [`f0: ${d.f0_method}`]),
    el("span", { class: "badge" }, [`index_rate: ${fmtNum(d.index_rate, 2)}`]),
    el("span", { class: "badge" }, [`filter_radius: ${d.filter_radius}`]),
    el("span", { class: "badge" }, [`rms_mix: ${fmtNum(d.rms_mix_rate, 2)}`]),
    el("span", { class: "badge" }, [`protect: ${fmtNum(d.protect, 2)}`]),
  ]);
  if (!applied) return chips;
  return el("div", {}, [
    chips,
    el("div", { class: "notice mt", style: "margin-bottom:0" }, [
      el("span", { class: "n-icon" }, [icon("check", 16)]),
      el("div", {}, [
        el("p", { style: "margin:0" }, [
          "لهذا النموذج إعدادات محسّنة مطبّقة من المحسّن الذكي ",
          `(${fmtNum(applied.scores?.composite ?? 0, 3)} نقطة مركّبة${applied.scores?.validated_composite ? ` · محقّقة ${fmtNum(applied.scores.validated_composite, 3)}` : ""}).`,
        ]),
      ]),
    ]),
  ]);
}

async function loadCapabilities(ui, state) {
  try {
    state.capabilities = await get("/optimizer/capabilities");
  } catch (err) {
    state.capabilities = null;
    return;
  }
  // f0 methods checkboxes
  const box = ui.f0Methods;
  clear(box);
  const methods = state.capabilities.f0_methods || ["rmvpe", "fcpe", "pm"];
  for (const m of methods) {
    box.append(el("label", { class: "check" }, [
      el("input", { type: "checkbox", value: m, checked: true }),
      el("span", {}, [m]),
    ]));
  }
  // embedder status hint
  const emb = state.capabilities.embedders || {};
  if (emb.reliability === "low") {
    toast("تنبيه: لا يتوفر أي نموذج بصمة عصبي (WavLM/ECAPA) — تقييم الهوية سيعمل بموثوقية منخفضة.", "warn", 8000);
  }
  // weights defaults
  if (!state.weights) state.weights = { ...state.capabilities.weights };
  renderWeights(ui, state);
}

function renderWeights(ui, state) {
  const box = ui.weightsList;
  clear(box);
  for (const [key, label] of Object.entries(AXIS_LABELS)) {
    const val = state.weights?.[key] ?? 0;
    const input = el("input", {
      type: "range", min: "0", max: "0.5", step: "0.01", value: String(val),
    });
    const num = el("span", { class: "slider-val" }, [fmtNum(val, 2)]);
    input.addEventListener("input", () => {
      state.weights[key] = Number(input.value);
      clear(num); num.append(document.createTextNode(fmtNum(input.value, 2)));
    });
    box.append(el("div", { class: "slider-row", title: AXIS_HINTS[key] }, [
      el("div", { class: "slider-head" }, [
        el("label", {}, [label]), num,
      ]),
      input,
    ]));
  }
}

/* ============================ profiles & samples ============================ */

async function refreshProfiles(ui, state) {
  const model = ui.model.value;
  const profileSel = ui.profile;
  state.profiles = [];
  clear(profileSel);
  if (!model) {
    profileSel.append(el("option", { value: "default" }, ["default"]));
    updateStartButton(ui, state);
    return;
  }
  try {
    const data = await get(`/optimizer/profiles?model_name=${encodeURIComponent(model)}`);
    state.profiles = data.profiles || [];
  } catch (err) {
    state.profiles = [];
  }
  const names = new Set(state.profiles.map(p => p.name));
  names.add("default");
  for (const name of names) {
    profileSel.append(el("option", { value: name }, [name]));
  }
  await refreshSamples(ui, state);
  updateStartButton(ui, state);
}

async function refreshSamples(ui, state) {
  const model = ui.model.value;
  const profile = ui.profile.value || "default";
  const listEl = ui.refList;
  clear(listEl);
  state.samples = [];
  if (!model) return;
  try {
    const data = await get(`/optimizer/reference-samples?model_name=${encodeURIComponent(model)}&profile_name=${encodeURIComponent(profile)}`);
    state.samples = data.samples || [];
  } catch (err) { /* empty list is fine */ }
  if (!state.samples.length) {
    listEl.append(el("span", { class: "hint" }, ["لا عينات محفوظة بعد لهذا الملف المرجعي."]));
    return;
  }
  for (const s of state.samples) {
    listEl.append(el("div", { class: "opt-sample-row" }, [
      el("button", {
        class: "btn ghost small", type: "button", title: "استماع",
        onclick: () => playSampleInline(model, profile, s.id),
      }, [icon("speaker", 14)]),
      el("span", { class: "grow" }, [s.file]),
      el("span", { class: "dim small" }, [`${fmtMB(s.size_mb)} MB`]),
      el("button", {
        class: "btn ghost small", type: "button",
        onclick: async () => {
          try {
            await fetchDeleteSample(model, profile, s.id);
            await refreshSamples(ui, state);
            toast("حُذفت العينة.", "success");
          } catch (err) { toast(err.message, "error"); }
        },
      }, [icon("trash", 13)]),
    ]));
  }
}

function playSampleInline(model, profile, sampleId) {
  const url = `/api/optimizer/reference-samples/audio?model_name=${encodeURIComponent(model)}&profile_name=${encodeURIComponent(profile)}&sample_id=${encodeURIComponent(sampleId)}`;
  const a = el("audio", { controls: true, src: url, style: "width:100%;margin-top:4px" });
  const existing = document.getElementById("opt-sample-inline-player");
  if (existing) existing.remove();
  a.id = "opt-sample-inline-player";
  document.getElementById("opt-ref-list")?.append(a);
  a.play().catch(() => {});
}

async function fetchDeleteSample(model, profile, id) {
  const { del } = await import("../api.js");
  const { ApiError } = await import("../api.js");
  const token = localStorage.getItem("ivs_token");
  const res = await fetch(
    `/api/optimizer/reference-samples/${encodeURIComponent(id)}?model_name=${encodeURIComponent(model)}&profile_name=${encodeURIComponent(profile)}`,
    { method: "DELETE", headers: token ? { Authorization: `Bearer ${token}` } : {} },
  );
  if (!res.ok) throw new Error(`فشل الحذف (${res.status})`);
}

/* ============================ wiring ============================ */

function bindUI(root, state) {
  const ui = {};
  for (const [k, id] of Object.entries({
    model: "opt-model", loadModel: "opt-load-model",
    profile: "opt-profile", profileNew: "opt-profile-new",
    refUpload: "opt-ref-upload", refFiles: "opt-ref-files",
    refList: "opt-ref-list", analyze: "opt-analyze",
    profileStatus: "opt-profile-status",
    inputUpload: "opt-input-upload", inputFile: "opt-input-file",
    inputInfo: "opt-input-info",
    trials: "opt-trials", timeBudget: "opt-timebudget",
    strategy: "opt-strategy", embedder: "opt-embedder",
    pitch: "opt-pitch", pitchMin: "opt-pitch-min", pitchMax: "opt-pitch-max",
    pitchRangeRow: "opt-pitch-range-row",
    f0Methods: "opt-f0-methods",
    weightsList: "opt-weights-list", weightsReset: "opt-weights-reset",
    normalize: "opt-normalize", denoise: "opt-denoise",
    start: "opt-start",
    phase: "opt-phase", progressNum: "opt-progress-num",
    message: "opt-message", elapsed: "opt-elapsed",
    currentTrial: "opt-current-trial", liveBest: "opt-live-best",
    resources: "opt-resources",
    pause: "opt-pause", resume: "opt-resume", cancel: "opt-cancel",
    resultsBody: "opt-results-body",
    abGrid: "opt-ab-grid", abPlay: "opt-ab-play", abStop: "opt-ab-stop",
    abLoop: "opt-ab-loop", abBlind: "opt-ab-blind", abReveal: "opt-ab-reveal",
  })) {
    ui[k] = document.getElementById(id);
  }
  ui.progressBars = Array.from(
    document.querySelectorAll("#opt-live .progress .bar"));
  return ui;
}

function wireConfigUI(ui, state) {
  // mode cards
  for (const card of document.querySelectorAll(".opt-mode-card")) {
    card.addEventListener("click", () => {
      document.querySelectorAll(".opt-mode-card")
        .forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      state.mode = card.dataset.mode;
      ui.trials.value = String(MODES[state.mode].trials);
    });
  }
  // pitch policy toggles range inputs
  ui.pitch.addEventListener("change", () => {
    ui.pitchRangeRow.style.display =
      ui.pitch.value === "range" ? "" : "none";
  });
  ui.pitchRangeRow.style.display = "none";
  // weights reset
  ui.weightsReset.addEventListener("click", () => {
    state.weights = { ...(state.capabilities?.weights || {}) };
    renderWeights(ui, state);
  });
  // model change reloads profiles
  ui.model.addEventListener("change", () => refreshProfiles(ui, state));
  ui.profile.addEventListener("change", () => refreshSamples(ui, state));
  // load model button
  ui.loadModel.addEventListener("click", async () => {
    const name = ui.model.value;
    if (!name) return;
    setBusy(ui.loadModel, true, "جارٍ التحميل…");
    try {
      await post(`/models/${encodeURIComponent(name)}/load`);
      await refreshEngineCard(ui, state);
      toast(`تم تحميل النموذج «${name}».`, "success");
    } catch (err) {
      toast(err.message, "error", 9000);
    } finally { setBusy(ui.loadModel, false); }
  });
  // reference uploads
  ui.refUpload.addEventListener("click", () => ui.refFiles.click());
  ui.refFiles.addEventListener("change", async () => {
    const files = Array.from(ui.refFiles.files || []);
    if (!files.length) return;
    await uploadReferenceSamples(ui, state, files);
    ui.refFiles.value = "";
  });
  // analyze profile
  ui.analyze.addEventListener("click", () => analyzeReference(ui, state));
  // input audio
  ui.inputUpload.addEventListener("click", () => ui.inputFile.click());
  ui.inputFile.addEventListener("change", () => {
    const f = ui.inputFile.files && ui.inputFile.files[0];
    if (!f) return;
    setInputAudio(ui, state, f);
  });
}

async function uploadReferenceSamples(ui, state, files) {
  const model = ui.model.value;
  if (!model) { toast("اختر نموذجًا أولًا.", "warn"); return; }
  const profile = profileSelection(ui);
  let ok = 0;
  for (const f of files) {
    const form = new FormData();
    form.append("file", f, f.name);
    form.append("model_name", model);
    form.append("profile_name", profile);
    try {
      await postForm("/optimizer/reference-samples", form);
      ok += 1;
    } catch (err) {
      toast(`${f.name}: ${err.message}`, "error");
    }
  }
  if (ok) toast(`تم حفظ ${ok} عينة مرجعية في الملف «${profile}».`, "success");
  await refreshSamples(ui, state);
}

function profileSelection(ui) {
  const sel = ui.profile.value;
  const fresh = (ui.profileNew.value || "").trim();
  return (fresh && fresh !== sel) ? fresh.replace(/\s+/g, "_") : sel;
}

async function analyzeReference(ui, state) {
  const model = ui.model.value;
  const profile = profileSelection(ui);
  if (!model) { toast("اختر نموذجًا أولًا.", "warn"); return; }
  if (!state.samples.length) { toast("أضف عينة مرجعية واحدة على الأقل.", "warn"); return; }
  setBusy(ui.analyze, true, "جارٍ التحليل…");
  try {
    const result = await post(
      `/optimizer/analyze-reference?model_name=${encodeURIComponent(model)}&profile_name=${encodeURIComponent(profile)}`);
    renderProfileStatus(ui, result, state);
    ui.profileNew.value = "";
    // the profile now exists server-side — refresh both the profile list
    // (the start button requires an analysed profile) and the samples list
    await refreshProfiles(ui, state);
    ui.profile.value = profile;
    await refreshSamples(ui, state);
    toast("اكتمل تحليل العينات وبناء بصمة المتحدث.", "success");
  } catch (err) {
    toast(err.message, "error", 9000);
  } finally { setBusy(ui.analyze, false); }
}

function renderProfileStatus(ui, result, state) {
  const box = ui.profileStatus;
  clear(box);
  const rel = result.reliability || {};
  const levelClass = rel.level === "low" ? "warn" : "";
  const note = el("div", { class: `notice ${levelClass} mt`, style: "margin-bottom:0" }, [
    el("span", { class: "n-icon" }, [icon("info", 16)]),
    el("div", {}, [
      el("p", { style: "margin:0" }, [
        `بصمة المتحدث جاهزة: ${result.n_samples} عينة · موثوقية: ${
          { ok: "جيدة", low: "منخفضة", warn: "تحتاج مراجعة", empty: "فارغة" }[rel.level] || rel.level
        }`,
      ]),
      (rel.notes || []).length
        ? el("ul", { class: "opt-reliability" },
            rel.notes.map(n => el("li", {}, [n])))
        : null,
    ]),
  ]);
  box.append(note);
  // keep it also for later runs
  state.lastProfileAnalysis = result;
}

function setInputAudio(ui, state, file) {
  state.inputAudio = { id: null, file };
  const info = ui.inputInfo;
  clear(info);
  info.append(
    el("div", { class: "latin small dim" }, [file.name]),
    el("div", { class: "small" }, [`${fmtMB(file.size / (1024 * 1024))} MB`]),
  );
  const probe = el("audio", { src: URL.createObjectURL(file), preload: "metadata" });
  probe.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(probe.duration)) {
      info.append(el("div", { class: "small" }, [`المدة: ${fmtDuration(probe.duration)}`]));
      if (probe.duration < 4) {
        info.append(el("div", { class: "small", style: "color:#e0a13c" }, [
          "تنبيه: المدة قصيرة — التحقق على مقاطع منفصلة قد يتعذر.",
        ]));
      }
    }
  });
  updateStartButton(ui, state);
}

function updateStartButton(ui, state) {
  const hasModel = Boolean(ui.model.value) || Boolean(state.loadedModel);
  const hasInput = Boolean(state.inputAudio);
  const hasProfile = Boolean(state.profiles.find(
    p => p.name === (ui.profile.value || "default") && p.n_samples > 0));
  ui.start.disabled = !(hasModel && hasInput && hasProfile);
  ui.start.title = hasProfile ? "" : "يلزم: نموذج + ملف مرجعي محلّل + ملف صوتي";
}

async function startOptimization(ui, state) {
  const model = ui.model.value;
  const profile = profileSelection(ui);
  if (!state.inputAudio) { toast("اختر الملف الصوتي أولاً.", "warn"); return; }
  setBusy(ui.start, true, "جارٍ الرفع…");
  try {
    // 1) upload input audio
    const form = new FormData();
    form.append("file", state.inputAudio.file, state.inputAudio.file.name);
    const stored = await postForm("/optimizer/input", form);
    state.inputAudio.path = stored.stored.path;
    state.inputAudio.id = stored.stored.id;
    // 2) gather f0 methods
    const f0Methods = Array.from(ui.f0Methods.querySelectorAll("input:checked"))
      .map(i => i.value);
    // 3) start
    const startForm = new FormData();
    startForm.append("model_name", model);
    startForm.append("input_path", stored.stored.path);
    startForm.append("profile_name", profile);
    startForm.append("mode", state.mode);
    startForm.append("max_trials", ui.trials.value || "24");
    startForm.append("time_budget_s", String((Number(ui.timeBudget.value) || 0) * 60));
    startForm.append("strategy", ui.strategy.value);
    startForm.append("pitch_policy", ui.pitch.value);
    startForm.append("pitch_min", ui.pitchMin.value);
    startForm.append("pitch_max", ui.pitchMax.value);
    startForm.append("embedder_prefer", ui.embedder.value);
    startForm.append("denoise", ui.denoise.checked ? "true" : "false");
    startForm.append("normalize", ui.normalize.checked ? "true" : "false");
    startForm.append("weights_json", JSON.stringify(state.weights || {}));
    startForm.append("f0_methods_json", JSON.stringify(f0Methods));
    const started = await postForm("/optimizer/start", startForm);
    state.lastSessionId = started.session_id;
    document.getElementById("opt-live")?.classList.remove("hidden");
    document.getElementById("opt-results")?.classList.add("hidden");
    document.getElementById("opt-ab")?.classList.add("hidden");
    toast("بدأ التحسين الذكي.", "success");
    refreshOptimizerStatus(ui, state);
  } catch (err) {
    toast(err.message, "error", 10000);
    if (err.detail) toast(err.detail, "error", 10000);
  } finally {
    setBusy(ui.start, false);
  }
}

/* ============================ controls ============================ */

function wireControlsUI(ui, state) {
  ui.start.addEventListener("click", () => startOptimization(ui, state));
  ui.pause.addEventListener("click", async () => {
    try { const r = await post("/optimizer/pause"); toast(r.message || "…", "info"); }
    catch (err) { toast(err.message, "error"); }
  });
  ui.resume.addEventListener("click", async () => {
    try { const r = await post("/optimizer/resume"); toast(r.message || "…", "info"); }
    catch (err) { toast(err.message, "error"); }
  });
  ui.cancel.addEventListener("click", async () => {
    const ok = await confirmDialog({
      title: "إلغاء التحسين",
      message: "سيُوقف البحث بعد اكتمال التجربة الحالية، وتبقى النتائج محفوظة. المتابعة؟",
      confirmText: "إلغاء البحث", danger: true,
    });
    if (!ok) return;
    try { const r = await post("/optimizer/cancel"); toast(r.message || "…", "info"); }
    catch (err) { toast(err.message, "error"); }
  });
  // A/B controls
  ui.abPlay.addEventListener("click", () => playAllAB(ui, state));
  ui.abStop.addEventListener("click", () => stopAllAB(ui, state));
  ui.abBlind.addEventListener("change", () => {
    state.blindMode = ui.abBlind.checked;
    ui.abReveal.hidden = !state.blindMode;
    if (state.results) renderAB(ui, state);
  });
  ui.abReveal.addEventListener("click", () => {
    ui.abBlind.checked = false;
    state.blindMode = false;
    ui.abReveal.hidden = true;
    if (state.results) renderAB(ui, state);
  });
}

/* ============================ live polling ============================ */

function pollStatus(ui, state) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (document.visibilityState !== "visible") return;
    if (document.getElementById("opt-phase")) {
      await refreshOptimizerStatus(ui, state);
    }
  }, 1500);
}

async function refreshOptimizerStatus(ui, state) {
  let s;
  try { s = await get("/optimizer/status"); } catch { return; }
  if (!s.running && !s.paused && s.phase === "idle") {
    // no session: keep latest results visible if any
    if (state.lastSessionId && state.results) return;
    return;
  }
  document.getElementById("opt-live")?.classList.remove("hidden");
  const phaseEl = ui.phase;
  if (phaseEl) {
    phaseEl.textContent = PHASE_LABELS[s.phase] || s.phase;
    phaseEl.className = `badge dot opt-live-phase ${s.phase}`;
  }
  const pct = Math.round((s.progress || 0) * 100);
  for (const bar of ui.progressBars) bar.style.width = `${pct}%`;
  ui.progressNum.textContent = `${pct}%`;
  ui.message.textContent = s.message || "…";
  ui.elapsed.textContent = s.elapsed_s ? `الزمن: ${fmtDuration(s.elapsed_s)}` : "—";
  // current trial params
  const ct = ui.currentTrial;
  if (ct) {
    clear(ct);
    const cur = s.current_trial;
    if (cur && cur.params) {
      ct.append(...paramChips(cur.params, cur.number));
    } else {
      ct.append(el("span", { class: "dim" }, ["—"]));
    }
  }
  // live best
  if (s.best && ui.liveBest) {
    renderAxisTiles(ui.liveBest, s.best.axes, s.best.composite);
  }
  // resources
  if (ui.resources) renderResources(ui.resources, s.resources || {});
  // phase transitions -> refresh results
  if (["completed", "stopped", "failed"].includes(s.phase)) {
    if (state.lastSessionId !== s.session_id || !state.results) {
      state.lastSessionId = s.session_id;
      await loadResults(ui, state);
    }
  }
}

function paramChips(params, number) {
  const chips = [];
  if (number !== undefined && number !== null) {
    chips.push(el("span", { class: "badge" }, [`تجربة ${number}`]));
  }
  chips.push(el("span", { class: "badge" }, [`f0: ${params.f0_method}`]));
  chips.push(el("span", { class: "badge" }, [`index: ${fmtNum(params.index_rate, 2)}`]));
  chips.push(el("span", { class: "badge" }, [`filter: ${params.filter_radius}`]));
  chips.push(el("span", { class: "badge" }, [`rms: ${fmtNum(params.rms_mix_rate, 2)}`]));
  chips.push(el("span", { class: "badge" }, [`protect: ${fmtNum(params.protect, 2)}`]));
  if (params.pitch_shift) {
    chips.push(el("span", { class: "badge" }, [`pitch: ${params.pitch_shift}`]));
  }
  return chips;
}

function renderAxisTiles(container, axes, composite) {
  clear(container);
  if (composite !== undefined && composite !== null) {
    container.append(axisTile("النتيجة المركّبة", composite, "composite"));
  }
  for (const [key, label] of Object.entries(AXIS_LABELS)) {
    const v = axes?.[key];
    if (v === undefined || v === null) continue;
    container.append(axisTile(label, v, key));
  }
}

function axisTile(label, value, key) {
  const v = Number(value);
  const cls = v >= 0.75 ? "" : v >= 0.5 ? "warn" : "bad";
  const pct = Math.round(Math.max(0, Math.min(1, v)) * 100);
  return el("div", { class: "stat-tile opt-axis-tile", title: AXIS_HINTS[key] || label }, [
    el("div", { class: "t-label" }, [label]),
    el("div", { class: "t-value" }, [
      el("span", {}, [fmtNum(v, 3)]),
      el("div", { class: `opt-axis-bar ${cls}` }, [
        el("span", { style: `width:${pct}%` }),
      ]),
    ]),
  ]);
}

function renderResources(container, r) {
  clear(container);
  const rows = [];
  if (r.cpu_percent !== undefined) rows.push(["CPU", `${fmtNum(r.cpu_percent, 0)}%`]);
  if (r.ram_used_mb) rows.push(["RAM", `${r.ram_used_mb} / ${r.ram_total_mb} MB`]);
  if (r.vram_used_mb !== undefined) {
    rows.push(["VRAM", r.vram_used_mb !== null ? `${r.vram_used_mb} / ${r.vram_total_mb} MB` : "—"]);
  }
  for (const [k, v] of rows) {
    container.append(el("div", { class: "stat-tile" }, [
      el("div", { class: "t-label" }, [k]),
      el("div", { class: "t-value" }, [v]),
    ]));
  }
  if (!rows.length) {
    container.append(el("span", { class: "hint" }, ["لا تتوفر قراءات موارد."]));
  }
}

/* ============================ results ============================ */

async function loadResults(ui, state) {
  try {
    const sid = state.lastSessionId ? `?session_id=${encodeURIComponent(state.lastSessionId)}` : "";
    state.results = await get(`/optimizer/results${sid}`);
    if (state.results.session_id) state.lastSessionId = state.results.session_id;
    document.getElementById("opt-results")?.classList.remove("hidden");
    renderResults(ui, state);
    document.getElementById("opt-ab")?.classList.remove("hidden");
    renderAB(ui, state);
  } catch (err) {
    // no results yet — fine
  }
}

function renderResults(ui, state) {
  const r = state.results;
  if (!r) return;
  const body = ui.resultsBody;
  clear(body);

  // ---- headline ----
  const best = r.best;
  if (!best) {
    body.append(el("div", { class: "notice warn" }, [
      el("span", { class: "n-icon" }, [icon("alert", 16)]),
      el("p", { style: "margin:0" }, [
        "لم يُعثر على أفضل نتيجة (لا تجربة ناجحة ضمن حدود الجودة). جرّب توسيع الميزانية أو مراجعة العينات المرجعية.",
      ]),
    ]));
  } else {
    const bestCard = el("div", { class: "card", style: "box-shadow:none" }, [
      el("h3", { class: "small dim" }, ["أفضل الإعدادات المكتشفة"]),
      el("div", { class: "opt-trial-params" }, paramChips(best.params, best.number)),
      best.validated_composite !== null && best.validated_composite !== undefined
        ? el("p", { class: "small muted" }, [
            `نتيجة مركّبة على مقاطع البحث: ${fmtNum(best.composite, 4)} · على مقاطع التحقق المنفصلة: ${fmtNum(best.validated_composite, 4)}`,
          ])
        : el("p", { class: "small muted" }, [`النتيجة المركّبة: ${fmtNum(best.composite, 4)}`]),
    ]);
    // axes tiles
    const axesBox = el("div", { class: "opt-axes-grid mt" });
    renderAxisTiles(axesBox, best.axes, best.composite);

    // ---- baseline vs optimized ----
    const comp = el("div", { class: "mt" }, [
      el("h3", { class: "small dim" }, ["خط الأساس مقابل المحسّن (لكل معيار)"]),
    ]);
    const baseAxes = (r.baseline?.evaluation?.axes) || {};
    for (const [key, label] of Object.entries(AXIS_LABELS)) {
      const b = baseAxes[key];
      const o = best.axes?.[key];
      if (o === undefined || o === null) continue;
      const bb = Number(b ?? 0);
      const oo = Number(o);
      const delta = oo - bb;
      const up = delta > 0.0005, down = delta < -0.0005;
      comp.append(el("div", { class: "opt-compare-row" }, [
        el("span", { class: "muted" }, [label]),
        el("div", { class: "opt-compare-track", title: AXIS_HINTS[key] }, [
          el("div", { class: "bar-base", style: `width:${Math.round(Math.max(0, Math.min(1, bb)) * 100)}%` }),
          el("div", { class: "bar-opt", style: `width:${Math.round(Math.max(0, Math.min(1, oo)) * 100)}%` }),
        ]),
        el("span", { class: "dim small" }, [fmtNum(oo, 3)]),
        el("span", { class: `opt-compare-delta small ${up ? "up" : ""} ${down ? "down" : ""}` }, [
          up ? "▲" : down ? "▼" : "•", ` ${fmtNum(Math.abs(delta), 3)}`,
        ]),
      ]));
    }

    // ---- chart ----
    const chartBox = el("div", { class: "mt" }, [
      el("h3", { class: "small dim" }, ["تطور النتائج عبر التجارب"]),
      el("div", { class: "opt-chart-wrap" }, [drawChart(r.chart)]),
      el("div", { class: "opt-chart-legend" }, [
        el("span", {}, [el("span", { class: "k", style: `background:${AXIS_COLORS.composite}` }), "المركّبة"]),
        el("span", {}, [el("span", { class: "k", style: `background:${AXIS_COLORS.speaker}` }), "تشابه الهوية"]),
        el("span", {}, [el("span", { class: "k", style: `background:${AXIS_COLORS.clarity}` }), "الوضوح"]),
        el("span", {}, [el("span", { class: "k", style: `background:${AXIS_COLORS.naturalness}` }), "الطبيعية"]),
        el("span", {}, [el("span", { class: "k", style: `background:${AXIS_COLORS.artifact}` }), "خلوّه من التشوهات"]),
      ]),
    ]);

    // ---- reliability notes ----
    const notes = el("div", { class: "mt" }, [
      el("h3", { class: "small dim" }, ["ملاحظات الموثوقية (إلزامية القراءة)"]),
      el("ul", { class: "guide-list" }, (r.reliability_notes || []).map(n =>
        el("li", {}, [n]))),
      el("p", { class: "small muted", style: "margin:0" }, [
        "جميع الدرجات مقاييس نسبية داخل نطاق هذا البحث وبياناته — ليست نسبة تطابق حقيقية ولا احتمال هوية.",
      ]),
    ]);

    // ---- validation table ----
    let validationTable = null;
    if ((r.validation || []).length) {
      const header = el("tr", {}, [
        el("th", {}, ["تجربة"]), el("th", {}, ["الإعدادات"]),
        el("th", {}, ["نقاط البحث"]), el("th", {}, ["نقاط التحقق"]),
        el("th", {}, ["الهبوط"]), el("th", {}, ["تشابه الهوية"]),
      ]);
      const rows = r.validation.map(v => el("tr", {}, [
        el("td", { class: "num" }, [String(v.number)]),
        el("td", { class: "num small" }, [
          `idx ${fmtNum(v.params.index_rate, 2)} · ${v.params.f0_method} · pr ${fmtNum(v.params.protect, 2)}`,
        ]),
        el("td", { class: "num" }, [fmtNum(v.search_composite, 3)]),
        el("td", { class: "num" }, [fmtNum(v.validation_composite, 3)]),
        el("td", { class: "num" }, [
          v.drop === null || v.drop === undefined ? "—" :
            `${fmtNum(v.drop * 100, 1)}%`,
        ]),
        el("td", { class: "num" }, [
          fmtNum(v.evaluation?.axes?.speaker_similarity ?? 0, 3),
        ]),
      ]));
      validationTable = el("div", { class: "mt" }, [
        el("h3", { class: "small dim" }, ["التحقق على مقاطع منفصلة (لم تُستخدم في البحث)"]),
        el("div", { class: "table-wrap" }, [
          el("table", { class: "tbl" }, [
            el("thead", {}, [header]),
            el("tbody", {}, rows),
          ]),
        ]),
      ]);
    }

    // ---- pareto ----
    let paretoBox = null;
    if ((r.pareto || []).length > 1) {
      paretoBox = el("div", { class: "mt" }, [
        el("h3", { class: "small dim" }, ["جبهة باريتو (Pareto) — توازن التشابه/الطبيعية/الوضوح"]),
        el("div", { class: "opt-chip-row" }, r.pareto.map(p =>
          el("span", { class: "badge" }, [
            `تجربة ${p.number}: تشابه ${fmtNum(p.axes?.speaker_similarity ?? 0, 3)} · طبيعية ${fmtNum(p.axes?.naturalness ?? 0, 3)} · وضوح ${fmtNum(p.axes?.clarity ?? 0, 3)}`,
          ]))),
      ]);
    }

    // ---- trials table ----
    const trialsHeader = el("tr", {}, [
      el("th", {}, ["#"]), el("th", {}, ["طريقة f0"]),
      el("th", {}, ["Index"]), el("th", {}, ["Filter"]),
      el("th", {}, ["RMS"]), el("th", {}, ["Protect"]),
      el("th", {}, ["Pitch"]), el("th", {}, ["المركّبة"]),
      el("th", {}, ["الهوية"]), el("th", {}, ["الوضوح"]),
      el("th", {}, ["البوابات"]), el("th", {}, ["زمن"]),
    ]);
    const trialsRows = r.trials.map(t => el("tr", {
      style: t.error ? "opacity:.45" : "",
    }, [
      el("td", { class: "num" }, [String(t.number)]),
      el("td", { class: "num" }, [t.error ? "—" : t.params.f0_method]),
      el("td", { class: "num" }, [t.error ? "—" : fmtNum(t.params.index_rate, 2)]),
      el("td", { class: "num" }, [t.error ? "—" : String(t.params.filter_radius)]),
      el("td", { class: "num" }, [t.error ? "—" : fmtNum(t.params.rms_mix_rate, 2)]),
      el("td", { class: "num" }, [t.error ? "—" : fmtNum(t.params.protect, 2)]),
      el("td", { class: "num" }, [t.error ? "—" : String(t.params.pitch_shift ?? 0)]),
      el("td", { class: "num" }, [t.composite === null ? "خطأ" : fmtNum(t.composite, 3)]),
      el("td", { class: "num" }, [fmtNum(t.axes?.speaker_similarity ?? 0, 3)]),
      el("td", { class: "num" }, [fmtNum(t.axes?.clarity ?? 0, 3)]),
      el("td", {}, [t.error ? "فشلت" : t.gates_passed
        ? el("span", { class: "badge" }, ["مجتازة"])
        : el("span", { class: "badge danger" }, [(t.failed_gates || []).join("، ") || "مرفوضة"])]),
      el("td", { class: "num" }, [`${fmtNum(t.duration_s, 1)}ث`]),
    ]));
    const trialsTable = el("div", { class: "mt" }, [
      el("h3", { class: "small dim" }, ["سجل التجارب"]),
      el("div", { class: "table-wrap" }, [
        el("table", { class: "tbl" }, [
          el("thead", {}, [trialsHeader]),
          el("tbody", {}, trialsRows),
        ]),
      ]),
    ]);

    // ---- actions ----
    const actions = el("div", { class: "opt-actions mt" }, [
      el("button", { class: "btn primary", type: "button", id: "opt-apply" }, [
        icon("check", 15), " تطبيق أفضل الإعدادات على هذا النموذج",
      ]),
      el("button", { class: "btn", type: "button", id: "opt-export" }, [
        icon("download", 15), " تصدير تقرير التحسين (JSON)",
      ]),
      el("span", { class: "hint" }, [
        "التطبيق يربط الإعدادات بهذا النموذج فقط، وتبقى الإعدادات السابقة قابلة للاسترجاع.",
      ]),
    ]);

    body.append(bestCard, axesBox, comp, chartBox, validationTable,
                paretoBox, notes, trialsTable, actions);
    document.getElementById("opt-apply")?.addEventListener("click", () => applyBest(ui, state));
    document.getElementById("opt-export")?.addEventListener("click", () => exportReport(ui, state));
  }
}

function drawChart(chart) {
  // tiny dependency-free SVG line chart
  const NS = "http://www.w3.org/2000/svg";
  const W = 640, H = 240, PAD = 34;
  const series = [
    { key: "composite", color: AXIS_COLORS.composite },
    { key: "speaker", color: AXIS_COLORS.speaker },
    { key: "clarity", color: AXIS_COLORS.clarity },
    { key: "naturalness", color: AXIS_COLORS.naturalness },
    { key: "artifact", color: AXIS_COLORS.artifact },
  ];
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%");
  const numbers = chart?.numbers || [];
  if (numbers.length < 2) {
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", String(W / 2)); txt.setAttribute("y", String(H / 2));
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("fill", "currentColor");
    txt.setAttribute("opacity", "0.5");
    txt.textContent = "لا توجد بيانات كافية للرسم بعد";
    svg.append(txt);
    return svg;
  }
  const n = numbers.length;
  const x = (i) => PAD + (i * (W - 2 * PAD)) / (n - 1);
  const y = (v) => H - PAD - Math.max(0, Math.min(1, v)) * (H - 2 * PAD);
  // grid
  for (let g = 0; g <= 4; g += 1) {
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", String(PAD)); line.setAttribute("x2", String(W - PAD));
    line.setAttribute("y1", String(y(g / 4))); line.setAttribute("y2", String(y(g / 4)));
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("stroke-opacity", "0.12");
    svg.append(line);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", "6"); t.setAttribute("y", String(y(g / 4) + 4));
    t.setAttribute("fill", "currentColor"); t.setAttribute("opacity", "0.45");
    t.setAttribute("font-size", "10");
    t.textContent = (g / 4).toFixed(1);
    svg.append(t);
  }
  for (const s of series) {
    const data = chart[s.key] || [];
    if (!data.length) continue;
    const poly = document.createElementNS(NS, "polyline");
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", s.color);
    poly.setAttribute("stroke-width", "2");
    poly.setAttribute("points",
      data.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" "));
    svg.append(poly);
  }
  // x labels
  const xt = document.createElementNS(NS, "text");
  xt.setAttribute("x", String(W - PAD)); xt.setAttribute("y", String(H - 8));
  xt.setAttribute("fill", "currentColor"); xt.setAttribute("opacity", "0.45");
  xt.setAttribute("font-size", "10");
  xt.setAttribute("text-anchor", "end");
  xt.textContent = `التجربة (1 → ${n})`;
  svg.append(xt);
  return svg;
}

/* ============================ apply & export ============================ */

async function applyBest(ui, state) {
  const r = state.results;
  if (!r || !r.best) { toast("لا توجد نتيجة لتطبيقها.", "warn"); return; }
  const btn = document.getElementById("opt-apply");
  setBusy(btn, true, "جارٍ التطبيق…");
  try {
    const applied = await post(
      `/optimizer/apply?session_id=${encodeURIComponent(r.session_id)}`);
    toast("تم تطبيق أفضل الإعدادات وربطها بالنموذج — ستظهر في تبويب التحويل.", "success");
    if (applied.previous) {
      toast("الإعدادات السابقة محفوظة ويمكن استرجاعها بإعادة التطبيق.", "info");
    }
    await refreshEngineCard(ui, state);
  } catch (err) {
    toast(err.message, "error", 9000);
  } finally { setBusy(btn, false); }
}

async function exportReport(ui, state) {
  const r = state.results;
  if (!r) { toast("لا توجد نتائج للتصدير.", "warn"); return; }
  try {
    const report = await get(
      `/optimizer/report?session_id=${encodeURIComponent(r.session_id)}`);
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: `optimizer_report_${r.session_id}.json` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast("تم تصدير التقرير.", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}

/* ============================ A/B ============================ */

function renderAB(ui, state) {
  const grid = ui.abGrid;
  clear(grid);
  const r = state.results;
  if (!r) return;
  const sid = r.session_id;
  const loop = ui.abLoop.checked;
  const cards = [];

  // 1) reference sample (first stored sample)
  const model = r.model;
  const profile = r.profile || "default";
  cards.push({
    kind: "ref", title: "الصوت المرجعي الأصلي",
    sub: "عينة من المتحدث الذي دُرّب النموذج عليه",
    url: `/api/optimizer/reference-samples/audio?model_name=${encodeURIComponent(model)}&profile_name=${encodeURIComponent(profile)}&sample_id=${encodeURIComponent(state.samples[0]?.id || "")}`,
    available: Boolean(state.samples.length),
  });
  // 2) input segment
  cards.push({
    kind: "input", title: "المدخل (قبل التحويل)",
    sub: "المقطع المستخدم في البحث",
    url: `/api/optimizer/results/audio?session_id=${encodeURIComponent(sid)}&name=input_segment`,
    available: true,
  });
  // 3) baseline
  if (r.baseline?.audio) {
    cards.push({
      kind: "baseline", title: "خط الأساس (الإعدادات الافتراضية)",
      sub: "قبل التحسين", url: `/api/optimizer/results/audio?session_id=${encodeURIComponent(sid)}&name=baseline`,
      available: true,
    });
  }
  // 4) best
  if (r.best) {
    cards.push({
      kind: "best", title: "أفضل نتيجة (المحسّنة)",
      sub: "الإعدادات المنتقاة بعد التحقق",
      url: `/api/optimizer/results/audio?session_id=${encodeURIComponent(sid)}&name=best`,
      available: true,
    });
  }
  // 5) top alternatives from kept candidate audio (validation entries)
  for (const v of (r.validation || []).slice(0, 3)) {
    cards.push({
      kind: `alt-${v.number}`, title: `بديل مرشّح (تجربة ${v.number})`,
      sub: "من أفضل النتائج البديلة",
      url: `/api/optimizer/results/audio?session_id=${encodeURIComponent(sid)}&name=validation_${v.number}`,
      available: true,
    });
  }

  const playable = cards.filter(c => c.available);
  // blind shuffle (deterministic per render, labels hidden)
  let order = playable.slice();
  if (state.blindMode) {
    order.sort((a, b) => hashStr(a.kind) - hashStr(b.kind));
  }
  const blindNames = ["النسخة أ", "النسخة ب", "النسخة ج", "النسخة د", "النسخة هـ", "النسخة و"];

  order.forEach((c, i) => {
    const blind = state.blindMode;
    const title = blind ? blindNames[i] : c.title;
    const sub = blind ? "التسمية مخفية (وضع Blind)" : c.sub;
    const audio = el("audio", {
      controls: true, preload: "metadata", src: c.url,
      dataset: { kind: c.kind },
    });
    if (loop) {
      audio.addEventListener("ended", () => { audio.currentTime = 0; audio.play().catch(() => {}); });
    }
    grid.append(el("div", { class: `opt-ab-card${blind ? " blind" : ""}` }, [
      el("div", { class: "ab-head" }, [
        el("span", { class: "ab-title" }, [title]),
        el("span", { class: "badge" }, [blind ? "?" : labelForKind(c.kind)]),
      ]),
      el("span", { class: "ab-sub" }, [sub]),
      audio,
    ]));
  });

  if (!playable.length) {
    grid.append(el("span", { class: "hint" }, ["لا توجد مقاطع للاستماع بعد."]));
  }
}

function labelForKind(kind) {
  if (kind === "ref") return "مرجع";
  if (kind === "input") return "مدخل";
  if (kind === "baseline") return "أساس";
  if (kind === "best") return "الأفضل";
  return "بديل";
}

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

function playAllAB(ui, state) {
  const players = Array.from(ui.abGrid.querySelectorAll("audio"));
  for (const a of players) {
    a.currentTime = 0;
    a.play().catch(() => {});
  }
}

function stopAllAB(ui, state) {
  for (const a of ui.abGrid.querySelectorAll("audio")) a.pause();
}
