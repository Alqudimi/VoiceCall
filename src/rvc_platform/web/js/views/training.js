/* ============================================================
   training.js — التدريب: تجهيز البيانات، بدء/إيقاف/استكمال،
   مراقبة حية (خسائر + رسم بياني)، نقاط حفظ، سجلات، تصدير.
   ============================================================ */

import {
  get, post, getText, toast, el, clear, icon, setBusy,
  TRAIN_STATES, fmtNum, fmtPct, fmtTime, fmtMB, clamp,
} from "../api.js";

export default { mount };

const LOSS_KEYS = ["disc", "gen", "mel", "kl", "fm"];
const LOSS_COLORS = { disc: "#f4675a", gen: "#34d39b", mel: "#5aa2e8", kl: "#f5b23c", fm: "#b48ef7" };
const LOSS_AR = { disc: "مميّز", gen: "مولّد", mel: "Mel", kl: "KL", fm: "ميزة المطابقة", lr: "معدل التعلم" };

const STAGE_AR = {
  idle: "لم يبدأ",
  starting: "بدء التجهيز",
  preprocess: "قص وتجهيز الصوت",
  f0: "استخراج الطبقة (F0)",
  features: "استخراج الميزات",
  index: "بناء الفهرس",
  done: "اكتمل التجهيز",
  failed: "فشل",
};

const CHART_KEEP = 300;

async function mount(root) {
  const state = {
    prepareExperiment: "",
    currentExperiment: "",
    prepareTimer: null,
    monitorTimer: null,
    logTimer: null,
    lastStep: null,
    lastUpdatedAt: null,
    history: { disc: [], gen: [], mel: [], kl: [], fm: [] },
  };

  root.replaceChildren(el("div", { class: "stack" }, [
    buildGpuNotice(),
    el("div", { class: "grid" }, [
      buildFormCard(),
      buildPrepareCard(),
    ]),
    buildMonitorCard(),
    buildRunsCard(),
    el("div", { class: "grid" }, [
      buildCheckpointsCard(),
      buildLogsCard(),
    ]),
  ]));

  const ui = {
    exp: root.querySelector("#tr-exp"),
    dataset: root.querySelector("#tr-dataset"),
    sr: root.querySelector("#tr-sr"),
    epochs: root.querySelector("#tr-epochs"),
    batch: root.querySelector("#tr-batch"),
    lr: root.querySelector("#tr-lr"),
    lrDecay: root.querySelector("#tr-lr-decay"),
    saveEvery: root.querySelector("#tr-save-every"),
    f0: root.querySelector("#tr-f0"),
    f0Method: root.querySelector("#tr-f0-method"),
    speaker: root.querySelector("#tr-speaker"),
    btnPrepare: root.querySelector("#tr-prepare"),
    btnStart: root.querySelector("#tr-start"),
    prepBox: root.querySelector("#tr-prepare-box"),
    monBox: root.querySelector("#tr-monitor-box"),
    monBadge: root.querySelector("#tr-monitor-badge"),
    monBar: root.querySelector("#tr-monitor-bar"),
    monLabel: root.querySelector("#tr-monitor-label"),
    monLosses: root.querySelector("#tr-losses-body"),
    chart: root.querySelector("#tr-chart"),
    runsBox: root.querySelector("#tr-runs-box"),
    btnStop: root.querySelector("#tr-stop"),
    btnResume: root.querySelector("#tr-resume"),
    btnExport: root.querySelector("#tr-export"),
    ckptBox: root.querySelector("#tr-ckpt-box"),
    logsBox: root.querySelector("#tr-logs"),
    btnLogs: root.querySelector("#tr-logs-refresh"),
  };

  wireForm(ui, state);
  drawChart(ui.chart, state.history); // placeholder frame until first losses arrive

  // start live monitoring immediately, then load runs
  state.monitorTimer = setInterval(pollStatus, 3000);
  state.logTimer = setInterval(() => {
    if (document.visibilityState === "visible" && state.currentExperiment) {
      refreshLogs(state, ui);
    }
  }, 10000);
  await Promise.allSettled([pollStatus(), loadRuns(state, ui)]);
  refreshCheckpoints(state, ui);
  refreshLogs(state, ui);

  async function pollStatus() {
    try {
      const status = await get("/training/status");
      renderMonitor(state, ui, status);
      const msg = status?.message || "";
      if (msg && msg !== state.currentExperiment && !msg.includes(" ")) {
        // backend puts the run dir name into `message`
        state.currentExperiment = msg;
        refreshCheckpoints(state, ui);
        refreshLogs(state, ui);
      }
    } catch (err) {
      clear(ui.monBox);
      ui.monBox.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
    }
  }

  function cleanup() {
    if (state.prepareTimer) clearInterval(state.prepareTimer);
    if (state.monitorTimer) clearInterval(state.monitorTimer);
    if (state.logTimer) clearInterval(state.logTimer);
    if (state._ckptHandler) window.removeEventListener("ivs-ckpt-refresh", state._ckptHandler);
  }
  return cleanup;
}

/* ------------------------------ static UI ------------------------------ */

function buildGpuNotice() {
  return el("div", { class: "notice warn" }, [
    el("span", { class: "n-icon" }, [icon("alert", 18)]),
    el("p", {}, ["التدريب يحتاج GPU عمليًا — على Colab استخدم سكربت التثبيت الرسمي، وعلى CPU سيكون بطيئًا جدًا."]),
  ]);
}

function buildFormCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["إعداد التدريب", el("span", { class: "sub latin" }, ["Config"])]),
    el("div", { class: "form-grid" }, [
      field("tr-exp", "اسم التجربة ( Experiment )", el("input", {
        type: "text", id: "tr-exp", placeholder: "my_voice_v1",
        pattern: "[A-Za-z0-9_-]{1,64}", title: "أحرف إنجليزية وأرقام وشرطة فقط",
      }), "أحرف إنجليزية/أرقام/_/- فقط (حتى 64 حرفًا)."),
      field("tr-dataset", "مسار مجلد البيانات على الخادم", el("input", {
        type: "text", id: "tr-dataset", placeholder: "/content/dataset/wavs", dir: "ltr",
      }), "مسار مطلق لمجلد التسجيلات الصوتية على جهاز الخادم."),
      field("tr-sr", "معدل العينة", el("select", { id: "tr-sr" }, [
        el("option", { value: "32k" }, ["32 kHz"]),
        el("option", { value: "40k", selected: true }, ["40 kHz"]),
        el("option", { value: "48k" }, ["48 kHz"]),
      ])),
      field("tr-speaker", "رقم المتحدث ( speaker_id )", el("input", { type: "number", id: "tr-speaker", min: "0", max: "109", value: "0" })),
      field("tr-epochs", "عدد الدورات ( epochs )", el("input", { type: "number", id: "tr-epochs", min: "1", max: "10000", value: "200" })),
      field("tr-batch", "حجم الدفعة ( batch size )", el("input", { type: "number", id: "tr-batch", min: "1", max: "64", value: "6" })),
      field("tr-lr", "معدل التعلم ( learning rate )", el("input", { type: "number", id: "tr-lr", step: "any", value: "0.0001" })),
      field("tr-lr-decay", "تضاؤل معدل التعلم ( lr_decay )", el("input", { type: "number", id: "tr-lr-decay", step: "any", value: "0.999875" })),
      field("tr-save-every", "حفظ كل ( epochs )", el("input", { type: "number", id: "tr-save-every", min: "1", max: "200", value: "10" })),
    ]),
    el("div", { class: "row mt" }, [
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", id: "tr-f0", checked: true }),
        el("span", {}, ["تدريب بالطبقة ", el("span", { class: "opt-latin" }, ["f0"])]),
      ]),
      el("div", { class: "field grow" }, [
        el("label", { for: "tr-f0-method" }, ["طريقة f0"]),
        el("select", { id: "tr-f0-method" }, [
          el("option", { value: "rmvpe", selected: true }, ["rmvpe"]),
          el("option", { value: "fcpe" }, ["fcpe"]),
          el("option", { value: "pm" }, ["pm"]),
        ]),
      ]),
    ]),
    el("div", { class: "row mt" }, [
      el("button", { class: "btn", type: "button", id: "tr-prepare" }, [icon("cpu", 15), " تجهيز البيانات"]),
      el("button", { class: "btn primary", type: "button", id: "tr-start" }, [icon("play", 15), " بدء التدريب"]),
    ]),
    el("p", { class: "small muted", style: "margin:10px 0 0" }, [
      "\"تجهيز البيانات\" يعالج مجلد التسجيلات (قص + F0 + ميزات + فهرس). \"بدء التدريب\" يطلق عملية مستقلة على الخادم ويمكن إيقافها لاحقًا.",
    ]),
  ]);
}

function field(id, label, control, hint) {
  return el("div", { class: "field" }, [
    el("label", { for: id }, [label]),
    control,
    hint ? el("span", { class: "hint" }, [hint]) : null,
  ]);
}

function buildPrepareCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["تجهيز البيانات", el("span", { class: "sub latin" }, ["Prepare"])]),
    el("div", { id: "tr-prepare-box" }, [
      el("p", { class: "muted small" }, ["لم يبدأ التجهيز بعد — املأ اسم التجربة والمسار واضغط \"تجهيز البيانات\"."]),
    ]),
  ]);
}

function buildMonitorCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "مراقبة التدريب الحية",
      el("span", { class: "sub latin" }, ["Live Monitor"]),
      el("span", { class: "badge idle dot", id: "tr-monitor-badge" }, ["—"]),
    ]),
    el("div", { class: "progress", id: "tr-monitor-bar-wrap" }, [el("div", { class: "bar", id: "tr-monitor-bar" })]),
    el("div", { class: "progress-label" }, [el("span", { id: "tr-monitor-label" }, ["—"])]),
    el("div", { class: "row mt", id: "tr-monitor-meta" }, []),
    el("div", { class: "table-wrap mt" }, [
      el("table", { class: "tbl" }, [
        el("thead", {}, [el("tr", {}, [
          el("th", {}, ["الخسارة (Loss)"]),
          el("th", {}, ["القيمة الحالية"]),
        ])]),
        el("tbody", { id: "tr-losses-body" }, []),
      ]),
    ]),
    el("div", { class: "chart-box mt" }, [
      el("canvas", { class: "chart", id: "tr-chart", width: "600", height: "220" }),
      el("div", { class: "chart-legend" }, LOSS_KEYS.map((k) =>
        el("span", { class: "li" }, [
          el("span", { class: "sw", style: `background:${LOSS_COLORS[k]}` }),
          document.createTextNode(`${LOSS_AR[k]} (${k})`),
        ]),
      )),
    ]),
    el("div", { class: "row mt" }, [
      el("button", { class: "btn danger", type: "button", id: "tr-stop" }, [icon("stop", 15), " إيقاف"]),
      el("button", { class: "btn", type: "button", id: "tr-resume" }, [icon("play", 15), " استكمال"]),
      el("button", { class: "btn success", type: "button", id: "tr-export" }, [icon("download", 15), " تصدير النموذج"]),
    ]),
    el("p", { class: "small muted", style: "margin:8px 0 0" }, [
      "التصدير ينسج آخر نقطة حفظ إلى سجل النماذج لتجهيزها للتحويل. الاستكمال يُكمل آخر تجربة (أو المحددة).",
    ]),
    el("div", { id: "tr-monitor-box", class: "mt" }, []),
  ]);
}

function buildRunsCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "التجارب المحفوظة",
      el("span", { class: "sub latin" }, ["Runs"]),
    ]),
    el("div", { id: "tr-runs-box" }, [
      el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
    ]),
  ]);
}

function buildCheckpointsCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "نقاط الحفظ",
      el("span", { class: "sub latin" }, ["Checkpoints"]),
      el("button", { class: "btn sm", type: "button", id: "tr-ckpt-refresh", onclick: () => window.dispatchEvent(new Event("ivs-ckpt-refresh")) }, [icon("refresh", 14), " تحديث"]),
    ]),
    el("div", { id: "tr-ckpt-box" }, []),
  ]);
}

function buildLogsCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "سجل التدريب",
      el("span", { class: "sub latin" }, ["train.log"]),
      el("button", { class: "btn sm", type: "button", id: "tr-logs-refresh" }, [icon("refresh", 14), " تحديث"]),
    ]),
    el("div", { class: "small muted" }, ["آخر 200 سطر — نص خام من الخادم."]),
    el("pre", { class: "log-viewer", id: "tr-logs" }, ["—"]),
  ]);
}

/* ------------------------------ form wiring ------------------------------ */

function wireForm(ui, state) {
  ui.btnPrepare.addEventListener("click", () => runPrepare(state, ui));
  ui.btnStart.addEventListener("click", () => runStart(state, ui));
  ui.btnStop.addEventListener("click", () => runStop(state, ui));
  ui.btnResume.addEventListener("click", () => runResume(state, ui));
  ui.btnExport.addEventListener("click", () => runExport(state, ui));
  ui.btnLogs.addEventListener("click", () => refreshLogs(state, ui));
  state._ckptHandler = () => refreshCheckpoints(state, ui);
  window.addEventListener("ivs-ckpt-refresh", state._ckptHandler);
}

function formPayload(ui) {
  return {
    experiment_name: ui.exp.value.trim(),
    dataset_path: ui.dataset.value.trim(),
    sample_rate: ui.sr.value,
    epochs: Number(ui.epochs.value || 200),
    batch_size: Number(ui.batch.value || 6),
    learning_rate: Number(ui.lr.value || 1e-4),
    lr_decay: Number(ui.lrDecay.value || 0.999875),
    save_every_epoch: Number(ui.saveEvery.value || 10),
    f0: ui.f0.checked,
    f0_method: ui.f0Method.value,
    speaker_id: Number(ui.speaker.value || 0),
    version: "v2",
  };
}

function validateForm(ui) {
  const payload = formPayload(ui);
  if (!payload.experiment_name) {
    toast("أدخل اسم التجربة (أحرف إنجليزية وأرقام و - _ فقط).", "warn");
    return null;
  }
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(payload.experiment_name)) {
    toast("اسم التجربة غير صالح — استخدم أحرفًا إنجليزية وأرقامًا وشرطات فقط.", "warn");
    return null;
  }
  if (!payload.dataset_path) {
    toast("أدخل مسار مجلد البيانات على الخادم.", "warn");
    return null;
  }
  return payload;
}

/* ------------------------------ prepare ------------------------------ */

async function runPrepare(state, ui) {
  const payload = validateForm(ui);
  if (!payload) return;
  setBusy(ui.btnPrepare, true, "جارٍ بدء التجهيز…");
  try {
    await post("/training/prepare", {
      experiment_name: payload.experiment_name,
      dataset_path: payload.dataset_path,
      sample_rate: payload.sample_rate,
      version: payload.version,
      f0: payload.f0,
      f0_method: payload.f0_method,
      speaker_id: payload.speaker_id,
    });
    state.prepareExperiment = payload.experiment_name;
    state.currentExperiment = payload.experiment_name;
    toast("بدأ تجهيز البيانات في الخلفية.", "success");
    startPreparePolling(state, ui);
  } catch (err) {
    toast(err.message, "error", 10000);
    renderPrepare(state, ui, { stage: "failed", error: err.message });
  } finally {
    setBusy(ui.btnPrepare, false);
  }
}

function startPreparePolling(state, ui) {
  if (state.prepareTimer) clearInterval(state.prepareTimer);
  state.prepareTimer = setInterval(async () => {
    if (!state.prepareExperiment) { clearInterval(state.prepareTimer); state.prepareTimer = null; return; }
    try {
      const qsExp = `?experiment=${encodeURIComponent(state.prepareExperiment)}`;
      const status = await get(`/training/prepare/status${qsExp}`);
      renderPrepare(state, ui, status);
      if (status.stage === "done" || status.stage === "failed") {
        clearInterval(state.prepareTimer);
        state.prepareTimer = null;
        if (status.stage === "done") toast("اكتمل تجهيز البيانات.", "success");
      }
    } catch (err) {
      toast(err.message, "error");
      clearInterval(state.prepareTimer);
      state.prepareTimer = null;
    }
  }, 2000);
}

function renderPrepare(state, ui, status) {
  const box = ui.prepBox;
  clear(box);
  const stage = status?.stage || "idle";
  const stageAr = STAGE_AR[stage] || stage;
  const frac = Number(status?.fraction || 0);

  box.append(el("div", { class: "row" }, [
    el("span", { class: `badge ${stage === "done" ? "ok" : stage === "failed" ? "error" : stage === "idle" ? "idle" : "warn"} dot` }, [stageAr]),
    status?.file ? el("span", { class: "latin small dim" }, [status.file]) : null,
  ]));

  if (stage === "idle") {
    box.append(el("p", { class: "muted small" }, ["لم يبدأ التجهيز بعد."]));
    return;
  }

  if (stage === "failed") {
    box.append(el("div", { class: "notice danger" }, [el("p", {}, [status?.error || "فشل التجهيز بسبب غير معروف."])]));
    return;
  }

  box.append(el("div", { class: "mt" }, [
    el("div", { class: `progress ${stage === "done" ? "" : "amber"}` }, [
      el("div", { class: "bar", style: `width:${(clamp(frac, 0, 1) * 100).toFixed(1)}%` }),
    ]),
    el("div", { class: "progress-label" }, [
      el("span", {}, [`${stageAr} — ${status?.file || ""}`]),
      el("span", { class: "latin" }, [fmtPct(frac)]),
    ]),
  ]));

  if (status?.report) {
    box.append(el("div", { class: "notice ok mt" }, [
      el("p", { class: "latin small" }, [JSON.stringify(status.report)]),
    ]));
  }
  if (status?.updated_at) {
    box.append(el("p", { class: "small muted mt", style: "margin-bottom:0" }, [
      `آخر تحديث: ${fmtTime(status.updated_at)}`,
    ]));
  }
}

/* ------------------------------ start / stop / resume / export ------------------------------ */

async function runStart(state, ui) {
  const payload = validateForm(ui);
  if (!payload) return;
  setBusy(ui.btnStart, true, "جارٍ بدء التدريب…");
  try {
    const res = await post("/training/start", {
      experiment_name: payload.experiment_name,
      dataset_path: payload.dataset_path,
      sample_rate: payload.sample_rate,
      epochs: payload.epochs,
      batch_size: payload.batch_size,
      learning_rate: payload.learning_rate,
      lr_decay: payload.lr_decay,
      save_every_epoch: payload.save_every_epoch,
      save_every_weights: true,
      save_latest: true,
      cache_in_gpu: false,
      preload_base_models: true,
      gpu: null,
      f0: payload.f0,
      f0_method: payload.f0_method,
      version: payload.version,
      speaker_id: payload.speaker_id,
      resume: true,
      clean_start: false,
    });
    state.currentExperiment = res?.run || payload.experiment_name;
    toast(`انطلق التدريب (PID ${res?.pid ?? "؟"}) — تجربة ${state.currentExperiment}.`, "success");
    await loadRuns(state, ui);
    refreshCheckpoints(state, ui);
    refreshLogs(state, ui);
  } catch (err) {
    toast(err.message, "error", 10000);
  } finally {
    setBusy(ui.btnStart, false);
  }
}

async function runStop(state, ui) {
  setBusy(ui.btnStop, true, "جارٍ الإيقاف…");
  try {
    const res = await post("/training/stop");
    if (res?.running === false) toast("أُوقف التدريب.", "success");
    else toast("لا توجد عملية تدريب نشطة.", "info");
    await loadRuns(state, ui);
  } catch (err) { toast(err.message, "error"); }
  finally { setBusy(ui.btnStop, false); }
}

async function runResume(state, ui) {
  setBusy(ui.btnResume, true, "جارٍ الاستكمال…");
  try {
    const experiment = state.currentExperiment || "";
    const path = `/training/resume${experiment ? `?experiment=${encodeURIComponent(experiment)}` : ""}`;
    const res = await post(path);
    toast(`استُكمل التدريب (PID ${res?.pid ?? "؟"}) — ${res?.run || experiment || "أحدث تجربة"}.`, "success");
    await loadRuns(state, ui);
  } catch (err) { toast(err.message, "error"); }
  finally { setBusy(ui.btnResume, false); }
}

async function runExport(state, ui) {
  const experiment = state.currentExperiment;
  if (!experiment) {
    toast("حدد تجربة من قائمة التجارب أولًا (أو شغّل تدريبًا).", "warn");
    return;
  }
  setBusy(ui.btnExport, true, "جارٍ التصدير…");
  try {
    const res = await post("/training/export", { experiment, epoch: null });
    toast(`تم التصدير: ${res?.model_path || ""}${res?.index_path ? " + فهرس" : ""}`, "success");
    if (res?.registered) toast(`سُجِّل النموذج \"${res.registered.name}\" في تبويب النماذج.`, "success", 8000);
  } catch (err) { toast(err.message, "error", 10000); }
  finally { setBusy(ui.btnExport, false); }
}

/* ------------------------------ live monitor ------------------------------ */

function renderMonitor(state, ui, status) {
  const box = ui.monBox;
  clear(box);
  const training = status || {};
  const st = training.state || "idle";
  ui.monBadge.className = `badge ${st} dot`;
  ui.monBadge.textContent = TRAIN_STATES[st] || st;

  const total = training.total_epochs || 0;
  const epoch = training.epoch || 0;
  const frac = total > 0 ? clamp(epoch / total, 0, 1) : 0;
  ui.monBar.style.width = `${(frac * 100).toFixed(1)}%`;
  clear(ui.monLabel);
  ui.monLabel.append(document.createTextNode(`الدورات: ${fmtNum(epoch, 0)} / ${fmtNum(total, 0)} (${fmtPct(frac)})`));

  const meta = document.getElementById("tr-monitor-meta");
  clear(meta);
  meta.append(
    el("span", { class: "badge" }, [`خطوة عامة: ${fmtNum(training.global_step || 0, 0)}`]),
    training.steps_per_epoch ? el("span", { class: "badge" }, [`خطوات/دورة: ${fmtNum(training.steps_per_epoch, 0)}`]) : null,
    training.updated_at ? el("span", { class: "badge" }, [`آخر تحديث: ${fmtTime(training.updated_at)}`]) : null,
    (training.message && training.message.includes(" ")) ? el("span", { class: "badge info" }, [training.message]) : null,
  );

  renderLosses(state, ui, training.latest_losses || {});

  if (st === "running" || st === "preparing") {
    box.append(el("p", { class: "small muted" }, ["تحديث تلقائي كل 3 ثوانٍ…"]));
  } else if (training.message && !training.message.includes(" ")) {
    box.append(el("p", { class: "small muted" }, [`سياق التجربة الحالية: ${training.message}`]));
  } else {
    box.append(el("p", { class: "small muted" }, [training.message || "لا توجد عملية تدريب."]));
  }
}

function renderLosses(state, ui, losses) {
  const tbody = ui.monLosses;
  clear(tbody);
  const keys = ["disc", "gen", "mel", "kl", "fm", "lr"];
  let any = false;
  for (const k of keys) {
    if (losses[k] === undefined || losses[k] === null) continue;
    any = true;
    tbody.append(el("tr", {}, [
      el("td", {}, [`${LOSS_AR[k] || k} — `, el("span", { class: "latin small dim" }, [k])]),
      el("td", { class: "num" }, [fmtNum(losses[k], 6)]),
    ]));
  }
  if (!any) tbody.append(el("tr", {}, [el("td", { colspan: "2", class: "muted" }, ["لا توجد قيم خسارة بعد."])]));

  // append to client-side history when the trainer advances
  const marker = `${losses.lr ?? ""}|${losses.disc ?? ""}|${losses.gen ?? ""}|${losses.mel ?? ""}|${losses.kl ?? ""}|${losses.fm ?? ""}`;
  if (any && marker !== state.lastStep) {
    state.lastStep = marker;
    for (const k of LOSS_KEYS) {
      if (losses[k] !== undefined && losses[k] !== null && Number.isFinite(Number(losses[k]))) {
        state.history[k].push(Number(losses[k]));
        if (state.history[k].length > CHART_KEEP) state.history[k].shift();
      }
    }
    drawChart(ui.chart, state.history);
  }
}

/* ------------------------------ chart ------------------------------ */

function drawChart(canvas, history) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.width || 600;
  const height = 220;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  // frame + grid
  ctx.strokeStyle = "#1f2b45";
  ctx.lineWidth = 1;
  for (let gy = 0; gy <= 4; gy++) {
    const y = (height - 16) * gy / 4 + 8;
    ctx.beginPath();
    ctx.moveTo(34, y);
    ctx.lineTo(width - 8, y);
    ctx.stroke();
  }

  const n = Math.max(...LOSS_KEYS.map((k) => history[k].length));
  if (!n) {
    ctx.fillStyle = "#6f7d9c";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("no loss history yet", width / 2, height / 2);
    return;
  }

  const norm = (k) => {
    const arr = history[k];
    if (!arr.length) return { lo: 0, hi: 1 };
    let lo = Math.min(...arr);
    let hi = Math.max(...arr);
    if (hi === lo) { hi = lo + 1e-9; }
    return { lo, hi };
  };
  const scales = {};
  for (const k of LOSS_KEYS) scales[k] = norm(k);

  const xAt = (i) => 34 + (width - 46) * (n > 1 ? i / (n - 1) : 0);
  const yAt = (k, v) => {
    const s = scales[k];
    const f = clamp((v - s.lo) / (s.hi - s.lo), 0, 1);
    return height - 16 - f * (height - 32);
  };

  for (const k of LOSS_KEYS) {
    const arr = history[k];
    if (!arr.length) continue;
    ctx.strokeStyle = LOSS_COLORS[k];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = xAt(i);
      const y = yAt(k, v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // endpoint marker
    ctx.fillStyle = LOSS_COLORS[k];
    ctx.beginPath();
    ctx.arc(xAt(arr.length - 1), yAt(k, arr[arr.length - 1]), 2.4, 0, Math.PI * 2);
    ctx.fill();
  }

  // axis labels (normalized 0..1 per-series)
  ctx.fillStyle = "#6f7d9c";
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  for (let gy = 0; gy <= 4; gy++) {
    const y = (height - 16) * (4 - gy) / 4 + 8;
    ctx.fillText(String(gy / 4), 30, y + 3);
  }
}

/* ------------------------------ runs list ------------------------------ */

async function loadRuns(state, ui) {
  const box = ui.runsBox;
  try {
    const runs = await get("/training/runs");
    clear(box);
    if (!Array.isArray(runs) || !runs.length) {
      box.append(el("p", { class: "muted" }, ["لا توجد تجارب محفوظة بعد."]));
      return;
    }
    const tbody = el("tbody");
    for (const run of runs) {
      const isCurrent = run.experiment === state.currentExperiment;
      const tr = el("tr", {}, [
        el("td", { class: "latin" }, [run.experiment]),
        el("td", { class: "num" }, [String(run.epoch ?? 0)]),
        el("td", { class: "num" }, [String(run.global_step ?? 0)]),
        el("td", { class: "latin small" }, [fmtTime(run.updated_at)]),
        el("td", {}, [run.has_checkpoints ? el("span", { class: "badge ok" }, ["نعم"]) : el("span", { class: "badge idle" }, ["لا"])]),
        el("td", {}, [
          el("button", {
            class: `btn sm ${isCurrent ? "success" : ""}`, type: "button",
            onclick: () => selectRun(state, ui, run.experiment),
          }, [isCurrent ? "محددة حاليًا" : "تحديد"]),
        ]),
      ]);
      tbody.append(tr);
    }
    box.append(el("div", { class: "table-wrap" }, [
      el("table", { class: "tbl" }, [
        el("thead", {}, [el("tr", {}, [
          el("th", {}, ["التجربة"]),
          el("th", {}, ["الدورة"]),
          el("th", {}, ["الخطوة"]),
          el("th", {}, ["آخر تحديث"]),
          el("th", {}, ["نقاط حفظ"]),
          el("th", {}, [""]),
        ])]),
        tbody,
      ]),
    ]));
    if (!state.currentExperiment) {
      state.currentExperiment = runs[0]?.experiment || "";
      refreshCheckpoints(state, ui);
      refreshLogs(state, ui);
    }
  } catch (err) {
    clear(box);
    box.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
  }
}

function selectRun(state, ui, experiment) {
  state.currentExperiment = experiment;
  loadRuns(state, ui);
  refreshCheckpoints(state, ui);
  refreshLogs(state, ui);
  toast(`تم تحديد سياق التجربة: ${experiment}`, "info");
}

/* ------------------------------ checkpoints + logs ------------------------------ */

async function refreshCheckpoints(state, ui) {
  const box = ui.ckptBox;
  if (!state.currentExperiment) {
    clear(box);
    box.append(el("p", { class: "muted small" }, ["حدد تجربة لعرض نقاط الحفظ."]));
    return;
  }
  try {
    const list = await get(`/training/runs/${encodeURIComponent(state.currentExperiment)}/checkpoints`);
    clear(box);
    if (!Array.isArray(list) || !list.length) {
      box.append(el("p", { class: "muted small" }, ["لا توجد نقاط حفظ لهذه التجربة بعد."]));
      return;
    }
    const tbody = el("tbody");
    for (const c of list) {
      tbody.append(el("tr", {}, [
        el("td", { class: "latin" }, [c.run_id || "—"]),
        el("td", {}, [el("span", { class: `badge ${String(c.kind).startsWith("G") ? "ok" : "info"}` }, [c.kind])]),
        el("td", { class: "num" }, [String(c.epoch ?? 0)]),
        el("td", { class: "num" }, [String(c.global_step ?? 0)]),
        el("td", { class: "latin small" }, [fmtTime(c.created_at)]),
        el("td", { class: "num" }, [fmtMB(c.size_mb)]),
        el("td", {}, [resumeFromButton(state, ui, c)]),
      ]));
    }
    box.append(el("div", { class: "table-wrap" }, [
      el("table", { class: "tbl" }, [
        el("thead", {}, [el("tr", {}, [
          el("th", {}, ["المعرّف"]),
          el("th", {}, ["النوع"]),
          el("th", {}, ["الدورة"]),
          el("th", {}, ["الخطوة"]),
          el("th", {}, ["التاريخ"]),
          el("th", {}, ["الحجم"]),
          el("th", {}, ["إجراء"]),
        ])]),
        tbody,
      ]),
    ]));
  } catch (err) {
    clear(box);
    box.append(el("p", { class: "muted small" }, [err.message]));
  }
}

function resumeFromButton(state, ui, c) {
  const name = Path(c.path).name; // c.path comes from the server (G_*.pth)
  const btn = el("button", { class: "btn sm ghost", type: "button" }, ["استكمال من هنا"]);
  btn.addEventListener("click", async () => {
    if (!name) { toast("لا يمكن تحديد نقطة الحفظ.", "warn"); return; }
    setBusy(btn, true, "جارٍ الاستكمال…");
    try {
      await post(`/training/resume?experiment=${encodeURIComponent(state.currentExperiment)}&checkpoint=${encodeURIComponent(name)}`);
      toast(`بدأ استكمال التدريب من ${name}.`, "success");
    } catch (err) {
      toast(err.message, "error", 10000);
    } finally { setBusy(btn, false); }
  });
  return btn;
}

function Path(p) { // minimal basename helper without importing node:path
  const s = String(p || "");
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  return { name: i >= 0 ? s.slice(i + 1) : s };
}

async function refreshLogs(state, ui) {
  const box = ui.logsBox;
  if (!state.currentExperiment) {
    box.textContent = "—";
    return;
  }
  try {
    const text = await getText(`/training/runs/${encodeURIComponent(state.currentExperiment)}/logs?lines=200`);
    box.textContent = text && text.trim() ? text : "(السجل فارغ)";
    box.scrollTop = box.scrollHeight;
  } catch (err) {
    box.textContent = err.message;
  }
}
