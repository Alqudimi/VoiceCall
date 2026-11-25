/* ============================================================
   convert.js — تحويل الملفات: اختيار ملف صوتي، إعدادات التحويل،
   تنفيذ POST /api/convert، ثم مقارنة الأصلي والمحوّل.
   ============================================================ */

import {
  get, postFormWithProgress, getBlob, toast, el, clear, icon, setBusy,
  fmtNum, fmtDuration, fmtMB, f0MethodSelect,
} from "../api.js";

export default { mount };

const AUDIO_ACCEPT = ".wav,.mp3,.flac,.ogg,.m4a,.aac,.opus,.webm,audio/*";

async function mount(root) {
  const state = {
    file: null,
    fileUrl: null,
    fileDuration: null,
    settings: null,
    result: null,
    resultUrl: null,
  };

  root.replaceChildren(el("div", { class: "stack" }, [
    buildEngineCard(),
    el("div", { class: "grid" }, [
      buildInputCard(),
      buildSettingsCard(),
    ]),
    buildResultCard(),
  ]));

  const input = {
    dz: root.querySelector("#convert-dropzone"),
    input: root.querySelector("#convert-file"),
    btn: root.querySelector("#convert-btn"),
  };

  wireDropzone(input, state);
  wireRecorder(state);

  input.btn.addEventListener("click", () => runConvert(state, input));

  await loadSettings(state);

  function cleanup() {
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
    if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  }
  return cleanup;
}

/* ------------------------------ engine state ------------------------------ */

function buildEngineCard() {
  return el("div", { class: "card", id: "convert-engine" }, [
    el("h2", { class: "card-title" }, [
      "حالة المحرك",
      el("span", { class: "sub latin" }, ["Engine"]),
    ]),
    el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
  ]);
}

async function loadSettings(state) {
  try {
    state.settings = await get("/convert/settings");
    renderEngine(state, state.settings.engine);
    applyDefaults(state.settings);
  } catch (err) {
    const card = document.getElementById("convert-engine");
    if (card) {
      clear(card);
      card.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
    }
  }
}

function renderEngine(state, engine) {
  const card = document.getElementById("convert-engine");
  if (!card || !engine) return;
  clear(card);
  card.append(el("h2", { class: "card-title" }, ["حالة المحرك", el("span", { class: "sub latin" }, ["Engine"])]));
  const ready = Boolean(engine.status === "ready" && engine.loaded_model);
  if (state) state.engineReady = ready;
  const badge = el("span", { class: `badge ${engine.status} dot` }, [
    engine.status === "ready" ? "جاهز" : engine.status === "loading" ? "قيد التحميل" : engine.status === "busy" ? "مشغول" : engine.status === "error" ? "خطأ" : "غير متاح",
  ]);
  card.append(el("div", { class: "row" }, [badge, engine.loaded_model
    ? el("span", { class: "dim" }, [`النموذج: ${engine.loaded_model.name}`])
    : null]));
  if (engine.message) {
    card.append(el("p", { class: "small latin muted", style: "direction:ltr;text-align:left;margin:8px 0 0" }, [engine.message]));
  }
  if (!ready) {
    card.append(el("div", { class: "notice warn mt" }, [
      el("span", { class: "n-icon" }, [icon("alert", 18)]),
      el("p", {}, [
        "يتطلب التحويل نموذجًا محمّلًا في المحرك — ",
        el("a", { href: "#models" }, ["حمّل نموذجًا أولًا من تبويب النماذج"]),
        ".",
      ]),
    ]));
  }
  const idxRow = document.getElementById("row-index_rate");
  if (idxRow) idxRow.classList.toggle("hidden", !(engine.loaded_model && engine.loaded_model.index_path));
  updateConvertButton(state);
}

/* ------------------------------ input card ------------------------------ */

function buildInputCard() {
  const fileInput = el("input", { type: "file", accept: AUDIO_ACCEPT, id: "convert-file" });
  const dropzone = el("div", { class: "dropzone", id: "convert-dropzone", tabindex: "0", role: "button" }, [
    el("span", { class: "dz-icon" }, [icon("upload", 26)]),
    el("span", { class: "dz-title" }, ["اسحب ملفًا صوتيًا هنا أو اضغط للاختيار"]),
    el("span", { class: "dz-sub" }, ["WAV / MP3 / FLAC / OGG / M4A — الملف يُرسل كاملًا إلى الخادم للتحويل."]),
    fileInput,
  ]);
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["الملف المصدر", el("span", { class: "sub latin" }, ["Input"])]),
    dropzone,
    el("div", { class: "record-box mt", id: "convert-record-box" }, [
      el("div", { class: "row" }, [
        el("button", { class: "btn", type: "button", id: "convert-record-btn" }, [
          icon("mic", 16), " تسجيل من الميكروفون",
        ]),
        el("button", { class: "btn danger hidden", type: "button", id: "convert-record-stop" }, [
          icon("stop", 16), " إيقاف التسجيل",
        ]),
        el("span", { class: "badge hidden", id: "convert-record-timer" }, ["00:00"]),
        el("span", { class: "level-meter hidden", id: "convert-record-level" }, [el("span")]),
      ]),
      el("p", { class: "hint", id: "convert-record-hint" }, [
        "بديل عن الملف: سجّل صوتك مباشرة ثم حوّله — يُعامل التسجيل كملف مدخل.",
      ]),
    ]),
    el("div", { class: "mt", id: "convert-fileinfo" }, []),
  ]);
}

/* ------------------------------ mic recording ------------------------------ */

function wireRecorder(state) {
  const btn = document.getElementById("convert-record-btn");
  const stopBtn = document.getElementById("convert-record-stop");
  const timer = document.getElementById("convert-record-timer");
  const level = document.getElementById("convert-record-level");
  const hint = document.getElementById("convert-record-hint");
  if (!btn || !navigator.mediaDevices || !window.MediaRecorder) return;

  let recorder = null;
  let chunks = [];
  let stream = null;
  let tick = null;
  let audioCtx = null;
  let analyser = null;
  let raf = null;
  let startedAt = 0;

  const supported = MediaRecorder.isTypeSupported
    && MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";

  btn.addEventListener("click", async () => {
    if (recorder && recorder.state === "recording") return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      hint.textContent = "تعذّر الوصول إلى الميكروفون (يتطلب HTTPS أو localhost، و إذن المتصفح): " + err.message;
      toast("تعذّر بدء التسجيل: " + err.message, "error");
      return;
    }
    chunks = [];
    recorder = new MediaRecorder(stream, supported ? { mimeType: supported } : undefined);
    recorder.ondataavailable = (ev) => { if (ev.data && ev.data.size) chunks.push(ev.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      const ext = (recorder.mimeType || "audio/webm").includes("ogg") ? "ogg" : "webm";
      const file = new File([blob], `mic_recording_${Date.now()}.${ext}`, { type: blob.type });
      // the recording flows through the exact same file-conversion pipeline
      selectFile(file, state);
      teardown();
      toast("تم التسجيل — جاهز للتحويل.", "success");
    };
    recorder.start(250);
    startedAt = Date.now();
    btn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    timer.classList.remove("hidden");
    level.classList.remove("hidden");
    // live timer + level meter
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const bar = level.querySelector("span");
    const loop = () => {
      analyser.getByteTimeDomainData(buf);
      let peak = 0;
      for (let i = 0; i < buf.length; i += 1) peak = Math.max(peak, Math.abs(buf[i] - 128));
      bar.style.width = `${Math.min(100, (peak / 128) * 100)}%`;
      raf = requestAnimationFrame(loop);
    };
    loop();
    tick = setInterval(() => {
      const s = Math.floor((Date.now() - startedAt) / 1000);
      timer.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
    }, 500);
  });

  stopBtn.addEventListener("click", () => {
    if (recorder && recorder.state === "recording") recorder.stop();
  });

  function teardown() {
    if (tick) clearInterval(tick);
    if (raf) cancelAnimationFrame(raf);
    if (audioCtx) audioCtx.close().catch(() => {});
    if (stream) stream.getTracks().forEach((t) => t.stop());
    recorder = null; stream = null; audioCtx = null; analyser = null; tick = null; raf = null;
    btn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    timer.classList.add("hidden");
    level.classList.add("hidden");
    timer.textContent = "00:00";
  }
}

function wireDropzone(input, state) {
  const { dz, input: fileInput } = input;
  dz.addEventListener("click", () => fileInput.click());
  dz.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fileInput.click(); }
  });
  for (const evt of ["dragenter", "dragover"]) {
    dz.addEventListener(evt, (ev) => { ev.preventDefault(); dz.classList.add("over"); });
  }
  for (const evt of ["dragleave", "drop"]) {
    dz.addEventListener(evt, (ev) => { ev.preventDefault(); dz.classList.remove("over"); });
  }
  dz.addEventListener("drop", (ev) => {
    const files = Array.from(ev.dataTransfer?.files || []);
    const audio = files.find((f) => f.type.startsWith("audio") || /\.(wav|mp3|flac|ogg|m4a|aac|opus|webm)$/i.test(f.name));
    if (audio) selectFile(audio, state);
    else toast("الملف المُسقط ليس ملفًا صوتيًا مدعومًا.", "error");
  });
  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    if (f) selectFile(f, state);
  });
}

function selectFile(file, state) {
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = file;
  state.fileDuration = null;
  state.fileUrl = URL.createObjectURL(file);
  const info = document.getElementById("convert-fileinfo");
  clear(info);
  const nameEl = el("div", { class: "latin small dim" }, [file.name]);
  const sizeEl = el("div", { class: "small muted" }, [fmtMB(file.size / (1024 * 1024))]);
  const durEl = el("div", { class: "small muted" }, ["جارٍ قراءة المدة…"]);
  info.append(nameEl, sizeEl, durEl);
  const probe = el("audio", { src: state.fileUrl, preload: "metadata" });
  probe.addEventListener("loadedmetadata", () => {
    state.fileDuration = Number.isFinite(probe.duration) ? probe.duration : null;
    clear(durEl);
    durEl.append(document.createTextNode(state.fileDuration ? `المدة: ${fmtDuration(state.fileDuration)}` : "تعذّر تحديد المدة"));
  });
  probe.addEventListener("error", () => {
    clear(durEl);
    durEl.append(document.createTextNode("تعذّر قراءة بيانات الملف الصوتي في المتصفح (سيحاول الخادم معالجته)."));
  });
  updateConvertButton(state);
}

function updateConvertButton(state) {
  const btn = document.getElementById("convert-btn");
  if (!btn) return;
  btn.disabled = !(state && state.file && state.engineReady);
}

/* ------------------------------ settings card ------------------------------ */

function buildSettingsCard() {
  const srSelect = el("select", { id: "cv-out-sr" }, [
    el("option", { value: "" , selected: true }, ["معدل النموذج الأصلي (بدون تغيير)"]),
    el("option", { value: "32000" }, ["32 kHz"]),
    el("option", { value: "40000" }, ["40 kHz"]),
    el("option", { value: "44100" }, ["44.1 kHz"]),
    el("option", { value: "48000" }, ["48 kHz"]),
  ]);

  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["إعدادات التحويل", el("span", { class: "sub latin" }, ["Settings"])]),

    sliderRow("pitch_shift", "انزياح النغمة (Pitch)", "semitones", -24, 24, 1, 0, "cv-pitch", "cv-pitch-val"),
    el("div", { class: "field" }, [
      el("label", {}, ["طريقة استخراج الطبقة — ", el("span", { class: "latin" }, ["f0 method"])]),
      f0MethodSelect("rmvpe", "cv-f0"),
    ]),
    sliderRow("index_rate", "نسبة الفهرس — ", "Index Rate", 0, 1, 0.01, 0, "cv-index", "cv-index-val"),
    sliderRow("filter_radius", "نصف قطر المرشح — ", "Filter Radius", 0, 7, 1, 3, "cv-filter", "cv-filter-val"),
    sliderRow("rms_mix_rate", "مزج الشدة (RMS Mix)", "0..1", 0, 1, 0.01, 0.25, "cv-rms", "cv-rms-val"),
    sliderRow("protect", "حماية السواكن — ", "Protect", 0, 0.5, 0.01, 0.33, "cv-protect", "cv-protect-val"),
    el("div", { class: "field" }, [
      el("label", {}, ["معدل العينة الناتج — ", el("span", { class: "latin" }, ["output SR"])]),
      srSelect,
      el("span", { class: "hint" }, ["اتركه \"بدون تغيير\" لإخراج بمعدل النموذج."]),
    ]),
    el("div", { class: "row mt" }, [
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", id: "cv-denoise" }),
        el("span", {}, ["إزالة الضجيج قبل التحويل — ", el("span", { class: "opt-latin" }, ["denoise"])]),
      ]),
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", id: "cv-normalize", checked: true }),
        el("span", {}, ["توحيد مستوى الصوت — ", el("span", { class: "opt-latin" }, ["normalize"])]),
      ]),
    ]),
    el("div", { id: "convert-progress", class: "mt hidden" }, [
      el("div", { class: "progress amber", id: "convert-upload-progress" }, [el("div", { class: "bar" })]),
      el("div", { class: "progress-label" }, [
        el("span", { id: "convert-upload-progress-label" }, ["…"]),
      ]),
    ]),
    el("button", { class: "btn primary block mt", type: "button", id: "convert-btn", disabled: true }, [
      icon("play", 16), " تحويل الملف",
    ]),
  ]);
}

function sliderRow(name, label, latinLabel, min, max, step, def, inputId, valId) {
  const input = el("input", {
    type: "range", id: inputId, min: String(min), max: String(max),
    step: String(step), value: String(def), dataset: { name },
  });
  const val = el("span", { class: "slider-val", id: valId }, [formatValue(def, step)]);
  input.addEventListener("input", () => {
    clear(val);
    val.append(document.createTextNode(formatValue(input.value, step)));
  });
  return el("div", { class: "slider-row", id: `row-${name}` }, [
    el("div", { class: "slider-head" }, [
      el("label", { for: inputId }, [label, " ", el("span", { class: "opt-latin" }, [latinLabel])]),
      val,
    ]),
    input,
  ]);
}

function formatValue(value, step) {
  const n = Number(value);
  if (step >= 1) return String(Math.round(n));
  return n.toFixed(2);
}

function applyDefaults(settingsData) {
  if (!settingsData) return;
  const d = settingsData.defaults || {};
  setSlider("cv-pitch", d.pitch_shift);
  setSlider("cv-index", d.index_rate);
  setSlider("cv-filter", d.filter_radius);
  setSlider("cv-rms", d.rms_mix_rate);
  setSlider("cv-protect", d.protect);
  const f0 = document.getElementById("cv-f0");
  if (f0 && d.f0_method) f0.value = d.f0_method;
  const denoise = document.getElementById("cv-denoise");
  if (denoise) denoise.checked = Boolean(d.denoise);
  const normalize = document.getElementById("cv-normalize");
  if (normalize) normalize.checked = d.normalize !== false;
  for (const [id, step] of [["cv-pitch", 1], ["cv-index", 0.01], ["cv-filter", 1], ["cv-rms", 0.01], ["cv-protect", 0.01]]) {
    const input = document.getElementById(id);
    if (!input) continue;
    const label = input.closest(".slider-row").querySelector(".slider-val");
    if (label) { clear(label); label.append(document.createTextNode(formatValue(input.value, step))); }
  }
}

function setSlider(id, value) {
  const input = document.getElementById(id);
  if (input && value !== undefined && value !== null) input.value = String(value);
}

/* ------------------------------ convert run ------------------------------ */

async function runConvert(state, input) {
  const btn = input.btn;
  if (!state.file) { toast("اختر ملفًا صوتيًا أولًا.", "warn"); return; }
  let engine = null;
  try { engine = (await get("/convert/settings")).engine; }
  catch (err) { toast(err.message, "error"); return; }
  if (!engine.loaded_model || engine.status !== "ready") {
    toast("لا يوجد نموذج محمّل في المحرك — حمّل نموذجًا من تبويب النماذج أولًا.", "error");
    renderEngine(state, engine);
    return;
  }

  const form = new FormData();
  form.append("file", state.file, state.file.name);
  form.append("pitch_shift", document.getElementById("cv-pitch").value);
  form.append("f0_method", document.getElementById("cv-f0").value);
  form.append("index_rate", document.getElementById("cv-index").value);
  form.append("filter_radius", document.getElementById("cv-filter").value);
  form.append("rms_mix_rate", document.getElementById("cv-rms").value);
  form.append("protect", document.getElementById("cv-protect").value);
  const sr = document.getElementById("cv-out-sr").value;
  // omit the field entirely when "model native" is chosen so the server default (None) applies
  if (sr) form.append("output_sample_rate", sr);
  form.append("denoise", document.getElementById("cv-denoise").checked ? "true" : "false");
  form.append("normalize", document.getElementById("cv-normalize").checked ? "true" : "false");

  const progressWrap = document.getElementById("convert-progress");
  const bar = progressWrap.querySelector(".bar");
  const label = document.getElementById("convert-upload-progress-label");
  progressWrap.classList.remove("hidden");
  setBusy(btn, true, "جارٍ التحويل…");

  try {
    const result = await postFormWithProgress("/convert", form, (frac) => {
      bar.style.width = `${Math.round(frac * 100)}%`;
      clear(label);
      label.append(document.createTextNode(`رفع الملف: ${Math.round(frac * 100)}%`));
    });
    clear(label);
    label.append(document.createTextNode("المعالجة على الخادم…"));
    bar.style.width = "100%";
    state.result = result;
    await renderResult(state, result);
    toast("تم التحويل بنجاح.", "success");
  } catch (err) {
    toast(err.message, "error", 10000);
    if (err.detail) toast(err.detail, "error", 10000);
  } finally {
    setBusy(btn, false);
    updateConvertButton(state);
    progressWrap.classList.add("hidden");
    bar.style.width = "0%";
  }
}

/* ------------------------------ result ------------------------------ */

function buildResultCard() {
  return el("div", { class: "card hidden", id: "convert-result" }, [
    el("h2", { class: "card-title" }, ["النتيجة", el("span", { class: "sub latin" }, ["Result"])]),
    el("div", { id: "convert-result-body" }, []),
  ]);
}

async function renderResult(state, result) {
  const card = document.getElementById("convert-result");
  const body = document.getElementById("convert-result-body");
  card.classList.remove("hidden");
  clear(body);
  body.append(el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "جارٍ جلب الصوت المحوّل…"]));

  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  try {
    const blob = await getBlob(`/convert/results/${encodeURIComponent(result.output_path)}/download`);
    state.resultUrl = URL.createObjectURL(blob);
  } catch (err) {
    clear(body);
    body.append(el("div", { class: "notice danger" }, [el("p", {}, [`تعذّر جلب الناتج: ${err.message}`])]));
    return;
  }

  clear(body);
  const originalAudio = el("audio", { controls: true, src: state.fileUrl, preload: "metadata" });
  const convertedAudio = el("audio", { controls: true, src: state.resultUrl, preload: "metadata" });
  const download = el("a", {
    class: "btn primary", href: state.resultUrl, download: result.output_path || "converted.wav",
  }, [icon("download", 15), " تنزيل الناتج"]);

  const stats = el("div", { class: "latency-grid" }, [
    tile("مدة الملف", fmtDuration(result.duration_s)),
    tile("زمن المعالجة", fmtDuration(result.processing_s)),
    tile("معامل الزمن الحقيقي (RTF)", fmtNum(result.realtime_factor, 3)),
  ]);

  body.append(
    el("div", { class: "grid" }, [
      el("div", { class: "card", style: "box-shadow:none" }, [
        el("h3", { class: "small dim" }, ["الأصلي"]),
        originalAudio,
      ]),
      el("div", { class: "card", style: "box-shadow:none" }, [
        el("h3", { class: "small dim" }, ["المحوّل"]),
        convertedAudio,
      ]),
    ]),
    el("div", { class: "mt" }, [stats]),
    el("div", { class: "row mt" }, [download]),
  );

  const settingsSummary = result.settings || {};
  const keys = {
    pitch_shift: "انزياح النغمة",
    f0_method: "طريقة f0",
    index_rate: "نسبة الفهرس",
    filter_radius: "نصف قطر المرشح",
    rms_mix_rate: "مزج RMS",
    protect: "حماية",
    denoise: "إزالة الضجيج",
    normalize: "توحيد المستوى",
  };
  const chips = el("div", { class: "row small mt" });
  for (const [k, label] of Object.entries(keys)) {
    if (settingsSummary[k] === undefined) continue;
    const v = typeof settingsSummary[k] === "boolean" ? (settingsSummary[k] ? "نعم" : "لا") : settingsSummary[k];
    chips.append(el("span", { class: "badge" }, [`${label}: ${v}`]));
  }
  body.append(el("div", { class: "mt" }, [chips]));
  body.append(el("p", { class: "small muted mt", style: "margin-bottom:0" }, [
    "حُفظ هذا التحويل في ",
    el("a", { href: "#history" }, ["سجل التحويلات"]),
    " مع كل إعداداته.",
  ]));
}

function tile(label, value) {
  return el("div", { class: "stat-tile" }, [
    el("div", { class: "t-label" }, [label]),
    el("div", { class: "t-value" }, [value]),
  ]);
}
