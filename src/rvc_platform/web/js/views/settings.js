/* ============================================================
   settings.js — الإعدادات: رمز المصادقة، النفق العام، كشف النظام،
   سجلات المنصة، وبطاقة تعريف المنصة.
   ============================================================ */

import {
  get, post, toast, el, clear, icon, setBusy,
  getToken, setToken, clearToken, fmtMB, fmtGB, fmtTime,
} from "../api.js";

export default { mount };

async function mount(root) {
  root.replaceChildren(el("div", { class: "grid" }, [
    buildTokenCard(),
    buildTunnelCard(),
    buildDetectCard(),
    el("div", { class: "card span-2" }, [buildLogsCard()]),
    buildAboutCard(),
  ]));

  const ui = {
    tokenInput: root.querySelector("#st-token"),
    btnTokenSave: root.querySelector("#st-token-save"),
    btnTokenClear: root.querySelector("#st-token-clear"),
    btnTokenTest: root.querySelector("#st-token-test"),
    tokenState: root.querySelector("#st-token-state"),
    tunnelBox: root.querySelector("#st-tunnel-box"),
    btnTunnelStart: root.querySelector("#st-tunnel-start"),
    btnTunnelStop: root.querySelector("#st-tunnel-stop"),
    detectBox: root.querySelector("#st-detect-box"),
    logsBox: root.querySelector("#st-logs"),
    btnLogs: root.querySelector("#st-logs-refresh"),
    logLines: root.querySelector("#st-logs-lines"),
    aboutVersion: root.querySelector("#st-about-version"),
  };

  ui.tokenInput.value = getToken();
  wireToken(ui);
  wireTunnel(ui);
  ui.btnLogs.addEventListener("click", () => loadLogs(ui));

  await Promise.allSettled([
    loadTunnelStatus(ui),
    loadDetect(ui),
    loadLogs(ui),
    loadAbout(ui),
  ]);

  function cleanup() { /* no intervals — all polling stopped with the view */ }
  return cleanup;
}

/* ------------------------------ token ------------------------------ */

function buildTokenCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["رمز المصادقة", el("span", { class: "sub latin" }, ["Bearer Token"])]),
    el("div", { class: "field" }, [
      el("label", { for: "st-token" }, ["الرمز (من ملف logs/auth.token أو إعداد الخادم)"]),
      el("input", { type: "password", id: "st-token", placeholder: "ألصق الرمز هنا", autocomplete: "off" }),
      el("span", { class: "hint" }, ["يُحفظ محليًا في المتصفح (localStorage) ويُرسل مع كل طلب عبر ترويسة Authorization."]),
    ]),
    el("div", { class: "row mt" }, [
      el("button", { class: "btn primary", type: "button", id: "st-token-save" }, ["حفظ الرمز"]),
      el("button", { class: "btn danger", type: "button", id: "st-token-clear" }, ["مسح الرمز"]),
      el("button", { class: "btn", type: "button", id: "st-token-test" }, ["اختبار الرمز"]),
    ]),
    el("div", { class: "mt small", id: "st-token-state" }, ["—"]),
    el("div", { class: "notice info mt" }, [
      el("p", {}, ["إذا كان النفق العام مفعّلًا فإن الخادم يفرض الرمز إجباريًا على كل نداءات API."]),
    ]),
  ]);
}

function wireToken(ui) {
  renderTokenState(ui);
  ui.btnTokenSave.addEventListener("click", () => {
    const value = ui.tokenInput.value.trim();
    if (!value) {
      toast("أدخل رمزًا أولًا أو استخدم زر المسح.", "warn");
      return;
    }
    setToken(value);
    renderTokenState(ui);
    toast("تم حفظ الرمز — سيُطبَّق على الطلبات القادمة.", "success");
  });
  ui.btnTokenClear.addEventListener("click", () => {
    clearToken();
    ui.tokenInput.value = "";
    renderTokenState(ui);
    toast("تم مسح الرمز.", "info");
  });
  ui.btnTokenTest.addEventListener("click", async () => {
    setBusy(ui.btnTokenTest, true, "جارٍ الاختبار…");
    try {
      const system = await get("/system");
      toast(`الرمز يعمل — المحرك: ${system.engine}.`, "success");
    } catch (err) {
      toast(err.message, "error", 10000);
    } finally {
      setBusy(ui.btnTokenTest, false);
    }
  });
  ui.tokenInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") ui.btnTokenSave.click();
  });
}

function renderTokenState(ui) {
  const has = Boolean(getToken());
  clear(ui.tokenState);
  ui.tokenState.append(
    el("span", { class: `badge ${has ? "ok" : "warn"}` }, [has ? "رمز محفوظ" : "لا يوجد رمز"]),
    document.createTextNode(has ? " — يُستخدم للطلبات وWebSocket." : " — أضفه إن كان الخادم يطلب المصادقة."),
  );
}

/* ------------------------------ tunnel ------------------------------ */

function buildTunnelCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["النفق العام", el("span", { class: "sub latin" }, ["Cloudflare Tunnel"])]),
    el("p", { class: "small muted" }, [
      "ينشئ رابط trycloudflare.com مؤقتًا يعرض هذه المنصة للإنترنت. أثناء تشغيل النفق تُفرض المصادقة بالرمز.",
    ]),
    el("div", { class: "row" }, [
      el("button", { class: "btn primary", type: "button", id: "st-tunnel-start" }, [icon("play", 15), " بدء النفق"]),
      el("button", { class: "btn danger", type: "button", id: "st-tunnel-stop" }, [icon("stop", 15), " إيقاف النفق"]),
    ]),
    el("div", { class: "mt", id: "st-tunnel-box" }, [
      el("p", { class: "muted small" }, ["جارٍ جلب حالة النفق…"]),
    ]),
  ]);
}

function wireTunnel(ui) {
  ui.btnTunnelStart.addEventListener("click", async () => {
    setBusy(ui.btnTunnelStart, true, "جارٍ إنشاء النفق (قد يستغرق ~30 ثانية)…");
    try {
      const res = await post("/tunnel/start");
      toast(res?.message || "بدأ النفق.", "success", 8000);
      await loadTunnelStatus(ui);
    } catch (err) {
      toast(err.message, "error", 12000);
      if (err.detail) toast(err.detail, "error", 12000);
    } finally {
      setBusy(ui.btnTunnelStart, false);
    }
  });
  ui.btnTunnelStop.addEventListener("click", async () => {
    setBusy(ui.btnTunnelStop, true, "جارٍ الإيقاف…");
    try {
      await post("/tunnel/stop");
      toast("أُوقف النفق.", "success");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      setBusy(ui.btnTunnelStop, false);
      loadTunnelStatus(ui);
    }
  });
}

async function loadTunnelStatus(ui) {
  const box = ui.tunnelBox;
  try {
    const status = await get("/tunnel/status");
    renderTunnelStatus(box, status);
  } catch (err) {
    clear(box);
    box.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
  }
}

function renderTunnelStatus(box, status) {
  clear(box);
  const running = Boolean(status?.running);
  box.append(el("div", { class: "row" }, [
    running ? el("span", { class: "badge ok dot" }, ["النفق يعمل"]) : el("span", { class: "badge idle" }, ["النفق متوقف"]),
    status?.pid ? el("span", { class: "small latin dim" }, [`pid ${status.pid}`]) : null,
    status?.started_at ? el("span", { class: "small dim" }, [`منذ ${fmtTime(status.started_at)}`]) : null,
  ]));
  const urlText = status?.url || "";
  const urlEl = el("span", { class: `url${urlText ? "" : " empty"}` }, [urlText || "لا يوجد رابط بعد."]);
  const btnCopy = el("button", { class: "btn sm", type: "button", title: "نسخ الرابط" }, [icon("copy", 14), " نسخ"]);
  if (!urlText) btnCopy.disabled = true;
  else btnCopy.addEventListener("click", () => copyText(urlText, "تم نسخ رابط النفق."));
  box.append(el("div", { class: "url-box mt" }, [urlEl, btnCopy]));
  if (urlText) {
    box.append(el("p", { class: "small muted mt", style: "margin-bottom:0" }, [
      "افتح الرابط من أي متصفح مع تفعيل الرمز المحفوظ أعلاه. الرابط مؤقت ويتغير مع كل بدء.",
    ]));
  }
}

/* ------------------------------ system detect ------------------------------ */

function buildDetectCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["كشف النظام", el("span", { class: "sub latin" }, ["Detect"])]),
    el("div", { id: "st-detect-box" }, [
      el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
    ]),
    el("button", { class: "btn sm mt", type: "button", onclick: () => reloadDetect() }, [icon("refresh", 14), " إعادة الكشف"]),
  ]);
}

async function loadDetect(ui) {
  const box = ui.detectBox;
  try {
    const detect = await get("/system/detect");
    renderDetect(box, detect);
  } catch (err) {
    clear(box);
    box.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
  }
}

async function reloadDetect() {
  const box = document.getElementById("st-detect-box");
  if (!box) return;
  clear(box);
  box.append(el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]));
  try {
    renderDetect(box, await get("/system/detect"));
  } catch (err) {
    clear(box);
    box.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
  }
}

function renderDetect(box, detect) {
  clear(box);
  const d = detect || {};
  const gpus = Array.isArray(d.gpus) ? d.gpus : [];
  const kvRows = [
    ["نظام التشغيل", `${d.os || "—"} ${d.os_release || ""}`],
    ["المعمارية", d.arch || "—"],
    ["Python", d.python || "—"],
    ["CUDA", d.cuda || "—"],
    ["عدد GPUs", String(gpus.length)],
    ["RAM", fmtMB(d.ram_mb)],
    ["المساحة الحرة", fmtGB(d.disk_free_gb)],
    ["ffmpeg", d.ffmpeg ? "متوفر" : "غير متوفر"],
    ["cloudflared", d.cloudflared ? "متوفر" : "غير متوفر"],
    ["Colab", d.colab ? "نعم" : "لا"],
    ["Drive", d.drive_mounted ? "مرفق" : "غير مرفق"],
    ["uv", d.uv ? String(d.uv) : "غير مثبت"],
  ];
  const dl = el("dl", { class: "kv" });
  for (const [k, v] of kvRows) {
    dl.append(el("dt", {}, [k]), el("dd", {}, [document.createTextNode(String(v))]));
  }
  box.append(dl);
  for (const gpu of gpus) {
    box.append(el("div", { class: "notice ok mt" }, [
      el("p", {}, [
        `GPU ${gpu.index ?? 0}: ${gpu.name || "—"} — ${fmtMB(gpu.memory_mb)}` +
        (gpu.compute_capability && gpu.compute_capability.length ? ` (CC ${gpu.compute_capability.join(".")})` : ""),
      ]),
    ]));
  }
}

/* ------------------------------ logs ------------------------------ */

function buildLogsCard() {
  const linesSelect = el("select", { id: "st-logs-lines" }, [
    el("option", { value: "100" }, ["100 سطر"]),
    el("option", { value: "200", selected: true }, ["200 سطر"]),
    el("option", { value: "500" }, ["500 سطر"]),
  ]);
  return el("div", { class: "stack" }, [
    el("h2", { class: "card-title" }, [
      "سجلات المنصة",
      el("span", { class: "sub latin" }, ["platform.log"]),
      el("button", { class: "btn sm", type: "button", id: "st-logs-refresh" }, [icon("refresh", 14), " تحديث"]),
      linesSelect,
    ]),
    el("pre", { class: "log-viewer", id: "st-logs" }, ["…"]),
  ]);
}

async function loadLogs(ui) {
  const lines = ui.logLines.value || "200";
  try {
    const data = await get(`/logs?lines=${encodeURIComponent(lines)}`);
    const arr = Array.isArray(data?.lines) ? data.lines : [];
    ui.logsBox.textContent = arr.length ? arr.join("\n") : "(السجل فارغ)";
    ui.logsBox.scrollTop = ui.logsBox.scrollHeight;
  } catch (err) {
    ui.logsBox.textContent = err.message;
  }
}

/* ------------------------------ about ------------------------------ */

function buildAboutCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["عن المنصة", el("span", { class: "sub latin" }, ["About"])]),
    el("dl", { class: "kv" }, [
      el("dt", {}, ["اسم المنصة"]),
      el("dd", {}, ["استوديو الصوت العراقي — Iraqi Voice Studio"]),
      el("dt", {}, ["الإصدار"]),
      el("dd", { class: "latin", id: "st-about-version" }, ["—"]),
      el("dt", {}, ["الواجهة"]),
      el("dd", {}, ["تطبيق صفحة واحدة (SPA) بدون اعتمادات خارجية — يعمل محليًا وفي Colab."]),
      el("dt", {}, ["المحرك"]),
      el("dd", { class: "latin" }, ["RVC v2 · HuBERT · RMVPE/FCPE/PM"]),
    ]),
  ]);
}

async function loadAbout(ui) {
  try {
    const system = await get("/system");
    ui.aboutVersion.textContent = `v${system.app_version || "؟"}`;
  } catch (err) {
    ui.aboutVersion.textContent = err.message;
  }
}

/* ------------------------------ clipboard ------------------------------ */

async function copyText(text, successMsg) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      toast(successMsg, "success", 2500);
      return;
    }
  } catch { /* fall through to legacy path */ }
  try {
    const area = el("textarea", { style: "position:fixed;opacity:0;top:0;left:0" });
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    toast(successMsg, "success", 2500);
  } catch {
    toast("تعذّر النسخ تلقائيًا — انسخ الرابط يدويًا.", "warn");
  }
}
