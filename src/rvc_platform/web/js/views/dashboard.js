/* ============================================================
   dashboard.js — لوحة التحكم: نظام، معالج، محرك، تدريب،
   جلسات لحظية، نفق، والأصول (HuBERT/RMVPE/pretrained).
   ============================================================ */

import {
  bus, get, post, toast, el, clear, icon,
  ENGINE_LABELS, TRAIN_STATES, fmtNum, fmtMB, fmtGB, fmtMs, fmtPct, fmtTime,
  setBusy,
} from "../api.js";

export default { mount };

async function mount(root) {
  const offSystem = bus.on("system", (system) => update(system));

  root.replaceChildren(
    el("div", { class: "stack" }, [
      buildHero(),
      el("div", { class: "grid" }, [
        buildSystemCard(),
        buildComputeCard(),
        buildEngineCard(),
        buildTrainingCard(),
        buildRealtimeCard(),
        buildTunnelCard(),
        el("div", { class: "card span-2", id: "dash-assets" }, [
          el("h2", { class: "card-title" }, [
            "الأصول المسبقة",
            el("span", { class: "sub latin" }, ["HuBERT · RMVPE · Pretrained"]),
          ]),
          el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "جارٍ التحقق من الأصول…"]),
        ]),
      ]),
    ])
  );

  let statsTimer = null;
  let assetsTimer = null;

  async function pollStats() {
    try {
      const stats = await get("/realtime/stats");
      const box = document.getElementById("dash-rt-stats");
      if (box) renderRealtimeStats(box, stats);
    } catch { /* latency panel shows '—' on failure */ }
  }
  statsTimer = setInterval(pollStats, 5000);
  pollStats();

  await Promise.allSettled([refreshSystemOnce(), loadAssets()]);
  refreshTunnelOnce();

  async function refreshSystemOnce() {
    try { update(await get("/system")); }
    catch { update(null); }
  }
  async function refreshTunnelOnce() {
    try {
      const tunnel = await get("/tunnel/status");
      renderTunnel(tunnel);
    } catch { renderTunnel(null); }
  }

  function update(system) {
    renderSystem(system);
    renderCompute(system);
    renderEngine(system);
    renderTraining(system?.training || null);
    renderRealtimeCount(system);
    if (system?.tunnel) renderTunnel(system.tunnel);
  }

  return function cleanup() {
    offSystem();
    if (statsTimer) clearInterval(statsTimer);
    if (assetsTimer) clearInterval(assetsTimer);
    if (assetsPoller) { clearInterval(assetsPoller); assetsPoller = null; }
  };
}

/* ------------------------------ hero ------------------------------ */

function buildHero() {
  return el("div", { class: "card", id: "dash-hero" }, [
    el("div", { class: "spread" }, [
      el("div", {}, [
        el("h2", { class: "card-title", style: "margin:0" }, ["ملخص المنصة"]),
        el("p", { class: "muted small", style: "margin:4px 0 0" }, [
          "تحديث تلقائي كل 3 ثوانٍ أثناء ظهور الصفحة.",
        ]),
      ]),
      el("span", { class: "badge info dot", id: "dash-hero-engine" }, ["—"]),
    ]),
    el("div", { class: "row mt", id: "dash-hero-msg", style: "min-height:24px" }, []),
  ]);
}

/* ------------------------------ system card ------------------------------ */

function buildSystemCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "النظام",
      el("span", { class: "sub latin" }, ["System"]),
    ]),
    el("dl", { class: "kv", id: "dash-system" }, []),
  ]);
}

function renderSystem(system) {
  const dl = document.getElementById("dash-system");
  if (!dl) return;
  clear(dl);
  const badge = document.getElementById("dash-hero-engine");
  const msgBox = document.getElementById("dash-hero-msg");
  if (badge && system) {
    badge.className = `badge ${system.engine} dot`;
    badge.textContent = `المحرك: ${ENGINE_LABELS[system.engine] || system.engine}`;
  }
  if (msgBox) {
    clear(msgBox);
    if (!system) {
      msgBox.append(el("span", { class: "badge error" }, ["تعذّر الاتصال بالخادم"]));
    } else if (system.engine_message) {
      msgBox.append(el("span", { class: "small dim latin" }, [system.engine_message]));
    }
  }
  if (!system) {
    dl.append(...kv("إصدار المنصة", "—"), ...kv("نظام التشغيل", "—"));
    return;
  }
  dl.append(
    ...kv("إصدار المنصة", el("span", { class: "latin" }, [`v${system.app_version}`])),
    ...kv("نظام التشغيل", el("span", { class: "latin" }, [`${system.os} / ${system.arch}`])),
    ...kv("Python", el("span", { class: "latin" }, [system.python])),
    ...kv("المساحة الحرة", fmtGB(system.disk_free_gb)),
    ...kv("ذاكرة RAM", `${fmtMB(system.ram.used_mb)} / ${fmtMB(system.ram.total_mb)}`),
    ...kv("آخر تحديث", el("span", { class: "latin small" }, [fmtTime(system.timestamp)])),
  );
}

function kv(key, value) {
  const dt = el("dt", {}, [key]);
  const dd = el("dd", {});
  dd.append(value instanceof Node ? value : document.createTextNode(String(value)));
  return [dt, dd];
}

/* ------------------------------ compute card ------------------------------ */

function buildComputeCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "المعالج والذاكرة",
      el("span", { class: "sub latin" }, ["Device"]),
    ]),
    el("dl", { class: "kv", id: "dash-compute" }, []),
    el("div", { class: "stack", id: "dash-compute-bars", style: "margin-top:12px" }, []),
  ]);
}

function bar(label, used, total, unit = "MB") {
  const frac = total > 0 ? Math.min(1, used / total) : 0;
  return el("div", {}, [
    el("div", { class: "progress-label" }, [
      label,
      el("span", { class: "latin" }, [`${fmtNum(used, 0)} / ${fmtNum(total, 0)} ${unit} (${fmtPct(frac)})`]),
    ]),
    el("div", { class: "progress amber" }, [el("div", { class: "bar", style: `width:${(frac * 100).toFixed(1)}%` })]),
  ]);
}

function renderCompute(system) {
  const dl = document.getElementById("dash-compute");
  const bars = document.getElementById("dash-compute-bars");
  if (!dl || !bars) return;
  clear(dl); clear(bars);
  if (!system) { dl.append(...kv("الجهاز", "—")); return; }
  dl.append(
    ...kv("الجهاز", el("span", { class: "latin" }, [system.device])),
    ...kv("الدقة", system.is_half ? "FP16 (نصف دقة)" : "FP32 (دقة كاملة)"),
    ...kv("بطاقة GPU", system.gpu?.name ? el("span", { class: "latin" }, [system.gpu.name]) : "—"),
  );
  if (system.gpu && system.gpu.vram_total_mb) {
    bars.append(bar("VRAM — ذاكرة الفيديو", system.gpu.vram_used_mb || 0, system.gpu.vram_total_mb));
  } else {
    bars.append(el("p", { class: "muted small", style: "margin:0" }, ["لا توجد GPU نشطة (وضع CPU)."]));
  }
  if (system.ram && system.ram.total_mb) {
    bars.append(bar("RAM — ذاكرة النظام", system.ram.used_mb || 0, system.ram.total_mb));
  }
}

/* ------------------------------ engine card ------------------------------ */

function buildEngineCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "المحرك والنموذج المحمّل",
      el("span", { class: "sub latin" }, ["Engine"]),
    ]),
    el("div", { id: "dash-engine" }, [
      el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
    ]),
  ]);
}

function renderEngine(system) {
  const box = document.getElementById("dash-engine");
  if (!box) return;
  clear(box);
  if (!system) {
    box.append(el("p", { class: "muted" }, ["لا يوجد اتصال."]));
    return;
  }
  const state = system.engine;
  box.append(el("div", { class: "row" }, [
    el("span", { class: `badge ${state} dot` }, [ENGINE_LABELS[state] || state]),
    system.conversion_server
      ? el("span", { class: "badge ok" }, ["خدمة التحويل متاحة"])
      : el("span", { class: "badge warn" }, ["خدمة التحويل غير جاهزة"]),
  ]));
  if (system.engine_message) {
    box.append(el("p", { class: "small latin muted", style: "margin:8px 0 0;direction:ltr;text-align:left" }, [system.engine_message]));
  }
  const loaded = system.loaded_model;
  if (loaded) {
    box.append(el("div", { class: "mt" }, [
      el("dl", { class: "kv" }, [
        ...kv("الاسم", loaded.name),
        ...kv("معدل العينة", `${fmtNum(loaded.sample_rate / 1000, 1)} kHz`),
        ...kv("الطبقة (f0)", loaded.f0 ? "مدعومة" : "غير مدعومة"),
        ...kv("الإصدار", el("span", { class: "latin" }, [loaded.version])),
        ...kv("الحجم", fmtMB(loaded.size_mb)),
        ...kv("الفهرس", loaded.index_path ? el("span", { class: "latin small" }, ["مرفق"]) : "لا يوجد"),
      ]),
    ]));
    box.append(el("div", { class: "row mt" }, [
      el("a", { class: "btn sm", href: "#convert" }, ["الانتقال إلى التحويل"]),
      el("a", { class: "btn sm ghost", href: "#realtime" }, ["البث اللحظي"]),
    ]));
  } else {
    box.append(el("p", { class: "muted small mt" }, ["لا يوجد نموذج محمّل — حمّل نموذجًا من تبويب النماذج."]));
    box.append(el("a", { class: "btn sm primary mt", href: "#models" }, ["فتح تبويب النماذج"]));
  }
}

/* ------------------------------ training card ------------------------------ */

function buildTrainingCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "التدريب",
      el("span", { class: "sub latin" }, ["Training"]),
    ]),
    el("div", { id: "dash-training" }, [
      el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
    ]),
  ]);
}

function renderTraining(training) {
  const box = document.getElementById("dash-training");
  if (!box) return;
  clear(box);
  if (!training) {
    box.append(el("p", { class: "muted" }, ["لا توجد عملية تدريب حالية."]));
    box.append(el("a", { class: "btn sm", href: "#training" }, ["فتح تبويب التدريب"]));
    return;
  }
  const stateLabel = TRAIN_STATES[training.state] || training.state;
  box.append(el("div", { class: "row" }, [
    el("span", { class: `badge ${training.state} dot` }, [stateLabel]),
    el("span", { class: "small dim" }, [
      `الدورة ${fmtNum(training.epoch, 0)} / ${fmtNum(training.total_epochs, 0)}`,
    ]),
    training.global_step ? el("span", { class: "small latin dim" }, [`step ${training.global_step}`]) : null,
  ]));
  const frac = training.total_epochs > 0 ? training.epoch / training.total_epochs : 0;
  box.append(el("div", { class: "mt" }, [
    el("div", { class: "progress" }, [el("div", { class: "bar", style: `width:${(frac * 100).toFixed(1)}%` })]),
    el("div", { class: "progress-label" }, ["تقدم الدورات", fmtPct(frac)]),
  ]));
  const losses = training.latest_losses || {};
  const keys = Object.keys(losses);
  if (keys.length) {
    const lossRow = el("div", { class: "row small mt" });
    for (const k of keys) lossRow.append(el("span", { class: "badge" }, [`${k}: ${fmtNum(losses[k], 4)}`]));
    box.append(lossRow);
  }
  if (training.message) box.append(el("p", { class: "small muted mt", style: "margin-bottom:0" }, [training.message]));
  box.append(el("a", { class: "btn sm mt", href: "#training" }, ["عرض التفاصيل"]));
}

/* ------------------------------ realtime card ------------------------------ */

function buildRealtimeCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "الجلسات اللحظية",
      el("span", { class: "sub latin" }, ["Realtime"]),
    ]),
    el("div", { id: "dash-rt-count" }, [
      el("div", { class: "row" }, [
        el("span", { class: "badge info" }, ["— جلسة"]),
      ]),
    ]),
    el("div", { class: "mt", id: "dash-rt-stats" }, []),
    el("a", { class: "btn sm mt", href: "#realtime" }, ["فتح البث اللحظي"]),
  ]);
}

function renderRealtimeCount(system) {
  const box = document.getElementById("dash-rt-count");
  if (!box || !system) return;
  clear(box);
  box.append(el("div", { class: "row" }, [
    el("span", { class: "badge info" }, [`${fmtNum(system.realtime_sessions, 0)} جلسة نشطة`]),
  ]));
}

function renderRealtimeStats(box, stats) {
  clear(box);
  const agg = stats?.aggregate || {};
  if (!agg || agg.count === undefined || !agg.count) {
    box.append(el("p", { class: "muted small", style: "margin:0" }, ["لا توجد قياسات حتى الآن."]));
    return;
  }
  const grid = el("div", { class: "latency-grid" });
  grid.append(
    tile("عدد الكتل", fmtNum(agg.count, 0)),
    tile("أدنى زمن", fmtMs(agg.min_ms)),
    tile("متوسط زمن الاستدلال", fmtMs(agg.avg_ms)),
    tile("p95", fmtMs(agg.p95_ms)),
    tile("أقصى زمن", fmtMs(agg.max_ms)),
    tile("الجلسات", fmtNum(agg.sessions ?? 0, 0)),
  );
  box.append(grid);
}

function tile(label, value) {
  return el("div", { class: "stat-tile" }, [
    el("div", { class: "t-label" }, [label]),
    el("div", { class: "t-value" }, [value]),
  ]);
}

/* ------------------------------ tunnel card ------------------------------ */

function buildTunnelCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "النفق العام",
      el("span", { class: "sub latin" }, ["Cloudflare Tunnel"]),
    ]),
    el("div", { id: "dash-tunnel" }, [
      el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
    ]),
  ]);
}

function renderTunnel(tunnel) {
  const box = document.getElementById("dash-tunnel");
  if (!box) return;
  clear(box);
  if (!tunnel) {
    box.append(el("p", { class: "muted" }, ["حالة النفق غير معروفة."]));
    return;
  }
  box.append(el("div", { class: "row" }, [
    tunnel.running
      ? el("span", { class: "badge ok dot" }, ["يعمل"])
      : el("span", { class: "badge idle" }, ["متوقف"]),
    tunnel.pid ? el("span", { class: "small latin dim" }, [`pid ${tunnel.pid}`]) : null,
  ]));
  if (tunnel.url) {
    box.append(el("div", { class: "url-box mt" }, [
      el("span", { class: "url" }, [tunnel.url]),
    ]));
  } else if (tunnel.running) {
    box.append(el("p", { class: "small muted mt" }, ["النفق يعمل لكن الرابط غير متوفر بعد."]));
  }
  box.append(el("a", { class: "btn sm mt", href: "#settings" }, ["إدارة النفق من الإعدادات"]));
}

/* ------------------------------ assets ------------------------------ */

async function loadAssets() {
  const card = document.getElementById("dash-assets");
  if (!card) return;
  try {
    const assets = await get("/assets");
    renderAssets(card, assets);
  } catch (err) {
    clear(card);
    card.append(
      el("h2", { class: "card-title" }, ["الأصول المسبقة"]),
      el("div", { class: "notice danger" }, [el("p", {}, [err.message])]),
    );
  }
}

function renderAssets(card, assets) {
  clear(card);
  const summary = assets?._summary || {};
  const entries = Object.entries(assets || {}).filter(([k]) => k !== "_summary");

  card.append(el("h2", { class: "card-title" }, [
    "الأصول المسبقة",
    el("span", { class: "sub latin" }, ["HuBERT · RMVPE · Pretrained v2"]),
    el("button", {
      class: "btn sm", type: "button", id: "assets-refresh",
      onclick: () => loadAssets(),
      title: "تحديث حالة الأصول",
    }, [icon("refresh", 14), " تحديث"]),
  ]));

  const badges = el("div", { class: "row" }, [
    el("span", { class: `badge ${summary.hubert_ready ? "ok" : "bad"}` }, [`HuBERT: ${summary.hubert_ready ? "جاهز" : "مفقود"}`]),
    el("span", { class: `badge ${summary.rmvpe_ready ? "ok" : "bad"}` }, [`RMVPE: ${summary.rmvpe_ready ? "جاهز" : "مفقود"}`]),
    el("span", { class: `badge ${summary.pretrained_ready ? "ok" : "warn"}` }, [`Pretrained: ${summary.pretrained_ready ? "جاهز" : "مفقود"}`]),
  ]);
  card.append(badges);

  const table = el("div", { class: "table-wrap mt" });
  const tbody = el("tbody");
  for (const [name, info] of entries) {
    tbody.append(el("tr", {}, [
      el("td", { class: "latin small" }, [name]),
      el("td", {}, [
        info.present
          ? el("span", { class: "badge ok" }, ["موجود"])
          : el("span", { class: "badge bad" }, ["مفقود"]),
      ]),
      el("td", { class: "num" }, [fmtMB(info.size_mb)]),
    ]));
  }
  table.append(el("table", { class: "tbl" }, [
    el("thead", {}, [el("tr", {}, [
      el("th", {}, ["الملف"]), el("th", {}, ["الحالة"]), el("th", {}, ["الحجم"]),
    ])]),
    tbody,
  ]));
  card.append(table);

  const allReady = summary.hubert_ready && summary.rmvpe_ready && summary.pretrained_ready;
  const actions = el("div", { class: "row mt" }, []);
  const btnDownload = el("button", { class: "btn primary", type: "button" }, [icon("download", 15), " تنزيل الأصول الناقصة"]);
  btnDownload.addEventListener("click", () => downloadAssets(btnDownload));
  actions.append(btnDownload);
  if (allReady) {
    actions.append(el("span", { class: "badge ok" }, ["كل الأصول جاهزة"]));
  }
  card.append(actions);
  card.append(el("p", { class: "small muted mt", style: "margin-bottom:0" }, [
    "التنزيل يجري في الخلفية من مستودع RVC الرسمي (HuggingFace) — راقب الحالة بالتحديث الدوري أو زر التحديث.",
  ]));
}

let assetsPoller = null;

async function downloadAssets(button) {
  setBusy(button, true, "جارٍ بدء التنزيل…");
  try {
    const res = await post("/assets/download");
    if (res && res.started) {
      toast("بدأ تنزيل الأصول في الخلفية — سيُحدَّث المؤشر تلقائيًا.", "success");
      startAssetsPolling();
    } else {
      toast(res?.message || "التنزيل قيد التنفيذ بالفعل.", "warn");
      startAssetsPolling();
    }
  } catch (err) {
    toast(err.message, "error");
  } finally {
    setBusy(button, false);
  }
}

function startAssetsPolling() {
  if (assetsPoller) clearInterval(assetsPoller);
  const card = () => document.getElementById("dash-assets");
  assetsPoller = setInterval(async () => {
    const node = card();
    if (!node) { clearInterval(assetsPoller); assetsPoller = null; return; }
    try {
      const assets = await get("/assets");
      renderAssets(node, assets);
      const s = assets?._summary || {};
      if (s.hubert_ready && s.rmvpe_ready && s.pretrained_ready) {
        clearInterval(assetsPoller);
        assetsPoller = null;
        toast("اكتمل تنزيل جميع الأصول.", "success");
      }
    } catch { /* transient poll failure — keep trying */ }
  }, 3000);
}
