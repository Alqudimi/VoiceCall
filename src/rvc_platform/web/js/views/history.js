/* ============================================================
   history.js — سجل التحويلات: كل عمليات تحويل الملفات السابقة،
   إعدادات كل تحويل، سرعته، مع تشغيل وتنزيل كل نتيجة.
   ============================================================ */

import {
  get, getBlob, toast, el, clear, icon, setBusy, fmtNum, fmtMB, fmtTime, fmtDuration,
} from "../api.js";

export default { mount };

const PAGE_SIZE = 20;

async function mount(root) {
  let all = [];
  let filter = "";

  root.replaceChildren(el("div", { class: "stack" }, [
    buildIntro(),
    buildSummaryCard(),
    buildListCard(),
  ]));

  const ui = {
    summary: root.querySelector("#hs-summary"),
    list: root.querySelector("#hs-list"),
    search: root.querySelector("#hs-search"),
    refresh: root.querySelector("#hs-refresh"),
    more: root.querySelector("#hs-more"),
    shown: 0,
  };

  ui.search.addEventListener("input", () => {
    filter = ui.search.value.trim().toLowerCase();
    renderList(ui, all, filter);
  });
  ui.refresh.addEventListener("click", async () => {
    setBusy(ui.refresh, true, "جارٍ التحديث…");
    try { await load(); } finally { setBusy(ui.refresh, false); }
  });
  ui.more.addEventListener("click", () => {
    ui.shown += PAGE_SIZE;
    renderList(ui, all, filter);
  });

  await load();

  async function load() {
    try {
      all = await get("/convert/history?limit=200");
      renderSummary(ui, all);
      ui.shown = PAGE_SIZE;
      renderList(ui, all, filter);
    } catch (err) {
      clear(ui.list);
      ui.list.append(el("div", { class: "notice danger" }, [
        el("span", { class: "n-icon" }, [icon("alert", 16)]),
        el("p", { style: "margin:0" }, [err.message]),
      ]));
    }
  }

  function cleanup() { /* one-shot page: no intervals */ }
  return cleanup;
}

/* ------------------------------ intro ------------------------------ */

function buildIntro() {
  return el("div", { class: "notice info" }, [
    el("span", { class: "n-icon" }, [icon("wave", 18)]),
    el("div", {}, [
      el("p", { style: "font-weight:700;margin:0 0 4px" }, ["سجل التحويلات"]),
      el("p", { class: "small", style: "margin:0" }, [
        "كل عملية تحويل ملف تُسجَّل هنا مع إعداداتها الكاملة وسرعتها — ",
        el("span", { class: "dim" }, ["السجل محفوظ على الخادم ويبقى بعد إعادة التشغيل."]),
      ]),
    ]),
  ]);
}

/* ------------------------------ summary ------------------------------ */

function buildSummaryCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "نظرة عامة",
      el("span", { class: "sub latin" }, ["Overview"]),
    ]),
    el("div", { class: "latency-grid", id: "hs-summary" }, []),
  ]);
}

function renderSummary(ui, all) {
  const box = ui.summary;
  if (!box) return;
  clear(box);
  const list = Array.isArray(all) ? all : [];
  const total = list.length;
  const totalMb = list.reduce((a, r) => a + (Number(r.size_mb) || 0), 0);
  const rtfs = list.map((r) => Number(r.realtime_factor)).filter(Number.isFinite);
  const avgRtf = rtfs.length ? rtfs.reduce((a, b) => a + b, 0) / rtfs.length : null;
  const durations = list.map((r) => Number(r.duration_s)).filter(Number.isFinite);
  const totalDur = durations.length ? durations.reduce((a, b) => a + b, 0) : 0;
  box.append(
    tile("عدد التحويلات", fmtNum(total, 0)),
    tile("إجمالي المدة المحوّلة", fmtDuration(totalDur)),
    tile("إجمالي حجم النواتج", fmtMB(totalMb)),
    tile("متوسط معامل الزمن (RTF)", avgRtf !== null ? fmtNum(avgRtf, 3) : "—"),
  );
}

function tile(label, value) {
  return el("div", { class: "stat-tile" }, [
    el("div", { class: "t-label" }, [label]),
    el("div", { class: "t-value" }, [value]),
  ]);
}

/* ------------------------------ list ------------------------------ */

function buildListCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "التحويلات السابقة",
      el("input", {
        type: "search", id: "hs-search", placeholder: "ابحث باسم الملف أو الإعدادات…",
        style: "max-width:260px;margin-inline-start:auto",
      }),
      el("button", { class: "btn sm", type: "button", id: "hs-refresh" }, [
        icon("refresh", 14), " تحديث",
      ]),
    ]),
    el("div", { id: "hs-list" }, [
      el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
    ]),
    el("div", { class: "row mt" }, [
      el("button", { class: "btn ghost", type: "button", id: "hs-more" }, [
        "عرض المزيد",
      ]),
    ]),
  ]);
}

function renderList(ui, all, filter) {
  const box = ui.list;
  if (!box) return;
  clear(box);
  const list = Array.isArray(all) ? all : [];
  const filtered = filter
    ? list.filter((r) => `${r.file} ${JSON.stringify(r.settings || {})}`.toLowerCase().includes(filter))
    : list;
  ui.more.classList.toggle("hidden", filtered.length <= ui.shown);

  if (!filtered.length) {
    box.append(el("p", { class: "muted" }, [
      list.length ? "لا نتائج مطابقة للبحث." : "لا توجد تحويلات مسجّلة بعد — نفّذ أول تحويل من تبويب تحويل الملفات.",
    ]));
    return;
  }

  const shown = filtered.slice(0, ui.shown);
  const tbody = el("tbody");
  for (const row of shown) tbody.append(historyRow(row));

  box.append(el("div", { class: "table-wrap" }, [
    el("table", { class: "tbl" }, [
      el("thead", {}, [el("tr", {}, [
        el("th", {}, ["الملف"]),
        el("th", {}, ["التاريخ"]),
        el("th", {}, ["المدة"]),
        el("th", {}, ["زمن المعالجة"]),
        el("th", {}, ["RTF"]),
        el("th", {}, ["الحجم"]),
        el("th", {}, ["إجراءات"]),
      ])]),
      tbody,
    ]),
  ]));
  box.append(el("p", { class: "small muted", style: "margin:8px 0 0" }, [
    `يُعرض ${shown.length} من ${filtered.length} تحويل${filter ? " (مُصفّى)" : ""}.`,
  ]));
}

function historyRow(row) {
  const playBtn = el("button", { class: "btn sm primary", type: "button" }, [
    icon("play", 14), " تشغيل",
  ]);
  playBtn.addEventListener("click", () => togglePlay(row, playBtn));

  const dlBtn = el("button", { class: "btn sm", type: "button" }, [
    icon("download", 14), " تنزيل",
  ]);
  dlBtn.addEventListener("click", async () => {
    setBusy(dlBtn, true, "جارٍ التنزيل…");
    try {
      const blob = await getBlob(`/convert/results/${encodeURIComponent(row.id)}/download`);
      const url = URL.createObjectURL(blob);
      const a = el("a", { href: url, download: row.file });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      toast(`تم تنزيل "${row.file}".`, "success");
    } catch (err) {
      toast(err.message, "error");
    } finally { setBusy(dlBtn, false); }
  });

  return el("tr", {}, [
    el("td", { class: "latin small" }, [row.file]),
    el("td", { class: "latin small" }, [fmtTime(row.created_at)]),
    el("td", { class: "num" }, [row.duration_s != null ? fmtDuration(row.duration_s) : "—"]),
    el("td", { class: "num" }, [row.processing_s != null ? `${fmtNum(row.processing_s, 2)} s` : "—"]),
    el("td", { class: "num" }, [row.realtime_factor != null ? fmtNum(row.realtime_factor, 3) : "—"]),
    el("td", { class: "num" }, [fmtMB(row.size_mb)]),
    el("td", {}, [el("div", { class: "actions" }, [playBtn, dlBtn])]),
  ]);
}

/* expandable inline player + settings chips under the row */
function togglePlay(row, btn) {
  const existing = document.querySelector(`tr[data-hs-detail="${CSS.escape(row.id)}"]`);
  if (existing) {
    existing.remove();
    btn.replaceChildren(icon("play", 14), document.createTextNode(" تشغيل"));
    return;
  }
  const player = el("audio", {
    controls: true, preload: "metadata",
    src: `/api/convert/results/${encodeURIComponent(row.id)}/download`,
    style: "width:100%",
  });
  const chips = el("div", { class: "row small" });
  const s = row.settings || {};
  const labels = {
    pitch_shift: "النغمة", f0_method: "طريقة f0", index_rate: "نسبة الفهرس",
    rms_mix_rate: "مزج RMS", protect: "حماية", denoise: "إزالة ضجيج", normalize: "توحيد مستوى",
  };
  for (const [k, label] of Object.entries(labels)) {
    if (s[k] === undefined) continue;
    const v = typeof s[k] === "boolean" ? (s[k] ? "نعم" : "لا") : s[k];
    chips.append(el("span", { class: "badge" }, [`${label}: ${v}`]));
  }
  const tr = el("tr", { class: "detail-row", dataset: { hsDetail: row.id } }, [
    el("td", { colspan: "7" }, [el("div", { class: "stack" }, [player, chips])]),
  ]);
  btn.closest("tr").after(tr);
  player.play().catch(() => { /* autoplay guard: user can press play manually */ });
  btn.replaceChildren(icon("stop", 14), document.createTextNode(" إيقاف"));
  player.addEventListener("ended", () => {
    tr.remove();
    btn.replaceChildren(icon("play", 14), document.createTextNode(" تشغيل"));
  });
}
