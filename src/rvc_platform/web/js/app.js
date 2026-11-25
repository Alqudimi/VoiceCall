/* ============================================================
   app.js — SPA bootstrap: hash router, tab bar, auth token,
   periodic /api/system polling (visibility-aware), topbar chips.
   ============================================================ */

import {
  bus, get, getToken, ENGINE_LABELS, el, qsa, toast,
} from "./api.js";
import dashboard from "./views/dashboard.js";
import models from "./views/models.js";
import convert from "./views/convert.js";
import history from "./views/history.js";
import realtime from "./views/realtime.js";
import training from "./views/training.js";
import settings from "./views/settings.js";
import guide from "./views/guide.js";

const ROUTES = [
  { id: "dashboard", label: "لوحة التحكم", view: dashboard },
  { id: "models", label: "النماذج", view: models },
  { id: "convert", label: "تحويل الملفات", view: convert },
  { id: "history", label: "سجل التحويلات", view: history },
  { id: "realtime", label: "البث اللحظي", view: realtime },
  { id: "training", label: "التدريب", view: training },
  { id: "settings", label: "الإعدادات", view: settings },
  { id: "guide", label: "الدليل", view: guide },
];

const viewRoot = document.getElementById("view-root");
const tabsNav = document.getElementById("tabs");
let currentCleanup = null;
let currentRoute = null;

/* ------------------------------ nav ------------------------------ */

function buildTabs() {
  for (const route of ROUTES) {
    const link = el("a", { href: `#${route.id}`, dataset: { route: route.id } }, [route.label]);
    link.setAttribute("role", "tab");
    tabsNav.append(link);
  }
}

function markActiveTab(routeId) {
  for (const link of qsa("a", tabsNav)) {
    link.classList.toggle("active", link.dataset.route === routeId);
  }
}

/* ------------------------------ router ------------------------------ */

async function route() {
  let hash = (location.hash || "").replace(/^#/, "").split("?")[0].split("/")[0];
  if (!ROUTES.some((r) => r.id === hash)) {
    hash = "dashboard";
    if (location.hash !== `#${hash}`) {
      location.hash = `#${hash}`;
    }
  }
  if (currentRoute === hash && currentCleanup) return;

  if (currentCleanup) {
    try { currentCleanup(); } catch (err) { console.error("cleanup failed", err); }
    currentCleanup = null;
  }
  currentRoute = hash;

  markActiveTab(hash);
  viewRoot.replaceChildren();
  viewRoot.append(el("div", { class: "loading-line" }, [
    el("span", { class: "spin" }),
    document.createTextNode("جارٍ تحميل القسم…"),
  ]));

  const routeDef = ROUTES.find((r) => r.id === hash);
  let cleanup = null;
  try {
    cleanup = await routeDef.view.mount(viewRoot);
  } catch (err) {
    console.error("view mount failed", err);
    viewRoot.replaceChildren();
    viewRoot.append(renderError(err));
  }
  if (currentRoute === hash) {
    currentCleanup = typeof cleanup === "function" ? cleanup : null;
  } else if (typeof cleanup === "function") {
    cleanup();
  }
}

function renderError(err) {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, ["تعذّر تحميل القسم"]),
    el("div", { class: "notice danger" }, [
      el("span", { class: "n-icon" }),
      el("p", {}, [err?.message || "خطأ غير معروف"]),
    ]),
    el("button", {
      class: "btn primary",
      type: "button",
      onclick: () => { currentRoute = null; route(); },
    }, ["إعادة المحاولة"]),
  ]);
}

/* ------------------------------ topbar chips ------------------------------ */

const chipConn = document.getElementById("chip-conn");
const chipEngine = document.getElementById("chip-engine");
const chipToken = document.getElementById("chip-token");

function renderChips(system) {
  if (chipConn) {
    chipConn.dataset.state = system ? "ok" : "error";
    chipConn.querySelector(".chip-label").textContent = system ? "متصل" : "غير متصل";
  }
  if (chipEngine) {
    const state = system ? system.engine : "unavailable";
    chipEngine.dataset.state = state;
    chipEngine.querySelector(".chip-label").textContent =
      system ? `المحرك: ${ENGINE_LABELS[state] || state}` : "المحرك: غير متصل";
    chipEngine.title = system?.engine_message || "";
  }
}

function renderTokenChip() {
  if (!chipToken) return;
  const hasToken = Boolean(getToken());
  chipToken.dataset.state = hasToken ? "ok" : "warn";
  chipToken.querySelector(".chip-label").textContent = hasToken ? "الرمز: محفوظ" : "الرمز: غير محدد";
}

/* ------------------------------ system poll (visibility-aware) ------------------------------ */

let pollTimer = null;
const POLL_MS = 3000;

async function refreshSystem() {
  try {
    const system = await get("/system");
    renderChips(system);
    bus.emit("system", system);
  } catch (err) {
    if (err.status === 401) renderChips(null);
    else {
      renderChips(null);
      bus.emit("system", null);
    }
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    if (document.visibilityState === "visible") refreshSystem();
  }, POLL_MS);
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") refreshSystem();
});

bus.on("token", () => {
  renderTokenChip();
  refreshSystem();
});

/* ------------------------------ footer version ------------------------------ */

async function refreshFooterVersion() {
  try {
    const system = await get("/system");
    const node = document.getElementById("footer-version");
    if (node) node.textContent = `v${system.app_version || "?"}`;
  } catch { /* footer is cosmetic; errors surface elsewhere */ }
}

/* ------------------------------ boot ------------------------------ */

window.addEventListener("hashchange", route);

buildTabs();
renderTokenChip();
route();
refreshSystem();
startPolling();
refreshFooterVersion();

/* expose a tiny debug handle (no data fabrication — live endpoint check) */
window.ivs = { health: () => get("/health").catch((e) => { toast(e.message, "error"); return null; }) };
