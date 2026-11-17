/* ============================================================
   api.js — shared client for the Iraqi Voice Studio API.
   - bearer-token auth (localStorage "ivs_token")
   - fetch wrapper + Arabic error mapping
   - toasts, confirm modal, tiny event bus
   - formatting + DOM helpers
   - realtime WebSocket wrapper
   ============================================================ */

export const TOKEN_KEY = "ivs_token";
export const API_BASE = "/api";

/* ------------------------------ errors ------------------------------ */

export class ApiError extends Error {
  constructor(message, status = 0, detail = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

const TOKEN_MSG = "الرمز غير صالح أو مفقود — أضف الرمز من تبويب الإعدادات";

const HTTP_AR = {
  400: "طلب غير صالح (400)",
  401: TOKEN_MSG,
  403: "الوصول مرفوض (403)",
  404: "العنصر غير موجود (404)",
  409: "تعارض في الحالة — تحقق من النموذج المحمّل (409)",
  413: "الملف أكبر من الحد المسموح (413)",
  422: "بيانات غير مقبولة (422)",
  500: "خطأ داخلي في الخادم (500)",
  502: "فشل في خدمة خارجية (502)",
  503: "الخدمة غير متاحة حاليًا (503)",
};

function mapStatus(status, detail) {
  if (status === 401) return TOKEN_MSG;
  const base = HTTP_AR[status] || `خطأ في الشبكة (${status})`;
  return detail ? `${base} — ${detail}` : base;
}

/* ------------------------------ token ------------------------------ */

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function setToken(value) {
  try {
    const v = (value || "").trim();
    if (v) localStorage.setItem(TOKEN_KEY, v);
    else localStorage.removeItem(TOKEN_KEY);
    bus.emit("token", v);
    return v;
  } catch {
    toast("تعذّر حفظ الرمز في هذا المتصفح (تخزين محجوب)", "error");
    return "";
  }
}

export function clearToken() {
  return setToken("");
}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/* ------------------------------ event bus ------------------------------ */

export const bus = {
  _subs: new Map(),
  on(event, handler) {
    if (!this._subs.has(event)) this._subs.set(event, new Set());
    this._subs.get(event).add(handler);
    return () => this._subs.get(event).delete(handler);
  },
  emit(event, data) {
    const subs = this._subs.get(event);
    if (subs) for (const fn of subs) {
      try { fn(data); } catch (e) { console.error("bus handler failed", e); }
    }
  },
};

/* ------------------------------ toasts ------------------------------ */

let toastRoot = null;

function ensureToastRoot() {
  if (!toastRoot) toastRoot = document.getElementById("toasts");
  return toastRoot;
}

export function toast(message, type = "info", timeoutMs = 0) {
  const root = ensureToastRoot();
  if (!root) return;
  const item = el("div", { class: `toast ${type}` });
  const msg = el("div", { class: "t-msg" });
  msg.textContent = message;
  const close = el("button", { class: "t-close", type: "button", title: "إغلاق" });
  close.setAttribute("aria-label", "إغلاق التنبيه");
  close.textContent = "\u00d7";
  item.append(msg, close);
  root.append(item);
  const ttl = timeoutMs || (type === "error" ? 9000 : 4500);
  const kill = () => {
    item.style.opacity = "0";
    setTimeout(() => item.remove(), 250);
  };
  close.addEventListener("click", kill);
  setTimeout(kill, ttl);
  while (root.children.length > 5) root.firstElementChild.remove();
}

/* ------------------------------ confirm modal ------------------------------ */

export function confirmDialog({ title = "تأكيد", message = "", confirmText = "تأكيد", cancelText = "إلغاء", danger = false } = {}) {
  return new Promise((resolve) => {
    const root = document.getElementById("modal-root");
    if (!root) { resolve(false); return; }

    const backdrop = el("div", { class: "backdrop" });
    const box = el("div", { class: "modal", role: "dialog", "aria-modal": "true" });
    const h = el("h3", {}, [title]);
    const p = el("p", {}, [message]);
    const actions = el("div", { class: "modal-actions" });
    const cancel = el("button", { class: "btn ghost", type: "button" }, [cancelText]);
    const ok = el("button", { class: `btn ${danger ? "danger" : "primary"}`, type: "button" }, [confirmText]);
    actions.append(cancel, ok);
    box.append(h, p, actions);
    root.append(backdrop, box);
    root.classList.add("open");

    let done = false;
    const finish = (result) => {
      if (done) return;
      done = true;
      root.classList.remove("open");
      backdrop.remove();
      box.remove();
      resolve(result);
    };
    cancel.addEventListener("click", () => finish(false));
    ok.addEventListener("click", () => finish(true));
    backdrop.addEventListener("click", () => finish(false));
    document.addEventListener("keydown", function esc(ev) {
      if (ev.key === "Escape") { finish(false); document.removeEventListener("keydown", esc); }
    });
    ok.focus();
  });
}

/* ------------------------------ core request ------------------------------ */

async function parseError(res) {
  let detail = "";
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") detail = data.detail;
    else if (Array.isArray(data?.detail) && data.detail.length) {
      detail = data.detail.map((d) => d?.msg || "").filter(Boolean).join("; ");
    }
  } catch { /* non-json body */ }
  return new ApiError(mapStatus(res.status, detail), res.status, detail);
}

export async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch (err) {
    throw new ApiError("تعذّر الاتصال بالخادم — تأكد أن المنصة تعمل", 0, String(err));
  }
  if (!res.ok) {
    if (res.status === 401) toast(TOKEN_MSG, "error");
    throw await parseError(res);
  }
  if (res.status === 204) return null;
  const ctype = res.headers.get("content-type") || "";
  if (ctype.includes("application/json")) return res.json();
  return res;
}

export function get(path) { return request(path, { method: "GET" }); }

export function post(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? "" : JSON.stringify(body),
  });
}

export function postForm(path, formData) {
  return request(path, { method: "POST", body: formData });
}

export function del(path) { return request(path, { method: "DELETE" }); }

/** Plain-text GET (training logs). */
export async function getText(path) {
  const res = await request(path, { method: "GET" });
  if (typeof res === "string") return res;
  return res.text();
}

/** Binary GET honoring auth — returns a Blob (audio results, etc.). */
export async function getBlob(path) {
  const headers = authHeaders();
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { headers });
  } catch (err) {
    throw new ApiError("تعذّر الاتصال بالخادم", 0, String(err));
  }
  if (!res.ok) {
    if (res.status === 401) toast(TOKEN_MSG, "error");
    throw await parseError(res);
  }
  return res.blob();
}

/** Multipart POST with genuine upload progress (XMLHttpRequest). */
export function postFormWithProgress(path, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}${path}`);
    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.responseType = "text";
    xhr.upload.addEventListener("progress", (ev) => {
      if (ev.lengthComputable && onProgress) onProgress(ev.loaded / ev.total);
    });
    xhr.addEventListener("error", () => reject(new ApiError("تعذّر الاتصال أثناء الإرسال", 0)));
    xhr.addEventListener("timeout", () => reject(new ApiError("انتهت مهلة الإرسال", 0)));
    xhr.addEventListener("load", () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) { resolve(data); return; }
      if (xhr.status === 401) toast(TOKEN_MSG, "error");
      let detail = "";
      if (typeof data?.detail === "string") detail = data.detail;
      reject(new ApiError(mapStatus(xhr.status, detail), xhr.status, detail));
    });
    xhr.send(formData);
  });
}

/* ------------------------------ websocket wrapper ------------------------------ */

/**
 * Opens the realtime WebSocket with the optional bearer token.
 * Returns a small wrapper: sendText(obj), sendBinary(arrayBuffer), close(),
 * and event hooks assigned by the caller (onopen / ontext / onbinary / onclose / onerror).
 */
export function openRealtimeWS(token = getToken()) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}${API_BASE}/realtime/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  const wrapper = {
    socket: ws,
    onopen: null,
    ontext: null,
    onbinary: null,
    onclose: null,
    onerror: null,
    readyState: () => ws.readyState,
    sendText(obj) {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
    },
    sendBinary(buffer) {
      if (ws.readyState === WebSocket.OPEN) ws.send(buffer);
    },
    close() {
      try { if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close(1000, "client-stop"); }
      catch { /* already closed */ }
    },
  };

  ws.addEventListener("open", () => { if (wrapper.onopen) wrapper.onopen(); });
  ws.addEventListener("message", (ev) => {
    if (typeof ev.data === "string") {
      let msg = null;
      try { msg = JSON.parse(ev.data); } catch { msg = { type: "error", message: "bad json" }; }
      if (wrapper.ontext) wrapper.ontext(msg);
    } else if (wrapper.onbinary) {
      wrapper.onbinary(ev.data);
    }
  });
  ws.addEventListener("close", (ev) => { if (wrapper.onclose) wrapper.onclose(ev.code, ev.reason); });
  ws.addEventListener("error", () => { if (wrapper.onerror) wrapper.onerror(); });
  return wrapper;
}

/* ------------------------------ DOM helpers ------------------------------ */

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, String(value));
  }
  const flat = [];
  const collect = (child) => {
    if (Array.isArray(child)) child.forEach(collect);
    else flat.push(child);
  };
  collect(Array.isArray(children) ? children : [children]);
  for (const child of flat) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function qs(selector, root = document) { return root.querySelector(selector); }
export function qsa(selector, root = document) { return Array.from(root.querySelectorAll(selector)); }

export function clear(node) { while (node.firstChild) node.firstChild.remove(); return node; }

export function icon(name, size = 18) {
  const paths = {
    upload: "M12 3 L12 15 M7 8 L12 3 L17 8 M4 17 L4 19 L20 19 L20 17",
    download: "M12 15 L12 3 M7 10 L12 15 L17 10 M4 17 L4 19 L20 19 L20 17",
    wave: "M3 12 L7 12 L9 7 L12 17 L15 7 L17 12 L21 12",
    alert: "M12 3 L22 20 L2 20 Z M12 10 L12 15 M12 17.5 L12 18.5",
    check: "M4 12 L9 17 L20 6",
    mic: "M12 3 a4 4 0 0 1 4 4 v5 a4 4 0 0 1 -8 0 v-5 a4 4 0 0 1 4 -4 M6 12 a6 6 0 0 0 12 0 M12 18 L12 22",
    stop: "M6 6 h12 v12 h-12 Z",
    play: "M7 4 L19 12 L7 20 Z",
    copy: "M8 8 h11 v13 h-11 Z M5 16 V4 h11",
    trash: "M4 6 h16 M9 6 V4 h6 v2 M6 6 l1 14 h10 l1-14",
    info: "M12 21 a9 9 0 1 1 0-18 a9 9 0 0 1 0 18 Z M12 11 L12 16 M12 8 L12 8.5",
    refresh: "M20 12 a8 8 0 1 1 -3 -6.2 M20 4 L20 9 L15 9",
    cpu: "M8 8 h8 v8 h-8 Z M5 5 h14 v14 h-14 Z M12 2 V5 M12 19 V22 M2 12 H5 M19 12 H22",
  };
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size);
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.style.flex = "0 0 auto";
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", paths[name] || paths.info);
  svg.append(p);
  return svg;
}

/* ------------------------------ formatting ------------------------------ */

const NF = (digits) => new Intl.NumberFormat("en-US", { maximumFractionDigits: digits });

export function fmtNum(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return NF(digits).format(n);
}

export function fmtMB(mb) {
  const n = Number(mb);
  if (!Number.isFinite(n) || n <= 0) return "0 MB";
  if (n >= 1024) return `${fmtNum(n / 1024, 2)} GB`;
  return `${fmtNum(n, 1)} MB`;
}

export function fmtGB(gb) {
  const n = Number(gb);
  if (!Number.isFinite(n) || n <= 0) return "—";
  return `${fmtNum(n, 1)} GB`;
}

export function fmtPct(fraction) {
  const n = Number(fraction);
  if (!Number.isFinite(n) || n < 0) return "0%";
  return `${Math.round(Math.min(n, 1) * 100)}%`;
}

/** ms formatting for latency numbers. */
export function fmtMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n === 0) return "—";
  return `${fmtNum(n, 1)} ms`;
}

/** epoch timestamp (seconds or ms) → local Arabic-friendly string. */
export function fmtTime(ts) {
  if (!ts) return "—";
  const date = new Date(ts < 1e12 ? ts * 1000 : ts);
  if (isNaN(date.getTime())) return "—";
  const pad = (x) => String(x).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function fmtDuration(seconds) {
  const n = Number(seconds);
  if (!Number.isFinite(n)) return "—";
  const m = Math.floor(n / 60);
  const s = Math.round(n % 60);
  return m > 0 ? `${m} د ${s} ث` : `${s} ث`;
}

export function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

export function encodeURIComponentSafe(value) {
  return encodeURIComponent(String(value ?? ""));
}

/* ------------------------------ common labels ------------------------------ */

export const ENGINE_LABELS = {
  ready: "جاهز",
  loading: "قيد التحميل",
  busy: "مشغول",
  unavailable: "غير متاح",
  error: "خطأ",
};

export const TRAIN_STATES = {
  idle: "خامل",
  preparing: "تجهيز",
  running: "يعمل",
  paused: "متوقف مؤقتًا",
  stopping: "جارٍ الإيقاف",
  completed: "اكتمل",
  failed: "فشل",
  stopped: "أُوقف",
};

export const F0_METHODS = [
  { value: "rmvpe", label: "RMVPE — الأدق للكلام العربي" },
  { value: "fcpe", label: "FCPE — سريع وموفّر" },
  { value: "pm", label: "PM — بديل على المعالج (CPU)" },
];

/** Reusable: builds a <select> of f0 methods. */
export function f0MethodSelect(selected = "rmvpe", id = "") {
  const select = el("select", { class: "f0-method", ...(id ? { id } : {}) });
  for (const m of F0_METHODS) {
    const opt = el("option", { value: m.value }, [m.label]);
    if (m.value === selected) opt.selected = true;
    select.append(opt);
  }
  return select;
}

/* ------------------------------ button busy state ------------------------------ */

export function setBusy(button, busy, busyText = "جارٍ التنفيذ…") {
  if (!button) return;
  if (busy) {
    if (!button.dataset.origHtml) button.dataset.origHtml = button.innerHTML;
    button.disabled = true;
    button.innerHTML = "";
    button.append(el("span", { class: "spin" }), document.createTextNode(` ${busyText}`));
  } else {
    button.disabled = false;
    if (button.dataset.origHtml) button.innerHTML = button.dataset.origHtml;
  }
}
