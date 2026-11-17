/* ============================================================
   models.js — النماذج: قائمة، تحميل إلى المحرك، رفع .pth،
   تفاصيل، حذف مع تأكيد. + لافتة الموافقة الأخلاقية.
   ============================================================ */

import {
  bus, get, post, postFormWithProgress, del, getBlob, toast, confirmDialog,
  el, clear, icon, setBusy, fmtMB, fmtTime, fmtNum, ENGINE_LABELS,
} from "../api.js";

export default { mount };

async function mount(root) {
  root.replaceChildren();
  root.append(
    buildConsentBanner(),
    buildLoadedCard(),
    buildUploadCard(),
    buildListCard(),
  );

  const offSystem = bus.on("system", (system) => renderLoaded(system?.loaded_model || null, system));
  await Promise.allSettled([refreshLoaded(), refreshList()]);

  function cleanup() { offSystem(); }
  return cleanup;
}

/* ------------------------------ consent banner ------------------------------ */

function buildConsentBanner() {
  return el("div", { class: "notice warn" }, [
    el("span", { class: "n-icon" }, [icon("alert", 18)]),
    el("p", {}, [
      "ارفع فقط نماذج لأصوات تملك حق استخدامها. استنساخ صوت شخص دون إذنه غير مسموح.",
    ]),
  ]);
}

/* ------------------------------ loaded model ------------------------------ */

function buildLoadedCard() {
  return el("div", { class: "card", id: "models-loaded" }, [
    el("h2", { class: "card-title" }, [
      "النموذج المحمّل في المحرك",
      el("span", { class: "sub latin" }, ["Loaded Model"]),
    ]),
    el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
  ]);
}

async function refreshLoaded() {
  try {
    const system = await get("/system");
    renderLoaded(system.loaded_model || null, system);
  } catch (err) {
    const card = document.getElementById("models-loaded");
    if (card) {
      clear(card);
      card.append(el("h2", { class: "card-title" }, ["النموذج المحمّل في المحرك"]));
      card.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
    }
  }
}

function renderLoaded(loaded, system) {
  const card = document.getElementById("models-loaded");
  if (!card) return;
  clear(card);
  card.append(el("h2", { class: "card-title" }, [
    "النموذج المحمّل في المحرك",
    el("span", { class: "sub latin" }, ["Loaded Model"]),
  ]));
  if (!loaded) {
    card.append(el("p", { class: "muted" }, ["لا يوجد نموذج محمّل — اختر نموذجًا من القائمة واضغط \"تحميل إلى المحرك\"."]));
    return;
  }
  const engineState = system ? system.engine : "unknown";
  card.append(el("div", { class: "row" }, [
    el("span", { class: `badge ${engineState} dot` }, [`المحرك: ${ENGINE_LABELS[engineState] || engineState}`]),
    el("span", { class: "badge ok" }, [loaded.name]),
  ]));
  card.append(el("dl", { class: "kv mt" }, [
    dtdd("معدل العينة", `${fmtNum(loaded.sample_rate / 1000, 1)} kHz`),
    dtdd("الطبقة (f0)", loaded.f0 ? "مدعومة" : "غير مدعومة"),
    dtdd("الإصدار", el("span", { class: "latin" }, [loaded.version])),
    dtdd("الحجم", fmtMB(loaded.size_mb)),
    dtdd("الفهرس", loaded.index_path
      ? el("span", { class: "latin small" }, ["مرفق (index)"])
      : "لا يوجد"),
  ]));
  const btnUnload = el("button", { class: "btn danger", type: "button" }, ["تفريغ المحرك"]);
  btnUnload.addEventListener("click", async () => {
    setBusy(btnUnload, true, "جارٍ التفريغ…");
    try {
      await post("/models/unload");
      toast("تم تفريغ المحرك.", "success");
      await refreshLoaded();
    } catch (err) { toast(err.message, "error"); }
    finally { setBusy(btnUnload, false); }
  });
  card.append(el("div", { class: "row mt" }, [btnUnload, el("a", { class: "btn", href: "#convert" }, ["التحويل الآن"])]));
}

function dtdd(key, value) {
  return [
    el("dt", {}, [key]),
    el("dd", {}, [value instanceof Node ? value : document.createTextNode(String(value))]),
  ];
}

/* ------------------------------ upload ------------------------------ */

function buildUploadCard() {
  const fileInput = el("input", { type: "file", accept: ".pth" });
  const dzTitle = el("span", { class: "dz-title" }, ["اسحب ملف .pth هنا أو اضغط للاختيار"]);
  const dzSub = el("span", { class: "dz-sub" }, ["ملف نموذج RVC فقط — الحد الأقصى 300 MB. إذا وُجد ملف .index بنفس الاسم بجواره فسيُرفع معه تلقائيًا في الخادم."]);
  const dropzone = el("div", { class: "dropzone", tabindex: "0", role: "button", "aria-label": "اختيار ملف نموذج" }, [
    el("span", { class: "dz-icon" }, [icon("upload", 26)]),
    dzTitle, dzSub, fileInput,
  ]);

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fileInput.click(); }
  });
  for (const evt of ["dragenter", "dragover"]) {
    dropzone.addEventListener(evt, (ev) => { ev.preventDefault(); dropzone.classList.add("over"); });
  }
  for (const evt of ["dragleave", "drop"]) {
    dropzone.addEventListener(evt, (ev) => { ev.preventDefault(); dropzone.classList.remove("over"); });
  }
  dropzone.addEventListener("drop", (ev) => {
    const files = Array.from(ev.dataTransfer?.files || []);
    const pth = files.find((f) => f.name.toLowerCase().endsWith(".pth"));
    if (pth) uploadFile(pth, dropzone);
    else toast("الملف ليس نموذج .pth صالحًا.", "error");
  });
  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    if (f) uploadFile(f, dropzone);
  });

  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "رفع نموذج جديد",
      el("span", { class: "sub latin" }, ["Upload .pth"]),
    ]),
    dropzone,
  ]);
}

async function uploadFile(file, dropzone) {
  if (!file.name.toLowerCase().endsWith(".pth")) {
    toast("يُقبل فقط ملف بامتداد .pth", "error");
    return;
  }
  const oldTitle = dropzone.querySelector(".dz-title").textContent;
  dropzone.querySelector(".dz-title").textContent = `جارٍ رفع: ${file.name} (${fmtMB(file.size / (1024 * 1024))})`;
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const res = await postFormWithProgress("/models/upload", form, () => {});
    const name = res?.registered?.name || file.name;
    toast(`تم تسجيل النموذج \"${name}\" بنجاح.`, "success");
    await refreshList();
  } catch (err) {
    toast(err.message, "error", 12000);
    if (err.detail) toast(err.detail, "error", 12000);
  } finally {
    dropzone.querySelector(".dz-title").textContent = oldTitle;
  }
}

/* ------------------------------ list ------------------------------ */

function buildListCard() {
  return el("div", { class: "card" }, [
    el("h2", { class: "card-title" }, [
      "النماذج المتوفرة",
      el("button", {
        class: "btn sm", type: "button", id: "models-refresh",
        onclick: () => refreshList(),
      }, [icon("refresh", 14), " تحديث"]),
    ]),
    el("div", { id: "models-list" }, [
      el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "…"]),
    ]),
  ]);
}

async function refreshList() {
  const box = document.getElementById("models-list");
  if (!box) return;
  try {
    const models = await get("/models");
    renderModels(box, Array.isArray(models) ? models : []);
  } catch (err) {
    clear(box);
    box.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
  }
}

function renderModels(box, models) {
  clear(box);
  if (!models.length) {
    box.append(el("p", { class: "muted" }, ["لا توجد نماذج مسجّلة — ارفع أول نموذج من الأعلى."]));
    return;
  }
  const tbody = el("tbody");
  for (const m of models) {
    tbody.append(modelRow(m));
    tbody.append(detailRow(m));
  }
  box.append(el("div", { class: "table-wrap" }, [
    el("table", { class: "tbl" }, [
      el("thead", {}, [el("tr", {}, [
        el("th", {}, ["الاسم"]),
        el("th", {}, ["معدل العينة (SR)"]),
        el("th", {}, ["f0"]),
        el("th", {}, ["الإصدار"]),
        el("th", {}, ["الحجم"]),
        el("th", {}, ["الفهرس"]),
        el("th", {}, ["التفاصيل"]),
        el("th", {}, ["إجراءات"]),
      ])]),
      tbody,
    ]),
  ]));
}

function modelRow(m) {
  const tr = el("tr", { dataset: { name: m.name } }, [
    el("td", { class: "latin" }, [m.name]),
    el("td", { class: "num" }, [`${fmtNum(m.sample_rate / 1000, 1)} kHz`]),
    el("td", {}, [m.f0 ? el("span", { class: "badge ok" }, ["نعم"]) : el("span", { class: "badge info" }, ["لا"])]),
    el("td", { class: "latin" }, [m.version || "—"]),
    el("td", { class: "num" }, [fmtMB(m.size_mb)]),
    el("td", {}, [m.index_path ? el("span", { class: "badge ok" }, ["مرفق"]) : el("span", { class: "muted small" }, ["—"])]),
    el("td", {}, [m.info ? el("span", { class: "small dim" }, [m.info]) : el("span", { class: "muted small" }, ["—"])]),
    el("td", {}, [actionsCell(m)]),
  ]);
  return tr;
}

function actionsCell(m) {
  const wrap = el("div", { class: "actions" });

  const btnLoad = el("button", { class: "btn sm primary", type: "button" }, ["تحميل إلى المحرك"]);
  btnLoad.addEventListener("click", async () => {
    setBusy(btnLoad, true, "جارٍ تحميل النموذج…");
    try {
      await post(`/models/${encodeURIComponent(m.name)}/load`);
      toast(`تم تحميل \"${m.name}\" إلى المحرك.`, "success");
      await refreshLoaded();
    } catch (err) {
      toast(err.message, "error", 10000);
    } finally { setBusy(btnLoad, false); }
  });

  const btnDetails = el("button", { class: "btn sm ghost", type: "button" }, ["تفاصيل"]);
  btnDetails.addEventListener("click", () => toggleDetails(m.name));

  const btnDownload = el("button", { class: "btn sm ghost", type: "button" }, ["تنزيل"]);
  btnDownload.addEventListener("click", async () => {
    setBusy(btnDownload, true, "جارٍ التنزيل…");
    try {
      const blob = await getBlob(`/models/${encodeURIComponent(m.name)}/download`);
      const url = URL.createObjectURL(blob);
      const a = el("a", { href: url, download: `${m.name}.pth` });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      toast(`تم تنزيل "${m.name}".`, "success");
    } catch (err) {
      toast(err.message, "error");
    } finally { setBusy(btnDownload, false); }
  });

  const btnDelete = el("button", { class: "btn sm danger", type: "button" }, ["حذف"]);
  btnDelete.addEventListener("click", async () => {
    const ok = await confirmDialog({
      title: "حذف النموذج",
      message: `سيُحذف النموذج \"${m.name}\" نهائيًا من الخادم (مع فهرسه إن وُجد). هل أنت متأكد؟`,
      confirmText: "حذف نهائي",
      cancelText: "إلغاء",
      danger: true,
    });
    if (!ok) return;
    setBusy(btnDelete, true, "جارٍ الحذف…");
    try {
      await del(`/models/${encodeURIComponent(m.name)}`);
      toast(`تم حذف \"${m.name}\".`, "success");
      await Promise.allSettled([refreshList(), refreshLoaded()]);
    } catch (err) {
      toast(err.message, "error");
    } finally { setBusy(btnDelete, false); }
  });

  wrap.append(btnLoad, btnDetails, btnDownload, btnDelete);
  return wrap;
}

function detailRow(m) {
  const cell = el("td", { colspan: "8", class: "hidden", dataset: { detailFor: m.name } }, [
    el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "جارٍ جلب التفاصيل…"]),
  ]);
  const tr = el("tr", { class: "detail-row" }, [cell]);
  return tr;
}

async function toggleDetails(name) {
  const row = document.querySelector(`td[data-detail-for="${CSS.escape(name)}"]`);
  if (!row) return;
  if (!row.classList.contains("hidden")) {
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  clear(row);
  row.append(el("div", { class: "loading-line" }, [el("span", { class: "spin" }), "جارٍ جلب التفاصيل…"]));
  try {
    const info = await get(`/models/${encodeURIComponent(name)}`);
    clear(row);
    row.append(el("div", { class: "stack" }, [
      el("dl", { class: "kv" }, [
        dtdd("الاسم", el("span", { class: "latin" }, [info.name || name])),
        dtdd("معدل العينة (SR)", info.sr ? `${fmtNum(info.sr / 1000, 1)} kHz` : "—"),
        dtdd("الطبقة (f0)", info.f0 === true ? "مدعومة" : info.f0 === false ? "غير مدعومة" : "—"),
        dtdd("الإصدار", el("span", { class: "latin" }, [info.version || "—"])),
        dtdd("الحجم", fmtMB(info.size_mb)),
        dtdd("المعلومة الداخلية", el("span", { class: "latin small" }, [info.info || "—"])),
        dtdd("الفهرس (index)", info.index ? el("span", { class: "latin small" }, [info.index]) : "لا يوجد"),
        dtdd("مرجع فهرس داخل الملف", info.has_index_ref ? "موجود" : "لا يوجد"),
        dtdd("محمّل في المحرك", info.engine_loaded ? el("span", { class: "badge ok" }, ["نعم"]) : el("span", { class: "badge idle" }, ["لا"])),
        ...(info.speakers && info.speakers.length ? dtdd("المتحدثون", el("span", { class: "latin small" }, [info.speakers.join(", ")])) : []),
      ]),
    ]));
  } catch (err) {
    clear(row);
    row.append(el("div", { class: "notice danger" }, [el("p", {}, [err.message])]));
  }
}
