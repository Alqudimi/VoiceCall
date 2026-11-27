#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants and helpers for the changes-report pipeline.

All modules of scripts/changes_report/ import from here so the classification
and Arabic labels stay consistent between the charts, the map and CHANGES.md.
"""
import re

# ---- path classification -------------------------------------------------

CATEGORIES = ("source", "tests", "docs", "ci", "scripts", "configs",
               "notebooks", "reports", "root")

CAT_AR = {
    "source":    "شيفرة المصدر",
    "tests":     "الاختبارات",
    "docs":      "التوثيق",
    "ci":        "أتمتة CI/CD",
    "scripts":   "السكربتات",
    "configs":   "الإعدادات",
    "notebooks": "الدفاتر التفاعلية",
    "reports":   "التقارير",
    "root":      "ملفات الجذر",
}

CAT_COLOR = {
    "source":    "#2563EB",
    "tests":     "#059669",
    "docs":      "#B45309",
    "ci":        "#7C3AED",
    "scripts":   "#64748B",
    "configs":   "#87503D",
    "notebooks": "#0EA5E9",
    "reports":   "#BE123C",
    "root":      "#64748B",
}


def classify(path):
    """Map a repository path to one of CATEGORIES."""
    p = path.lower()
    if p.startswith("src/"):
        return "source"
    if p.startswith("tests/"):
        return "tests"
    if p.startswith(".github/"):
        return "ci"
    if p.startswith("configs/"):
        return "configs"
    if p.startswith("scripts/"):
        return "scripts"
    if p.startswith("notebooks/"):
        return "notebooks"
    if p.startswith("reports/"):
        return "reports"
    if p.startswith("docs/") or p.endswith(".md") or p == "mkdocs.yml":
        return "docs"
    return "root"


# ---- commit type classification -------------------------------------------

TYPE_INFO = {
    "t-init":  ("تأسيس المشروع", "#DBEAFE", "#1D4ED8"),
    "t-feat":  ("ميزة جديدة",    "#D1FAE5", "#047857"),
    "t-fix":   ("إصلاح",         "#FFE4E6", "#BE123C"),
    "t-docs":  ("توثيق",         "#FEF3C7", "#92400E"),
    "t-rel":   ("إصدار",         "#EDE9FE", "#5B21B6"),
    "t-chore": ("صيانة",         "#F1F5F9", "#475569"),
    "t-dev":   ("تطوير",         "#E0F2FE", "#0369A1"),
}


def commit_type(subject):
    """Return (label_ar, type_class, bg_fill, border_color) for a commit subject."""
    s = (subject or "").strip()
    low = s.lower()
    if low.startswith("feat"):
        tcls = "t-feat"
    elif low.startswith("fix"):
        tcls = "t-fix"
    elif low.startswith("docs"):
        tcls = "t-docs"
    elif low.startswith(("chore", "style", "refactor", "perf", "test", "build", "ci")):
        tcls = "t-chore"
    elif re.match(r"^v\d+\.\d+", low):
        tcls = "t-rel"
    elif "initial" in low:
        tcls = "t-init"
    else:
        tcls = "t-dev"
    label = TYPE_INFO[tcls][0]
    return label, tcls, TYPE_INFO[tcls][1], TYPE_INFO[tcls][2]


# ---- Arabic calendar names -------------------------------------------------

AR_DAYS = {0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
           4: "الجمعة", 5: "السبت", 6: "الأحد"}

AR_MONTHS = {1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو",
             6: "يونيو", 7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر",
             11: "نوفمبر", 12: "ديسمبر"}


def ar_date(datestr):
    """'2026-11-27' -> '27 نوفمبر' (Arabic day + month)."""
    y, m, d = (int(x) for x in datestr.split("-")[:3])
    return f"{d} {AR_MONTHS[m]}"


def ar_weekday(datestr):
    """Arabic weekday name for an ISO date string."""
    from datetime import date as _date
    y, m, d = (int(x) for x in datestr.split("-")[:3])
    return AR_DAYS[_date(y, m, d).weekday()]


def fmt(n):
    """Thousands-separated number, e.g. 71509 -> '71,509'."""
    return f"{n:,}"


def truncate(text, n=40):
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"
