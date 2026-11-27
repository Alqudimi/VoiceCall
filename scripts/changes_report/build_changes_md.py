#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-generate the comprehensive Arabic CHANGES.md from commits.json, plus
summary.json (aggregates reused by the wiki builder) and release_notes.md
(body of the rolling GitHub release).

The generated document covers EVERY commit and EVERY file-change event with
per-file additions/deletions — 100% coverage, fully reproducible from git.
"""
import argparse
import ast
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import (CAT_AR, CATEGORIES, ar_date, ar_weekday, fmt, truncate)

EXT_LINES = (".py", ".js", ".css", ".md", ".sh", ".cmd", ".ps1", ".json",
             ".yml", ".yaml", ".html")


def git(repo, *args, default=""):
    try:
        res = subprocess.run(["git", "-C", str(repo), *args], check=True,
                             capture_output=True, text=True)
        return res.stdout
    except subprocess.CalledProcessError:
        return default


def compute_summary(repo_root, commits):
    """All aggregates used by CHANGES.md / wiki / release notes."""
    s = {}
    s["n_commits"] = len(commits)
    s["total_ins"] = sum(c["total_ins"] for c in commits)
    s["total_del"] = sum(c["total_del"] for c in commits)
    s["n_file_events"] = sum(c["n_files"] for c in commits)
    uniq = set()
    for c in commits:
        for fl in c["files"]:
            uniq.add(fl["path"])
    s["n_unique_files"] = len(uniq)

    days = defaultdict(list)
    for c in commits:
        days[c["date"][:10]].append(c)
    s["n_days"] = len(days)
    s["days"] = []
    for d in sorted(days):
        dc = days[d]
        s["days"].append({
            "date": d,
            "weekday_ar": ar_weekday(d),
            "date_ar": ar_date(d),
            "n_commits": len(dc),
            "n_files": sum(c["n_files"] for c in dc),
            "ins": sum(c["total_ins"] for c in dc),
            "del": sum(c["total_del"] for c in dc),
            "subjects": [c["subject"] for c in dc],
        })
    busiest = max(s["days"], key=lambda x: x["n_commits"])
    s["busiest_day"] = busiest["date_ar"]
    s["busiest_day_commits"] = busiest["n_commits"]

    tcount = Counter(c["type_label"] for c in commits)
    s["types"] = dict(tcount)

    ccount = Counter()
    fcount = Counter()
    for c in commits:
        for fl in c["files"]:
            ccount[fl["cat"]] += 1
            fcount[fl["path"]] += 1
    s["cats"] = dict(ccount)
    s["top_files"] = fcount.most_common(15)

    author_counts = Counter((c["author_name"], c["author_email"]) for c in commits)
    s["authors"] = [{"name": n, "email": e, "n": v} for (n, e), v in author_counts.most_common()]
    s["via_bot"] = sum(1 for c in commits if c["author_email"] != c["committer_email"])

    tags = []
    for c in commits:
        for name, tdate in zip(c["tags"], c["tag_dates"]):
            tags.append({"name": name, "sha_short": c["short"], "date": tdate,
                         "subject": c["subject"]})
    tags.sort(key=lambda t: t["date"])
    s["tags"] = tags
    s["tags_total"] = len(tags)

    first, last = commits[0], commits[-1]
    s["first_commit"] = {"date": first["date"], "short": first["short"],
                         "subject": first["subject"],
                         "n_files": first["n_files"],
                         "ins": first["total_ins"], "del": first["total_del"]}
    s["last_commit"] = {"date": last["date"], "short": last["short"],
                        "subject": last["subject"]}
    dt1 = datetime.strptime(first["date"], "%Y-%m-%d %H:%M")
    dt2 = datetime.strptime(last["date"], "%Y-%m-%d %H:%M")
    dur = dt2 - dt1
    total_h = dur.total_seconds() / 3600
    d, rem = divmod(total_h, 24)

    def ar_count(n, one, two, few, many):
        if n == 1:
            return one
        if n == 2:
            return two
        if 3 <= n <= 10:
            return f"{int(n)} {few}"
        return f"{int(n)} {many}"

    d_txt = ar_count(d, "يوم واحد", "يومان", "أيام", "يومًا")
    h_txt = ar_count(rem, "ساعة واحدة", "ساعتان", "ساعات", "ساعة")
    s["duration_text"] = f"{d_txt} و{h_txt}" if d >= 1 else h_txt
    s["avg_per_day"] = round(len(commits) / len(days), 1)

    biggest = max(commits, key=lambda c: c["total_ins"] + c["total_del"])
    s["biggest"] = {"short": biggest["short"], "subject": biggest["subject"],
                    "n_files": biggest["n_files"],
                    "ins": biggest["total_ins"], "del": biggest["total_del"],
                    "seq": biggest["seq"]}

    s["latest_commits"] = [{
        "seq": c["seq"], "short": c["short"], "date": c["date"],
        "subject": c["subject"], "ins": c["total_ins"],
        "del": c["total_del"], "n_files": c["n_files"],
    } for c in commits[-5:]]
    return s


def identity_from_pyproject(repo_root):
    ident = {}
    path = Path(repo_root) / "pyproject.toml"
    try:
        import tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f).get("project", {})
        ident["name"] = data.get("name", "")
        ident["version"] = data.get("version", "")
        ident["description"] = data.get("description", "")
        ident["python"] = data.get("requires-python", "")
        lic = data.get("license", {})
        ident["license"] = lic.get("text", "") if isinstance(lic, dict) else str(lic)
        authors = data.get("authors", [])
        ident["authors"] = ", ".join(a.get("name", "") for a in authors if isinstance(a, dict))
    except FileNotFoundError:
        pass
    return ident


def tree_stats(repo_root):
    files = [l for l in git(repo_root, "ls-files").splitlines() if l.strip()]
    lines_by_ext = Counter()
    files_by_ext = Counter()
    for fl in files:
        ext = Path(fl).suffix.lower()
        files_by_ext[ext] += 1
    for fl in files:
        ext = Path(fl).suffix.lower()
        if ext in EXT_LINES:
            p = Path(repo_root) / fl
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                lines_by_ext[ext] += text.count("\n") + 1
            except OSError:
                pass

    def count_functions(subdir):
        total = 0
        base = Path(repo_root) / subdir
        if not base.exists():
            return 0
        for p in base.rglob("*.py"):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total += 1
        return total

    src_funcs = count_functions("src")
    test_funcs = count_functions("tests")
    workflows = len(list((Path(repo_root) / ".github" / "workflows").glob("*"))) \
        if (Path(repo_root) / ".github" / "workflows").exists() else 0
    return {
        "tracked_files": len(files),
        "py_files": files_by_ext.get(".py", 0),
        "py_lines": lines_by_ext.get(".py", 0),
        "js_lines": lines_by_ext.get(".js", 0),
        "css_lines": lines_by_ext.get(".css", 0),
        "md_files": files_by_ext.get(".md", 0),
        "md_lines": lines_by_ext.get(".md", 0),
        "sh_files": files_by_ext.get(".sh", 0) + files_by_ext.get(".cmd", 0) + files_by_ext.get(".ps1", 0),
        "json_files": files_by_ext.get(".json", 0),
        "test_files": sum(1 for f in files if f.startswith("tests/") and f.endswith(".py")),
        "src_functions": src_funcs,
        "test_functions": test_funcs,
        "workflows": workflows,
    }


def slug_from_repo(repo_root, provided):
    if provided:
        return provided
    env_slug = os.environ.get("GITHUB_REPOSITORY", "")
    if env_slug:
        return env_slug
    url = git(repo_root, "remote", "get-url", "origin").strip()
    if "github.com" in url:
        tail = url.split("github.com/")[-1].replace(".git", "")
        return tail.strip("/")
    return ""


def render_changes_md(commits, summ, repo_display, slug, release_tag, generated):
    N = len(commits)
    commit_base = f"https://github.com/{slug}/commit/" if slug else ""
    L = []
    add = L.append

    add(f"# CHANGES.md — السجل الشامل المولَّد آليًا لتغييرات مستودع {repo_display}\n")
    add("> **هذا الملف يُولَّد آليًا بالكامل** بواسطة سير عمل GitHub Actions «Changes Report» عند كل دفع "
        f"جديد إلى الفرع الرئيسي — آخر تحديث: **{generated} UTC** — الرأس الموثَّق: "
        f"`{commits[-1]['hash']}`. لا تُعدَّل هذا الملف يدويًا؛ أي تعديل يدوي سيُستبدل مع "
        "التحديث التالي. مصدر الحقيقة: `git log --numstat` على كامل سجل الفرع.\n")
    add("---\n")

    # ---------------- §0 ----------------
    add("## 0. كيف تقرأ هذا الملف\n")
    add("هذا السجل مولَّد مباشرة من بيانات Git الخام بلا أي تدخل يدوي: لكل التزام بطاقة "
        "تعريفية كاملة (الهاش الكامل والمختصر، التاريخ والوقت، المؤلف، الوسوم إن وجدت، "
        "النوع، عدد الملفات، الأسطر المضافة والمحذوفة) **وجدول كامل بكل ملف تعرّض للتغيير "
        "في ذلك التزام تحديدًا مع أرقام الإضافات والحذوفات لكل ملف** — أي تغطية "
        f"**{fmt(summ['n_file_events'])}/{fmt(summ['n_file_events'])}** حدث تغيير ملف "
        f"من أصل {N} التزامًا، بنسبة **100%** بلا استثناء.\n")
    add("| إذا كنت تريد... | اذهب إلى... |")
    add("|---|---|")
    add("| الأرقام الإجمالية والحسّية في جدول واحد | §3 «الإحصاءات الكلية» |")
    add("| الوسوم كلها مرتبطة بالتزاماتها | §4 «ملحق الوسوم» |")
    add("| تفاصيل التزام بعينه وملفاته كاملة | §5 «السجل التفصيلي الكامل» |")
    add("| الرسوم البيانية المرافقة | §6 «ملحق الرسوم» |")
    add("| كيف يُولَّد هذا الملف وكيف تُعيد توليده | §7 «المنهجية» |\n")

    # ---------------- §1 ----------------
    add("## 1. الملخص التنفيذي — قصة السجل بالأرقام\n")
    first, biggest = summ["first_commit"], summ["biggest"]
    add(f"يفتتح سجل التغييرات التزام «{first['subject']}» ({first['short']}) بتاريخ "
        f"**{first['date']}** الذي أدخل **{fmt(first['n_files'])} ملفًا** دفعة واحدة "
        f"(**+{fmt(first['ins'])}** سطرًا مضافًا) — وهو {'أكبر' if first['ins'] >= biggest['ins'] else 'من أكبر'} "
        "التزامات السجل على الإطلاق، ما يعني أن المشروع وُلد ببنية كاملة منذ أول لحظة ثم "
        "تطوّر فوقها.\n")
    add(f"يمتد السجل الزمني على **{summ['duration_text']}** موزعة على **{summ['n_days']} يومًا "
        f"نشطًا** (متوسط **{summ['avg_per_day']} التزام/يوم نشط**)، أنتجت **{fmt(N)} التزامًا** "
        f"عدّلت **{fmt(summ['n_file_events'])} حدثًا لتغيير الملفات** عبر **{fmt(summ['n_unique_files'])} "
        f"ملفًا فريدًا**، بإجمالي **+{fmt(summ['total_ins'])} سطرًا مضافًا** و**−{fmt(summ['total_del'])} "
        "سطرًا محذوفًا** — أي أن النمو الصافي للشيفرة "
        f"**+{fmt(summ['total_ins'] - summ['total_del'])} سطرًا**.\n")
    add(f"أكبر التزام في السجل هو **«{biggest['subject']}»** ({biggest['short']} — التزام رقم "
        f"#{biggest['seq']:02d}) بموجة **+{fmt(biggest['ins'])}/−{fmt(biggest['del'])}** عبر "
        f"{fmt(biggest['n_files'])} ملفًا. أما اليوم الأنشط فهو **{summ['busiest_day']}** "
        f"بواقع **{summ['busiest_day_commits']} التزامات**.\n")
    type_txt = "، ".join(f"**{k}** ({v})" for k, v in sorted(summ["types"].items(), key=lambda kv: -kv[1]))
    add(f"توزيع الالتزامات حسب النوع: {type_txt}. ومرّت **{summ['via_bot']}** التزامًا عبر "
        "هوية مُلتزِم (committer) مختلفة عن هوية المؤلف — دليل على إنجاز مُؤتمَت.\n")
    day_story = " ← ".join(f"{d['date_ar']} ({d['n_commits']})" for d in summ["days"])
    add(f"خيط الأيام النشطة: {day_story}.\n")

    # ---------------- §2 ----------------
    add("## 2. بطاقة هوية المستودع\n")
    add("| البند | القيمة |")
    add("|---|---|")
    if slug:
        add(f"| **المستودع** | [`{slug}`](https://github.com/{slug}) |")
    if summ["identity"]["name"]:
        add(f"| **اسم الحزمة** | `{summ['identity']['name']}` |")
        add(f"| **الإصدار الحالي في pyproject** | `{summ['identity']['version']}` |")
        add(f"| **الوصف** | {summ['identity']['description']} |")
        add(f"| **متطلب Python** | `{summ['identity']['python']}` |")
        add(f"| **الترخيص** | {summ['identity']['license']} |")
        add(f"| **المؤلفون المسجلون** | {summ['identity']['authors']} |")
    t = summ["tree"]
    add(f"| **الملفات المتتبعة حاليًا** | {fmt(t['tracked_files'])} |")
    add(f"| **أسطر Python** | {fmt(t['py_lines'])} في {fmt(t['py_files'])} ملفًا |")
    add(f"| **أسطر JavaScript** | {fmt(t['js_lines'])} |")
    add(f"| **أسطر CSS** | {fmt(t['css_lines'])} |")
    add(f"| **وثائق Markdown** | {fmt(t['md_files'])} ملفًا / {fmt(t['md_lines'])} سطرًا |")
    add(f"| **ملفات الاختبارات** | {fmt(t['test_files'])} |")
    add(f"| **دوال Python المُعرَّفة** | {fmt(t['src_functions'])} في المصدر + {fmt(t['test_functions'])} في الاختبارات |")
    add(f"| **سير عمل CI/CD** | {fmt(t['workflows'])} |")
    author_line = "، ".join(f"{a['name']} ({a['n']} التزامًا)" for a in summ["authors"])
    add(f"| **المساهمون (من سجل Git)** | {author_line} |\n")

    # ---------------- §3 ----------------
    add("## 3. الإحصاءات الكلية\n")
    add("### 3.1 الأرقام الرئيسية\n")
    add("| القياس | القيمة |")
    add("|---|---|")
    add(f"| إجمالي الالتزامات | **{N}** |")
    add(f"| أول التزام | `{first['short']}` — {first['date']} — «{first['subject']}» |")
    add(f"| آخر التزام | `{summ['last_commit']['short']}` — {summ['last_commit']['date']} — «{summ['last_commit']['subject']}» |")
    add(f"| عمر السجل | **{summ['duration_text']}** |")
    add(f"| أيام نشطة | {summ['n_days']} يومًا |")
    add(f"| إجمالي أحداث تغيير الملفات | **{fmt(summ['n_file_events'])}** |")
    add(f"| ملفات فريدة | **{fmt(summ['n_unique_files'])}** |")
    add(f"| الأسطر المضافة | **+{fmt(summ['total_ins'])}** |")
    add(f"| الأسطر المحذوفة | **−{fmt(summ['total_del'])}** |")
    add(f"| الوسوم | **{summ['tags_total']}** |\n")

    add("### 3.2 النشاط اليومي\n")
    add("| اليوم | التاريخ | الالتزامات | أحداث الملفات | إضافات | حذوفات |")
    add("|---|---|---|---|---|---|")
    for d in summ["days"]:
        add(f"| {d['weekday_ar']} | {d['date']} | {d['n_commits']} | {fmt(d['n_files'])} "
            f"| **+{fmt(d['ins'])}** | −{fmt(d['del'])} |")
    add("")
    add("### 3.3 توزيع أحداث التغيير حسب فئة الملف\n")
    add("| الفئة | أحداث التغيير | النسبة |")
    add("|---|---|---|")
    for cat in CATEGORIES:
        n = summ["cats"].get(cat, 0)
        if n == 0:
            continue
        pct = n / summ["n_file_events"] * 100
        add(f"| {CAT_AR[cat]} | {fmt(n)} | {pct:.1f}% |")
    add("")
    add("### 3.4 أكثر الملفات تعديلًا (عدد مرات الظهور في الالتزامات)\n")
    add("| # | الملف | مرات التغيير |")
    add("|---|---|---|")
    for i, (path, n) in enumerate(summ["top_files"], 1):
        add(f"| {i} | `{path}` | **{n}** |")
    add("")

    # ---------------- §4 ----------------
    add("## 4. ملحق الوسوم — كل الوسوم مربوطة بالتزاماتها\n")
    if summ["tags_total"]:
        add("| # | الوسم | الالتزام | تاريخ الوسم | رسالة الالتزام |")
        add("|---|---|---|---|---|")
        for i, t in enumerate(summ["tags"], 1):
            sha = t["sha_short"]
            linked = f"[`{sha}`]({commit_base}{sha})" if commit_base else f"`{sha}`"
            add(f"| {i} | `{t['name']}` | {linked} | {t['date']} | {truncate(t['subject'], 60)} |")
    else:
        add("لا توجد وسوم في هذا المستودع.")
    add("")

    # ---------------- §5 ----------------
    add("## 5. السجل التفصيلي الكامل — يومًا بيوم، التزامًا بالتزام، ملفًا بملف\n")
    add(f"> **تغطية {N}/{N} التزامًا**: لكل التزام بطاقة كاملة **وجدول بكل ملف متأثر فيه** "
        "مع أرقام +إضافات/−حذوفات لكل ملف. الملفات الثنائية تُعلَّم «ثنائي».\n")
    days = defaultdict(list)
    for c in commits:
        days[c["date"][:10]].append(c)
    for d in sorted(days):
        dcommits = days[d]
        dins = sum(c["total_ins"] for c in dcommits)
        ddel = sum(c["total_del"] for c in dcommits)
        dfiles = sum(c["n_files"] for c in dcommits)
        add(f"### اليوم {ar_date(d)} — {ar_weekday(d)} "
            f"({len(dcommits)} التزامًا · {fmt(dfiles)} حدث ملف · +{fmt(dins)}/−{fmt(ddel)})\n")
        for c in dcommits:
            sha = c["hash"]
            short = c["short"]
            add(f"#### التزام {c['seq']:02d}/{N} — «{c['subject']}»\n")
            add("| البند | القيمة |")
            add("|---|---|")
            linked = f"`{sha}` ([{short}]({commit_base}{sha}))" if commit_base else f"`{sha}`"
            add(f"| الهاش | {linked} |")
            add(f"| التاريخ والوقت | {c['date']} |")
            add(f"| المؤلف | {c['author_name']} `<{c['author_email']}>` |")
            if c["author_email"] != c["committer_email"]:
                add(f"| المُلتزِم | {c['committer_name']} `<{c['committer_email']}>` (مؤتمت) |")
            if c["tags"]:
                add(f"| الوسوم | {'، '.join('`' + t + '`' for t in c['tags'])} |")
            add(f"| النوع | {c['type_label']} |")
            add(f"| الملفات المتأثرة | **{c['n_files']}** |")
            add(f"| الحجم | **+{fmt(c['total_ins'])} / −{fmt(c['total_del'])}** |")
            cat_txt = "، ".join(f"{CAT_AR[k]}: {v}" for k, v in c["cats"].items())
            add(f"| توزيع الفئات | {cat_txt} |\n")
            if c["files"]:
                add("**الملفات المتأثرة في هذا الالتزام:**\n")
                add("| الفئة | الملف | إضافات | حذوفات |")
                add("|---|---|---|---|")
                for fl in c["files"]:
                    if fl["binary"]:
                        stats = "ثنائي | ثنائي"
                    else:
                        stats = f"**+{fmt(fl['ins'])}** | −{fmt(fl['del'])}"
                    add(f"| {CAT_AR[fl['cat']]} | `{fl['path']}` | {stats} |")
                add("")
    # ---------------- §6 ----------------
    add(f"## 6. ملحق الرسوم البيانية المرافقة (مرفقة بإصدار `{release_tag}`)\n")
    add("| الأصل في الإصدار | المحتوى |")
    add("|---|---|")
    add(f"| `{repo_display}_Commits_Changes_Map.png` | الخريطة الشاملة الكبرى: بطاقة لكل التزام مع كل ملفاته وأرقامه |")
    add(f"| `{repo_display}_Commits_Changes_Map.html` | نسخة تفاعلية من الخريطة (خطوط مدمجة) |")
    add(f"| `{repo_display}_Commits_Overview_Chart.png` | الإضافات/الحذوفات وعدد الملفات لكل التزام |")
    add(f"| `{repo_display}_Changes_Analytics_Dashboard.png` | لوحة التحليل: النشاط اليومي، الفئات، النمو التراكمي، أكثر الملفات |")
    add("| `CHANGES.md` | هذا الملف نفسه |")
    if slug:
        add(f"\nروابط التحميل المباشرة: `https://github.com/{slug}/releases/download/{release_tag}/<اسم-الأصل>` — "
            "وتُحدَّث الأصول تلقائيًا مع كل دفعة جديدة.\n")

    # ---------------- §7 ----------------
    add("## 7. المنهجية وإعادة الإنتاج\n")
    add("- **مصدر البيانات:** `git log --numstat --pretty=...` على كامل سجل الفرع (بدون تقريب، "
        "بدون تقدير يدوي) + `git for-each-ref refs/tags` للوسوم + `git ls-files` لبطاقة الهوية.\n")
    add("- **الخطوات:** (1) استخراج `commits.json`؛ (2) توليد الرسمين الإحصائيين؛ (3) بناء خريطة "
        "HTML ثم تصويرها PNG؛ (4) توليد هذا الملف و`summary.json` وملاحظات الإصدار؛ "
        "(5) دفع CHANGES.md إلى المستودع؛ (6) تحديث إصدار GitHub بالأصول؛ (7) تحديث الويكي.\n")
    add("- **إعادة التوليد يدويًا:** من تبويب Actions شغِّل سير العمل «Changes Report» "
        "(workflow_dispatch)، أو محليًا: `bash scripts/changes_report/run_pipeline.sh`.\n")
    add(f"- **آخر توليد:** {generated} UTC على الرأس `{commits[-1]['hash']}`.\n")
    add("---\n")
    add(f"> **التغطية المعلنة:** {N}/{N} التزامًا، {fmt(summ['n_file_events'])}/{fmt(summ['n_file_events'])} "
        f"حدث تغيير ملف، {summ['tags_total']}/{summ['tags_total']} وسمًا — **100%**.\n")
    return "\n".join(L)


def render_release_notes(summ, repo_display, slug, release_tag, generated):
    L = []
    add = L.append
    add(f"# تقرير التغييرات الشامل — {repo_display}\n")
    add(f"تقرير مولَّد آليًا بتاريخ **{generated} UTC** على الرأس `{summ['last_commit']['short']}` "
        "— يُحدَّث هذا الإصدار تلقائيًا عند كل دفع جديد إلى الفرع الرئيسي.\n")
    add("## الأرقام الرئيسية\n")
    add("| القياس | القيمة |")
    add("|---|---|")
    add(f"| الالتزامات الموثقة | **{summ['n_commits']}** |")
    add(f"| أحداث تغيير الملفات | **{fmt(summ['n_file_events'])}** |")
    add(f"| ملفات فريدة | **{fmt(summ['n_unique_files'])}** |")
    add(f"| أسطر مضافة | **+{fmt(summ['total_ins'])}** |")
    add(f"| أسطر محذوفة | **−{fmt(summ['total_del'])}** |")
    add(f"| أيام تطوير نشطة | {summ['n_days']} |")
    add(f"| الوسوم | {summ['tags_total']} |\n")
    add("## آخر الالتزامات\n")
    add("| # | الهاش | التاريخ | الرسالة | +/− | ملفات |")
    add("|---|---|---|---|---|---|")
    for c in summ["latest_commits"]:
        add(f"| {c['seq']:02d} | `{c['short']}` | {c['date']} | {truncate(c['subject'], 40)} "
            f"| +{fmt(c['ins'])}/−{fmt(c['del'])} | {c['n_files']} |")
    add("")
    add("## الأصول المرفقة\n")
    add(f"- `{repo_display}_Commits_Changes_Map.png` — الخريطة الشاملة الكبرى (كل التزام + كل ملف متأثر)")
    add(f"- `{repo_display}_Commits_Changes_Map.html` — نسخة تفاعلية من الخريطة")
    add(f"- `{repo_display}_Commits_Overview_Chart.png` — مقارنة أحجام الالتزامات")
    add(f"- `{repo_display}_Changes_Analytics_Dashboard.png` — لوحة التحليل الإحصائي")
    add("- `CHANGES.md` — التوثيق النصي الكامل\n")
    if slug:
        add("\n## الويكي\n")
        add(f"- [الصفحة الرئيسية](https://github.com/{slug}/wiki)")
        add(f"- [السجل الكامل (Changelog)](https://github.com/{slug}/wiki/Changelog)")
        add(f"- [خريطة التغييرات](https://github.com/{slug}/wiki/Changes-Map)\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dist/changes-report/commits.json")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="CHANGES.md")
    ap.add_argument("--dist", default="dist/changes-report")
    ap.add_argument("--repo", default="VoiceCall", help="display name")
    ap.add_argument("--slug", default="", help="OWNER/REPO (else auto-detected)")
    ap.add_argument("--release-tag", default="changes-report")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        commits = json.load(f)
    summ = compute_summary(args.repo_root, commits)
    summ["repo_display"] = args.repo
    summ["repo_slug"] = slug_from_repo(args.repo_root, args.slug)
    summ["release_tag"] = args.release_tag
    summ["identity"] = identity_from_pyproject(args.repo_root)
    summ["tree"] = tree_stats(args.repo_root)
    summ["head_short"] = commits[-1]["short"]
    summ["head_full"] = commits[-1]["hash"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    md = render_changes_md(commits, summ, args.repo, summ["repo_slug"],
                           args.release_tag, generated)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"written: {out} ({len(md) / 1024:.0f} KB)")

    dist = Path(args.dist)
    dist.mkdir(parents=True, exist_ok=True)
    with open(dist / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    notes = render_release_notes(summ, args.repo, summ["repo_slug"],
                                 args.release_tag, generated)
    with open(dist / "release_notes.md", "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"written: {dist / 'summary.json'} + {dist / 'release_notes.md'}")


if __name__ == "__main__":
    main()
