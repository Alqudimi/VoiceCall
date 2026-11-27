#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update the GitHub Wiki pages of the repository from the changes report.

Writes (and rewrites on every run):
  Home.md        — overview, headline stats, latest commits, images
  Changelog.md   — the full generated CHANGES.md document
  Changes-Map.md — the story + the three charts embedded from the release
  _Sidebar.md    — navigation
  _Footer.md     — auto-generated footer (timestamp + trigger sha)

Wiki git repo: https://github.com/<slug>.wiki.git
Authentication: secrets.WIKI_TOKEN (a classic PAT with repo scope) if set,
otherwise the workflow GITHUB_TOKEN is attempted. On failure the script
prints an actionable ::warning:: and exits 0 so the pipeline stays green.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import fmt, truncate


def sh(cmd, cwd=None, check=True):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=False)
    if check and res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{res.stderr.strip()}")
    return res.returncode, res.stdout, res.stderr


def warn(msg):
    print(f"::warning::{msg}")


def asset_url(slug, tag, filename):
    return f"https://github.com/{slug}/releases/download/{tag}/{filename}"


def build_pages(summ, changes_md, repo_display, slug, tag, generated, trigger_sha):
    pages = {}

    # ---------- _Sidebar.md ----------
    pages["_Sidebar.md"] = "\n".join([
        f"**{repo_display} — تقرير التغييرات**",
        "",
        "- [[الرئيسية|Home]]",
        "- [[السجل الكامل|Changelog]]",
        "- [[خريطة التغييرات|Changes-Map]]",
        "",
        f"[إصدار التقرير الكامل (الأصول الخمسة)](https://github.com/{slug}/releases/tag/{tag})",
        "",
    ])

    # ---------- _Footer.md ----------
    pages["_Footer.md"] = (
        f"آخر تحديث تلقائي: **{generated} UTC** — الرأس: `{trigger_sha[:7]}` — "
        "يُولَّد بواسطة سير عمل «Changes Report».\n"
    )

    # ---------- Home.md ----------
    t = summ["types"]
    types_line = "، ".join(f"**{k}**: {v}" for k, v in
                           sorted(t.items(), key=lambda kv: -kv[1]))
    home = []
    home.append(f"# {repo_display} — تقرير التغييرات الشامل (تلقائي)\n")
    home.append("هذه الصفحة **تُحدَّث تلقائيًا عند كل دفع جديد إلى الفرع الرئيسي** "
                 "بواسطة سير عمل GitHub Actions «Changes Report»، وتعرض ملخص سجل "
                 "التغييرات الكامل المقاس من بيانات Git الخام.\n")
    home.append("## الأرقام الرئيسية\n")
    home.append("| القياس | القيمة |")
    home.append("|---|---|")
    home.append(f"| إجمالي الالتزامات | **{summ['n_commits']}** |")
    home.append(f"| أحداث تغيير الملفات | **{fmt(summ['n_file_events'])}** |")
    home.append(f"| ملفات فريدة | **{fmt(summ['n_unique_files'])}** |")
    home.append(f"| أسطر مضافة | **+{fmt(summ['total_ins'])}** |")
    home.append(f"| أسطر محذوفة | **−{fmt(summ['total_del'])}** |")
    home.append(f"| أيام تطوير نشطة | {summ['n_days']} |")
    home.append(f"| عمر السجل | {summ['duration_text']} |")
    home.append(f"| الوسوم | {summ['tags_total']} |\n")
    home.append(f"توزيع الالتزامات حسب النوع: {types_line}.\n")
    home.append("## آخر الالتزامات (5)\n")
    home.append("| # | الهاش | التاريخ | الرسالة | +/− | ملفات |")
    home.append("|---|---|---|---|---|---|")
    for c in summ["latest_commits"]:
        home.append(f"| {c['seq']:02d} | [`{c['short']}`](https://github.com/{slug}/commit/{c['short']}) "
                     f"| {c['date']} | {truncate(c['subject'], 44)} "
                     f"| +{fmt(c['ins'])}/−{fmt(c['del'])} | {c['n_files']} |")
    home.append("")
    home.append("## نظرة سريعة على التغييرات\n")
    home.append(f'<img width="880" src="{asset_url(slug, tag, repo_display + "_Changes_Analytics_Dashboard.png")}" '
                f'alt="لوحة التحليل الإحصائي">\n')
    home.append("## التنقل\n")
    home.append("- [[Changelog]] — السجل التفصيلي الكامل لكل التزام وكل ملف")
    home.append("- [[Changes-Map]] — خريطة التغييرات بالرسوم البيانية")
    home.append(f"- [إصدار التقرير (تنزيل كل الأصول)](https://github.com/{slug}/releases/tag/{tag})\n")
    pages["Home.md"] = "\n".join(home)

    # ---------- Changelog.md ----------
    pages["Changelog.md"] = (
        "> هذه الصفحة نسخة ويكي من `CHANGES.md` المولَّد آليًا في المستودع، "
        f"وتُحدَّث تلقائيًا عند كل دفع جديد. آخر تحديث: {generated} UTC.\n\n"
        + changes_md
    )

    # ---------- Changes-Map.md ----------
    cmap = []
    cmap.append("# خريطة التغييرات — كل التزامات وكل الملفات المتأثرة\n")
    cmap.append("جميع الرسوم أدناه مولَّدة آليًا من بيانات Git وتُحدَّث مع كل دفع جديد. "
                "النسخ كاملة الدقة متاحة للتنزيل من "
                f"[إصدار التقرير](https://github.com/{slug}/releases/tag/{tag}).\n")
    cmap.append("## النشاط اليومي والفئات والنمو وأكثر الملفات تعديلًا\n")
    cmap.append(f'<img width="900" src="{asset_url(slug, tag, repo_display + "_Changes_Analytics_Dashboard.png")}" '
                'alt="لوحة التحليل الإحصائي">\n')
    cmap.append("## مقارنة أحجام الالتزامات (الإضافات والحذوفات وعدد الملفات لكل التزام)\n")
    cmap.append(f'<img width="900" src="{asset_url(slug, tag, repo_display + "_Commits_Overview_Chart.png")}" '
                'alt="مقارنة أحجام الالتزامات">\n')
    cmap.append("## الخريطة الشاملة الكبرى (بطاقة لكل التزام + كل ملف متأثر فيه)\n")
    cmap.append(f'<img width="900" src="{asset_url(slug, tag, repo_display + "_Commits_Changes_Map.png")}" '
                'alt="الخريطة الشاملة">\n')
    cmap.append(f"[تنزيل الخريطة كاملة الدقة (PNG)]({asset_url(slug, tag, repo_display + '_Commits_Changes_Map.png')}) — "
                f"[النسخة التفاعلية (HTML)]({asset_url(slug, tag, repo_display + '_Commits_Changes_Map.html')})\n")
    cmap.append("## النشاط اليومي بالأرقام\n")
    cmap.append("| اليوم | الالتزامات | أحداث الملفات | إضافات | حذوفات |")
    cmap.append("|---|---|---|---|---|")
    for d in summ["days"]:
        cmap.append(f"| {d['weekday_ar']} {d['date_ar']} | {d['n_commits']} | {fmt(d['n_files'])} "
                    f"| +{fmt(d['ins'])} | −{fmt(d['del'])} |")
    cmap.append("")
    pages["Changes-Map.md"] = "\n".join(cmap)
    return pages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="dist/changes-report/summary.json")
    ap.add_argument("--changes", default="CHANGES.md")
    ap.add_argument("--slug", default=os.environ.get("GITHUB_REPOSITORY", ""))
    ap.add_argument("--repo", default="", help="display name; default: part after /")
    ap.add_argument("--release-tag", default="changes-report")
    ap.add_argument("--work", default="dist/wiki-work")
    ap.add_argument("--trigger-sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    args = ap.parse_args()

    if not args.slug:
        sys.exit("changes-report: no repository slug provided (use --slug OWNER/REPO)")
    repo_display = args.repo or args.slug.split("/")[-1]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    with open(args.summary, encoding="utf-8") as f:
        summ = json.load(f)
    changes_md = Path(args.changes).read_text(encoding="utf-8")

    pages = build_pages(summ, changes_md, repo_display, args.slug,
                        args.release_tag, generated, args.trigger_sha)

    token = os.environ.get("WIKI_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if not token:
        warn("no WIKI_TOKEN/GH_TOKEN provided - skipping wiki update")
        return
    remote = os.environ.get("WIKI_REMOTE_URL", "").strip() or \
        f"https://x-access-token:{token}@github.com/{args.slug}.wiki.git"

    work = Path(args.work)
    if work.exists():
        subprocess.run(["rm", "-rf", str(work)], check=True)
    work.parent.mkdir(parents=True, exist_ok=True)

    # ---- clone (empty wiki is fine) ----
    rc, _out, err = sh(["git", "clone", remote, str(work)], check=False)
    if rc != 0:
        warn(f"could not clone the wiki repository ({args.slug}.wiki.git). "
             "Make sure the Wiki is enabled in repository Settings -> Features "
             "and the token has write access. Details: " + err.strip()[:300])
        return

    # ---- write pages ----
    for name, content in pages.items():
        (work / name).write_text(content, encoding="utf-8")

    # ---- commit + push ----
    sh(["git", "-C", str(work), "config", "user.name", "github-actions[bot]"])
    sh(["git", "-C", str(work), "config", "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com"])
    sh(["git", "-C", str(work), "add", "-A"])
    rc, _out, _err = sh(["git", "-C", str(work), "commit", "-m",
                         f"wiki: auto-update changes report @ {args.trigger_sha[:7]}"],
                        check=False)
    if rc != 0:
        print("wiki: no changes to commit (already up to date)")
        return
    rc, _out, err = sh(["git", "-C", str(work), "push", "origin", "HEAD:master"], check=False)
    if rc != 0:
        warn("wiki push failed. If the default GITHUB_TOKEN was used, create a classic "
             "PAT with repo scope and add it as the WIKI_TOKEN secret, then re-run the "
             "workflow. Details: " + err.strip()[:300])
        return
    print("wiki updated successfully")


if __name__ == "__main__":
    main()
