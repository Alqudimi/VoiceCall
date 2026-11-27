#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the two statistical charts for the changes report.

Figure 1: <Repo>_Commits_Overview_Chart.png
    Per-commit additions / deletions (asinh scale) + files-touched panel,
    covering EVERY commit on the branch.
Figure 2: <Repo>_Changes_Analytics_Dashboard.png
    Daily activity, file-event categories, cumulative growth, top files.
Arabic labels are reshaped (arabic-reshaper + python-bidi) for matplotlib.
"""
import argparse
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter

from common import AR_MONTHS, CAT_AR, truncate


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dist/changes-report/commits.json")
    ap.add_argument("--out-dir", default="dist/changes-report")
    ap.add_argument("--fonts", default=".cache/changes-report-fonts")
    ap.add_argument("--repo", default="VoiceCall", help="display name of the repo")
    return ap


def ar(text):
    """Reshape + bidi Arabic text so matplotlib renders it correctly."""
    return get_display(arabic_reshaper.reshape(text))


def main():
    args = build_argparser().parse_args()
    for name in ("NotoSansArabic.ttf", "NotoKufiArabic.ttf", "JetBrainsMono-Regular.ttf"):
        fm.fontManager.addfont(f"{args.fonts}/{name}")
    plt.rcParams["font.family"] = ["Noto Sans Arabic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    kufi = fm.FontProperties(family="Noto Kufi Arabic")

    with open(args.data, encoding="utf-8") as f:
        commits = json.load(f)
    repo = args.repo
    N = len(commits)

    # ---- palette ----
    C_INS, C_DEL, C_FILES = "#4C6EF5", "#CC6677", "#3AAFA9"
    C_GRID, C_TXT = "#CBD5E1", "#1F2937"

    # =====================================================================
    # FIGURE 1 — per-commit ins/del + files (all commits)
    # =====================================================================
    fig = plt.figure(figsize=(19, 17.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.06)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)

    ypos = list(range(N))
    axA.invert_yaxis()
    ins_vals = [c["total_ins"] for c in commits]
    del_vals = [c["total_del"] for c in commits]
    fil_vals = [c["n_files"] for c in commits]
    max_churn = max(c["total_ins"] + c["total_del"] for c in commits)

    h = 0.36
    axA.barh([y - h / 2 - 0.02 for y in ypos], ins_vals, height=h,
             color=C_INS, label=ar("إضافات (+ أسطر)"), zorder=3,
             edgecolor="white", linewidth=0.4)
    axA.barh([y + h / 2 + 0.02 for y in ypos], del_vals, height=h,
             color=C_DEL, label=ar("حذوفات (− أسطر)"), zorder=3,
             edgecolor="white", linewidth=0.4)
    axA.set_xscale("asinh")
    axA.set_xlim(0, max(ins_vals) * 1.25)

    for y, iv, dv in zip(ypos, ins_vals, del_vals):
        if iv >= 1:
            axA.text(iv * 1.15, y - h / 2 - 0.02, f"+{iv:,}", va="center", ha="left",
                     fontsize=7.3, color="#3B5BDB", fontfamily="JetBrains Mono", zorder=4)
        if dv >= 1:
            axA.text(dv * 1.15, y + h / 2 + 0.02, f"−{dv:,}", va="center", ha="left",
                     fontsize=7.3, color="#A24B57", fontfamily="JetBrains Mono", zorder=4)

    axA.set_yticks(ypos)
    axA.set_yticklabels([f"#{c['seq']:02d} {c['short']}  {truncate(c['subject'])}" for c in commits],
                        fontfamily="JetBrains Mono", fontsize=8.1, color=C_TXT)
    axA.set_xlabel(ar("حجم التغيير (عدد الأسطر) — مقياس asinh يُظهر الالتزامات الصغيرة والكبيرة معًا"),
                   fontsize=11, fontproperties=kufi, color=C_TXT, labelpad=10)
    axA.set_title(ar(f"الإضافات والحذوفات لكل التزام ({N} التزامًا)"),
                  fontsize=15, fontproperties=kufi, color=C_TXT, pad=14, loc="right")
    axA.xaxis.grid(True, color=C_GRID, alpha=0.35, linewidth=0.7, zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.tick_params(axis="x", labelsize=9)
    axA.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))

    axB.barh(ypos, fil_vals, height=0.62, color=C_FILES, zorder=3,
             edgecolor="white", linewidth=0.4)
    for y, fv in zip(ypos, fil_vals):
        axB.text(fv + max(fil_vals) * 0.02, y, str(fv), va="center", ha="left",
                 fontsize=8, color="#1D7A74", fontfamily="JetBrains Mono", zorder=4)
    axB.set_xlim(0, max(fil_vals) * 1.18)
    axB.set_xlabel(ar("عدد الملفات المتأثرة"), fontsize=11, fontproperties=kufi,
                   color=C_TXT, labelpad=10)
    axB.set_title(ar("عدد الملفات المتأثرة في كل التزام"),
                  fontsize=15, fontproperties=kufi, color=C_TXT, pad=14, loc="right")
    axB.xaxis.grid(True, color=C_GRID, alpha=0.35, linewidth=0.7, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axB.spines[s].set_visible(False)
    axB.tick_params(axis="x", labelsize=9)
    axB.tick_params(axis="y", length=0, labelleft=False)
    axB.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))

    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.44, -0.012),
               ncol=2, frameon=False, fontsize=11.5, prop=kufi)
    fig.suptitle(ar(f"خريطة التغييرات الشاملة لمستودع {repo} — كل التزامات الفرع ({N})"),
                 fontsize=19, fontproperties=kufi, color="#111827", x=0.02, ha="left")
    fig.savefig(f"{args.out_dir}/{repo}_Commits_Overview_Chart.png",
                dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"saved: {repo}_Commits_Overview_Chart.png")

    # =====================================================================
    # FIGURE 2 — analytics dashboard
    # =====================================================================
    day_totals = defaultdict(lambda: [0, 0, 0])
    for c in commits:
        d = c["date"][:10]
        day_totals[d][0] += c["total_ins"]
        day_totals[d][1] += c["total_del"]
        day_totals[d][2] += 1
    days = sorted(day_totals.keys())
    days_ar = [f"{d[8:]} {AR_MONTHS[int(d[5:7])]}" for d in days]

    cat_counter, file_counter = Counter(), Counter()
    for c in commits:
        for fl in c["files"]:
            cat_counter[fl["cat"]] += 1
            file_counter[fl["path"]] += 1

    cum_ins, cum_del, a, b = [], [], 0, 0
    for c in commits:
        a += c["total_ins"]
        b += c["total_del"]
        cum_ins.append(a)
        cum_del.append(b)
    total_events = sum(c["n_files"] for c in commits)

    fig = plt.figure(figsize=(19, 13), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.14, wspace=0.10)

    # (1) daily activity — stacked
    ax1 = fig.add_subplot(gs[0, 0])
    x = range(len(days))
    ins_d = [day_totals[d][0] for d in days]
    del_d = [day_totals[d][1] for d in days]
    cnt_d = [day_totals[d][2] for d in days]
    ax1.bar(x, ins_d, color=C_INS, label=ar("إضافات"), width=0.62, zorder=3)
    ax1.bar(x, del_d, bottom=ins_d, color=C_DEL, label=ar("حذوفات"), width=0.62, zorder=3)
    for i, (iv, dv, cv) in enumerate(zip(ins_d, del_d, cnt_d)):
        ax1.text(i, iv + dv + max(t[0] + t[1] for t in day_totals.values()) * 0.03,
                 f"{cv} " + ar("التزامًا") + f"\n+{iv:,} / −{dv:,}",
                 ha="center", va="bottom", fontsize=9, color=C_TXT, linespacing=1.4)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([ar(d) for d in days_ar], fontsize=9.5, fontproperties=kufi)
    ax1.set_ylim(0, max(i + d for i, d in zip(ins_d, del_d)) * 1.30)
    ax1.set_title(ar("حجم التغييرات اليومية (أسطر مضافة/محذوفة وعدد الالتزامات)"),
                  fontsize=13.5, fontproperties=kufi, color=C_TXT, pad=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
               fontsize=11, prop=kufi)
    ax1.grid(axis="y", color=C_GRID, alpha=0.3, zorder=0)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # (2) category distribution — horizontal
    ax2 = fig.add_subplot(gs[0, 1])
    cats_sorted = cat_counter.most_common()
    names = [ar(CAT_AR[k]) for k, _ in cats_sorted]
    vals = [v for _, v in cats_sorted]
    ax2.barh(range(len(vals)), vals, color="#3AAFA9", height=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax2.text(v + max(vals) * 0.02, i, str(v), va="center", ha="left",
                 fontsize=9.5, color=C_TXT, fontfamily="JetBrains Mono")
    ax2.set_yticks(range(len(vals)))
    ax2.set_yticklabels(names, fontsize=11, fontproperties=kufi)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(vals) * 1.15)
    ax2.set_title(ar(f"توزيع أحداث تغيير الملفات حسب الفئة ({total_events:,} حدثًا)"),
                  fontsize=13.5, fontproperties=kufi, color=C_TXT, pad=12)
    ax2.xaxis.grid(True, color=C_GRID, alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # (3) cumulative growth
    ax3 = fig.add_subplot(gs[1, 0])
    xs = [c["seq"] for c in commits]
    ax3.plot(xs, cum_ins, color=C_INS, linewidth=2.4, zorder=3,
             label=ar("تراكمي الأسطر المضافة"))
    ax3.plot(xs, cum_del, color=C_DEL, linewidth=2.0, zorder=3,
             label=ar("تراكمي الأسطر المحذوفة"))
    ax3.fill_between(xs, cum_ins, alpha=0.12, color=C_INS, zorder=2)
    ax3.text(xs[-1] + 0.4, cum_ins[-1], f"{cum_ins[-1]:,}", va="center",
             fontsize=10, color="#3B5BDB", fontfamily="JetBrains Mono")
    ax3.text(xs[-1] + 0.4, cum_del[-1], f"{cum_del[-1]:,}", va="center",
             fontsize=10, color="#A24B57", fontfamily="JetBrains Mono")
    first = commits[0]
    ax3.annotate(ar(f"الالتزام الأول «{truncate(first['subject'], 26)}» +{first['total_ins']:,}"),
                 xy=(1, first["total_ins"]),
                 xytext=(max(4, N // 6), first["total_ins"] * 1.9),
                 fontsize=9.5, fontproperties=kufi, color=C_TXT,
                 arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.1))
    ax3.annotate(ar(f"آخر التزام «{truncate(commits[-1]['subject'], 26)}»"),
                 xy=(N, cum_ins[-1]), xytext=(N * 0.55, cum_ins[-1] * 0.86),
                 fontsize=9.5, fontproperties=kufi, color=C_TXT,
                 arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.1))
    ax3.set_xlim(0.5, N + N * 0.1)
    ax3.set_xlabel(ar(f"رقم الالتزام بالتسلسل الزمني (#01 حتى #{N:02d})"),
                   fontsize=10.5, fontproperties=kufi)
    ax3.set_ylabel(ar("الأسطر التراكمية"), fontsize=10.5, fontproperties=kufi)
    ax3.set_title(ar("منحنى النمو التراكمي لقاعدة الشيفرة عبر الالتزامات"),
                  fontsize=13.5, fontproperties=kufi, color=C_TXT, pad=12)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    ax3.legend(loc="upper left", frameon=False, fontsize=10.5, prop=kufi)
    ax3.grid(color=C_GRID, alpha=0.3, zorder=0)
    ax3.set_axisbelow(True)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    # (4) top modified files
    ax4 = fig.add_subplot(gs[1, 1])
    top = file_counter.most_common(12)
    names_f = [t for t, _ in top][::-1]
    vals_f = [v for _, v in top][::-1]
    ax4.barh(range(len(vals_f)), vals_f, color=C_FILES, height=0.62, zorder=3)
    for i, v in enumerate(vals_f):
        ax4.text(v + max(vals_f) * 0.02, i, f"{v}×", va="center", ha="left",
                 fontsize=9.5, color="#1D7A74", fontfamily="JetBrains Mono")
    ax4.set_yticks(range(len(vals_f)))
    ax4.set_yticklabels([truncate(n, 42) for n in names_f],
                        fontfamily="JetBrains Mono", fontsize=8.6, color=C_TXT)
    ax4.set_xlim(0, max(vals_f) * 1.16)
    ax4.set_title(ar("أكثر الملفات تعديلًا (عدد مرات التغيير عبر الالتزامات)"),
                  fontsize=13.5, fontproperties=kufi, color=C_TXT, pad=12)
    ax4.xaxis.grid(True, color=C_GRID, alpha=0.3, zorder=0)
    ax4.set_axisbelow(True)
    for s in ("top", "right"):
        ax4.spines[s].set_visible(False)

    fig.suptitle(ar(f"لوحة التحليل الإحصائي لتغييرات {repo}"),
                 fontsize=18, fontproperties=kufi, color="#111827", x=0.02, ha="left")
    fig.savefig(f"{args.out_dir}/{repo}_Changes_Analytics_Dashboard.png",
                dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"saved: {repo}_Changes_Analytics_Dashboard.png")


if __name__ == "__main__":
    main()
