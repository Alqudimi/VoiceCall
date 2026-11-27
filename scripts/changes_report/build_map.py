#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the comprehensive RTL-Arabic HTML changes map (self-contained file,
fonts embedded as base64) covering EVERY commit and EVERY file-change event
with per-file +additions/−deletions counts."""
import argparse
import base64
import html as H
import json
from datetime import datetime
from pathlib import Path

from common import AR_MONTHS, CAT_AR, CAT_COLOR, TYPE_INFO, ar_weekday, fmt


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dist/changes-report/commits.json")
    ap.add_argument("--out", default="dist/changes-report/map.html")
    ap.add_argument("--fonts", default=".cache/changes-report-fonts")
    ap.add_argument("--repo", default="VoiceCall")
    return ap


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def type_style(tcls):
    _label, bg, brd = TYPE_INFO[tcls]
    return bg, brd


def main():
    args = build_argparser().parse_args()
    repo = args.repo

    f_sans = b64(f"{args.fonts}/NotoSansArabic.ttf")
    f_kufi = b64(f"{args.fonts}/NotoKufiArabic.ttf")
    f_mono = b64(f"{args.fonts}/JetBrainsMono-Regular.ttf")

    with open(args.data, encoding="utf-8") as f:
        commits = json.load(f)

    total_ins = sum(c["total_ins"] for c in commits)
    total_del = sum(c["total_del"] for c in commits)
    total_files_events = sum(c["n_files"] for c in commits)
    uniq = set()
    for c in commits:
        for fl in c["files"]:
            uniq.add(fl["path"])
    days = {}
    for c in commits:
        days.setdefault(c["date"][:10], []).append(c)
    n_days = len(days)
    first_d = commits[0]["date"][:10]
    last_d = commits[-1]["date"][:10]

    # ---------------- mini overview strip ----------------
    max_churn = max(c["total_ins"] + c["total_del"] for c in commits)
    strip_bars = []
    for c in commits:
        churn = c["total_ins"] + c["total_del"]
        hgt = 12 + 76 * (churn / max_churn)
        bg, brd = type_style(c["type_class"])
        tip = f"#{c['seq']:02d} {c['short']} · +{fmt(c['total_ins'])}/−{fmt(c['total_del'])}"
        strip_bars.append(
            f'<div class="mb" style="height:{hgt:.0f}px;background:{bg};'
            f'border-top:2px solid {brd}" title="{H.escape(tip)}"></div>')
    strip_html = "\n".join(strip_bars)

    from collections import Counter
    tcount = Counter(c["type_class"] for c in commits)
    legend_items = []
    for tcls in ("t-init", "t-feat", "t-fix", "t-docs", "t-rel", "t-chore", "t-dev"):
        if tcount.get(tcls, 0) == 0:
            continue
        label = TYPE_INFO[tcls][0]
        bg, brd = type_style(tcls)
        legend_items.append(
            f'<span class="lg"><i style="background:{bg};border-top:2px solid {brd}"></i> '
            f"{label} ({tcount[tcls]})</span>")
    legend_html = "\n".join(legend_items)

    # ---------------- day sections + commit cards ----------------
    sections = []
    for d in sorted(days.keys()):
        dcommits = days[d]
        dins = sum(c["total_ins"] for c in dcommits)
        ddel = sum(c["total_del"] for c in dcommits)
        dfiles = sum(c["n_files"] for c in dcommits)
        dt = datetime.strptime(d, "%Y-%m-%d")
        day_name = ar_weekday(d)
        month = AR_MONTHS[dt.month]

        cards = []
        for c in dcommits:
            churn = c["total_ins"] + c["total_del"]
            wfrac = max(churn / max_churn, 0.012) * 100
            ins_frac = (c["total_ins"] / churn * 100) if churn else 0
            del_frac = 100 - ins_frac
            bg, _brd = type_style(c["type_class"])

            cat_chips = []
            for cat in ("source", "tests", "docs", "ci", "scripts", "configs",
                        "notebooks", "reports", "root"):
                n = c["cats"].get(cat)
                if n:
                    cat_chips.append(
                        f'<span class="cchip"><i style="background:{CAT_COLOR[cat]}"></i>'
                        f"{CAT_AR[cat]} <b>{n}</b></span>")
            cat_chips_html = "\n".join(cat_chips)

            fchips = []
            for fl in c["files"]:
                path = H.escape(fl["path"])
                if fl["binary"]:
                    stats = '<b class="bin" dir="rtl">ثنائي</b>'
                elif fl["ins"] == 0 and fl["del"] == 0:
                    stats = '<b class="fz">0/0</b>'
                else:
                    stats = (f'<b class="fi">{fmt(fl["ins"])}</b>'
                             f'<b class="fd">{fmt(fl["del"])}</b>')
                fchips.append(
                    f'<div class="fchip" dir="ltr" title="{path}">'
                    f'<i class="fdot" style="background:{CAT_COLOR[fl["cat"]]}"></i>'
                    f'<span class="fpath">{path}</span>'
                    f'<span class="fstats">{stats}</span></div>')
            fchips_html = "\n".join(fchips)

            cards.append(f'''
<article class="commit">
  <header class="chead">
    <span class="seq">#{c["seq"]:02d}</span>
    <code class="hash" dir="ltr">{c["short"]}</code>
    <span class="when">{c["date"][11:16]}</span>
    <span class="type" style="background:{bg}">{c["type_label"]}</span>
    <code class="subject" dir="ltr">{H.escape(c["subject"])}</code>
  </header>
  <div class="chips">
    <span class="chip"><b>{c["n_files"]}</b> ملفًا متأثرًا</span>
    <span class="chip ci"><b dir="ltr">+{fmt(c["total_ins"])}</b> سطرًا مضافًا</span>
    <span class="chip cd"><b dir="ltr">−{fmt(c["total_del"])}</b> سطرًا محذوفًا</span>
    <span class="chip cats">{cat_chips_html}</span>
  </div>
  <div class="mag" title="الحجم النسبي للتغيير">
    <div class="magtrack"><div class="magbar" style="width:{wfrac:.2f}%">
      <span class="mi" style="width:{ins_frac:.2f}%"></span><span class="md" style="width:{del_frac:.2f}%"></span>
    </div></div>
  </div>
  <div class="flabel">الملفات المتأثرة ({c["n_files"]}):</div>
  <div class="fgrid">{fchips_html}</div>
</article>''')

        sections.append(f'''
<section class="day">
  <div class="dayhead">
    <span class="dnum">{dt.day}</span>
    <span class="dmeta">{day_name} {month} {dt.year}</span>
    <span class="dtot">{len(dcommits)} التزامًا · {dfiles} حدث تغيير ملف · <b class="pi" dir="ltr">+{fmt(dins)} / −{fmt(ddel)}</b></span>
  </div>
  <div class="daybody">{" ".join(cards)}</div>
</section>''')

    day_sections = "".join(sections)

    css = """
@font-face { font-family:'Noto Sans Arabic'; src:url(data:font/ttf;base64,__SANS__) format('truetype'); font-weight:100 900; }
@font-face { font-family:'Noto Kufi Arabic'; src:url(data:font/ttf;base64,__KUFI__) format('truetype'); font-weight:100 900; }
@font-face { font-family:'JetBrains Mono'; src:url(data:font/ttf;base64,__MONO__) format('truetype'); }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#F1F5F9; font-family:'Noto Sans Arabic'; color:#1F2937;
       width:1680px; margin:0 auto; padding:0 24px 40px; font-size:13px; line-height:1.55; }
h1,h2,h3,.kufi { font-family:'Noto Kufi Arabic'; }
.hero { background:linear-gradient(135deg,#1E293B 0%,#243447 60%,#2C3E56 100%);
        border-radius:14px; color:#F1F5F9; padding:26px 30px 22px; margin:18px 0 14px; }
.hero h1 { font-size:26px; color:#fff; margin-bottom:6px; }
.hero p.sub { color:#CBD5E1; font-size:13.5px; }
.hero .repo { font-family:'JetBrains Mono'; direction:ltr; display:inline-block;
              color:#93C5FD; font-size:12px; }
.statrow { display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }
.stat { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14);
        border-radius:10px; padding:8px 14px; min-width:150px; }
.stat .v { font-family:'JetBrains Mono','Noto Sans Arabic',sans-serif; font-size:17px;
           color:#F9FAFB; direction:ltr; text-align:right; }
.stat .l { font-size:11px; color:#94A3B8; margin-top:2px; }
.overview { background:#fff; border:1px solid #E2E8F0; border-radius:12px;
            padding:16px 20px; margin-bottom:14px; }
.overview h2 { font-size:15px; color:#0F172A; margin-bottom:4px; }
.overview p.note { font-size:11.5px; color:#64748B; margin-bottom:10px; }
.strip { direction:ltr; display:flex; align-items:flex-end; gap:3px; height:96px;
         padding:4px 6px; background:#F8FAFC; border:1px solid #E2E8F0;
         border-radius:8px; overflow:hidden; }
.mb { flex:1 1 0; border-radius:3px 3px 0 0; min-width:8px; }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.lg { display:inline-flex; align-items:center; gap:6px; background:#fff;
      border:1px solid #E2E8F0; border-radius:20px; padding:3.5px 12px; font-size:11.5px; }
.lg i { width:10px; height:10px; border-radius:3px; display:inline-block; }
.howto { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px;
         padding:14px 18px; margin-bottom:16px; font-size:12.5px; }
.howto b { color:#1D4ED8; }
.howto ol { padding-right:18px; margin-top:6px; }
.howto li { margin-bottom:3px; }
.day { margin-bottom:20px; }
.dayhead { display:flex; align-items:center; gap:12px; background:#243447;
           color:#E2E8F0; border-radius:10px; padding:9px 16px; }
.dnum { font-family:'JetBrains Mono'; font-size:20px; color:#93C5FD; }
.dmeta { font-size:13px; font-weight:600; }
.dtot { margin-right:auto; font-size:12px; color:#CBD5E1; }
.dtot .pi { color:#6EE7B7; font-weight:700; }
.daybody { padding:12px 0 0; }
.commit { background:#fff; border:1px solid #E2E8F0; border-radius:11px;
          padding:14px 18px 16px; margin-bottom:14px;
          box-shadow:0 1px 2px rgba(15,23,42,.04); }
.chead { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.seq { font-family:'JetBrains Mono'; font-weight:700; font-size:15px; color:#0F172A;
       background:#F1F5F9; border:1px solid #E2E8F0; border-radius:8px; padding:2px 9px; }
.hash { font-family:'JetBrains Mono'; font-size:12.5px; color:#475569;
        background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:2px 8px; }
.when { font-family:'JetBrains Mono'; font-size:11.5px; color:#94A3B8; }
.type { font-size:11px; font-weight:700; border-radius:20px; padding:2.5px 12px; }
.subject { font-family:'JetBrains Mono'; font-size:12px; color:#1F2937;
           flex:1 1 40%; min-width:260px; }
.chips { display:flex; flex-wrap:wrap; gap:7px; align-items:center; margin-top:10px; }
.chip { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:20px;
        padding:3px 12px; font-size:11.5px; }
.chip b { font-family:'JetBrains Mono'; }
.chip.ci { background:#EFF6FF; border-color:#BFDBFE; color:#1E40AF; }
.chip.cd { background:#FFF1F2; border-color:#FECDD3; color:#9F1239; }
.chip.cats { display:inline-flex; flex-wrap:wrap; gap:6px; background:none;
             border:none; padding:0; }
.cchip { display:inline-flex; align-items:center; gap:5px; background:#F8FAFC;
         border:1px solid #E2E8F0; border-radius:14px; padding:1.5px 9px;
         font-size:10.5px; color:#475569; }
.cchip i { width:8px; height:8px; border-radius:2px; }
.cchip b { font-family:'JetBrains Mono'; color:#1F2937; }
.mag { margin-top:10px; }
.magtrack { background:#F1F5F9; border-radius:5px; height:10px; overflow:hidden;
            border:1px solid #E2E8F0; }
.magbar { height:100%; display:flex; overflow:hidden; border-radius:4px; }
.mi { background:#4C6EF5; height:100%; display:inline-block; }
.md { background:#E4A3AB; height:100%; display:inline-block; }
.flabel { margin-top:11px; font-size:11.5px; font-weight:700; color:#334155; }
.fgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr));
         gap:5px; margin-top:7px; }
.fchip { display:flex; align-items:center; gap:7px; background:#FAFBFD;
         border:1px solid #E6EAF1; border-radius:7px; padding:3.5px 9px;
         font-size:11px; min-width:0; }
.fdot { width:9px; height:9px; border-radius:3px; flex:0 0 auto; }
.fpath { font-family:'JetBrains Mono'; font-size:10.8px; color:#334155;
         white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1 1 auto; }
.fstats { flex:0 0 auto; font-family:'JetBrains Mono'; font-size:10.3px;
          text-align:right; }
.fi { color:#047857; font-weight:700; margin-left:7px; }
.fd { color:#BE123C; font-weight:700; margin-left:4px; }
.fz { color:#94A3B8; }
.bin { background:#FEF3C7; color:#92400E; border-radius:8px; padding:0 7px;
       font-family:'Noto Sans Arabic'; font-size:10px; font-weight:700; }
footer.foot { text-align:center; color:#64748B; font-size:11.5px;
              padding:18px 0 4px; }
"""
    css = (css.replace("__SANS__", f_sans).replace("__KUFI__", f_kufi)
               .replace("__MONO__", f_mono))

    page = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>خريطة التغييرات الشاملة — {H.escape(repo)}</title>
<style>{css}</style>
</head>
<body>
<div class="hero">
  <h1>خريطة التغييرات الشاملة لمستودع {H.escape(repo)}</h1>
  <p class="sub">رسم بياني تفصيلي يوثّق كل التزامات الفرع الحالي — <b dir="ltr">{len(commits)}</b>
  التزامًا — مع كل ملف تعرّض للتغيير في كل التزام وأعداد الأسطر المضافة والمحذوفة لكل ملف —
  تغطية كاملة 100%.</p>
  <div class="statrow">
    <div class="stat"><div class="v">{len(commits)}</div><div class="l">التزامًا موثقًا</div></div>
    <div class="stat"><div class="v">{fmt(total_files_events)}</div><div class="l">حدث تغيير ملف</div></div>
    <div class="stat"><div class="v">{len(uniq)}</div><div class="l">ملفًا فريدًا</div></div>
    <div class="stat"><div class="v">+{fmt(total_ins)}</div><div class="l">سطرًا مضافًا</div></div>
    <div class="stat"><div class="v">−{fmt(total_del)}</div><div class="l">سطرًا محذوفًا</div></div>
    <div class="stat"><div class="v">{n_days}</div><div class="l">أيام تطوير نشطة</div></div>
    <div class="stat"><div class="v" dir="ltr">{first_d} – {last_d}</div><div class="l">فترة التطوير</div></div>
  </div>
</div>

<div class="overview">
  <h2>نظرة سريعة: قصة تطور المشروع ({len(commits)} التزامًا)</h2>
  <p>كل عمود = التزام واحد بالترتيب الزمني (#01 يسارًا حتى #{len(commits):02d} يمينًا)، ارتفاعه يعكس حجم التغيير، ولونه يحدد نوعه.</p>
  <div class="strip">{strip_html}</div>
  <div class="legend">{legend_html}
  </div>
</div>

<div class="howto">
  <b>كيف تقرأ هذه الخريطة؟</b>
  <ol>
    <li>كل بطاقة أدناه تمثل <b>التزامًا واحدًا</b> بالترتيب الزمني من #01 (الأقدم) حتى #{len(commits):02d} (الأحدث)، مُجمّعة حسب يوم التطوير.</li>
    <li>في رأس كل بطاقة: رقم الالتزام، بصمة الالتزام (hash)، الوقت، نوع التغيير، ورسالة الالتزام الأصلية بالإنجليزية.</li>
    <li>أسفل الرأس: عدد الملفات المتأثرة، الأسطر المضافة (+) والمحذوفة (−)، وتوزيع الملفات على الفئات.</li>
    <li>الشريط المزدوج الأزرق/الوردي يمثل الحجم النسبي للتغيير مقارنة بأكبر التزام في السجل (+{fmt(max_churn)} سطرًا).</li>
    <li>قائمة <b>«الملفات المتأثرة»</b> تعرض كل ملف تعرّض للتغيير في هذا التزام تحديدًا، مع نقطة ملوّنة تدل على فئة الملف، وأرقام <b>+إضافات/−حذوفات</b> لكل ملف. كلمة «ثنائي» تعني ملفًا غير نصي.</li>
  </ol>
</div>

{day_sections}

<footer class="foot">
  وُلّدت هذه الخريطة آليًا من سجل Git — إجمالي أحداث تغيير الملفات: {fmt(total_files_events)} —
  تُحدَّث تلقائيًا عند كل دفع جديد بواسطة سير عمل «Changes Report».
</footer>
</body>
</html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"HTML map written: {out} ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
