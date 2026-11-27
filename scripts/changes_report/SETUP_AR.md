# Changes Report — الأتمتة الكاملة لتوثيق التغييرات في مستودع VoiceCall

منظومة GitHub Actions تؤتمت **كل شيء** تمّ في التحليل السابق، وتُعيد توليده من جديد **عند كل التزام يُدفع إلى الفرع `main`**:

1. تحليل كامل لسجل Git (كل الالتزامات + كل الملفات المتأثرة في كل التزام + الإضافات/الحذوفات لكل ملف).
2. توليد الرسوم البيانية الثلاثة (خريطة التغييرات الشاملة الكبرى + رسم مقارنة الأحجام + لوحة التحليل الإحصائي) بمحتوى عربي كامل.
3. توليد ملف التوثيق `CHANGES.md` الشامل تلقائيًا ودفعه إلى المستودع نفسه.
4. **تحديث إصدار GitHub متجدد** (اسم الوسم: `changes-report`) يحمل كل الأصول الخمسة ويُحدَّث مع كل دفعة.
5. **تحديث الـ Wiki** تلقائيًا بصفحات: الرئيسية (Home) + السجل الكامل (Changelog) + خريطة التغييرات (Changes-Map) + شريط تنقل وشريط سفلي.

---

## 1. التركيب (خطوتان فقط)

### الخطوة 1: انسخ الملفات إلى جذر مستودعك

انقل محتويات هذا الحزمة كما هي داخل المستودع (تندمج مع المجلدات الموجودة دون الكتابة فوق أي شيء):

```
.github/workflows/changes-report.yml   <- سير العمل الجديد
scripts/changes_report/                <- محرّك التحليل والتوليد (9 ملفات)
SETUP_AR.md                            <- هذا الدليل (اختياري)
```

أوامر مثال (من داخل نسخة محلية من المستودع بعد فك الضغط بجانبها):

```bash
cp -r VoiceCall-Automation/.github/  /path/to/VoiceCallRepo/
cp -r VoiceCall-Automation/scripts/   /path/to/VoiceCallRepo/
git add .github/workflows/changes-report.yml scripts/changes_report
git commit -m "ci: add automated changes-report pipeline (charts + CHANGES.md + release + wiki)"
git push origin main
```

بمجرد الدفع، يبدأ سير العمل «Changes Report» تلقائيًا على هذا الالتزام نفسه.

### الخطوة 2: فعّل الـ Wiki

من إعدادات المستودع على GitHub:
**Settings → General → Features → ✅ Wikis**

هذا كل شيء — عند أول تشغيل ستُنشأ صفحات الويكي تلقائيًا حتى لو كان الويكي فارغًا تمامًا.

---

## 2. (اختياري لكن مُوصى به) توكن موثوق للـ Wiki

دفع التحديثات إلى الويكي عبر التوكن الافتراضي `GITHUB_TOKEN` قد يُرفض في بعض المستودعات (الويكي مستودع منفصل تقنيًا). للضمان الكامل:

1. أنشئ **Classic PAT** من: GitHub → Settings → Developer settings → Personal access tokens (classic) → Generate new token، وفعّل صلاحية **repo** كاملة.
2. أضفه كسرّ في مستودع VoiceCall باسم `WIKI_TOKEN`:
   **Repository → Settings → Secrets and variables → Actions → New repository secret**
3. لا يتغير شيء في السير — هو يقرأ `WIKI_TOKEN` تلقائيًا إن وُجد، وإلا جرّب `GITHUB_TOKEN` وأصدر تنبيهًا واضحًا عند الفشل (مع بقية النتائج تُنشر طبيعيًا).

---

## 3. ماذا يحدث بالضبط عند كل push إلى main؟

| المرحلة | ما يُنفَّذ | المخرج |
|---|---|---|
| 1 | `parse_commits.py` | `commits.json`: كل الالتزامات + كل ملف متأثر بأرقامه |
| 2 | `make_charts.py` | رسم مقارنة الأحجام + لوحة التحليل الإحصائي (PNG) |
| 3 | `build_map.py` + `shoot_map.py` | خريطة HTML تفاعلية + صورتها PNG الشاملة |
| 4 | `build_changes_md.py` | `CHANGES.md` الشامل + `summary.json` + ملاحظات الإصدار |
| 5 | commit إلى main | `CHANGES.md` المُحدَّث يدخل المستودع نفسه |
| 6 | `publish_release.sh` | إصدار `changes-report` يُنشأ من جديد على الالتزام الحالي بأصوله الخمسة |
| 7 | `update_wiki.py` | صفحات الويكي: Home + Changelog + Changes-Map + Sidebar + Footer |

**روابط التحميل الثابتة** (تُستخدم داخل الويكي وتبقى صالحة دائمًا):

```
https://github.com/Alqudimi/VoiceCall/releases/download/changes-report/VoiceCall_Commits_Changes_Map.png
https://github.com/Alqudimi/VoiceCall/releases/download/changes-report/VoiceCall_Commits_Changes_Map.html
https://github.com/Alqudimi/VoiceCall/releases/download/changes-report/VoiceCall_Commits_Overview_Chart.png
https://github.com/Alqudimi/VoiceCall/releases/download/changes-report/VoiceCall_Changes_Analytics_Dashboard.png
https://github.com/Alqudimi/VoiceCall/releases/download/changes-report/CHANGES.md
```

ملاحظات تصميمية مهمة:

- **لا حلقة تكرار لا نهائية:** الدفع بواسطة `GITHUB_TOKEN` لا يُطلق workflows مرة أخرى (سلوك GitHub الرسمي) — إضافة إلى وسم `[skip ci]` احتياطًا. سير عملك الخاص `auto-release.yml` لن يُطلقه الالتزام التوثيقي للبوت.
- **لا تعارض مع `auto-release.yml` الخاص بك:** سير عمله ينشئ وسوم `v1.0.0-...` على كل push؛ إصدارنا يستخدم وسمًا منفصلًا ثابت الاسم `changes-report` (يُحذف ويُعاد إنشاؤه في كل مرة على الالتزام الأحدث) — لذا لا يتقاطعان أبدًا، ولا يسرق إصدار التقرير شارة "Latest" الحقيقية (`--latest=false`).
- **التقارير في الويكي/الإصدار فقط دون تثقيل المستودع:** الصور (بضعة ميغابايتات تتغير كل مرة) تُخزَّن في الإصدار والويكي، بينما يدخل المستند النصي `CHANGES.md` وحده إلى git — فلا يتضخم تاريخ المستودع.
- **التشغيل اليدوي:** من تبويب Actions ← Changes Report ← Run workflow.

---

## 4. التشغيل محليًا (بدون GitHub)

```bash
pip install -r scripts/changes_report/requirements.txt
playwright install chromium

mkdir -p .cache/changes-report-fonts
# حمّل الخطوط الثلاثة (NotoSansArabic / NotoKufiArabic / JetBrainsMono) من صفوف google/fonts وJetBrains

FONTS_DIR=.cache/changes-report-fonts \
bash scripts/changes_report/run_pipeline.sh
```

المتغيرات البيئية الاختيارية: `OUT_DIR`، `FONTS_DIR`، `REPO_NAME`، `RELEASE_TAG`.

---

## 5. تفاصيل المخرجات

### الويكي (5 صفحات تُحدَّث كل مرة)

| الصفحة | المحتوى |
|---|---|
| `Home` | الأرقام الرئيسية، آخر 5 التزامات، لوحة التحليل مدرجة، روابط التنقل |
| `Changelog` | نسخة كاملة من `CHANGES.md` (كل التزام ببطاقته وكل ملفاته) |
| `Changes-Map` | الرسوم الثلاثة مدرجة من روابط الإصدار + جدول النشاط اليومي |
| `_Sidebar` | شريط تنقل دائم |
| `_Footer` | ختم "آخر تحديث تلقائي: ... — الرأس: ..." |

### الإصدار المتجدد (`changes-report`)

5 أصول: الخريطة PNG (كاملة الدقة 2100×20964 حاليًا) + الخريطة HTML التفاعلية + رسم المقارنة + لوحة التحليل + `CHANGES.md` — مع جسم إصدار عربي جاهز يتضمن الأرقام الرئيسية وآخر الالتزامات.

### `CHANGES.md` داخل المستودع

يُولَّد من جديد بالكامل في كل تشغيل — **تنبيه:** هذا سيستبدل أي نسخة يدوية موجودة من الملف (إن أردت الاحتفاظ بنسخة يدوية منفصلة، انقلها إلى اسم آخر مثل `CHANGES.manual.md` قبل أول تشغيل).

---

## 6. استكشاف الأخطاء

| المشكلة | السبب والحل |
|---|---|
| تحذير «wiki push failed» | التوكن الافتراضي مُنع من الدفع للويكي — أنشئ `WIKI_TOKEN` (القسم 2) وأعد التشغيل |
| تحذير «could not commit CHANGES.md» | حماية الفرع تمنع دفع البوت — إمّا السماح لـ Actions بالدفع على main، أو اعتمد على نسخة الإصدار/الويكي (كلاهما محدَّث على أي حال) |
| صفحة الويكي فارغة عند أول تشغيل | طبيعي في أول مرة؛ الصفحات تُنشأ بالتشغيل الأول نفسه |
| بطء التحميل | تنزيل الخطوط + Chromium مرة واحدة لكل تشغيل؛ إجمالي التشغيل النموذجي ~6-10 دقائق |

---

## 7. ملفات المنظومة

```
.github/workflows/changes-report.yml   — سير العمل (الزناد + المراحل السبع)
scripts/changes_report/
  common.py            — التصنيفات والتسميات العربية المشتركة
  parse_commits.py     — git log --numstat -> commits.json (مصدر الحقيقة الوحيد)
  make_charts.py       — الرسم المقارن + لوحة التحليل (matplotlib)
  build_map.py         — خريطة HTML عربية RTL بخطوط مدمجة
  shoot_map.py         — تصوير الخريطة كاملةً (Playwright، شرائح + لحام)
  build_changes_md.py  — CHANGES.md + summary.json + release_notes.md
  update_wiki.py       — توليد صفحات الويكي والدفع إليها
  run_pipeline.sh      — تشغيل البايبلاين كاملًا محليًا أو في CI
  publish_release.sh  — إنشاء/تحديث الإصدار المتجدد (gh CLI)
  requirements.txt    — الاعتماديات
```
