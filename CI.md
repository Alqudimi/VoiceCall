# CI/CD — منظومة GitHub Actions الكاملة

> هذه الوثيقة تشرح كل ملفات `.github/` في المستودع: ماذا تفعل، متى تنطلق،
> وما الصلاحيات والسرّ التي تحتاجها. جميع الفحوص تعمل بلا أي إعداد إضافي
> منك — فقط ادفع إلى GitHub وستنطلق تلقائياً.

## الخريطة الكاملة

| # | الملف | متى ينطلق | الغرض |
|---|---|---|---|
| 1 | `workflows/ci.yml` | push إلى main + كل PR | الاختبارات الكاملة (unit+integration مع coverage) + فحص البناء والتغليف و CLI smoke |
| 2 | `workflows/reusable-tests.yml` | `workflow_call` فقط | قالب اختبارات قابل لإعادة الاستخدام (تستدعيه ci و nightly) |
| 3 | `workflows/lint.yml` | push إلى main + كل PR | ruff (أخطاء حقيقية فقط) + codespell (أخطاء إملائية) |
| 4 | `workflows/validate.yml` | push إلى main + كل PR | تحقق هيكلي لكل الأصول: shellcheck، bash -n، PowerShell parse، `node --check` لكل JS، `compileall` لكل بايثون، فحص JSON/YAML للإعدادات والـ notebook، و actionlint لملفات workflows نفسها |
| 5 | `workflows/colab.yml` | تغييرات notebooks/ أو scripts/colab/ (أسبوعي) | اتساق الـ notebook مع المثبّت مع المستودع + إمكانية الوصول لعنوان التثبيت الموحد |
| 6 | `workflows/codeql.yml` | push إلى main + PR + أسبوعي | تحليل CodeQL (لغة Python) — النتائج في Security > Code scanning |
| 7 | `workflows/security.yml` | push إلى main + tags + PR + أسبوعي | خط أمني شامل: gitleaks (سرّ مسربة)، pip-audit (CVEs في التبعيات المقفلة)، bandit (تحليل ساكن)، dependency-review (فرق التبعيات في PR)، trivy (مسح ملفات)، SBOM (SPDX) |
| 8 | `workflows/docker.yml` | PR: بناء فقط / main و tags v*: بناء + نشر إلى ghcr.io | صورة حاوية CPU جاهزة مع HEALTHCHECK |
| 9 | `workflows/release.yml` | دفع وسم `v*` (مثل `v1.0.1`) | بناء sdist+wheel من الوسم نفسه، اختبار الحزمة في venv نظيف، SBOM، ثم GitHub Release مع المرفقات وملاحظات تلقائية |
| 10 | `workflows/pypi.yml` | عند نشر release (اختياري) | نشر `rvc-platform` إلى PyPI — **معطّل افتراضياً**، يفعَّل بضبط متغيّر/سرّ (انظر أدناه) |
| 11 | `workflows/release-drafter.yml` | كل push إلى main | يحديث مسودة ملاحظات الإصدار القادم تلقائياً من وسوم PRs |
| 12 | `workflows/docs-pages.yml` | تغييرات `*.md` أو mkdocs.yml | بناء موقع توثيق mkdocs-material ونشره على GitHub Pages |
| 13 | `workflows/nightly.yml` | يومياً 03:00 UTC | الاختبارات الكاملة من جديد + تقرير انجراف التبعيات (`uv lock --upgrade`) + فحص تغليف |
| 14 | `workflows/cleanup.yml` | أسبوعياً (الأحد 03:30 UTC) | حذف مخازن Actions للبيانات خاملة >7 أيام + تشغيلات مكتملة أقدم من 30 يوماً |
| 15 | `workflows/stale.yml` | يومياً | وسم `stale` للمسائل/PRs غير النشطة 30 يوماً وإغلاقها بعد 7 أيام، مع استثناءات (pinned/security/upstream) |
| 16 | `workflows/labeler.yml` | كل PR | وسوم المجال تلقائياً من المسارات المتغيّرة (ci/colab/docs/web-ui/api/…) |
| 17 | `workflows/size-label.yml` | كل PR | وسم الحجم size:XS→XL من عدد الأسطر |
| 18 | `workflows/pr-title.yml` | فتح/تعديل PR | فرض عناوين conventional commits (`feat:`/`fix:`…) لتغذية ملاحظات الإصدار |
| 19 | `workflows/first-interaction.yml` | أول مشكلة/PR من مساهم جديد | رسالة ترحيب بالعربية مع توجيه للتوثيق |
| 20 | `workflows/lock-threads.yml` | يومياً | قفل النقاشات المغلقة الخاملة لمدة سنة (يمنع "الإحياء") |
| 21 | `workflows/label-sync.yml` | تغيير labels.yml | مزامنة مجموعة الوسوم القانونية من `.github/labels.yml` |
| 22 | `workflows/todo-to-issue.yml` | push إلى main يمس src/ أو scripts/ | تحويل علامات TODO/FIXME إلى مسائل متتبعة (وتغلقها عند إزالة العلامة) |
| 23 | `workflows/auto-merge.yml` | PRs من dependabot | دمج تلقائي للتحديثات الصغيرة (patch/minor) متى اخضرّت CI |
| 24 | `workflows/markdown-links.yml` | أسبوعياً + تغييرات md | فحص صحة كل روابط التوثيق بـ lychee (استشاري لا يُفشل) |

ملفات الدعم: `dependabot.yml` (تحديثات uv/github-actions/docker أسبوعية)،
`CODEOWNERS`، `labels.yml` + `labeler.yml` + `release-drafter.yml`، قوالب
الإبلاغ (`ISSUE_TEMPLATE/*` مع قالب مخصص لمشاكل Colab)، وقالب PR.

## ماذا يحدث عند أول دفعة (push) إلى GitHub؟

كل شيء يعمل فوراً بلا إعدادات:

1. **CI / Lint / Validate / Colab** تنطلق على push إلى main، وعلى كل PR.
2. **CodeQL** يفتح تحليلاً في Security > Code scanning.
3. **Security** يفحص التبعيات والسرّ المضمنة؛ SBOM يُنشأ كمُرفَق.
4. **Docker** يبني الصورة ويدفعها إلى `ghcr.io/alqudimi/voicecall:latest`.
5. **Docs** تنشر التوثيق على `https://alqudimi.github.io/VoiceCall/`
   (فعّل Pages مرة واحدة: Settings → Pages → Source: **GitHub Actions**).
6. **Release drafter** يجهّز مسودة الإصدار القادمة.

## دورة الإصدار (Release)

```bash
# 1) ادمج PRs إلى main — مسودة ملاحظات الإصدار تُحدَّث تلقائياً
# 2) عندما تقرر الإصدار: انشر المسودة من صفحة Releases أو وسم مباشرة:
git tag v1.0.1 && git push origin v1.0.1
# 3) release.yml يبني الحزمة، يختبرها، يرفق SBOM، ويصدر GitHub Release
```

## النشر إلى PyPI (اختياري — معطّل افتراضياً)

فعّله فقط إن أردت نشر الحزمة عمومياً:

1. Settings → Secrets and variables → Actions → **Variables** → أضف
   `PYPI_PUBLISH = true`
2. Settings → Secrets and variables → Actions → **Secrets** → أضف
   `PYPI_API_TOKEN` (توكن من pypi.org)
3. (اختياري للبيئات) Settings → Environments → أنشئ بيئة باسم `pypi`
4. عند نشر أي GitHub Release سينشر `pypi.yml` الحزمة تلقائياً.

## الفحوص التي تحتاج معرفة خاصة

- **bandit** يستثني `B324` (hashlib.md5) عمداً: MD5 تُستخدم كبصمات غير
  أمنية لفحص تكرار البيانات في `data-check` — ليست للاستخدامات الكريبتوغرافية.
- **ruff** يعمل بقواعد "عيوب حقيقية فقط" (انظر `[tool.ruff]` في pyproject.toml)،
  والكود المُدمج (vendored) في `src/rvc_platform/engine/rvc` مستثنى لأنه مشتق
  من مشروع RVC المنبعي (انظر ATTRIBUTION.md).
- **pip-audit** يقرأ `uv.lock` عبر `uv export` ويتحقق من كل التبعيات
  مقابل قاعدة بيانات OSV/PyPI Advisories.
- **markdown-links** استشاري (`continue-on-error`) لأن الروابط الخارجية
  تتعطل لظروف خارجية — لكنه يرفع تقريراً كمُرفَق أسبوعياً.

## الحماية الموصى بها للفرع (Branch protection)

في Settings → Branches أضف قاعدة لـ `main`:

- Require: CI, Lint, Validate, CodeQL, Colab
- Require approvals: 1 (CODEOWNERS يجعل مراجعتك إلزامية)
- Dismiss stale approvals عند دفعات جديدة
- Require conversation resolution

## التشغيل المحلي لما يعادل CI قبل الدفع

```bash
make test                # == job الاختبارات
uvx ruff check .         # == job الـ lint
uvx codespell .          # == job الإملاء
python3 scripts/ci/check_colab_assets.py   # == job الـ Colab
bash -n scripts/colab/install_colab.sh     # فحص صياغة المثبّت
uv build                 # == job التغليف
```

## الالتزامات مقابل السرّ

كل الملفات تعمل بـ `GITHUB_TOKEN` المدمج وبأقل صلاحيات ممكنة
(`contents: read` افتراضياً، وتوسَع فقط حيث يلزم: `packages:write` لبناء
Docker، `contents:write` للإصدارات، `issues:write` للبوتات…). لا حاجة
لأي PAT أو سرّ إضافي مطلقاً إلا `PYPI_API_TOKEN` إن فعّلت النشر إلى PyPI.
