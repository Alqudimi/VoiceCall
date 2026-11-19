# Iraqi Voice Studio — منصة تحويل الصوت RVC v2

**الإصدار:** 1.0.0 | **اسم الحزمة:** `rvc-platform` | **الترخيص:** MIT

[![CI](https://github.com/Alqudimi/VoiceCall/actions/workflows/ci.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/ci.yml)
[![Lint](https://github.com/Alqudimi/VoiceCall/actions/workflows/lint.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/lint.yml)
[![Colab](https://github.com/Alqudimi/VoiceCall/actions/workflows/colab.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/colab.yml)
[![Security](https://github.com/Alqudimi/VoiceCall/actions/workflows/security.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/security.yml)
[![CodeQL](https://github.com/Alqudimi/VoiceCall/actions/workflows/codeql.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/codeql.yml)
[![Docker](https://github.com/Alqudimi/VoiceCall/actions/workflows/docker.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/docker.yml)
[![Docs](https://github.com/Alqudimi/VoiceCall/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/docs-pages.yml)
[![Nightly](https://github.com/Alqudimi/VoiceCall/actions/workflows/nightly.yml/badge.svg)](https://github.com/Alqudimi/VoiceCall/actions/workflows/nightly.yml)

منصة إنتاجية (production-grade) لتحويل الصوت (Voice Conversion) مبنية على محرك RVC v2،
متخصصة في اللهجة العراقية: خط تدريب كامل، تحويل ملفات، بث لحظي عبر WebSocket،
وواجهة ويب عربية بواجهة RTL، مع واجهة سطر أوامر موحدة وواجهة API كاملة.

> تنبيه مهم: النظام **يحوّل الطابع الصوتي (timbre)** للصوت، ولا يقوم بالترجمة ولا
> بتوليد كلام من نص. المتحدث يحتفظ بكلماته ونطقه؛ ما يتغير هو "هوية" الصوت الناتجة.
> جودة اللهجة العراقية تأتي من جودة بيانات التدريب، لا من المنصة نفسها.

---

## جدول المحتويات

| الملف | الموضوع |
|---|---|
| [INSTALLATION.md](INSTALLATION.md) | التثبيت خطوة بخطوة (Ubuntu / Windows / Colab / GPU) |
| [TRAINING.md](TRAINING.md) | تدريب نموذج صوتي جديد من بياناتك |
| [REALTIME.md](REALTIME.md) | البث اللحظي عبر WebSocket وقياس زمن الاستجابة |
| [API.md](API.md) | توثيق كامل لواجهة REST/WebSocket |
| [CONFIGURATION.md](CONFIGURATION.md) | كل مفاتيح الإعداد ومتغيرات البيئة |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | الأعطال الشائعة وحلولها |
| [CONTRIBUTING.md](CONTRIBUTING.md) | دليل المساهمة في الكود |
| [CI.md](CI.md) | منظومة GitHub Actions الكاملة (CI/CD/الأمان) |
| [SECURITY.md](SECURITY.md) | سياسة الأمان والإبلاغ عن الثغرات |
| ARCHITECTURE.md | بنية النظام الداخلية (وثيقة مستقلة) |

---

## القدرات الأساسية

| القدرة | الوصف | القيد |
|---|---|---|
| تدريب نماذج صوت | خط كامل: فحص بيانات (بصمت MD5 + نسبة صمت)، تقطيع، استخراج F0 وHuBERT، فهرس FAISS، تدريب GAN مع استئناف دقيق — تلقائي أو **من نقطة محددة**، مع حفظ طارئ عند الانهيار، و**تتبّع أفضل نموذج** (تحقق محجوز أو خسارة التدريب) يُرفع ويُحدّث في `best/` على HuggingFace، و**كل عمليات القراءة/الكتابة/الرفع عبر خيوط خلفية** لا تبطئ التدريب | GPU واحد فقط (لا يوجد DDP متعدد GPU) |
| تحويل الملفات | Audio-to-Audio: WAV/MP3/FLAC/OGG/M4A/OPUS/WebM/AAC (عبر ffmpeg) + **تسجيل مباشر من الميكروفون** داخل تبويب التحويل | WAV/FLAC تعمل بدون ffmpeg |
| البث اللحظي | من ميكروفون المتصفح إلى صوت محوّل عبر WebSocket، **مع اختيار الميكروفون وجهاز الإخراج** | يتطلب GPU من فئة CUDA (RTX) ونموذج f0 |
| واجهة ويب عربية | لوحة تحكم RTL: لوحة التحكم / النماذج / تحويل الملفات / البث اللحظي / التدريب / الإعدادات | `http://127.0.0.1:8320` |
| نشر مؤقت | نفق Cloudflare سريع (quick tunnel) بعنوان HTTPS مؤقت | العنوان يتغير مع كل تشغيل |
| واجهة API | REST + WebSocket مع مصادقة Bearer اختيارية | المصادقة تصبح إلزامية مع النفق |

---

## تثبيت خلال 5 دقائق (Quickstart)

المتطلبات: Python 3.12 فقط (`>=3.12,<3.13`) و `uv`. لا حاجة لأي GPU لهذه الخطوات.

```bash
# 1) تثبيت uv (مدير البيئات والحزم)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) جلب المشروع
git clone https://github.com/Alqudimi/VoiceCall.git iraqi-voice-platform
cd iraqi-voice-platform

# 3) إنشاء البيئة وتثبيت كل المجموعات (dev + training)
make install          # = uv sync --all-groups

# 4) تنزيل الأصول المدربة مسبقًا من HuggingFace
make download-models  # HuBERT + RMVPE + قواعد pretrained_v2

# 5) تشغيل لوحة التحكم
make web              # http://127.0.0.1:8320  (يعمل بالحظر؛ Ctrl-C للإيقاف)
```

بعد ذلك، أول تحويل ملف (بنموذج جاهز أو مرفوع):

```bash
# تشغيل اختبار المحرك (اختياري)
make check            # = rvc-platform install check

# تحويل ملف صوتي بنموذج موجود في models/
make inference INPUT=data/uploads/voice.wav MODEL=my_model PITCH=0 INDEX_RATE=0
# الناتج يُخزن في data/conversions/
```

على Windows أو Colab استخدم الأمر المكافئ في CLI (كل هدف make له مقابل 1:1):

```bash
uv run rvc-platform download-models
uv run rvc-platform web --port 8320
```

---

## مخططات سير العمل

### مسار التدريب (من بيانات خام إلى نموذج)

```
data/raw/<dataset>          (ملفات صوت خام: wav/mp3/flac/...)
        |
        |  make prepare-dataset DATASET=... EXPERIMENT=... SR=40k F0_METHOD=rmvpe
        |  [1] check        : فحص الفساد والتكرار (md5) وإحصاء المدد
        |  [2] preprocess   : تقطيع واعٍ للصمت، مقاطع 3.7s، مرشح highpass 48Hz، تطبيع
        |                     -> runs/<exp>/0_gt_wavs  +  1_16k_wavs
        |  [3] f0           : استخراج النغمة rmvpe|fcpe|pm -> 2a_f0 / 2b-f0nsf
        |  [4] features     : HuBERT 768-d -> 3_feature768
        |  [5] index        : فهرس FAISS added_IVF...index
        |  [6] filelist.txt + config.json
        v
make train                 (GAN trainer — عمليات فرعية قابلة للاستئناف)
        |   checkpoints/G_<step>.pth  D_<step>.pth   (حالة كاملة)
        |   checkpoints/G_2333333.pth  (مرآة "الأحدث دائمًا")
        |   checkpoints/state.json    (بصمة البيانات + حالة الاستئناف)
        |   checkpoints/best.json     (أفضل نموذج: مقياس/نتيجة/دورة)
        |   exports/*.pth             (نقاط تصدير دورية + <exp>_best_*.pth)
        |   [كل الكتابات عبر خيط I/O خلفي — الحلقة لا تنتظر القرص أبدًا]
        |   [أفضل نموذج + التصديرات تُرفع خلفيًا إلى HuggingFace
        |    مع إعادة محاولة و commit ذري، وتُحدّث best/ في مكانها]
        v
make export EXPERIMENT=<exp>   ->  models/<exp>.pth (+ .index اختياري)
```

### مسار تحويل الملفات

```
input.wav (أي صيغة مدعومة)
        |
        |  تحميل بمعدل 16kHz -> highpass 48Hz -> [denoise طيفي اختياري]
        |  تقطيع ذكي عند نقاط الهدوء -> لكل مقطع:
        |     HuBERT(768) -> مزج FAISS (index_rate) -> رفع دقة x2
        |     F0 (rmvpe/fcpe/pm + مرشح وسيط) -> حماية الصوات (protect)
        |     SynthesizerTrn NSF-HiFiGAN -> مزج غلاف RMS -> resample
        v
data/conversions/<...>.wav   (int16)
```

### مسار البث اللحظي

```
ميكروفون المتصفح (AudioWorklet)
        |  مقاطع Int16 أحادية @16kHz، كل 160ms افتراضيًا
        v
WebSocket  /api/realtime/ws   (إطارات TEXT للتحكم + BINARY للصوت)
        |  نافذة انزلاقية: lookback 200ms + crossfade + search + block
        |  مخزن نغمة 1024 إطار + بوابة ضجيج -40dB + مزج RMS + SOLA
        v
Int16 PCM بمعدل النموذج  ->  تشغيل فوري في المتصفح
```

---

## ما تحتاج معرفته بصراحة (القيود)

هذه المنصة مبنية لتكون صادقة عن قدراتها:

1. **بدون GPU (CPU فقط):** التدريب بطيء جدًا وغير عملي، والبث اللحظي **غير متاح**
   (زمن الاستدلال يتجاوز ميزانية المقاطع). استخدم تحويل الملفات على CPU.
2. **البث اللحظي** يتطلب GPU من فئة CUDA (مثل RTX) ونموذجًا مدربًا بـ f0 (pitch-guided)
   واتصالًا مستقرًا. الأرقام المرجعية من المشروع الأصلي RVC: ~170ms شاملًا على GPU
   مناسب، و~90ms مع تعريفات ASIO (Windows فقط). **قِس جهازك بنفسك** عبر
   `make realtime MODEL=<name>`.
   (قياس مرجعي فعلي من بيئة Linux CPU-only بـ vCPU×2: متوسط 519ms لكل مقطع
   160ms — أي أن اللحظية الحقيقية تتطلب GPU.)
3. **التدريب على GPU واحد فقط** — DDP متعدد GPU غير منفذ (قرارة تصميم متوافقة مع
   Colab حيث GPU واحد بطبيعته).
4. **نفق Cloudflare السريع** يعطي عنوان HTTPS مؤقتًا (`https://xxx.trycloudflare.com`)
   يتغير مع كل تشغيل — ليس عنوانًا دائمًا. عند تشغيله تصبح المصادقة إلزامية.
5. **جودة اللهجة العراقية** رهينة بجودة وكمية بيانات التدريب: الحد الأدنى 10 دقائق،
   والمثالي 10–60 دقيقة من كلام عراقي نظيف.

---

## استخدام أخلاقي

تقنية تحويل الصوت قوية وحساسة. عند استخدام Iraqi Voice Studio:

- استنسخ فقط أصواتًا تملك حق استخدامها (صوتك، أو بموافقة موثقة صريحة من صاحب الصوت).
- لا تستخدم المنصة للانتحال، أو انتحال شخصية، أو التضليل، أو تجاوز أي قانون أو اتفاقية.
- تحمل مسؤولية الاستخدام: هذا المشروع أداة هندسية، والمسؤولية القانونية والأخلاقية
  عن أي صوت منتج تقع على مستخدمه.

---

## البنية المختصرة للمستودع

```
iraqi-voice-platform/
├── Makefile                     أهداف موحدة (كل هدف يقابل أمر CLI)
├── pyproject.toml               Python >=3.12,<3.13 + uv + المجموعات
├── configs/default.yaml         الإعداد الافتراضي (انظر CONFIGURATION.md)
├── configs/training/{32k,40k,48k}.json   قوالب hyper-parameters للتدريب
├── scripts/colab|ubuntu|windows/         مثبتات لكل منصة (INSTALLATION.md)
├── notebooks/                   دفتر Colab المكافئ للتدريب السحابي
├── src/rvc_platform/
│   ├── core/                    كيانات، واجهات (Protocols)، أخطاء، أحداث
│   ├── application/             خدمات حالات الاستخدام (نماذج، تحويل، تدريب، بث، نفق)
│   ├── engine/rvc/              محرك RVC v2 (nets مُضمّنة + inference/realtime/checkpoint)
│   ├── training/                خط البيانات + المدرّب + نقاط الحفظ + worker
│   ├── realtime/                (البث عبر application/realtime_service.py + engine/rvc/realtime.py)
│   ├── infrastructure/          إعداد، تسجيل، DSP صوتي، تخزين، عمليات، أمان
│   ├── api/                     تطبيق FastAPI + المسارات
│   ├── cli/                     rvc-platform (typer)
│   ├── web/                     صفحة SPA عربية RTL
│   └── colab/                   مشغّل Colab (colab_runner.py)
├── tests/                       unit + integration (pytest)
├── assets/                      الأصول المدربة مسبقًا (تُنزّل، لا تُرفع مع git)
├── models/                      نماذج الأصوات المسجلة (.pth + .index)
├── data/                        raw/ uploads/ conversions/
└── runs/                        مجلدات تجارب التدريب
```

> ملاحظة: ملفات `assets/` و `models/` و `data/` و `runs/` و `logs/` تُنشأ عند أول
> تشغيل وتُدار محليًا؛ يمكن إعادة توجيهها بمتغيرات بيئة (انظر CONFIGURATION.md).

---

## الأوامر: خريطة make ↔ CLI الكاملة

كل هدف make يقابل أمرًا واحدًا في CLI (`uv run rvc-platform ...`) — استخدم CLI على
Windows/Colab حيث لا يوجد make. المجموعات: `install`, `env`, `train`, `model`
وأوامر مستقلة.

| الغرض | make | CLI |
|---|---|---|
| تثبيت/مزامنة البيئة | `make install` | `uv run rvc-platform env sync` |
| فحص استيرادات سريع | `make check` | `uv run rvc-platform install check` |
| فحص عميق (GPU/CUDA/الأصول) | `make doctor FIX=1` | `uv run rvc-platform install doctor --fix` |
| نظرة عامة على البيئة والمسارات | `make env` | `uv run rvc-platform env info` |
| عجلات CUDA |  `make env-gpu CUDA=cu128` | `uv run rvc-platform env gpu cu128` |
| تحديث الاعتماديات | `make update` | `uv run rvc-platform env update` |
| توليد uv.lock | `make lock` | `uv run rvc-platform env lock` |
| إعادة إنتاج من lockfile | `make verify` | `uv run rvc-platform env verify` |
| تنزيل الأصول | `make download-models` | `uv run rvc-platform download-models` |
| فحص بيانات | `make data-check DATASET=...` | `uv run rvc-platform data-check <dataset>` |
| استيراد مجلد صوت وتجهيزه | `make prepare-audio-dataset SOURCE=... NAME=... SR=40k` | `uv run rvc-platform prepare-audio-dataset <source> <name> --sr 40k` |
| تقطيع/تطبيع فقط | `make preprocess DATASET=... EXPERIMENT=... SR=40k` | `uv run rvc-platform preprocess <dataset> <exp> --sr 40k` |
| استخراج ميزات تجربة قائمة | `make extract-features EXPERIMENT=... F0_METHOD=rmvpe` | `uv run rvc-platform extract-features <exp> --f0-method rmvpe` |
| خط البيانات الكامل | `make prepare-dataset DATASET=... EXPERIMENT=... SR=40k F0_METHOD=rmvpe` | `uv run rvc-platform prepare-dataset <dataset> <exp> --sr 40k --f0-method rmvpe` |
| تدريب | `make train` | `uv run rvc-platform train start --dataset ... --experiment ... --epochs 200 --batch 6 --sr 40k` |
| استئناف | `make train-resume` | `uv run rvc-platform train start --experiment <exp>` |
| استئناف من نقطة محددة | — | `uv run rvc-platform train start --experiment <exp> --resume-from G_48.pth` (أو `latest`) |
| إيقاف تعاوني | `make train-stop` | `uv run rvc-platform train stop` (يقرأ `temp/trainer.pid` الذي يكتبه worker) |
| نقاط الحفظ | `make checkpoints EXPERIMENT=<exp>` | `uv run rvc-platform train checkpoints <exp>` |
| تصدير نموذج | `make export EXPERIMENT=<exp>` | `uv run rvc-platform train export <exp>` |
| قائمة النماذج | `make model-list` | `uv run rvc-platform model list` |
| تفاصيل نموذج | `make model-info MODEL=<name>` | `uv run rvc-platform model info <name>` |
| تحميل على خادم حي | `make model-load MODEL=<name> PORT=8320` | `POST /api/models/<name>/load` |
| تحويل ملف | `make inference INPUT=x.wav MODEL=y PITCH=0 INDEX_RATE=0` | `uv run rvc-platform inference x.wav y --pitch 0 --f0-method rmvpe --index-rate 0 --filter-radius 3 --rms 0.25 --protect 0.33 --output-sr null` |
| الويب (مع نفق) | `make web [TUNNEL=1] PORT=8320` | `uv run rvc-platform web --port 8320 [--tunnel]` |
| بدء/إيقاف كلي | `make start` / `make stop` / `make restart` | `make stop` = إيقاف الويب + نفق CLI (إن وُجد) |
| إيقاف الويب | `make web-stop` / `make stop` | `uv run rvc-platform web-stop` |
| حالة الويب الحية | `make web-status` | `uv run rvc-platform web-status` |
| قياس زمن البث (بلا ميكروفون) | `make realtime MODEL=y BLOCKS=40 BLOCK_MS=160` | `uv run rvc-platform realtime-benchmark y --blocks 40 --block-ms 160` |
| حالة البث اللحظي الحي | `make realtime-status` | `GET /api/realtime/stats` |
| حالة المنصة | `make status` | `uv run rvc-platform status` |
| السجل | `make logs LINES=120` | `uv run rvc-platform logs --lines 120` |
| تنظيف | `make clean` / `make clean-all` | `uv run rvc-platform clean [--all]` |
| نفق عام | `make tunnel` / `make tunnel-stop` / `make tunnel-status` | `POST /api/tunnel/start` / `POST /api/tunnel/stop` / `GET /api/tunnel/status` |
| أدوات Colab/Drive | — | `uv run rvc-platform colab <restore|snapshot|status|env>` |

متغيرات make المتاحة: `SR` (40k) `F0_METHOD` (rmvpe) `EPOCHS` (200) `BATCH` (6)
`EXPERIMENT` (iraqi_voice) `DATASET` (data/raw/iraqi_dataset) `PORT` (8320)
`MODEL` `INPUT` `PITCH` (0) `INDEX_RATE` (0) `BLOCKS` (40) `BLOCK_MS` (160)
`TUNNEL` `FIX`.

قائمة الأهداف كاملة: `make help`. قائمة الأوامر: `uv run rvc-platform --help`.

---

## الاختبارات

```bash
make test                # كل الاختبارات (pytest -q)
make test-unit           # اختبارات الوحدة فقط
make test-integration    # التكامل: تحميل نقاط حفظ تخليقية، خطوات تدريب صغيرة على بيانات تخليقية،
                         # فحص دخاني للـAPI، جلسة WebSocket تخليقية
```

الاختبارات الموسومة `gpu` تُتخطى تلقائيًا بدون GPU فعلي، و`slow` للبطيئة.

---

## الإسناد (Attribution)

- المحرك مبني على المشروع الأصلي
  [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  (رخصة MIT)؛ الملفات المضمّنة موثقة المصدر في `src/rvc_platform/engine/rvc/ATTRIBUTION.md`.
- الأصول المدربة مسبقًا من إصدار HuggingFace الرسمي `lj1995/VoiceConversionWebUI`.
- SOLA crossfade مستلهم من DDSP-SVC عبر upstream.

## الرخصة

MIT — انظر ملف الترخيص في المستودع.
