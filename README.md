# Iraqi Voice Studio — منصة تحويل الصوت RVC v2

**الإصدار:** 1.9.0 | **اسم الحزمة:** `rvc-platform` | **الترخيص:** MIT

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
| [UPGRADE_REPORT_AR.md](UPGRADE_REPORT_AR.md) | تقرير الترقية البحثية: EMA / MRSTFT / MSD / المقياس المركب / تعزيز لحظي / إحماء+جيبي / مرشّح كلام (مراجع + أدلة) |
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
| تدريب نماذج صوت | خط كامل: فحص بيانات (بصمت MD5 + نسبة صمت)، تقطيع، استخراج F0 وHuBERT، فهرس FAISS، تدريب GAN مع استئناف دقيق — تلقائي أو **من نقطة محددة**، مع حفظ طارئ عند الانهيار، و**تتبّع أفضل نموذج** بـ**مقياس مركب** (val_mel + تشابه متحدث ECAPA + F0-RMSE) يُرفع ويُحدّث في `best/` على HuggingFace، و**وصفة الجودة البحثية مفعّلة افتراضيًا**: EMA للمولّد + خسارة MRSTFT + تمييزات MSD (scales 2/4) + إيقاف مبكر ضد الإفراط + تعزيز لحظي للجانب الحقيقي + إحماء/ذيل جيبي لمعدل التعلم + مرشّح نسبة الكلام — انظر UPGRADE_REPORT_AR.md | GPU واحد فقط (لا يوجد DDP متعدد GPU) |
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
| استئناف | `make train-resume` | `uv run rvc-platform train start --experiment <exp>` (v1.3.1: `--epochs/--batch/--lr/--save-every` تُطبَّق فعليًا على الجلسة القائمة دون إعادة تجهيز؛ SR وF0 مقفولان على ما جُهّز) |
| استئناف من نقطة محددة | — | `uv run rvc-platform train start --experiment <exp> --resume-from G_48.pth` (أو `latest`) |
| إيقاف تعاوني | `make train-stop` | `uv run rvc-platform train stop` (يقرأ `temp/trainer.pid` الذي يكتبه worker) |
| نقاط الحفظ | `make checkpoints EXPERIMENT=<exp>` | `uv run rvc-platform train checkpoints <exp>` |
| تصدير نموذج | `make export EXPERIMENT=<exp>` | `uv run rvc-platform train export <exp>` |
| قائمة النماذج | `make model-list` | `uv run rvc-platform model list` |
| تقييم موضوعي لنقطة تحقق | — | `uv run rvc-platform train evaluate <exp> [--checkpoint G_48.pth]` (val_mel + تشابه متحدث + composite → `runs/<exp>/eval_report.json`) |
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

## محاذاة الاستدلال الفوري والتحويل v1.9.0 — الجولة الثامنة (رtrvc/realtime_gui/pipeline الرسمية)

جُلبت ملفات المستودع الرسمي الحالية — `infer/rtrvc.py` (الاستدلال الفوري الجديد)، `realtime_gui.py` (999 سطراً)، `infer/vc/pipeline.py`، `infer/audio.py`، `configs/config.py`، `train/train_index.py` — وقورنت كلها سطراً بسطر:

| المكوّن | النتيجة |
|---|---|
| الاستدلال الفوري (rtrvc) | SOLA/cache_pitch/k=8/interpolate×2 **مطابق** + إصلاح 6 انحرافات (أدناه) |
| vc pipeline | change_rms/protect blend/p_len guard/opt_ts chunking/feats0 — **مطابق حرفياً** |
| get_f0 (pm/rmvpe/fcpe) | thred 0.03، local_argmax/0.006، pm بوسط pad — **مطابق** |
| بناء فهرس FAISS | 16√N≤N/39 خلايا IVF، MiniBatchKMeans 10000/256·n_cpu فوق 200k، دفعات 8192 — **مطابق حرفياً** |
| fcpe/torchgate | fcpe **مطابق**؛ torchgate غير مُستخدم عندنا (التنقية عندنا noisereduce اختياري) |

**الانحرافات المُصلحة (7):**
1. **بوابة الصمت الفورية**: كانت تعزل كل كتلة (تقصّ هجوم أول كلمة عند حدود كل كتلة) — الآن ذاكرة rms_buffer 40ms عبر الكتل + إسقاط إطارين + إزاحة ±5ms، **مُتحقّق رقمياً مطابق لrealtime_gui** في كل أحجام الكتل
2. عتبة البوابة الافتراضية -60dB (معطّلة) كما الرسمي — كانت -40 مفعّلة
3. crossfade افتراضي 50ms (الرسمي 0.05s) — كان 8ms
4. نافذة lookback الافتراضية 2500ms (الرسمي extra_time 2.5s) — كانت 200ms
5. `align_corners=True` في مزج غلاف RMS الفوري (كالرسمي)
6. **ميزانية تقسيم الاستدلال تُختار بـis_half لا بنوع الجهاز** (configs/config.py: 3/10/60/65 لfull) — كانت تعطي CUDA+fp32 ميزانية CPU
7. فك ffmpeg أصبح **f32le كامل الدقة** (infer/audio.py الرسمي الحالي) — كنا نُكمّم إلى int16؛ و`per` للتقطيع 3.0s الرسمية الحالية (كانت 3.7 القديمة)

11 اختبار انحدار جديد (`test_round8_alignment.py`) يثبّت كل ما سبق. الجناح: **229 وحدة + 51 تكامل = 280/280 أخضر**.

## إصلاح القوالب لكل (نسخة، معدل) v1.8.0 — الجولة السابعة (إصلاح توافق checkpoints)

جُلبت `train/train.py` و`train/preprocess.py` و`train/process_ckpt.py` وكل ملفات `configs/v1,v2` الرسمية وقورنت حرفياً:

| المكوّن | النتيجة |
|---|---|
| حلقة التدريب (train.py) | **مطابقة**: نفس توليفة الخسائر (gen+fm+mel·45+kl·1) تحت autocast معطّل، AdamW بbetas [0.8,0.99]/eps 1e-9، ExponentialLR بgamma 0.999875 — وحفظنا أكمل (scheduler/scaler/RNG كاملة بينما الرسمي يفقدها) |
| preprocess.py | **مطابق**: نفس القاطع (-42dB/1500/400/15/500)، نفس التطبيع (alpha 0.75/max 0.9)، حارس peak>2.5 — والرسمي نفسه يستخدم `lfilter` السببي هنا (filtfilt معلّق عنده) كما عندنا تماماً |
| process_ckpt.py | **مطابق حرفياً**: نفس مصفوفة config ذات الـ18 عنصراً، نفس استثناء enc_q، نفس half()، نفس المفاتيح (weight/config/info/sr/f0/version) |

**الانحراف الجوهري المكتشف — القوالب كانت مشتركة بين v1 وv2 لنفس المعدل**: الرسمي يقدّم قوالب منفصلة لكل (نسخة، معدل) وتختلف بنية المولّد بينهما لنفس المعدل! (v1_32k upsample=[10,4,2,2,2] مقابل v2_32k [10,8,2,2]؛ وv1_48k segment=11520 مقابل v2_48k 17280). قالبنا المشترك كان يعني: **أي وزة v1 مدرّبة رسمياً لن تُحمّل عندنا** (والعكس). الإصلاح:

- **6 قوالب رسمية حرفية**: `v1_32k/40k/48k.json` + `v2_32k/48k.json` منزّلة verbatim من configs الرسمية + نسختان بالاسم القديم للتوافق الخلفي (=v2)
- **`load_template(sr, version)`** يميّز النسخة، مع **رفض صريح لـ40k+v2** (لا وجود رسمياً لـv2 بـ40k — configs/v2 فيها 32k/48k فقط؛ الصمت كان سيولّد checkpoints غير متوافقة مع أدوات RVC الرسمية)
- الافتراضي أصبح متسقاً: `40k+v1` (المعدل الافتراضي نفسه مع قالبه الرسمي الصحيح)
- **22 اختبار انحدار جديد** (`test_official_templates.py`): أشكال المولّد لكل النسخ/المعدلات، ثوابت التدريب، segment لكل قالب، اختلاف v1/v2 لنفس المعدل (الخطأ الذي أُصلح)، رفض 40k-v2، عدم وجود قالب v2_40k مخترَع، وتوافق hop/win
- اختبارات التكامل نُقّلت كلها للأزواج الرسمية (v2/48k أو v1/40k) وقصاصاتها صارت مضاعفات hop الصحيحة

الجناح الكامل: **197/197 أخضر** (146 وحدة + 51 تكامل).

## تدقيق معماري شامل + مجموعة الصمت الرسمية v1.7.0 — الجولة السادسة (الأعمق)

جولة تدقيق آلية كاملة على مستوى السطر: نُزّلت ملفات المعمارية الرسمية الخمسة (attentions/commons/models/modules/transforms — 2486 سطراً) وقورنت هيكلياً وعددياً مع نسختنا:

| المكوّن | النتيجة |
|---|---|
| `transforms.py` (PRQ flow transform) | **مطابق هيكلياً بالكامل** (أداة مقارنة مطبّعة: أجسام متطابقة) |
| `models.py` | **4/4 أصناف SynthesizerTrn** بنفس البنية؛ حارس `n*upp != har` بنفس الشرط الرسمي |
| `commons.py` | slice_segments/slice_segments2 **بنفس الدلالات** (تجهير gather بنفس النوافذ) |
| `attentions.py` / `modules.py` | فروق `super().__init__()` التجميلية فقط |
| rmvpe الداخلية | cents_mapping/pad-32/thred=0.03/htk/clamp — **مطابقة بالكامل** |
| get_f0 الرسمي | 3 طرق (pm/rmvpe/fcpe): pm عندنا مطابق (عتبة 0.6، أرضية 50، خطوة 10ms، الحشو المركزي) |
| `train/data_utils.py` | phone repeat×2، حد 900، `.spec.pt` بنمط الحفظ نفسه، طول التقدير `//3·hop`، ترتيب الكوليت التنازلي — **مطابق سطراً بسطر** |

**الفجوة الجوهرية الوحيدة المكتشفة — مجموعة الصمت الرسمية (mute) كانت غائبة تماماً**: الرسمي يصنّف `mute.zip` "مطلوبة لتدريب v1/v2": يفكّها إلى `logs/mute/` (صمت 10ث لكل معدل عينة + ميزات HuBERT معدة + F0 صفرية) ويضمّ **سطرين لكل متحدث** في filelist بالمرجع — تُدرّب النموذج صمت→صمت وتمنع "طنين" التحويل على الدخل الهادئ. نفّذنا التكافؤ الوظيفي الكامل:

- **`training/mute_asset.py`** (جديد): يولّد الأصل بنفس بنية الرسمي (mute32k/40k/48k.wav + 3_feature768/256/mute.npy + 2a_f0/2b-f0nsf صفرية)، ويستخرج الميزات عبر خط HuBERT عندنا نفسه (أو أصفاراً عند غياب الأصل مع تحذير وإعادة توليد). **أولوية للنسخة الرسمية**: لو فُكّ mute.zip الأصلية إلى logs/mute تُستخدم حرفياً.
- **`write_filelist`** الآن يضمّ **سطرين لكل speaker_id** بنفس جرعة الرسمي، مخلوطين بالبذرة الحتمية نفسها (1234)، ويستنتج معدل العينة تلقائياً.
- **5 اختبارات انحدار** (`test_mute_asset.py`): البنية الرسمية، جرعة السطرين، النسخة non-f0، التكامل مع filelist (كل المسارات موجودة، 3+2 أسطر)، وحتمية الخلط. الجناح: **259/259 أخضر**.

التبعيات دُقّقت مقابل `requirments_*_py312.txt` الرسمية: كل مكتبات مسار v1/v2 الحرجة (numpy/scipy/librosa/transformers<4.50/parselmouth/torchfcpe/faiss) بنفس النطاقات؛ ملفات المعمارية الرسمية لا تحتاج local-attention/einops/hyper-connections لمسارنا.

## تدقيق التطابق الرسمي الشامل v1.6.0 — جولة رابعة على كل مكوّنات الأنباق المتبقية

قارنتُ حرفياً مع المستودع الرسمي (بالبنية الجديدة بعد إعادة الهيكلة) كل مكون لم يُدقّق بعد:

| المكوّن | المرجع الرسمي | النتيجة |
|---|---|---|
| قالب التدريب 48k | `configs/v2/48k.json` | **مطابق 100%** — كل الحقول (lr=1e-4، betas=[0.8,0.99]، eps=1e-9، lr_decay=0.999875، segment=17280، c_mel=45، c_kl=1.0، hop=480، وكل معمارية النموذج) |
| المعالجة المسبقة | `train/preprocess.py` | **مطابق** — قاطع الرسمي (-42dB/1500/400/15/500)، تطبيع الجهارة (alpha=0.75/max=0.9)، تخطي المقاطع الشاذة (peak>2.5)، قصاصات 3.7ث بتراكب 0.3ث |
| استخراج الميزات | `infer/hubert.py` | **مطابق** — طبقة 9+final_proj لv1 (256-بعد)، الطبقة الأخيرة لv2 (768-بعد)، layer_norm عند الحاجة |
| تصدير النموذج .pth | `train/process_ckpt.py` | **مطابق حرفياً** — نفس ترتيب config الـ18، إسقاط enc_q، تحويل half، مفاتيح sr/f0/version/info |
| المحرك اللحظي | `infer/rtrvc.py` | **مطابق** — f0_extractor_frame=block+800 مع تدوير rmvpe 5120، ذاكرة النغمة 1024 إطار بنفس منطق الإزاحة، skip_head//2 للمزج، حارس (ix>=0).all() |
| change_rms | `infer/vc/pipeline.py` | **مطابق حرفياً** — نفس frame/hop نصف-ثانية، interpolate خطي، المزج الأُسّي |
| مصادر التنزيل | `lj1995/VoiceConversionWebUI` | **مطابقة** — hubert_base، rmvpe.pt/onnx، f0G/f0D لكل المعدلات (تحققت HTTP 200 من المستودع الرسمي) |

**عيبان حقيقيان اكتُشفا وعُولجا في طبقة إدارة النماذج** (`model_repository.py`):

1. **اقتلاع فهرس نموذج آخر**: النموذج `ahmed.pth` كان يلتقط فهرس `ahmed2.index` (نموذج متحدث آخر!) عند غياب فهرسه الخاص — فيُمزج طابع متحدث بمتحدث آخر بصمت. والأسوأ: حذف `ahmed` كان يحذف فهرس `ahmed2` أيضاً. أُصلح بمطابقة الفهرس `<stem>.index` الدقيقة ثم المتغيرات بحدَّي شرطة سفلية فقط.
2. **انهيار السجل**: ملف `registry.json` صالح JSONياً لكن ببنية خاطئة (مثل `{"models": []}`) كان يُسقط عمليات التسجيل/الحذف/التعليق بخطأ بدل الشفاء الذاتي.

كما دقّقت وحدتي **security** (compare_digest ثابت الزمن، حارس WS 4401) و**processes** (مجموعات عمليات معزولة، إخلاء SIGTERM→SIGKILL مرتب) — كلتاهما سليمتان بلا ملاحظات.

**10 اختبارات انحدار جديدة** (`test_repository_index_isolation.py`): عزل الفهرس الدقيق، عدم اقتلاع فهرس نموذج آخر، بقاء فهرس الغير عند الحذف، حذف متغيرات الشرطة السفلية، وشفاء السجل في 4 حالات تلف + الحذف بسجل مكسور. الجناح الكامل: **254/254 أخضر** (203 وحدة + 51 تكامل).

## تدقيق ما بقي من نقاط الشك v1.6.0 — الجولة الخامسة (إغلاق 1% الأخيرة)

جرت هذه الجولة على كل ما لم يُدقّق رقمياً حتى الآن، بجلب المراجع الرسمية مباشرة:

| نقطة الشك | المرجع الرسمي | النتيجة |
|---|---|---|
| صيغة تحويل Hz→bin الخشنة (`*254/(max-min)+1`) — شُكّ أنها يجب أن تكون 255 | `infer/vc/pipeline.py` get_f0 | **مطابقة حرفياً** — الرسمي يستخدم 254 بنفس الصيغة والتثبيتات (1..255) وrint→int32 |
| بناء فهرس FAISS (n_ivf=16·√N بحد N//39، nprobe=1، IVF,Flat، تدريب دفعات 8192، KMeans إلى 10000 مراكز فوق 200000 صف، خلط عشوائي قبل الحفظ، نمط اسم added_IVF...) | `train/train_index.py` | **مطابق سطراً بسطر** بما فيه مصيدة N<39 (max(1, ...)) |
| mel spectrogram (hann، reflect، normalized=False، onesided، sqrt(re²+im²+2e-7)) | `train/mel_processing.py` | **مطابق** |
| ثوابت rmvpe (128 mel، 16kHz، 1024/160، 30–8000Hz) | `infer/rmvpe.py` | **مطابقة** |
| kl_loss/feature_loss/generator/discriminator | `train/losses.py` | **مطابقة** |

كما أُغلقت في هذه الجولة حالة **فهرس ناقص الصفوف**: فهرس IVF بلا متجهات مخزّنة (trained_*) أو متضرر كان قد يُستخدم في المزج بصمت — أُضيف حارس `(ix >= 0).all()` في مساري المزج (الملف + اللحظي) قبل أي استخدام، مطابقاً لحارس الرسمي.

## مواءمة الاستدلال الحرفية v1.5.0 — خمسة إصلاحات من المصدر الرسمي مباشرة

جلبت `infer/vc/pipeline.py` + `infer/vc/modules.py` + `train/train.py` الرسمية (البنية الجديدة للمستودع بعد إعادة الهيكلة) وقارنت مسار التحويل حرفياً. النتيجة: معظم المسار كان مطابقاً (أوزان مزج الفهرس 1/score²، حماية الصوائت pitchff، scale_factor=2، حشو reflect، تقطيع opt_ts، FAISS بنفس معاملاته، حلقة التدريب بجدولة لكل حقبة). خمسة انحرافات مؤكدة أُصلحت:

| # | الانحراف | الإصلاح الرسمي |
|---|---|---|
| 1 | مرشح التمريض العالي بـ`lfilter` السببي (يشوّه الطور ويميع هجمات الصوت) | `signal.filtfilt` صفري الطور — حرفياً كالرسمي |
| 2 | Praat pitch_floor=65Hz | **50Hz الرسمي** — أصوات الذكور العراقية العميقة (50-65Hz) كانت تُقرأ غير مُصوّتة وتُقتطع! |
| 3 | حشو فائض طول f0 لليمين فقط (ينزاح زمنياً عن إطارات HuBERT) | الحشو المركزي الرسمي `(p_len-len+1)//2` |
| 4 | median filter بنواة 2r+1 (=7 افتراضياً) يمسح vibrato وديناميكية النغمة | الرسمي الحالي **أزاله من الاستدلال** — الافتراضي الآن OFF، والتراثي المحفوظ نواة-3 ثابتة عند radius>2 |
| 5 | لا حماية ذروة للدخل | الحارس الرسمي `if max(|audio|)/0.95 > 1: scale` — يحمي ميزات HuBERT من القطع |

الإصلاح #2 وحده جوهري للاستخدام العراقي: المتحدثون الرجال ببيئات تسجيل منزلية كثيراً ما يستقرون تحت 65Hz نبرةً أساسية — والنظام القديم كان يصفر نغمتهم ثم يستوفيها خطياً (لهجة مسطّحة آلية). #4 يمسّ كل تحويل: التمهيد بنواة 7 كان يمحو تعرّج النغمة الطبيعي للعربية العراقية.

**11 اختبار انحدار جديد** (`test_official_inference_fixes.py`): تناظر الدفع صفري الطور، تتبّع نغمة 55Hz، بقاء نقطة الهجوم في موضعها، نواة-3 تزيل شوكة إطار واحد ولا تمس النغمة الملساء، الحارس يُطبّق فقط عند التجاوز. الجناح الكامل: **254/254 أخضر** (203 وحدة + 51 تكامل).

## محرّك توصية الحقب الرسمي v1.4.1 — FAQ Q9 كمقياس آلي

تدقيق معماري شامل (قالب 40k مطابق حرفياً 100% لـ`configs/v1/40k.json` الرسمي — تحقّق مباشر من المستودع؛ سلسلة v2+40k ومولّد 768-بُعد وHuBERT طبقة-9 + final_proj كلها مطابقة للرسمي؛ الأصول من `lj1995/VoiceConversionWebUI` الرسمي).

**الجديد — قياس جودة البيانات يتحول إلى توصية حقب رسمية آلية:**
- فحص البيانات الآن يقيس **SNR لكل ملف** (نسبة الصوت لأرضية الضجيج، بالمئينات 80 مقابل 10 لسلامة من النقرات) و**أرضية الضجيج** — إضافة للمدة/الصمت/التكرارات السابقة.
- التقرير (`dataset_report.json`) يشمل الآن `mean_snr_db` و`noise_floor_db` و`epoch_recommendation` مشتقة حرفياً من نص FAQ Q9 الرسمي: **ضجيجي (<20dB) → 20-30 حقبة فقط** ("رفعها أكثر لن يحسّن الجودة — يتعلم الضجيج")، جيد → 50-150، **نظيف (≥35dB) → حتى 200**.
- أمر `train start` يقرأ التقرير ويعرض التوجيه قبل بدء الساعات: إذا طلبت حقباً فوق توصية بياناتك المقيسة، تُنبَّه مع سبب القياس، مع نصيحة التوقف المبكر عند استقرار val_mel/speaker-sim.
- كل شيء best-effort: تقرير قديم بلا SNR = لا تحذير، لا كسر توافق.

**مصادر:** `github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI` (faq_en.md Q9/Q10، configs/v1/40k.json، hubert التحقق من models.py الرسمي) + قائمة `pretrained_v2` من `huggingface.co/lj1995/VoiceConversionWebUI`.

## إصلاحات ومواءمة رسمية v1.4.0 — تدقيق مقابل المصادر الأولية

جميع التغييرات أدناه مُدقّقة مقابل المصادر الرسمية (RVC الرسمي، أوراق HiFi-GAN/VITS/BigVGAN، وثائق huggingface_hub، وبطاقة نموذج speechbrain):

1. **[حرج] اسم نموذج ECAPA كان خاطئاً** — `spkrec-ecapa-voxcelebx` غير موجود في HuggingFace أصلاً (يرجع 401). حتى مع تثبيت speechbrain كان مقياس تشابه المتحدث يتعطل بصمت. صُحّح إلى `spkrec-ecapa-voxceleb` (EER 0.80%، Apache-2.0، يتطلب إدخال 16kHz — نعيد العينة تلقائياً).
2. **[حرج] `index_rate` الافتراضي كان 0.0** — أي أن مزج الاسترجاع (الآلية الرسمية في RVC لسحب الطابع نحو المتحدث وتقليل تسريب الطابع، FAQ Q11) كان معطّلاً كلياً في كل عمليات التحويل. الافتراضي الآن **0.75** كما في WebUI الرسمي؛ `1.0` يفرض طابع مجموعة التدريب بالكامل.
3. **مواءمة توازن GAN مع الرسمي** — كل المدربات الرسمية (HiFi-GAN، VITS، BigVGAN، so-vits-svc، RVC) تستخدم **نفس معدل التعلم للمولّد والمميّز**؛ حتى كود TTUR المرجعي يعطي D الأعلى (3x) لا الأدنى. الافتراضي `--d-lr-scale` أصبح **1.0** (كان 0.75 بلا سند أولي).
4. **قص تدرج المولّد (BigVGAN §3.4)** — خيار جديد `--clip-grad-norm` (مثال 500؛ الافتراضي 0=مطابق للرسمي). BigVGAN يوثّق أن قص النorm العام يمنع "الانهيار المبكر للمولّد"، ونوصي به عند بدايات غير مستقرة.
5. **تحذير التوكن الرسمي** — رسالة غياب HF_TOKEN الآن توجه إلى huggingface.co/settings/tokens وتوصي بالتوكن fine-grained (الخيار الرسمي للإنتاج) مع صياغة الإعداد في كولاب.

**مصادر التدقيق:**
- RVC الرسمي: `github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI` (docs/en/faq_en.md Q9-Q11، webui.py، train/train.py، train/train_index.py)
- HiFi-GAN: `arxiv.org/abs/2010.05646` + `github.com/jik876/hifi-gan` (λ_mel=45, λ_fm=2, ExponentialLR لكل-حقبة)
- VITS: `arxiv.org/abs/2106.06103` (تسوس 0.999^{1/8}/حقبة)
- BigVGAN: `arxiv.org/abs/2206.04658` (قص التدرج §3.4، snake activation)
- huggingface_hub: `huggingface.co/docs/huggingface_hub/guides/upload` (رفع أول بأول، إعادة المحاولة المدمجة 429/5xx، `HF_XET_HIGH_PERFORMANCE=1` للرفع الأسرع)
- speechbrain: `huggingface.co/speechbrain/spkrec-ecapa-voxceleb` (16kHz، VoxCeleb1+2)

## إصلاحات v1.3.3

1. **رفع HuggingFace لا يعمل أثناء التدريب** (خطأ استيقاظ الخيط: المهام كانت تُضاف للطابور دون إيقاظ خيط الرفع النائم — لا يرفع شيء حتى إيقاف التدريب): أُصلح بـ`notify()` فوري عند كل إضافة، مع تحذير صريح مرة واحدة عند غياب `HF_TOKEN` يشرح طريقة الإصلاح.
2. **اختيار أفضل نموذج أصمّ لهوية المتحدث**: عند غياب speechbrain كان الاختيار يعتمد على `val_mel` وحده (مقياس طيفي لا يسمع "شخصية" الصوت). الآن: `composite` الكامل عند توفر speechbrain، وإلا `composite_f0 = val_mel + 0.5×(F0-RMSE/100)` — متاح دائماً مع rmvpe ويُبقي أمانة النبرة في الاختيار. تغيّر المقياس عند الاستئناف يُعاد ضبط سجل الأفضلية تلقائياً.
3. **كاشف صحة GAN**: تصاعد `val_kl` المستمر (علامة سيطرة المميّز — الصوت يبتعد عن المتحدث بينما val_mel يتحسن شكلياً) أصبح يُكشف ويُحذّر مع التوصية.

## إصلاحات v1.3.2

1. **انهيار أول خطوة تدريب تحت `torch.compile` على كولاب** (`expected size 256==256, stride 1==320`): خطأ تخطيط في Inductor مع PyTorch 2.7. أُصلح جذرياً في `commons.py` (تقطيع متجه بدل حلقة الصفوف) و`models.py` (إعادة تشكيل مصدر NSF بدل transpose)، مع حارس `_CompileGuard` في المدرب يسقط تلقائياً إلى eager إن فشلت نواة مُجمَّعة — التدريب لا يتوقف بسبب compile أبداً.
2. **حذف عجلات GPU عند تشغيل السكربتات** (`Uninstalled 2 packages`): `uv run` العارية تعيد مزامنة `uv.lock` فترجع torch إلى CPU. كل خلايا النوتبوك الآن تضبط `UV_NO_SYNC=1`، وخلية التدريب تفحص عجلة GPU وتصلحها تلقائياً، وسكربت التثبيت يحذّر ويشفئ CPU-torch ذاتياً في النهاية.

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
