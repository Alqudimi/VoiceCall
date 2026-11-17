# حل الأعطال — TROUBLESHOOTING

دليل عملي لأشهر الأعطال في Iraqi Voice Studio. قبل كل شيء، اجمع الحقائق:

```bash
make doctor               # فحص GPU/CUDA/الأصول/الاستيرادات
make status               # حالة الخادم الحي أو لقطة دون اتصال
make logs LINES=120        # آخر سجلات logs/platform.log
curl -s http://127.0.0.1:8320/api/health
```

---

## 1. التثبيت والبيئة

| العرَض | السبب المحتمل | الحل |
|---|---|---|
| `uv: command not found` | uv غير مثبت | `curl -LsSf https://astral.sh/uv/install.sh \| sh` ثم أعد فتح الطرفية |
| رفض Python بالإصدار | غير 3.12 (يتطلب `>=3.12,<3.13`) | ثبّت Python 3.12 وأعد `uv sync --all-groups` |
| فشل استيراد عند `make check` | بيئة ناقصة/فاسدة | `make verify` (إعادة إنشاء من `uv.lock` + فحص)؛ إن استمر: احذف `.venv` وأعد `make install` |
| عجلة torch لا تعمل بعد تحديث | مزج CPU/GPU wheels | `make env-gpu CUDA=cu124` بنفس إصدار torch الحالي فقط |
| `psutil` مفقود عند web-stop | مجموعة dev غير مثبتة | `uv sync --all-groups` |
| سكربت المثبّت يرفض المتابعة | فحوص Ubuntu/arch/python ناقصة | ثبّت الناقص كما يقول السكربت (`apt`) ثم أعد تشغيله |

## 2. GPU / CUDA

| العرَض | السبب | الحل |
|---|---|---|
| `cuda available: False` مع GPU موجود | عجلات CPU لا تزال مثبتة | `make env-gpu CUDA=cu124` (أو cu121/cu118/cu128 بحسب درايفر NVIDIA) |
| fp16 لا يتفعل | GPU غير مؤهل أو `precision: fp32` | تحقق من `make doctor`؛ اضبط `engine.precision: auto` |
| `CUDA out of memory` أثناء التدريب | batch كبير للـVRAM | أنزل `--batch` (6–8GB ≈ 6؛ 24GB ≈ 12)؛ أعد التشغيل لتحرير الذاكرة |
| `CUDA out of memory` عند البث | جلسات متعددة/نموذج آخر محمّل | جلسة واحدة، `POST /api/models/unload` ما لا تستخدمه |
| أداء GPU أضعف من المتوقع | تشغيل على iGPU/توفير طاقة | `make doctor` لعرض اسم GPU وVRAM؛ قارن بـ`make realtime MODEL=<name>` |

## 3. الأصول المدربة مسبقًا

| العرَض | السبب | الحل |
|---|---|---|
| خطأ: assets missing عند التدريب/التحويل | لم تُنزّل بعد | `make download-models` أو `rvc-platform install doctor --fix` أو `POST /api/assets/download` |
| فشل تنزيل/انقطاع | شبكة | التنزيل يستأنف تلقائيًا (byte-range على ملف `.part`) — أعد المحاولة؛ تحقق من الوصول إلى huggingface.co |
| `hubert_ready: false` رغم الملفات | بنية `assets/hubert_base/` ناقصة (يلزم `config.json` + `pytorch_model.bin`) | احذف المجلد وأعد `download-models` |
| `pretrained_ready: false` | نقص `f0G40k.pth`/`f0D40k.pth` | أعد التنزيل (الملفات موثقة في INSTALLATION.md قسم 5) |

## 4. الخادم والـAPI والمصادقة

| العرَض | السبب | الحل |
|---|---|---|
| `port 8320 already in use` | خادم سابق شغّال | `make web-status` ثم `make web-stop`؛ أو `PORT=8330 make web` |
| `401 missing or invalid bearer token` | المصادقة مفعلة (نفق غالبًا) | اقرأ `logs/auth.token` وأرسل `Authorization: Bearer <token>` |
| WS يُغلق برمز `4401` | توكن لم يُمرر في `?token=` | المتصفح لا يرسل ترويسات على WebSocket — استخدم `ws://…?token=<token>` (REALTIME.md) |
| نتيجة تحويل `409` | لا نموذج محمّل | `POST /api/models/{name}/load` أولًا |
| رفض رفع صوت `400` | فارغ/أكبر من 40MB/صيغة غير مدعومة | ارفع ملفًا أصغر بصيغة مدعومة (WAV/MP3/FLAC/OGG/M4A/OPUS/WebM/AAC) |
| رفض رفع نموذج `422` | ليس checkpoint RVC صالحًا | النظام يفحص الملف قبل التسجيل — استخدم `.pth` من RVC v1/v2 سليم |
| `502` عند بدء النفق | cloudflared غائب أو فشل البدء | ثبّت cloudflared (INSTALLATION.md قسم 7)؛ تُعاد المحاولة 3 مرات تلقائيًا |

## 5. تحويل الملفات

| العرَض | السبب | الحل |
|---|---|---|
| `cannot decode <file> without ffmpeg` | ffmpeg غائب وغير WAV/FLAC | ثبّت ffmpeg أو استخدم WAV/FLAC |
| صوت الناتج بضجيج/رخاوة | دخل ضعيف أو إعدادات | فعّل `denoise`؛ جرّب `index_rate` أعلى (مع فهرس)؛ `protect` 0.33 يحمي الصوات العراقية |
| نغمة الناتج غير طبيعية | `pitch_shift` غير ملائم أو F0 سيئة | اضبط `pitch_shift` (±24)؛ جرّب `rmvpe` إن كنت على `fcpe`/`pm` |
| بطء التحويل | CPU | طبيعي على CPU (أبطأ من الزمن الحقيقي) — استخدم GPU أو انتظر |
| الناتج عالي/منخفض الجهارة | مزازجة RMS | اضبط `rms_mix_rate` (0 = جهارة المصدر، 1 = جهارة النموذج) |
| فشل تحويل OPUS/WebM/AAC | ffmpeg غائب | ثبّت ffmpeg — هذه الصيغ تُفك عبره فقط |

## 6. التدريب

| العرَض | السبب | الحل |
|---|---|---|
| خطأ بصمة البيانات عند الاستئناف | تغيّرت بيانات التجربة عن وقت نقطة الحفظ | كما يقول الخطأ: اسم تجربة جديد أو `--clean-start` (TRAINING.md قسم 5) |
| التدريب بطيء جدًا | CPU أو GPU ضعيف | تحقق `make doctor`؛ على CPU البطء متوقع — استخدم Colab/GPU |
| `no existing run and no --dataset given` | تجربة غير موجودة بلا بيانات | مرّر `--dataset` أو ابدأ بـ`prepare-dataset` |
| انقطاع mid-training | OOM/انقطاع كهرباء | الحفظ ذري؛ أعد `make train-resume` (أو `POST /api/training/resume`) — يكمل من آخر حالة كاملة (epoch/step/RNG/optimizer) |
| خسائر لا تنخفض | بيانات قليلة/مختلطة الأصوات | 10–60 دقيقة من صوت واحد نظيف (TRAINING.md قسم 1) |
| `/api/training/status` يظهر idle رغم تشغيل CLI | التدريب بالـCLI عملية مستقلة عن الخادم | راقب CLI مباشرة أو شغّل التدريب عبر الـAPI/الويب |
| توقف المدرب برمز 2 | خطأ إعداد | راجع `runs/<exp>/run_config.json` ورسالة الخطأ |
| توقف المدرب برمز 3 | فشل تشغيلي أثناء الخطوة | راجع `train.log` و`events.jsonl` ثم أعد الاستئناف |

أكواد خروج المدرب: `0` نجاح/إيقاف تعاوني، `2` خطأ إعداد، `3` فشل تشغيلي.

## 7. البث اللحظي

| العرَض | السبب | الحل |
|---|---|---|
| صوت متقطع (underruns) | `avg_ms` ≥ `block_ms` | قِس: `make realtime MODEL=<name>`؛ ارفع `block_ms` (حتى 500) أو GPU أقوى |
| رفض الجلسة: يتطلب نموذج f0 | النموذج بلا pitch | درّب/حمّل نموذج f0 (البث لا يعمل بلا نغمة) |
| كسور/طقطقة عند حدود المقاطع | crossfade منخفض جدًا | ارفع `realtime.crossfade_ms` (حتى 40ms) |
| نغمة متذبذبة | F0 غير مستقر | `rmvpe` بدل `fcpe`؛ ارفع `lookback_ms` |
| البث لا يبدأ على CPU | الحد المتوقع | البث يتطلب CUDA RTX-class — على CPU استخدم تحويل الملفات (REALTIME.md قسم 1) |
| RTT عبر النفق مرتفع | مسار الشبكة | طبيعي؛ خفّف `block_ms` على GPU قوي لموازنة الشبكة |
| تأخير ثابت حوالي block_ms | طبيعة التجميع | التأخير يتناسب مع `block_ms` + الشبكة — لا يوجد وضع "صفر كمون" |

## 8. النفق

| العرَض | السبب | الحل |
|---|---|---|
| `502 tunnel` عند `/api/tunnel/start` | cloudflared غير مثبت | ثبّته — يُعاد المحاولة 3 مرات تلقائيًا قبل الفشل |
| العنوان تغيّر بعد إعادة التشغيل | طبيعة quick tunnel | العنوان مؤقت بتصميمه (جديد كل تشغيل) — لا يوجد اسم دائم |
| النفق يعمل لكن API يرفض 401 | المصادقة تفعّلت تلقائيًا مع النفق | مقصود أمنيًا — استخدم التوكن (`logs/auth.token` أو `server.auth_token`) |
| الخادم محلي يعمل والنفق 502 | الخادم على منفذ مختلف | تأكد `server.port` وتشغيل الويب قبل `make tunnel` |

## 9. البيانات والصيغ

| العرَض | السبب | الحل |
|---|---|---|
| `data-check` يظهر ملفات corrupt | ملفات تالفة | احذفها من `data/raw/` — التقرير يسميها |
| تحذير: أقل من 10 دقائق | بيانات قليلة | اجمع 10–60 دقيقة لصوت نظيف (TRAINING.md) |
| تكرارات md5 مكتشفة | ملفات مكررة | احذف النسخ — التدريب على التكرارات يضخّم تحيزًا |
| ملفات `.aiff` مقبولة بالفحص وترفض لاحقًا | الترميز يحتاج ffmpeg | ثبّت ffmpeg أو حوّل إلى WAV |

## 10. Windows وColab — خصوصيات

### Windows

| العرَض | الحل |
|---|---|
| PowerShell يرفض السكربت | `powershell -ExecutionPolicy Bypass -File scripts/windows/install_windows.ps1` |
| لا Makefile افتراضيًا | استخدم CLI: `uv run rvc-platform <command>` — كل هدف make له مكافئ |
| البطء الشديد بعد التثبيت | تحقق من GPU: `uv run rvc-platform install check` — إن كان CPU wheels نفّذ `env gpu cu124` |
| ASIO لا يظهر عبر المتصفح | أرقام ASIO (~90ms) خاصة بتطبيقات upstream على Windows — البث عبر المتصفح يستخدم قناة الصوت الاعتيادية |

### Colab

| العرَض | الحل |
|---|---|
| انقطعت الجلسة وضاع التقدم | شغّل مثبّت Colab مجددًا — يستأنف من آخر checkpoint متزامن في Drive (`RVC_Iraqi/`) |
| GPU لم يُمنح | من `Runtime > Change runtime type` اختر GPU ثم أعد المثبّت |
| مساحة Drive ممتلئة | نظّف `runs/` القديمة أو ارفع `RVC_RUNS_DIR` إلى مسار أوسع |

## 11. أين تجد كل سجل

| السجل | الموقع |
|---|---|
| المنصة | `logs/platform.log` (يتدور كل 10MB × 5) |
| التوكن (إن وُلّد) | `logs/auth.token` |
| النفق | `logs/tunnel.log` و`GET /api/tunnel/logs` |
| العمليات المُدارة | `logs/processes/` |
| تدريب تجربة | `runs/<exp>/train.log` + `events.jsonl` + `progress.json` |

## 12. تنظيف وإعادة الضبط

```bash
make clean                # يحذف .cache/ و.tmp/ فقط (يبقى كل شيء آخر)
make clean-all            # + runs/ + data/uploads + data/conversions
# انتبه: clean-all لا يمس models/ أو assets/
```

---

## 13. عند عدم وجود حل هنا

1. `make doctor FIX=1` — فحص شامل مع إصلاح الأصول.
2. `make logs LINES=200` و`GET /api/logs?lines=200` — السجلات أولًا.
3. سجلات التدريب: `runs/<exp>/train.log` و`events.jsonl` و`progress.json`.
4. سجل النفق: `GET /api/tunnel/logs` و`logs/tunnel.log`.
5. راجع [API.md](API.md) (أكواد الأخطاء) و[REALTIME.md](REALTIME.md) (البروتوكول)
   و[TRAINING.md](TRAINING.md) (دقة الاستئناف) و[CONFIGURATION.md](CONFIGURATION.md).

عند الإبلاغ عن علة: أرفق مخرجات `make doctor` و`make status` والأسطر ذات الصلة من
السجل (راجع CONTRIBUTING.md — قسم إبلاغ الأعطال).
