# واجهة برمجة التطبيقات — API

توثيق كامل لواجهة Iraqi Voice Studio (FastAPI) على `http://127.0.0.1:8320`.
جميع المسارات تحت `/api`. المستند التفاعلي (Swagger): `http://127.0.0.1:8320/api/docs`.

---

## 1. الأساسيات والمصادقة

- **النقل:** HTTP/JSON (وWebSocket للبث — القسم 6).
- **المصادقة (Bearer):** معطلة افتراضيًا على localhost. عند تفعيلها يتطلب كل مسار
  `/api/*` (عدا `/api/health`) الترويسة `Authorization: Bearer <token>`.
- **تفعيل تلقائي:** عند نجاح `POST /api/tunnel/start` تصبح المصادقة **إلزامية**
  فورًا على كل الـAPI (رسالة الاستجابة تذكّرك).
- **التوكن:** إن لم يُضبط `server.auth_token` يُولَّد ويُكتب في `logs/auth.token`.
- **الاستثناء الدائم:** `GET /api/health`.
- **أكواد الأخطاء:** `400` طلب/بيانات غير صالحة، `401` توكن مفقود/خاطئ، `404`
  غير موجود، `409` تعارض (مثل: لا نموذج محمّل)، `422` ملف صوت/نموذج غير صالح،
  `502` فشل النفق.

كل الأمثلة تفترض:

```bash
export API=http://127.0.0.1:8320
export TOKEN=$(cat logs/auth.token)   # عند تفعيل المصادقة
```

---

## 2. النظام والأصول

### GET /api/system — الحالة الشاملة

- المصادقة: مطلوبة إن فُعّلت. الاستجابة: كائن حالة يضم
  `os`, `arch`, `python`, `device`, `is_half`, `gpu {name, vram_total_mb}`, `engine`,
  `engine_message`, `loaded_model`, `training {state, epoch, total_epochs}`,
  `realtime_sessions`, `tunnel` (الحقول التمثيلية — القيم من جهازك).

```bash
curl -s $API/api/system -H "Authorization: Bearer $TOKEN"
```

```json
{"os":"Linux","arch":"x86_64","python":"3.12.7","device":"cuda:0","is_half":true,
 "gpu":{"name":"NVIDIA GeForce RTX 3060","vram_total_mb":12288},
 "engine":"ready","loaded_model":{"name":"iraqi_voice","sample_rate":40000,"f0":true},
 "training":{"state":"running","epoch":42,"total_epochs":200},
 "realtime_sessions":1,"tunnel":{"running":false}}
```

### GET /api/system/detect — كشف البيئة

- المصادقة: إن فُعّلت. الاستجابة: ملخص بروفايل النظام
  (OS/معمارية/Python/الجهاز/fp16/CUDA).

```bash
curl -s $API/api/system/detect
```

### GET /api/health — فحص الحياة (معفى دائمًا من المصادقة)

```bash
curl -s $API/api/health
# {"ok": true, "detail": "Iraqi Voice Studio API is alive"}
```

### GET /api/assets — حالة الأصول المدربة مسبقًا

- المصادقة: إن فُعّلت. الاستجابة: لكل أصل `path`, `present`, `size_mb`
  + `_summary {hubert_ready, rmvpe_ready, pretrained_ready, all_present}`.

```bash
curl -s $API/api/assets
```

### POST /api/assets/download — تنزيل الأصول الناقصة بالخلفية

- المصادقة: إن فُعّلت. الطلب: بلا جسم.
- الاستجابة: `{"started": true, "message": "downloading in background; poll /api/assets"}`.
- عند جريان تنزيل قائم: `202` مع `{"started": false, "message": "download already running"}`.

```bash
curl -s -X POST $API/api/assets/download -H "Authorization: Bearer $TOKEN"
curl -s $API/api/assets      # متابعة التقدم
```

### GET /api/logs — سجل المنصة

- المصادقة: إن فُعّلت. المعامل: `lines` (int, افتراضي 200).
- الاستجابة: `{"lines": [...]}` — آخر أسطر `logs/platform.log`.

```bash
curl -s "$API/api/logs?lines=80" -H "Authorization: Bearer $TOKEN"
```

---

## 3. النماذج — `/api/models*`

النموذج = ملف `.pth` (checkpoint استدلال RVC) مع `.index` اختياري بجانبه في `models/`.
كل الاستجابات النموذجية بصيغة كائن ModelInfo: `name`, `sample_rate`, `f0`,
`version`, `info`, `size_mb`, `index_path` (إن وجد).

### GET /api/models — قائمة النماذج

- المصادقة: إن فُعّلت. الاستجابة: مصفوفة ModelInfo.

```bash
curl -s $API/api/models
```

### GET /api/models/{name} — تفاصيل checkpoint

- المصادقة: إن فُعّلت. المعامل: `name`. أخطاء: `404`.
- الاستجابة: بيانات فحص checkpoint + `name`, `index`, `engine_loaded`.

```bash
curl -s $API/api/models/iraqi_voice
```

### POST /api/models/{name}/load — تحميل إلى المحرك

- المصادقة: إن فُعّلت. الاستجابة: `{"loaded": {…ModelInfo…}, "engine": "<status>"}`.
- أخطاء: `404` غير موجود، `500` فشل تحميل.

```bash
curl -s -X POST $API/api/models/iraqi_voice/load -H "Authorization: Bearer $TOKEN"
```

### POST /api/models/unload — إفراغ المحرك

- المصادقة: إن فُعّلت. الاستجابة: `{"loaded": null}`.

```bash
curl -s -X POST $API/api/models/unload -H "Authorization: Bearer $TOKEN"
```

### POST /api/models/upload — رفع نموذج (multipart)

- المصادقة: إن فُعّلت. الطلب: `multipart/form-data` بحقل `file`: ملف `.pth` فقط
  (حد 300 MB)، يُفحص كونه checkpoint RVC صالحًا قبل التسجيل.
- الاستجابة: `{"registered": {…ModelInfo…}}`.
- أخطاء: `400` ليس `.pth`/فارغ/تجاوز 300MB، `422` checkpoint غير صالح.

```bash
curl -s -X POST $API/api/models/upload -H "Authorization: Bearer $TOKEN" \
     -F "file=@my_voice.pth"
```

### DELETE /api/models/{name} — حذف نموذج

- المصادقة: إن فُعّلت. الاستجابة: `{"deleted": "<name>"}`. أخطاء: `404`.
  (يُفرغ المحرك أولًا إن كان النموذج محمّلًا.)

```bash
curl -s -X DELETE $API/api/models/old_voice -H "Authorization: Bearer $TOKEN"
```

### GET /api/models/{name}/download — تنزيل ملف النموذج

- المصادقة: إن فُعّلت. الاستجابة: ملف ثنائي `.pth` (`application/octet-stream`). أخطاء: `404`.

```bash
curl -s -o iraqi_voice.pth "$API/api/models/iraqi_voice/download" \
     -H "Authorization: Bearer $TOKEN"
```

---

## 4. تحويل الملفات — `/api/convert*`

### GET /api/convert/settings — الافتراضيات الحية

- المصادقة: إن فُعّلت. الاستجابة: `defaults` (كل مفاتيح التحويل) +
  `engine {status, message, loaded_model}` + `f0_methods: ["rmvpe","fcpe","pm"]`.

```bash
curl -s $API/api/convert/settings
```

### POST /api/convert — تحويل ملف مرفوع (multipart + form)

- المصادقة: إن فُعّلت. أخطاء: `400` رفع غير صالح (فارغ/أكبر من 40MB/صيغة غير
  مدعومة)، `409` لا نموذج محمّل، `422` فشل فك/تحويل.

| الحقل | النوع | الافتراضي | المدى/القيم |
|---|---|---|---|
| `file` | ملف | إلزامي | WAV/MP3/FLAC/OGG/M4A/OPUS/WebM/AAC — حد 40MB (WAV/FLAC تعمل بدون ffmpeg) |
| `pitch_shift` | int | 0 | -24..24 (أنصاف نغمة) |
| `f0_method` | str | `rmvpe` | `rmvpe` \| `fcpe` \| `pm` |
| `index_rate` | float | 0.0 | 0..1 (مزج FAISS) |
| `filter_radius` | int | 3 | 0..7 (مرشح وسيط للنغمة) |
| `rms_mix_rate` | float | 0.25 | 0..1 (مزج غلاف الجهارة) |
| `protect` | float | 0.33 | 0..0.5 (حماية الصوات) |
| `output_sample_rate` | int | null | 16000..192000 (null = معدل النموذج) |
| `denoise` | bool | false | تنظيف طيفي للدخل |
| `normalize` | bool | true | تطبيع الدخل |

```bash
curl -s -X POST $API/api/convert -H "Authorization: Bearer $TOKEN" \
     -F "file=@input.wav" -F "pitch_shift=0" -F "f0_method=rmvpe" \
     -F "index_rate=0" -F "protect=0.33"
```

```json
{
  "output_path": "1738067_ab12cd_myvoice_iraqi_voice.wav",
  "duration_s": 12.4,
  "processing_s": 1.9,
  "realtime_factor": 0.153,
  "settings": {"pitch_shift": 0, "f0_method": "rmvpe", "index_rate": 0.0,
               "filter_radius": 3, "rms_mix_rate": 0.25, "protect": 0.33,
               "denoise": false, "normalize": true}
}
```

`realtime_factor = processing_s / duration_s` (أقل من 1 = أسرع من الزمن الحقيقي).
الناتج في `data/conversions/`.

### GET /api/convert/results/{result_id}/download — جلب النتيجة

- المصادقة: إن فُعّلت. المعامل: `result_id` = قيمة `output_path` من استجابة
  التحويل. الاستجابة: ملف WAV (`audio/wav`). أخطاء: `404`.

```bash
curl -s -o out.wav "$API/api/convert/results/1738067_ab12cd_myvoice_iraqi_voice.wav/download" \
     -H "Authorization: Bearer $TOKEN"
```

---

## 5. التدريب — `/api/training*`

### GET /api/training/status

- المصادقة: إن فُعّلت. الاستجابة: `state` (idle/preparing/running/stopping/
  completed/failed/stopped), `epoch`, `total_epochs`, `global_step`,
  `steps_per_epoch`, `latest_losses`, `started_at`, `updated_at`, `message`.

```bash
curl -s $API/api/training/status
```

### GET /api/training/runs — كل التجارب

- الاستجابة: مصفوفة: `experiment`, `run_dir`, `epoch`, `global_step`,
  `updated_at`, `has_checkpoints`, `hf_repo_id`, `hf_url`,
  `best` (`{epoch, global_step, metric, score, improved_at}` أو null)،
  و`hf_best` (بيانات أفضل نموذج المرفوع إلى `best/` في المستودع، إن وُجد).

```bash
curl -s $API/api/training/runs
```

### POST /api/training/prepare — تجهيز بيانات بالخلفية

- المصادقة: إن فُعّلت. الطلب JSON — الحقول:

| الحقل | النوع | الافتراضي |
|---|---|---|
| `experiment_name` | str | إلزامي (نمط `[A-Za-z0-9_-]{1,64}`) |
| `dataset_path` | str | إلزامي |
| `sample_rate` | str | `40k` (32k/40k/48k) |
| `version` | str | `v2` |
| `f0` | bool | true |
| `f0_method` | str | `rmvpe` |
| `speaker_id` | int | 0 (0..109) |

الاستجابة: `{"job": "<job_id>", "experiment": "<name>"}` — المراحل كما في
TRAINING.md (check→preprocess→f0→features→index). أخطاء: `400`.

```bash
curl -s -X POST $API/api/training/prepare -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"experiment_name":"iraqi_voice","dataset_path":"data/raw/myvoice"}'
```

### GET /api/training/prepare/status — تقدم التجهيز

- المعامل: `experiment` (query). الاستجابة: `{"stage": "...", "fraction": 0.4, ...}`
  أو `{"stage":"idle"}` إن لم يوجد نشاط.

```bash
curl -s "$API/api/training/prepare/status?experiment=iraqi_voice"
```

### POST /api/training/start — بدء/استئناف تدريب

- المصادقة: إن فُعّلت. الطلب JSON — الحقول (كلها اختيارية):

| الحقل | الافتراضي | المدى |
|---|---|---|
| `experiment_name` | null | تجربة جديدة أو قائمة (استئناف) |
| `dataset_path` | null | مطلوب للجديد |
| `sample_rate` | `40k` | 32k/40k/48k |
| `epochs` | 200 | 1..10000 |
| `batch_size` | 6 | 1..64 |
| `learning_rate` | 1e-4 | > 0 |
| `lr_decay` | 0.999875 | معامل ExponentialLR |
| `save_every_epoch` | 10 | 1..200 |
| `save_every_weights` / `save_latest` | true | نقاط التصدير/المرآة |
| `save_best` | true | تتبّع أفضل نموذج ورفعه إلى `best/` |
| `val_split` | 0.0 | 0..0.5 — نسبة بيانات التحقق لقياس الأفضل |
| `cache_in_gpu` | false | تخزين الميزات على GPU |
| `preload_base_models` | true | قواعد pretrained_v2 |
| `gpu` | null | فهرس GPU |
| `f0` / `f0_method` / `version` | true / rmvpe / v2 | |
| `resume` / `clean_start` | true / false | سلوك الاستئناف |
| `speaker_id` | 0 | 0..109 |

الاستجابة: `{"pid": <pid>, "run": "<name>", "config": "runs/<name>/run_config.json"}`
(المدرب عملية فرعية: `python -m rvc_platform.training.worker --config …`).
أخطاء: `400` بيانات/إعداد غير صالحة — ومنها تعارض بصمة البيانات عند الاستئناف
(الحل: اسم تجربة جديد أو clean_start — TRAINING.md).

```bash
curl -s -X POST $API/api/training/start -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"experiment_name":"iraqi_voice","dataset_path":"data/raw/myvoice","epochs":200,"batch_size":6}'
```

### POST /api/training/stop — إيقاف تعاوني

- الاستجابة: `{"running": false, "was_pid": <pid>, "clean": true}` (يُحفظ checkpoint
  أولًا ثم يخرج المدرب برمز 0) أو `{"running": false}` إن لم يكن قيد التشغيل.

```bash
curl -s -X POST $API/api/training/stop -H "Authorization: Bearer $TOKEN"
```

### POST /api/training/resume?experiment=<name>&checkpoint=<G_step.pth|latest> — استئناف

- المعاملات: `experiment` (query, اختياري — بدونها أحدث تجربة) و
  `checkpoint` (query, اختياري): اسم نقطة مثل `G_48.pth` أو مسار كامل أو
  الكلمة `latest` — يستكمل من تلك النقطة بالذات بدل الأحدث تلقائيًا.
- الاستجابة: كما `/start`. أخطاء: `404` لا تجارب/لا نقطة.

```bash
curl -s -X POST "$API/api/training/resume?experiment=iraqi_voice" \
     -H "Authorization: Bearer $TOKEN"

# استكمال من نقطة محددة (متاح كزر في واجهة الويب لكل صف في جدول النقاط):
curl -s -X POST "$API/api/training/resume?experiment=iraqi_voice&checkpoint=G_48.pth" \
     -H "Authorization: Bearer $TOKEN"
```

### GET /api/training/runs/{experiment}/checkpoints

- الاستجابة: مصفوفة: `kind` (G/D), `epoch`, `global_step`, `size_mb`, `path`.
- أخطاء: `404` لا تجربة.

```bash
curl -s $API/api/training/runs/iraqi_voice/checkpoints
```

### GET /api/training/runs/{experiment}/logs — سجل التدريب

- المعامل: `lines` (int, 200). الاستجابة: نص خام (`text/plain`) من `train.log`.

```bash
curl -s "$API/api/training/runs/iraqi_voice/logs?lines=50"
```

### GET /api/training/runs/{experiment}/events — أحداث JSONL

- المعامل: `tail` (int, 100). الاستجابة: `{"events": [...]}` — آخر أحداث
  `events.jsonl` (progress/started/resumed/finished/failed).

```bash
curl -s "$API/api/training/runs/iraqi_voice/events?tail=20"
```

### POST /api/training/export — تصدير إلى models/

- الطلب JSON: `experiment` (إلزامي), `epoch` (اختياري — بدونه آخر تصدير).
- الاستجابة: `{"model_path": "...", "registered": {…ModelInfo…}, "index": ...}`.
- أخطاء: `404` تجربة/epoch بلا تصدير.

```bash
curl -s -X POST $API/api/training/export -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"experiment":"iraqi_voice"}'
```

---

## 6. البث اللحظي — `/api/realtime*`

### GET /api/realtime/info — جاهزية البث

- المصادقة: إن فُعّلت. الاستجابة: `available` (نموذج f0 محمّل؟), `block_ms`,
  `crossfade_ms`, `lookback_ms`, `silence_threshold_db`, `f0_method`, `loaded_model`,
  `note` (وصف البروتوكول).

```bash
curl -s $API/api/realtime/info
```

### GET /api/realtime/stats — إحصاءات الكمون

- المصادقة: إن فُعّلت. الاستجابة: `aggregate {count, min_ms, avg_ms, p95_ms,
  max_ms, sessions}` + `sessions[]`: لكل جلسة `session_id`, `blocks`, `server {…}`,
  `bridge_overhead_ms`, `block_ms`, `client_rtt_p95_ms`, `engine_status`.

```bash
curl -s $API/api/realtime/stats
```

### WS /api/realtime/ws — جلسة البث

- المصادقة عبر `?token=` فقط (المتصفح لا يرسل ترويسات على WebSocket — إغلاق
  `4401` عند فشلها).
- البروتوكول: TEXT تحكم (`start`/`settings`/`ping`/`stats`) + BINARY Int16 mono
  @16kHz دخل / Int16 mono @معدل النموذج خرج.

التوثيق الكامل إطارًا بإطار مع مثال JS: [REALTIME.md](REALTIME.md) القسم 4.
قياس بلا ميكروفون (محرك حقيقي بمقاطع تخليقية):

```bash
uv run rvc-platform realtime-benchmark iraqi_voice --blocks 40 --block-ms 160
```

---

## 7. النفق — `/api/tunnel*`

### POST /api/tunnel/start

- المصادقة: إن فُعّلت (وتصبح إلزامية فور نجاحه). الطلب: بلا جسم.
- الاستجابة: `{"running": true, "url": "https://<random>.trycloudflare.com",
  "message": "tunnel active — auth token is now required for API access"}`.
- أخطاء: `502` cloudflared غير مثبت/فشل (يُعاد المحاولة حتى 3 مرات).
- **تذكير صريح:** العنوان **مؤقت** — جديد مع كل تشغيل.

```bash
curl -s -X POST $API/api/tunnel/start -H "Authorization: Bearer $TOKEN"
```

### POST /api/tunnel/stop

```bash
curl -s -X POST $API/api/tunnel/stop -H "Authorization: Bearer $TOKEN"
```

### GET /api/tunnel/status

- الاستجابة: `running` + `url` (إن كان نشطًا).

```bash
curl -s $API/api/tunnel/status
```

### GET /api/tunnel/logs

- المعامل: `lines` (int, 60). الاستجابة: `{"lines": [...]}` من سجل cloudflared.

```bash
curl -s "$API/api/tunnel/logs?lines=40"
```

---

## 8. صفحة الويب والملفات الثابتة

| المسار | الوظيفة |
|---|---|
| `GET /` | لوحة التحكم العربية RTL: لوحة التحكم / النماذج / تحويل الملفات / البث اللحظي / التدريب / الإعدادات |
| `GET /static/*` | أصول الواجهة (CSS/JS) |
| `GET /api/docs` | Swagger UI التفاعلي (OpenAPI) |

الواجهة تحفظ التوكن في `localStorage` وتمرره للـWebSocket بـ`?token=`.

---

## 9. سيناريو كامل بالـcurl (من الصفر إلى تحويل)

```bash
export API=http://127.0.0.1:8320
export TOKEN=$(cat logs/auth.token)

# 1) حالة + أصول
curl -s $API/api/system -H "Authorization: Bearer $TOKEN"
curl -s $API/api/assets

# 2) تجهيز + تدريب
curl -s -X POST $API/api/training/prepare -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"experiment_name":"iraqi_voice","dataset_path":"data/raw/myvoice"}'
curl -s "$API/api/training/prepare/status?experiment=iraqi_voice"
curl -s -X POST $API/api/training/start -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"experiment_name":"iraqi_voice"}'
curl -s $API/api/training/status

# 3) تصدير وتحميل
curl -s -X POST $API/api/training/export -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"experiment":"iraqi_voice"}'
curl -s -X POST $API/api/models/iraqi_voice/load -H "Authorization: Bearer $TOKEN"

# 4) تحويل ملف
curl -s -X POST $API/api/convert -H "Authorization: Bearer $TOKEN" \
     -F "file=@input.wav" -F "pitch_shift=0" -F "f0_method=rmvpe"
curl -s -o out.wav "$API/api/convert/results/<output_path>/download" \
     -H "Authorization: Bearer $TOKEN"
```
