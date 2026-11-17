# الإعداد — CONFIGURATION

توثيق كامل لإعداد Iraqi Voice Studio: طبقات التحميل، كل مفتاح في `configs/default.yaml`،
متغيرات البيئة، وإعادة توجيه المسارات.

---

## 1. طبقات التحميل (الأسبقية)

```
code defaults (PlatformConfig)          الأدنى
        |
configs/default.yaml                    الإعداد المرفق
        |
configs/user.yaml  أو  $RVC_PLATFORM_CONFIG   ملف المستخدم
        |
RVC_* environment variables             الأعلى — تتغلب على كل ما سبق
```

- ملف المستخدم: `configs/user.yaml` افتراضيًا، أو أي مسار عبر `RVC_PLATFORM_CONFIG`.
- الدمج عميق (deep merge): يكفي أن يذكر ملف المستخدم المفاتيح التي يريد تغييرها.
- الكتابة البرمجية مدعومة أيضًا (`ConfigLoader.write_user`) مع استثناء القيم
  المستمدة من متغيرات البيئة وقت الكتابة.

مثال `configs/user.yaml`:

```yaml
server:
  port: 9000
engine:
  device: cuda:0
  precision: fp16
realtime:
  block_ms: 200
```

---

## 2. القسم `server`

| المفتاح | الافتراضي | النوع | الشرح |
|---|---|---|---|
| `host` | `127.0.0.1` | str | استخدم `0.0.0.0` فقط خلف نفق أو على Colab |
| `port` | `8320` | int (1024..65535) | منفذ الخادم |
| `auth_enabled` | `false` | bool | يُفعَّل **تلقائيًا** عند بدء نفق |
| `auth_token` | `""` | str | فارغ → يُولَّد توكن ويُكتب في `logs/auth.token` عند التفعيل |
| `cors_origins` | `["*"]` | list[str] | أصول CORS المسموحة |
| `max_upload_mb` | `200` | int (1..2048) | حد التحقق العام لرفع الصوت (خدمة التحويل تفرض 40MB) |

## 3. القسم `engine`

| المفتاح | الافتراضي | الشرح |
|---|---|---|
| `device` | `auto` | `auto` \| `cpu` \| `cuda:<index>` (مثل `cuda:0`) |
| `f0_method` | `rmvpe` | مستخرج النغمة الافتراضي: `rmvpe` (الأدق) \| `fcpe` (الأسرع) \| `pm` (CPU) |
| `precision` | `auto` | `auto` \| `fp16` \| `fp32` — auto تعني fp16 على GPU مؤهل |

> مفتاح إضافي في المخطط (schema) غير موجود في default.yaml: `engine.cuda_graph`
> (bool, افتراضي `false`) — مسار التقاط/إعادة تشغيل CUDA graph على CUDA فقط.

## 4. القسم `conversion` (الافتراضيات التي يبدأ منها كل تحويل)

| المفتاح | الافتراضي | المدى | الشرح |
|---|---|---|---|
| `pitch_shift` | 0 | -24..24 | إزاحة نغمة بأنصاف النغمة |
| `index_rate` | 0.0 | 0..1 | نسبة مزج ميزات FAISS مع HuBERT (0 = بدون فهرس) |
| `filter_radius` | 3 | 0..7 | مرشح وسيط للنغمة (0 يعني تعطيله عمليًا) |
| `rms_mix_rate` | 0.25 | 0..1 | مزج غلاف جهارة المصدر مع الناتج |
| `protect` | 0.33 | 0..0.5 | حماية الصوات/الحروف غير المجهورة من التشويه |
| `denoise` | false | bool | تنظيف طيفي (spectral gate) للدخل قبل التحويل |
| `normalize` | true | bool | تطبيع مستوى الدخل |
| `output_sample_rate` | null | 16000..192000 | null = معدل النموذج |

## 5. القسم `realtime`

| المفتاح | الافتراضي | المدى | الشرح |
|---|---|---|---|
| `block_ms` | 160 | 60..500 | حجم مقطع البث؛ يجب أن يبقى زمن استدلال الخادم دونه |
| `lookback_ms` | 200 | 100..1000 | سياق F0/التاريخ المغذّى مع كل مقطع |
| `crossfade_ms` | 8 | 0..40 | مزج SOLA بين المقاطع المتتالية |
| `silence_threshold_db` | -40 | -70..-20 | بوابة ضجيج المقاطع |
| `keep_alive_s` | 60 | 5..600 | إبقاء الجلسة حية |
| `f0_method` | rmvpe | rmvpe/fcpe/pm | مستخرج نغمة البث |

## 6. القسم `training`

| المفتاح | الافتراضي | الشرح |
|---|---|---|
| `sample_rate` | `"40k"` | `32k` \| `40k` \| `48k` |
| `version` | `v2` | `v1` \| `v2` (v2+f0 هو المسار الموصى به) |
| `f0` | true | تدريب موجّه بالنغمة (مطلوب للبث اللحظي) |
| `f0_method` | `rmvpe` | لاستخراج ميزات البيانات (rmvpe/fcpe/pm) |
| `epochs` | 200 | 1..10000 |
| `batch_size` | 6 | 1..64 — VRAM 6–8GB ≈ 6؛ 24GB ≈ 12 |
| `learning_rate` | `1.0e-4` | لـ AdamW |
| `lr_decay` | `0.999875` | معامل ExponentialLR لكل خطوة |
| `save_every_epoch` | 10 | كل N epochs نقطة + تصدير |
| `save_latest` | true | تحديث مرايا `G_2333333.pth`/`D_2333333.pth` |
| `save_every_weights` | true | نقاط تصدير استدلال دورية إلى `exports/` |
| `cache_in_gpu` | false | تخزين ميزات التدريب على GPU |
| `preload_base_models` | true | البدء من قواعد `pretrained_v2` (f0G/f0D) |
| `gpu` | null | فهرس GPU صريح (null = auto) |

## 7. القسم `tunnel`

| المفتاح | الافتراضي | الشرح |
|---|---|---|
| `enabled` | false | بدء النفق عند إقلاع الويب (مكافئ `--tunnel`) |
| `binary` | `cloudflared` | اسم/مسار الثنائي |
| `retries` | 3 | محاولات البدء |
| `start_timeout_s` | 30 | مهلة انتظار عنوان trycloudflare |

## 8. القسم `downloads`

| المفتاح | الافتراضي | الشرح |
|---|---|---|
| `huggingface_base` | `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main` | مصدر الأصول (إصدار RVC الرسمي) |
| `chunk_mb` | 8 | حجم مقطع التنزيل |
| `connect_timeout_s` | 30 | مهلة الاتصال |

(خريطة `files` الداخلية تربط المسارات البعيدة بمحلية `assets/` — hubert_base،
rmvpe.pt/onnx، وpretrained_v2 لكل المعدلات.)

## 9. القسم `logging`

| المفتاح | الافتراضي | الشرح |
|---|---|---|
| `level` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `max_bytes` | 10485760 | تدوير `logs/platform.log` (10 MB) |
| `backups` | 5 | عدد النسخ المدوّرة |

---

## 10. متغيرات البيئة (تتغلب على كل شيء)

| المتغير | القسم.المفتاح | مثال |
|---|---|---|
| `RVC_SERVER_HOST` | `server.host` | `0.0.0.0` |
| `RVC_SERVER_PORT` | `server.port` | `9000` |
| `RVC_AUTH_ENABLED` | `server.auth_enabled` | `true` |
| `RVC_AUTH_TOKEN` | `server.auth_token` | `mytoken` |
| `RVC_DEVICE` | `engine.device` | `cuda:0` |
| `RVC_F0_METHOD` | `engine.f0_method` | `fcpe` |
| `RVC_BLOCK_MS` | `realtime.block_ms` | `200` |
| `RVC_TRAIN_SR` | `training.sample_rate` | `48k` |
| `RVC_TUNNEL_ENABLED` | `tunnel.enabled` | `true` |
| `RVC_LOG_LEVEL` | `logging.level` | `DEBUG` |

مثال (Colab/سيرفر عام):

```bash
RVC_SERVER_HOST=0.0.0.0 RVC_AUTH_ENABLED=true RVC_BLOCK_MS=200 \
    uv run rvc-platform web --port 8320
```

القيم تُحوَّل تلقائيًا: `true/false` → bool، والأرقام → int/float.

---

## 11. إعادة توجيه المسارات

| المتغير | يعيد توجيه | الافتراضي |
|---|---|---|
| `RVC_PLATFORM_ROOT` | الجذر الكامل (مثل مجلد Drive في Colab) | جذر المستودع |
| `RVC_MODELS_DIR` | `models/` | `<root>/models` |
| `RVC_DATA_DIR` | `data/` | `<root>/data` |
| `RVC_ASSETS_DIR` | `assets/` | `<root>/assets` |
| `RVC_RUNS_DIR` | `runs/` | `<root>/runs` |
| `RVC_CACHE_DIR` | `.cache/` | `<root>/.cache` |

مثال Colab (كل شيء في Drive):

```bash
export RVC_PLATFORM_ROOT=/content/drive/MyDrive/RVC_Iraqi
uv run rvc-platform train start --experiment iraqi_voice
```

> ملاحظة: `logs/` و`.tmp/` يتبعان الجذر (`RVC_PLATFORM_ROOT`) دائمًا.

---

## 12. وصفات جاهزة (وصفات إعداد شائعة)

### GPU مخصص للتدريب والبث

```yaml
engine:
  device: cuda:0
  precision: fp16
realtime:
  block_ms: 160
training:
  sample_rate: "40k"
  batch_size: 6
```

### خدمة عامة مؤقتة (نفق + مصادقة)

```bash
make web TUNNEL=1        # يبدأ الويب + النفق (المصادقة تصبح إلزامية تلقائيًا)
cat logs/auth.token       # التوكن المولد
```

### وضع متصفح/عميل خارجي على الشبكة المحلية

```yaml
server:
  host: 0.0.0.0
  auth_enabled: true
  auth_token: "choose-a-long-random-token"
```

> لا تفتح `0.0.0.0` بدون مصادقة — واجهة الـAPI كاملة الصلاحيات على جهازك وبياناتك.

---

## 13. أين تُقرأ هذه القيم في الكود

| القسم | المخطط |
|---|---|
| كل الأقسام | `src/rvc_platform/infrastructure/config/schema.py` |
| التحميل والدمج والبيئة | `src/rvc_platform/infrastructure/config/loader.py` |
| المسارات وإعادة التوجيه | `src/rvc_platform/infrastructure/paths.py` |
| الافتراضيات المرفقة | `configs/default.yaml` |
