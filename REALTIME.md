# البث اللحظي — REALTIME

توثيق البث اللحظي (تحويل الصوت أثناء الكلام) في Iraqi Voice Studio: المعمارية،
بروتوكول WebSocket إطارًا بإطار، قياس زمن الاستجابة بصدق، وقيود المتصفح.

---

## 1. متى يعمل البث اللحظي (بصراحة)

البث اللحظي **ليس** متاحًا على أي جهاز. يعمل عندما تتحقق كل هذه الشروط:

| الشرط | التفصيل |
|---|---|
| GPU من فئة CUDA | RTX وأفضل؛ زمن استدلال الخادم لكل مقطع يجب أن يبقى **أقل من `block_ms`** |
| نموذج f0 (pitch-guided) | النماذج بلا f0 ترفض الجلسة بخطأ واضح |
| اتصال مستقر | المستعرض ↔ الخادم بلا قفزات؛ عبر النفق يزيد زمن الشبكة |
| اتصال WebSocket | من تبويب "البث اللحظي" في الويب أو أي عميل مخصص |

على **CPU فقط**: زمن الاستدلال يتجاوز ميزانية المقاطع (160ms افتراضيًا) ويصبح
الصوت متقطعًا — استخدم **تحويل الملفات** بدل البث (انظر TRAINING.md وREADME.md).

**أرقام مرجعية من upstream (كما يعلنها مشروع RVC الأصلي، وليست قياساتنا):**
~170ms شاملًا (end-to-end) على GPU مناسب، و~90ms مع تعريفات ASIO — على Windows فقط.
لا نعد بأرقام مماثلة على جهازك: **قِس بنفسك** (القسم 6).

---

## 2. المعمارية

```
[المتصفح]  getUserMedia -> AudioWorklet
     مقاطع Int16 أحادية القناة @ 16 kHz، كل block_ms (160ms افتراضيًا)
          |
          |  WebSocket  ws://host:8320/api/realtime/ws?token=<token>
          |  TEXT frames: تحكم   |   BINARY frames: صوت
          v
[الخادم]  RealtimeSession  ->  RealTimeRVC engine (لكل اتصال محركه الخاص)
     نافذة انزلاقية: lookback 200ms (سياق F0/تاريخ) + crossfade 8ms + search + block
     HuBERT محتوى + F0 بمخزن pitch طوله 1024 إطار (استمرارية النغمة)
     skip_head/return_length: فقط المقطع الجديد يُبعث
     SOLA crossfade على الإخراج (بحث overlap-add بين المقاطع)
     بوابة ضجيج دخل -40 dB + مزج غلاف RMS
          |
          v
  Int16 PCM أحادية القناة @ معدل النموذج (tgt_sr) — يعاد إلى المتصفح إطارًا ثنائيًا
          |
          v
[المتصفح]  تشغيل فوري (PlaybackStream)
```

الاستدلال يعمل في خيط عامل (worker thread) عبر جسر asyncio-to-thread، فلا يعلّق
حلقة الأحداث، ويقاس زمن كل مقطع فعليًا (min/avg/p95/max).

---

## 3. إعدادات البث (من configs/default.yaml → realtime)

| المفتاح | الافتراضي | المدى | المعنى |
|---|---|---|---|
| `block_ms` | 160 | 60..500 | حجم مقطع الصوت؛ 160ms ≈ ميزانية upstream الافتراضية |
| `lookback_ms` | 200 | 100..1000 | سياق F0/التاريخ مع كل مقطع |
| `crossfade_ms` | 8 | 0..40 | مدة مزج SOLA بين المقاطع |
| `silence_threshold_db` | -40 | -70..-20 | بوابة ضجيج: المقاطع الأدنى من هذا تُتجاوز |
| `keep_alive_s` | 60 | 5..600 | إبقاء الجلسة حية |
| `f0_method` | rmvpe | rmvpe/fcpe/pm | مستخرج النغمة للبث |

موازنة صريحة: تقليل `block_ms` يقلل الكمون الظاهري لكنه يضغط ميزانية GPU
(يجب أن ينزل زمن الاستدلال معه)؛ زيادته تزيد الكمون وتزيد سلاسة الاحتمال على
GPU أضعف. غيرها عبر `configs/user.yaml` أو `RVC_BLOCK_MS` (CONFIGURATION.md).

---

## 4. بروتوكول WebSocket إطارًا بإطار

### الاتصال

```
ws://127.0.0.1:8320/api/realtime/ws                # بدون مصادقة مفعلة
ws://127.0.0.1:8320/api/realtime/ws?token=<TOKEN>  # عند تفعيل المصادقة
```

**قيود المتصفح الأساسية:** لا يمكن إرسال ترويسات HTTP مخصصة من `WebSocket` في
المتصفح — لذلك التوكن يُمرر **في سلسلة الاستعلام `?token=`** فقط (الخادم يرفض
برمز إغلاق `4401` عند فشل التحقق). توثيق المصادقة الكامل في API.md.

### إطارات TEXT من العميل إلى الخادم

| الإطار | الحمولة | الوصف |
|---|---|---|
| `start` | `{"type":"start","model":"<name>","settings":{...}}` | يبدأ/يعيد تحميل محرك الجلسة بالنموذج و الإعدادات الأولية |
| `settings` | `{"type":"settings","settings":{...}}` | تحديث حي للإعدادات (بلا إعادة تحميل) |
| `ping` | `{"type":"ping"}` | نبضة؛ الرد `pong` مع عدد المقاطع |
| `stats` | `{"type":"stats"}` | طلب إحصاءات زمن الجلسة الحالية |

الإعدادات المسموح تحديثها: `pitch_shift`, `index_rate`, `f0_method`, `rms_mix_rate`,
`silence_threshold_db`.

### إطارات TEXT من الخادم إلى العميل

| الإطار | الحمولة | الوصف |
|---|---|---|
| `started` | `{"type":"started","sample_rate":<tgt_sr>,"block_ms":<n>,"channels":1,"format":"int16","model":"<name>","latency_budget_ms":<n>}` | تأكيد بدء الجلسة |
| `settings_applied` | `{"type":"settings_applied"}` | تم تطبيق التحديث |
| `pong` | `{"type":"pong","blocks":<n>}` | رد النبضة |
| `stats` | `{"type":"stats", ...latency fields...}` | إحصاءات الجلسة |
| `error` | `{"type":"error","message":"..."}` | خطأ (نموذج غير موجود/بلا f0/JSON سيئ/إطار تحكم مجهول) |

### إطارات BINARY

- **العميل → الخادم:** Int16 mono PCM @ **16 kHz** بالضبط، بمقاطع بحجم `block_ms`
  (2560 عينة عند 160ms). يجب إرسال `{"type":"start"}` أولًا وإلا يأتي إطار
  `error` بالنص `"send {type:start} first"`.
- **الخادم → العميل:** Int16 mono PCM @ **معدل النموذج** (`sample_rate` من إطار
  `started`)، بمقاطع مصطفة على البلوك، تُشغَّل فورًا.

### مثال JS مبسّط (pseudo-snippet)

```javascript
const ws = new WebSocket("ws://127.0.0.1:8320/api/realtime/ws?token=" + token);
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "start",
    model: "iraqi_voice",
    settings: { pitch_shift: 0, index_rate: 0.0, f0_method: "rmvpe" }
  }));
};

ws.onmessage = (ev) => {
  if (typeof ev.data === "string") {
    const msg = JSON.parse(ev.data);
    if (msg.type === "started") {
      // msg.sample_rate / msg.block_ms / msg.format === "int16"
      startCapture(msg.block_ms);        // AudioWorklet: Int16 @16kHz
      startPlayback(msg.sample_rate);    // AudioContext بمعدل النموذج
    }
    // "error" / "settings_applied" / "pong" / "stats" ...
  } else {
    playInt16(ev.data);                  // بايتات Int16 PCM → تشغيل فوري
  }
};

// داخل AudioWorklet: كل block_ms اجمع 2560 عينة Int16 وأرسلها
ws.send(int16Buffer.buffer);             // BINARY frame → الخادم

// قياس RTT من جهة العميل (اختياري): أرسل ping وسجّل زمن pong
ws.send(JSON.stringify({ type: "ping" }));
```

(هذا مقطع توضيحي للبروتوكول؛ التنفيذ الفعلي في واجهة الويب يتكفل بالتقاط
AudioWorklet والتشغيل. متصفحك لا يسمح بترويسات مخصصة على WebSocket — لذلك `?token=`.)

---

## 5. نقاط REST المساندة

| المسار | الوظيفة |
|---|---|
| `GET /api/realtime/info` | جاهزية البث (يتطلب نموذج f0 محمّلًا) + الإعدادات الحية + ملاحظة البروتوكول |
| `GET /api/realtime/stats` | إحصاءات مجمّعة لكل الجلسات + لكل جلسة (انظر القسم 6) |

التفصيل الكامل والـcurl في [API.md](API.md).

---

## 6. قياس زمن الاستجابة (Latency) بصدق

### ما الذي يُقاس فعلًا

- **زمن الخادم لكل مقطع** (server-side): من استلام مقطع Int16 إلى تجهيز الناتج —
  محفوظة لآخر 512 مقطع لكل جلسة، وتُعرض `min_ms / avg_ms / p95_ms / max_ms`.
- **جسر الخيط** (`bridge_overhead_ms`): زمن استدعاء `process` الكامل من حلقة asyncio.
- **RTT من العميل** (`client_rtt_p95_ms`): ما يبلّغه المتصفح (ping→pong) — اختياري.

المصدران: `GET /api/realtime/stats`، أو إطار `{"type":"stats"}` داخل الجلسة.

### المعيار الذهبي: بدون ميكروفون

`make realtime MODEL=<name>` يشغّل **المحرك الحقيقي** بمقاطع تخليقية (موجة جيبية
150Hz مع تغير سعة + ضجيج) — لا حاجة لميكروفون ولا متصفح:

```bash
make realtime MODEL=iraqi_voice BLOCKS=40 BLOCK_MS=160
# = uv run rvc-platform realtime-benchmark iraqi_voice --blocks 40 --block-ms 160
```

الناتج JSON يضم: `device`, `block_ms`, `samples`, `min_ms`, `avg_ms`, `p95_ms`,
`max_ms`, و`verdict` الذي يصدر الحكم بلا مجاملة:

```json
{
  "verdict": "real-time OK (avg < block budget)"          // نجاح
}
```

أو:

```json
{
  "verdict": "WARNING: avg 214 ms exceeds block budget 160 ms"   // CPU مثالًا
}
```

### كيف تقرأ الأرقام (بلا خداع)

| القياس | المعنى الصادق |
|---|---|
| `avg_ms` من المقياس | زمن استدلال الخادم فقط — هذا ما يجب أن يبقى < `block_ms` |
| RTT المتصفح | يضيف شبكة + إدخال/إخراج المتصفح؛ أعلى من زمن الخادم دائمًا |
| 170ms (upstream) | رقم معلن من مشروع RVC الأصلي شاملًا على GPU مناسب — **مرجع، لا وعد** |
| 90ms ASIO (upstream) | مع تعريفات ASIO على Windows فقط — غير متاح عبر المتصفح |

لتقليص الكمون على GPU مُتوسط: ارفع `block_ms` مؤقتًا (حتى 500)، أو استخدم
`fcpe` بدل `rmvpe` (أسرع، أقل دقة نغميًا)، وتأكد أن الجلسة فريدة (كل اتصال يبني
محركه الخاص — جلستان على GPU ضعيف تتقاسمان العتاد).

---

## 7. البث عبر النفق العام

يمكن بث الجلسة عبر نفق Cloudflare (راجع API.md / TROUBLESHOOTING.md):
`POST /api/tunnel/start` ثم اربط WebSocket بعنوان trycloudflare مع `?token=`.
كن واعيًا: زمن الشبكة عبر النفق يضاف إلى RTT، والعنوان مؤقت (يتغير مع كل تشغيل)،
والمصادقة تصبح إلزامية تلقائيًا.

---

## 8. قصص فشل معروفة

| العرَض | السبب الشائع | الحل |
|---|---|---|
| صوت متقطع/متأخر جدًا | `avg_ms` ≥ `block_ms` (GPU غير كافٍ أو CPU) | قِس بـ`make realtime`؛ استخدم تحويل الملفات أو GPU أقوى |
| إغلاق WS برمز 4401 | توكن خاطئ/مفقود في `?token=` | انسخ التوكن من `logs/auth.token` أو إعداداتك |
| `error: send {type:start} first` | إرسال بايت قبل `start` | أرسل إطار `start` أولًا |
| `error: real-time conversion requires a pitch-guided (f0) model` | النموذج بلا f0 | درّب/حمّل نموذج f0 (TRAINING.md) |
| نغمة "تقفز" بين المقاطع | F0 غير مستقر أو `fcpe` على صوت خافت | جرّب `rmvpe`، ورفع `lookback_ms` |

المزيد: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
