# المساهمة — CONTRIBUTING

مرحبًا بكل مساهمة جادة: إصلاحات، تحسين أداء موثق بالقياس، توثيق أدق، اختبارات.
هذا المستند يشرح بيئة التطوير، الثوابت المعمارية غير القابلة للتفاوض، وسير المساهمة.

---

## 1. بيئة التطوير

```bash
git clone <repository-url> iraqi-voice-platform
cd iraqi-voice-platform
make install               # uv sync --all-groups (dev + training)
make check                 # فحص الاستيرادات والإصدارات
make test                  # كل الاختبارات
```

- Python **3.12 فقط**. `uv` مدير البيئة والحزم من `uv.lock` (قابل للتوليد بـ`make lock`).
- عجلات GPU ليست افتراضية:  `make env-gpu CUDA=cu128` عند الحاجة (انظر INSTALLATION.md).

### تشغيل الاختبارات

```bash
make test                 # pytest -q (الكل)
make test-unit            # tests/unit
make test-integration     # tests/integration
```

- اختبارات التكامل تشمل: تحميل checkpoint تخليقي، خطوات تدريب صغيرة على بيانات
  تخليقية، فحص دخاني للـAPI، وجلسة WebSocket تخليقية.
- العلامات: `@pytest.mark.gpu` (تُتخطى تلقائيًا بلا GPU فعلي) و`@pytest.mark.slow`
  (طويلة). لا تكتب اختبارًا يعتمد على أصول مدربة مسبقًا منزّلة — استخدم
  البيانات التخليقية.
- قاعدة الحجب: كل PR يجب أن يمر `make test` على CPU (المسار الافتراضي في CI/محليًا).

---

## 2. الثوابت المعمارية (اقرأها قبل أول PR)

البنية الكاملة موثقة في `ARCHITECTURE.md`. القواعد غير القابلة للتفاوض:

1. **جذر التركيب الوحيد:** `src/rvc_platform/api/app.py` هو المكان الوحيد الذي تُختار
   فيه التنفيذات الملموسة (Composition Root). أي شيء بعده يعتمد على الواجهات
   (`core/interfaces.py` Protocols) — لا استيراد تنفيذ مباشر من طبقة أدنى.

2. **الطبقات:** `core` (كيانات/أخطاء/أحداث/واجهات) ← `application` (خدمات حالات
   الاستخدام) ← `engine` / `training` / `realtime` / `infrastructure` ← `api` / `cli` /
   `web`. الاعتماد يتجه نحو الداخل فقط.

3. **لا تُخترع مكوّنات RVC من جديد:** الشبكات والخسائر وRMVPE وcheckpoint format هي
   **حساسة للتوافق** (كل اسم tensor وشكل ثابت جزء من صيغة RVC v1/v2). المكوّنات
   المضمّنة (vendored) من upstream بشهادة MIT موثقة في
   `src/rvc_platform/engine/rvc/ATTRIBUTION.md`. أي تعديل يجري على ملف مضمّن يجب أن
   يكون مدونًا هناك.

4. **توافق checkpoint:** قارئ النقاط يجب أن يستمر في قراءة نقاط upstream vanilla
   (`{model, optimizer, learning_rate, iteration}`) — لا تكتب صيغة تحطم التوافق.

5. **الأخطاء والكائنات:** نطاق أخطاء `core/errors.py` (مثل `ModelNotFoundError`,
   `DatasetError`, `TrainingError`, `RealtimeError`, `TunnelError`,
   `SecurityError`, `ConfigurationError`) — لا ترمِ استثناءات عامة من الخدمات؛
   والكيانات في `core/entities.py` (مثل `ModelInfo`, `ConversionResult`,
   `TrainingProgress`, `LatencyStats`).

6. **الأمان:** كل مسار `/api/*` جديد يمر عبر حارس المصادقة تلقائيًا (`app.py`) —
   استثناء واحد فقط مسموح نمطيًا (healthcheck). الملفات المرفوعة غير موثوقة:
   فحص صيغة/حجم قبل أي معالجة.

7. **الصدق في الواجهة:** لا تعد بأداء لا يقاس. كل رقم في الوثائق إما ثابت
   من upstream (منسوب) أو من `realtime-benchmark`/الاختبارات.

---

## 3. سير المساهمة

1. **Issue أولًا** للميزات الجديدة (مناقشة النطاق قبل الكود).
2. **فرع** بصيغة `feat/<slug>` أو `fix/<slug>` من `main`.
3. **Commit**: رسائل واضحة بصيغة `type: وصف قصير` (مثل
   `fix(trainer): guard resume against dataset signature change`).
4. **الاختبارات**: كل سلوك جديد يرافقه اختبار؛ الأخطاء المصلحة يرافقها
   اختبار انحدار (regression).
5. **التوثيق**: أي أمر/مسار/مفتاح إعداد جديد يجب أن يظهر في README.md/API.md/
   CONFIGURATION.md/INSTALLATION.md في نفس PR — الوثائق جزء من التعاقد، لا إضافة لاحقة.
6. **قبل الإرسال**:

```bash
make test
make verify        # إعادة إنتاج من uv.lock + فحص استيرادات
make doctor
```

7. **PR** يصف: ماذا/لماذا، القياسات (إن ادّعيت أداءً)، وكسر التوافق (إن وجد).

---

## 4. معايير الكود

- Python 3.12 حديث: type hints، `from __future__ import annotations`، dataclasses/
  pydantic كما في الكود القائم.
- Docstrings موجزة رأس الملف لكل وحدة (ماذا/لماذا/البروتوكول) — التقليد قائم في
  كل ملف ويتابع.
- لا أسرار في الكود أو الوثائق: التوكنات تُقرأ من `logs/auth.token` أو الإعداد.
- الاسماء: اللاتينية snake_case؛ لغة الـUI للويب عربية RTL — أي نص واجهة جديد
  عربي مع الحفاظ على المعرفات اللاتينية.
- لا إيموجي في الكود أو الوثائق.

### تحديث المكونات المضمّنة (vendoring)

```bash
python scripts/dev_vendor_rvc.py
```

السكربت idempotent يعيد نسخ الملفات من upstream المرجعي ويعيد ربط الاستيرادات —
بعد تشغيله راجع `ATTRIBUTION.md` وشغّل كل الاختبارات (توافق checkpoint أولًا).

---

## 5. إبلاغ عن أعطال (Bugs)

أرفق دائمًا:

1. إصدار المنصة: `uv run rvc-platform --help` (يظهر الإصدار) أو `make status`.
2. مخرجات `make doctor` كاملة.
3. مقتطف من السجل: `make logs LINES=100` أو `GET /api/logs?lines=100`
   (وللتدريب: `runs/<exp>/train.log` + `events.jsonl`).
4. الجهاز: GPU/VRAM/CPU، ونظام التشغيل، وPython.
5. خطوات إعادة إنتاج مصغرة.

للثغرات الأمنية: لا تفتح issue عامًا — تواصل مع المشرفين مباشرة (راجع ملف
التواصل في المستودع).

---

## 6. الأخلاقيات

المساهمة في هذه المنصة تعني قبول ميثاق الاستخدام: أدواتنا لتحويل الصوت بحقوق
معلومة فقط. لا نقبل مساهمات هدفها الانتحال أو التمويه أو التحايل على المصادقة.
انظر إشعار الاستخدام الأخلاقي في README.md.

---

## 7. خريطة الوثائق (حيث يجب أن يتحدث تغييرك)

| غيّرتَ | حدّث |
|---|---|
| مسار API/سلوك استجابة | API.md |
| مفتاح إعداد/متغير بيئة | CONFIGURATION.md |
| أمر CLI أو هدف make | README.md (+ INSTALLATION.md إن كان تثبيتًا) |
| سلوك التدريب/الاستئناف | TRAINING.md |
| بروتوكول البث/الكمون | REALTIME.md |
| مثبّت/سكربت تثبيت | INSTALLATION.md |
| عطل شائع جديد | TROUBLESHOOTING.md |
| ثابت معماري/بنية | ARCHITECTURE.md |

شكرًا مقدمًا — مساهمتك تجعل الصوت العراقي أفضل توثيقًا وأداءً.
