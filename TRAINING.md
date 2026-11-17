# التدريب — TRAINING

دليل تدريب نموذج صوتي جديد بـ RVC v2 عبر Iraqi Voice Studio: من البيانات الخام إلى
ملف `.pth` جاهز في `models/`.

---

## 1. قبل أن تبدأ: بيانات جيدة = نموذج جيد

المنصة لا "تخلق" لهجة عراقية؛ هي تتعلمها من بياناتك. RVC يحوّل الطابع الصوتي
(timbre) فقط — النص والنطق يبقيان من صوت المتحدث المصدر وقت التحويل.

| العامل | الحد الأدنى | الموصى به |
|---|---|---|
| مدة الصوت الصافي | 10 دقيقة | 10–60 دقيقة |
| المحتوى | كلام واحد المتحدث بلا خلط أصوات | كلام عراقي نظيف متنوع (نبرات مختلفة) |
| الجودة | بلا ضجيج ثقيل/صدى/موسيقى خلفية | تسجيل واحد ثابت الميكروفون والمستوى |
| الصيغ | wav/mp3/flac/ogg/opus/m4a/aac/webm/aiff | WAV أو FLAC بمدة 16kHz+ غير ضرورية — يُعاد الترميز |

قواعد عملية:
- ملفات قصيرة عديدة أفضل من ملف واحد ضخم (التقطيع يتم داخليًا على أي حال).
- تجنب الضجيج والموسيقى؛ ضجيج الخلفية يتعلمه النموذج.
- لا تخلط أكثر من متحدث واحد في التجربة (المتحدث الواحد هو الهدف).
- **موافقة صاحب الصوت إلزامية أخلاقيًا وقانونيًا قبل التدريب على صوته.**

---

## 2. فحص البيانات (اختياري لكن مستحسن)

```bash
make data-check DATASET=data/raw/myvoice
# أو: uv run rvc-platform data-check data/raw/myvoice
```

التقرير يضم: عدد الملفات/الصالحة، الملفات الفاسدة، غير المدعومة، **التكرارات عبر
md5**، المدد (كلية/أدنى/أقصى)، والملفات الفارغة. إن كانت المدة أقل من 10 دقائق
تحذّرك المنصة صراحة من تدهور الجودة.

---

## 3. خط تجهيز البيانات (Pipeline)

```bash
make prepare-dataset DATASET=data/raw/myvoice EXPERIMENT=iraqi_voice SR=40k F0_METHOD=rmvpe
```

المكافئ CLI: `uv run rvc-platform prepare-dataset data/raw/myvoice iraqi_voice --sr 40k --f0-method rmvpe`

المراحل بالترتيب:

| # | المرحلة | ماذا تفعل | الناتج في `runs/<exp>/` |
|---|---|---|---|
| 1 | `check` | فحص الفساد والتكرار (md5) وإحصاء المدد وإعطاء توصية | تقرير على الشاشة |
| 2 | `preprocess` | تقطيع واعٍ للصمت (Slicer)، مقاطع 3.7s بتراكب 0.3s، مرشح highpass 48Hz (order 5)، تطبيع ذروة (alpha 0.75 / max 0.9) | `0_gt_wavs/` (معدل الهدف) + `1_16k_wavs/` (16kHz) |
| 3 | `f0` | استخراج النغمة: `rmvpe` (الأفضل) \| `fcpe` (الأسرع) \| `pm` (CPU) | `2a_f0/` + `2b-f0nsf/` (`.wav.npy`) |
| 4 | `features` | HuBERT 768-dim للمحتوى | `3_feature768/` (`.npy`) |
| 5 | `index` | فهرس FAISS للميزات | `added_IVF<n>_Flat_nprobe_1_<name>_<version>.index` |
| 6 | `filelist` | قائمة العينات `gt\|feature\|f0\|f0nsf\|spk` | `filelist.txt` |
| — | config | كتابة إعداد التشغيل | `config.json` + `run_config.json` |

المراحل كلها idempotent (يمكن إعادة تشغيلها). لتشغيل مرحلة منفردة على تجربة قائمة:

```bash
make preprocess DATASET=... EXPERIMENT=... SR=40k
make extract-features EXPERIMENT=iraqi_voice F0_METHOD=rmvpe
```

خيارات المعدل: `SR=32k|40k|48k` (افتراضي 40k). `F0_METHOD` للبيانات: `rmvpe|fcpe|pm`
(المشروع لا يشحن harvest/dio/crepe).

### شكل مجلد التجربة

```
runs/iraqi_voice/
├── config.json          hyper-parameters الفعلية للتدريب
├── run_config.json      إعداد التشغيل القابل للاستئناف
├── filelist.txt         قائمة العينات
├── 0_gt_wavs/  1_16k_wavs/  2a_f0/  2b-f0nsf/  3_feature768/
├── added_IVF....index   فهرس FAISS
├── checkpoints/         G_*.pth, D_*.pth, state.json
├── exports/             نقاط تصدير دورية *.pth + النهائي <name>.pth
├── train.log  progress.json  events.jsonl
```

---

## 4. تشغيل التدريب

```bash
make train   # = uv run rvc-platform train start --dataset data/raw/iraqi_dataset \
#              --experiment iraqi_voice --epochs 200 --batch 6 --sr 40k
```

عبر CLI (يتحكم بأكثر من خيار):

```bash
uv run rvc-platform train start \
    --dataset data/raw/myvoice \
    --experiment iraqi_voice \
    --epochs 200 --batch 6 --sr 40k \
    --lr 1e-4 --save-every 10 --f0-method rmvpe
```

| الخيار | الافتراضي | المدى/المعنى |
|---|---|---|
| `--dataset` | — | مسار البيانات (لبدء تجربة جديدة) |
| `--experiment / -e` | `iraqi_voice` | اسم التجربة (مجلد `runs/<name>`) |
| `--sr` | `40k` | `32k / 40k / 48k` |
| `--epochs` | `200` | 1..10000 |
| `--batch` | `6` | 1..64 — لـ VRAM 6–8GB جرّب 6؛ لـ 24GB يمكنك 12 |
| `--lr` | `1e-4` | معدل تعلم AdamW |
| `--save-every` | `10` | حفظ كل N epochs |
| `--gpu` | — | فهرس GPU صريح |
| `--f0-method` | `rmvpe` | طريقة F0 لاستخراج ميزات البيانات |
| `--no-pretrained` | — | بدون القواعد المدربة v2 (تدريب من الصفر — غير موصى به) |
| `--clean-start` | — | تجاهل أي checkpoints قائمة وابدأ من جديد |

- عبر **CLI** يعمل التدريب في العملية نفسها لعرض الأحداث الحية (JSONL) على الشاشة.
- عبر **API/Web** يعمل كعملية فرعية منفصلة:
  `python -m rvc_platform.training.worker --config runs/<name>/run_config.json`.
- المتغيرات عبر make: `EXPERIMENT`, `DATASET`, `SR`, `F0_METHOD`, `EPOCHS`, `BATCH`.

### خوارزمية التدريب (GAN) بثوابت upstream

- بنية v2 مع f0 (الافتراضي) — أو v1 عبر الإعداد (لا يوصى به).
- المولّد/المميّز بـ **AdamW** بـ betas `(0.8, 0.99)` وeps `1e-9`، lr `1e-4`.
- جدولة **ExponentialLR** بمعامل decay `0.999875` لكل خطوة.
- **AMP fp16** على GPU المؤهل (GradScaler)؛ fp32 على CPU.
- الخسائر الأصلية لـ RVC (mel + KL + GAN losses) — نفس أرقام upstream
  (انظر `configs/training/40k.json` لقيم `c_mel/c_kl` …).

---

## 5. نقاط الحفظ والاستئناف (Resume)

في `runs/<exp>/checkpoints/`:

| الملف | المعنى |
|---|---|
| `G_<step>.pth` / `D_<step>.pth` | لقطات كاملة الحالة: `model`, `optimizer`, `scheduler`, `scaler`, `epoch`, `global_step`, `rng` — للـ Generator/Discriminator |
| `G_2333333.pth` / `D_2333333.pth` | **مرآة "الأحدث دائمًا"** يُحدّث مع كل حفظ (متوافقة مع تسمية upstream) |
| `state.json` | `epoch`, `global_step`, **بصمة البيانات (dataset signature)**, `run_config`, ذيل الخسائر, زمن الحفظ |

كتابة كل `.pth` ذرية (`.tmp` ثم rename) حتى لا تفسد نقطة عند انقطاع الطاقة.

كذلك تُصدّر دوريًا نقاط استدلال أخف إلى `runs/<exp>/exports/*.pth` (كل
`save_every_epoch`) وتُكتب نتيجة نهائية `<exp>.pth` عند الاكتمال.

### الاستئناف

- **تلقائي**: إعادة `make train` / `train start --experiment <name>` بدون `--dataset`
  تحمّل آخر حالة كاملة وتكمل من مكانها (epoch/step/RNG/optimizer/scheduler/scaler).
- عبر الـAPI: `POST /api/training/resume` (بصيغة `?experiment=` اختيارية — بدونها
  تختار أحدث تجربة).
- **تعارض بصمة البيانات**: إذا تغيّرت البيانات (ملفات/محتوى) عن التي دربّت عليها
  النقطة، ترفض المنصة الاستئناف بخطأ واضح يطلب إما **تغيير اسم التجربة** أو
  `--clean-start` — هذا يمنع استئنافًا فاسدًا بصمتًا.
- **توافقية upstream**: نقاط RVC الخام (`G_/D_` بصيغة upstream الأصيلة
  `{model, optimizer, learning_rate, iteration}`) قابلة للاستئناف أيضًا عبر مسار
  legacy يقرأ منها ما توفر.

### الإيقاف التعاوني

```bash
make train-stop        # = uv run rvc-platform train stop
# أو: kill -TERM <pid>
```

`SIGTERM` يعني: أكمل الخطوة الحالية، احفظ checkpoint، اخرج برمز 0. أكواد الخروج:
`0` نجاح/إيقاف، `2` خطأ إعداد، `3` فشل تشغيلي.

### مراقبة التقدم

- CLI الحي: أحداث JSONL (`training.progress`, `training.started`, `training.resumed`,
  `training.finished`, `training.failed`).
- ملفات: `progress.json` (ذرية، دائمًا حديثة) و`events.jsonl` (سجل تراكمي) و`train.log`.
- الـAPI: `GET /api/training/status` و`GET /api/training/runs/{exp}/events`.
- لوحة الويب: تبويب "التدريب".
- TensorBoard (مع مجموعة training): سجلات التدريب تكتب منحنيات الخسائر.

---

## 6. التصدير إلى models/

```bash
make export EXPERIMENT=iraqi_voice           # آخر تصدير
uv run rvc-platform train export iraqi_voice --epoch 180   # تصدير epoch محدد
```

التصدير ينسخ نقطة من `exports/` إلى `models/` ويسجلها في السجل، ومعها أحدث
`added_IVF*.index` إن وُجد (يُنسخ بجانب النموذج). بعدها يظهر النموذج في:

```bash
make model-list           # uv run rvc-platform model list
make model-info MODEL=iraqi_voice    # بيانات checkpoint وفقرة
```

وسيكون متاحًا للتحميل في الويب/API والاستخدام في التحويل والبث.

---

## 7. التدريب على Google Colab

سير مقترح (GPU واحد ممنوح من Colab — T4/L4/A100):

1. نفّذ مثبّت Colab (انظر INSTALLATION.md القسم 4) — يجهز البيئة والأصول ويربط
   Drive بمجلد `RVC_Iraqi/`.
2. ارفع بياناتك (مثلاً إلى Drive ثم انسخها إلى `data/raw/`).
3. جهّز ودرّب:

```bash
uv run rvc-platform prepare-dataset data/raw/myvoice iraqi_voice --sr 40k --f0-method rmvpe
uv run rvc-platform train start --experiment iraqi_voice --epochs 200 --batch 6
```

4. عند انقطاع الجلسة (المقص الزمني في Colab حقيقي): أعد تشغيل المثبّت — يستأنف
   تلقائيًا من آخر checkpoint متزامن في Drive.
5. صدّر النموذج وحمّله من `models/`.

ملاحظة صريحة: Colab تمنح **GPU واحدًا** بطبيعتها — وهذا متوافق مع قرار المنصة:
التدريب أحادي العملية/أحادي GPU (DDP متعدد GPU غير منفذ).

---

## 8. القيود والصراحة الهندسية

- **GPU واحد فقط** للتدريب؛ لا يوجد DDP. على CPU يعمل الكود لكن بطيئًا جدًا
  (احسب ساعات إلى أيام لكل epoch).
- **البث اللحظي** (انظر REALTIME.md) يتطلب GPU من فئة RTX ونموذج f0.
- **جودة اللهجة العراقية** تعتمد كليًا على نظافة بياناتك وكميتها؛ المنصة توفر
  الأدوات (RMVPE افتراضيًا، `protect` يحرس الصوات العراقية كـ /q/ الم emphatic)،
  لكن البيانات هي الحد الأقصى للجودة.
- العناية بصمة البيانات: أي تعديل في البيانات بعد بدء التدريب يستوجب اسم تجربة
  جديدًا أو `--clean-start`.

---

## 9. من دفتر المراجع

- الإعداد الكامل للمفاتيح: [CONFIGURATION.md](CONFIGURATION.md) — قسم `training`.
- API التدريب (prepare/start/stop/resume/export): [API.md](API.md).
- أعطال التدريب الشائعة: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
