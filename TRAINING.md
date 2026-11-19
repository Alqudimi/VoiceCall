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

## 2.5 استيراد مجلد صوت خارجي وتجهيزه تلقائيًا

إذا كانت تسجيلاتك موزعة في مجلد خارجي وبصيغ مختلفة، استخدم وحدة الاستيراد الجديدة.
تبحث الوحدة داخل المجلد والمجلدات الفرعية، تفك الترميز عبر FFmpeg، تحول الصوت إلى
Mono بمعدل التدريب، تقص الصمت من البداية والنهاية مع هامش يحافظ على بدايات الكلمات،
تحذف التكرارات بعد فك الترميز، وتتجاوز الملفات التالفة أو الصامتة أو القصيرة. الناتج
يوضع تلقائيًا في `data/raw/<اسم_المجموعة>`، مع `import_report.json` و
`import_manifest.csv` يشرحان كل ملف تم قبوله أو استبعاده.

```bash
uv run rvc-platform prepare-audio-dataset /path/to/recordings my_voice --sr 40k
# أو عبر Make:
make prepare-audio-dataset SOURCE=/path/to/recordings NAME=my_voice SR=40k
```

الصيغ الشائعة المدعومة تشمل WAV وFLAC وMP3 وOGG/Opus وM4A/AAC وWMA وWebM وAIFF
وAMR وAPE وAC3 وDTS وMKA وغيرها مما يستطيع FFmpeg فك ترميزه. لا يستبدل الأمر
مجموعة موجودة تلقائيًا؛ استخدم `--force` أو `FORCE=--force` فقط إذا كنت متأكدًا.
بعد نجاحه شغّل `make data-check DATASET=data/raw/my_voice` ثم `make prepare-dataset`.

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
| `G_final.pth` / `D_final.pth` | لقطة الاكتمال (أعلى `global_step`) — ترتيبها في اختيار الاستكمال أعلى من المرقّمة |
| `G_2333333.pth` / `D_2333333.pth` | **مرآة "الأحدث دائمًا"** يُحدّث مع كل حفظ (متوافقة مع تسمية upstream) |
| `state.json` | `epoch`, `global_step`, **بصمة البيانات (dataset signature)**, `run_config`, ذيل الخسائر, زمن الحفظ |

كتابة كل `.pth` ذرية (`.tmp` ثم rename) حتى لا تفسد نقطة عند انقطاع الطاقة.

كذلك تُصدّر دوريًا نقاط استدلال أخف إلى `runs/<exp>/exports/*.pth` (كل
`save_every_epoch`) وتُكتب نتيجة نهائية `<exp>.pth` عند الاكتمال.

### الاستئناف

- **تلقائي**: إعادة `make train` / `train start --experiment <name>` بدون `--dataset`
  تحمّل آخر حالة كاملة وتكمل من مكانها (epoch/step/RNG/optimizer/scheduler/scaler).
- عبر الـAPI: `POST /api/training/resume?experiment=<name>` (بدونها تختار أحدث تجربة)
  ويدعم أيضًا **نقطة محددة**: `&checkpoint=G_48.pth` (اسم الملف أو مساره) أو
  `&checkpoint=latest`.
- **استكمال من نقطة محددة** (CLI):
  `rvc-platform train start --experiment <name> --resume-from G_48.pth`
  (يقبل اسم ملف نقطة، مسارًا كاملًا، أو الكلمة `latest`). زر "استكمال من هنا"
  متاح لكل صف في جدول نقاط الحفظ داخل واجهة الويب.
- **حفظ طارئ عند الانهيار**: أي استثناء غير متوقع داخل حلقة التدريب (OOM،
  دفعة فاسدة...) يحفظ checkpoint طارئًا كامل الحالة قبل الإبلاغ عن الفشل —
  يبقى التدريب قابلًا للاستكمال من آخر خطوة ناجحة.
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
   Drive بمجلد `RVC_Iraqi/`. يعمل التدريب افتراضيًا على `/content` المحلي الأسرع،
   ثم تنسخ خلية Snapshot النتائج المكتملة إلى Drive.
2. ارفع بياناتك (مثلاً إلى Drive ثم انسخها إلى `data/raw/`).
3. جهّز ودرّب:

```bash
uv run rvc-platform prepare-dataset data/raw/myvoice iraqi_voice --sr 40k --f0-method rmvpe
uv run rvc-platform train start --experiment iraqi_voice --epochs 200 --batch 6
```

4. عند انقطاع الجلسة (المقص الزمني في Colab حقيقي): شغّل خلية Snapshot قبل
   الانقطاع قدر الإمكان، ثم أعد تشغيل المثبّت — يستعيد آخر checkpoint متزامن من
   Drive إلى `/content` ويستأنف تلقائيًا. يتحقق Snapshot من ملفات `.pth` عبر
   `snapshot_manifest.json`.
5. صدّر النموذج وحمّله من `models/`.

ملاحظة صريحة: Colab تمنح **GPU واحدًا** بطبيعتها — وهذا متوافق مع قرار المنصة:
التدريب أحادي العملية/أحادي GPU (DDP متعدد GPU غير منفذ).

---

## 8. القيود والصراحة الهندسية

- **GPU واحد فقط** للتدريب؛ لا يوجد DDP. على CPU يعمل الكود لكن بطيئًا جدًا
  (احسب ساعات إلى أيام لكل epoch).
- افتراضيًا يستخدم DataLoader نفس سياسة المشروع السابقة: عاملان على CUDA وصفر
  على CPU، مع `pin_memory` و`non_blocking`، ولا يغيّر sampler أو ترتيب الدفعات.
  يمكن إجراء benchmark اختياري عبر `RVC_NUM_WORKERS` و`RVC_PREFETCH_FACTOR` و
  `RVC_PERSISTENT_WORKERS=1`، لكن زيادة هذه القيم قد تستهلك RAM في Colab.
- استخراج F0 وHuBERT يكتب ملفات NPY ذريًا عبر ملف `.part` ثم `os.replace`،
  ويتحقق من صحة الملفات قبل اعتبارها cache مكتملة؛ لذلك إعادة التشغيل بعد انقطاع
  لا تعتمد على ملف جزئي.
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


## 10. Hugging Face الخاص وإعادة التدريب من الصفر

أنشئ رمز Hugging Face بصلاحية الكتابة، واحفظه في بيئة خادم التدريب فقط:

```bash
export HF_TOKEN="hf_..."
# أو HUGGINGFACE_HUB_TOKEN
```

في الواجهة أدخل معرف المستودع بصيغة `owner/repository`، أو استخدم:

```bash
uv run rvc-platform train start --dataset data/raw/myvoice \
  --experiment my_voice --epochs 200 --hf-repo owner/private-voice
```

تُنشأ الوجهة كمستودع نموذج **خاص**، ويرفع التدريب أحدث نموذج مصدّر مع كل تصدير دوري، ثم يحدّثه بالتصدير النهائي، ويرفع أحدث فهرس FAISS فقط؛ لا تُرفع البيانات الصوتية أو الملفات الوسيطة أو الرمز. تظهر وصلة المستودع في قائمة التجارب، وفشل الرفع يُسجّل مستقلًا عن نتيجة التدريب.
إذا كان المستودع موجودًا وعامًا، يرفض النظام الرفع دون تغيير خصوصيته؛ حوّله إلى خاص في Hugging Face أو اختر مستودعًا خاصًا آخر.

لبدء تدريب كامل جديد على تجربة موجودة، فعّل «إعادة التدريب بالكامل من البداية» في الواجهة أو استخدم `--clean-start`:

```bash
uv run rvc-platform train start --dataset data/raw/myvoice \
  --experiment my_voice --clean-start --epochs 200
```

يمسح هذا الخيار checkpoints والتصديرات وحالة التقدم والسجلات القديمة، ويمنع أي استعادة تلقائية أو legacy أو نقطة `resume_from` قديمة؛ يحتفظ بملفات تجهيز البيانات كي لا يعيد استخراج الميزات دون حاجة. الخيار يُستهلك لمرة واحدة ثم يُحفظ التشغيل كقابل للاستئناف؛ لذلك زر «استكمال» بعد بدء التدريب النظيف سيكمل **التدريب الجديد** ولن يمسح نقاطه. الخيار غير متوافق مع `--resume-from`.
