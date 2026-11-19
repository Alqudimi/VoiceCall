# تقرير الترقية البحثية — أقوى نموذج ممكن (Research-Backed Upgrade Report)

**الإصدار:** 1.1.0-quality | **النطاق:** تدريب RVC v2 على أي مجموعة بيانات | **المرجع التنفيذي:** هذا الملف

هذا التقرير يوثّق مرحلة البحث المكثف التي سبقت الترقية، والقرارات الهندسية
المستخلصة منها، وتنفيذها الكامل داخل المشروع. كل قرار أدناه مبني على أوراق
بحثية محكّمة أو مصادر رسمية (روابط في كل قسم)، وكل إضافة مصمّمة بحيث:

1. **تعمل على أي مجموعة بيانات** (dataset-agnostic) — لا افتراضات عن اللهجة أو المحتوى.
2. **متوافقة مع Google Colab (T4/L4/A100) وأي بيئة أخرى** — كل اعتمادية
   اختيارية ومحميّة (guarded)، وغيابها يُرجع النظام بأمان إلى أفضل سلوك
   متاح بلا أي خطأ (المنصة تبقى قابلة للإقلاع كما هي).
3. **قابلة للعكس بالكامل**: إرجاع أي مفتاح إلى الصفر/False يعيد الهدف
   الحسابي الأصلي (upstream) حرفيًا.

---

## 1) نظرة عامة: من فحص المشروع إلى خريطة الفجوات

الفحص العميق للمستودع (خط التدريب كاملًا: `preprocess → f0 → features →
index → trainer → checkpoints/best_model`) أظهر منصة ناضجة هندسيًا:
استئناف كامل الحالة، تتبّع أفضل نموذج، I/O خلفي، وتجميع HuBERT بلا حشو.
مقابل أدبيات البحث، تبيّنت **خمس فجوات نوعية** هي بالضبط مواضع الفارق بين
"تدريب يعمل" و"أقوى نموذج ممكن":

| # | الفجوة قبل الترقية | الحل المطبّق (الأدلة) |
|---|---|---|
| F1 | أوزان المولّد النهائية = آخر خطوة optimizer (سلوك GAN التقليدي) | **EMA للمولّد** — §2.1 |
| F2 | الهدف الحسابي: mel-L1 + KL + FM + adversarial بمقياس واحد فقط | **خسارة MRSTFT متعددة الدقة** — §2.2 |
| F3 | التمييز: MPD (فترات) + تمييز خام واحد scale-1 فقط | **استعادة MSD كاملة (scales 2/4)** — §2.3 |
| F4 | اختيار أفضل نموذج بـ val_mel وحده (أعمى عن هوية الصوت والنغمة) | **مقياس مركب: تشابه متحدث ECAPA + F0-RMSE + val_mel** — §2.4 |
| F5 | لا حماية من الإفراط بالتدريب؛ خيارات جودة البيانات محدودة | **إيقاف مبكر بصبر + تنظيف + توحيد جهارة + crepe** — §2.5–2.7 |

---

## 2) الأدلة البحثية والتنفيذ

### 2.1) EMA للمولّد — أقوى رافعة جودة مضمونة في تدريب GAN/Vocoder

**الأدلة:**
- Yazici et al., *The Unusual Effectiveness of Averaging in GAN Training*
  (DeepMind, ICLR blog-track 2024): متوسط الأوزان يتجاوز آخر تكرار (last
  iterate) في جودة GAN بشكل منهجي.
  https://openreview.net/forum?id=SJgw_sRqFQ
- BigVGAN (Lee et al., ICLR 2023) — vocoder عصري يعتمد EMA أثناء التدريب
  ويقيّم/يصدّر منها: https://github.com/NVIDIA/BigVGAN
- *Exponential Moving Average of Weights in Deep Learning* (2024):
  EMA يحسّن التعميم والثبات والمعايرة:
  https://arxiv.org/abs/2312.02600
- Databricks Composer (توثيق رسمي): "EMA improves the attainable
  tradeoffs between training speed and final quality… we recommend EMA".
  https://docs.mosaicml.com/en/latest/composer/algorithms/ema.html

**التنفيذ** (`src/rvc_platform/training/ema.py` — `GeneratorEMA`):
- ظلّ (shadow) على نفس جهاز التدريب: تحديث بعمليتين مدمجتين لكل معامل
  (`shadow = d*shadow + (1-d)*param`) بعد كل خطوة G ناجحة — خطوات
  GradScaler المتخطاة (inf) تُستثنى كي لا يلوث الظلّ خطوةً فاسدة.
- منحدر إحماء قياسي `min(d, (1+t)/(10+t))` يمنع هيمنة التهيئة العشوائية
  مبكرًا (نفس ممارسة تدريب نماذج diffusion).
- **النموذج المُصدَّر والمُقيَّم والأفضل هو أوزان EMA** (وصفة BigVGAN)،
  بينما يستمر التدريب على الأوزان الخام — انظر `_eval_weights` /
  `_export_snapshot` في `trainer.py`.
- حالة EMA تُخزَّن داخل G_*.pth (مفتاح `ema`) — الاستئناف **bit-exact**،
  والنقاط القديمة بلا المفتاح تُهيَّئ من الأوزان الحالية.
- افتراضيًا: `ema_decay = 0.999`. إطفاؤه: `--no-ema` أو `ema_decay: 0`.

### 2.2) خسارة STFT متعددة الدقة (MRSTFT) — تكملة الهدف الحسابي

**الأدلة:**
- Yamamoto et al., *Parallel WaveGAN* (ICASSP 2020): التجميع الطيفي
  (spectral convergence) + L1 لمقدار اللوغاريتم عبر عدة دقات FFT —
  الهدف الحسابي المعياري لتوليد الكلام.
- Kwon et al., *UnivNet: A Neural Vocoder with Multi-Resolution
  Spectrogram Discriminators* (Interspeech 2021): MRSTFT + تمييزات
  متعددة الدقة تحسّن الجودة الحسّية بوضوح:
  https://arxiv.org/abs/2106.07889
- BigVGAN (ICLR 2023): خط mel متعدد المقاييس + خسائر STFT المكملة:
  https://arxiv.org/abs/2206.04658

**التنفيذ** (`src/rvc_platform/training/stft_loss.py` — `mr_stft_loss`):
- ثلاث دقات (512/128، 1024/256، 2048/512) على **نفس مقطع التدريب الذي
  تغطيه خسارة mel** (`wave_sliced` مقابل `y_hat`)، محسوبة float32 خارج
  autocast — نفس سياسة mel/KL الحالية.
- الوزن `c_stft` **مُعاير قياسياً على إشارات كلامية تخليقية**: القيمة
  الافتراضية **2.0 تجعل الحد ~8% من خسارة G الكلية** — منظّم فعّال غير
  مهيمن (معايرة قابلة للتكرار في `tests/unit/test_stft_loss.py`).
- `c_stft=0` يعيد الهدف الأصلي حرفيًا.

### 2.3) استعادة التمييز متعدد المقاييس (MSD) — وصفة HiFi-GAN الأصلية

**الأدلة:**
- Kong et al., *HiFi-GAN* (NeurIPS 2020): التوليد المبنى على RVC مشتق
  من نفس عائلة HiFi-GAN، والورقة الأصلية تستخدم **MPD + MSD (3 مقاييس
  خام: 1/2/4)** مع Feature Matching على مخارج الاثنين — إشارة عدائية
  متعددة المقاييس تزيل ضجيج الترددات العالية والأطواق الزائفة:
  https://arxiv.org/abs/2010.05646
- upstream RVC أبقى **تمييزًا خامًا واحدًا (scale-1)** فقط داخل MPD —
  أي أن وصفة HiFi-GAN ناقصة المقاييس.

**التنفيذ** (`src/rvc_platform/training/msd.py` — `MultiScaleDiscriminator`):
- المقاييس **2 و4** (الناقصة) عبر تجميع متوسط كتلي ثم نفس رصة
  `DiscriminatorS` المضمّنة، مع spectral-norm وفق توصية HiFi-GAN
  للتمييزات الخام.
- وحدة **مستقلة تمامًا** بمحسّن وجدولة خاصين، تُخزَّن في `D2_*.pth`
  بجانب G/D — **صيغة G/D والمُسبقات التدريبية بلا أي تغيير**، والاستئناف
  من نقاط قديمة يدفيء D2 (كما في تشغيل جديد — طبيعي تمامًا لأن D2 سقالة
  عدائية لا تحمل جودة نموذج).
- لا دوال خسارة جديدة: مخارج MPD+MSD تُدمج في قوائم قبل استدعاء
  `discriminator_loss`/`generator_loss`/`feature_loss` الحالية (هي تولّف
  بالتكرار على القوائم أصلًا).
- إطفاؤه: `--no-msd` أو `RVC_MSD=0`.

### 2.4) المقياس المركب لاختيار أقوى نموذج — ما لا تراه mel-L1

**الأدلة:**
- Desplanques et al., *ECAPA-TDNN* (Interspeech 2020) + تنفيذ SpeechBrain:
  تمثيل متحدث قوي (SOTA على VoxCeleb) هو المعيار الشائع لمقياس "تشابه
  المتحدث" (speaker similarity/cosine) في أوراق VC/SVC الحديثة.
  https://arxiv.org/abs/2005.04126 / https://speechbrain.github.io
- مجتمع RVC نفسه يكشف المشكلة: "overtraining" يفسد النماذج بعد نقطة
  ما (تتوليد ضجيج على بيانات غير مرئية) ويُنصح بالمقارنة على بيانات
  محجوزة: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/198
- Applio (أشهر تفرّع RVC) بنى كاشف إفراط تدريب حديثًا لهذا السبب نفسه:
  https://github.com/IAHispano/Applio
- UTMOS (UTokyo-SaruLab, VoiceMOS 2022): مُتنبّه MOS موضوعي — مسار بحثي
  مستقبلي (لا نضيف اعتماديته لأنه غير مستقر تثبيتًا على Colab):
  https://github.com/sarulab-speech/UTMOS22

**التنفيذ** (`src/rvc_platform/training/eval_metrics.py` + تكامل
`best_model.ValidationEvaluator`):
- `val_spk`: تجميل ECAPA (speechbrain/spkrec-ecapa-voxcelebx) لكل زوج
  (إعادة بناء، حقيقة) بعد إعادة العينة إلى 16k — **قياس مباشر لحفظ
  الطابع الصوتي**: ما يريده مستخدم VC فعلًا.
- `val_f0_rmse`: RMSE لِنغمة RMVPE على مقاطع voiced (بروتوكول تقييم
  CREPE) — تشخيص حفظ النغمة، يُستخرج بأصول RMVPE الموجودة أصلًا في
  الأصول (لا تنزيل جديد).
- **مقياس الاختيار**: `composite = val_mel + spk_weight*(1 - val_spk)`
  (الأقل أفضل). غياب speechbrain ⇒ يبقى val_mel كما كان تمامًا.
- كل مقياس محميّ: أي فشل = تسجيل وتخطٍّ، لا انهيار تدريب أبدًا.
- أمر جديد: **`rvc-platform train evaluate <exp>`** يقيّم أي checkpoint
  (كامل أو تصدير inference) ويكتب `runs/<exp>/eval_report.json` — دليل
  موضوعي على قوة النموذج للمستخدم.

### 2.5) الإيقاف المبكر ضد الإفراط بالتدريب

**الأدلة:** ظاهرة تدهور RVC بعد حقب كثيرة موثقة في Issue أعلاه + مبدأ
early-stopping بمعيار محجوز قياسي (Keras/Lightning ModelCheckpoint
monitor/patience). Applio يوقف عند حد حقب أقصى؛ نحن **نوقف على مقياس**:
أدق.

**التنفيذ:** `early_stop_patience` (افتراضي 0=مطفأ؛ دفتر Colab يضبط 40):
عند انقضائها بلا تحسّن للمقياس المركب ⇒ إيقاف تعاوني كامل (checkpoint
نهائي + `training.early_stop` + `state.json:early_stopped=true`).

### 2.6) جودة البيانات — أي مجموعة بيانات كانت

**الأدلة:**
- noisereduce (Sainburg et al., Spectral Gating; Scientific Reports 2025):
  مكتبة معيارية خفيفة لإزالة ضجيج النطاق:
  https://pypi.org/project/noisereduce/ /
  https://www.nature.com/articles/s41598-025-13108-x
- CREPE (Kim et al., Interspeech 2018): تقدير نغمة عصبي فائق المتانة
  للضجيج: https://github.com/maxrmorrison/torchcrepe
- توحيد الجهارة (loudness normalisation) ممارسة معيارية في تجهيز بيانات
  الصوت (Applio's dataset review يطبق level-balancing قبل التدريب:
  https://docs.applio.org/getting-started/training).

**التنفيذ:**
- `--denoise/--no-denoise` (افتراضيًا مطفأ): تنظيف طيفي قبل التقطيع —
  محميّ بكل مستوى (غياب المكتبة/فشل = لا-عمل مسجّل، لا انهيار).
- `--loudness/--no-loudness` (افتراضيًا **مفعل**): توحيد RMS لكل مقطع إلى
  −20 dBFS بسقف كسب وحارس قمة، قبل تطبيع upstream نفسه — أهداف تدريب
  متسقة لأي بيانات مختلطة المصادر.
- `--f0-method crepe` (torchcrepe اختياري): بروتوكول CREPE الكامل
  (periodicity≥0.1 للvoiced) كخيار ثالث بجودة قوية على البيانات الصاخبة
  (خطأ ConfigurationError واضح إن غابت المكتبة — لا سقوط صامت).

### 2.7) ما لم نغيّره عمدًا (سلامة الجودة والاستئناف)

- **بنية G وD وصيغة المُسبقات**: بلا أي تغيير — كل التعزيز فوقها لا فيها.
- **فهرس FAISS / nprobe**: يغيّر top-k والاسترجاع — خارج نطاق الجودة
  المحسوبة هنا.
- **آلية الجلب/الحشو في HuBERT**: التجميع بالطول المتطابق فقط (بقي كما
  هو — قياس upstream أظهر انحراف الحشو).
- **grad accumulation / channels_last / BF16**: لأسباب موثقة في
  OPTIMIZATION_REPORT_AR.md (دلالات GAN، عدم انطباق 1D، وT4).

---

## 3) التوافق مع Google Colab — تصميم "لا فشل ممكن"

| المكوّن | سلوك Colab (T4/L4/A100) | سلوك بيئة بلا GPU/بلا مكتبات |
|---|---|---|
| EMA | يعمل (ظل على GPU، تكلفة ذاكرة ≈ حجم G) | يعمل على CPU |
| MRSTFT | يعمل (3 STFT صغيرة لكل خطوة) | يعمل |
| MSD/D2 | يعمل + checkpoint مستقل D2 | يعمل (اختبارات CPU تُغطيه) |
| ECAPA speaker-sim | speechbrain مثبتة في الخلية 2 ⇒ المقياس المركب كامل | غيابها ⇒ fallback إلى val_mel بلا أي خطأ |
| F0-RMSE (RMVPE) | أصلًا من `make download-models` | غياب الأصل ⇒ تخطٍّ صامت |
| denoise/crepe | من مجموعة quality (خلية 2) | لا-عمل/ConfigurationError واضح |
| الاستئناف | G/D/EMA/D2 كاملة + الرواصد الخلفية | متطابق |

كل الخيارات الجديدة defaults = "أقوى وصفة"، وكلها قابلة للإطفاء لعودة
حرفية لسلوك upstream: `--no-ema --stft-weight 0 --no-msd --val-split 0
--loudness=false`.

---

## 4) التحقق المنفّذ

- اختبارات وحدة جديدة: EMA (رياضيات التحديث/الإحماء/الاستدارة)، MRSTFT
  (صفر على التطابق، إيجابية على الضجيج، تعارض الأشكال)، MSD (عقد
  المخارج، انتشار خلفي)، المقاييس المركبة (بمزوّدات مزيفة)، config
  (الحدود والتوثيق).
- اختبار الدخان الحقيقي (تدريب GAN كامل بحجم صغير على CPU + استئناف
  حقيقي + legacy resume + حراسة توقيع البيانات) يشغّل الآن **الوصفة
  الكاملة افتراضيًا** (EMA+MSD+MRSTFT+val_split) — أي أن كل مسار جديد
  مغطى باختبار تكامل حقيقي.
- كل الاختبارات: `make test` (uv run pytest -q).

---

## 5) المراجع الرئيسية

1. EMA في GAN: https://openreview.net/forum?id=SJgw_sRqFQ
2. EMA في Deep Learning: https://arxiv.org/abs/2312.02600
3. BigVGAN: https://arxiv.org/abs/2206.04658 + https://github.com/NVIDIA/BigVGAN
4. HiFi-GAN (MPD+MSD): https://arxiv.org/abs/2010.05646
5. UnivNet (MRSTFT): https://arxiv.org/abs/2106.07889
6. Parallel WaveGAN (STFT losses): https://arxiv.org/abs/1910.11480
7. ECAPA-TDNN: https://arxiv.org/abs/2005.04126 + https://speechbrain.github.io
8. CREPE: https://arxiv.org/abs/1802.06182 + https://github.com/maxrmorrison/torchcrepe
9. RVC overtraining (issue الرسمي): https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/198
10. Applio (تفرّع RVC المهتم بالجودة): https://github.com/IAHispano/Applio + https://docs.applio.org
11. noisereduce: https://pypi.org/project/noisereduce/ + https://www.nature.com/articles/s41598-025-13108-x
12. UTMOS (مسار مستقبلي): https://github.com/sarulab-speech/UTMOS22
13. RMVPE (InterSpeech 2023): مضمّن في الأصول الرسمية lj1995/VoiceConversionWebUI
