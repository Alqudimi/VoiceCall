# تقرير تحسين سرعة تدريب VoiceCall

## ما تم تطبيقه

حافظت النسخة المحسّنة على بنية RVC v2، وAMP/FP16، وAdamW، وجدولة التعلم، وحجم الدفعة، و`DistributedBucketSampler`، وFAISS `IVFFlat`، وصيغة checkpoints. هذه العناصر بقيت كما هي لأنها جزء من مسار الجودة أو من دلالات الاستئناف.

تمت إضافة إعدادات اختيارية إلى DataLoader من خلال متغيرات البيئة: `RVC_NUM_WORKERS` و`RVC_PREFETCH_FACTOR` و`RVC_PERSISTENT_WORKERS`. بعد دمج تحسينات التسريع الكاملة (انظر `SPEEDUP.md`) صارت القيم الافتراضية سريعة تلقائيًا: عدد العمال يتكيف مع الأنوية (عاملان على Colab المجاني T4، وحتى 6 على L4/A100)، العمال دائمون بين الحقب (لا إعادة fork كل epoch)، و`prefetch_factor=4`، مع إمكانية العودة للسلوك القديم حرفيًا عبر `RVC_NUM_WORKERS=2 RVC_PERSISTENT_WORKERS=0 RVC_PREFETCH_FACTOR=2`. كما أصبحت `zero_grad(set_to_none=True)` صريحة لتقليل الذاكرة وتحسين وضوح المسار.

أصبحت ملفات F0 وHuBERT تُكتب إلى ملف مؤقت مع `fsync` ثم `os.replace`. قبل تخطي أي cache، يتم فتح ملف NPY والتحقق من أنه مقروء وغير فارغ وقيمه finite. هذا لا يغيّر القيم، لكنه يمنع أن يتحول انقطاع Colab أثناء الكتابة إلى cache تالفة تبدو مكتملة.

غيّر مثبت Colab استراتيجية التخزين: تتم استعادة runs وmodels من Drive إلى `/content`، ويحدث التدريب وقراءة الميزات والكتابة المرحلية على القرص المحلي. يبقى Drive وجهة النسخة الدائمة، وتُشغّل خلية `colab snapshot` بعد إيقاف التدريب تعاونيًا. ينسخ snapshot الملفات عبر ملفات `.part`، ويكتب `snapshot_manifest.json` يحوي الحجم وSHA-256 للـcheckpoints والنماذج والفهارس.

## ما فُعّل بعد الدمج وما بقي معطّلًا عمدًا

بعد الدمج فُعّلت `torch.compile` للمولّد والمميّز افتراضيًا على CUDA مع ضمانات أمان تعالج المخاوف الأصلية: `suppress_errors=True` يعيد التنفيذ إلى eager تلقائيًا عند أي فشل ترجمة (لا يمكن أن يكسر تدريبًا)، `cache_size_limit=32` يستوعب أشكال الـ buckets المتغيرة دون ترجمات لا نهائية، و`RVC_TORCH_COMPILE=0` يعطّلها كليًا. كذلك فُعّل TF32 عبر `set_float32_matmul_precision("high")` وfused AdamW على CUDA: الأول لا يعمل على T4 أصلًا ولا يمسّ مسار fp16 autocast الرئيسي، والثاني نفس خوارزمية AdamW حرفيًا في نواة مدمجة واحدة وفق توثيق PyTorch الرسمي.

بقي معطلًا عمدًا: BF16 (غير مدعوم على T4)، و`channels_last` (لا ينطبق على الالتفافات الصوتية 1D)، وgradient accumulation (يغيّر عدد تحديثات GAN وجدولة التعلم حتى لو بقي الحجم الفعّال متشابهًا)، والتجميع مع حشو أطوال مختلفة في HuBERT (قياس فعلي أظهر انحرافًا كبيرًا؛ التجميع بالطول المتطابق فقط).

لم أغيّر `nprobe` أو نوع فهرس FAISS، ولم أضف FAISS GPU إلى الاعتماديات. هذه تغييرات قد تؤثر في top-k والمسافات وجودة التحويل وليست تحسينات I/O محايدة.

## تشغيل اختبارات الأداء الاختيارية

على GPU، ابدأ بالقيم الافتراضية ثم قارن تشغيلًا قصيرًا على نفس البيانات:

```bash
RVC_NUM_WORKERS=2 RVC_PREFETCH_FACTOR=2 uv run rvc-platform train start --experiment iraqi_voice
```

إذا كانت RAM كافية، يمكن تجربة:

```bash
RVC_NUM_WORKERS=4 RVC_PREFETCH_FACTOR=2 RVC_PERSISTENT_WORKERS=1 \
  uv run rvc-platform train start --experiment iraqi_voice
```

لا تعتمد الخيار الثاني في Colab قبل مراقبة RAM وVRAM وزمن انتظار البيانات. إذا ظهر OOM أو لم يرتفع استخدام GPU، ارجع إلى القيم الافتراضية.

## القيود

لا يمكن إثبات رقم تسريع مطلق داخل هذه الحاوية لأنها لا تملك GPU تدريبيًا أو بيانات صوتية حقيقية. كما أن Google Colab لا يضمن نوع GPU أو مدة الجلسة أو حدود الاستخدام، لذلك يجب تسجيل اسم GPU ونسخة Torch/CUDA وقياس warm-up وmedian وp95 داخل كل جلسة.

## المراجع

[1]: https://docs.pytorch.org/docs/stable/amp.html "PyTorch Automatic Mixed Precision"
[2]: https://docs.pytorch.org/docs/stable/data.html "PyTorch Data Loading"
[3]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html "PyTorch pinned memory and non-blocking transfers"
[4]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html "PyTorch torch.compile"
[5]: https://docs.pytorch.org/docs/stable/notes/randomness.html "PyTorch reproducibility"
[6]: https://research.google.com/colaboratory/faq.html "Google Colab FAQ"
[7]: https://github.com/facebookresearch/faiss/wiki/FAQ "FAISS FAQ"
[8]: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU "FAISS on the GPU"
