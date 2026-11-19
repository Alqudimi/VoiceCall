# تقرير تحسين سرعة تدريب VoiceCall

## ما تم تطبيقه

حافظت النسخة المحسّنة على بنية RVC v2، وAMP/FP16، وAdamW، وجدولة التعلم، وحجم الدفعة، و`DistributedBucketSampler`، وFAISS `IVFFlat`، وصيغة checkpoints. هذه العناصر بقيت كما هي لأنها جزء من مسار الجودة أو من دلالات الاستئناف.

تمت إضافة إعدادات اختيارية إلى DataLoader من خلال متغيرات البيئة: `RVC_NUM_WORKERS` و`RVC_PREFETCH_FACTOR` و`RVC_PERSISTENT_WORKERS=1`. القيم الافتراضية مطابقة للمشروع السابق، لذلك لا يتغير السلوك دون طلب صريح. كما أصبحت `zero_grad(set_to_none=True)` صريحة لتقليل الذاكرة وتحسين وضوح المسار.

أصبحت ملفات F0 وHuBERT تُكتب إلى ملف مؤقت مع `fsync` ثم `os.replace`. قبل تخطي أي cache، يتم فتح ملف NPY والتحقق من أنه مقروء وغير فارغ وقيمه finite. هذا لا يغيّر القيم، لكنه يمنع أن يتحول انقطاع Colab أثناء الكتابة إلى cache تالفة تبدو مكتملة.

غيّر مثبت Colab استراتيجية التخزين: تتم استعادة runs وmodels من Drive إلى `/content`، ويحدث التدريب وقراءة الميزات والكتابة المرحلية على القرص المحلي. يبقى Drive وجهة النسخة الدائمة، وتُشغّل خلية `colab snapshot` بعد إيقاف التدريب تعاونيًا. ينسخ snapshot الملفات عبر ملفات `.part`، ويكتب `snapshot_manifest.json` يحوي الحجم وSHA-256 للـcheckpoints والنماذج والفهارس.

## ما لم أفعّله عمدًا

لم أفعّل `torch.compile` أو CUDA Graphs أو `max-autotune` افتراضيًا، لأن أطوال RVC متغيرة، وحلقة GAN تحتوي قصًا وأقنعة وتبديلًا بين المولّد والمميّز. قد تؤدي هذه الميزات إلى إعادة ترجمة كثيرة أو استهلاك VRAM أو اختلافات عددية.

لم أفعّل TF32 أو BF16 أو fused AdamW أو `channels_last` أو gradient accumulation. TF32 مناسب للتجارب على L4/A100 فقط وقد يغيّر المسار العددي، ولا يعمل كتحسين مفيد على T4. أما gradient accumulation فيغيّر عدد تحديثات GAN وجدولة التعلم حتى لو بقي الحجم الفعّال متشابهًا.

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
