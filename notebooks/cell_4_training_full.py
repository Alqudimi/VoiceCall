# @title 4) التدريب الكامل — كل خيارات المنصة متاحة (يُستكمل تلقائيًا من آخر Checkpoint)
# مهم: نمرر دائمًا --dataset حتى تُطبَّق كل القيم أدناه فعليًا حتى على الجلسات
# المستكملة (بدونه يتجاهل الأمر epochs/batch/lr وغيرها ويقرأها من ملف قديم).
# كل الأعلام مُتحقَّق منها 1:1 ضد CLI المنصة (rvc-platform train start --help).
import json, os, shlex

EXPERIMENT = "iraqi_voice"  # @param {type:"string"} (نفس اسم خلية التجهيز 3)
DATASET_IN_DRIVE = "Voice"  # @param {type:"string"} (نفس مسار خلية 3)
BASEPATH = "/content/drive/MyDrive/"  # @param ["/content/drive/MyDrive/", "/content/"]

# ═══════════════ الإعدادات الأساسية ═══════════════
EPOCHS = 300  # @param {type:"integer"} (300 لبيانات ~40 دقيقة؛ FAQ: 50-200 نظيفة، 200-500 متوسطة)
BATCH = 6  # @param {type:"integer"} (T4 FP32 @ 48k + MRD + Prior + ECAPA: 6 آمن ومستقر ~14 GB؛ خفّض إلى 4 إن ظهرت CUDA OOM)
LR = 0.0001  # @param {type:"number"} (معدل التعلم — لا ترفعه مع القواعد المسبقة)
SAVE_EVERY = 10  # @param {type:"integer"} (احفظ نقطة تحقق كل N حقبة)
F0_METHOD = "rmvpe"  # @param ["rmvpe", "fcpe", "pm", "crepe"] (مقفول على ما جُهّز؛ rmvpe الأقوى)
GPU_INDEX = ""  # @param {type:"string"} (فارغ = اختيار تلقائي)
RESUME_FROM = ""  # @param {type:"string"} (اختياري: G_12600.pth أو latest — فارغ = الأحدث تلقائيًا)
CLEAN_START = False  # @param {type:"boolean"} (امسح كل نقاط التحقق وابدأ من الصفر!)
NO_PRETRAINED = False  # @param {type:"boolean"} (تخطي القاعدة المسبقة — ليس موصى به)

# ═══════════ المرحلة 1: رافعات الجودة (UPGRADE_REPORT §1) ═══════════
EMA_DECAY = 0.999  # @param {type:"number"} (متوسط أوزان المولد؛ 0 = تعطيل)
STFT_WEIGHT = 2.0  # @param {type:"number"} (خسارة STFT متعددة الدقة؛ 0 = الهدف الأصلي)
USE_MSD = True  # @param {type:"boolean"} (تمييزات متعددة المقاييس — HiFi-GAN)
SPK_WEIGHT = 1.0  # @param {type:"number"} (وزن تشابه المتحدث ECAPA في اختيار الأفضل)
EARLY_STOP = 30  # @param {type:"integer"} (حقب صبر قبل الإيقاف؛ 0 = تعطيل)
VAL_SPLIT = 0.10  # @param {type:"number"} (نسبة التحقق المحجوز للتقييم الموضوعي؛ 0 = تقييم على بيانات التدريب)
SAVE_BEST = True  # @param {type:"boolean"} (تتبّع وتصدير أفضل نموذج ونشره إلى best/ على HF)
# ملاحظة: save_latest / save_every_weights / f0_eval مفعّلة افتراضيًا في config
# (تُعدّل فقط بتحرير runs/<exp>/run_config.json — ليست أعلام CLI).

# ═══════════ المرحلة 3: محسنات الاستنساخ الفائق (SOTA Voice Cloning) ═══════════
USE_MRD = True  # @param {type:"boolean"} (مميز STFT متعدد الدقة UnivNet/BigVGAN لمنع الرنين المعدني)
MS_MEL_WEIGHT = 10.0  # @param {type:"number"} (خسارة Multi-Scale Mel من 4 مقاييس — BigVGAN-v2)
PRIOR_MEL_WEIGHT = 8.0  # @param {type:"number"} (مطابقة مسار الاستدلال enc_p -> flow -> dec أثناء التدريب)
SPK_LOSS_WEIGHT = 2.0  # @param {type:"number"} (خسارة هوية المتحدث ECAPA-TDNN القابلة للتفاضل)
FEAT_NOISE = 0.015  # @param {type:"number"} (تنظيم ضوضاء HuBERT لمنع تحفظ المحتوى وحذف الشوائب)
AUGMENT = True  # @param {type:"boolean"} (تعزيز لحظي: ضجيج/جهارة/حذف — مهم للبيانات الصغيرة/الصاخبة)
WARMUP = 1  # @param {type:"integer"} (حقب إحماء خطي لمعدل التعلم؛ 0 = تعطيل)
COSINE = 0.1  # @param {type:"number"} (ذيل جيب التمام؛ 0 = تعطيل، 0.1 موصى به)
D_LR_SCALE = 1.0  # @param {type:"number"} (1.0 = مطابق upstream المُتحقَّق رسميًا؛ 0.9 يبطئ المميّز إن انهار مبكرًا)
CLIP_GRAD_NORM = 500  # @param {type:"number"} (قص تدريج المولد ضد الانهيار — BigVGAN §3.4؛ 0 = تعطيل = upstream)

# ═══════════ أداء Colab ═══════════
TORCH_COMPILE = "off"  # @param ["auto", "on", "off"] (تسريع torch.compile — auto يفعّله على T4/L4 فقط)
NUM_WORKERS = 2  # @param {type:"integer"} (عمّال تحميل البيانات؛ 2-4 لـColab)
PERSISTENT_WORKERS = True  # @param {type:"boolean"} (إبقاء العمّال أحياء بين الحقب — يسرّع الحقب القصيرة)
PREFETCH_FACTOR = 4  # @param {type:"integer"} (دفعات محمّلة مسبقًا لكل عامل؛ 2-8)
MAX_VAL_BATCHES = 64  # @param {type:"integer"} (سقف دفعات التقييم كل حقبة — 8-16 يسرّع الحقب الطويلة جدًا)

# ═══════════ رفع HuggingFace ═══════════
HF_REPO = "eng7mi/Iraqi"  # @param {type:"string"} (مستودع خاص owner/name لرفع النماذج؛ فارغ = تعطيل)
HF_UPLOAD_ATTEMPTS = 5  # @param {type:"integer"} (محاولات رفع HF مع إعادة محاولة أسية)
IO_FLUSH_TIMEOUT = 900  # @param {type:"integer"} (مهلة كتابة checkpoint بالثواني قبل إنقاذ طارئ)

# ═══════════ تشخيص ما قبل التدريب ═══════════
RUN_DATA_CHECK = True  # @param {type:"boolean"} (فحص SNR/صمت/مدة البيانات قبل البدء — لا يعيد أي مرحلة)
EVAL_AFTER_TRAIN = True  # @param {type:"boolean"} (تقييم موضوعي شامل بعد انتهاء التدريب: val_mel + val_spk + F0-RMSE)

# -------- تثبيت SR وF0 تلقائيًا من الجلسة المجهَّزة (يمنع عدم التوافق) --------
_run_cfg = f"/content/iraqi-voice-platform/runs/{EXPERIMENT}/run_config.json"
if os.path.isfile(_run_cfg) and not CLEAN_START:
    _saved = json.load(open(_run_cfg))
    F0_METHOD = _saved.get("f0_method", F0_METHOD)
    SR = _saved.get("sample_rate", "48k")
    print(f">>> جلسة موجودة: نثبت SR={SR} وF0={F0_METHOD} كما جُهّزت (لا تغيّرهما إلا بإعادة التجهيز)")
else:
    SR = "48k"  # @param ["32k", "40k", "48k"]
    print(f">>> جلسة جديدة: SR={SR} (تجهيز جديد)")

os.environ["UV_NO_SYNC"] = "1"
os.environ["RVC_TORCH_COMPILE"] = TORCH_COMPILE
os.environ["RVC_NUM_WORKERS"] = str(NUM_WORKERS)
os.environ["RVC_PERSISTENT_WORKERS"] = "1" if PERSISTENT_WORKERS else "0"
os.environ["RVC_PREFETCH_FACTOR"] = str(PREFETCH_FACTOR)
os.environ["RVC_MAX_VAL_BATCHES"] = str(MAX_VAL_BATCHES)
os.environ["RVC_HF_UPLOAD_ATTEMPTS"] = str(HF_UPLOAD_ATTEMPTS)
os.environ["RVC_IO_FLUSH_TIMEOUT"] = str(IO_FLUSH_TIMEOUT)

dataset_path = f"{BASEPATH}{DATASET_IN_DRIVE}"
q = shlex.quote
flags = [
    f"--experiment {q(EXPERIMENT)}",
    f"--dataset {q(dataset_path)}",
    f"--sr {SR}", f"--epochs {EPOCHS}", f"--batch {BATCH}", f"--lr {LR}",
    f"--save-every {SAVE_EVERY}", f"--f0-method {q(F0_METHOD)}",
    # المرحلة 1
    f"--ema-decay {EMA_DECAY}", f"--stft-weight {STFT_WEIGHT}",
    f"--spk-weight {SPK_WEIGHT}", f"--early-stop {EARLY_STOP}",
    f"--val-split {VAL_SPLIT}",
    "--save-best" if SAVE_BEST else "--no-save-best",
    "" if USE_MSD else "--no-msd",
    # المرحلة 2
    "--augment" if AUGMENT else "--no-augment",
    f"--warmup {WARMUP}", f"--cosine {COSINE}", f"--d-lr-scale {D_LR_SCALE}",
    f"--clip-grad-norm {CLIP_GRAD_NORM}",
    # المرحلة 3 (SOTA)
    "" if USE_MRD else "--no-mrd",
    f"--ms-mel-weight {MS_MEL_WEIGHT}",
    f"--prior-mel-weight {PRIOR_MEL_WEIGHT}",
    f"--spk-loss-weight {SPK_LOSS_WEIGHT}",
    f"--feat-noise {FEAT_NOISE}",
    # متنوعة
    f"--gpu {GPU_INDEX.strip()}" if GPU_INDEX.strip() else "",
    "--clean-start" if CLEAN_START else "",
    "--no-pretrained" if NO_PRETRAINED else "",
    f"--resume-from {q(RESUME_FROM)}" if RESUME_FROM.strip() else "",
    f"--hf-repo {q(HF_REPO)}" if HF_REPO.strip() else "",
]
cmd = "cd /content/iraqi-voice-platform && uv run rvc-platform train start " + " ".join(f for f in flags if f)

# فحص بيانات اختياري (لا يعيد تجهيزًا — تقرير فقط)
precheck = ""
if RUN_DATA_CHECK:
    precheck = (f"cd /content/iraqi-voice-platform && uv run rvc-platform data-check "
                f"{q(dataset_path)} || true")

# تقييم موضوعي بعد التدريب (أفضل نموذج تلقائيًا)
postcheck = ""
if EVAL_AFTER_TRAIN:
    postcheck = (f"cd /content/iraqi-voice-platform && uv run rvc-platform train evaluate "
                 f"{q(EXPERIMENT)} || true")

print(">>>", precheck or "(تخطي الفحص)")
print(">>>", cmd)
if postcheck:
    print(">>>", postcheck)
