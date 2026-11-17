# التثبيت — INSTALLATION

دليل تثبيت Iraqi Voice Studio (`rvc-platform` v1.0.0) على Linux (Ubuntu)، Windows،
وGoogle Colab، مع إعداد GPU وتنزيل الأصول المدربة مسبقًا.

---

## 1. المتطلبات

| المتطلب | التفصيل |
|---|---|
| نظام التشغيل | Ubuntu LTS / Windows 10+ / macOS / Google Colab (Linux) |
| Python | **3.12 فقط** (`>=3.12,<3.13`) — لا تدعم المنصة 3.11 أو 3.13 |
| مدير البيئة | `uv` (يدير البيئة الافتراضية والاعتماديات من `uv.lock`) |
| GPU (اختياري) | CUDA-compatible (فئة RTX مفضلة للتدريب والبث اللحظي) |
| ffmpeg (اختياري) | لفك ترميز MP3/M4A/OGG/OPUS/WebM/AAC — WAV/FLAC تعمل بدونه |
| cloudflared (اختياري) | للنفق العام المؤقت |
| مساحة قرص | ~2 GB أصول مدربة مسبقًا + مساحة بياناتك ونماذجك |

**بصراحة:** بدون GPU يعمل كل شيء لكن التدريب بطيء جدًا والبث اللحظي غير متاح.
مع GPU يصبح التدريب عمليًا والبث ممكنًا (انظر REALTIME.md).

---

## 2. Ubuntu — المثبّت الرسمي

المثبّت `scripts/ubuntu/install_ubuntu.sh` يفحص: إصدار Ubuntu، المعمارية، `python3.12`،
`uv`، `nvidia-smi` وCUDA، `ffmpeg`، و`build-essential`، ثم يثبّت الناقص عبر `apt`
ويجهز البيئة:

```bash
git clone <repository-url> iraqi-voice-platform
cd iraqi-voice-platform
bash scripts/ubuntu/install_ubuntu.sh
```

أو من خارج المستودع (بعد الاستنساخ):

```bash
cd iraqi-voice-platform
bash scripts/ubuntu/install_ubuntu.sh            # بيئة CPU (افتراضي torch CPU)
bash scripts/ubuntu/install_ubuntu.sh --with-assets   # + تنزيل الأصول المدربة مسبقًا (اختياري)
```

المثبّت:
1. يتحقق من Ubuntu المدعوم والمعمارية (amd64/arm64).
2. يتحقق من `python3.12` و`uv` ويثبّت الناقص من مستودع النظام أو astral.sh.
3. يفحص GPU: `nvidia-smi` وإصدار CUDA (اختياري — يستمر بدون GPU مع تحذير صريح).
4. يثبّت `ffmpeg` و`build-essential` إن كانا ناقصين (via `apt`).
5. ينشئ البيئة: `uv sync --all-groups`.
6. مع `--with-assets` ينفذ `rvc-platform download-models`.

بعد التثبيت، لو لديك NVIDIA GPU فعّل عجلات CUDA (القسم 5).

---

## 3. Windows — المثبّت الرسمي

هناك ملفان: `scripts/windows/install_windows.ps1` (سكربت PowerShell) و
`scripts/windows/install_windows.cmd` (غلاف cmd يستدعي PowerShell):

```bat
:: من موجه الأوامر (cmd) بعد الاستنساخ:
cd iraqi-voice-platform
scripts\windows\install_windows.cmd
```

```powershell
# أو مباشرة من PowerShell:
cd iraqi-voice-platform
powershell -ExecutionPolicy Bypass -File scripts\windows\install_windows.ps1
```

السكربت يفحص: Windows 10+، المعمارية، Python 3.12، `uv`، `ffmpeg`، `cloudflared`،
وGPU (اختياري). **بدون NVIDIA GPU** يستمر التثبيت على وضع CPU، ويُوثَّق في السكربت
أن البث اللحظي لن يعمل وأن التدريب سيكون بطيئًا — التحويل عبر الملفات يبقى متاحًا.

على Windows استخدم CLI بدل make (لا يوجد Makefile افتراضيًا):

```powershell
uv run rvc-platform download-models
uv run rvc-platform web
```

---

## 4. Google Colab — التدريب السحابي

المثبّت `scripts/colab/install_colab.sh` مصمم للخلية الأولى في Colab:

```bash
curl -fsSL <repo>/scripts/colab/install_colab.sh | bash
```

> ملاحظة مهمة: عنوان السكربت `<repo>` يعتمد على مكان استضافتك للمستودع — على
> GitHub مثلًا يكون بصيغة raw:
> `https://raw.githubusercontent.com/<user>/<repo>/main/scripts/colab/install_colab.sh`.
> استبدل العنوان بما يوافق مستودعك.

ما يفعله المثبّت:
1. يكتشف بيئة Colab وGPU وCUDA تلقائيًا (T4/L4/A100).
2. يستنسخ/ينزّل المشروع ويثبّت `uv` ثم `uv sync --all-groups` مع عجلات CUDA.
3. ينزّل الأصول المدربة مسبقًا (`download-models`).
4. يربط Google Drive: مجلد `RVC_Iraqi/` للتجربة، عبر `RVC_PLATFORM_ROOT` و`drive_sync`
   (مزامنة نقاط الحفظ إلى Drive حتى لا تضيع عند انقطاع الجلسة).
5. يستأنف تلقائيًا من آخر checkpoint موجود عند إعادة التشغيل.

يوجد أيضًا دفتر جupyter مكافئ: `notebooks/RVC_Training_Colab.ipynb` — نفس الخطوات
في خلايا قابلة للتشغيل. تفاصيل التدريب في TRAINING.md (قسم Colab).

---

## 5. التثبيت اليدوي (أي منصة)

```bash
# 1) uv
curl -LsSf https://astral.sh/uv/install.sh | sh          # Linux/macOS
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2) المشروع
git clone <repository-url> iraqi-voice-platform
cd iraqi-voice-platform

# 3) البيئة: كل المجموعات (dev: pytest/pytest-asyncio/psutil، training: tensorboard/scikit-learn/matplotlib)
make install                    # = uv sync --all-groups
# أو مباشرة:
uv sync --all-groups

# 4) تحقق
make check                      # = rvc-platform install check
make doctor                     # = rvc-platform install doctor  (فحص GPU/CUDA/الأصول)
```

عجلات torch الافتراضية هي **CPU** (قرارة تصميم: كل بيئة تقلع حتى بدون GPU). مجموعة
`onnx` اختيارية: `uv sync --extra onnx` (تشمل `onnxruntime`).

### تنزيل الأصول المدربة مسبقًا

```bash
make download-models                              # CLI
uv run rvc-platform download-models                # مكافئ CLI
uv run rvc-platform install doctor --fix           # فحص + تنزيل الناقص
```

الأصول تُجلب من إصدار HuggingFace الرسمي `lj1995/VoiceConversionWebUI`:

| الأصل | المسار المحلي بعد التنزيل |
|---|---|
| HuBERT (صيغة transformers) | `assets/hubert_base/` (`config.json`, `preprocessor_config.json`, `pytorch_model.bin`) |
| RMVPE (مستخرج النغمة) | `assets/rmvpe/rmvpe.pt` (و`rmvpe.onnx` أيضًا) |
| القواعد المدربة v2 (40k) | `assets/pretrained_v2/f0G40k.pth` + `f0D40k.pth` |
| القواعد المدربة v2 (32k/48k) | `assets/pretrained_v2/f0G32k.pth/f0D32k.pth` و`f0G48k.pth/f0D48k.pth` |

التنزيل يدعم استئناف البايت (byte-range) والكتابة الذرية (`.part` ثم rename)،
ويعمل في الخلفية عبر API: `POST /api/assets/download` مع متابعة عبر `GET /api/assets`.

---

## 6. إعداد GPU (عجلات CUDA)

الافتراضي CPU. للتبديل إلى GPU (مثال cu124):

```bash
make env-gpu CUDA=cu124
# = rvc-platform env gpu cu124
# = uv pip install --index-url https://download.pytorch.org/whl/cu124 \
#     'torch==<version>+cu124' 'torchaudio==<version>+cu124'
```

الإصدارات المدعومة: `cu118 | cu121 | cu124 | cu128`.

تحقق بعدها:

```bash
make check
# python 3.12.x | torch 2.x.x+cu124 | cuda available: True
```

ملاحظات:
- الأمر يثبّت عجلة CUDA **بنفس إصدار torch الحالي** في البيئة.
- fp16 يُفعّل تلقائيًا على GPU المؤهل (يعتمد على `engine.precision: auto`)،
  ويمكن إجباره إلى `fp32` عبر CONFIGURATION.md.
- على Colab مثبّت المثبّت عجلات CUDA تلقائيًا مع الجهاز الممنوح.

---

## 7. أدوات النظام الاختيارية

| الأداة | الغرض | بدونها |
|---|---|---|
| `ffmpeg` | فك ترميز الصيغ غير WAV/FLAC في التحويل والبيانات | WAV/FLAC فقط |
| `cloudflared` | النفق العام المؤقت | لا يوجد نفق (خطأ واضح عند `make tunnel`) |
| `build-essential` | أدوات بناء native (Ubuntu) | بعض حزم pip قد تفشل بناؤها |
| `tensorboard` | مراقبة منحنيات التدريب (مجموعة training) | لا توجد لوحة TensorBoard |

```bash
# Ubuntu
sudo apt install ffmpeg cloudflared build-essential
# Windows: ثبّت ffmpeg وcloudflared يدويًا (winget أو Chocolatey أو الموقع الرسمي)
```

---

## 8. التحقق النهائي من صحة التثبيت

```bash
uv run rvc-platform install doctor
```

تقرير الطبيب يشمل:
- بروفايل النظام: OS/معمارية/Python/الجهاز (cpu|cuda)/fp16.
- حالة الأصول: `hubert_ready`, `rmvpe_ready`, `pretrained_ready`, `all_present`.
- فحص استيراد: `torch`, `faiss`, `librosa`, `transformers`, `fastapi`.

مع `--fix` ينزّل الأصول الناقصة مباشرة:

```bash
uv run rvc-platform install doctor --fix
```

ثم شغّل الخادم وتأكد من الصحة:

```bash
make web          # http://127.0.0.1:8320
curl http://127.0.0.1:8320/api/health
# {"ok": true, "detail": "Iraqi Voice Studio API is alive"}
```

---

## 9. شكل مجلد المشروع بعد التثبيت

```
iraqi-voice-platform/
├── .venv/                البيئة (uv)
├── assets/               الأصول المدربة مسبقًا (hubert_base/, rmvpe/, pretrained_v2/)
├── configs/              default.yaml + user.yaml (إن أنشأته) + training/
├── data/
│   ├── raw/              بياناتك الخام للتدريب (أنت تضعها هنا)
│   ├── uploads/          ملفات التحويل المرفوعة
│   └── conversions/      نتائج التحويل
├── models/               نماذج الأصوات (.pth + .index اختياري)
├── runs/<experiment>/   مجلدات التدريب (checkpoints/exports/…)
├── logs/                 platform.log, auth.token (عند التفعيل), processes/, tunnel.log
├── .cache/  .tmp/        مؤقتات (يحذفها `make clean`)
└── scripts/ colab/ ubuntu/ windows/  المثبّتات
```

إعادة توجيه المسارات (Colab/أقراص منفصلة) عبر متغيرات البيئة — التفصيل في
[CONFIGURATION.md](CONFIGURATION.md).

---

## 10. الخطوة التالية

- تجهيز بيانات وتدريب نموذج: [TRAINING.md](TRAINING.md)
- البث اللحظي وقياس زمن الاستجابة: [REALTIME.md](REALTIME.md)
- أعطال التثبيت: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
