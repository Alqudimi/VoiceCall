# Cloudflare Tunnel — دليل التشغيل والتثبيت

هذا المشروع يستخدم **Cloudflare Quick Tunnel** لإنشاء عنوان HTTPS مؤقت يصل إلى خادم المنصة المحلي. لا يحتاج إلى حساب Cloudflare أو دومين ثابت. هذا الوضع مناسب للتجربة والعرض المؤقت، وليس استضافة إنتاجية: العنوان يتغير مع كل تشغيل، ولا يوجد ضمان وقت تشغيل، والرابط متاح لأي شخص يعرفه ما لم تُضف طبقة حماية أخرى. راجع [وثائق Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) و[صفحة تنزيل cloudflared الرسمية](https://developers.cloudflare.com/tunnel/downloads/).

## ما الذي يحدث عند التشغيل

1. المثبت يثبت `cloudflared` الرسمي في مسار المستخدم، ويتحقق من إصدار البرنامج قبل حفظه. على Linux لا يحتاج هذا إلى صلاحية root؛ الملف يوضع تحت `~/.local/bin`. على Windows يوضع تحت `%LOCALAPPDATA%\IraqiVoiceStudio\bin` ويضاف المسار إلى `PATH` الخاص بالمستخدم.
2. ابدأ خادم VoiceCall. عند طلب Tunnel يفحص التطبيق `/api/health` محليًا أولًا، ثم يشغل `cloudflared tunnel --url http://127.0.0.1:<port>` ويتعرف إلى عنوان `trycloudflare.com` من سجل العملية.
3. يجري التطبيق فحص `/api/health` عبر العنوان العام قبل إعلان النجاح. إذا كان DNS المحلي لا يستطيع حل اسم `trycloudflare.com` لكنه يرى سجل اتصال Cloudflare مسجلًا، يعرض التطبيق الرابط مع تحذير بدل رفضه؛ قد يستغرق نشر DNS عدة ثوانٍ. اختبر الرابط من شبكة أخرى قبل الاعتماد عليه.
4. قبل تشغيل النفق يفعّل التطبيق Bearer authentication. إذا لم يحدد المستخدم رمزًا من الإعدادات، ينشأ رمز عشوائي محفوظ في `logs/auth.token` بأذونات ملف مقيدة على Linux، ويبقى ثابتًا عبر إعادة تشغيل التطبيق.
5. افتح الرابط العام. في تبويب **Settings / الإعدادات** ألصق الرمز الموجود في `logs/auth.token`. يخزن المتصفح الرمز محليًا ويرسله لترويسات HTTP، أما WebSocket فيمرره في معامل الاتصال لأن واجهة المتصفح لا تتيح تعيين Authorization header عند إنشاء WebSocket. لا تضع الرابط أو الرمز في منشور عام.

النفق يوجه إلى التطبيق على loopback؛ لذلك لا يحتاج خادم الويب إلى الاستماع على كل واجهات الشبكة. لا تستخدم إعدادات CORS مفتوحة أو تنشر الخدمة مباشرة على عنوان عام.

## تثبيت مباشر من GitHub

الأوامر أدناه تشير إلى `main` في المستودع `Alqudimi/VoiceCall`. لكي تحصل على التعديلات الأخيرة التي تعمل عليها في نسختك، يجب أولًا رفع تلك التعديلات إلى GitHub. قبل رفعها، رابط Raw على GitHub سيعيد النسخة الموجودة حاليًا في الفرع وليس الملفات المحلية غير المنشورة.

### Ubuntu / Linux

شغّل المثبت مباشرة في Bash. يستنسخ المستودع إلى `~/VoiceCall` إذا لم يكن موجودًا، ولا يحذف مجلدًا سابقًا. يمكن تخطي تنزيل أوزان RVC الكبيرة مع `--skip-assets`، أو تخطي تثبيت النفق مع `--skip-tunnel`:

```bash
curl -fsSL https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/ubuntu/install_ubuntu.sh | bash -s -- --skip-assets
```

المثبت يثبت أدوات البناء والصوت و`uv` و`cloudflared`؛ ثم ينشئ `.venv` ويثبت التبعيات من `uv.lock`. عند عدم وجود صلاحيات apt، يتطلب تثبيت المتطلبات النظامية يدويًا ثم يمكن إعادة المحاولة هكذا:

```bash
cd ~/VoiceCall
bash scripts/ubuntu/install_ubuntu.sh --no-apt --skip-assets
```

يُدعم تنزيل `cloudflared` الرسمي لـ`x86_64` و`aarch64` وARMv7 و32-bit x86 وفق ملف الإصدار المناسب. المثبت يختبر `cloudflared --version` بعد التنزيل.

### Google Colab

من خلية Colab جديدة شغّل:

```bash
!curl -fsSL https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/colab/install_colab.sh | bash
```

لإرجاء تنزيل الأصول الثقيلة:

```bash
!curl -fsSL https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/colab/install_colab.sh -o /tmp/ivs-install.sh
!SKIP_ASSETS=1 bash /tmp/ivs-install.sh
```

المثبت يركب Google Drive عند الموافقة، يثبت `uv` والتبعيات، يكتشف CUDA المتاح، يختار Torch المناسب عندما يمكن التعرف إلى نسخة التعريف، ويثبت `cloudflared` إلى `~/.local/bin`. إذا لم يُمنح GPU يستمر في وضع CPU مع تحذير. بيئة Colab مؤقتة؛ يجب حفظ ملفات التدريب على Drive، ويُقطع الرابط العام عند إيقاف الجلسة أو النفق.

### Windows 10/11 — PowerShell

نفذ في PowerShell. هذه الطريقة تنزل ملف PowerShell إلى مجلد مؤقت ثم تشغله، فيستنسخ المشروع إلى `%USERPROFILE%\VoiceCall`:

```powershell
$installer = Join-Path $env:TEMP 'ivs-install-windows.ps1'
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/windows/install_windows.ps1' -OutFile $installer
powershell -NoProfile -ExecutionPolicy Bypass -File $installer -SkipAssets
```

إذا كان Git غير موجود، يحاول المثبت توفيره عبر `winget`. يثبت `cloudflared` من أصل Cloudflare الرسمي `cloudflared-windows-amd64.exe`، ويفحص `--version`، ويضيفه إلى PATH للمستخدم. إذا غيّرت PATH، افتح PowerShell جديدًا بعد التثبيت.

للتثبيت من مستودع مستنسخ، شغّل `scripts\windows\install_windows.ps1` مباشرة. ولتغيير مجلد الاستنساخ استخدم `-InstallDir 'D:\VoiceCall'`. يقتصر المثبت الحالي على Windows x64.

### خيارات مثبتات الأنظمة

- Ubuntu: `--skip-assets` لتأجيل الأصول، و`--skip-tunnel` لتأجيل `cloudflared`، و`--no-apt` عندما تكون متطلبات النظام قد ثبتت مسبقًا. يغير `RVC_INSTALL_DIR` مسار clone و`IRAQI_VOICE_REPO` عنوان المستودع.
- Colab: `SKIP_ASSETS=1` و`SKIP_TUNNEL=1` و`MOUNT_DRIVE=0` و`DRIVE_DIR=RVC_Iraqi`. يمكن تغيير مسار clone بمتغير `RVC_INSTALL_DIR` ومصدره بمتغير `IRAQI_VOICE_REPO`.
- Windows: `-SkipAssets` و`-SkipTunnel` و`-RepoUrl` و`-InstallDir` و`-GpuVariant`.

## بدء النفق وإيقافه

### بدء التطبيق والنفق معًا

```bash
cd ~/VoiceCall
uv run rvc-platform web --tunnel
```

تظل الطرفية مشغولة بخادم الويب. يعرض الطرفية الرابط عند نجاح فحصه. افتح رابط Quick Tunnel في المتصفح، ثم أدخل رمز `logs/auth.token` من تبويب الإعدادات.

### تشغيلهما كلًا على حدة

في الطرفية الأولى:

```bash
cd ~/VoiceCall
uv run rvc-platform web
```

في الطرفية الثانية:

```bash
cd ~/VoiceCall
make tunnel
```

أوامر دورة الحياة:

```bash
make tunnel-status
make tunnel-stop
make web-stop
```

أمر إيقاف الويب يرسل إيقافًا مرتبًا إلى خادم API، والذي يغلق النفق الذي يديره. على Windows أو Colab دون `make` استخدم `uv run rvc-platform web`, `uv run rvc-platform web-stop`، أو واجهة API مع Bearer token. سجلات `cloudflared` موجودة في `logs/processes/cloudflared/cloudflared.log`، والحالة عبر `make tunnel-status`.

## API: الطلبات والاستجابات والأخطاء

Base URL محلي افتراضيًا هو `http://127.0.0.1:8320`. عند التفعيل عبر Tunnel يجب إرسال `Authorization: Bearer <token>` لكل API باستثناء `/api/health`. ابدأ النفق محليًا قبل التفعيل من localhost، أو أرسل رمز الإعداد إن كان مفعلًا أصلًا.

### تشغيل النفق — `POST /api/tunnel/start`

```bash
curl -sS -X POST http://127.0.0.1:8320/api/tunnel/start
```

استجابة نجاح `200`:

```json
{
  "running": true,
  "url": "https://<random>.trycloudflare.com",
  "message": "tunnel active — auth token is now required for API access"
}
```

خطأ `502` يعني أن `cloudflared` غير مثبت، أو أن الخادم المحلي لم يستجب، أو أن اتصال Cloudflare لم ينجح. راجع سجل Cloudflared وتحقق من الاتصال الخارجي. الخطأ `401` يعني أن المصادقة كانت مفعلة مسبقًا، ويلزم ترويسة Bearer صحيحة.

### الحالة — `GET /api/tunnel/status`

```bash
curl -sS http://127.0.0.1:8320/api/tunnel/status \
  -H "Authorization: Bearer $TOKEN"
```

الاستجابة تشمل `running` و`url` (أو `null` إذا لم يعمل) و`pid` و`started_at` و`log_path`.

### الإيقاف — `POST /api/tunnel/stop`

```bash
curl -sS -X POST http://127.0.0.1:8320/api/tunnel/stop \
  -H "Authorization: Bearer $TOKEN"
```

الاستجابة: `{"running":false,"clean":true}` عند الإيقاف الناجح. إذا `clean` false، افحص السجل ثم أوقف `cloudflared` يدويًا. الرابط يتوقف فور انتهاء عملية cloudflared.

### السجل — `GET /api/tunnel/logs?lines=60`

```bash
curl -sS 'http://127.0.0.1:8320/api/tunnel/logs?lines=60' \
  -H "Authorization: Bearer $TOKEN"
```

الاستجابة: `{"lines":["...", "..."]}`. عند عدم تشغيل العملية قد تكون القائمة فارغة.

## اختبار الوظيفة

بعد تثبيت `cloudflared`:

1. افحص `uv run rvc-platform install check` وتأكد من ظهور الحزم الأساسية.
2. شغل `uv run rvc-platform web` واتأكد محليًا من `curl -fsS http://127.0.0.1:8320/api/health`.
3. شغل `make tunnel` أو `uv run rvc-platform web --tunnel` وانتظر استجابة النجاح التي تتضمن رابط HTTPS واتصال Cloudflare المسجل. إذا كانت بيئة الخادم تقيد DNS، تحقق من الرابط خارجيًا بعد بضع ثوانٍ من انتشاره.
4. اختبر الرابط من جهاز/شبكة أخرى: `curl -fsS https://<random>.trycloudflare.com/api/health`، ثم افتح الصفحة وأدخل رمز المصادقة. اختبر أن `/api/system` يرجع `401` بلا رمز و`200` مع الرمز.
5. نفذ `make tunnel-stop` ثم تحقق من أن الرابط توقف، وبعدها `make web-stop`.

لا يمكن لفحص محلي وحده إثبات وصول الشبكات الخارجية من مزود الإنترنت لديك؛ يجب إجراء خطوة 4 على جهازك أو بيئة الشبكة التي ستستخدمها.

## متطلبات الشبكة والقيود

- يتطلب Cloudflare Tunnel اتصالات صادرة إلى شبكة Cloudflare. يستخدم التطبيق سلوك البروتوكول الافتراضي لـ`cloudflared` ليختار النقل الممكن، بدل فرض HTTP/2 على الشبكات التي تمنع TCP.
- Quick Tunnel عنوان عشوائي مؤقت، يتبدل في كل مرة. لا يوجد SLA، وهو للاختبار والتطوير فقط. وثائق Cloudflare تذكر حد 200 طلب متزامن لكل Quick Tunnel وعدم دعم SSE؛ WebSocket متاح عبر HTTP/HTTPS لكن أداء الصوت يعتمد على الشبكة والجهاز وGPU.
- العنوان غير محمي بمجرد عشوائيته؛ استخدم رمز المنصة، ولا تكشف `logs/auth.token`. لإضافة حاجز بريد إلكتروني على Quick Tunnel، راجع Cloudflare `--allowed-mail` في الوثائق الرسمية. هذا الخيار ليس مفعّلًا افتراضيًا هنا.
- إيقاف العملية أو انقطاع جلسة Colab ينهي الرابط. للنطاق الثابت والاستعمال طويل الأمد أنشئ Named Tunnel مربوطًا بحساب Cloudflare بدل Quick Tunnel.

## المراجع الرسمية

- [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
- [تنزيل cloudflared من Cloudflare](https://developers.cloudflare.com/tunnel/downloads/)
- [مستودع cloudflared الرسمي](https://github.com/cloudflare/cloudflared)
