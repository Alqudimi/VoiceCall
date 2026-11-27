# سكربتات جلسة القياس الحقيقية (Real-Data Session Scripts)

هذه السكربتات هي التي شُغّلت فعليًا لإنتاج كل نتائج
[OPTIMIZER_PHASE2_REAL_BENCHMARK_AR.md](../../OPTIMIZER_PHASE2_REAL_BENCHMARK_AR.md)
على بيانات المستخدم الحقيقية (37.57 دقيقة، 18 ملفًا) والنموذج المدرَّب فعليًا
`eng7mi/Iraqi-v3`. كل رقم في التقرير قابل لإعادة الإنتاج منها.

> **ملاحظة بيئة:** المسارات المضمّنة (مثل `/home/z/my-project/VoiceCall/src`)
> تعكس بيئة التشغيل الأصلية لجلسة القياس. عند إعادة التشغيل في تثبيتك المحلي،
> استبدلها بجذر مستودعك أو صدّر `PYTHONPATH="$REPO/src"` (السكربتات تعمل
> مباشرة داخل جذر المستودع بعد هذا التعديل).

## السكربتات

| السكربت | الوظيفة | الأوامر المرتبطة في `make` / CLI |
|---|---|---|
| `analyze_voice_dataset.py` | التحليل العميق لجودة البيانات + بناء بصمة المتحدث المرجعية (WavLM + ECAPA) عبر مكدس المرحلة الثانية للمشروع نفسه | `make data-check` |
| `download_prep_assets.py` | تنزيل أصول التجهيز الرسمية (HuBERT / RMVPE) | `make assets-download` |
| `run_stage.py` | تشغيل مرحلة واحدة من خط التجهيز في المقدمة، قابلة للاستئناف والمجزأة: `check \| preprocess \| f0 \| features \| filelist \| index \| all_small` | `make prepare-*` |
| `optimizer_benchmark_real.py` | القياس الحقيقي الكامل للمحسّن (نموذج مدرَّب + فهرس حقيقيان + تقسيم ملفات منفصل) — `--trials 10` | الواجهة ← تبويب «التحسين الذكي» |
| `save_best_settings.py` | حفظ أفضل الإعدادات المكتشفة في متجر المشروع (قابلة للاستعادة بزر «استعادة») | — |
| `download_user_model.py` | تنزيل النموذج المدرَّب من HuggingFace عبر خدمة المشروع (التوكن من `RVC_HF_TOKEN` فقط — لا يُخزَّن أبدًا) | `make model-download` |
| `patch_benchmark_report.py` | تدقيق القيم في التقرير المولَّد قبل النشر | — |

## إعادة إنتاج النتائج (خطوات مرتبة)

```bash
# 0) ضع البيانات الخام في data/raw/iraqi_voice ثم:
PYTHONPATH=src python scripts/real_data/analyze_voice_dataset.py   # جودة + بصمة
PYTHONPATH=src python scripts/real_data/download_prep_assets.py    # أصول رسمية
PYTHONPATH=src python scripts/real_data/run_stage.py all_small     # تجهيز كامل قابل للاستئناف

# 1) النموذج المدرَّب (مستودع خاص):
export RVC_HF_TOKEN=hf_...        # توكنك — من البيئة فقط
PYTHONPATH=src python scripts/real_data/download_user_model.py

# 2) القياس الحقيقي:
PYTHONPATH=src python scripts/real_data/optimizer_benchmark_real.py --trials 10
```

النتائج الخام المقابلة موجودة في
[reports/iraqi_voice/](../../reports/iraqi_voice/)
(جودة البيانات، تحليل كل ملف، أفضل الإعدادات، JSON القياس الكامل،
وأمثلة صوتية: خط الأساس / الأفضل / الاختبار النهائي / الإدخال).
