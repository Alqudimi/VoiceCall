#!/usr/bin/env python3
"""Safe in-place project updater shared by Ubuntu, Windows, and Colab.

It updates only a clean Git checkout tracking origin/<branch>, uses fast-forward
only, preserves ignored runtime/user files, and syncs dependencies only when the
tracked dependency manifests changed. It never runs update hooks from the repo.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from urllib.parse import urlparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_FILES = {"pyproject.toml", "uv.lock"}


class UpdateError(RuntimeError):
    pass


def run(args: list[str], *, cwd: Path | None = None, check: bool = True,
        capture: bool = False, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args, cwd=cwd or ROOT, text=True, check=False,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            env=env,
        )
    except FileNotFoundError as exc:
        raise UpdateError(f"الأداة غير موجودة: {args[0]}") from exc
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        raise UpdateError(f"فشل الأمر ({result.returncode}): {' '.join(args)}" + (f"\n{detail}" if detail else ""))
    return result


def output(args: list[str], *, check: bool = True) -> str:
    return run(args, capture=True, check=check).stdout.strip()


def ensure_services_stopped() -> None:
    runtime_root = Path(os.environ.get("RVC_PLATFORM_ROOT", str(ROOT))).expanduser().resolve()
    pid_dir = runtime_root / ".tmp"
    active: list[str] = []
    for name, label in (("web.pid", "خادم الويب"), ("trainer.pid", "عامل التدريب")):
        pid_file = pid_dir / name
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        try:
            os.kill(pid, 0)
            active.append(f"{label} (PID {pid})")
        except ProcessLookupError:
            continue  # stale PID file; never delete it automatically
        except PermissionError:
            active.append(f"{label} (PID {pid}; لا تتوفر صلاحية فحصه)")
        except OSError:
            continue
    if active:
        raise UpdateError(
            "أوقف الخدمات قبل التحديث لحماية التدريب والعمليات الجارية: "
            + "، ".join(active)
            + ". أوقفها بأمر rvc-platform web-stop ثم أعد المحاولة."
        )


def validate_checkout() -> tuple[str, str]:
    ensure_services_stopped()
    top = Path(output(["git", "rev-parse", "--show-toplevel"])).resolve()
    if top != ROOT.resolve():
        raise UpdateError(f"شغّل المُحدّث من نسخة المشروع نفسها؛ جذر Git هو {top}")
    if output(["git", "rev-parse", "--is-inside-work-tree"]) != "true":
        raise UpdateError("المجلد الحالي ليس مستودع Git")

    status = output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        preview = "\n".join(status.splitlines()[:12])
        more = "\n…" if len(status.splitlines()) > 12 else ""
        raise UpdateError(
            "أوقفت التحديث لأن نسخة العمل تحتوي تغييرات/ملفات غير متتبعة. "
            "لم يتم تغيير أي ملف. احفظ عملك أو انقله خارج المستودع ثم أعد المحاولة:\n"
            + preview + more
        )

    branch = output(["git", "branch", "--show-current"])
    if not branch:
        raise UpdateError("المستودع في وضع detached HEAD؛ انتقل إلى فرع قبل التحديث")
    upstream = output(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
    if not upstream.startswith("origin/"):
        raise UpdateError(f"الفرع يتتبع {upstream!r} بدل origin؛ لن أخمّن مصدر التحديث")
    if upstream != f"origin/{branch}":
        raise UpdateError(f"الفرع {branch!r} لا يتتبع origin/{branch} (المتتبع الحالي: {upstream})")

    remote = output(["git", "remote", "get-url", "origin"])
    parsed = urlparse(remote)
    github_https = parsed.scheme == "https" and parsed.hostname == "github.com" and bool(parsed.path)
    github_ssh = (
        parsed.scheme == "ssh" and parsed.hostname == "github.com"
        or re.fullmatch(r"git@github\.com:[^\s]+", remote) is not None
    )
    if not (github_https or github_ssh):
        raise UpdateError(f"مصدر origin ليس GitHub معروفًا (المضيف: {parsed.hostname or 'غير معروف'})")
    return branch, upstream


def detect_cuda_variant() -> str | None:
    """Return cu118/cu126/cu128 if this project's existing venv uses CUDA torch."""
    venv_python = Path(sys.executable)
    # The wrapper runs through `uv run --no-sync`, so sys.executable is the venv.
    probe = subprocess.run(
        [str(venv_python), "-c", "import torch; print(torch.__version__)"] ,
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if probe.returncode:
        return None
    match = re.search(r"\+cu(118|126|128)(?:\.|$)", probe.stdout.strip())
    return f"cu{match.group(1)}" if match else None


def restore_cuda(variant: str) -> None:
    # Query the torch version selected by the just-synced lock; don't reuse the
    # previous torch version when the project deliberately upgraded it.
    python = sys.executable
    version = output([python, "-c", "import torch; print(torch.__version__.split('+')[0])"])
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
        raise UpdateError(f"لم أتمكن من تحديد نسخة torch بعد المزامنة: {version!r}")
    index = f"https://download.pytorch.org/whl/{variant}"
    run([
        "uv", "pip", "install", "--python", python, "--index-url", index,
        f"torch=={version}+{variant}", f"torchaudio=={version}+{variant}",
    ])
    check = subprocess.run(
        [python, "-c", "import torch, torchaudio; assert torch.version.cuda; print(torch.__version__)"],
        cwd=ROOT, text=True, check=False,
    )
    if check.returncode:
        raise UpdateError("أُعيد تثبيت حزمة CUDA لكن تعذر استيرادها/التحقق منها؛ راجع سجل الأمر أعلاه")


def pending_dependency_sync() -> bool:
    return output(["git", "config", "--local", "--get", "voicecall.updatePendingDeps"], check=False) == "true"


def mark_dependency_sync(pending: bool) -> None:
    if pending:
        run(["git", "config", "--local", "voicecall.updatePendingDeps", "true"])
    else:
        run(["git", "config", "--local", "--unset", "voicecall.updatePendingDeps"], check=False)


def installed_cli() -> Path:
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "rvc-platform.exe"
    return ROOT / ".venv" / "bin" / "rvc-platform"


def main() -> int:
    parser = argparse.ArgumentParser(description="تحديث آمن للمشروع دون إعادة التثبيت من الصفر")
    parser.add_argument("--dry-run", action="store_true", help="جلب وفحص التحديث وعرض خطته دون دمج أو مزامنة")
    parser.add_argument("--skip-deps", action="store_true", help="تحديث الكود فقط؛ لا يُنصح به إذا تغيرت الاعتماديات")
    args = parser.parse_args()

    try:
        branch, upstream = validate_checkout()
        current = output(["git", "rev-parse", "HEAD"])
        pending_sync = pending_dependency_sync()
        print(f"[update] الفرع: {branch} | HEAD: {current[:12]}")
        run(["git", "fetch", "--no-tags", "origin", f"{branch}:refs/remotes/origin/{branch}"])
        target = output(["git", "rev-parse", upstream])
        if current == target and not pending_sync:
            print("[update] المشروع محدّث بالفعل؛ لا توجد تغييرات جديدة.")
            return 0
        ancestor = (run(["git", "merge-base", "--is-ancestor", current, target], check=False)
                    if current != target else None)
        if ancestor is not None and ancestor.returncode:
            raise UpdateError(
                f"الفرع المحلي {current[:12]} متباعد أو متقدم على {target[:12]}؛ "
                "لن أستخدم reset أو rebase تلقائيًا. راجع المستودع يدويًا."
            )
        changed = (output(["git", "diff", "--name-only", f"{current}..{target}"]).splitlines()
                   if current != target else [])
        sync_needed = bool(DEPENDENCY_FILES.intersection(changed)) or pending_sync
        print(f"[update] التغييرات الجديدة: {len(changed)} ملفًا")
        if sync_needed:
            print("[update] ستُزامن الاعتماديات من uv.lock" + (" (إعادة محاولة معلقة)" if pending_sync else ""))
        else:
            print("[update] ملفات الاعتماديات لم تتغير؛ لن يعاد تثبيت الحزم")
        if args.skip_deps and sync_needed:
            print("[warn] --skip-deps مختار رغم تغيّر ملفات الاعتماديات؛ قد لا يعمل البرنامج حتى المزامنة.")
        if args.dry_run:
            print(f"[dry-run] سيُطبّق fast-forward فقط إلى {target[:12]} على {branch}.")
            return 0

        cuda_variant = detect_cuda_variant() if sync_needed and not args.skip_deps else None
        if cuda_variant:
            print(f"[update] اكتُشفت عجلات PyTorch CUDA ({cuda_variant})؛ سأعيد تثبيتها بعد مزامنة القفل.")
        if current != target:
            run(["git", "merge", "--ff-only", target])

        if sync_needed and not args.skip_deps:
            mark_dependency_sync(True)
            run(["uv", "sync", "--frozen", "--all-groups"])
            if cuda_variant:
                restore_cuda(cuda_variant)
            mark_dependency_sync(False)
        elif sync_needed and args.skip_deps:
            mark_dependency_sync(True)

        cli = installed_cli()
        if cli.is_file():
            env = os.environ.copy()
            if cuda_variant:
                env["UV_NO_SYNC"] = "1"
            run([str(cli), "install", "check"], env=env)
        else:
            print("[warn] لم أجد CLI داخل .venv؛ تجاوزت فحص الاستيرادات.")

        if "scripts/migrations/" in "\n".join(changed):
            print("[notice] وصلت ملفات migrations جديدة. لم أشغّلها تلقائيًا؛ راجع تعليماتها قبل المتابعة.")
        print(f"[ok] اكتمل التحديث إلى {target[:12]}.")
        print("[info] لم تُحذف الملفات المتجاهلة في Git (مثل configs/user.yaml وdata/ وruns/ وmodels/ و.env).")
        if args.skip_deps and sync_needed:
            print("[warn] نفّذ مزامنة البيئة يدويًا عند الاستعداد: uv sync --frozen --all-groups")
        return 0
    except UpdateError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[error] أوقف المستخدم عملية التحديث.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
