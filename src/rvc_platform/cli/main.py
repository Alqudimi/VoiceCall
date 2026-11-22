"""Unified CLI — every Makefile target maps 1:1 onto a command here.

Designed to be the full replacement for `make` on Windows / Colab:
    rvc-platform <group> <command> ...        (see --help on each group)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rvc_platform import __version__
from rvc_platform.infrastructure.paths import get_paths

app = typer.Typer(
    name="rvc-platform",
    help="Iraqi Voice Studio — RVC v2 conversion platform (train / convert / realtime)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_PID_FILE = get_paths().temp / "web.pid"
UVICORN_ARGS = ["-m", "uvicorn", "rvc_platform.api.app:create_app", "--factory"]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *cmd], **kwargs)


def _web_pid() -> int | None:
    try:
        return int(_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _port_open(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _api(port: int, path: str = "/api/system") -> dict | None:
    try:
        import httpx

        resp = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=5.0)
        return resp.json() if resp.status_code < 500 else None
    except Exception:
        return None


def _print_status(status: dict) -> None:
    table = Table(title=f"Iraqi Voice Studio v{__version__}")
    table.add_column("component")
    table.add_column("state")
    table.add_row("OS / arch", f"{status.get('os')} / {status.get('arch')}")
    table.add_row("Python", str(status.get("python")))
    table.add_row("Device", f"{status.get('device')} fp16={status.get('is_half')}")
    table.add_row(
        "GPU",
        f"{(status.get('gpu') or {}).get('name')} "
        f"{(status.get('gpu') or {}).get('vram_total_mb') or ''}MB",
    )
    table.add_row("Engine", f"{status.get('engine')} {status.get('engine_message', '')}")
    table.add_row(
        "Loaded model",
        (status.get("loaded_model") or {}).get("name", "—"),
    )
    training = status.get("training") or {}
    table.add_row(
        "Training",
        f"{training.get('state')} epoch {training.get('epoch')}/{training.get('total_epochs')}",
    )
    table.add_row("Realtime sessions", str(status.get("realtime_sessions")))
    table.add_row("Tunnel", json.dumps(status.get("tunnel") or {"running": False}))
    console.print(table)


# --------------------------------------------------------------------------- #
# install / environment
# --------------------------------------------------------------------------- #
install_app = typer.Typer(help="Installation & environment management")
app.add_typer(install_app, name="install")


@install_app.command("doctor")
def doctor(fix: bool = typer.Option(False, "--fix", help="download missing assets")):
    """Deep environment check: GPU/CUDA/deps/assets; --fix downloads assets."""
    from rvc_platform.utils.environment import detect_system

    prof = detect_system()
    console.print_json(json.dumps(prof.summary, ensure_ascii=False))
    from rvc_platform.application.downloads import asset_status

    assets = asset_status()
    console.print_json(json.dumps(assets, ensure_ascii=False))
    missing = not assets["_summary"]["all_present"]
    if missing and fix:
        from rvc_platform.application.downloads import download_assets

        download_assets(progress=lambda name, frac: console.print(
            f"[dim]downloading {name}: {frac * 100:.0f}%"))
        console.print("[green]assets downloaded[/green]")
    elif missing:
        console.print("[yellow]assets missing — run `rvc-platform install doctor --fix` "
                      "or `make download-models`[/yellow]")
    # import sanity
    for mod in ("torch", "faiss", "librosa", "transformers", "fastapi"):
        try:
            __import__(mod)
            console.print(f"[green]import {mod} OK[/green]")
        except Exception as exc:
            console.print(f"[red]import {mod} FAILED: {exc}[/red]")


@install_app.command("check")
def check():
    """Quick dependency check (imports + versions)."""
    import torch

    console.print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
                  f"cuda available: {torch.cuda.is_available()}")
    for mod in ("faiss", "librosa", "transformers", "fastapi", "soundfile",
                "parselmouth", "uvicorn"):
        try:
            m = __import__(mod)
            console.print(f"[green]{mod}[/green] {getattr(m, '__version__', '?')}")
        except Exception as exc:
            console.print(f"[red]{mod}[/red] {exc}")


env_app = typer.Typer(help="uv environment management")
app.add_typer(env_app, name="env")


@env_app.command("sync")
def sync(groups: bool = typer.Option(True, "--all-groups/--no-groups")):
    """uv sync (all dependency groups by default)."""
    cmd = ["uv", "sync"] + (["--all-groups"] if groups else [])
    raise typer.Exit(os.system(" ".join(cmd)))


_GPU_VARIANTS = ("cu118", "cu126", "cu128")
# NOTE: for the locked torch 2.7.x the PyTorch index publishes ONLY cu118 /
# cu126 / cu128 wheels (verified against download.pytorch.org). cu121/cu124
# do not exist for this version and would fail resolution.


@env_app.command("gpu")
def env_gpu(cuda: str = typer.Argument("cu128", help="cu118|cu126|cu128")):
    """Install CUDA torch wheels into the project venv (CPU -> GPU switch).

    IMPORTANT — persistence: a plain ``uv run`` re-syncs the environment from
    the lockfile and would REVERT these GPU wheels back to CPU. After this
    command, run the platform with ``UV_NO_SYNC=1 uv run ...`` (or
    ``uv run --no-sync ...``, or ``.venv/bin/rvc-platform ...``) so the venv
    is used as-is. On Colab the installer wires UV_NO_SYNC automatically.
    """
    if cuda not in _GPU_VARIANTS:
        raise typer.BadParameter(
            f"supported: {', '.join(_GPU_VARIANTS)} — cu121/cu124 have no "
            "wheels for the locked torch 2.7.x"
        )
    import torch

    version = torch.__version__.split("+")[0]
    console.print(f"installing torch=={version}+{cuda} ...")
    cmd = (f"uv pip install --index-url https://download.pytorch.org/whl/{cuda} "
           f"'torch=={version}+{cuda}' 'torchaudio=={version}+{cuda}'")
    code = os.system(cmd)
    if code != 0:
        console.print("[red]GPU torch install FAILED — the venv still has the "
                      "previous torch build[/red]")
        raise typer.Exit(1)
    # honest verification in a FRESH interpreter: this process still holds the
    # old CPU torch in memory, so re-importing here would lie.
    probe = (
        "import torch, torchaudio; "
        "print(torch.__version__); "
        "print('cuda', torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
    )
    check = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if check.returncode != 0:
        console.print(f"[red]torch/torchaudio import failed after GPU install:\n"
                      f"{check.stderr.strip()[:500]}[/red]")
        raise typer.Exit(1)
    out_lines = [line for line in check.stdout.strip().splitlines() if line]
    torch_ver = out_lines[0] if out_lines else "?"
    cuda_ok = len(out_lines) > 1 and out_lines[1].endswith("True")
    gpu_name = out_lines[2] if len(out_lines) > 2 else "-"
    if cuda_ok:
        console.print(f"[green]GPU torch active: {torch_ver} on {gpu_name}[/green]")
    else:
        console.print(f"[yellow]torch {torch_ver} installed but CUDA is NOT "
                      "available — check nvidia-smi / driver, or pick a variant "
                      "matching your driver (cu118 for 11.x, cu126 for 12.0–12.7, "
                      "cu128 for 12.8+ / 13.x)[/yellow]")
    console.print("next: keep the GPU wheels by exporting UV_NO_SYNC=1 before "
                  "`uv run` (plain `uv run` re-syncs back to CPU torch)")


@env_app.command("lock")
def lock():
    raise typer.Exit(os.system("uv lock"))


@env_app.command("update")
def update():
    raise typer.Exit(os.system("uv sync --upgrade --all-groups"))


@env_app.command("verify")
def verify():
    """Re-create the env from the lockfile and verify imports."""
    code = os.system("uv sync --frozen --all-groups")
    if code != 0:
        raise typer.Exit(code)
    check()


@env_app.command("info")
def env_info():
    from rvc_platform.infrastructure.paths import get_paths as gp

    p = gp()
    console.print(f"project root : {p.root}\nassets        : {p.assets}\n"
                  f"models        : {p.models}\ndata          : {p.data}\n"
                  f"runs          : {p.runs}\ncache         : {p.cache}")


@app.command("clean")
def clean(all: bool = typer.Option(False, "--all", help="also remove runs & data")):
    """Remove caches and temp files (kept: models, assets, runs, data)."""
    import shutil

    p = get_paths()
    for d in (p.cache, p.temp):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            console.print(f"removed {d}")
    if all:
        for d in (p.runs, p.data / "uploads", p.data / "conversions"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                console.print(f"removed {d}")


# --------------------------------------------------------------------------- #
# assets / data
# --------------------------------------------------------------------------- #
@app.command("download-models")
def download_models():
    """Download HuBERT / RMVPE / pretrained v2 bases (official HF release)."""
    from rvc_platform.application.downloads import download_assets

    download_assets(progress=lambda name, frac: console.print(
        f"[dim]{name}: {frac * 100:.0f}%[/dim]"))
    console.print("[green]all pretrained assets ready[/green]")


@app.command("data-check")
def data_check(dataset: Path):
    """Inventory + quality check of a dataset folder."""
    from rvc_platform.training.preprocess import DatasetChecker

    report = DatasetChecker().check(dataset)
    console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
    if not report.meets_recommendation:
        console.print("[yellow]< 10 minutes of audio — quality will suffer "
                      "(collect 10-60 min of clean Iraqi speech)[/yellow]")


@app.command("prepare-audio-dataset")
def prepare_audio_dataset(
    source: Path = typer.Argument(..., help="External folder containing audio files (recursive)"),
    dataset_name: str = typer.Argument(..., help="Safe dataset name, e.g. my_voice"),
    sr: str = typer.Option("40k", "--sr", help="Output rate: 32k, 40k, or 48k"),
    min_duration: float = typer.Option(0.35, "--min-duration", help="Minimum usable clip duration in seconds"),
    trim_margin_ms: int = typer.Option(120, "--trim-margin-ms", help="Silence margin kept around speech"),
    force: bool = typer.Option(False, "--force", help="Replace an existing data/raw/<name> directory"),
):
    """Decode, clean, deduplicate, and install an external audio folder in data/raw."""
    from rvc_platform.training.dataset_import import AudioDatasetImporter, parse_sample_rate

    importer = AudioDatasetImporter(
        target_sr=parse_sample_rate(sr),
        min_duration_s=min_duration,
        trim_margin_ms=trim_margin_ms,
    )
    report = importer.import_folder(
        source, dataset_name, force=force,
        progress=lambda frac, name: console.print(f"[dim]import: {frac * 100:.0f}% ({name})[/dim]"),
    )
    console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
    console.print(f"[green]dataset ready: {report.output_path}[/green]")
    console.print(
        f"Next: make prepare-dataset DATASET={report.output_path} "
        f"EXPERIMENT={dataset_name} SR={sr} F0_METHOD=rmvpe"
    )


@app.command("preprocess")
def preprocess(
    dataset: Path,
    experiment: str,
    sr: str = typer.Option("48k", "--sr"),
    version: str = typer.Option("v2", "--version", "-v", help="model version: v2 (default) or v1"),
    denoise: bool = typer.Option(False, "--denoise/--no-denoise",
                                  help="best-effort spectral-gate denoise before slicing (noisereduce)"),
    loudness: bool = typer.Option(True, "--loudness/--no-loudness",
                                  help="per-chunk RMS loudness normalisation (-20 dBFS)"),
):
    from rvc_platform.training.run_config import TrainingRunConfig

    cfg = TrainingRunConfig(experiment_name=experiment, dataset_path=dataset,
                            sample_rate=sr, version=version, denoise=denoise, loudness=loudness)
    DataPipelineRunner(cfg).stage("preprocess")


@app.command("extract-features")
def extract_features(
    experiment: str,
    method: str = typer.Option("rmvpe", "--f0-method", "--method"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="model version: v2 (default) or v1"),
):
    """Extract F0 (+HuBERT features) for an existing run with the chosen method."""
    from rvc_platform.training.run_config import TrainingRunConfig

    cfg = TrainingRunConfig.load(get_paths().runs / experiment)
    cfg.f0_method = method
    if version:
        cfg.version = version
    DataPipelineRunner(cfg).stage("features", "f0")


@app.command("prepare-dataset")
def prepare_dataset(
    dataset: Path,
    experiment: str,
    sr: str = typer.Option("48k", "--sr"),
    version: str = typer.Option("v2", "--version", "-v", help="model version: v2 (default, 768-dim) or v1 (256-dim)"),
    method: str = typer.Option("rmvpe", "--f0-method", "--method"),
    denoise: bool = typer.Option(False, "--denoise/--no-denoise",
                                  help="best-effort spectral-gate denoise before slicing (noisereduce)"),
    loudness: bool = typer.Option(True, "--loudness/--no-loudness",
                                  help="per-chunk RMS loudness normalisation (-20 dBFS)"),
    speech_filter: float = typer.Option(
        0.25, "--speech-filter",
        help="drop chunks whose voiced-frame ratio (energy VAD) is below this "
             "value; 0 disables (recommended 0.2-0.3)",
    ),
):
    """Full pipeline: check -> preprocess -> F0 -> HuBERT -> filelist -> index."""
    from rvc_platform.training.run_config import TrainingRunConfig

    cfg = TrainingRunConfig(experiment_name=experiment, dataset_path=dataset,
                            sample_rate=sr, version=version, f0_method=method,
                            denoise=denoise, loudness=loudness,
                            speech_filter_ratio=max(0.0, min(1.0, speech_filter)))
    DataPipelineRunner(cfg).all()


class DataPipelineRunner:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _progress(self, stage: str, frac: float, name: str) -> None:
        console.print(f"[dim]{stage}: {frac * 100:.0f}% ({name})[/dim]")

    def stage(self, *stages: str):
        from rvc_platform.training.data_pipeline import DataPipeline

        pipeline = DataPipeline(self.cfg)
        mapping = {
            "check": pipeline.check,
            "preprocess": lambda: pipeline.preprocess(self._progress),
            "f0": lambda: pipeline.extract_f0(self._progress),
            "features": lambda: pipeline.extract_features(self._progress),
            "index": lambda: pipeline.build_index(self._progress),
            "filelist": pipeline.write_filelist,
        }
        out = {}
        for stage in stages:
            out[stage] = mapping[stage]()
            console.print_json(json.dumps({stage: out[stage]}, ensure_ascii=False, default=str))
        return out

    def all(self):
        from rvc_platform.training.data_pipeline import DataPipeline

        pipeline = DataPipeline(self.cfg)
        report = pipeline.prepare_all(progress=self._progress)
        console.print_json(json.dumps(report, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
train_app = typer.Typer(help="training lifecycle")
app.add_typer(train_app, name="train")


@train_app.command("start")
def train_start(
    dataset: Optional[Path] = typer.Option(
        None, "--dataset",
        help="raw dataset folder — only needed the FIRST time (before cell 3 "
             "prepares the run); on an existing run it just moves the pointer",
    ),
    experiment: str = typer.Option("iraqi_voice", "--experiment", "-e"),
    sr: Optional[str] = typer.Option(
        None, "--sr",
        help="sample rate for a NEW run; on a prepared run it must match the "
             "prepared value (locked — re-prepare to change)",
    ),
    version: Optional[str] = typer.Option(
        None, "--version", "-v",
        help="model version: v2 (default, 768-dim) or v1 (256-dim)",
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs",
        help="total epoch target (resume continues up to it); unset = keep saved",
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch", help="batch size; unset = keep the saved value"),
    lr: Optional[float] = typer.Option(
        None, "--lr", help="learning rate; unset = keep the saved value"),
    save_every: Optional[int] = typer.Option(
        None, "--save-every", help="checkpoint interval; unset = keep saved"),
    gpu: Optional[int] = typer.Option(None, "--gpu"),
    f0_method: Optional[str] = typer.Option(
        None, "--f0-method",
        help="F0 extractor for a NEW run; on a prepared run it must match "
             "the prepared value (locked — re-prepare to change)",
    ),
    no_pretrained: bool = typer.Option(False, "--no-pretrained"),
    clean_start: bool = typer.Option(False, "--clean-start"),
    hf_repo: Optional[str] = typer.Option(None, "--hf-repo", help="private HF model repo (owner/name)"),
    save_best: Optional[bool] = typer.Option(
        None, "--save-best/--no-save-best",
        help="track the best-so-far model and publish it to best/ on the Hub",
    ),
    val_split: Optional[float] = typer.Option(
        None, "--val-split",
        help="held-out data fraction (0-0.5) scoring the best model; 0 = train-mel fallback",
    ),
    # ---- researched quality levers (UPGRADE_REPORT_AR.md) ----
    no_ema: bool = typer.Option(
        False, "--no-ema",
        help="disable the generator EMA (BigVGAN-style averaging of G weights)",
    ),
    ema_decay: Optional[float] = typer.Option(
        None, "--ema-decay",
        help="EMA decay (default 0.999); 0 or --no-ema disables",
    ),
    stft_weight: Optional[float] = typer.Option(
        None, "--stft-weight",
        help="multi-resolution STFT loss weight (default 2.0; 0 = upstream objective)",
    ),
    no_msd: bool = typer.Option(
        False, "--no-msd",
        help="disable the extra multi-scale raw discriminators (HiFi-GAN MSD)",
    ),
    spk_weight: Optional[float] = typer.Option(
        None, "--spk-weight",
        help="speaker-similarity weight in the composite best-model metric (needs speechbrain)",
    ),
    early_stop: Optional[int] = typer.Option(
        None, "--early-stop",
        help="stop cooperatively after N epochs without a new best model (0 = off)",
    ),
    resume_from: Optional[str] = typer.Option(
        None, "--resume-from",
        help="explicit checkpoint: G_<step>.pth name, absolute path, or 'latest'",
    ),
    # ---- Phase-2 researched quality levers (UPGRADE_REPORT_AR.md §2.8+) ----
    augment: Optional[bool] = typer.Option(
        None, "--augment/--no-augment",
        help="online waveform augmentation: light noise/gain/dropout on the real side",
    ),
    warmup: Optional[int] = typer.Option(
        None, "--warmup",
        help="linear LR warmup epochs for fine-tune stability (default 1; 0 = off)",
    ),
    cosine: Optional[float] = typer.Option(
        None, "--cosine",
        help="cosine-tail LR target fraction (e.g. 0.1; 0 = pure exponential decay)",
    ),
    d_lr_scale: Optional[float] = typer.Option(
        None, "--d-lr-scale",
        help="discriminator LR scale (1.0 = upstream parity — official GAN repos use identical G/D LR; e.g. 0.9 slows D)",
    ),
    clip_grad_norm: Optional[float] = typer.Option(
        None, "--clip-grad-norm",
        help="clip generator gradient global norm (BigVGAN §3.4 anti-collapse; e.g. 500; 0 = off = upstream)",
    ),
):
    """Start (or continue) a training run. Follows live progress on stdout."""
    from rvc_platform.training.run_config import TrainingRunConfig

    paths = get_paths()
    run_dir = paths.runs / experiment
    run_config_path = run_dir / "run_config.json"
    existing_cfg = (TrainingRunConfig.load(run_dir)
                    if run_config_path.is_file() else None)
    quality_overrides = {
        **({"ema_decay": 0.0} if no_ema else
           ({"ema_decay": ema_decay} if ema_decay is not None else {})),
        **({"c_stft": stft_weight} if stft_weight is not None else {}),
        **({"use_msd": False} if no_msd else {}),
        **({"spk_weight": spk_weight} if spk_weight is not None else {}),
        **({"early_stop_patience": early_stop} if early_stop is not None else {}),
        # Phase-2 levers
        **({"augment": augment} if augment is not None else {}),
        **({"warmup_epochs": warmup} if warmup is not None else {}),
        **({"cosine_frac": cosine} if cosine is not None else {}),
        **({"d_lr_scale": d_lr_scale} if d_lr_scale is not None else {}),
        **({"clip_grad_norm": clip_grad_norm} if clip_grad_norm is not None else {}),
    }
    if dataset is not None:
        if existing_cfg is not None:
            # v1.3.1: the run is already prepared — --dataset only moves the
            # pointer; prepare settings (denoise/loudness/speech-filter) and
            # the locked SR/F0 must survive, never reset to defaults.
            cfg = existing_cfg
            if sr is not None and sr != cfg.sample_rate:
                raise typer.BadParameter(
                    f"--sr {sr} does not match the prepared run ({cfg.sample_rate}); "
                    "the data is already sliced/extracted at that rate — "
                    "re-run prepare-dataset (and expect a new run) to change it")
            if f0_method is not None and f0_method != cfg.f0_method:
                raise typer.BadParameter(
                    f"--f0-method {f0_method} does not match the prepared run "
                    f"({cfg.f0_method}); F0 files are already extracted — "
                    "re-run prepare-dataset to change it")
            cfg.dataset_path = dataset
            console.print(f"[yellow]prepared run {experiment}: --dataset only "
                          "moved the raw-data pointer (no re-preparation)[/yellow]")
        else:
            cfg = TrainingRunConfig(
                experiment_name=experiment, dataset_path=dataset,
                sample_rate=sr or "48k", version=version or "v2",
                epochs=epochs or 300, batch_size=batch_size or 8,
                learning_rate=lr or 1e-4, save_every_epoch=save_every or 10,
                gpu=gpu, f0_method=f0_method or "rmvpe",
                preload_base_models=not no_pretrained, clean_start=clean_start,
                hf_repo_id=hf_repo,
                **({"save_best": save_best} if save_best is not None else {}),
                **({"val_split": val_split} if val_split is not None else {}),
                **quality_overrides,
            )
    elif existing_cfg is not None:
        cfg = existing_cfg
        # v1.3.1: apply the training knobs onto an existing run WITHOUT
        # --dataset — the prepared pipeline (chunks/F0/features/filelist)
        # lives inside runs/<exp>/ and is never re-executed by train start.
        if version is not None and version != cfg.version:
            raise typer.BadParameter(
                f"--version {version} does not match the prepared run ({cfg.version}); "
                "re-run prepare-dataset (and expect a new run) to change it")
        if sr is not None and sr != cfg.sample_rate:
            raise typer.BadParameter(
                f"--sr {sr} does not match the prepared run ({cfg.sample_rate}); "
                "the data is already sliced/extracted at that rate — "
                "re-run prepare-dataset (and expect a new run) to change it")
        if f0_method is not None and f0_method != cfg.f0_method:
            raise typer.BadParameter(
                f"--f0-method {f0_method} does not match the prepared run "
                f"({cfg.f0_method}); F0 files are already extracted — "
                "re-run prepare-dataset to change it")
        console.print(f"[green]continuing prepared run[/green] {experiment} "
                      f"(sr={cfg.sample_rate}, f0={cfg.f0_method}, "
                      f"dataset prepared at {cfg.dataset_path})")
    else:
        console.print("[red]no existing run and no --dataset given[/red]")
        raise typer.Exit(2)
    if existing_cfg is not None:
        # knobs always apply on top of the loaded (prepared) config
        if epochs is not None:
            cfg.epochs = epochs
        if batch_size is not None:
            cfg.batch_size = batch_size
        if lr is not None:
            cfg.learning_rate = lr
        if save_every is not None:
            cfg.save_every_epoch = save_every
        if gpu is not None:
            cfg.gpu = gpu
    cfg.clean_start = clean_start
    if hf_repo is not None:
        cfg.hf_repo_id = hf_repo
    if save_best is not None:
        cfg.save_best = save_best
    if val_split is not None:
        cfg.val_split = val_split
    for key, value in quality_overrides.items():
        setattr(cfg, key, value)
    cfg.resume = not clean_start
    if resume_from is not None:
        if clean_start:
            raise typer.BadParameter("--resume-from cannot be combined with --clean-start")
        cfg.resume_from = Path(resume_from)
        cfg.resume = True
        console.print(f"[cyan]explicit resume point: {resume_from}[/cyan]")

    # official epoch guidance (RVC FAQ Q9) derived from the measured dataset
    # quality (SNR) in the prepare-dataset report — surfaced BEFORE hours are
    # spent; best-effort: a missing/legacy report simply skips the hint
    try:
        import json as _json
        report_path = run_dir / "dataset_report.json"
        if report_path.is_file():
            report = _json.loads(report_path.read_text(encoding="utf-8"))
            rec = report.get("epoch_recommendation") or {}
            snr = report.get("mean_snr_db")
            if rec and snr is not None and cfg.epochs > rec.get("max", 0):
                console.print(
                    f"[yellow]official guidance:[/yellow] dataset measured "
                    f"SNR {snr} dB ({rec.get('quality')}) — FAQ Q9 recommends "
                    f"[b]{rec.get('min')}-{rec.get('max')} epochs[/b]; you asked "
                    f"for {cfg.epochs}. {rec.get('reason')}"
                )
                console.print(
                    "[yellow]training anyway (--epochs was explicit); if "
                    "val_mel/speaker-sim plateau, stop early via "
                    "'rvc-platform train stop' (checkpoint kept).[/yellow]"
                )
    except Exception:  # noqa: BLE001 — guidance is best-effort, never blocks
        pass

    from rvc_platform.training.worker import main as worker_main
    # run in-process so the CLI shows live JSONL events; the worker registers
    # its pid in temp/trainer.pid so `train stop` reaches it from any terminal
    cfg.save()

    def _forward_events() -> None:
        # the trainer runs in this process; worker_main blocks until done
        code = worker_main(["--config", str(run_dir / "run_config.json")])
        raise SystemExit(code)

    try:
        _forward_events()
    except KeyboardInterrupt:
        # the worker installs its own SIGINT handler (cooperative stop +
        # checkpoint save); this branch only guards a race before install
        console.print("[yellow]interrupted[/yellow]")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise


@train_app.command("evaluate")
def train_evaluate(
    experiment: str = typer.Argument(..., help="Experiment (run) name under runs/"),
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint",
        help="G checkpoint to score: 'latest', G_<step>.pth name, absolute path, "
             "or an exports/*.pth inference file (default: best export if any, else latest)",
    ),
    val_split: Optional[float] = typer.Option(
        None, "--val-split", help="held-out fraction (defaults to the run config)",
    ),
    batch: Optional[int] = typer.Option(None, "--batch", help="validation batch size"),
    gpu: Optional[int] = typer.Option(None, "--gpu"),
):
    """Objective evaluation of a generator on held-out data (UPGRADE_REPORT_AR.md).

    Reports val_mel (reconstruction), val_spk (ECAPA speaker similarity, when
    speechbrain is installed), val_f0_rmse (pitch, when rmvpe asset exists) and
    the composite selection score; writes runs/<exp>/eval_report.json.
    """
    import torch

    from rvc_platform.core.errors import ConfigurationError
    from rvc_platform.training.best_model import (
        ValidationEvaluator,
        derive_validation_split,
    )
    from rvc_platform.training.run_config import TrainingRunConfig
    from rvc_platform.training.trainer import build_generator
    from rvc_platform.utils.device import select_device

    run_dir = get_paths().runs / experiment
    if not (run_dir / "run_config.json").is_file():
        console.print(f"[red]no run config under {run_dir}[/red]")
        raise typer.Exit(2)
    cfg = TrainingRunConfig.load(run_dir)
    hps = cfg.to_hparams()
    hps.data["training_files"] = str(run_dir / "filelist.txt")
    if batch is not None:
        hps.train["batch_size"] = batch

    split = derive_validation_split(
        Path(hps.data["training_files"]),
        float(val_split if val_split is not None else cfg.val_split),
    )
    if split is None or split.n_val == 0:
        console.print("[red]no validation split available — pass --val-split > 0 "
                      "on a run with a filelist[/red]")
        raise typer.Exit(2)

    # ---- resolve the checkpoint to score -----------------------------------
    ckpt_dir = run_dir / "checkpoints"
    candidates: list[Path] = []
    if checkpoint:
        raw = Path(checkpoint)
        candidates.append(raw if raw.is_absolute() else ckpt_dir / checkpoint)
        if str(checkpoint).strip().lower() in ("latest", "final"):
            candidates.append(ckpt_dir / "G_final.pth")
    else:
        best_json = ckpt_dir / "best.json"
        if best_json.is_file():
            try:
                import json as _json

                meta = _json.loads(best_json.read_text(encoding="utf-8"))
                name = meta.get("export")
                if name:
                    candidates.append(run_dir / "exports" / name)
            except (OSError, ValueError):
                pass
        candidates.append(ckpt_dir / "G_final.pth")
        numbered = sorted(ckpt_dir.glob("G_*.pth"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(numbered[:2])
    weight_path = next((p for p in candidates if p.is_file()), None)
    if weight_path is None:
        console.print(f"[red]no checkpoint found for {experiment} "
                      f"(tried {[str(c) for c in candidates]})[/red]")
        raise typer.Exit(2)

    device = select_device(gpu).device
    net_g = build_generator(hps, device)
    payload = torch.load(weight_path, map_location="cpu", weights_only=False)
    if "model" in payload:
        state = payload["model"]
    elif "weight" in payload:
        state = {k: v.float() for k, v in payload["weight"].items()}
    else:
        raise ConfigurationError(f"{weight_path} has neither 'model' nor 'weight'")
    _missing, unexpected = net_g.load_state_dict(state, strict=False)
    if unexpected:
        console.print(f"[yellow]ignoring {len(unexpected)} unexpected keys[/yellow]")

    evaluator = ValidationEvaluator(hps, device, False, split.val_path)
    scores = evaluator.evaluate(net_g)
    if not scores:
        console.print("[red]evaluation produced no scores (empty validation set?)[/red]")
        raise typer.Exit(2)

    report = {
        "experiment": experiment,
        "checkpoint": str(weight_path),
        "epoch": int(payload.get("epoch", payload.get("iteration", 0)) or 0),
        "device": device,
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in scores.items()},
        "metric_semantics": {
            "val_mel": "held-out mel L1 reconstruction (lower is better)",
            "val_spk": "ECAPA speaker-similarity (higher is better; needs speechbrain)",
            "val_f0_rmse": "voiced F0 RMSE Hz (lower is better; needs rmvpe asset)",
            "composite": "val_mel + spk_weight*(1-val_spk) — selection metric (lower is better)",
        },
        "evaluated_at": time.time(),
    }
    out_path = run_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    console.print_json(json.dumps(report, ensure_ascii=False))
    console.print(f"[green]report saved: {out_path}[/green]")


@train_app.command("stop")
def train_stop():
    """Send SIGTERM to the running trainer subprocess (cooperative stop)."""
    pid = _read_pid("trainer")
    if pid is None:
        console.print("[yellow]no trainer process registered[/yellow]")
        return
    os.kill(pid, signal.SIGTERM)
    console.print(f"SIGTERM -> trainer pid {pid}")


@train_app.command("checkpoints")
def train_checkpoints(experiment: str = typer.Argument(...)):
    from rvc_platform.training import checkpointing

    ckpts = checkpointing.list_checkpoints(get_paths().runs / experiment / "checkpoints")
    table = Table(title=f"checkpoints: {experiment}")
    for col in ("kind", "epoch", "global_step", "size_mb", "path"):
        table.add_column(col)
    for c in ckpts:
        table.add_row(c.kind, str(c.epoch), str(c.global_step),
                      str(round(c.size_mb, 1)), str(c.path))
    console.print(table)


@train_app.command("export")
def train_export(experiment: str, epoch: Optional[int] = typer.Option(None)):
    from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
    from rvc_platform.application.training_service import TrainingService
    from rvc_platform.infrastructure.processes.manager import ProcessManager

    svc = TrainingService(ProcessManager(get_paths().logs / "processes"))
    result = svc.export(experiment, epoch, model_repo=ModelRepository(get_paths().models))
    console.print_json(json.dumps(result, ensure_ascii=False))


def _read_pid(name: str) -> int | None:
    f = get_paths().temp / f"{name}.pid"
    try:
        pid = int(f.read_text().strip())
        import psutil

        if psutil.pid_exists(pid):
            return pid
    except (OSError, ValueError):
        pass
    return None


# --------------------------------------------------------------------------- #
# colab / drive persistence
# --------------------------------------------------------------------------- #
colab_app = typer.Typer(help="Colab/Drive persistence utilities")
app.add_typer(colab_app, name="colab")


@colab_app.command("restore")
def colab_restore(drive_dir: str = typer.Option("RVC_Iraqi", "--drive-dir")):
    """Pull runs/ + models/ from Google Drive into this session."""
    from rvc_platform.colab import colab_runner

    console.print_json(json.dumps(colab_runner.restore(drive_dir), default=str))


@colab_app.command("snapshot")
def colab_snapshot(drive_dir: str = typer.Option("RVC_Iraqi", "--drive-dir")):
    """Push runs/ + models/ to Google Drive (run before a session ends)."""
    from rvc_platform.colab import colab_runner

    console.print_json(json.dumps(colab_runner.snapshot(drive_dir), default=str))


@colab_app.command("status")
def colab_status(drive_dir: str = typer.Option("RVC_Iraqi", "--drive-dir")):
    """Drive availability, active redirects and known checkpoints."""
    from rvc_platform.colab import colab_runner

    console.print_json(json.dumps(colab_runner.status(drive_dir), default=str))


@colab_app.command("env")
def colab_env(drive_dir: str = typer.Option("RVC_Iraqi", "--drive-dir")):
    """Print the recommended Drive env wiring for this session."""
    from rvc_platform.colab import colab_runner

    console.print_json(json.dumps(colab_runner.colab_env(drive_dir), default=str))


# --------------------------------------------------------------------------- #
# models / inference
# --------------------------------------------------------------------------- #
model_app = typer.Typer(help="model management")
app.add_typer(model_app, name="model")


@model_app.command("list")
def model_list():
    from rvc_platform.infrastructure.persistence.model_repository import ModelRepository

    models = ModelRepository(get_paths().models).list_models()
    if not models:
        console.print("[yellow]no models yet — train one or upload a .pth[/yellow]")
        return
    table = Table(title="voice models")
    for col in ("name", "sr", "f0", "version", "size_mb", "info", "index"):
        table.add_column(col)
    for m in models:
        table.add_row(m.name, str(m.sample_rate), str(m.f0), m.version,
                      str(round(m.size_mb, 1)), m.info,
                      str(m.index_path) if m.index_path else "—")
    console.print(table)


@model_app.command("info")
def model_info(name: str):
    from rvc_platform.engine.rvc.checkpoint import inspect_checkpoint

    console.print_json(json.dumps(inspect_checkpoint(get_paths().models / name),
                                  ensure_ascii=False))


@app.command("inference")
def inference(
    input: Path = typer.Argument(..., help="audio file"),
    model: str = typer.Argument(..., help="model name in models/"),
    output: Path = typer.Option(None, "--output", "-o"),
    pitch: int = typer.Option(0, "--pitch", "-t"),
    f0_method: str = typer.Option("rmvpe", "--f0-method"),
    index_rate: float = typer.Option(0.0, "--index-rate"),
    filter_radius: int = typer.Option(3, "--filter-radius"),
    rms_mix_rate: float = typer.Option(0.25, "--rms"),
    protect: float = typer.Option(0.33, "--protect"),
    output_sr: Optional[int] = typer.Option(None, "--output-sr"),
    denoise: bool = typer.Option(False, "--denoise"),
):
    """Convert one audio file to the target voice (Audio-to-Audio)."""
    from rvc_platform.core.entities import ConversionSettings
    from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
    from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
    from rvc_platform.engine.rvc.inference import RVCInferenceEngine
    from rvc_platform.utils.device import select_device

    repo = ModelRepository(get_paths().models)
    info = repo.get(model)
    profile = select_device(None)
    engine = RVCInferenceEngine(profile, AudioProcessingEngine())
    index_path = info.index_path if info.index_path else None
    engine.load_model(info.path, index_path)
    settings = ConversionSettings(
        pitch_shift=pitch, f0_method=f0_method, index_rate=index_rate,
        filter_radius=filter_radius, rms_mix_rate=rms_mix_rate, protect=protect,
        output_sample_rate=output_sr, denoise=denoise,
    )
    out = output or get_paths().data / "conversions" / f"{Path(input).stem}_{model}.wav"
    result = engine.convert_file(Path(input), Path(out), settings)
    console.print(f"[green]converted -> {result.output_path}[/green]")
    console.print_json(json.dumps(result.speed_report))


# --------------------------------------------------------------------------- #
# web / realtime / tunnel / process control
# --------------------------------------------------------------------------- #
@app.command("web")
def web(
    host: str = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    tunnel: bool = typer.Option(False, "--tunnel", help="also start cloudflared"),
):
    """Launch the web API + dashboard (blocks; Ctrl-C to stop)."""
    from rvc_platform.infrastructure.config.loader import get_config

    config = get_config()
    host = host or (config.server.host if not tunnel else "0.0.0.0")
    port = port or config.server.port
    if _port_open(port):
        console.print(f"[yellow]port {port} already in use — server may be running[/yellow]")
    get_paths().temp.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Iraqi Voice Studio -> http://{host}:{port}[/green]")
    cmd = [sys.executable, *UVICORN_ARGS, "--host", host, "--port", str(port)]
    # new session: uvicorn (and its tunnel child) get their own process group so
    # `web-stop` can terminate the whole tree with one SIGTERM->SIGKILL sequence
    proc = subprocess.Popen(cmd, start_new_session=True)
    _PID_FILE.write_text(str(proc.pid))
    if tunnel:
        time.sleep(2.5)
        _start_tunnel(port)
    try:
        proc.wait()
    except KeyboardInterrupt:
        _terminate_group(proc.pid)
        proc.wait()
    finally:
        try:
            _PID_FILE.unlink()
        except OSError:
            pass


def _terminate_group(pid: int) -> None:
    """SIGTERM the process group, escalate to SIGKILL after 8 s."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _start_tunnel(port: int) -> None:
    import subprocess as sp

    log = get_paths().logs / "tunnel.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        proc = sp.Popen(
            ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}",
             "--no-autoupdate"],
            stdout=fh, stderr=sp.STDOUT, start_new_session=True,
        )
    (get_paths().temp / "tunnel.pid").write_text(str(proc.pid))
    import re

    url = None
    deadline = time.time() + 30
    while time.time() < deadline and url is None:
        data = log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", data)
        url = match.group(0) if match else None
        if url is None:
            time.sleep(0.5)
    if url:
        console.print(f"[green]public URL: {url}[/green]")
        (get_paths().temp / "tunnel.url").write_text(url)
    else:
        console.print("[red]tunnel URL not detected; see logs/tunnel.log[/red]")


def _stop_tunnel() -> bool:
    """Terminate a CLI-spawned cloudflared (if any). Returns stopped?"""
    pid_file = get_paths().temp / "tunnel.pid"
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        _terminate_group(pid)
    except (ProcessLookupError, PermissionError):
        pass
    pid_file.unlink(missing_ok=True)
    (get_paths().temp / "tunnel.url").unlink(missing_ok=True)
    return True


@app.command("web-status")
def web_status(port: Optional[int] = None):
    from rvc_platform.infrastructure.config.loader import get_config

    port = port or get_config().server.port
    if _port_open(port):
        status = _api(port) or {}
        _print_status(status)
    else:
        console.print("[yellow]web server is not running[/yellow]")


@app.command("web-stop")
def web_stop(port: Optional[int] = None):
    from rvc_platform.infrastructure.config.loader import get_config

    port = port or get_config().server.port
    if _stop_tunnel():
        console.print("[green]stopped CLI tunnel[/green]")
    pid = _web_pid()
    if pid and pid != os.getpid():
        try:
            _terminate_group(pid)
            console.print(f"[green]stopped web pid {pid}[/green]")
            _PID_FILE.unlink(missing_ok=True)
            return
        except ProcessLookupError:
            pass
    # fallback: find uvicorn by port
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "uvicorn" in cmdline and f"--port {port}" in cmdline.replace("  ", " "):
                proc.terminate()
                console.print(f"[green]terminated {proc.info['pid']}[/green]")
        _PID_FILE.unlink(missing_ok=True)
    except ImportError:
        console.print("[yellow]install psutil for reliable stop[/yellow]")


@app.command("realtime-benchmark")
def realtime_benchmark(
    model: str = typer.Argument(..., help="f0 model name in models/"),
    blocks: int = typer.Option(40, "--blocks", help="number of synthetic blocks"),
    block_ms: int = typer.Option(160, "--block-ms"),
    crossfade_ms: int = typer.Option(50, "--crossfade-ms",
                                     help="block crossfade (upstream 0.05 s)"),
):
    """Measure REAL engine latency with synthetic blocks (no mic needed)."""
    from rvc_platform.engine.rvc.realtime import RealTimeRVC
    from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
    from rvc_platform.utils.device import select_device

    repo = ModelRepository(get_paths().models)
    info = repo.get(model)
    if not info.f0:
        console.print("[red]real-time requires an f0 (pitch-guided) model[/red]")
        raise typer.Exit(2)
    profile = select_device(None)
    engine = RealTimeRVC(profile, block_ms=block_ms, crossfade_ms=crossfade_ms)
    console.print(f"loading {info.name} on {profile.device} ...")
    engine.load_model(info.path, info.index_path)
    import numpy as np

    rng = np.random.default_rng(7)
    freq = 16000
    chunk = engine.block_frame_16k
    console.print(f"feeding {blocks} synthetic blocks ({block_ms} ms each) ...")
    for i in range(blocks):
        t = np.arange(chunk) / freq + i * block_ms / 1000.0
        wave = 0.2 * np.sin(2 * np.pi * 150 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        wave += 0.02 * rng.standard_normal(chunk).astype(np.float32)
        engine.process_block(wave.astype(np.float32))
    stats = engine.latency()
    console.print_json(json.dumps({
        "device": profile.device,
        "block_ms": block_ms,
        "samples": stats.count,
        "min_ms": stats.min_ms,
        "avg_ms": stats.avg_ms,
        "p95_ms": stats.p95_ms,
        "max_ms": stats.max_ms,
        "verdict": (
            "real-time OK (avg < block budget)" if stats.avg_ms < block_ms
            else f"WARNING: avg {stats.avg_ms} ms exceeds block budget {block_ms} ms"
        ),
    }))


@app.command("status")
def status(port: Optional[int] = None):
    """Overall platform status (via running web API or offline snapshot)."""
    from rvc_platform.infrastructure.config.loader import get_config

    port = port or get_config().server.port
    if _port_open(port):
        _print_status(_api(port) or {})
    else:
        from rvc_platform.utils.environment import detect_system

        prof = detect_system()
        console.print_json(json.dumps(prof.summary, ensure_ascii=False))
        console.print("[dim]web server offline — start with `make web`[/dim]")


@app.command("logs")
def logs(lines: int = typer.Option(80, "--lines")):
    log_file = get_paths().logs / "platform.log"
    if log_file.is_file():
        tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        console.print("\n".join(tail))
    else:
        console.print("[yellow]no platform.log yet — start the web server (`make web`)[/yellow]")


def app_run() -> None:  # entry point
    app()


if __name__ == "__main__":
    app_run()
