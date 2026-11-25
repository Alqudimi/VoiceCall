"""RVCInferenceEngine — file-mode Audio-to-Audio conversion.

Faithful re-implementation of the upstream conversion pipeline
(``infer/vc/pipeline.py``) on our architecture:

    load@16k -> highpass -> [denoise] -> pad -> smart split (quiet-point)
    -> per slice: HuBERT feats -> FAISS blend -> upsample x2 -> F0
    -> voiceless-protect blend -> SynthesizerTrn.infer -> trim pads
    -> concat -> RMS envelope mixing -> resample -> peak-safe write

Iraqi-dialect quality rationale (docs/TRAINING.md): RMVPE default, index_rate
tunable per recording, and `protect` guarding voiceless consonants
(عربية emphatics and Iraqi /q/ realisations stay crisp).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from rvc_platform.core.entities import (
    ConversionResult,
    ConversionSettings,
    EngineStatus,
    ModelInfo,
)
from rvc_platform.core.errors import (
    AssetMissingError,
    ConfigurationError,
    ConversionError,
    ModelNotFoundError,
)
from rvc_platform.engine.rvc import checkpoint as ckpt_mod
from rvc_platform.engine.rvc import f0 as f0_mod
from rvc_platform.engine.rvc.hubert import extract_hubert_features, load_hubert_model
from rvc_platform.engine.rvc.index import FeatureIndex
from rvc_platform.infrastructure.audio import pipeline as audio_engine_mod
from rvc_platform.infrastructure.paths import get_paths
from rvc_platform.utils.device import DeviceProfile

logger = logging.getLogger(__name__)

_SR_HUBERT = 16_000
_WINDOW = 160  # 10 ms frames at 16 kHz


def change_rms(
    src_audio: np.ndarray, src_sr: int,
    dst_audio: np.ndarray, dst_sr: int,
    rate: float,
) -> np.ndarray:
    """Blend the output loudness envelope with the source one (smooth 125 ms grid)."""
    frame1 = max(2, (src_sr // 4) * 2)
    hop1 = max(1, src_sr // 8)
    frame2 = max(2, (dst_sr // 4) * 2)
    hop2 = max(1, dst_sr // 8)
    rms1 = librosa.feature.rms(y=src_audio, frame_length=frame1, hop_length=hop1)
    rms2 = librosa.feature.rms(y=dst_audio, frame_length=frame2, hop_length=hop2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=dst_audio.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=dst_audio.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    dst_audio = dst_audio * (
        torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return dst_audio


def _shift_formants_praat(audio_16k: np.ndarray, formant_shift_semitones: float) -> np.ndarray:
    """Shift vocal-tract formants independently of F0 via Praat PSOLA (best-effort)."""
    if abs(formant_shift_semitones) < 0.05 or audio_16k.size < 320:
        return audio_16k
    try:
        import parselmouth

        ratio = float(2.0 ** (formant_shift_semitones / 12.0))
        snd = parselmouth.Sound(np.asarray(audio_16k, dtype=np.float64), sampling_frequency=_SR_HUBERT)
        shifted = parselmouth.praat.call(
            snd, "Change gender", 50.0, 1100.0, ratio, 0.0, 1.0, 1.0
        )
        out = shifted.values[0].astype(np.float32)
        if out.shape[0] == audio_16k.shape[0]:
            return out
        if out.shape[0] > audio_16k.shape[0]:
            return out[: audio_16k.shape[0]]
        return np.pad(out, (0, audio_16k.shape[0] - out.shape[0])).astype(np.float32)
    except Exception:
        logger.debug("formant shift unavailable — continuing without it", exc_info=True)
        return audio_16k


class RVCInferenceEngine:
    """VoiceModelEngine implementation around a single loaded RVC model."""

    def __init__(
        self,
        device_profile: DeviceProfile,
        audio: audio_engine_mod.AudioProcessingEngine,
        cuda_graph: bool = False,
    ) -> None:
        self.profile = device_profile
        self.device = device_profile.device
        self.is_half = device_profile.is_half and self.device.startswith("cuda")
        self.audio = audio
        self.x_pad, self.x_query, self.x_center, self.x_max = device_profile.split_params
        self._lock = threading.RLock()
        self._hubert: torch.nn.Module | None = None
        self._f0_cache = f0_mod.CachedF0()
        self.net_g: torch.nn.Module | None = None
        self.cpt: dict | None = None
        self.index: FeatureIndex | None = None
        self.model_info: ModelInfo | None = None
        self.tgt_sr = 40_000
        self.if_f0 = 1
        self.version = "v2"
        self._status = EngineStatus.UNAVAILABLE
        self._status_message = "no model loaded"

    # ------------------------------------------------------------------ status
    @property
    def status(self) -> str:
        return self._status.value

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def loaded_model(self) -> ModelInfo | None:
        return self.model_info

    # ------------------------------------------------------------------ assets
    def _ensure_hubert(self) -> torch.nn.Module:
        if self._hubert is None:
            hubert_dir = get_paths().hubert
            if not (hubert_dir / "config.json").is_file():
                raise AssetMissingError(
                    "HuBERT content encoder (assets/hubert_base)",
                    "run `make download-models` or `rvc-platform doctor --fix`",
                )
            logger.info("loading HuBERT content encoder (%s)", self.device)
            self._hubert = load_hubert_model(self.device, self.is_half)
        return self._hubert

    def share_hubert(self, hubert: torch.nn.Module | None) -> None:
        """Reuse an already-loaded encoder (e.g. from the realtime engine)."""
        if hubert is not None:
            self._hubert = hubert

    # ------------------------------------------------------------------ load
    def load_model(self, model_path: Path, index_path: Path | None) -> ModelInfo:
        with self._lock:
            self._status = EngineStatus.LOADING
            try:
                net_g, cpt = ckpt_mod.load_checkpoint(
                    Path(model_path), self.device, self.is_half
                )
                if index_path is not None and Path(index_path).is_file():
                    try:
                        self.index = FeatureIndex(Path(index_path))
                    except Exception as exc:
                        logger.warning("index load failed (%s) — continuing without", exc)
                        self.index = None
                else:
                    self.index = None
                self.net_g = net_g
                self.cpt = cpt
                self.tgt_sr = int(cpt["config"][-1])
                self.if_f0 = int(cpt.get("f0", 1))
                self.version = str(cpt.get("version", "v1"))
                path = Path(model_path)
                stat = path.stat()
                self.model_info = ModelInfo(
                    name=path.stem,
                    path=path,
                    sample_rate=self.tgt_sr,
                    f0=self.if_f0 == 1,
                    version=self.version,
                    info=str(cpt.get("info", "")),
                    size_mb=stat.st_size / (1024 * 1024),
                    created_at=stat.st_mtime,
                    index_path=Path(index_path) if index_path else None,
                )
                self._status = EngineStatus.READY
                self._status_message = (
                    f"{self.model_info.name} "
                    f"(sr={self.tgt_sr}, f0={self.if_f0}, {self.version})"
                )
                logger.info("model loaded: %s", self._status_message)
                return self.model_info
            except (ModelNotFoundError, ConversionError, Exception) as exc:
                self._status = EngineStatus.ERROR
                self._status_message = str(exc)
                raise

    def unload(self) -> None:
        with self._lock:
            self.net_g = None
            self.cpt = None
            self.index = None
            self.model_info = None
            self._status = EngineStatus.UNAVAILABLE
            self._status_message = "no model loaded"
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                torch.cuda.empty_cache()

    def _require_model(self) -> None:
        if self.net_g is None:
            raise ConversionError(
                f"no model loaded ({self._status_message}); load a model first"
            )

    # ------------------------------------------------------------------ F0
    def _extract_raw_f0(self, audio_16k: np.ndarray, p_len: int, method: str,
                        filter_radius: int) -> np.ndarray:
        extractor = self._f0_cache.get(method, self.device, self.is_half)
        f0 = extractor.extract(np.ascontiguousarray(audio_16k))
        if f0.shape[0] < p_len:
            f0 = np.pad(f0, (0, p_len - f0.shape[0]))
        f0 = f0[:p_len]
        # Clean isolated 1-2 frame octave spikes while PRESERVING unvoiced 0.0
        # frames so SineGen U/V excitation and voiceless `protect` both work!
        f0 = f0_mod.clean_f0_octaves(f0)
        f0 = f0_mod.median_filter_f0(f0, filter_radius)
        return f0

    def get_f0(self, audio_16k: np.ndarray, p_len: int, semitones: int,
               method: str, filter_radius: int):
        f0 = self._extract_raw_f0(audio_16k, p_len, method, filter_radius)
        f0 = f0_mod.shift_f0(f0, semitones)
        coarse, fine = f0_mod.f0_to_coarse(f0)
        return coarse.astype(np.int64), fine.astype(np.float32)

    # ------------------------------------------------------------------ core
    def _vc(self, audio0: np.ndarray, pitch, pitchf, settings: ConversionSettings) -> np.ndarray:
        feats = torch.from_numpy(np.ascontiguousarray(audio0, dtype=np.float32))
        if self.is_half:
            feats = feats.half()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        with torch.no_grad():
            feats = extract_hubert_features(
                self._ensure_hubert(),
                feats.to(self.device),
                self.version,
                padding_mask=padding_mask,
            )
        feats0 = None
        if settings.protect < 0.5 and pitch is not None:
            feats0 = feats.clone()

        if self.index is not None and settings.index_rate > 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            blended = self.index.blend(npy, settings.index_rate)
            feats = (
                torch.from_numpy(blended).unsqueeze(0).to(self.device)
                if not self.is_half else
                torch.from_numpy(blended.astype("float16")).unsqueeze(0).to(self.device)
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if feats0 is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = audio0.shape[0] // _WINDOW
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        if feats0 is not None and pitch is not None and pitchf is not None:
            # voiceless-consonant protection (upstream logic)
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = settings.protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        p_len_tensor = torch.tensor([p_len], device=self.device).long()
        sid = torch.tensor([0], device=self.device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                synthesized = self.net_g.infer(feats, p_len_tensor, pitch, pitchf, sid)[0]
            else:
                synthesized = self.net_g.infer(feats, p_len_tensor, sid)[0]
            audio1 = synthesized[0, 0].data.cpu().float().numpy()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return audio1

    # ------------------------------------------------------------------ public
    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        settings: ConversionSettings,
        progress: Callable[[float], None] | None = None,
    ) -> ConversionResult:
        self._require_model()
        settings.validate()
        started = time.perf_counter()
        input_path, output_path = Path(input_path), Path(output_path)

        audio16k = self.audio.load_any(input_path, _SR_HUBERT)
        if audio16k.size == 0:
            raise ConversionError(f"empty audio: {input_path.name}")
        duration = audio16k.size / _SR_HUBERT
        # upstream input peak-guard (infer/vc/modules.py vc_single):
        # scale down ONLY when peaks exceed 0.95 — clipped input saturates
        # HuBERT features before conversion even starts.
        audio_max = float(np.abs(audio16k).max() / 0.95)
        if audio_max > 1.0:
            audio16k = (audio16k / audio_max).astype(np.float32)

        # pre-pass conditioning (optional denoise is ours, highpass = upstream)
        # filtfilt (zero-phase) is the official filter: lfilter's causal
        # phase distortion smears HuBERT features at onsets/attacks.
        bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=_SR_HUBERT)
        audio16k = signal.filtfilt(bh, ah, audio16k)
        if settings.denoise:
            audio16k = self.audio.denoise(audio16k, _SR_HUBERT)
        if abs(getattr(settings, "formant_shift", 0.0)) >= 0.05:
            audio16k = _shift_formants_praat(audio16k, float(settings.formant_shift))

        sr = self.tgt_sr
        t_pad = self.x_pad * _SR_HUBERT
        t_pad_tgt = self.x_pad * sr
        t_pad2 = t_pad * 2
        t_query = self.x_query * _SR_HUBERT
        t_center = self.x_center * _SR_HUBERT
        t_max = self.x_max * _SR_HUBERT

        audio_pad = np.pad(audio16k.astype(np.float32), (_WINDOW // 2, _WINDOW // 2), mode="reflect")
        opt_ts: list[int] = []
        if audio_pad.shape[0] > t_max:
            audio_sum = np.zeros_like(audio16k)
            for i in range(_WINDOW):
                audio_sum += np.abs(audio_pad[i : i - _WINDOW])
            for t in range(t_center, audio16k.shape[0], t_center):
                opt_ts.append(
                    t - t_query
                    + np.where(
                        audio_sum[t - t_query : t + t_query]
                        == audio_sum[t - t_query : t + t_query].min()
                    )[0][0]
                )
        audio_pad = np.pad(audio16k.astype(np.float32), (t_pad, t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // _WINDOW

        pitch = pitchf = None
        if self.if_f0 == 1:
            effective_pitch = float(settings.pitch_shift)
            if getattr(settings, "auto_pitch", False):
                f0_stats = (self.cpt or {}).get("f0_stats") if isinstance(self.cpt, dict) else None
                tgt_med = float((f0_stats or {}).get("median_hz") or 0.0)
                if tgt_med > 20.0:
                    raw_f0 = self._extract_raw_f0(
                        audio_pad, p_len, settings.f0_method, settings.filter_radius
                    )
                    voiced = raw_f0[raw_f0 > 20.0]
                    if voiced.size >= 5:
                        src_med = float(np.median(voiced))
                        auto_semi = float(np.clip(
                            round(12.0 * np.log2(tgt_med / src_med)), -18.0, 18.0
                        ))
                        logger.info("auto-pitch: source median=%.1f Hz -> target=%.1f Hz => %+g st",
                                    src_med, tgt_med, auto_semi)
                        effective_pitch += auto_semi
            coarse, fine = self.get_f0(
                audio_pad, p_len, effective_pitch, settings.f0_method,
                settings.filter_radius,
            )
            pitch = torch.tensor(coarse, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(fine, device=self.device).unsqueeze(0).float()

        chunks: list[np.ndarray] = []
        total_chunks = max(1, len(opt_ts) + 1)
        s = 0
        t = None
        for t in opt_ts + [None]:
            if t is None:
                seg = audio_pad[s:]
                seg_pitch = pitch[:, s // _WINDOW :] if pitch is not None else None
                seg_pitchf = pitchf[:, s // _WINDOW :] if pitchf is not None else None
            else:
                t = t // _WINDOW * _WINDOW
                seg = audio_pad[s : t + t_pad2 + _WINDOW]
                seg_pitch = pitch[:, s // _WINDOW : (t + t_pad2) // _WINDOW] if pitch is not None else None
                seg_pitchf = pitchf[:, s // _WINDOW : (t + t_pad2) // _WINDOW] if pitchf is not None else None
                s = t
            out = self._vc(seg, seg_pitch, seg_pitchf, settings)[t_pad_tgt : -t_pad_tgt]
            chunks.append(out)
            if progress:
                progress(len(chunks) / total_chunks)

        if len(chunks) > 1:
            # Tiny 1 ms seam smoothing across split boundaries without altering sample alignment
            edge = min(48, min(len(c) for c in chunks) // 4)
            if edge > 2:
                fade_in = np.linspace(0.0, 1.0, edge, dtype=np.float32)
                fade_out = 1.0 - fade_in
                for idx in range(len(chunks) - 1):
                    mid = 0.5 * (chunks[idx][-1] + chunks[idx + 1][0])
                    chunks[idx][-edge:] = chunks[idx][-edge:] * fade_out + mid * fade_in
                    chunks[idx + 1][:edge] = chunks[idx + 1][:edge] * fade_in + mid * fade_out
            audio_opt = np.concatenate(chunks)
        else:
            audio_opt = chunks[0] if chunks else np.zeros(0, dtype=np.float32)
        if settings.rms_mix_rate != 1 and audio_opt.size:
            audio_opt = change_rms(audio16k, _SR_HUBERT, audio_opt, sr, settings.rms_mix_rate)
        if settings.output_sample_rate and settings.output_sample_rate >= 16_000 and settings.output_sample_rate != sr:
            audio_opt = librosa.resample(audio_opt, orig_sr=sr, target_sr=settings.output_sample_rate)
            sr = settings.output_sample_rate
        if settings.normalize:
            audio_opt = self.audio.normalize(audio_opt, 0.96)
        peak = float(np.max(np.abs(audio_opt))) if audio_opt.size else 0.0
        if peak > 0.99:
            audio_opt = audio_opt / peak * 0.96

        self.audio.write_wav(output_path, audio_opt.astype(np.float32), sr)
        processing = time.perf_counter() - started
        return ConversionResult(
            input_path=input_path,
            output_path=output_path,
            duration_seconds=duration,
            processing_seconds=processing,
            settings=settings,
            realtime_factor=processing / duration if duration else 0.0,
        )
