"""RealTimeRVC — true streaming voice conversion engine.

Implements the upstream real-time algorithm
(``infer/rtrvc.py`` + ``realtime_gui.py`` driving loop):

* fixed 10 ms frame grid (``zc``) for every buffer size;
* sliding input window = extra (F0 context) + crossfade + search + block;
* per-block HuBERT + pitch with a 1024-frame pitch cache for continuity;
* ``skip_head``/``return_length`` slicing so only the new block is emitted;
* SOLA (overlap-add search) output crossfade — borrowed from DDSP-SVC via
  upstream, credited in ATTRIBUTION.md;
* optional input noise gate and RMS-envelope mixing.

Our additions: per-block latency instrumentation (min/avg/p95/max),
safe mid-stream parameter updates (pitch, index_rate, f0 method), and a
pure-Python driving loop decoupled from any audio device backend — the
transport (WebSocket) is supplied by the realtime service layer.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from rvc_platform.core.entities import EngineStatus, LatencyRecord, LatencyStats, ModelInfo
from rvc_platform.core.errors import RealtimeError
from rvc_platform.engine.rvc import checkpoint as ckpt_mod
from rvc_platform.engine.rvc import f0 as f0_mod
from rvc_platform.engine.rvc.hubert import extract_hubert_features, load_hubert_model
from rvc_platform.engine.rvc.index import FeatureIndex
from rvc_platform.engine.rvc.speaker_guard import SpeakerGuard
from rvc_platform.utils.device import DeviceProfile

logger = logging.getLogger(__name__)

_SR = 16_000
_UNIT = 160            # 10 ms @ 16 kHz
_PITCH_CACHE = 1024


class RealTimeRVC:
    """One realtime conversion session bound to one loaded model."""

    def __init__(
        self,
        profile: DeviceProfile,
        block_ms: int = 160,
        crossfade_ms: int = 50,
        extra_ms: int = 2500,
        silence_threshold_db: float = -60.0,
        f0_method: str = "rmvpe",
    ) -> None:
        self.profile = profile
        self.device = profile.device
        self.is_half = profile.is_half and self.device.startswith("cuda")
        # ---- block geometry (all in 10 ms units like upstream) ----
        self.block_units = max(1, round(block_ms / 10))
        self.crossfade_units = max(0, round(crossfade_ms / 10))
        self.sola_search_units = 1
        self.extra_units = max(1, round(extra_ms / 10))
        self.block_frame_16k = self.block_units * _UNIT
        self.sola_buffer_units = min(self.crossfade_units, 4 * 1)  # upstream cap
        self.sila_search_frame = self.sola_search_units * _UNIT
        self.skip_head = self.extra_units
        self.return_units = (
            self.block_units + self.sola_buffer_units + self.sola_search_units
        )
        # ---- buffers ----
        total_units = self.extra_units + self.crossfade_units + self.sola_search_units + self.block_units
        self.input_wav = torch.zeros(total_units * _UNIT, device=self.device, dtype=torch.float32)
        self.cache_pitch = torch.zeros(_PITCH_CACHE, device=self.device, dtype=torch.long)
        self._rms_buffer = np.zeros(_UNIT * 4, dtype=np.float32)  # 40 ms gate history
        self._gate_hold_frames = 0
        self._gate_gain = 0.0
        self._silent_blocks = 0
        self.cache_pitchf = torch.zeros(_PITCH_CACHE, device=self.device, dtype=torch.float32)
        self.tgt_sr = 40_000
        self._init_sola_buffers(_UNIT)
        # ---- model slots ----
        self.net_g: torch.nn.Module | None = None
        self.hubert: torch.nn.Module | None = None
        self.index: FeatureIndex | None = None
        self.model_info: ModelInfo | None = None
        self.if_f0 = 1
        self.version = "v2"
        self.resample_kernel: dict[int, Resample] = {}
        # ---- live settings (mutex-guarded mid-stream changes) ----
        self._settings_lock = threading.Lock()
        self.pitch_shift = 0
        self.index_rate = 0.0
        self.f0_method = str(f0_method or "rmvpe")
        self.rms_mix_rate = 0.0
        self.silence_threshold_db = silence_threshold_db
        self.strict_isolation = False
        self.speaker_guard = SpeakerGuard()
        # ---- instrumentation ----
        self._records: deque[LatencyRecord] = deque(maxlen=512)
        self._blocks = 0
        self._status = EngineStatus.UNAVAILABLE

    def _init_sola_buffers(self, unit_samples: int) -> None:
        """Allocate SOLA crossfade buffers in target-sample-rate 10 ms units."""
        self._sola_unit = max(1, int(unit_samples))
        self.sila_search_frame = self.sola_search_units * self._sola_unit
        sola_len = self.sola_buffer_units * self._sola_unit
        self.sola_buffer = torch.zeros(sola_len, device=self.device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, sola_len, device=self.device, dtype=torch.float32)
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0, 1.0, steps=max(1, sola_len),
                    device=self.device, dtype=torch.float32,
                )
            ) ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self._fade_ok = self.sola_buffer_units > 0

    # ------------------------------------------------------------------ setup
    @property
    def status(self) -> str:
        return self._status.value

    def load_model(self, model_path: Path, index_path: Path | None) -> ModelInfo:
        net_g, cpt = ckpt_mod.load_checkpoint(Path(model_path), self.device, self.is_half)
        self.net_g = net_g
        self.cpt = cpt
        if self.hubert is None:
            hubert_dir = __import__(
                "rvc_platform.infrastructure.paths", fromlist=["get_paths"]
            ).get_paths().hubert
            if not (hubert_dir / "config.json").is_file():
                from rvc_platform.core.errors import AssetMissingError

                raise AssetMissingError(
                    "HuBERT content encoder (assets/hubert_base)",
                    "run `make download-models` or `rvc-platform doctor --fix`",
                )
            self.hubert = load_hubert_model(self.device, self.is_half)
        self.tgt_sr = int(cpt["config"][-1])
        self._init_sola_buffers(self.tgt_sr // 100)
        self.if_f0 = int(cpt.get("f0", 1))
        self.version = str(cpt.get("version", "v1"))
        if index_path is not None and Path(index_path).is_file():
            try:
                self.index = FeatureIndex(Path(index_path))
            except Exception as exc:
                logger.warning("index load failed (%s)", exc)
                self.index = None
        else:
            self.index = None
        path = Path(model_path)
        self.model_info = ModelInfo(
            name=path.stem, path=path, sample_rate=self.tgt_sr,
            f0=self.if_f0 == 1, version=self.version,
            info=str(cpt.get("info", "")),
            size_mb=path.stat().st_size / (1024 * 1024),
            created_at=path.stat().st_mtime,
            index_path=Path(index_path) if index_path else None,
        )
        # warm-up pass (first CUDA kernels are slow; also compiles RMVPE)
        try:
            probe = 0.05 * torch.sin(
                2 * np.pi * 220.0
                * torch.arange(self.input_wav.shape[0], device=self.device, dtype=torch.float32)
                / _SR
            )
            self.input_wav.copy_(probe)
            self._infer(self.input_wav)
            self.reset()
        except Exception as exc:
            logger.warning("warm-up failed (continuing): %s", exc)
        self._status = EngineStatus.READY
        return self.model_info

    def warmup(self) -> None:
        """Warm up full process_block (FAISS + F0 + HuBERT + SOLA) before live audio."""
        if self.net_g is None:
            return
        try:
            saved_strict = self.strict_isolation
            self.strict_isolation = False
            t = np.arange(self.block_frame_16k, dtype=np.float32) / _SR
            probe = (0.15 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
            self.process_block(probe)
            self.reset()
            self._records.clear()
            self._blocks = 0
            self.strict_isolation = saved_strict
        except Exception as exc:
            logger.warning("block warm-up failed (continuing): %s", exc)

    def reset(self) -> None:
        self.input_wav.zero_()
        self.cache_pitch.zero_()
        self.cache_pitchf.zero_()
        self.sola_buffer.zero_()
        self._rms_buffer.fill(0.0)
        self._gate_hold_frames = 0
        self._gate_gain = 0.0
        self._silent_blocks = 0
        self.speaker_guard.reset()

    def update_settings(self, **kwargs) -> None:
        """Safe mid-stream updates (applied between blocks)."""
        with self._settings_lock:
            if "pitch_shift" in kwargs:
                self.pitch_shift = int(kwargs["pitch_shift"])
            if "index_rate" in kwargs:
                self.index_rate = float(kwargs["index_rate"])
            if "f0_method" in kwargs:
                self.f0_method = str(kwargs["f0_method"])
            if "rms_mix_rate" in kwargs:
                self.rms_mix_rate = float(kwargs["rms_mix_rate"])
            if "silence_threshold_db" in kwargs:
                self.silence_threshold_db = float(kwargs["silence_threshold_db"])
            if "strict_isolation" in kwargs:
                self.strict_isolation = bool(kwargs["strict_isolation"])
            if "speaker_profile" in kwargs:
                prof = kwargs["speaker_profile"]
                if prof is None:
                    self.speaker_guard.clear_profile()
                else:
                    self.speaker_guard.set_profile(prof)

    def latency(self) -> LatencyStats:
        return LatencyStats.from_records(list(self._records))

    # ------------------------------------------------------------------ F0
    def _get_f0(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        with self._settings_lock:
            method = self.f0_method
            semitones = self.pitch_shift
        extractor = f0_mod.SHARED_CACHE.get(method, self.device, self.is_half)
        f0 = extractor.extract(np.ascontiguousarray(x))
        f0 = f0_mod.interp_unvoiced(f0)
        f0 = f0_mod.shift_f0(f0, semitones)
        coarse, fine = f0_mod.f0_to_coarse(f0)
        return (
            torch.tensor(coarse, device=self.device).long(),
            torch.tensor(fine, device=self.device).float(),
        )

    # ------------------------------------------------------------------ infer
    def _infer(self, input_wav: torch.Tensor) -> torch.Tensor:
        """Upstream RVC.infer: returns raw converted audio for the window."""
        with torch.no_grad():
            feats = input_wav.float().view(1, -1) if not self.is_half else input_wav.half().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            feats = extract_hubert_features(
                self.hubert, feats.to(self.device), self.version, padding_mask=padding_mask,
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            with self._settings_lock:
                index_rate = self.index_rate
            if self.index is not None and index_rate > 0:
                npy = feats[0][self.skip_head // 2 :].cpu().numpy().astype("float32")
                score, ix = self.index.index.search(npy, k=8)
                if (ix >= 0).all():
                    weight = np.square(1 / np.maximum(score, 1e-8))
                    weight /= weight.sum(axis=1, keepdims=True)
                    blended = np.sum(
                        self.index.vectors[ix] * np.expand_dims(weight, axis=2), axis=1
                    ) if self.index.vectors is not None else npy
                    if self.is_half:
                        blended = blended.astype("float16")
                    feats[0][self.skip_head // 2 :] = (
                        torch.from_numpy(blended).to(self.device) * index_rate
                        + (1 - index_rate) * feats[0][self.skip_head // 2 :]
                    )
            p_len = input_wav.shape[0] // _UNIT
            # hubert 20 ms frames -> model 10 ms frames (x2), clipped to p_len
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            feats = feats[:, :p_len, :]
            if self.if_f0 == 1:
                f0_extractor_frame = self.block_frame_16k + 800
                if self.f0_method == "rmvpe":
                    f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
                pitch, pitchf = self._get_f0(
                    input_wav[-f0_extractor_frame:].cpu().numpy()
                )
                shift = self.block_frame_16k // _UNIT
                self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
                self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
                self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
                self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
                cache_pitch = self.cache_pitch[None, -p_len:]
                cache_pitchf = self.cache_pitchf[None, -p_len:]
                p_len_tensor = torch.LongTensor([p_len]).to(self.device)
                sid = torch.LongTensor([0]).to(self.device)
                audio = self.net_g.infer(
                    feats, p_len_tensor, cache_pitch, cache_pitchf, sid,
                    self.skip_head, self.return_units, self.return_units,
                )[0]
            else:
                p_len_tensor = torch.LongTensor([p_len]).to(self.device)
                sid = torch.LongTensor([0]).to(self.device)
                audio = self.net_g.infer(
                    feats, p_len_tensor, sid,
                    self.skip_head, self.return_units, self.return_units,
                )[0]
        # normalise to 1-D (upstream returns (1, T); every consumer here is 1-D)
        return audio.reshape(-1).float()

    # ------------------------------------------------------------------ block
    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Feed one 16 kHz mono block; return tgt-sr audio for that block.

        ``block`` must have ``self.block_frame_16k`` samples (the caller —
        realtime service — guarantees this; shorter blocks are zero-padded
        which also handles microphone stalls).
        """
        if self.net_g is None:
            raise RealtimeError("realtime engine has no model loaded")
        t_start = time.perf_counter()
        if block.ndim != 1:
            block = block.mean(axis=1)
        expected = self.block_frame_16k
        if block.shape[0] > expected:            # caller misbehaviour -> keep tail
            block = block[-expected:]
        if block.shape[0] < expected:
            block = np.pad(block, (0, expected - block.shape[0]))

        # input noise gate with hysteresis + hangover + smooth gain ramp
        # (10 ms grid): avoids chopping syllables between words while blocking
        # distant quiet room sounds; slides the window strictly by `expected`
        # so input_wav never duplicates 20 ms chunks or drifts vs cache_pitch.
        with self._settings_lock:
            threshold_db = self.silence_threshold_db
            rms_mix_rate = self.rms_mix_rate
            strict_isolation = self.strict_isolation
        block = block.astype(np.float32, copy=True)
        accepted, _guard_reason = self.speaker_guard.evaluate(
            block, threshold_db, strict=strict_isolation
        )
        if not accepted:
            block.fill(0.0)
        n_write = expected                      # samples written into the window
        gate_has_audio = True
        if threshold_db > -60.0:
            x = np.append(self._rms_buffer, block)
            env = librosa.feature.rms(
                y=x, frame_length=_UNIT * 4, hop_length=_UNIT,
            )[:, 2:]
            self._rms_buffer[:] = x[-_UNIT * 4:]
            x = x[2 * _UNIT - _UNIT // 2:]
            env_db = librosa.amplitude_to_db(env, ref=1.0)[0]
            open_db = threshold_db
            close_db = max(-68.0, threshold_db - 8.0)
            hangover_frames = 22  # ~220 ms hold across consonant gaps & pauses
            gate_has_audio = False
            for i in range(env_db.shape[0]):
                db_val = float(env_db[i])
                if self._gate_hold_frames > 0:
                    if db_val >= close_db:
                        self._gate_hold_frames = hangover_frames
                    else:
                        self._gate_hold_frames -= 1
                    target = 1.0
                elif db_val >= open_db:
                    self._gate_hold_frames = hangover_frames
                    target = 1.0
                else:
                    target = 0.0
                prev_gain = self._gate_gain
                if target > self._gate_gain:
                    self._gate_gain = min(1.0, self._gate_gain + 0.5)
                elif target < self._gate_gain:
                    self._gate_gain = max(0.0, self._gate_gain - 0.25)
                if prev_gain > 0.0 or self._gate_gain > 0.0:
                    gate_has_audio = True
                start = i * _UNIT
                end = min(x.shape[0], (i + 1) * _UNIT)
                if end > start:
                    if prev_gain == 0.0 and self._gate_gain == 0.0:
                        x[start:end] = 0.0
                    elif prev_gain != self._gate_gain:
                        ramp = np.linspace(prev_gain, self._gate_gain, end - start, dtype=np.float32)
                        x[start:end] *= ramp
                    elif self._gate_gain < 1.0:
                        x[start:end] *= self._gate_gain
            block = x[_UNIT // 2:]
            n_write = block.shape[0]           # expected + 2 units (20 ms)
        else:
            self._gate_hold_frames = 0
            self._gate_gain = 1.0

        # slide window by `expected` (one block), then overwrite tail (block + 20 ms)
        self.input_wav[:-expected] = self.input_wav[expected:].clone()
        self.input_wav[-n_write:] = torch.from_numpy(block).to(self.device)

        tgt_unit = max(1, self.tgt_sr // 100)
        out_samples = self.block_units * tgt_unit
        if not gate_has_audio:
            self._silent_blocks += 1
        else:
            self._silent_blocks = 0

        if self._silent_blocks >= 3:
            if self.if_f0 == 1:
                shift = self.block_units
                self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
                self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
                self.cache_pitch[-shift:] = 0
                self.cache_pitchf[-shift:] = 0.0
            self.sola_buffer.zero_()
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            self._records.append(
                LatencyRecord(captured_at=time.time(), inference_ms=inference_ms,
                              total_ms=inference_ms)
            )
            self._blocks += 1
            return np.zeros(out_samples, dtype=np.float32)

        infer_wav = self._infer(self.input_wav)   # (1, samples)

        # RMS envelope mixing against the source block
        if rms_mix_rate < 1:
            src_len = min(self.return_units * _UNIT, self.input_wav.shape[-1])
            src = self.input_wav[-src_len:].cpu().numpy()
            rms1 = librosa.feature.rms(
                y=src, frame_length=min(_UNIT * 4, len(src)), hop_length=_UNIT, center=False
            )
            rms2 = librosa.feature.rms(
                y=infer_wav.cpu().numpy(),
                frame_length=min(tgt_unit * 4, infer_wav.shape[-1]),
                hop_length=tgt_unit,
                center=False,
            )
            n = infer_wav.shape[-1]
            r1 = F.interpolate(
                torch.from_numpy(rms1).float().to(infer_wav.device).unsqueeze(0),
                size=n, mode="linear", align_corners=True,
            )[0, 0]
            r2 = F.interpolate(
                torch.from_numpy(rms2).float().to(infer_wav.device).unsqueeze(0),
                size=n, mode="linear", align_corners=True,
            )[0, 0]
            r2 = torch.max(r2, torch.zeros_like(r2) + 1e-3)
            infer_wav = infer_wav * torch.pow(r1 / r2, 1.0 - rms_mix_rate)

        # SOLA crossfade (upstream/DDSP-SVC) in target-SR units
        if self._fade_ok:
            fade = self.sola_buffer_units * self._sola_unit
            conv_input = infer_wav[None, None, : fade + self.sila_search_frame]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input ** 2, self.sola_den_kernel) + 1e-8)
            sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
            infer_wav = infer_wav[sola_offset:]
            infer_wav[:fade] = infer_wav[:fade] * self.fade_in_window
            infer_wav[:fade] = infer_wav[:fade] + self.sola_buffer * self.fade_out_window
            self.sola_buffer[:] = infer_wav[out_samples : out_samples + fade]
        result = infer_wav[:out_samples]

        inference_ms = (time.perf_counter() - t_start) * 1000.0
        block_ms = self.block_units * 10.0
        self._records.append(
            LatencyRecord(captured_at=time.time(), inference_ms=inference_ms,
                          total_ms=inference_ms)  # transport added by service
        )
        self._blocks += 1
        out_np = result.cpu().numpy().astype(np.float32)
        if strict_isolation and accepted:
            self.speaker_guard.record_output(out_np, self.tgt_sr)
        return out_np

