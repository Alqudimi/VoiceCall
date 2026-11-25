"""Strict real-time speaker isolation: speech-only VAD, proximity gate, and voiceprint lock.

Designed to run on 16 kHz mono 160 ms blocks in < 2 ms with zero external
model downloads, ensuring that only the primary close speaker is forwarded to
RVC while rejecting background noise, distant voices, other speakers, and
speaker-to-mic self-echo.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np

_SR = 16_000
_N_FFT = 512
_HOP = 160
_N_MFCC = 24
_CENTROID_K = 8


@dataclass
class SpeechAnalysis:
    speech_db: float
    speech_ratio: float
    flatness: float
    harmonicity: float
    zcr: float
    f0_hz: float
    mfcc_vec: np.ndarray


@dataclass
class SpeakerProfile:
    centroids: np.ndarray  # (K, 23) L2-normalized MFCC c1..c23 centroids
    f0_hz: float
    ref_db: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "centroids": self.centroids.tolist(),
            "f0_hz": float(self.f0_hz),
            "ref_db": float(self.ref_db),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerProfile:
        centroids = np.asarray(data["centroids"], dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[0] == 0 or centroids.shape[1] != _N_MFCC - 1:
            raise ValueError("invalid speaker profile centroids shape")
        norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
        return cls(
            centroids=centroids / norms,
            f0_hz=float(data.get("f0_hz", 0.0)),
            ref_db=float(data.get("ref_db", -24.0)),
        )


def _cosine_max(vec: np.ndarray, bank: np.ndarray) -> float:
    if bank.size == 0:
        return 0.0
    v = vec / (np.linalg.norm(vec) + 1e-9)
    sims = bank @ v
    return float(np.max(sims))


def _frame_mfcc(wav_16k: np.ndarray) -> np.ndarray:
    """Return (T, 23) MFCC frames (c1..c23, energy c0 excluded)."""
    if wav_16k.shape[0] < _N_FFT:
        wav_16k = np.pad(wav_16k, (0, _N_FFT - wav_16k.shape[0]))
    mel = librosa.feature.melspectrogram(
        y=wav_16k, sr=_SR, n_fft=_N_FFT, hop_length=_HOP,
        n_mels=40, fmin=80.0, fmax=7200.0, center=False,
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel + 1e-10), n_mfcc=_N_MFCC)[1:]
    return mfcc.T.astype(np.float32)


def analyze_block(wav_16k: np.ndarray) -> SpeechAnalysis:
    """Extract fast speech-VAD and speaker-timbre features from a 16 kHz block."""
    x = np.asarray(wav_16k, dtype=np.float32)
    if x.shape[0] < _N_FFT:
        x = np.pad(x, (0, _N_FFT - x.shape[0]))

    win = np.hanning(_N_FFT).astype(np.float32)
    frames = librosa.util.frame(x, frame_length=_N_FFT, hop_length=_HOP)
    spec = np.abs(np.fft.rfft(frames * win[:, None], axis=0)) ** 2
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / _SR)
    speech_mask = (freqs >= 120.0) & (freqs <= 3800.0)

    total_e = float(np.sum(spec)) + 1e-12
    speech_e = float(np.sum(spec[speech_mask]))
    speech_ratio = speech_e / total_e

    # Speech-band RMS in dBFS (approx. scaled consistently with full-band RMS)
    mean_frame_power = float(np.mean(np.sum(spec[speech_mask], axis=0))) / ((_N_FFT * _N_FFT) / 4.0)
    speech_rms = float(np.sqrt(max(mean_frame_power, 1e-18)))
    # Fallback / blend with time-domain RMS for consistent dB calibration
    td_rms = float(np.sqrt(np.mean(x ** 2) + 1e-18))
    eff_rms = min(td_rms, max(td_rms * np.sqrt(max(speech_ratio, 0.0)), speech_rms))
    speech_db = float(20.0 * np.log10(eff_rms + 1e-9))

    p_sp = spec[speech_mask] + 1e-12
    flatness = float(np.mean(np.exp(np.mean(np.log(p_sp), axis=0)) / np.mean(p_sp, axis=0)))

    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))

    # Normalized autocorrelation harmonicity & coarse F0 in 60..450 Hz
    seg = x[:min(1280, x.shape[0])]
    seg = seg - float(np.mean(seg))
    ac = np.correlate(seg, seg, mode="full")[seg.shape[0] - 1 :]
    ac0 = float(ac[0]) + 1e-12
    lo_lag, hi_lag = 35, min(266, ac.shape[0] - 1)
    if hi_lag > lo_lag:
        sub = ac[lo_lag:hi_lag] / ac0
        best_idx = int(np.argmax(sub))
        harmonicity = float(sub[best_idx])
        if harmonicity >= 0.30:
            lag = lo_lag + best_idx
            # Avoid locking onto 2x/3x harmonic (formant) peaks when true period has >= 88% strength
            for mult in (3, 2):
                cand = lag * mult
                if cand + 3 < hi_lag:
                    win_lo = max(lo_lag, cand - 3)
                    win_hi = min(hi_lag, cand + 4)
                    sub_peak = int(np.argmax(ac[win_lo:win_hi]))
                    cand_lag = win_lo + sub_peak
                    if (ac[cand_lag] / ac0) >= 0.88 * harmonicity:
                        lag = cand_lag
                        break
            f0_hz = float(_SR / lag)
        else:
            f0_hz = 0.0
    else:
        harmonicity = 0.0
        f0_hz = 0.0

    mfccs = _frame_mfcc(x)
    mfcc_vec = np.mean(mfccs, axis=0)
    return SpeechAnalysis(
        speech_db=speech_db,
        speech_ratio=speech_ratio,
        flatness=flatness,
        harmonicity=harmonicity,
        zcr=zcr,
        f0_hz=f0_hz,
        mfcc_vec=mfcc_vec,
    )


def _cluster_centroids(frames: np.ndarray, k: int = _CENTROID_K) -> np.ndarray:
    norms = frames / (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-9)
    if norms.shape[0] <= k:
        return norms.astype(np.float32)
    centroids = [norms[0]]
    for _ in range(k - 1):
        sims = norms @ np.stack(centroids).T
        best = np.max(sims, axis=1)
        centroids.append(norms[int(np.argmin(best))])
    bank = np.stack(centroids)
    for _ in range(3):
        assign = np.argmax(norms @ bank.T, axis=1)
        updated = []
        for j in range(bank.shape[0]):
            members = norms[assign == j]
            c = np.mean(members, axis=0) if members.shape[0] else bank[j]
            updated.append(c / (np.linalg.norm(c) + 1e-9))
        bank = np.stack(updated)
    return bank.astype(np.float32)


def build_speaker_profile(wav_16k: np.ndarray, min_voiced_blocks: int = 3) -> SpeakerProfile:
    """Build a multi-centroid speaker voiceprint from a 16 kHz enrollment recording."""
    x = np.asarray(wav_16k, dtype=np.float32)
    block_len = 2560  # 160 ms
    voiced_mfccs: list[np.ndarray] = []
    f0s: list[float] = []
    dbs: list[float] = []
    for start in range(0, max(1, x.shape[0] - block_len + 1), block_len):
        blk = x[start : start + block_len]
        if blk.shape[0] < _N_FFT:
            continue
        a = analyze_block(blk)
        if (
            a.speech_db > -48.0
            and a.speech_ratio >= 0.50
            and a.flatness <= 0.30
            and a.harmonicity >= 0.32
            and a.zcr <= 0.55
        ):
            voiced_mfccs.append(_frame_mfcc(blk))
            dbs.append(a.speech_db)
            if a.f0_hz > 0.0:
                f0s.append(a.f0_hz)
    if len(voiced_mfccs) < min_voiced_blocks:
        raise ValueError(
            "لم يُلتقط كلام واضح كافٍ لبناء بصمة الصوت — تكلّم بوضوح قرب الميكروفون لمدة 3 ثوانٍ."
        )
    all_frames = np.concatenate(voiced_mfccs, axis=0)
    centroids = _cluster_centroids(all_frames, k=_CENTROID_K)
    f0_med = float(np.median(f0s)) if f0s else 0.0
    ref_db = float(np.clip(np.percentile(dbs, 75), -36.0, -12.0))
    return SpeakerProfile(centroids=centroids, f0_hz=f0_med, ref_db=ref_db)


class SpeakerGuard:
    """Per-session speech-only + proximity + voiceprint verifier."""

    def __init__(self) -> None:
        self.profile: SpeakerProfile | None = None
        self.noise_floor_db: float = -58.0
        self.speaker_ref_db: float | None = None
        self._hold_blocks: int = 0
        self._auto_frames: list[np.ndarray] = []
        self._auto_f0s: list[float] = []
        self._auto_dbs: list[float] = []
        self._cloned_centroids: deque[np.ndarray] = deque(maxlen=16)
        self._recent_mfccs: deque[np.ndarray] = deque(maxlen=3)
        self.last_reason: str = "silence"

    def reset(self) -> None:
        self._hold_blocks = 0
        self._recent_mfccs.clear()
        self._cloned_centroids.clear()
        self.last_reason = "silence"

    def set_profile(self, profile: SpeakerProfile | dict[str, Any] | None) -> None:
        if profile is None:
            self.clear_profile()
            return
        self.profile = profile if isinstance(profile, SpeakerProfile) else SpeakerProfile.from_dict(profile)
        self.speaker_ref_db = self.profile.ref_db
        self._auto_frames.clear()
        self._auto_f0s.clear()
        self._auto_dbs.clear()

    def clear_profile(self) -> None:
        self.profile = None
        self.speaker_ref_db = None
        self._auto_frames.clear()
        self._auto_f0s.clear()
        self._auto_dbs.clear()

    def record_output(self, audio_tgt_sr: np.ndarray, tgt_sr: int) -> None:
        """Store MFCC signature of recent converted output to detect speaker feedback."""
        if audio_tgt_sr.size == 0 or tgt_sr <= 0:
            return
        if tgt_sr != _SR:
            n_out = max(1, round(audio_tgt_sr.shape[0] * _SR / tgt_sr))
            idx = np.linspace(0.0, audio_tgt_sr.shape[0] - 1, n_out)
            wav_16k = np.interp(idx, np.arange(audio_tgt_sr.shape[0]), audio_tgt_sr).astype(np.float32)
        else:
            wav_16k = audio_tgt_sr.astype(np.float32, copy=False)
        if np.sqrt(np.mean(wav_16k ** 2)) < 0.01:
            return
        vec = np.mean(_frame_mfcc(wav_16k), axis=0)
        self._cloned_centroids.append((vec / (np.linalg.norm(vec) + 1e-9)).astype(np.float32))

    def _reject(self, reason: str) -> tuple[bool, str]:
        self._hold_blocks = 0
        self._recent_mfccs.clear()
        self.last_reason = reason
        return False, reason

    def evaluate(
        self,
        block_16k: np.ndarray,
        threshold_db: float,
        strict: bool = True,
    ) -> tuple[bool, str]:
        if not strict:
            self.last_reason = "ok"
            return True, "ok"

        a = analyze_block(block_16k)

        # 1) Non-speech noise rejection (fans, claps, knocks, hiss, static, rumble)
        is_speech_shape = (
            a.speech_ratio >= 0.50
            and a.flatness <= 0.30
            and a.zcr <= 0.55
        )
        if not is_speech_shape:
            if a.speech_db < -28.0:
                self.noise_floor_db = float(min(-40.0, 0.9 * self.noise_floor_db + 0.1 * a.speech_db))
            if self._hold_blocks > 0 and a.speech_db >= threshold_db - 8.0 and a.speech_ratio >= 0.35:
                # brief fricative/consonant mid-word: allow hold to decay gradually
                self._hold_blocks -= 1
                self.last_reason = "ok"
                return True, "ok"
            return self._reject("noise" if a.speech_db >= threshold_db else "silence")

        # 2) Proximity + noise-floor gate
        prox_floor = (self.speaker_ref_db - 13.0) if self.speaker_ref_db is not None else -60.0
        open_db = max(float(threshold_db), self.noise_floor_db + 10.0, prox_floor)
        close_db = open_db - 7.0

        is_opening = self._hold_blocks == 0
        if is_opening:
            if a.speech_db < open_db:
                if a.speech_db < open_db - 4.0:
                    self.noise_floor_db = float(min(-40.0, 0.9 * self.noise_floor_db + 0.1 * a.speech_db))
                return self._reject("far_voice" if a.speech_db >= float(threshold_db) - 5.0 else "silence")
            if a.harmonicity < 0.32:
                return self._reject("noise")
        else:
            if a.speech_db < close_db:
                self._hold_blocks -= 1
                if self._hold_blocks == 0:
                    return self._reject("silence")
                self.last_reason = "ok"
                return True, "ok"

        # 3) Speaker voiceprint verification (or auto-enrollment from first clean speech)
        self._recent_mfccs.append(a.mfcc_vec)
        rolling_vec = np.mean(np.stack(list(self._recent_mfccs)), axis=0)

        if self.profile is not None:
            sim_cur = _cosine_max(a.mfcc_vec, self.profile.centroids)
            sim_roll = _cosine_max(rolling_vec, self.profile.centroids)
            sim = max(sim_cur, sim_roll)

            # Pitch band check on strongly voiced frames (unless timbre centroid is a near-exact match)
            if self.profile.f0_hz > 0.0 and a.f0_hz > 0.0 and a.harmonicity >= 0.42 and sim < 0.96:
                ratio = a.f0_hz / self.profile.f0_hz
                if ratio < 0.55 or ratio > 1.82:
                    return self._reject("other_speaker")

            # Self-echo check against recent cloned output
            if self._cloned_centroids and a.harmonicity >= 0.35:
                echo_bank = np.stack(list(self._cloned_centroids))
                echo_sim = _cosine_max(a.mfcc_vec, echo_bank)
                if echo_sim > sim + 0.035 and sim < 0.92:
                    return self._reject("self_echo")

            min_sim = 0.85 if is_opening else 0.78
            if sim < min_sim:
                return self._reject("other_speaker")
        else:
            # Auto-lock onto the first close voiced speaker after ~0.8s (5 voiced blocks)
            if a.harmonicity >= 0.34:
                self._auto_frames.append(_frame_mfcc(block_16k))
                self._auto_dbs.append(a.speech_db)
                if a.f0_hz > 0.0:
                    self._auto_f0s.append(a.f0_hz)
                if len(self._auto_frames) >= 5:
                    all_f = np.concatenate(self._auto_frames, axis=0)
                    self.profile = SpeakerProfile(
                        centroids=_cluster_centroids(all_f, k=_CENTROID_K),
                        f0_hz=float(np.median(self._auto_f0s)) if self._auto_f0s else 0.0,
                        ref_db=float(np.clip(np.percentile(self._auto_dbs, 75), -36.0, -12.0)),
                    )
                    self.speaker_ref_db = self.profile.ref_db
                    self._auto_frames.clear()
                    self._auto_f0s.clear()
                    self._auto_dbs.clear()

        # Accepted close primary speaker
        self._hold_blocks = 2  # ~320 ms hold across brief consonants/syllable dips
        if a.speech_db >= open_db and a.harmonicity >= 0.35:
            if self.speaker_ref_db is None:
                self.speaker_ref_db = float(np.clip(a.speech_db, -36.0, -12.0))
            else:
                self.speaker_ref_db = float(np.clip(0.92 * self.speaker_ref_db + 0.08 * a.speech_db, -36.0, -12.0))
        self.last_reason = "ok"
        return True, "ok"
