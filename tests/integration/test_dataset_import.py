"""External audio-folder import tests."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from rvc_platform.training.dataset_import import AudioDatasetImporter, parse_sample_rate


def _tone(seconds: float, sr: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_import_decodes_trims_and_deduplicates(tmp_path):
    source = tmp_path / "incoming"
    source.mkdir()
    nested = source / "nested"
    nested.mkdir()
    padding = np.zeros(int(0.4 * 16_000), dtype=np.float32)
    signal = np.concatenate([padding, _tone(1.2), padding])
    sf.write(source / "a.wav", signal, 16_000, subtype="PCM_16")
    sf.write(nested / "same.wav", signal, 16_000, subtype="PCM_16")
    sf.write(source / "silent.wav", np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    (source / "notes.txt").write_text("ignore me", encoding="utf-8")

    out = tmp_path / "data" / "raw"
    report = AudioDatasetImporter(target_sr=40_000).import_folder(source, "speaker_a", output_root=out)

    assert report.included_files == 1
    assert len(report.duplicate_files) == 1
    assert any(item["reason"] == "silent_only" for item in report.skipped)
    written = list((out / "speaker_a").glob("voice_*.wav"))
    assert len(written) == 1
    audio, sr = sf.read(written[0])
    assert sr == 40_000
    assert 1.15 < len(audio) / sr < 1.55
    assert (out / "speaker_a" / "import_report.json").is_file()
    assert (out / "speaker_a" / "import_manifest.csv").is_file()


def test_import_refuses_to_replace_without_force(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sf.write(source / "x.wav", _tone(1.0), 16_000, subtype="PCM_16")
    out = tmp_path / "raw"
    importer = AudioDatasetImporter(target_sr=40_000)
    importer.import_folder(source, "voice", output_root=out)
    try:
        importer.import_folder(source, "voice", output_root=out)
    except Exception as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("existing dataset must not be replaced silently")


def test_sample_rate_parser():
    assert parse_sample_rate("40k") == 40_000
    assert parse_sample_rate("48000") == 48_000
