"""Regression tests for model-repository index isolation + registry safety.

Two real defects found in the v1.6.0 deep audit of persistence:
1. ``_find_index`` used a bare prefix glob (``<stem>*.index``): model
   ``ahmed`` silently grabbed ``ahmed2.index`` — ANOTHER speaker's retrieval
   index — and ``delete("ahmed")`` unlinked that foreign index too.
2. ``_read_registry`` returned valid-JSON-but-wrong-shape payloads as-is:
   ``{"models": []}`` or ``{}`` crashed register/delete/note with
   KeyError/TypeError instead of self-healing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rvc_platform.infrastructure.persistence.model_repository import ModelRepository


@pytest.fixture()
def repo(tmp_path: Path) -> ModelRepository:
    return ModelRepository(models_dir=tmp_path)


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class TestIndexIsolation:
    def test_exact_index_wins(self, repo: ModelRepository, tmp_path: Path):
        _touch(tmp_path / "ahmed.pth")
        _touch(tmp_path / "ahmed.index")
        _touch(tmp_path / "ahmed2.index")           # a DIFFERENT model's index
        assert repo._find_index(tmp_path / "ahmed.pth") == tmp_path / "ahmed.index"

    def test_no_cross_grab_of_other_models_index(self, repo, tmp_path):
        _touch(tmp_path / "ahmed.pth")
        _touch(tmp_path / "ahmed2.index")           # prefix collision, foreign
        assert repo._find_index(tmp_path / "ahmed.pth") is None

    def test_underscore_variant_still_found(self, repo, tmp_path):
        _touch(tmp_path / "ahmed.pth")
        _touch(tmp_path / "ahmed_added_IVF1024.index")
        found = repo._find_index(tmp_path / "ahmed.pth")
        assert found == tmp_path / "ahmed_added_IVF1024.index"

    def test_delete_does_not_remove_foreign_index(self, repo, tmp_path):
        _touch(tmp_path / "ahmed.pth")
        _touch(tmp_path / "ahmed.index")
        _touch(tmp_path / "ahmed2.index")           # must survive the delete
        _touch(tmp_path / "ahmed2.pth")
        repo.delete("ahmed", with_index=True)
        assert not (tmp_path / "ahmed.pth").exists()
        assert not (tmp_path / "ahmed.index").exists()
        assert (tmp_path / "ahmed2.index").exists(), "foreign index was deleted!"

    def test_delete_removes_own_underscore_variants(self, repo, tmp_path):
        _touch(tmp_path / "ahmed.pth")
        _touch(tmp_path / "ahmed_added_IVF1024.index")
        repo.delete("ahmed", with_index=True)
        assert not (tmp_path / "ahmed_added_IVF1024.index").exists()


class TestRegistrySelfHealing:
    def test_empty_object_registry(self, repo: ModelRepository, tmp_path):
        (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
        repo.note("anything", "a note")             # must not raise
        data = json.loads((tmp_path / "registry.json").read_text())
        assert data["models"]["anything"]["notes"] == "a note"

    def test_models_list_registry(self, repo, tmp_path):
        (tmp_path / "registry.json").write_text('{"models": []}', encoding="utf-8")
        repo.note("m", "n")                          # must not raise
        data = json.loads((tmp_path / "registry.json").read_text())
        assert data["models"]["m"]["notes"] == "n"

    def test_non_dict_registry(self, repo, tmp_path):
        (tmp_path / "registry.json").write_text('[1, 2, 3]', encoding="utf-8")
        repo.note("m", "n")                          # must not raise
        data = json.loads((tmp_path / "registry.json").read_text())
        assert data["models"]["m"]["notes"] == "n"

    def test_corrupt_json_still_recovers(self, repo, tmp_path):
        (tmp_path / "registry.json").write_text("{broken", encoding="utf-8")
        repo.note("m", "n")
        data = json.loads((tmp_path / "registry.json").read_text())
        assert data["models"]["m"]["notes"] == "n"

    def test_delete_with_broken_registry(self, repo, tmp_path):
        _touch(tmp_path / "m.pth")
        (tmp_path / "registry.json").write_text('{"models": 7}', encoding="utf-8")
        repo.delete("m")                             # must not raise
        assert not (tmp_path / "m.pth").exists()


class TestRegisterIndexIdempotent:
    """Regression from the REAL-data phase-02 benchmark session.

    ``ModelRepository.register`` was already guarded against copying a file
    onto itself, but ``register_index`` was not: re-registering the SAME
    index file (source == target) crashed with ``shutil.SameFileError``.
    This surfaced when the HuggingFace download service re-registered a
    freshly downloaded index that already sat at its final location.
    """

    def test_register_index_is_idempotent_same_file(self, repo, tmp_path):
        _touch(tmp_path / "ahmed.index", b"faiss-index-bytes")
        first = repo.register_index(tmp_path / "ahmed.index", "ahmed")
        assert first == (tmp_path / "ahmed.index").resolve()
        # re-registering the same file must be a no-op, not a crash
        second = repo.register_index(tmp_path / "ahmed.index", "ahmed")
        assert second == first
        assert (tmp_path / "ahmed.index").read_bytes() == b"faiss-index-bytes"

    def test_register_index_copies_when_source_differs(self, repo, tmp_path):
        _touch(tmp_path / "outside" / "src.index", b"other-bytes")
        target = repo.register_index(tmp_path / "outside" / "src.index", "ahmed")
        assert target == (tmp_path / "ahmed.index").resolve()
        assert target.read_bytes() == b"other-bytes"
        assert (tmp_path / "outside" / "src.index").read_bytes() == b"other-bytes"
