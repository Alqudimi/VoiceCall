"""Asset downloader: byte-range resume + atomic writes, with a fake HTTP server."""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rvc_platform.application import downloads
from rvc_platform.infrastructure.paths import get_paths


class FakeAssetHandler(BaseHTTPRequestHandler):
    # 200 KB payload served with Range support (exercises resume path)
    PAYLOAD = b"A" * (200 * 1024)

    def do_HEAD(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.PAYLOAD)))
        self.end_headers()

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(self.PAYLOAD):
                self.send_response(416)
                self.end_headers()
                return
            body = self.PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body)-1+start}/{len(self.PAYLOAD)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.PAYLOAD)))
            self.end_headers()
            self.wfile.write(self.PAYLOAD)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_asset_server():
    server = HTTPServer(("127.0.0.1", 0), FakeAssetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def test_download_writes_file_atomically(project_env, fake_asset_server):
    dest = get_paths().assets / "test" / "blob.bin"
    downloads._download_one(fake_asset_server + "/blob.bin", dest, chunk_mb=1, timeout=10)
    assert dest.is_file() and dest.stat().st_size == len(FakeAssetHandler.PAYLOAD)
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_resumes_from_partial(project_env, fake_asset_server):
    dest = get_paths().assets / "test" / "resume.bin"
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    # pretend a previous run wrote half the file
    part.write_bytes(FakeAssetHandler.PAYLOAD[: len(FakeAssetHandler.PAYLOAD) // 2])
    downloads._download_one(fake_asset_server + "/resume.bin", dest, chunk_mb=1, timeout=10)
    assert dest.stat().st_size == len(FakeAssetHandler.PAYLOAD)
    assert dest.read_bytes() == FakeAssetHandler.PAYLOAD


def test_download_fails_on_http_error(project_env, tmp_path):
    dest = tmp_path / "nope.bin"
    with pytest.raises(Exception):
        downloads._download_one(
            "http://127.0.0.1:1/nope.bin", dest, chunk_mb=1, timeout=1
        )


def test_download_assets_existing_file_skips_network(project_env, monkeypatch):
    """A present asset is reported 'exists' without any network access."""
    from rvc_platform.infrastructure.config.loader import get_config
    from rvc_platform.infrastructure.paths import get_paths

    cfg = get_config()
    monkeypatch.setattr(cfg.downloads, "huggingface_base", "http://127.0.0.1:9")
    monkeypatch.setattr(cfg.downloads, "connect_timeout_s", 0.5)

    present = get_paths().assets / cfg.downloads.files["rmvpe.pt"]
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_bytes(b"x")

    report = downloads.download_assets(only=["rmvpe.pt"])
    assert report["rmvpe.pt"].startswith("exists")


def test_download_assets_missing_file_raises_after_failure(project_env, monkeypatch):
    from rvc_platform.infrastructure.config.loader import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.downloads, "huggingface_base", "http://127.0.0.1:9")
    monkeypatch.setattr(cfg.downloads, "connect_timeout_s", 0.5)

    with pytest.raises(Exception):
        downloads.download_assets(only=["rmvpe.pt"])
