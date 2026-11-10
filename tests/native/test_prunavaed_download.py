"""Exercise the App's pinned ModelScope VAE downloader without remote traffic."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]


def main():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            requests.append((parsed.path, parse_qs(parsed.query)))
            payload = b"pruna-fixture-weights" * 32
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([str(native / "turbocider-prunavaed-download-tests"),
                            f"http://127.0.0.1:{server.server_port}", directory], check=True)
        assert requests == [("/api/v1/models/test/pruna/repo", {
            "Revision": ["a" * 40], "FilePath": ["vae/diffusion_pytorch_model.safetensors"]})] * 3, requests
        print("PASS fixed ModelScope revision/path, no metadata lookups, no extra download on reuse")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
