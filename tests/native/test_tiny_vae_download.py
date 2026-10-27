"""Exercise the native Tiny VAE downloader entirely on a local fixture server."""
from pathlib import Path
import subprocess
import tempfile
import threading
from http.server import ThreadingHTTPServer
from test_model_library_download import Handler, COUNTS, REVISION

ROOT = Path(__file__).resolve().parents[2]


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="tc-tiny-download-") as root:
            subprocess.run([str(ROOT / "build/native/turbocider-tiny-vae-download-tests"),
                            f"http://127.0.0.1:{server.server_port}", root], check=True)
        assert COUNTS.get(f"/test/tiny/resolve/{REVISION}/weights.bin") == 3, COUNTS
        assert not any("/api/" in path for path in COUNTS), COUNTS
        print("PASS: no metadata can replace pinned tiny assets; cache caused no extra network traffic")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
