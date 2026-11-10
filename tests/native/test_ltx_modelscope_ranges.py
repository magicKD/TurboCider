import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/models'))
from modelscope_ranges import download_ranges

PAYLOAD = bytes(range(256)) * 257


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        start, end = map(int, self.headers['Range'][6:].split('-'))
        self.server.ranges.append((start, end))
        self.send_response(206)
        claimed_start = start + 1 if self.server.bad_range else start
        self.send_header('Content-Range', f'bytes {claimed_start}-{end}/{len(PAYLOAD)}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        self.wfile.write(PAYLOAD[start:end + 1])


class LTXRangeDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.ranges, self.server.bad_range = [], False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}'
        self.asset = dict(repository='owner/repo', revision='a' * 40, filename='model.safetensors',
                          relative_path='diffusion_models/model.safetensors', bytes=len(PAYLOAD),
                          sha256=hashlib.sha256(PAYLOAD).hexdigest())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def target(self):
        return self.root / self.asset['relative_path']

    def run_download(self):
        return download_ranges(self.asset, self.root, connections=4, chunk_size=4096,
                               endpoint=self.endpoint, retries=1)

    def test_resume_prefix_and_partial_range_then_verified_publication(self):
        target = self.target()
        target.parent.mkdir()
        prefix = 5000
        target.with_name(target.name + '.part').write_bytes(PAYLOAD[:prefix])
        chunks = target.with_name(target.name + '.ranges')
        chunks.mkdir()
        (chunks / 'source.json').write_text(json.dumps(self.asset))
        (chunks / '5000-8191.range').write_bytes(PAYLOAD[5000:6000])
        result = self.run_download()
        self.assertEqual(result['status'], 'downloaded_verified')
        self.assertEqual(target.read_bytes(), PAYLOAD)
        self.assertIn((6000, 8191), self.server.ranges)
        self.assertFalse(chunks.exists())
        self.assertFalse(target.with_name(target.name + '.part').exists())
        count = len(self.server.ranges)
        self.assertEqual(self.run_download()['status'], 'verified_existing')
        self.assertEqual(len(self.server.ranges), count)

    def test_wrong_range_never_published(self):
        self.server.bad_range = True
        with self.assertRaisesRegex(ValueError, 'exact pinned range'):
            self.run_download()
        self.assertFalse(self.target().exists())

    def test_wrong_full_digest_never_published(self):
        self.asset['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
            self.run_download()
        self.assertFalse(self.target().exists())
        self.assertEqual(self.target().with_name(self.target().name + '.part').read_bytes(), PAYLOAD)

    def test_source_identity_mismatch_rejected(self):
        target = self.target()
        target.parent.mkdir()
        chunks = target.with_name(target.name + '.ranges')
        chunks.mkdir()
        (chunks / 'source.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source identity'):
            self.run_download()
        self.assertEqual(self.server.ranges, [])


if __name__ == '__main__':
    unittest.main()
