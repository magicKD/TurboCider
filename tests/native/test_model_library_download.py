"""Exercise native hub downloads with tiny local fixtures; no model downloads."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
CONFIG = b'{"fixture":true}'
WEIGHTS = b'tiny-fixture-weights' * 32
SLOW = b'x' * 131072
SUPPLEMENT = b'{"tokenizer_fixture":true}'
REVISION = 'a' * 40
COUNTS = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_oid(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value, headers=None):
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        COUNTS[path] = COUNTS.get(path, 0) + 1
        if path.endswith('/revision/main'):
            return self.reply({'sha': REVISION})
        if '/tree/' in path:
            repo = path.split('/')[4]
            if repo == 'supplement':
                return self.reply([{'type': 'file', 'path': 'extra/tokenizer.json', 'size': len(SUPPLEMENT), 'lfs': {'oid': sha(SUPPLEMENT)}}])
            if repo == 'unsafe':
                return self.reply([{'type': 'file', 'path': '../escape', 'size': 1}])
            if repo == 'slow':
                return self.reply([{'type': 'file', 'path': 'slow.bin', 'size': len(SLOW), 'lfs': {'oid': sha(SLOW)}}])
            if repo in ('badhash', 'badsize'):
                return self.reply([{'type': 'file', 'path': 'weights.bin', 'size': len(WEIGHTS) + (repo == 'badsize'),
                                    'lfs': {'oid': '0' * 64 if repo == 'badhash' else sha(WEIGHTS + b'!')}}])
            if 'page' not in query or repo == 'cycle':
                next_url = f'http://127.0.0.1:{self.server.server_port}{path}?page=2'
                return self.reply([{'type': 'file', 'path': 'config.json', 'size': len(CONFIG), 'oid': git_oid(CONFIG)}],
                                  {'Link': f'<{next_url}>; rel="next"'})
            return self.reply([{'type': 'file', 'path': 'weights.bin', 'size': len(WEIGHTS), 'lfs': {'oid': sha(WEIGHTS)}},
                               {'type': 'file', 'path': 'text_encoder/config.json', 'size': len(CONFIG), 'oid': git_oid(CONFIG)}])
        if path.endswith('/repo/files'):
            if 'Root' in query:
                entries = [{'Type': 'blob', 'Path': 'text_encoder/config.json', 'Size': len(CONFIG), 'Sha256': sha(CONFIG), 'Revision': REVISION}]
            else:
                entries = [{'Type': 'blob', 'Path': 'config.json', 'Size': len(CONFIG), 'Sha256': sha(CONFIG), 'Revision': REVISION},
                           {'Type': 'blob', 'Path': 'weights.bin', 'Size': len(WEIGHTS), 'Sha256': sha(WEIGHTS), 'Revision': REVISION},
                           {'Type': 'tree', 'Path': 'text_encoder'}]
            return self.reply({'Success': True, 'Data': {'Files': entries}})
        if path.endswith('/repo'):
            name = query['FilePath'][0]
        elif '/resolve/' in path:
            name = path.split('/resolve/')[1].split('/', 1)[1]
        else:
            self.send_error(404)
            return
        if name.startswith('text_encoder/'):
            raise AssertionError('Shared encoder must not be downloaded')
        if name == 'extra/tokenizer.json':
            return self.reply(SUPPLEMENT)
        if name == 'slow.bin':
            self.send_response(200); self.send_header('Content-Length', str(len(SLOW))); self.end_headers()
            try:
                for offset in range(0, len(SLOW), 1024):
                    self.wfile.write(SLOW[offset:offset + 1024]); self.wfile.flush(); time.sleep(.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        return self.reply(CONFIG if name == 'config.json' else WEIGHTS)


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='tc-hub-fixture-') as temporary:
            subprocess.run([str(ROOT / 'build/native/turbocider-hub-tests'),
                            f'http://127.0.0.1:{server.server_port}', temporary], check=True)
        assert COUNTS.get(f'/test/tiny/resolve/{REVISION}/weights.bin', 0) == 0, COUNTS
        print('PASS: all network traffic stayed on the tiny fixture server; no full model downloaded')
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    main()
