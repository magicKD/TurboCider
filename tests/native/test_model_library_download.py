"""Exercise native hub downloads with tiny local fixtures; no model downloads."""
import hashlib
import json
from pathlib import Path
import subprocess
import struct
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


def safetensors_fixture():
    header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
    header += b' ' * (-len(header) % 8)
    return struct.pack('<Q', len(header)) + header + b'\0\0'


FLUX_FILES = {
    'tokenizer/tokenizer.json': json.dumps({'model': {'type': 'BPE'}}).encode(),
    'transformer/config.json': json.dumps({'attention_head_dim': 128, 'in_channels': 128, 'num_attention_heads': 24,
                                         'num_layers': 5, 'num_single_layers': 20, 'joint_attention_dim': 7680,
                                         'guidance_embeds': False}).encode(),
    'text_encoder/config.json': json.dumps({'hidden_size': 2560, 'num_hidden_layers': 36,
                                          'num_attention_heads': 32, 'num_key_value_heads': 8}).encode(),
    'vae/config.json': json.dumps({'latent_channels': 32}).encode(),
    'transformer/model.safetensors.index.json': json.dumps({'weight_map': {'weight': 'model.safetensors'}}).encode(),
    **{f'{component}/model.safetensors': safetensors_fixture() for component in ('transformer', 'text_encoder', 'vae')},
}


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
            if repo == 'flux':
                return self.reply([{'type': 'file', 'path': name, 'size': len(data), 'lfs': {'oid': sha(data)}}
                                   for name, data in FLUX_FILES.items()])
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
        if path.startswith('/test/flux/resolve/'):
            return self.reply(FLUX_FILES[name])
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


def main(binary=ROOT / 'build/native/turbocider-hub-tests'):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='tc-hub-fixture-') as temporary:
            subprocess.run([str(binary),
                            f'http://127.0.0.1:{server.server_port}', temporary], check=True)
        assert COUNTS.get(f'/test/tiny/resolve/{REVISION}/weights.bin', 0) == 0, COUNTS
        assert sum(COUNTS.get(f'/test/flux/resolve/{REVISION}/{component}/model.safetensors', 0)
                   for component in ('transformer', 'text_encoder', 'vae')) == 1, COUNTS
        print('PASS: all network traffic stayed on the tiny fixture server; no full model downloaded')
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    main()
