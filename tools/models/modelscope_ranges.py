"""Pinned ModelScope range transfers with a resumable contiguous prefix.

Each asset has an OS file lock. Bounded range files are assembled durably into
the existing .part prefix, then published only after the full SHA256 matches.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
from download_qwen21_gguf import digest, download_asset, local_size


def download_ranges(asset, root, connections=4, chunk_size=64 << 20,
                    endpoint='https://modelscope.cn/models', retries=8):
    target = root / asset['relative_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + '.lock')
    local_size(lock_path)
    with lock_path.open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _download_locked(asset, root, target, connections, chunk_size, endpoint, retries)


def _download_locked(asset, root, target, connections, chunk_size, endpoint, retries):
    if not 1 <= connections <= 8 or chunk_size <= 0:
        raise ValueError('invalid connection count or range chunk size')
    partial = target.with_name(target.name + '.part')
    size = asset['bytes']
    offset = local_size(partial)
    if offset > size:
        raise ValueError('partial exceeds the pinned asset size')
    if target.exists() or target.is_symlink() or offset == size or connections == 1 or size <= chunk_size:
        return download_asset(asset, root, endpoints=(endpoint,), retries=retries)
    chunks = target.with_name(target.name + '.ranges')
    if chunks.is_symlink():
        raise ValueError('range directory must not be a symlink')
    chunks.mkdir(exist_ok=True)
    identity = chunks / 'source.json'
    local_size(identity)
    if identity.exists():
        if json.loads(identity.read_text()) != asset:
            raise ValueError('range source identity differs from the pinned asset')
    else:
        identity.write_text(json.dumps(asset, sort_keys=True) + '\n')
    segments = []
    start = offset
    while start < size:
        end = min(size, (start // chunk_size + 1) * chunk_size) - 1
        path = chunks / f'{start}-{end}.range'
        existing = local_size(path)
        if existing > end - start + 1:
            raise ValueError('range file exceeds its pinned extent')
        segments.append((start, end, path))
        start = end + 1
    progress = {start: local_size(path) for start, _, path in segments}
    mutex = threading.Lock()
    last_progress = time.monotonic()
    url = f"{endpoint.rstrip('/')}/{asset['repository']}/resolve/{asset['revision']}/{asset['filename']}"

    def transfer(segment):
        nonlocal last_progress
        start, end, path = segment
        expected = end - start + 1
        for attempt in range(retries):
            have = local_size(path)
            if have == expected:
                return
            request = urllib.request.Request(url, headers={
                'User-Agent': 'TurboCider-LTX-pinned-range-download',
                'Accept-Encoding': 'identity', 'Range': f'bytes={start + have}-{end}'})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start + have}-{end}/{size}':
                        raise ValueError('server did not honor the exact pinned range')
                    length = response.headers.get('Content-Length')
                    if length is not None and int(length) != expected - have:
                        raise ValueError('range Content-Length differs from the pinned extent')
                    if response.headers.get('Content-Encoding', 'identity') not in ('', 'identity'):
                        raise ValueError('compressed range cannot safely resume model bytes')
                    local_size(path)
                    with path.open('ab') as out:
                        while block := response.read(1 << 20):
                            if len(block) > expected - have:
                                raise ValueError('range response exceeds its pinned extent')
                            out.write(block)
                            have += len(block)
                            with mutex:
                                progress[start] = have
                                if time.monotonic() - last_progress >= 5:
                                    out.flush()
                                    print(json.dumps(dict(asset=asset['filename'], bytes=offset + sum(progress.values()),
                                                          total_bytes=size, connections=connections)), flush=True)
                                    last_progress = time.monotonic()
                        if have != expected:
                            raise OSError('range response ended early')
                return
            except OSError as error:
                print(json.dumps(dict(asset=asset['filename'], range_start=start, retry=attempt + 1,
                                      error_type=type(error).__name__)), flush=True)
                if attempt + 1 == retries:
                    raise
                time.sleep(min(attempt + 1, 3))

    print(json.dumps(dict(downloading=asset['filename'], resume_bytes=offset,
                          connections=connections, ranges=len(segments))), flush=True)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(transfer, segment) for segment in segments]
        try:
            for future in futures:
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    # A crash during assembly leaves a valid contiguous .part prefix. A range
    # is removed only after that prefix is durable; retries reuse the rest.
    local_size(partial)
    with partial.open('ab') as out:
        for start, end, path in segments:
            if out.tell() != start or local_size(path) != end - start + 1:
                raise ValueError('range assembly source or destination changed')
            with path.open('rb') as source:
                while block := source.read(4 << 20):
                    out.write(block)
            out.flush()
            os.fsync(out.fileno())
            path.unlink()
    if local_size(partial) != size or digest(partial) != asset['sha256']:
        raise ValueError('complete range download failed the pinned SHA256; not published')
    if target.exists() or target.is_symlink():
        raise ValueError('publication target appeared during range download')
    partial.rename(target)
    # Remove only files in this asset's private transfer directory.
    for path in chunks.iterdir():
        if re.fullmatch(r'\d+-\d+\.range', path.name):
            local_size(path)
            path.unlink()
    identity.unlink()
    if not any(chunks.iterdir()):
        chunks.rmdir()
    print(json.dumps(dict(verified=asset['filename'], bytes=size, sha256=asset['sha256'])), flush=True)
    return {**asset, 'status': 'downloaded_verified'}
