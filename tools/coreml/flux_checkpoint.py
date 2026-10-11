"""Read-only indexed FLUX source geometry and bounded safetensors access."""
from contextlib import ExitStack
import hashlib
import json
import mmap
from pathlib import Path
import struct


def geometry(config):
    single, heads, head_dim = (config.get(key) for key in ("num_single_layers", "num_attention_heads", "attention_head_dim"))
    if (single, heads, head_dim) not in ((20, 24, 128), (24, 32, 128)) or config.get("mlp_ratio", 3.0) != 3.0:
        raise ValueError("only original Klein4B/9B single-block geometry supported")
    return dict(hidden=heads * head_dim, heads=heads, single_layers=single, mlp_width=heads * head_dim * 3)


def generation(path):
    status = path.stat()
    return dict(device=status.st_dev, inode=status.st_ino, bytes=status.st_size,
        mtime_ns=status.st_mtime_ns, ctime_ns=status.st_ctime_ns, target=str(path.resolve(strict=True)))


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            result.update(chunk)
    return result.hexdigest()


class Checkpoint:
    def __init__(self, model):
        root = (Path(model) / "transformer").resolve(strict=True)
        single = root / "diffusion_pytorch_model.safetensors"
        index = root / "diffusion_pytorch_model.safetensors.index.json"
        if single.is_file() and index.is_file():
            raise ValueError("ambiguous flat/indexed checkpoint source; select one canonical layout")
        self.path = single if single.is_file() else index
        self.weight_map = None
        if self.path == index:
            data = json.loads(index.read_text())
            self.weight_map = data.get("weight_map")
            if not isinstance(self.weight_map, dict) or not self.weight_map or any(
                    not isinstance(name, str) or not name or not isinstance(shard, str) or
                    not shard or Path(shard).name != shard or "\\" in shard or shard in (".", "..")
                    for name, shard in self.weight_map.items()):
                raise ValueError("complete basename-only checkpoint index required")
            self.files = [root / name for name in sorted(set(self.weight_map.values()))]
        else:
            self.files = [single]
        if not all(path.is_file() for path in [self.path, *self.files]):
            raise ValueError("complete original checkpoint/index/shards required")
        self.generations = {path: generation(path) for path in set([self.path, *self.files])}
        self.stack = None

    def identity(self):
        self.check()
        result = dict(checkpoint=str(self.path), checkpoint_bytes=self.path.stat().st_size,
            checkpoint_sha256=sha(self.path))
        if self.weight_map is not None:
            result["checkpoint_shards"] = {path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in self.files}
        self.check()
        return result

    def check(self):
        if any(generation(path) != value for path, value in self.generations.items()):
            raise ValueError("original checkpoint source generation changed")

    def __enter__(self):
        self.stack, self.headers, self.maps = ExitStack(), {}, {}
        try:
            for path in self.files:
                stream = self.stack.enter_context(path.open("rb"))
                memory = self.stack.enter_context(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ))
                if len(memory) < 8:
                    raise ValueError("truncated safetensors source")
                size = struct.unpack("<Q", memory[:8])[0]
                if not 0 < size <= 100_000_000 or 8 + size > len(memory):
                    raise ValueError("invalid safetensors source header")
                header = json.loads(memory[8:8 + size])
                self.maps[path.name] = (memory, 8 + size)
                for name, item in header.items():
                    if name == "__metadata__":
                        continue
                    if name in self.headers:
                        raise ValueError("duplicate tensor across source shards")
                    self.headers[name] = (path.name, item)
            if self.weight_map is not None and any(name not in self.headers or self.headers[name][0] != shard
                    for name, shard in self.weight_map.items()):
                raise ValueError("checkpoint index tensor/shard binding missing or changed")
            self.check()
            return self
        except BaseException:
            self.stack.close()
            raise

    def tensor(self, name, shape, np, rows=None, columns=None):
        shard, item = self.headers[name]
        memory, offset = self.maps[shard]
        if item["shape"] != list(shape) or item["dtype"] not in ("BF16", "F16"):
            raise ValueError("unexpected original tensor shape/dtype: " + name)
        start, end = item["data_offsets"]
        count = int(np.prod(shape))
        if start < 0 or end - start != count * 2 or offset + end > len(memory):
            raise ValueError("invalid original tensor extent")
        if rows is not None:
            if len(rows) != 2 or any(type(value) is not int for value in rows) or not 0 <= rows[0] < rows[1] <= shape[0]:
                raise ValueError("bounded nonempty original row range required")
        if columns is not None:
            if len(shape) != 2 or len(columns) != 2 or any(type(value) is not int for value in columns) or not 0 <= columns[0] < columns[1] <= shape[1]:
                raise ValueError("bounded nonempty original column range required")
        view = np.frombuffer(memory, dtype="<u2" if item["dtype"] == "BF16" else "<f2", count=count, offset=offset + start).reshape(shape)
        try:
            if rows is not None:
                view = view[rows[0]:rows[1]]
            if columns is not None:
                view = view[:, columns[0]:columns[1]]
            if item["dtype"] == "BF16":
                words = view.astype(np.uint32) << 16
                return words.view(np.float32).astype(np.float16)
            return view.copy()
        finally:
            del view  # No exported mmap view survives success or an exception.

    def __exit__(self, *args):
        try:
            self.check()
        finally:
            self.stack.close()
