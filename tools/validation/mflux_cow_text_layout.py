#!/usr/bin/env python3
"""APFS clone for the language-only reference view; keep ALL source bytes."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import time

from flux9_bf16_framework_screen import generation, check_generations, write_json
from mflux_original_layout import mapped_header


def compact_header(header):
    mapped, bindings = mapped_header(header, "qwen-text")
    for index, row in enumerate(bindings):
        if not row["mapped"].startswith("model.language_model."):
            original_name = row["mapped"]
            name = f"__tc_unused_{index}"
            if name in mapped:
                raise ValueError("unused tensor alias collision")
            mapped[name] = mapped.pop(original_name)
            row["mapped"] = name
            row["reference_consumed"] = False
        else:
            row["reference_consumed"] = True
    return mapped, bindings


def check_fd(fd, identity):
    value = os.fstat(fd)
    if (not stat.S_ISREG(value.st_mode) or value.st_dev != identity["device"] or value.st_ino != identity["inode"] or
            value.st_size != identity["bytes"] or value.st_mtime_ns != identity["mtime_ns"] or value.st_ctime_ns != identity["ctime_ns"]):
        raise ValueError("held original source generation changed")


def rewrite_cow(source, destination):
    source, destination = Path(source), Path(destination)
    before = generation(source)
    if destination.exists() or destination.is_symlink():
        raise ValueError("fresh private clone target required")
    if source.stat().st_flags:
        raise ValueError("uncompressed, non-immutable original required for bounded header-only CoW")
    start = time.perf_counter()
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    clone = library.fclonefileat
    clone.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    clone.restype = ctypes.c_int
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        fd = os.open(source, os.O_RDONLY)
        check_fd(fd, before)
        prefix = os.pread(fd, 8, 0)
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 16 << 20:
            raise ValueError("bounded original header required")
        raw_header = os.pread(fd, length, 8)
        mapped, bindings = compact_header(json.loads(raw_header))
        encoded = json.dumps(mapped, separators=(",", ":")).encode()
        if len(encoded) > length:
            raise ValueError("lossless aliases do not fit original header; never shift payload in-place")
        encoded += b" " * (length - len(encoded))
        max_extent = max(value["data_offsets"][1] for name, value in mapped.items() if name != "__metadata__")
        if max_extent != before["bytes"] - 8 - length:
            raise ValueError("full physical source payload coverage required")
        if clone(fd, directory, destination.name.encode(), 0):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
        os.chmod(destination, 0o600)
        converted = os.open(destination, os.O_RDWR | os.O_NOFOLLOW)
        try:
            clone_identity = os.fstat(converted)
            if clone_identity.st_ino == before["inode"] and clone_identity.st_dev == before["device"]:
                raise ValueError("clone must be a distinct private inode")
            if os.pwrite(converted, prefix + encoded, 0) != 8 + length:
                raise ValueError("short private header write")
            os.fsync(converted)
            source_sha = hashlib.sha256(prefix + raw_header)
            derived_sha = hashlib.sha256(prefix + encoded)
            payload_sha, derived_payload = hashlib.sha256(), hashlib.sha256()
            offset = 8 + length
            while offset < before["bytes"]:
                size = min(4 << 20, before["bytes"] - offset)
                a, b = os.pread(fd, size, offset), os.pread(converted, size, offset)
                if len(a) != size or a != b:
                    raise ValueError("entire cloned payload must match held original byte-for-byte")
                source_sha.update(a); derived_sha.update(b)
                payload_sha.update(a); derived_payload.update(b)
                offset += size
            if payload_sha.digest() != derived_payload.digest():
                raise ValueError("payload checksum mismatch")
            check_fd(fd, before)
            check_generations({str(source): before})
        finally:
            os.close(converted)
        destination.chmod(0o400)
        return dict(source=str(source), source_generation=before, source_sha256=source_sha.hexdigest(),
            derived=str(destination), derived_generation=generation(destination), derived_sha256=derived_sha.hexdigest(),
            payload_sha256=payload_sha.hexdigest(), full_payload_equal=True, kind="qwen-text",
            header_recipe="unused names aliased only; all language mapping targets and ALL tensor payloads retained",
            allocation_recipe="fclonefileat held read-only source; distinct private inode; header offset/length unchanged",
            allocated_stat_scope="st_blocks may include shared APFS extents; not exclusive physical disk usage",
            derived_allocated_bytes=destination.stat().st_blocks * 512, preparation_seconds=time.perf_counter() - start,
            tensor_bindings=bindings)
    finally:
        os.close(directory)
        if fd is not None:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("fresh conversion receipt required")
    result = rewrite_cow(args.source.absolute(), args.output.absolute())
    write_json(args.receipt, result)
    print(json.dumps(dict(full_payload_equal=result["full_payload_equal"], allocated_bytes=result["derived_allocated_bytes"])))


if __name__ == "__main__":
    main()
