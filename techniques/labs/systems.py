"""Dependency, slot-lifetime, actual small-file IO and startup-cost experiments."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import os
import random
import tempfile
import time
from pathlib import Path

try:
    from .common import report
except ImportError:
    from common import report


def schedule(tasks):
    """Non-preemptive synthetic DAG list scheduler, NOT a device-overlap trace."""
    if not tasks or len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("unique nonempty tasks required")
    ids = {task["id"] for task in tasks}
    for task in tasks:
        if not math.isfinite(task["duration"]) or task["duration"] <= 0 or not set(task["deps"]) <= ids:
            raise ValueError("invalid duration/dependency")
    pending, finished, available, trace = list(tasks), {}, {}, []
    while pending:
        ready = [task for task in pending if set(task["deps"]) <= finished.keys()]
        if not ready:
            raise ValueError("dependency cycle")
        def earliest(task):
            return max([available.get(task["resource"], 0)] + [finished[name] for name in task["deps"]])
        task = min(ready, key=lambda item: (earliest(item), item["id"]))
        start = earliest(task)
        end = start + task["duration"]
        finished[task["id"]] = end
        available[task["resource"]] = end
        trace.append({**task, "start": start, "end": end})
        pending.remove(task)
    return {"makespan": max(finished.values()), "timeline": trace}


def require_edges(tasks, required):
    by_id = {task["id"]: set(task["deps"]) for task in tasks}
    if any(before not in by_id.get(after, set()) for before, after in required):
        raise ValueError("required mathematical/ownership edge missing")


def overlap(seed=17):
    tasks = [
        {"id": "attention", "resource": "gpu", "duration": 6., "deps": []},
        {"id": "input_pack", "resource": "gpu", "duration": 1., "deps": ["attention"]},
        {"id": "gpu_ffn", "resource": "gpu", "duration": 5., "deps": ["input_pack"]},
        {"id": "ane_ffn", "resource": "ane", "duration": 4., "deps": ["input_pack"]},
        {"id": "join", "resource": "gpu", "duration": .5, "deps": ["gpu_ffn", "ane_ffn"]},
    ]
    required = [("attention", "input_pack"), ("input_pack", "gpu_ffn"),
                ("input_pack", "ane_ffn"), ("gpu_ffn", "join"), ("ane_ffn", "join")]
    require_edges(tasks, required)
    parallel = schedule(tasks)
    serial = [{**task, "resource": "one_serial_resource"} for task in tasks]
    contended = [{**task, "duration": task["duration"] * (1.65 if task["id"] in ("gpu_ffn", "ane_ffn") else 1)} for task in tasks]
    illegal = [{**task, "deps": [] if task["id"] == "ane_ffn" else task["deps"]} for task in tasks]
    try:
        require_edges(illegal, required)
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected
    return report("overlap-dag", {"units": "synthetic milliseconds", "serial": schedule(serial),
                  "legal_fork_join": parallel, "coupled_bandwidth_scenario": schedule(contended),
                  "illegal_attention_overlap_rejected": rejected,
                  "note": "Resource labels and slowdown factors are assumptions, not physical Apple engine evidence."},
                  scope="discrete-event dependency simulator")


@dataclass
class Slot:
    generation: int = 0
    state: str = "empty"
    source: str = ""
    readers: set = None

    def __post_init__(self):
        self.readers = set()

    def fill(self, source):
        if self.state not in ("empty", "retired") or not source:
            raise ValueError("cannot overwrite an owned/in-flight/quarantined slot")
        self.generation += 1
        self.source = source
        self.state = "filling"
        return self.generation

    def ready(self, generation, *, success=True):
        if generation != self.generation or self.state != "filling":
            raise ValueError("stale producer completion")
        self.state = "ready" if success else "retired"

    def borrow(self, source, reader):
        if self.state not in ("ready", "in_flight") or self.source != source or reader in self.readers:
            raise ValueError("source mismatch or invalid consumer")
        self.readers.add(reader)
        self.state = "in_flight"
        return self.generation

    def complete(self, generation, reader):
        if generation != self.generation or reader not in self.readers:
            raise ValueError("stale/unknown reader completion")
        self.readers.remove(reader)
        if not self.readers:
            self.state = "retired"

    def timeout(self):
        if self.state != "in_flight":
            raise ValueError("timeout without submitted readers")
        self.state = "quarantined"


def lifetimes(seed=17):
    slot, controls = Slot(), []
    first = slot.fill("source-A:recipe-1")
    slot.ready(first)
    slot.borrow("source-A:recipe-1", "ane")
    slot.borrow("source-A:recipe-1", "gpu-restore")
    def rejects(name, function):
        try:
            function()
        except ValueError:
            controls.append(name)
        else:
            raise AssertionError("unsafe transition accepted: " + name)
    rejects("reuse-before-completion", lambda: slot.fill("source-B"))
    rejects("wrong-source-generation", lambda: slot.borrow("source-B", "next"))
    slot.timeout()
    rejects("timeout-is-not-reuse-permission", lambda: slot.fill("source-B"))
    slot.complete(first, "ane")
    rejects("one-of-two-readers-still-live", lambda: slot.fill("source-B"))
    slot.complete(first, "gpu-restore")
    second = slot.fill("source-B")
    rejects("late-old-producer", lambda: slot.ready(first))
    slot.ready(second, success=False)
    rejects("failed-producer-output-not-published", lambda: slot.borrow("source-B", "gpu"))
    third = slot.fill("source-C")
    slot.ready(third)
    slot.borrow("source-C", "gpu")
    slot.complete(third, "gpu")
    return report("slot-ownership", {"rejected_controls": controls, "final_state": slot.state,
                  "generations": slot.generation,
                  "scope_limit": "Logical state-machine witness. Real device tickets/events must prove physical completion."},
                  scope="ownership state-machine reference")


def read_exact(fd, offset, count):
    parts, consumed = [], 0
    while consumed < count:
        try:
            block = os.pread(fd, count - consumed, offset + consumed)
        except InterruptedError:
            continue
        if not block:
            raise IOError("short positional read; partial result must not be published")
        parts.append(block)
        consumed += len(block)
    return b"".join(parts)


def streaming_io(seed=17):
    rng, count, size = random.Random(seed), 12, 16384
    data = bytes(rng.randrange(256) for _ in range(count * size))
    expected = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix="tc-technique-io-") as directory:
        path = Path(directory) / "private-fixture.bin"
        path.write_bytes(data)
        fd = os.open(path, os.O_RDONLY)
        try:
            before = os.fstat(fd)
            def consume(reader):
                digest, start = hashlib.sha256(), time.perf_counter()
                for block in reader:
                    digest.update(block)
                return {"seconds": time.perf_counter() - start, "sha256": digest.hexdigest()}
            serial = consume(read_exact(fd, index * size, size) for index in range(count))
            def bounded():
                # Exactly one future; it owns bytes until the consumer finishes.
                with ThreadPoolExecutor(max_workers=1) as worker:
                    future = worker.submit(read_exact, fd, 0, size)
                    for index in range(count):
                        block = future.result()  # get/rethrow, not wait-and-ignore
                        future = worker.submit(read_exact, fd, (index + 1) * size, size) if index + 1 < count else None
                        yield block
            ahead = consume(bounded())
            assert serial["sha256"] == ahead["sha256"] == expected
            try:
                read_exact(fd, len(data) - 2, 4)
            except IOError:
                short_rejected = True
            else:
                short_rejected = False
            duplicate = os.dup(fd)
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                os.read(fd, 3)
                shared_cursor = os.lseek(duplicate, 0, os.SEEK_CUR)
                witness = os.pread(duplicate, 3, 0)
                unchanged_cursor = os.lseek(fd, 0, os.SEEK_CUR)
            finally:
                os.close(duplicate)
            after = os.fstat(fd)
            assert short_rejected and shared_cursor == unchanged_cursor == 3 and witness == data[:3]
            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        finally:
            os.close(fd)
    return report("bounded-io", {"bytes": len(data), "blocks": count, "serial": serial, "one_future": ahead,
                  "short_read_rejected": short_rejected, "dup_fd_shares_cursor": True,
                  "positional_read_preserves_cursor": True,
                  "limits": "Python os.pread allocates bytes; not native direct-fill/zero-copy. OS cache not cleared; no SSD throughput claim."},
                  scope="actual tiny private-file CPU IO, exploratory elapsed observations")


def startup(seed=17):
    gpu_setup, hybrid_setup, gpu_hot, hybrid_hot = .1, 1.2, .02, .012
    delta, gain = hybrid_setup - gpu_setup, gpu_hot - hybrid_hot
    crossover = max(1, math.floor(delta / gain) + 1) if gain > 0 else None
    rows = [{"requests": n, "gpu_total_seconds": gpu_setup + n * gpu_hot,
             "hybrid_total_seconds": hybrid_setup + n * hybrid_hot} for n in (1, 8, 64, 256)]
    return report("startup-amortization", {"assumed_seconds": {"gpu_setup": gpu_setup, "hybrid_setup": hybrid_setup,
                  "gpu_hot": gpu_hot, "hybrid_hot": hybrid_hot}, "first_strictly_profitable_request_count": crossover,
                  "scenarios": rows, "one_shot_faster": "gpu",
                  "revisit": "Replace assumptions with compile/verify/load/first-predict/subsequent spans from your machine."},
                  scope="synthetic startup break-even model")


def bucket_selection(seed=17):
    buckets = [512, 1024, 1056, 2112, 4224]
    requests = [1024 + 31, 4096 + 31, 1024 + 31, 1024 + 97]
    rows = []
    for actual in requests:
        valid = [bucket for bucket in buckets if bucket >= actual]
        chosen = min(valid) if valid else None
        rows.append({"actual_rows": actual, "bucket": chosen,
                     "padded_rows": chosen - actual if chosen else None,
                     "padding_fraction": (chosen - actual) / chosen if chosen else None})
    assert [row["bucket"] for row in rows[:3]] == [1056, 4224, 1056]
    return report("bucket-selection", {"request_sequence": rows, "available_buckets": buckets,
                  "note": "Caption/reference rows are part of shape; cache the lookup by actual geometry, not canvas alone."})
