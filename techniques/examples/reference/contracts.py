"""Locally implemented design capsules for graph, bank, pack and residency contracts.

Inspired by the documented designs of Splash and vpipe, not copied runtimes.
CPU models expose ownership/errors; they do not claim device synchronization.
"""
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import struct
import statistics
from types import MappingProxyType

try:
    from .common import bf16, report
except ImportError:
    from common import bf16, report


class OwnedGraph:
    def __init__(self):
        self.commands = []
        self.submitted = False

    def __copy__(self):
        raise TypeError("graph copies need an explicit rebinding policy")

    def __deepcopy__(self, memo):
        raise TypeError("graph copies need an explicit rebinding policy")

    def add(self, operation, owners, parameters):
        if self.submitted or not callable(operation):
            raise ValueError("graph is sealed or operation is invalid")
        # Copy plain parameters, retain the actual backing owners, not their data.
        self.commands.append((operation, tuple(owners), deepcopy(parameters)))

    def submit(self):
        if self.submitted or not self.commands:
            raise ValueError("one nonempty submission required")
        self.submitted = True
        ticket = CpuTicket(tuple(self.commands))
        self.commands.clear()
        return ticket


class CpuTicket:
    def __init__(self, commands):
        self.commands, self.cancelled = commands, False
        self.physically_complete, self.succeeded = False, False

    def cancel_or_timeout(self):
        if self.physically_complete:
            raise ValueError("cannot cancel a completed ticket")
        self.cancelled = True
        # Keep commands/owners until finish(), including when logically cancelled.

    def finish(self):
        if self.physically_complete:
            raise ValueError("completion consumed twice")
        try:
            for operation, owners, params in self.commands:
                operation(owners, params)
            self.succeeded = not self.cancelled
        finally:
            self.physically_complete = True
            self.commands = ()
        return self.succeeded


@dataclass
class Bank:
    generation: int = 0
    source: str = ""
    state: str = "free"
    discard: bool = False


class DualBank:
    """Fixed-capacity state machine: stage, ready, execute, drain, reuse."""
    def __init__(self):
        self.banks = [Bank(), Bank()]

    def _bank(self, token, states):
        if not isinstance(token, tuple) or len(token) != 2 or any(type(value) is not int for value in token):
            raise ValueError("bank token must be an integer (index, generation) pair")
        index, generation = token
        if type(index) is not int or not 0 <= index < len(self.banks):
            raise ValueError("unknown bank")
        bank = self.banks[index]
        if bank.generation != generation or bank.state not in states:
            raise ValueError("stale generation or invalid transition")
        return bank

    def claim(self, source):
        if not isinstance(source, str) or not source:
            raise ValueError("immutable source/recipe identity required")
        for index, bank in enumerate(self.banks):
            if bank.state == "free":
                bank.generation += 1
                bank.source, bank.state, bank.discard = source, "staging", False
                return index, bank.generation
        raise ValueError("both banks still owned; backpressure required")

    def stage_complete(self, token, success=True):
        if type(success) is not bool:
            raise ValueError("completion success must be explicit")
        bank = self._bank(token, {"staging"})
        bank.state = "ready" if success else "free"
        bank.discard = not success

    def launch(self, token, source):
        bank = self._bank(token, {"ready"})
        if bank.source != source:
            raise ValueError("wrong checkpoint/layer/recipe bank")
        bank.state = "executing"

    def timeout(self, token):
        bank = self._bank(token, {"executing"})
        bank.state, bank.discard = "quarantined", True

    def physical_complete(self, token, success=True):
        if type(success) is not bool:
            raise ValueError("completion success must be explicit")
        bank = self._bank(token, {"executing", "quarantined"})
        publish = success and not bank.discard
        bank.state = "free"
        return publish


def owned_graph(seed=17):
    graph, output, params = OwnedGraph(), [], {"value": 3}
    graph.add(lambda owners, p: owners[0].append(p["value"]), [output], params)
    params["value"] = 99
    ticket = graph.submit()
    assert ticket.finish() and output == [3]
    cancelled = OwnedGraph()
    cancelled.add(lambda owners, p: owners[0].append(p["value"]), [output], {"value": 4})
    pending = cancelled.submit()
    pending.cancel_or_timeout()
    retained = bool(pending.commands) and not pending.physically_complete
    assert not pending.finish() and output == [3, 4]
    return report("owned-command-graph", {
        "parameters_copied": output[0] == 3, "owners_retained_after_timeout": retained,
        "cancelled_work_still_drained": pending.physically_complete,
        "cancelled_output_publishable": pending.succeeded,
        "limits": "Deferred CPU execution, not a Metal fence. Cancellation discards publication but is not rollback of already-written scratch.",
    }, scope="CPU ownership/parameter snapshot model")


def dual_bank_handoff(seed=17):
    pool = DualBank()
    current, ahead = pool.claim("checkpoint-A:layer-0:recipe-1"), pool.claim("checkpoint-A:layer-1:recipe-1")
    pool.stage_complete(current)
    pool.launch(current, "checkpoint-A:layer-0:recipe-1")
    pool.stage_complete(ahead)
    pool.timeout(current)
    try:
        pool.claim("checkpoint-B")
    except ValueError:
        blocked = True
    else:
        blocked = False
    discarded = not pool.physical_complete(current)
    recovered = pool.claim("checkpoint-B:layer-0:recipe-2")
    pool.stage_complete(recovered)
    pool.launch(recovered, "checkpoint-B:layer-0:recipe-2")
    assert blocked and discarded and pool.physical_complete(recovered)
    return report("dual-bank-handoff", {
        "capacity": len(pool.banks), "timeout_blocks_reuse": blocked,
        "late_result_discarded": discarded, "next_generation": recovered[1],
        "fallback_contract": "Discard the entire failed chunk; run the full GPU boundary only with fresh, nonborrowed scratch. Ready/done signaling alone does not validate output.",
    }, scope="two-bank source/generation/completion state machine")


MAGIC = b"TCLAB01\0"
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4}


def aligned(value, alignment=64):
    return (value + alignment - 1) // alignment * alignment


def make_pack(tensors):
    """Teaching format only; offsets are relative to the aligned payload start."""
    entries, payload = [], bytearray()
    for name, shape, dtype, raw in tensors:
        offset = aligned(len(payload))
        payload.extend(b"\0" * (offset - len(payload)))
        payload.extend(raw)
        entries.append({"name": name, "shape": list(shape), "dtype": dtype, "offset": offset, "bytes": len(raw)})
    header = json.dumps({"version": 1, "endian": "little", "tensors": entries}, separators=(",", ":")).encode()
    prefix = MAGIC + struct.pack("<I", len(header)) + header
    raw = prefix + b"\0" * (aligned(len(prefix)) - len(prefix)) + payload
    PackedStore(raw)  # Writer must satisfy the same reader contract.
    return raw


class PackedStore:
    def __init__(self, source):
        self.owner = bytes(source)  # Immutable whole backing; views retain it too.
        if len(self.owner) < 12 or self.owner[:8] != MAGIC:
            raise ValueError("bad pack magic/truncated header")
        header_size = struct.unpack_from("<I", self.owner, 8)[0]
        if not 1 <= header_size <= 1 << 20 or 12 + header_size > len(self.owner):
            raise ValueError("invalid header size")
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate pack metadata key")
                result[key] = value
            return result
        header = json.loads(self.owner[12:12 + header_size], object_pairs_hook=unique_object)
        if not isinstance(header, dict) or type(header.get("version")) is not int or header.get("version") != 1 or header.get("endian") != "little":
            raise ValueError("unsupported pack version/endian")
        if not isinstance(header.get("tensors"), list) or not 1 <= len(header["tensors"]) <= 16384:
            raise ValueError("bounded nonempty tensor table required")
        self.payload_start, self.entries = aligned(12 + header_size), {}
        occupied = []
        for entry in header["tensors"]:
            if not isinstance(entry, dict) or not {"name", "shape", "dtype", "offset", "bytes"} <= entry.keys():
                raise ValueError("incomplete tensor table entry")
            name, shape, dtype = entry["name"], entry["shape"], entry["dtype"]
            offset, size = entry["offset"], entry["bytes"]
            if not isinstance(name, str) or not name or name in self.entries:
                raise ValueError("unique tensor names required")
            if not isinstance(shape, list) or not shape or any(type(dim) is not int or not 1 <= dim <= 1 << 20 for dim in shape):
                raise ValueError("invalid tensor dimensions")
            if not isinstance(dtype, str) or dtype not in DTYPE_BYTES or type(size) is not int or math.prod(shape) * DTYPE_BYTES[dtype] != size:
                raise ValueError("dtype/shape/payload mismatch")
            if type(offset) is not int or offset < 0 or offset % 64 or self.payload_start + offset + size > len(self.owner):
                raise ValueError("misaligned/out-of-bounds tensor")
            if any(offset < end and offset + size > begin for begin, end in occupied):
                raise ValueError("overlapping tensors")
            occupied.append((offset, offset + size))
            self.entries[name] = MappingProxyType({**entry, "shape": tuple(shape)})
        if not self.entries:
            raise ValueError("empty pack")
        self.sha256 = hashlib.sha256(self.owner).hexdigest()
        self.entries = MappingProxyType(self.entries)

    def view(self, name):
        entry = self.entries[name]
        start = self.payload_start + entry["offset"]
        return memoryview(self.owner)[start:start + entry["bytes"]]


def typed_refill(raw, elements, source_dtype, destination_dtype):
    """UNSERVABLE permits a separate slow path; FAILED never publishes partial data."""
    if source_dtype not in DTYPE_BYTES or destination_dtype not in DTYPE_BYTES:
        return {"status": "unservable", "payload": None}
    if type(elements) is not int or elements <= 0 or len(raw) != elements * DTYPE_BYTES[source_dtype]:
        return {"status": "failed", "payload": None}
    if source_dtype == destination_dtype:
        return {"status": "ready", "payload": bytes(raw)}
    if (source_dtype, destination_dtype) != ("F16", "BF16"):
        return {"status": "unservable", "payload": None}
    try:
        values = struct.unpack("<" + "e" * elements, raw)
        converted = b"".join(struct.pack("<I", struct.unpack("<I", struct.pack("<f", bf16(value)))[0])[2:] for value in values)
    except (ValueError, OverflowError, struct.error):
        return {"status": "failed", "payload": None}
    return {"status": "ready", "payload": converted}


def packed_weights(seed=17):
    source = struct.pack("<4e", 1., -.5, 2., .25)
    store = PackedStore(make_pack([("block.0.weight", (2, 2), "F16", source)]))
    view = store.view("block.0.weight")
    converted = typed_refill(view, 4, "F16", "BF16")
    values = [struct.unpack("<f", b"\0\0" + converted["payload"][i:i + 2])[0] for i in range(0, 8, 2)]
    wrong = [struct.unpack("<f", b"\0\0" + bytes(view[i:i + 2]))[0] for i in range(0, 8, 2)]
    assert values == [1., -.5, 2., .25] and values != wrong and view.readonly
    return report("packed-weight-store", {
        "source_sha256": store.sha256, "readonly_owner_retained": view.readonly,
        "payload_alignment": 64, "converted_values": values,
        "same_width_reinterpret_is_not_conversion": wrong != values,
        "truncated_refill_status": typed_refill(source[:-1], 4, "F16", "BF16")["status"],
        "limits": "Immutable Python memoryview, not OS mmap/Metal no-copy. Format is TCLAB01, not GGUF/Splash/vpipe format.",
    })


class FixedResidency:
    """Policy only: admissions never evict; caller confirms actual release."""
    def __init__(self, reserve=None):
        if reserve is not None and (type(reserve) is not int or reserve < 0):
            raise ValueError("reserve must be known nonnegative bytes or None")
        self.reserve, self.allocated_reserve = reserve, 0
        self.blocks, self.ceiling = {}, math.inf

    def admit(self, name, size, free_unallocated):
        if type(size) is not int or size <= 0 or type(free_unallocated) is not int or free_unallocated < 0:
            raise ValueError("valid bytes/headroom required")
        if type(self.allocated_reserve) is not int or self.allocated_reserve < 0:
            raise ValueError("allocated reserve must be nonnegative bytes")
        if name in self.blocks:
            raise ValueError("already resident")
        if self.reserve is None:
            return False
        remaining = max(0, self.reserve - self.allocated_reserve)
        if sum(self.blocks.values()) + size > self.ceiling or free_unallocated < remaining + 2 * size:
            return False
        self.blocks[name] = size
        return True

    def release_completed(self, name):
        # Call only AFTER last-reader completion and actual memory release.
        del self.blocks[name]
        self.ceiling = sum(self.blocks.values())

    def healthy_probe(self, bytes_to_probe):
        if type(bytes_to_probe) is not int or bytes_to_probe <= 0:
            raise ValueError("positive bounded probe required")
        self.ceiling += bytes_to_probe


def residency_policy(seed=17):
    trace, capacity, lru, lru_hits = list(range(8)) * 3, 3, OrderedDict(), []
    fixed, fixed_hits = set(range(capacity)), []
    for block in trace:
        lru_hits.append(block in lru)
        fixed_hits.append(block in fixed)
        lru[block] = True
        lru.move_to_end(block)
        if len(lru) > capacity:
            lru.popitem(last=False)
    policy = FixedResidency(reserve=100)
    policy.allocated_reserve = 80
    admitted = policy.admit("block-0", 10, 40)  # remaining20 + candidate10 + slack10
    policy.release_completed("block-0")
    ratcheted = not policy.admit("block-1", 10, 1000)
    policy.healthy_probe(10)
    recovered = policy.admit("block-1", 10, 40)
    assert admitted and ratcheted and recovered and not any(lru_hits[8:])
    return report("residency-policy", {
        "warm_scan_accesses": len(trace[8:]), "lru_warm_hits": sum(lru_hits[8:]),
        "fixed_warm_hits": sum(fixed_hits[8:]), "capacity_blocks": capacity,
        "allocated_reserve_not_counted_twice": admitted, "post_eviction_ratchet": ratcheted,
        "explicit_healthy_probe_can_reopen": recovered,
        "limits": "Synthetic sequential trace and policy bytes. Real resident/compressed pages, last readers and peer release must be measured by an adapter.",
    }, scope="synthetic residency policy and deterministic access trace")


class ProfitBreaker:
    """Bounded complete-boundary policy, not hardware monitoring or a fitter."""
    def __init__(self, identity, window=4, warmup=2, maximum_ratio=1.02):
        if not identity or type(window) is not int or not 2 <= window <= 64:
            raise ValueError("identity and bounded sample window required")
        if type(warmup) is not int or not 0 <= warmup <= 64 or not math.isfinite(maximum_ratio) or maximum_ratio < 1:
            raise ValueError("invalid warmup/noise policy")
        self.identity, self.window = identity, window
        self.warmup_left, self.maximum_ratio = warmup, maximum_ratio
        self.ratios, self.enabled = deque(maxlen=window), True

    def observe(self, identity, gpu_seconds, split_seconds):
        if identity != self.identity:
            raise ValueError("shape/source/recipe changed; requalify a fresh policy")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (gpu_seconds, split_seconds)):
            raise ValueError("complete finite positive paired boundaries required")
        if not self.enabled:
            return False
        if self.warmup_left:
            self.warmup_left -= 1
            return True
        self.ratios.append(split_seconds / gpu_seconds)
        if len(self.ratios) == self.window and statistics.median(self.ratios) > self.maximum_ratio:
            self.enabled = False
        return self.enabled


def hybrid_policy(seed=17):
    key = "M33:H128:F512:fp16:source-A:adapter-A"
    breaker = ProfitBreaker(key)
    for _ in range(2):
        assert breaker.observe(key, .01, .1)  # Cold/warmup is not a hot decision.
    for _ in range(3):
        assert breaker.observe(key, .01, .012)
    disabled = not breaker.observe(key, .01, .012)
    stays_off = not breaker.observe(key, .01, .008)
    fresh = ProfitBreaker(key + ":new-graph", warmup=0)
    reopened = all(fresh.observe(fresh.identity, .01, .008) for _ in range(4))
    assert disabled and stays_off and reopened
    return report("hybrid-policy", {
        "warmup_samples_excluded": 2, "hot_window": breaker.window,
        "sustained_regression_disables": disabled, "no_implicit_reenable": stays_off,
        "explicit_requalification_can_reopen": reopened,
        "limits": "Assumed paired complete-layer times; no physical contention attribution or production adaptation. Use matched controls without running full GPU twice on every real request.",
    }, scope="synthetic runtime-profit breaker with explicit requalification")
