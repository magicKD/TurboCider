"""CPU-only PMU provenance contracts and independent synthetic tensor witnesses.

Counter labels are claims, not calibrated units. Missing objects, empty buffers,
zeroed payloads and successful inference are deliberately different states.
"""
import hashlib
import math
import struct

from common import report

_WITNESSES = {}


def classify_stats(value):
    if type(value.get("object_present")) is not bool:
        raise ValueError("typed stats-object presence required")
    count, words = value.get("payload_bytes"), value.get("raw_words")
    if not value["object_present"]:
        if count is not None or words is not None:
            raise ValueError("missing object cannot carry invented raw counters")
        return {"state": "missing-object", "raw_by_index": None, "physical_duration_seconds": None,
                "counter_semantics_qualified": False}
    if type(count) is not int or not 0 <= count <= 4096 or not isinstance(words, list):
        raise ValueError("bounded payload size and explicit raw words required")
    if count % 8 or len(words) != count // 8:
        raise ValueError("truncated or inconsistent uint64 payload")
    if any(type(word) is not int or not 0 <= word < 2**64 for word in words):
        raise ValueError("typed unsigned raw words required")
    labels = value.get("labels_by_index", [None] * len(words))
    if not isinstance(labels, list) or len(labels) != len(words) or any(label is not None and not isinstance(label, str) for label in labels):
        raise ValueError("one claimed label per raw index required")
    state = ("empty-payload" if not count else "unknown-payload-schema" if count != 232 else
             "zero-filled-unqualified" if not any(words) else "raw-available-unqualified")
    return {"state": state, "raw_by_index": [{"index": i, "claimed_label": label, "value": word, "unit": None}
            for i, (label, word) in enumerate(zip(labels, words))],
            "physical_duration_seconds": None, "counter_semantics_qualified": False}


def counter_delta(before, after, *, bits, wrap_verified=False, reset_observed=False):
    """Arithmetic only after an explicit lifecycle/width decision; no auto wrap.

    The caller still has to prove cumulative, request-local and physical units
    using matched workloads. This helper does not grant that qualification.
    """
    if type(bits) is not int or not 1 <= bits <= 64 or type(wrap_verified) is not bool or type(reset_observed) is not bool:
        raise ValueError("explicit physical width and typed lifecycle evidence required")
    if any(type(value) is not int or not 0 <= value < 2**bits for value in (before, after)):
        raise ValueError("counter outside the explicit width")
    if reset_observed:
        return None
    if after < before and not wrap_verified:
        raise ValueError("decrease may be reset, per-request scope or wrap; never clamp to zero")
    return (after - before) % 2**bits


def witness_hashes(kind, channels, width, generation):
    """Full logical input/output hashes, independent of the native client.

    The declared synthetic input has period 17 in X. Reusing those exact CPU
    sums accelerates verification only; the digest still covers every element.
    It is not a model activation or a substitute for a real workload replay.
    """
    if (kind not in ("affine", "conv") or type(channels) is not int or not 1 <= channels <= 128
            or type(width) is not int or not 1 <= width <= 512 or type(generation) is not int or not 0 <= generation <= 8):
        raise ValueError("bounded synthetic witness required")
    key = kind, channels, width, generation
    if key in _WITNESSES:
        return dict(_WITNESSES[key])
    source = [[((c * 7 + x * 3 + generation * 5) % 17 - 8) / 8. for x in range(17)] for c in range(channels)]
    if kind == "affine":
        expected = [[2 * value + .25 for value in row] for row in source]
    else:
        expected = [[math.fsum(source[i][x] * (((c * 7 + i * 3) % 11 - 5) / 64.) for i in range(channels))
                     for x in range(17)] for c in range(channels)]
    inputs = [source[c][x % 17] for c in range(channels) for x in range(width)]
    outputs = [expected[c][x % 17] for c in range(channels) for x in range(width)]
    result = {name: hashlib.sha256(struct.pack("<" + "e" * len(values), *values)).hexdigest()
              for name, values in (("input_sha256", inputs), ("output_sha256", outputs))}
    if len(_WITNESSES) >= 256:
        _WITNESSES.clear()
    _WITNESSES[key] = result
    return dict(result)


def provenance(seed=17):
    missing = classify_stats({"object_present": False, "payload_bytes": None, "raw_words": None})
    zeroed = classify_stats({"object_present": True, "payload_bytes": 232, "raw_words": [0] * 29})
    wrapped = counter_delta(254, 2, bits=8, wrap_verified=True)
    assert wrapped == 4 and missing["physical_duration_seconds"] is None
    return report("pmu-provenance", {"missing": missing, "zeroed": zeroed,
                  "explicit_synthetic_8bit_wrap_delta": wrapped,
                  "constant_request_payload_is_not_a_cumulative_delta": [100, 100, 100],
                  "seed": seed, "actual_device_executed": False},
                  scope="CPU counter provenance/lifecycle examples; no physical counters or rates measured")
