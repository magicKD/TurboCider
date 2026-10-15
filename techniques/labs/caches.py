"""Content/recipe cache identity, residual replay and token merge references."""
import hashlib
import json
import math
import random

try:
    from .common import metrics, report
except ImportError:
    from common import metrics, report


def content_key(*, prompt, references, model, adapter, precision, route, sigma):
    document = {"prompt": prompt, "ordered_reference_sha256": [hashlib.sha256(value).hexdigest() for value in references],
                "model": model, "adapter": adapter, "precision": precision, "route": route, "sigma": sigma}
    return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()


def cache_identity(seed=17):
    base = dict(prompt="a blue teapot", references=[b"reference-A", b"reference-B"], model="verified-generation-1",
                adapter="base", precision="bf16", route="gpu", sigma=1.0)
    key = content_key(**base)
    changes = {"same-path-new-bytes": {"references": [b"changed-A", b"reference-B"]},
               "reference-order": {"references": list(reversed(base["references"]))},
               "adapter": {"adapter": "adapter-A:strength-1"}, "precision": {"precision": "w8a8-v2"},
               "route": {"route": "runtime-ane"}, "sigma": {"sigma": .5}, "source-generation": {"model": "verified-generation-2"}}
    invalidations = {name: content_key(**{**base, **change}) != key for name, change in changes.items()}
    assert all(invalidations.values()) and content_key(**base) == key
    return report("cache-identity", {"repeated_condition_hit": True, "invalidations": invalidations,
                  "seed_omitted_deliberately": "Only safe when the cached object is proven independent of noise/seed; not a universal KV rule.",
                  "negative_control": "A path-only key would miss same-path-new-bytes; dimensions alone would miss every content change."})


class ResidualReplay:
    def __init__(self, threshold, warmup=2, max_skips=2):
        if (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0 or
                type(warmup) is not int or warmup < 0 or type(max_skips) is not int or max_skips < 0):
            raise ValueError("invalid cache policy")
        self.threshold, self.warmup, self.max_skips = threshold, warmup, max_skips
        self.reset()

    def reset(self):
        self.previous, self.residual, self.last_step, self.key, self.skips = None, None, -1, None, 0

    def run(self, step, total, key, front, middle):
        if (type(step) is not int or type(total) is not int or not 0 <= step < total or
                not front or not all(math.isfinite(x) for x in front)):
            self.reset()
            raise ValueError("valid step and nonempty finite front required")
        if step != self.last_step + 1 or key != self.key or (self.previous is not None and len(front) != len(self.previous)):
            self.reset()
        delta = (math.fsum(abs(a - b) for a, b in zip(front, self.previous)) /
                 max(math.fsum(abs(x) for x in self.previous), 1e-12)) if self.previous else math.inf
        reuse = (self.residual is not None and step >= self.warmup and step != total - 1 and
                 self.skips < self.max_skips and delta < self.threshold)
        if reuse:
            result = [x + residual for x, residual in zip(front, self.residual)]
            self.skips += 1
        else:
            try:
                result = middle(front)
                if len(result) != len(front) or not all(math.isfinite(x) for x in result):
                    raise ValueError("middle output shape/nonfinite mismatch")
            except BaseException:
                self.reset()
                raise
            self.residual = [out - x for out, x in zip(result, front)]
            if not all(math.isfinite(x) for x in self.residual):
                self.reset()
                raise ValueError("residual subtraction overflow")
            self.skips = 0
        if not all(math.isfinite(x) for x in result):
            self.reset()
            raise ValueError("replayed output nonfinite")
        # Update comparison even on skipped steps. Own lists, never caller views.
        self.previous, self.last_step, self.key = list(front), step, key
        return result, reuse


def residual_reuse(seed=17):
    rng, steps = random.Random(seed), 20
    initial = [rng.uniform(-.4, .4) for _ in range(16)]
    def run(threshold):
        controller, x, calls, trace = ResidualReplay(threshold), list(initial), 0, []
        for step in range(steps):
            front = [math.tanh(value + .01 * (1 - step / steps)) for value in x]
            def middle(values):
                nonlocal calls
                calls += 1
                out = list(values)
                for _ in range(4):
                    out = [value + .008 * math.tanh(.9 * value + .1 * values[(index + 1) % len(values)]) for index, value in enumerate(out)]
                return out
            result, reused = controller.run(step, steps, "same-condition", front, middle)
            x = [value - .03 * velocity for value, velocity in zip(x, result)]
            trace.append({"step": step, "reused": reused})
        assert not trace[0]["reused"] and not trace[-1]["reused"]
        return x, calls, trace
    baseline, original_calls, _ = run(0)
    sweeps = []
    for threshold in (0, .03, .1):
        output, calls, trace = run(threshold)
        sweeps.append({"threshold": threshold, "middle_calls": calls, "saved_middle_calls": original_calls - calls,
                       "final_trajectory_error": metrics(baseline, output), "trace": trace})
    controller = ResidualReplay(1.0)
    controller.run(0, 20, "A", [1.0], lambda x: [1.1])
    controller.run(1, 20, "A", [1.0], lambda x: [1.1])
    _, gap_reuse = controller.run(7, 20, "A", [1.0], lambda x: [1.2])
    _, changed_reuse = controller.run(8, 20, "B", [1.0], lambda x: [1.3])
    assert not gap_reuse and not changed_reuse
    return report("residual-reuse", {"steps": steps, "baseline_middle_calls": original_calls, "sweep": sweeps,
                  "gap_and_condition_reset": True,
                  "limits": "Toy denoising recurrence. Saved calls are counted execution, not a device speedup or image-quality qualification."})


def token_merge(seed=17):
    rng = random.Random(seed)
    tokens = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(14)]
    protected = 2
    groups = [[index] for index in range(protected)] + [list(range(first, min(first + 3, len(tokens)))) for first in range(protected, len(tokens), 3)]
    masses = [1] * len(tokens)
    merged, merged_masses = [], []
    for group in groups:
        mass = sum(masses[index] for index in group)
        merged.append([math.fsum(tokens[index][dim] * masses[index] for index in group) / mass for dim in range(4)])
        merged_masses.append(mass)
    restored = [[0.] * 4 for _ in tokens]
    for group, token in zip(groups, merged):
        for index in group:
            restored[index] = list(token)
    weighted_original = [math.fsum(row[dim] for row in tokens) for dim in range(4)]
    weighted_merged = [math.fsum(row[dim] * mass for row, mass in zip(merged, merged_masses)) for dim in range(4)]
    assert restored[:protected] == tokens[:protected]
    assert metrics(weighted_original, weighted_merged)["max_abs"] < 1e-12
    return report("token-merge", {"input_tokens": len(tokens), "merged_tokens": len(merged), "groups": groups,
                  "protected_prefix_exact": True, "mass_preservation": metrics(weighted_original, weighted_merged),
                  "unmerge_error": metrics(tokens, restored), "original_position_ids": list(range(len(tokens))),
                  "revisit": "Try norm/density/semantic matching, stale compensation and frame-aware groups. Retain original RoPE positions; mean preservation is not attention equivalence."})
