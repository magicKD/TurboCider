"""Calibration and exact SwiGLU reparameterization for a partial ANE FFN.

The GPU complement receives the original activation. S1^-1 is consequently
applied inside the ANE graph, while S2 is folded into the up/down matrices.
"""

import hashlib


def load_calibration(path, rows, hidden, np, pad_rows=False, source_rows=None):
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise ValueError("calibration must be a real .npy file or per-block directory")
    read_rows = source_rows or rows
    if read_rows < rows:
        raise ValueError("calibration source rows cannot be shorter than export rows")
    paths = sorted(path.glob("*.npy")) if path.is_dir() else [path]
    if not paths or len(paths) > 256:
        raise ValueError("calibration must contain 1...256 real samples")
    samples = []
    digest = hashlib.sha256()
    for sample_path in paths:
        if sample_path.is_symlink() or not sample_path.is_file():
            raise ValueError("calibration sample must be a regular file")
        value = np.load(sample_path, mmap_mode="r", allow_pickle=False)
        if (value.dtype not in (np.float16, np.float32) or value.ndim not in (2, 3) or
                value.shape[-1] != hidden or
                (value.shape[-2] != read_rows and
                 not (pad_rows and value.shape[-2] == read_rows - 32))):
            raise ValueError(f"calibration must contain [{read_rows}, {hidden}] FP16/FP32 samples")
        digest.update(sample_path.name.encode())
        digest.update(b"\x00")
        with sample_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        for sample in (value if value.ndim == 3 else (value,)):
            if not np.isfinite(sample).all():
                raise ValueError("calibration contains nonfinite activations")
            samples.append(np.ascontiguousarray(sample[:rows])
                           if len(sample) >= rows else
                           np.pad(sample, ((0, rows - len(sample)), (0, 0))))
    if len(samples) > 256:
        raise ValueError("calibration exceeds 256 samples")
    return samples, digest.hexdigest()


def load_calibration_union(paths, rows, hidden, np, pad_rows=False, source_rows=None):
    """Combine real per-block sample directories without copying 2 GiB per prompt.

    A one-directory request retains the original digest/manifest identity.
    Multi-directory requests commit to every directory's content and order.
    """
    if not paths or len(paths) > 16:
        raise ValueError("calibration union requires 1...16 directories")
    if len(paths) == 1:
        return load_calibration(paths[0], rows, hidden, np, pad_rows=pad_rows,
                                source_rows=source_rows)
    combined = []
    digest = hashlib.sha256(b"turbocider.z_image.calibration_union.v1\x00")
    for path in paths:
        samples, source_digest = load_calibration(path, rows, hidden, np,
                                                  pad_rows=pad_rows,
                                                  source_rows=source_rows)
        digest.update(len(samples).to_bytes(4, "little"))
        digest.update(bytes.fromhex(source_digest))
        combined.extend(samples)
    if len(combined) > 256:
        raise ValueError("calibration union exceeds 256 samples")
    return combined, digest.hexdigest()


def channel_scale(a, w, alpha, np, low=1 / 16, high=16):
    if not 0 <= alpha <= 1 or not 0 < low <= high:
        raise ValueError("invalid SmoothQuant alpha or scale bounds")
    if a.shape != w.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(w).all():
        raise ValueError("invalid SmoothQuant statistics")
    if np.any(a < 0) or np.any(w < 0):
        raise ValueError("SmoothQuant statistics must be nonnegative")
    scale = np.power(np.maximum(a, 1e-8), alpha) / np.power(np.maximum(w, 1e-8), 1 - alpha)
    scale = np.clip(scale, low, high)
    scale[(a == 0) | (w == 0)] = 1
    return scale.astype(np.float32)


def representative_rows(sample, count, np):
    """Keep uniform image coverage and rare high-amplitude conditioning rows."""
    if sample.ndim != 2 or count < 1 or not len(sample):
        raise ValueError("invalid calibration row selection")
    if count >= len(sample):
        return np.arange(len(sample), dtype=np.int64)
    extreme_count = max(1, count // 4)
    regular = np.linspace(0, len(sample) - 1, count - extreme_count, dtype=np.int64)
    magnitude = np.max(np.abs(sample.astype(np.float32)), axis=1)
    extreme = np.argsort(magnitude)[-extreme_count:]
    return np.unique(np.concatenate((regular, extreme)))


def smooth_partial_ffn(gate, up, down, samples, alpha1, alpha2, np, hidden_rows=32,
                       outlier_rows=False, hidden_region_rows=None):
    if (gate.shape != up.shape or down.shape[1] != gate.shape[0] or
            not len(samples) or samples[0].shape[-1] != gate.shape[1] or hidden_rows < 1 or
            (hidden_region_rows is not None and hidden_region_rows < 1)):
        raise ValueError("invalid partial FFN/calibration geometry")
    gate = np.asarray(gate, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    a1 = np.zeros(gate.shape[1], dtype=np.float32)
    for sample in samples:
        for first in range(0, len(sample), 128):
            a1 = np.maximum(a1, np.max(np.abs(sample[first:first + 128].astype(np.float32)), axis=0))
    w1 = np.maximum(np.max(np.abs(gate), axis=0), np.max(np.abs(up), axis=0))
    s1 = channel_scale(a1, w1, alpha1, np)
    a2 = np.zeros(gate.shape[0], dtype=np.float32)
    for sample in samples:
        region = sample[:hidden_region_rows] if hidden_region_rows is not None else sample
        indexes = (representative_rows(region, hidden_rows, np) if outlier_rows else
                   np.linspace(0, len(region) - 1, min(hidden_rows, len(region)), dtype=np.int64))
        for first in range(0, len(indexes), 8):
            x = sample[indexes[first:first + 8]].astype(np.float32)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                g = x @ gate.T
                u = x @ up.T
                z = (g / (1 + np.exp(-np.clip(g, -80, 80)))) * u
            a2 = np.maximum(a2, np.max(np.abs(z), axis=0))
    s2 = channel_scale(a2, np.max(np.abs(down), axis=0), alpha2, np)
    transformed = (gate * s1[None, :], up * (s1[None, :] / s2[:, None]),
                   down * s2[None, :])
    if not all(np.isfinite(value).all() for value in transformed):
        raise ValueError("nonfinite SmoothQuant reparameterization")
    return (*transformed, s1, s2)


def calibrate_input_a8(gate, up, down, s1, samples, np, rows_per_sample=32,
                       outlier_rows=False):
    """Search input clipping thresholds by partial-FFN output NMSE.

    Include the maximum of *all* calibration rows as a non-clipping fallback.
    This models the first A8 boundary alone; hidden A8 is searched afterward.
    """
    if rows_per_sample < 1 or not samples:
        raise ValueError("invalid input A8 calibration")
    sampled = []
    maximum = 0.
    for sample in samples:
        if sample.shape[-1] != len(s1):
            raise ValueError("invalid input A8 calibration shape")
        ids = (representative_rows(sample, rows_per_sample, np) if outlier_rows else
               np.linspace(0, len(sample) - 1, min(rows_per_sample, len(sample)), dtype=np.int64))
        normalized = sample.astype(np.float32) / s1
        maximum = max(maximum, float(np.max(np.abs(normalized))))
        sampled.append(normalized[ids])
    x = np.concatenate(sampled, axis=0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = x @ gate.T
        hidden = (projected / (1 + np.exp(-np.clip(projected, -80, 80)))) * (x @ up.T)
        target = hidden @ down.T
    if not np.isfinite(target).all():
        raise ValueError("nonfinite input A8 calibration reference")
    denominator = max(float(np.sum(target.astype(np.float64) ** 2)), 1e-12)
    trials = []
    for percentile in (99., 99.5, 99.9, 99.95, 99.99, 99.999, 100.):
        threshold = (maximum if percentile == 100. else
                     float(np.percentile(np.abs(x), percentile)))
        scale = np.float16(max(threshold / 127., 1e-6))
        if not np.isfinite(scale) or scale == 0:
            continue
        reconstructed = np.clip(np.rint(x / float(scale)), -127, 127) * float(scale)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = reconstructed @ gate.T
            hidden = (projected / (1 + np.exp(-np.clip(projected, -80, 80)))) * (reconstructed @ up.T)
            output = hidden @ down.T
        if not np.isfinite(output).all():
            continue
        score = float(np.sum((output - target).astype(np.float64) ** 2) / denominator)
        trials.append({"percentile": percentile, "scale": float(scale), "nmse": score})
    if not trials:
        raise ValueError("no valid input A8 threshold")
    best = min(trials, key=lambda trial: trial["nmse"])
    return np.float16(best["scale"]), trials


def calibrate_hidden_a8(gate, up, down, s1, samples, input_scale,
                        activation_scale, np, rows_per_sample=128, outlier_rows=False):
    """Minimize partial-FFN output NMSE across real strided calibration rows.

    Unlike a max-only tensor threshold, this measures the downstream cost of
    clipping rare hidden outliers. Inputs simulate the first A8 boundary too.
    """
    if rows_per_sample < 1 or activation_scale < 1 or not samples:
        raise ValueError("invalid hidden A8 calibration")
    hidden = []
    for sample in samples:
        ids = (representative_rows(sample, rows_per_sample, np) if outlier_rows else
               np.linspace(0, len(sample) - 1, min(rows_per_sample, len(sample)), dtype=np.int64))
        for first in range(0, len(ids), 8):
            x = sample[ids[first:first + 8]].astype(np.float32) / s1
            x = np.clip(np.rint(x / input_scale), -127, 127) * input_scale
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                g = x @ gate.T
                u = x @ up.T
                z = (g / (1 + np.exp(-np.clip(g, -80, 80)))) * u / (activation_scale ** 2)
            if not np.isfinite(z).all():
                raise ValueError("nonfinite hidden A8 calibration")
            hidden.append(z)
    z = np.concatenate(hidden, axis=0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        target = z @ down.T
    if not np.isfinite(target).all():
        raise ValueError("nonfinite calibration partial output")
    denominator = max(float(np.sum(target.astype(np.float64) ** 2)), 1e-12)
    trials = []
    for percentile in (99., 99.5, 99.9, 99.95, 99.99, 99.999, 100.):
        threshold = float(np.percentile(np.abs(z), percentile))
        scale = np.float16(max(threshold / 127., 1e-6))
        if not np.isfinite(scale) or scale == 0:
            continue
        reconstructed = np.clip(np.rint(z / float(scale)), -127, 127) * float(scale)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            projected = reconstructed @ down.T
        if not np.isfinite(projected).all():
            continue
        score = float(np.sum((projected - target).astype(np.float64) ** 2) / denominator)
        trials.append({"percentile": percentile, "scale": float(scale), "nmse": score})
    if not trials:
        raise ValueError("no valid hidden A8 threshold")
    best = min(trials, key=lambda trial: trial["nmse"])
    return np.float16(best["scale"]), trials


def calibrate_hidden_channel_groups(gate, up, down, s1, samples, input_scale,
                                    activation_scale, groups, np, rows_per_sample=128):
    """Output-aware static hidden A8 thresholds for contiguous channel groups.

    Each group's down-projection columns correspond exactly to its gated/up
    rows. The incoming A8 boundary and SmoothQuant scales remain shared.
    """
    if (groups < 2 or gate.ndim != 2 or gate.shape != up.shape or
            down.ndim != 2 or down.shape[1] != gate.shape[0] or
            gate.shape[0] % groups):
        raise ValueError("invalid hidden A8 channel-group geometry")
    width = gate.shape[0] // groups
    return [calibrate_hidden_a8(
        gate[start:start + width], up[start:start + width],
        down[:, start:start + width], s1, samples, input_scale,
        activation_scale, np, rows_per_sample=rows_per_sample,
        outlier_rows=True)[0] for start in range(0, gate.shape[0], width)]


def calibrate_adaptive_hidden_a8(gate, up, s1, samples, input_scale,
                                 activation_scale, image_rows, bins, np,
                                 rows_per_sample=128):
    """Fit a small bank of constant A8 scales from training image rows only."""
    if (bins < 2 or rows_per_sample < 1 or image_rows < 1 or not samples or
            gate.shape != up.shape or gate.shape[1] != len(s1) or activation_scale <= 0):
        raise ValueError("invalid adaptive hidden A8 calibration")
    peaks = []
    for sample in samples:
        if sample.ndim != 2 or len(sample) < image_rows or sample.shape[1] != len(s1):
            raise ValueError("invalid adaptive hidden A8 training sample")
        region = sample[:image_rows]
        ids = representative_rows(region, rows_per_sample, np)
        x = np.clip(np.rint(region[ids].astype(np.float32) / s1 / input_scale),
                    -127, 127) * input_scale
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            g = x @ gate.T
            u = x @ up.T
            hidden = (g / (1 + np.exp(-np.clip(g, -80, 80)))) * u / activation_scale ** 2
        if not np.isfinite(hidden).all():
            raise ValueError("nonfinite adaptive hidden A8 calibration")
        peaks.extend(np.max(np.abs(hidden), axis=1))
    # More resolution around normal rows, and a non-clipping training ceiling
    # for the rare high-amplitude rows.
    quantiles = [100. * (1 - 2. ** -(index + 1)) for index in range(bins - 1)] + [100.]
    scales = np.maximum(np.percentile(peaks, quantiles) / 127., 1e-6)
    scales = np.unique(scales.astype(np.float16).astype(np.float32))
    if len(scales) < 2 or not np.isfinite(scales).all():
        raise ValueError("adaptive hidden A8 scale bank collapsed")
    return scales


def route_channel_indices(selected_groups, group_width, total_width, ane_width, np):
    """Expand ANE groups and return the sorted, exact GPU complement."""
    if (not isinstance(group_width, int) or group_width < 32 or
            total_width % group_width or ane_width % group_width or
            len(selected_groups) != ane_width // group_width or
            list(selected_groups) != sorted(set(selected_groups)) or
            any(not isinstance(index, int) or isinstance(index, bool) or
                index < 0 or index >= total_width // group_width
                for index in selected_groups)):
        raise ValueError("invalid channel routing groups")
    ane = np.concatenate([np.arange(group * group_width, (group + 1) * group_width)
                          for group in selected_groups])
    gpu = np.setdiff1d(np.arange(total_width), ane)
    if (len(ane) != ane_width or len(gpu) != total_width - ane_width or
            not np.array_equal(np.sort(np.concatenate((ane, gpu))), np.arange(total_width))):
        raise ValueError("channel routing does not cover the FFN")
    return ane, gpu


def screen_hidden_range_order(gate, up, s1, samples, input_scale,
                              activation_scale, np, rows_per_sample=32):
    """Training-only hidden-channel ordering by the A8-visible magnitude.

    Group the least dynamic hidden channels together before calibrating A8
    group scales. This is an exact offline permutation of gate/up/down weights,
    not a runtime gather. Held-out samples must be assessed separately.
    """
    if (gate.ndim != 2 or gate.shape != up.shape or
            s1.shape != (gate.shape[1],) or not samples or
            rows_per_sample < 1 or input_scale <= 0 or activation_scale < 1):
        raise ValueError("invalid hidden range ordering inputs")
    maximum = np.zeros(gate.shape[0], dtype=np.float32)
    for sample in samples:
        ids = representative_rows(sample, rows_per_sample, np)
        for first in range(0, len(ids), 16):
            x = sample[ids[first:first + 16]].astype(np.float32) / s1
            x = np.clip(np.rint(x / input_scale), -127, 127) * input_scale
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                g = x @ gate.T
                hidden = (g / (1 + np.exp(-np.clip(g, -80, 80)))) * (x @ up.T)
                hidden /= activation_scale ** 2
            if not np.isfinite(hidden).all():
                raise ValueError("nonfinite hidden range ordering")
            maximum = np.maximum(maximum, np.max(np.abs(hidden), axis=0))
    return np.argsort(maximum, kind="stable").astype(np.int32)


def screen_joint_hidden_channel_groups(gate, up, down, s1, samples, input_scale,
                                       activation_scale, groups, np, rows_per_sample=32):
    """Research-only: coordinate-search group A8 scales against combined output.

    Include the input A8 error in the target, since groupwise hidden rounding
    can either compound or cancel it. Callers must evaluate independent held-out
    samples; optimizing this calibration objective does not establish quality.
    """
    if (groups < 2 or gate.ndim != 2 or gate.shape != up.shape or
            down.ndim != 2 or down.shape[1] != gate.shape[0] or
            gate.shape[0] % groups or s1.shape != (gate.shape[1],) or
            not samples or rows_per_sample < 1 or input_scale <= 0 or
            activation_scale < 1):
        raise ValueError("invalid joint hidden A8 group geometry")
    selected = np.concatenate([
        sample[representative_rows(sample, rows_per_sample, np)].astype(np.float32)
        for sample in samples], axis=0)
    x = selected / s1
    x_a8 = np.clip(np.rint(x / input_scale), -127, 127) * input_scale
    step = gate.shape[0] // groups
    # All errors use the same full-precision FFN output reference, including
    # the first A8 boundary. This makes group interactions visible.
    reference = np.zeros((len(x), down.shape[0]), np.float32)
    projected = []
    scales = []
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for begin in range(0, gate.shape[0], step):
            end = begin + step
            g, u, d = gate[begin:end], up[begin:end], down[:, begin:end]
            def hidden(value):
                v = value @ g.T
                return (v / (1 + np.exp(-np.clip(v, -80, 80)))) * (value @ u.T) / (
                    activation_scale ** 2)
            reference += hidden(x) @ d.T
            z = hidden(x_a8)
            if not np.isfinite(z).all():
                raise ValueError("nonfinite joint hidden A8 calibration")
            trials = []
            options = []
            for percentile in (99., 99.5, 99.9, 99.95, 99.99, 99.999, 100.):
                threshold = float(np.percentile(np.abs(z), percentile))
                scale = np.float16(max(threshold / 127., 1e-6))
                if not np.isfinite(scale) or scale <= 0 or scale in trials:
                    continue
                trial = np.clip(np.rint(z / float(scale)), -127, 127) * float(scale)
                result = (trial @ d.T).astype(np.float32)
                if not np.isfinite(result).all():
                    continue
                trials.append(scale)
                options.append(result)
            if not options:
                raise ValueError("no finite joint hidden A8 candidate")
            scales.append(trials)
            projected.append(options)
    if not np.isfinite(reference).all():
        raise ValueError("nonfinite joint hidden A8 reference")
    # Start from each group's best local candidate, then improve the sum of
    # projected outputs. A fixed order and strict improvement make this
    # deterministic even when candidates tie.
    choices = [0] * groups
    total = sum((options[0] for options in projected), np.zeros_like(reference))
    def loss(value):
        delta = (value - reference).astype(np.float64)
        return float(np.sum(delta * delta))
    initial_loss = loss(total)
    for _ in range(3):
        changed = False
        for group, options in enumerate(projected):
            base = total - options[choices[group]]
            best = min(range(len(options)), key=lambda index: loss(base + options[index]))
            if best != choices[group]:
                total = base + options[best]
                choices[group] = best
                changed = True
        if not changed:
            break
    return [scales[index][choice] for index, choice in enumerate(choices)], {
        "initial_nmse": initial_loss / max(float(np.sum(reference.astype(np.float64) ** 2)), 1e-12),
        "joint_nmse": loss(total) / max(float(np.sum(reference.astype(np.float64) ** 2)), 1e-12),
    }


def verify_w8a8(spec, expected, allow_adaptive_scale=False,
                allow_input_region_scale=False):
    """Fail closed if either GEMM lacks dynamic activation Q/DQ or W8 storage.

    MIL representation does not establish physical ANE placement; that still
    requires a device trace and an end-to-end benchmark.
    """
    if spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("W8A8 requires an ML Program")
    found = set()
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            producers = {value.name: op for op in block.operations for value in op.outputs}

            def inputs(op, key):
                return [binding.name for arg in ([op.inputs[key]] if key in op.inputs else [])
                        for binding in arg.arguments if binding.WhichOneof("binding") == "name"]

            def path(name, wanted, allowed, seen=None):
                seen = set() if seen is None else seen
                if name in seen or name not in producers:
                    return False
                op = producers[name]
                if op.type == wanted:
                    return True
                if op.type not in allowed:
                    return False
                return any(path(child, wanted, allowed, seen | {name})
                           for child in inputs(op, "input") + inputs(op, "x") +
                           inputs(op, "data") + inputs(op, "values"))

            def activation_qdq(name, seen=None):
                seen = set() if seen is None else seen
                if name in seen or name not in producers:
                    return False
                op = producers[name]
                if op.type == "dequantize":
                    return any(path(child, "quantize", {"cast", "reshape", "identity"})
                               for child in inputs(op, "input"))
                if op.type == "mul" and ((allow_adaptive_scale and
                        op.outputs[0].name in ("a8_hidden_adaptive_restore",
                                               "a8_hidden_caption_restore",
                                               "a8_hidden_image_unresmooth")) or
                        (allow_input_region_scale and
                         op.outputs[0].name == "a8_input_region_restore")):
                    # Image rows may be restored after the *single* shared
                    # Q/DQ. The scale path must select only constant values;
                    # never permit a second unquantized activation data path.
                    def constant_scale_bank(value, visited=None):
                        visited = set() if visited is None else visited
                        if value in visited or value not in producers:
                            return False
                        candidate = producers[value]
                        if candidate.type == "const":
                            return True
                        if candidate.type != "select":
                            return False
                        def constant_branch(key):
                            arguments = candidate.inputs[key].arguments if key in candidate.inputs else []
                            if len(arguments) != 1:
                                return False
                            binding = arguments[0]
                            return (binding.WhichOneof("binding") == "value" or
                                    binding.WhichOneof("binding") == "name" and
                                    constant_scale_bank(binding.name, visited | {value}))
                        return constant_branch("a") and constant_branch("b")
                    x, y = inputs(op, "x"), inputs(op, "y")
                    if op.outputs[0].name in ("a8_hidden_image_unresmooth",
                                               "a8_input_region_restore"):
                        return (len(x) == 1 and len(y) == 1 and
                                activation_qdq(x[0], seen | {name}) and
                                y[0] in producers and producers[y[0]].type == "const")
                    return (len(x) == 1 and len(y) == 1 and
                            activation_qdq(x[0], seen | {name}) and
                            constant_scale_bank(y[0]))
                if op.type == "select":
                    # A token-dependent selector is admissible only when *both*
                    # candidate branches traverse their own activation Q/DQ.
                    branches = inputs(op, "a") + inputs(op, "b")
                    return (len(branches) == 2 and
                            all(activation_qdq(child, seen | {name}) for child in branches))
                if op.type not in {"concat", "cast", "reshape", "transpose", "identity"}:
                    return False
                children = (inputs(op, "values") + inputs(op, "input") +
                            inputs(op, "x") + inputs(op, "data"))
                return bool(children) and all(activation_qdq(child, seen | {name})
                                              for child in children)

            for op in block.operations:
                if op.type == "quantize" and any(
                        n == expected[0] for n in inputs(op, "input")):
                    raise ValueError("SwiGLU gate/up output must remain FP16, not shared A8")
                if op.type not in {"conv", "linear"}:
                    continue
                label = op.outputs[0].name
                if "name" in op.attributes:
                    names = op.attributes["name"].immediateValue.tensor.strings.values
                    if names:
                        label = names[0]
                if label not in expected:
                    continue
                activation = all(activation_qdq(n) for n in inputs(op, "x"))
                weight = any(path(n, "constexpr_blockwise_shift_scale",
                                  {"cast", "reshape", "transpose", "identity", "constexpr_lut_to_dense"})
                             for n in inputs(op, "weight"))
                # Do not accept weight-only INT8 or a stray, unrelated Q/DQ.
                if not activation or not weight:
                    raise ValueError(f"{label} does not have both activation and weight INT8")
                found.add(label)
    if found != set(expected):
        raise ValueError(f"W8A8 projection coverage mismatch: {found} != {set(expected)}")
    return sorted(found)
