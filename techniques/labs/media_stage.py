"""Strict PNG/source and recorded pixel-stage contracts; no device imports.

The public CPU/GPU stage is NOT a decoder, VideoToolbox placement test, resize,
VAE/request benchmark or a physical-overlap proof. Original file pixels stay local.
"""
import hashlib
import math
from pathlib import Path
import statistics
import struct
import zlib

from common import report

ARMS = ("product-loop", "vimage", "gpu-fused")
MAX_ABS = 1e-6


class SourceRefused(ValueError):
    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png_rgb(path):
    """Decode bounded, opaque, non-interlaced RGB/RGBA8 PNG without ImageIO.

    This verifies file sample bytes, NOT ICC/color-management/display semantics.
    Alpha must be entirely opaque. CRC, filters, exact stream end and cap checked.
    """
    path = Path(path)
    require(path.is_file() and 33 <= path.stat().st_size <= 8 * 1024**2, "bounded explicit PNG file required")
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature required")
    offset, header, payload, ended, idat_closed = 8, None, bytearray(), False, False
    while offset < len(data):
        require(offset + 12 <= len(data) and not ended, "complete single PNG required")
        size = struct.unpack_from(">I", data, offset)[0]
        require(size <= 8 * 1024**2 and offset + 12 + size <= len(data), "bounded complete PNG chunk required")
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + size]
        require(zlib.crc32(kind + body) == struct.unpack_from(">I", data, offset + 8 + size)[0], "PNG CRC differs")
        if header is None:
            require(kind == b"IHDR" and size == 13, "first unique PNG IHDR required")
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            require(not idat_closed, "consecutive PNG IDAT required")
            payload.extend(body)
        elif kind == b"IEND":
            require(size == 0 and payload, "complete nonempty PNG stream required")
            ended = True
        else:
            require(kind != b"IHDR" and kind not in (b"acTL", b"fcTL", b"fdAT") and kind[0] & 32, "unsupported critical/animated PNG")
            idat_closed |= bool(payload)
        offset += 12 + size
    require(ended and header is not None, "PNG end required")
    width, height, bits, color, compression, filtering, interlace = header
    require(1 <= width <= 2048 and 1 <= height <= 2048 and bits == 8 and color in (2, 6)
            and (compression, filtering, interlace) == (0, 0, 0), "bounded non-interlaced RGB/RGBA8 PNG required")
    channels = 3 if color == 2 else 4
    pitch = width * channels
    limit = height * (pitch + 1)
    decoder = zlib.decompressobj()
    filtered = decoder.decompress(bytes(payload), limit + 1)
    require(len(filtered) == limit and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            "bounded complete single PNG zlib stream required")
    pixels = bytearray(height * pitch)
    previous = bytearray(pitch)
    for y in range(height):
        method = filtered[y * (pitch + 1)]
        require(method <= 4, "unknown PNG scanline filter")
        row = bytearray(filtered[y * (pitch + 1) + 1:(y + 1) * (pitch + 1)])
        for i in range(pitch):
            left, above, upper_left = row[i - channels] if i >= channels else 0, previous[i], previous[i - channels] if i >= channels else 0
            if method == 1:
                predictor = left
            elif method == 2:
                predictor = above
            elif method == 3:
                predictor = (left + above) // 2
            elif method == 4:
                estimate = left + above - upper_left
                distances = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            else:
                predictor = 0
            row[i] = (row[i] + predictor) & 255
        pixels[y * pitch:(y + 1) * pitch] = row
        previous = row
    if channels == 4:
        alpha = pixels[3::4]
        if not all(v == 255 for v in alpha):
            raise SourceRefused("opaque file samples required; no hidden-RGB/color claim",
                                {"width": width, "height": height, "file_sha256": sha(data),
                                 "alpha_min": min(alpha), "alpha_max": max(alpha),
                                 "opaque_pixels": alpha.count(255), "total_pixels": width * height})
        rgb = bytearray(width * height * 3)
        for channel in range(3):
            rgb[channel::3] = pixels[channel::4]
        pixels = rgb
    return {"width": width, "height": height, "file_sha256": sha(data), "rgb_sha256": sha(pixels)}, bytes(pixels)


def synthetic_rgb(width=513, height=257, seed=17):
    require(type(width) is int and type(height) is int and type(seed) is int and 1 <= width <= 2048 and 1 <= height <= 2048 and 0 <= seed <= 10000,
            "bounded synthetic RGB geometry/seed required")
    return bytes((x * 13 + y * 7 + channel * 43 + (x * y) % 31 + seed * 17) % 256
                 for y in range(height) for x in range(width) for channel in range(3))


def source_bgra(rgb, width, height, generation):
    require(type(width) is int and type(height) is int and 1 <= width <= 2048 and 1 <= height <= 2048
            and type(generation) is int and len(rgb) == width * height * 3 and generation in (0, 1), "complete RGB and explicit two-generation control required")
    pitch = ((width * 4 + 63) // 64 + 1) * 64
    out = bytearray(b"\xa5" * (pitch * height))
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            position = y * pitch + x * 4
            out[position:position + 4] = bytes((rgb[index + 2] ^ generation, rgb[index + 1] ^ generation,
                                               rgb[index] ^ generation, (x * 11 + y * 3) % 256))
    return pitch, bytes(out)


def oriented_rgb(rgb, width, height, code, generation=0):
    require(type(code) is int and 0 <= code < 8 and type(generation) is int and generation in (0, 1)
            and type(width) is int and type(height) is int and 1 <= width <= 2048 and 1 <= height <= 2048 and len(rgb) == width * height * 3,
            "complete RGB and supported orientation required")
    rotated, flip_x, flip_y = bool(code & 4), bool(code & 1), bool(code & 2)
    ow, oh = (height, width) if rotated else (width, height)
    out = bytearray(len(rgb))
    # Independent forward scatter, unlike the product's inverse source gather.
    for sy in range(height):
        for sx in range(width):
            tx, ty = (sy, sx) if rotated else (sx, sy)
            if flip_x:
                tx = ow - 1 - tx
            if flip_y:
                ty = oh - 1 - ty
            start, target = (sy * width + sx) * 3, (ty * ow + tx) * 3
            out[target:target + 3] = bytes(v ^ generation for v in rgb[start:start + 3])
    return ow, oh, bytes(out)


def order(repeats):
    require(type(repeats) is int and repeats in (10, 30), "bounded measured groups required")
    result = []
    for group in range(repeats + 3):
        phase = "first" if group == 0 else "warmup" if group < 3 else "measured"
        trial = 0 if group == 0 else group - 1 if group < 3 else group - 3
        row = group if group < 3 else trial
        for position in range(3):
            arm = ARMS[(row % 3 + (2 - position if row // 3 % 2 else position)) % 3]
            result.append((phase, trial, group % 2, arm))
    return result


def verify(native):
    require(isinstance(native, dict) and native.get("schema") == "tc-media-stage-native-v1" and native.get("status") == "measured"
            and native.get("errors") == [] and native.get("production_qualified") is False
            and native.get("whole_request_qualified") is False and native.get("physical_overlap_qualified") is False
            and native.get("all_owned_gpu_work_completed") is True and native.get("product_pe_decoder_matches_independent_png") is True
            and native.get("mlx_version") == "0.32.0",
            "complete stage protocol required; never promote failed/request/physical evidence")
    require(native.get("gate") == {"max_abs": MAX_ABS, "relative_l2": 1e-6, "cosine_min": .999999}
            and native.get("output_layout") == "contiguous-FP32-1C1HW" and native.get("resize") is False,
            "fixed normalization/layout/no-resize boundary required")
    scenarios = native["scenarios"]
    require(len(scenarios) == 16 and {(s["source"], s["orientation"]) for s in scenarios} == {(kind, code) for kind in ("file", "synthetic") for code in range(8)},
            "actual PNG and odd rectangular source in all eight orientations required")
    ratios = []
    for scenario in scenarios:
        require(type(scenario["orientation"]) is int and 0 <= scenario["orientation"] < 8
                and scenario["id"] == f'{scenario["source"]}-o{scenario["orientation"]}', "typed unique orientation identity required")
        width, height = scenario["source_wh"]
        require(type(width) is int and type(height) is int and width > 0 and height > 0 and type(scenario["source_pitch"]) is int
                and scenario["source_pitch"] == ((width * 4 + 63) // 64 + 1) * 64,
                "explicit padded physical input required")
        ow, oh = (height, width) if scenario["orientation"] & 4 else (width, height)
        require(scenario["output_wh"] == [ow, oh] and scenario["product_source_formula_bound"] is True, "complete oriented consumer required")
        samples = scenario["samples"]
        require([(s["phase"], s["trial"], s["generation"], s["arm"]) for s in samples] == order(native["measured_groups"]), "all balanced same-generation calls required")
        for sample in samples:
            metric = sample["metrics"]
            require(type(sample["trial"]) is int and type(sample["generation"]) is int
                    and sample["numeric_pass"] is True and sample["host_guards_pass"] is True and sample["source_unchanged"] is True
                    and sample["output_dtype"] == "float32" and type(sample["output_elements"]) is int and sample["output_elements"] == ow * oh * 3,
                    "complete output/source/guard gate required")
            require(isinstance(sample["output_sha256"], str) and len(sample["output_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in sample["output_sha256"]), "complete FP32 output content identity required")
            require(metric["finite"] is True and type(metric["recovered_rgb_mismatches"]) is int and metric["recovered_rgb_mismatches"] == 0
                    and all(type(metric[k]) in (int, float) and math.isfinite(metric[k]) for k in ("max_abs", "relative_l2", "cosine"))
                    and 0 <= metric["max_abs"] <= MAX_ABS and 0 <= metric["relative_l2"] <= 1e-6 and .999999 <= metric["cosine"] <= 1 + 1e-10,
                    "fixed full-output FP64/byte-recovery numerical gate rejected")
            require(type(sample["wall_ns"]) is int and sample["wall_ns"] > 0 and sample["mlx_stream_drained"] is True,
                    "ready consumer and same-process host wall required")
            expected_bytes = scenario["source_pitch"] * height if sample["arm"] == "gpu-fused" else width * height * 3
            require(type(sample["logical_import_bytes"]) is int and sample["logical_import_bytes"] == expected_bytes
                    and sample["rgb_sha256"] == metric["recovered_rgb_sha256"] == scenario["rgb_sha256_by_generation"][sample["generation"]], "same complete source and representation bytes required")
            code = scenario["orientation"]
            vimage_calls = 1 + bool(code & 4) + (bool(code & 1) != bool(code & 4)) + bool(code & 2) if sample["arm"] == "vimage" else 0
            require(type(sample["vimage_api_calls"]) is int and sample["vimage_api_calls"] == vimage_calls,
                    "actual out-of-place rotation/reflection and channel-conversion calls required")
            events = sample["events"]
            require([e["name"] for e in events] == ["cpu-transform", "mlx-import", "consumer-ready"], "whole CPU/import/GPU boundary required")
            end = 0
            for event in events:
                require(type(event["start_ns"]) is int and type(event["duration_ns"]) is int and event["start_ns"] >= end
                        and event["duration_ns"] >= 0 and event["start_ns"] + event["duration_ns"] <= sample["wall_ns"]
                        and event["clock_domain"] == "same-process-uptime-raw-host", "associated host interval required")
                end = event["start_ns"] + event["duration_ns"]
        for key in ("wrong_stride_control", "wrong_channels_control", "stale_actual_output_control"):
            control = scenario[key]
            require(control["rejected"] is True and control["finite"] is True and type(control["recovered_rgb_mismatches"]) is int and control["recovered_rgb_mismatches"] > 0
                    and control["used_for_speed_ratio"] is False, "actual finite-but-wrong control required")
            require(control["new_gpu_dispatch"] is (key != "stale_actual_output_control"), "cached observation is not a new GPU dispatch")
        measured = {trial: {sample["arm"]: sample["wall_ns"] for sample in samples if sample["phase"] == "measured" and sample["trial"] == trial}
                    for trial in range(native["measured_groups"])}
        ratios.append({"id": scenario["id"], "median_wall_ns": {arm: statistics.median(row[arm] for row in measured.values()) for arm in ARMS},
                       "paired_loop_over_vimage": statistics.median(row["product-loop"] / row["vimage"] for row in measured.values()),
                       "paired_loop_over_gpu": statistics.median(row["product-loop"] / row["gpu-fused"] for row in measured.values()),
                       "paired_vimage_over_gpu": statistics.median(row["vimage"] / row["gpu-fused"] for row in measured.values()),
                       "paired_best_cpu_over_gpu": statistics.median(min(row["product-loop"], row["vimage"]) / row["gpu-fused"] for row in measured.values()),
                       "best_cpu_is_per_trial_oracle_not_a_deployable_policy": True,
                       "production_qualified": False})
    return {"completed_stage_calls": sum(len(s["samples"]) for s in scenarios), "actual_gpu_negative_calls": 32,
            "cached_negative_observations": 16, "component_comparisons": ratios,
            "whole_request_qualified": False, "physical_overlap_qualified": False}


def stage_contract(seed=17):
    rgb = synthetic_rgb(5, 3, seed)
    witnesses = []
    for code in range(8):
        width, height, transformed = oriented_rgb(rgb, 5, 3, code)
        require(sorted(tuple(transformed[i:i + 3]) for i in range(0, len(transformed), 3)) == sorted(tuple(rgb[i:i + 3]) for i in range(0, len(rgb), 3)), "orientation must preserve all pixels")
        witnesses.append({"orientation": code, "output_wh": [width, height], "rgb_sha256": sha(transformed)})
    return report("media-stage-contract", {"orientations": witnesses, "source_pitch_bytes": source_bgra(rgb, 5, 3, 0)[0],
                  "actual_device_executed": False, "retrieval_grants_qualification": False},
                  scope="CPU byte/address contracts only; not vImage/GPU performance, decode, resize, VAE or codec")
