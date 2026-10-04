"""CPU directory/target-fill tests plus an explicitly enabled pinned GGML oracle.

TC_GGML_ORACLE_ROOT selects a read-only llama.cpp checkout at ORACLE_COMMIT.
TURBOCIDER_TEST_GGUF_MODELS=1 enables bounded samples from local Z Q8/Q4 files.
Neither optional fixture is downloaded or used as a production dependency.
"""
from __future__ import annotations

import ctypes as C
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ORACLE_COMMIT = "64e9bceb2c3a856efed96feda784a50947049feb"
TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
         7: (32, 24), 8: (32, 34), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 30: (1, 2)}


def text(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def metadata(key, typ, value):
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
    return text(key) + struct.pack("<I", typ) + (text(value) if typ == 8 else struct.pack("<" + fmt[typ], value))


def fixture(entries, meta=(), alignment=32):
    """Entries: (name, type, disk dimensions, optional relative offset)."""
    header = struct.pack("<IIQQ", 0x46554747, 3, len(entries), len(meta)) + b"".join(meta)
    offset = 0
    payload_size = 0
    for entry in entries:
        name, typ, dims = entry[:3]
        position = entry[3] if len(entry) == 4 else offset
        elements, block_bytes = TYPES.get(typ, (1, 1))
        size = math.prod(dims) // elements * block_bytes
        header += text(name) + struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", dim) for dim in dims)
        header += struct.pack("<IQ", typ, position)
        offset = (position + size + alignment - 1) // alignment * alignment
        payload_size = max(payload_size, position + size)
    # Malformed huge-dimension fixtures must never allocate their claimed payload.
    return header + bytes((-len(header)) % alignment) + bytes(min(payload_size, 1 << 20))


def block(typ, rng):
    _, size = TYPES[typ]
    data = bytearray(rng.randbytes(size))
    if typ == 0: return struct.pack("<f", rng.uniform(-3, 3))
    if typ == 1: return struct.pack("<e", rng.uniform(-3, 3))
    if typ == 30: return struct.pack("<H", struct.unpack("<I", struct.pack("<f", rng.uniform(-3, 3)))[0] >> 16)
    offset = 208 if typ == 14 else 0
    data[offset:offset + 2] = struct.pack("<e", rng.choice([0., -0., 2**-24, -0.125, 0.03125, 1.5]))
    if typ in (3, 7, 12, 13): data[2:4] = struct.pack("<e", rng.choice([0., -0.5, 0.25, 2**-24]))
    return bytes(data)


class GGUFExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="tc-gguf-execution-")
        cls.path = Path(cls.temp.name)
        compiler = ["xcrun", "clang++", "-std=c++20", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "native/core")]
        sources = [str(ROOT / "tests/native/gguf_core_probe.cpp"), str(ROOT / "native/core/gguf_decode.cpp"), str(ROOT / "native/core/gguf_affine.cpp")]
        cls.probe = cls.path / "probe"
        subprocess.run([*compiler, *sources, "-o", str(cls.probe)], check=True)
        library = cls.path / "decode.dylib"
        subprocess.run([*compiler, "-dynamiclib", *sources, "-o", str(library)], check=True)
        cls.library = C.CDLL(str(library))
        cls.library.tc_gguf_test_error.restype = C.c_char_p
        cls.library.tc_gguf_test_half.argtypes = [C.c_uint16]
        cls.library.tc_gguf_test_half.restype = C.c_float
        cls.library.tc_gguf_test_round.argtypes = [C.c_uint32, C.c_float, C.POINTER(C.c_uint16)]
        cls.library.tc_gguf_test_decode.argtypes = [C.c_uint32, C.c_void_p] + [C.c_uint64]*7 + [C.c_uint32, C.c_void_p] + [C.c_uint64]*3 + [C.POINTER(C.c_uint64), C.c_uint64, C.c_int, C.POINTER(C.c_uint64)]
        cls.library.tc_gguf_test_affine.argtypes=[C.c_uint32,C.c_void_p,C.c_uint64,C.c_uint64,C.c_uint64,C.c_uint32,C.c_void_p,C.c_uint64]

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def inspect(self, data, mode="inspect", option=None):
        path = self.path / "fixture.gguf"
        path.write_bytes(data)
        args = [str(self.probe), mode, str(path)]
        if option is not None: args.append(option)
        return subprocess.run(args, capture_output=True, text=True, timeout=10)

    def decode(self, raw, typ, source_rows, source_columns, row_begin=0, rows=None,
               column_begin=0, columns=None, dtype=0, row_stride=None, column_stride=None,
               gather=None, cancel=False, target_size=None):
        rows = source_rows if rows is None else rows
        if gather is not None: rows = len(gather)
        columns = source_columns - column_begin if columns is None else columns
        item = 4 if dtype == 0 else 2
        column_stride = item if column_stride is None else column_stride
        row_stride = columns * column_stride + 8 if row_stride is None else row_stride
        size = (rows - 1) * row_stride + columns * column_stride + 16 if target_size is None else target_size
        target = (C.c_ubyte * size)(*([0xA5] * size))
        source = C.create_string_buffer(raw, len(raw))
        indices = (C.c_uint64 * len(gather))(*gather) if gather is not None else None
        stats = (C.c_uint64 * 4)()
        status = self.library.tc_gguf_test_decode(typ, source, len(raw), source_rows, source_columns,
            row_begin, rows, column_begin, columns, dtype, target, size, row_stride, column_stride,
            indices, len(gather) if gather is not None else 0, int(cancel), stats)
        return status, bytes(target), list(stats), self.library.tc_gguf_test_error().decode(), row_stride, column_stride

    def values(self, result, rows, columns, dtype=0):
        status, output, _, error, row_stride, column_stride = result
        self.assertEqual(status, 0, error)
        width = 4 if dtype == 0 else 2
        values = []
        touched = set()
        for row in range(rows):
            for column in range(columns):
                start = row * row_stride + column * column_stride
                touched.update(range(start, start + width))
                values.append(output[start:start + width])
        self.assertTrue(all(value == 0xA5 for i, value in enumerate(output) if i not in touched), "padding/guard overwritten")
        return b"".join(values)

    def test_all_registered_types_directory_and_native_gate(self):
        for typ, (elements, _) in TYPES.items():
            with self.subTest(typ=typ):
                raw = fixture([("w", typ, [max(elements, 32), 3])])
                self.assertEqual(self.inspect(raw).returncode, 0)
                legacy = self.inspect(raw, "legacy")
                self.assertEqual(legacy.returncode == 0, typ in (0, 1, 2, 3, 8, 30))

    def test_directory_bounds_alignment_duplicates_overlap_and_blocks(self):
        invalid = [fixture([("w", 8, [32, 2])])[:-1],
                   fixture([("w", 8, [32], 1)]),
                   fixture([("w", 8, [32]), ("w", 8, [32])]),
                   fixture([("a", 8, [32], 0), ("b", 8, [32], 0)]),
                   fixture([("w", 8, [31])]), fixture([("w", 12, [32])]),
                   fixture([("w", 0, [0])]), fixture([("w", 0, [2**32])])[:256],
                   fixture([("w", 8, [32])], [metadata("general.alignment", 4, 3)]),
                   fixture([("w", 8, [32])], [metadata("k", 4, 1), metadata("k", 4, 2)])]
        for raw in invalid:
            with self.subTest(size=len(raw)): self.assertNotEqual(self.inspect(raw).returncode, 0)

    def test_metadata_and_parser_limits(self):
        array = text("array") + struct.pack("<IIQ", 9, 8, 2) + text("a") + text("b")
        nested = text("nested") + struct.pack("<IIQ", 9, 9, 1) + struct.pack("<IQ", 4, 2) + struct.pack("<II", 1, 2)
        raw = fixture([("weight0", 8, [32]), ("weight1", 0, [1])], [array, nested])
        self.assertEqual(self.inspect(raw).returncode, 0)
        for limit in ("retained", "directory", "string", "count", "array"):
            with self.subTest(limit=limit): self.assertNotEqual(self.inspect(raw, "limit", limit).returncode, 0)
        for value in (2, 255):
            self.assertNotEqual(self.inspect(fixture([("w", 8, [32])], [metadata("bool", 7, value)])).returncode, 0)
        bool_array = text("bool_array") + struct.pack("<IIQB", 9, 7, 1, 2)
        self.assertNotEqual(self.inspect(fixture([("w", 8, [32])], [bool_array])).returncode, 0)

    def test_split_set_complete_and_negative(self):
        def shard(number, count=2, total=2, arch="qwen3", name=None):
            meta = [metadata("split.no", 2, number), metadata("split.count", 2, count),
                    metadata("split.tensors.count", 5, total), metadata("general.architecture", 8, arch)]
            return fixture([(name or f"w{number}", 8, [32])], meta)
        first, second = self.path / "first.gguf", self.path / "second.gguf"
        first.write_bytes(shard(0))
        for raw, expected in [(shard(1), 0), (shard(0), 1), (shard(1, arch="other"), 1),
                              (shard(1, total=3), 1), (shard(1, name="w0"), 1)]:
            second.write_bytes(raw)
            result = subprocess.run([str(self.probe), "split", str(first), str(second)], capture_output=True)
            self.assertEqual(result.returncode, expected)
        self.assertNotEqual(subprocess.run([str(self.probe), "split", str(first)], capture_output=True).returncode, 0)

    def test_half_conversion_exhaustive_finite(self):
        result = C.c_uint16()
        for bits in range(65536):
            if bits & 0x7c00 == 0x7c00: continue
            expected = struct.unpack("<e", struct.pack("<H", bits))[0]
            actual = self.library.tc_gguf_test_half(bits)
            self.assertEqual(struct.pack("<f", actual), struct.pack("<f", expected))
            self.assertEqual(self.library.tc_gguf_test_round(1, actual, C.byref(result)), 0)
            self.assertEqual(result.value, bits)

    def test_rounding_ties_subnormals_and_overflow(self):
        result = C.c_uint16()
        for value in [0., -0., 2**-25, -2**-25, 3*2**-25, 1 + 2**-11, 1 + 3*2**-11, 65504., -65504.]:
            self.assertEqual(self.library.tc_gguf_test_round(1, value, C.byref(result)), 0)
            self.assertEqual(result.value, struct.unpack("<H", struct.pack("<e", value))[0])
        for value in [65520., -65520., float("inf"), float("nan")]:
            self.assertNotEqual(self.library.tc_gguf_test_round(1, value, C.byref(result)), 0)
        for bits in [0x3f808000, 0x3f818000, 0xbf808000, 1, 0x80000001, 0x7f7fffff]:
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
            status = self.library.tc_gguf_test_round(2, value, C.byref(result))
            expected = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
            self.assertEqual(status == 0, expected & 0x7f80 != 0x7f80)
            if status == 0: self.assertEqual(result.value, expected)

    def test_q8_signed_scale_and_simd_stride_parity(self):
        codes = bytes([128, 255, 0, 1, 127] + list(range(27)))
        raw = (struct.pack("<e", 0.125) + codes) * 8
        expected = b"".join(struct.pack("<f", (c if c < 128 else c-256)*0.125) for c in codes) * 8
        self.assertEqual(self.values(self.decode(raw, 8, 2, 128), 2, 128), expected)
        dense = self.values(self.decode(raw, 8, 2, 128, dtype=2), 2, 128, 2)
        strided = self.values(self.decode(raw, 8, 2, 128, dtype=2, column_stride=4), 2, 128, 2)
        self.assertEqual(dense, strided)
        rng=random.Random(81)
        for trial in range(32):
            raw=b"".join(block(8,rng) for _ in range(8))
            for dtype in (0,1,2):
                vector=self.values(self.decode(raw,8,2,128,dtype=dtype),2,128,dtype)
                scalar=self.values(self.decode(raw,8,2,128,dtype=dtype,column_stride=8),2,128,dtype)
                self.assertEqual(vector,scalar,(trial,dtype))
        overflowing=struct.pack("<e",512.)+bytes([128]*32)
        self.assertNotEqual(self.decode(overflowing,8,1,32,dtype=1)[0],0)

    def test_native_affine_codes_scales_biases_source_values(self):
        rng=random.Random(818)
        for typ in (2,3,8):
            bits=8 if typ==8 else 4
            raw=b"".join(block(typ,rng) for _ in range(8))
            source=C.create_string_buffer(raw,len(raw))
            parts=[]
            for part,size in [(0,2*128*bits//8),(1,2*128//32*2),(2,2*128//32*2)]:
                out=(C.c_ubyte*(size+16))(*([0xA5]*(size+16)))
                status=self.library.tc_gguf_test_affine(typ,source,len(raw),2,128,part,out,size)
                self.assertEqual(status,0,self.library.tc_gguf_test_error())
                self.assertEqual(bytes(out)[size:],bytes([0xA5]*16));parts.append(bytes(out)[:size])
            scales=struct.unpack('<8e',parts[1]);biases=struct.unpack('<8e',parts[2])
            words=struct.unpack('<'+'I'*(len(parts[0])//4),parts[0])
            values=[]
            for i in range(256):
                code=(words[i//(32//bits)]>>((i%(32//bits))*bits))&((1<<bits)-1)
                values.append(code*scales[i//32]+biases[i//32])
            reconstructed=b''.join(struct.pack('<f',value) for value in values)
            golden=self.values(self.decode(raw,typ,2,128),2,128)
            # GGML's signed-code product and MLX's unsigned affine expression
            # can give opposite signed zero; the frozen D0 contract permits
            # zero canonicalization, not tolerance on any nonzero value.
            def canonical_zero(raw):
                return b''.join(struct.pack('<I',0 if bits&0x7fffffff==0 else bits)
                               for bits in struct.unpack('<256I',raw))
            self.assertEqual(canonical_zero(reconstructed),canonical_zero(golden))
            out=(C.c_ubyte*16)()
            self.assertNotEqual(self.library.tc_gguf_test_affine(typ,source,len(raw),2,128,99,out,16),0)

    def test_slice_gather_duplicate_order_and_guard_bytes(self):
        raw = b"".join(struct.pack("<f", float(x)) for x in range(60))
        result = self.decode(raw, 0, 3, 20, row_begin=1, rows=2, column_begin=3, columns=7, column_stride=8)
        expected = b"".join(struct.pack("<f", x) for row in (1, 2) for x in range(row*20+3, row*20+10))
        self.assertEqual(self.values(result, 2, 7), expected)
        result = self.decode(raw, 0, 3, 20, gather=[2, 0, 2], column_begin=5, columns=3)
        expected = b"".join(struct.pack("<f", x) for row in (2, 0, 2) for x in range(row*20+5, row*20+8))
        self.assertEqual(self.values(result, 3, 3), expected)
        self.assertEqual(result[2][3], 1024)

    def test_decode_invalid_geometry_nonfinite_and_cancellation(self):
        raw = struct.pack("<e", 1.) + bytes(32)
        cases = [{"source_columns": 31}, {"source_rows": 2}, {"rows": 2}, {"column_begin": 30, "columns": 3},
                 {"dtype": 99}, {"column_stride": 0}, {"row_stride": 1}, {"target_size": 1},
                 {"gather": [1]}, {"cancel": True}]
        for overrides in cases:
            args = dict(raw=raw, typ=8, source_rows=1, source_columns=32)
            args.update(overrides)
            with self.subTest(overrides=overrides): self.assertNotEqual(self.decode(**args)[0], 0)
        for scale in (float("inf"), float("nan")):
            self.assertNotEqual(self.decode(struct.pack("<e", scale)+bytes(32), 8, 1, 32)[0], 0)

    def test_random_all_types_target_stride_slice_parity(self):
        rng = random.Random(20260930)
        for typ, (elements, size) in TYPES.items():
            columns = max(elements * 3, 33 if elements == 1 else elements * 3)
            raw = b"".join(block(typ, rng) for _ in range(3 * columns // elements))
            full = self.values(self.decode(raw, typ, 3, columns), 3, columns)
            for begin, count in [(0, columns), (1, columns-2), (elements-1, min(elements+2, columns-elements+1))]:
                for dtype in (0, 1, 2):
                    with self.subTest(typ=typ, begin=begin, dtype=dtype):
                        out = self.values(self.decode(raw, typ, 3, columns, column_begin=begin, columns=count,
                                                     dtype=dtype, column_stride=8), 3, count, dtype)
                        values = [struct.unpack_from("<f", full, (row*columns+col)*4)[0]
                                  for row in range(3) for col in range(begin, begin+count)]
                        if dtype == 0: expected = b"".join(struct.pack("<f", value) for value in values)
                        elif dtype == 1: expected = b"".join(struct.pack("<e", value) for value in values)
                        else:
                            bits = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
                            expected = b"".join(struct.pack("<H", (b+0x7fff+((b>>16)&1))>>16) for b in bits)
                        self.assertEqual(out, expected)


class GGMLOracleTests(GGUFExecutionTests):
    @classmethod
    def setUpClass(cls):
        location = os.environ.get("TC_GGML_ORACLE_ROOT")
        if not location: raise unittest.SkipTest("explicit pinned GGML oracle checkout not selected")
        cls.oracle_root = Path(location).resolve()
        commit = subprocess.check_output(["git", "-C", str(cls.oracle_root), "rev-parse", "HEAD"], text=True).strip()
        if commit != ORACLE_COMMIT: raise RuntimeError("GGML oracle commit mismatch")
        subprocess.run(["git", "-C", str(cls.oracle_root), "diff", "--exit-code", "HEAD", "--", "ggml"], check=True)
        super().setUpClass()
        lib = cls.path / "ggml-oracle.dylib"
        # Compile unmodified pinned scalar functions, not the entire engine.
        # This avoids unresolved quantizer/backend symbols and ensures the
        # oracle does not link or call TurboCider's implementation.
        upstream = (cls.oracle_root/"ggml/src/ggml-quants.c").read_text()
        def function_body(signature):
            start = upstream.index(signature)
            opening = upstream.index("{", start)
            depth = 1
            end = opening + 1
            while depth:
                depth += (upstream[end] == "{") - (upstream[end] == "}")
                end += 1
            return upstream[start:end] + "\n"
        names = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q4_K", "q5_K", "q6_K"]
        extracted = '#include "ggml-quants.h"\n#include "ggml-impl.h"\n'
        extracted += function_body("static inline void get_scale_min_k4(")
        extracted += "\n".join(function_body("void dequantize_row_"+name+"(") for name in names)
        reference_source = cls.path/"ggml-oracle.c"
        reference_source.write_text(extracted)
        subprocess.run(["xcrun", "clang", "-std=c11", "-O2", "-ffp-contract=off", "-dynamiclib",
                        "-I", str(cls.oracle_root/"ggml/include"), "-I", str(cls.oracle_root/"ggml/src"),
                        str(reference_source),
                        "-o", str(lib)], check=True)
        cls.oracle = C.CDLL(str(lib))

    def oracle_values(self, raw, typ):
        name = {2:"q4_0",3:"q4_1",6:"q5_0",7:"q5_1",8:"q8_0",12:"q4_K",13:"q5_K",14:"q6_K"}[typ]
        function = getattr(self.oracle, "dequantize_row_"+name)
        function.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int64]
        elements, size = TYPES[typ]
        count = len(raw)//size*elements
        output = (C.c_float*count)()
        source = C.create_string_buffer(raw, len(raw))
        function(source, output, count)
        return bytes(output)

    def test_pinned_ggml_random_blocks_bit_exact(self):
        rng = random.Random(6409)
        for typ in (2,3,6,7,8,12,13,14):
            elements, _ = TYPES[typ]
            for trial in range(32):
                raw = b"".join(block(typ, rng) for _ in range(12))
                actual = self.values(self.decode(raw, typ, 3, elements*4), 3, elements*4)
                self.assertEqual(actual, self.oracle_values(raw, typ), (typ,trial))

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GGUF_MODELS") == "1", "local model samples not selected")
    def test_real_z_image_q8_q4_bounded_samples(self):
        for folder, filename in [("z-image-runtime-gguf-q8","z_image_turbo-Q8_0.gguf"),
                                 ("z-image-runtime-gguf-q4-main","z_image_turbo-Q4_0.gguf")]:
            path = ROOT/"models"/folder/filename
            if not path.is_file(): self.fail("required local model sample missing: "+filename)
            for tensor in ["layers.0.feed_forward.w1.weight","layers.0.feed_forward.w2.weight","layers.29.attention.qkv.weight"]:
                result = subprocess.run([str(self.probe), "tensor", str(path), tensor], text=True, capture_output=True, check=True)
                fields = result.stdout.splitlines()[1].split()
                typ, rows, columns, offset, size = map(int, fields[1:])
                elements, block_bytes = TYPES[typ]
                with path.open("rb") as source:
                    source.seek(offset); raw = source.read(min(rows,3)*columns//elements*block_bytes)
                actual = self.values(self.decode(raw, typ, min(rows,3), columns), min(rows,3), columns)
                golden = self.oracle_values(raw, typ)
                self.assertEqual(actual, golden, (filename,tensor))
                values = struct.unpack("<"+"f"*(len(golden)//4), golden)
                expected = b"".join(struct.pack("<H", (bits+0x7fff+((bits>>16)&1))>>16)
                    for bits in (struct.unpack("<I",struct.pack("<f",value))[0] for value in values))
                actual_bf16 = self.values(self.decode(raw, typ, min(rows,3), columns, dtype=2), min(rows,3), columns, 2)
                self.assertEqual(actual_bf16, expected, (filename,tensor,"BF16"))


if __name__ == "__main__": unittest.main(verbosity=2)
