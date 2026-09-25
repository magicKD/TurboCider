"""Small Core ML integration check for the exporter's A8/W8 coverage gate."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "z_image_smoothquant", ROOT / "tools/coreml/z_image_smoothquant.py")
SQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SQ)


class CoreMLW8A8Tests(unittest.TestCase):
    def test_shared_input_region_restore_requires_quantized_data_and_fixed_scale(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.optimize.coreml import (
            OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights,
        )

        def make_model(bypass=False, dynamic_scale=False):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 64), dtype=types.fp16)],
                        opset_version=ct.target.macOS15)
            def branch(x):
                ratio = np.full((1, 1, 1, 64), 2., dtype=np.float16)
                inverse = np.full((1, 1, 1, 64), .5, dtype=np.float16)
                prepared = mb.mul(x=x, y=ratio)
                reconstructed = mb.dequantize(input=mb.quantize(
                    input=prepared, scale=np.float16(.04), output_dtype="int8"),
                    scale=np.float16(.04))
                if dynamic_scale:
                    inverse = mb.add(x=mb.reduce_mean(x=x, axes=[1], keep_dims=True),
                                     y=inverse)
                projected_input = mb.mul(
                    x=prepared if bypass else reconstructed, y=inverse,
                    name="a8_input_region_restore")
                projected = mb.conv(x=projected_input,
                                    weight=np.ones((32, 16, 1, 1), dtype=np.float16),
                                    name="projected")
                gate, up = mb.split(x=projected, num_splits=2, axis=1)
                hidden = mb.mul(x=mb.silu(x=gate), y=up)
                hidden = mb.dequantize(input=mb.quantize(
                    input=hidden, scale=np.float16(.04), output_dtype="int8"),
                    scale=np.float16(.04))
                return mb.conv(x=hidden,
                               weight=np.ones((16, 16, 1, 1), dtype=np.float16), name="y")

            converted = ct.convert(branch, convert_to="mlprogram",
                                   minimum_deployment_target=ct.target.macOS15,
                                   compute_precision=ct.precision.FLOAT16,
                                   skip_model_load=True)
            w8 = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                         granularity="per_channel", block_size=32,
                                         weight_threshold=0)
            return linear_quantize_weights(converted, OptimizationConfig(
                global_config=None, op_name_configs={"projected": w8, "y": w8}))

        valid = make_model().get_spec()
        self.assertEqual(SQ.verify_w8a8(valid, ("projected", "y"),
                                        allow_input_region_scale=True),
                         ["projected", "y"])
        with self.assertRaisesRegex(ValueError, "activation and weight INT8"):
            SQ.verify_w8a8(valid, ("projected", "y"))
        for bypass, dynamic in ((True, False), (False, True)):
            with self.subTest(bypass=bypass, dynamic=dynamic), \
                    self.assertRaisesRegex(ValueError, "activation and weight INT8"):
                SQ.verify_w8a8(make_model(bypass, dynamic).get_spec(),
                                ("projected", "y"), allow_input_region_scale=True)

    def test_both_projections_have_w8_and_dynamic_a8(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.optimize.coreml import (
            OptimizationConfig, OpLinearQuantizerConfig,
            linear_quantize_activations, linear_quantize_weights,
        )

        rng = np.random.default_rng(42)
        first = rng.normal(0, 0.2, (32, 16, 1, 1)).astype(np.float16)
        last = rng.normal(0, 0.2, (16, 16, 1, 1)).astype(np.float16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 64), dtype=types.fp16)],
                    opset_version=ct.target.macOS15)
        def branch(x):
            quantized = mb.quantize(input=x, scale=np.float16(0.02), output_dtype="int8")
            x = mb.dequantize(input=quantized, scale=np.float16(0.02))
            gate, up = mb.split(x=mb.conv(x=x, weight=first, name="projected"),
                                num_splits=2, axis=1)
            hidden = mb.mul(x=mb.silu(x=gate), y=up)
            quantized = mb.quantize(input=hidden, scale=np.float16(0.04), output_dtype="int8")
            hidden = mb.dequantize(input=quantized, scale=np.float16(0.04))
            return mb.conv(x=hidden, weight=last, name="y")

        model = ct.convert(branch, convert_to="mlprogram",
                           minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        w8 = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                     granularity="per_channel", block_size=32, weight_threshold=0)
        config = OptimizationConfig(global_config=None,
                                    op_name_configs={name: w8 for name in ("projected", "y")})
        model = linear_quantize_weights(model, config)
        self.assertEqual(SQ.verify_w8a8(model.get_spec(), ("projected", "y")),
                         ["projected", "y"])

    def test_region_split_requires_both_quantized_activation_paths(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.optimize.coreml import (
            OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights,
        )

        def make_model(quantize_second):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 64), dtype=types.fp16)],
                        opset_version=ct.target.macOS15)
            def branch(x):
                pieces = []
                for index, (begin, end, scale) in enumerate(((0, 32, .02), (32, 64, .2))):
                    piece = mb.slice_by_index(x=x, begin=[0, 0, 0, begin],
                                              end=[1, 16, 1, end])
                    if index == 0 or quantize_second:
                        q = mb.quantize(input=piece, scale=np.float16(scale),
                                        output_dtype="int8")
                        piece = mb.dequantize(input=q, scale=np.float16(scale))
                    pieces.append(piece)
                first = mb.concat(values=pieces, axis=3)
                projected = mb.conv(x=first, weight=np.ones((32, 16, 1, 1),
                                                               dtype=np.float16), name="projected")
                gate, up = mb.split(x=projected, num_splits=2, axis=1)
                last = mb.mul(x=mb.silu(x=gate), y=up)
                second_pieces = []
                for begin, end in ((0, 32), (32, 64)):
                    piece = mb.slice_by_index(x=last, begin=[0, 0, 0, begin],
                                              end=[1, 16, 1, end])
                    q = mb.quantize(input=piece, scale=np.float16(.04),
                                    output_dtype="int8")
                    second_pieces.append(mb.dequantize(input=q, scale=np.float16(.04)))
                return mb.conv(x=mb.concat(values=second_pieces, axis=3),
                               weight=np.ones((16, 16, 1, 1), dtype=np.float16), name="y")

            converted = ct.convert(branch, convert_to="mlprogram",
                                   minimum_deployment_target=ct.target.macOS15,
                                   compute_precision=ct.precision.FLOAT16, skip_model_load=True)
            w8 = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                         granularity="per_channel", block_size=32,
                                         weight_threshold=0)
            return linear_quantize_weights(converted, OptimizationConfig(
                global_config=None, op_name_configs={"projected": w8, "y": w8}))

        self.assertEqual(SQ.verify_w8a8(make_model(True).get_spec(), ("projected", "y")),
                         ["projected", "y"])
        with self.assertRaisesRegex(ValueError, "activation and weight INT8"):
            SQ.verify_w8a8(make_model(False).get_spec(), ("projected", "y"))

    def test_nested_hidden_channel_groups_require_every_a8_path(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.optimize.coreml import (
            OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights,
        )

        def make_model(quantize_second_group):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 64), dtype=types.fp16)],
                        opset_version=ct.target.macOS15)
            def branch(x):
                x = mb.dequantize(input=mb.quantize(input=x, scale=np.float16(.02),
                                                   output_dtype="int8"), scale=np.float16(.02))
                projected = mb.conv(x=x, weight=np.ones((32, 16, 1, 1), np.float16),
                                    name="projected")
                gate, up = mb.split(x=projected, num_splits=2, axis=1)
                hidden = mb.mul(x=mb.silu(x=gate), y=up)
                image = mb.slice_by_index(x=hidden, begin=[0, 0, 0, 0], end=[1, 16, 1, 32])
                chunks = []
                for index in range(2):
                    part = mb.slice_by_index(x=image, begin=[0, index * 8, 0, 0],
                                             end=[1, (index + 1) * 8, 1, 32])
                    if index == 0 or quantize_second_group:
                        part = mb.dequantize(input=mb.quantize(
                            input=part, scale=np.float16(.04 * (index + 1)),
                            output_dtype="int8"), scale=np.float16(.04 * (index + 1)))
                    chunks.append(part)
                image = mb.concat(values=chunks, axis=1)
                caption = mb.slice_by_index(x=hidden, begin=[0, 0, 0, 32],
                                            end=[1, 16, 1, 64])
                caption = mb.dequantize(input=mb.quantize(
                    input=caption, scale=np.float16(.1), output_dtype="int8"),
                    scale=np.float16(.1))
                hidden = mb.concat(values=[image, caption], axis=3)
                return mb.conv(x=hidden, weight=np.ones((16, 16, 1, 1), np.float16), name="y")

            converted = ct.convert(branch, convert_to="mlprogram",
                                   minimum_deployment_target=ct.target.macOS15,
                                   compute_precision=ct.precision.FLOAT16, skip_model_load=True)
            w8 = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                         granularity="per_channel", block_size=32,
                                         weight_threshold=0)
            return linear_quantize_weights(converted, OptimizationConfig(
                global_config=None, op_name_configs={"projected": w8, "y": w8}))

        self.assertEqual(SQ.verify_w8a8(make_model(True).get_spec(), ("projected", "y")),
                         ["projected", "y"])
        with self.assertRaisesRegex(ValueError, "activation and weight INT8"):
            SQ.verify_w8a8(make_model(False).get_spec(), ("projected", "y"))

    def test_adaptive_hidden_scale_select_requires_all_a8_branches(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.optimize.coreml import (
            OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights,
        )

        def make_model(quantize_small):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 64), dtype=types.fp16)],
                        opset_version=ct.target.macOS15)
            def branch(x):
                x = mb.dequantize(input=mb.quantize(input=x, scale=np.float16(.02),
                                                   output_dtype="int8"), scale=np.float16(.02))
                projected = mb.conv(x=x, weight=np.ones((32, 16, 1, 1), np.float16),
                                    name="projected")
                gate, up = mb.split(x=projected, num_splits=2, axis=1)
                hidden = mb.mul(x=mb.silu(x=gate), y=up)
                image = mb.slice_by_index(x=hidden, begin=[0, 0, 0, 0], end=[1, 16, 1, 32])
                peak = mb.reduce_max(x=mb.abs(x=image), axes=[1], keep_dims=True)
                large = mb.dequantize(input=mb.quantize(
                    input=image, scale=np.float16(.16), output_dtype="int8"),
                    scale=np.float16(.16))
                small = (mb.dequantize(input=mb.quantize(
                    input=image, scale=np.float16(.04), output_dtype="int8"),
                    scale=np.float16(.04)) if quantize_small else image)
                image = mb.select(cond=mb.less_equal(x=peak, y=np.float16(5.08)),
                                  a=small, b=large)
                caption = mb.slice_by_index(x=hidden, begin=[0, 0, 0, 32],
                                            end=[1, 16, 1, 64])
                caption = mb.dequantize(input=mb.quantize(
                    input=caption, scale=np.float16(.2), output_dtype="int8"),
                    scale=np.float16(.2))
                hidden = mb.concat(values=[image, caption], axis=3)
                return mb.conv(x=hidden, weight=np.ones((16, 16, 1, 1), np.float16),
                               name="y")

            converted = ct.convert(branch, convert_to="mlprogram",
                                   minimum_deployment_target=ct.target.macOS15,
                                   compute_precision=ct.precision.FLOAT16, skip_model_load=True)
            w8 = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                         granularity="per_channel", block_size=32,
                                         weight_threshold=0)
            return linear_quantize_weights(converted, OptimizationConfig(
                global_config=None, op_name_configs={"projected": w8, "y": w8}))

        self.assertEqual(SQ.verify_w8a8(make_model(True).get_spec(), ("projected", "y")),
                         ["projected", "y"])
        with self.assertRaisesRegex(ValueError, "activation and weight INT8"):
            SQ.verify_w8a8(make_model(False).get_spec(), ("projected", "y"))

    def test_shared_adaptive_qdq_requires_quantized_data_and_constant_scale_bank(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.optimize.coreml import (
            OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights,
        )

        def make_model(bypass_data=False, dynamic_scale=False,
                       resmooth=False, bypass_resmooth=False,
                       caption_adaptive=False, bypass_caption=False):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 64), dtype=types.fp16)],
                        opset_version=ct.target.macOS15)
            def branch(x):
                x = mb.dequantize(input=mb.quantize(input=x, scale=np.float16(.02),
                                                   output_dtype="int8"), scale=np.float16(.02))
                projected = mb.conv(x=x, weight=np.ones((32, 16, 1, 1), np.float16),
                                    name="projected")
                gate, up = mb.split(x=projected, num_splits=2, axis=1)
                hidden = mb.mul(x=mb.silu(x=gate), y=up)
                image = mb.slice_by_index(x=hidden, begin=[0, 0, 0, 0], end=[1, 16, 1, 32])
                peak = mb.reduce_max(x=mb.abs(x=image), axes=[1], keep_dims=True)
                cond = mb.less_equal(x=peak, y=np.float16(5.08))
                const_small = np.full((1, 1, 1, 32), .5, dtype=np.float16)
                const_big = np.ones((1, 1, 1, 32), dtype=np.float16)
                ratio = mb.select(cond=cond, a=const_big, b=const_small)
                prepared = mb.mul(x=image, y=ratio)
                data = (prepared if bypass_data else mb.dequantize(
                    input=mb.quantize(input=prepared, scale=np.float16(.04),
                                      output_dtype="int8"), scale=np.float16(.04)))
                inverse = mb.select(cond=cond, a=image if dynamic_scale else const_big,
                                    b=const_small)
                restored = mb.mul(x=data, y=inverse, name="a8_hidden_adaptive_restore")
                if resmooth:
                    restored = mb.mul(
                        x=image if bypass_resmooth else restored,
                        y=np.full((1, 16, 1, 1), .75, dtype=np.float16),
                        name="a8_hidden_image_unresmooth")
                caption = mb.slice_by_index(x=hidden, begin=[0, 0, 0, 32],
                                            end=[1, 16, 1, 64])
                if caption_adaptive:
                    caption_peak = mb.reduce_max(x=mb.abs(x=caption), axes=[1],
                                                 keep_dims=True)
                    caption_cond = mb.less_equal(x=caption_peak, y=np.float16(5.08))
                    caption_ratio = mb.select(
                        cond=caption_cond,
                        a=np.ones((1, 1, 1, 32), dtype=np.float16),
                        b=np.full((1, 1, 1, 32), .5, dtype=np.float16))
                    caption_data = mb.mul(x=caption, y=caption_ratio)
                    if not bypass_caption:
                        caption_data = mb.dequantize(input=mb.quantize(
                            input=caption_data, scale=np.float16(.04),
                            output_dtype="int8"), scale=np.float16(.04))
                    caption = mb.mul(x=caption_data, y=caption_ratio,
                                     name="a8_hidden_caption_restore")
                else:
                    caption = mb.dequantize(input=mb.quantize(
                        input=caption, scale=np.float16(.2), output_dtype="int8"),
                        scale=np.float16(.2))
                combined = mb.concat(values=[restored, caption], axis=3)
                return mb.conv(x=combined, weight=np.ones((16, 16, 1, 1), np.float16),
                               name="y")

            model = ct.convert(branch, convert_to="mlprogram",
                               minimum_deployment_target=ct.target.macOS15,
                               compute_precision=ct.precision.FLOAT16, skip_model_load=True)
            w8 = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                         granularity="per_channel", block_size=32,
                                         weight_threshold=0)
            return linear_quantize_weights(model, OptimizationConfig(
                global_config=None, op_name_configs={"projected": w8, "y": w8}))

        self.assertEqual(SQ.verify_w8a8(make_model().get_spec(), ("projected", "y"),
                                         allow_adaptive_scale=True), ["projected", "y"])
        self.assertEqual(SQ.verify_w8a8(make_model(resmooth=True).get_spec(),
                                         ("projected", "y"), allow_adaptive_scale=True),
                         ["projected", "y"])
        self.assertEqual(SQ.verify_w8a8(make_model(caption_adaptive=True).get_spec(),
                                         ("projected", "y"), allow_adaptive_scale=True),
                         ["projected", "y"])
        for kwargs in ({"bypass_data": True}, {"dynamic_scale": True},
                       {"resmooth": True, "bypass_resmooth": True},
                       {"caption_adaptive": True, "bypass_caption": True}):
            with self.assertRaisesRegex(ValueError, "activation and weight INT8"):
                SQ.verify_w8a8(make_model(**kwargs).get_spec(), ("projected", "y"),
                                allow_adaptive_scale=True)


if __name__ == "__main__":
    unittest.main()
