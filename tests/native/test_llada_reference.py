#!/usr/bin/env python3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LLaDAReferenceTests(unittest.TestCase):
    def test_native_models_do_not_read_reference_environment(self):
        for path in (ROOT / 'native/models/llada').glob('*.cpp'):
            source = path.read_text()
            self.assertNotIn('getenv(', source, str(path))
            self.assertNotIn('TURBOCIDER_LLADA_REFERENCE_', source, str(path))
            self.assertNotIn('TURBOCIDER_LLADA_EAGER_BLOCKS', source, str(path))

    def test_reference_tools_compile_and_keep_persistent_protocol(self):
        worker = ROOT / "tools/validation/llada/python_worker.py"
        benchmark = ROOT / "tools/native/benchmark_llada_reference.py"
        source = worker.read_text()
        compile(source, str(worker), "exec")
        compile(benchmark.read_text(), str(benchmark), "exec")
        self.assertIn('"type": "ready"', source)
        self.assertIn('"type": "result"', source)
        self.assertIn('for line in sys.stdin', source)
        self.assertIn('mode not in {"text", "editing"}', source)
        self.assertIn('TURBOCIDER_LLADA_PROFILE', source)
        self.assertIn('block_rows', source)
        self.assertIn('MPS synchronized forward hooks; profiling only', source)
        self.assertIn('--keep-warmup', benchmark.read_text())

    def test_quality_gate_is_explicit_and_benchmark_enforced_on_request(self):
        gate = ROOT / "tools/native/quality_gate.py"
        benchmark = ROOT / "tools/native/benchmark_llada_hybrid.py"
        compile(gate.read_text(), str(gate), "exec")
        source = benchmark.read_text()
        self.assertIn('from quality_gate import image_metrics, passes', source)
        self.assertIn('--require-quality', source)
        self.assertIn('quality_gate_passed', source)
        self.assertIn('if args.require_quality', source)

    def test_edit_is_not_exposed_by_the_formal_native_contract(self):
        self.assertFalse((ROOT / "examples/requests/llada-image-turbo-edit-256.json").exists())
        module = (ROOT / "native/models/llada_module.cpp").read_text()
        self.assertIn('require(r.operation == "image.generate"', module)
        self.assertIn('d.operations = {"image.generate"}', module)

    def test_native_module_exposes_native_generation_and_explicit_ane_candidate(self):
        registry = (ROOT / "native/models/registry.cpp").read_text()
        module = (ROOT / "native/models/llada_module.cpp").read_text()
        session = (ROOT / "native/platform/apple/llada_session.mm").read_text()
        worker = (ROOT / "tools/validation/llada/python_worker.py").read_text()
        self.assertIn("llada_module()", registry)
        self.assertIn('d.operations = {"image.generate"}', module)
        self.assertIn('if (r.execution == "gpu_ane")', module)
        self.assertIn('require(r.operation == "image.generate"', module)
        self.assertIn('d.supports_gpu_ane = true', module)
        self.assertIn("uses_parent_mlx() const override { return true; }", session)
        self.assertIn("create_llada_image_native", session)
        self.assertNotIn("LLaDAWorker", session)
        self.assertNotIn("getenv(", session)
        self.assertNotIn("posix_spawn", session)
        self.assertIn('return native().generate(request, event, cancelled);', session)
        self.assertFalse((ROOT / "tools/native/llada_worker.py").exists())
        self.assertIn("class CoreMLFFNBridge", worker)
        self.assertIn("class HybridFeedForward", worker)
        self.assertIn("class ConditioningCache", worker)
        self.assertIn("TURBOCIDER_LLADA_CONDITIONING_CACHE_DIR", worker)
        self.assertIn("stage_input(hidden_states)", worker)
        self.assertIn("noise_refiner", worker)
        self.assertIn("pipeline.transformer.layers", worker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
