"""Keep local build/cache artifacts ignored without hiding portable examples."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless((ROOT / ".git").exists(), "requires a Git checkout")
class GitignoreTests(unittest.TestCase):
    def ignored(self, paths):
        child = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"],
                               input="\n".join(paths) + "\n", cwd=ROOT,
                               capture_output=True, text=True, check=False)
        self.assertIn(child.returncode, (0, 1), child.stderr)
        return set(child.stdout.splitlines())

    def test_nested_build_and_research_caches_stay_local(self):
        paths = ["techniques/integrations/.build/release/MFAReplay",
                 "techniques/integrations/.swiftpm/configuration/workspace-state.json",
                 "apps/macos/DerivedData/Build/temp.o", "apps/macos/tests.xcresult/Info.plist",
                 "apps/macos/release.xcarchive/Info.plist", "techniques/labs/.cache/compiled.bin",
                 ".aneforge_cache/weights.bin", ".aneforge-cache/tuning.json",
                 "experimental/ane/.aneforge_cache/model/weights.bin",
                 "tools/ane/.aneforge-cache/model/program.mil",
                 "techniques/integrations/.aneforge_cache/weights.bin",
                 "techniques/integrations/.aneforge-cache/program/weights.bin",
                 "techniques/labs/ane/.aneforge-cache/tuning.json",
                 "techniques/integrations/CMakeFiles/cmake.check_cache",
                 "techniques/integrations/CMakeCache.txt", "techniques/integrations/cmake_install.cmake",
                 "techniques/integrations/compile_commands.json"]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_existing_large_and_private_artifact_rules(self):
        paths = ["outputs/research/receipt.json", "models/checkpoint.safetensors",
                 ".deps/reference/README.md", ".venv/bin/python", ".env.local",
                 "techniques/review/raw/session.jsonl", "techniques/review/local/machine.json",
                 "techniques/labs/build/program", "techniques/labs/private/excerpt.json"]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_raw_profiling_captures_stay_local(self):
        paths = ["owned-tiny.gputrace", "capture.trace",
                 "techniques/labs/metal/owned-tiny.gputrace/metadata.json",
                 "techniques/labs/metal/capture.trace/instrument_data",
                 "techniques/labs/metal/owned-tiny.gputrace.zip",
                 "techniques/labs/metal/capture.trace.zip"]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_raw_bf16_archives_stay_local_at_any_depth(self):
        paths = ["ffn-bf16.tcbf16z", "techniques/integrations/ffn-bf16.tcbf16z",
                 "techniques/review/raw-pair.tcbf16z"]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_curated_sources_and_examples_remain_trackable(self):
        paths = ["techniques/candidates/projects.json", "techniques/labs/catalog.json",
                 "techniques/references/sources.json", "techniques/review/new-validation.json",
                 "techniques/examples/paired-runs.json", "techniques/integrations/mfa_gemm_bench.swift",
                 "examples/requests/new-request.json", "tests/fixtures/database.json",
                 "profiles/apple-device.example.json", "assets/example.png", ".env.example",
                 "techniques/sources/repos/mlx-affine-dispatch.md",
                 "techniques/assets/source-excerpts/ggml-dequant-64e9bce/dequantizers.metal",
                 "techniques/labs/metal/mlx_capture_probe.py",
                 "techniques/examples/profiling-summary.json", "tests/fixtures/tiny.npy",
                 "techniques/review/gpu-clock-evidence-20261016.json",
                 "techniques/review/qwen-attention-evidence-20261016.json",
                 "techniques/review/ane-pmu-evidence-20261016.json",
                 "techniques/review/aneforge-evidence-20261016.json",
                 "techniques/review/qwen-private-ffn-evidence-20261016.json",
                 "techniques/review/qwen-block-context-evidence-20261016.json",
                 "techniques/integrations/qwen_block_shared.mm",
                 "techniques/labs/ffn_cpu_witness.hpp",
                 "techniques/tests/test_block_context.py",
                 "techniques/labs/ane/pmu_probe.mm",
                 "techniques/labs/ane/qwen_witness.py",
                 "techniques/sources/official/metal-counters.md"]
        self.assertEqual(self.ignored(paths), set())


if __name__ == "__main__":
    unittest.main()
