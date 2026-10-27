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

    def test_runtime_ane_runs_and_legacy_output_directories_stay_local(self):
        paths = ["output/run/request.json", "Output/run/image.png",
                 "outputs/qwen-runtime/trial/report.json",
                 "outputs/qwen-runtime/native-source.tar.gz",
                 "results/qwen21/image.png", "results/z-image-runtime/spec.json",
                 "results/new-model/trial-memory.jsonl",
                 "build/runtime-ane/libturbocider.dylib",
                 "models/coreml/program.mlmodelc/metadata.json",
                 "tools/profiling/__pycache__/runtime_image_collect.cpython-311.pyc"]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_runtime_ane_source_and_portable_configuration_remain_trackable(self):
        paths = ["native/models/qwen21/runtime_private_qkv.cpp",
                 "native/backends/ane_fixed_program.hpp",
                 "tools/profiling/README.md", "tools/profiling/qwen_base/collect.py",
                 "tools/profiling/qwen_base/observer.mm",
                 "tools/profiling/qwen_base/CMakeLists.txt",
                 "tools/native/build_ane_w8_functions_probe.sh",
                 "tests/native/test_qwen_runtime_cache_audit.py",
                 "tests/fixtures/ane/descriptor.json", "tests/fixtures/ane/weights.bin",
                 "examples/requests/qwen-runtime.json",
                 "docs/benchmark/CURRENT_FASTEST_CONFIGS.md",
                 "docs/design/validation/portable-summary.json"]
        self.assertEqual(self.ignored(paths), set())

    def test_raw_bf16_archives_stay_local_at_any_depth(self):
        paths = ["ffn-bf16.tcbf16z", "techniques/integrations/ffn-bf16.tcbf16z",
                 "techniques/review/raw-pair.tcbf16z", "ffn-input.safetensors",
                 "techniques/review/ffn-reference.safetensors"]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_knowledge_notes_and_small_examples_remain_trackable(self):
        paths = ["techniques/README.md", "techniques/MAINTENANCE.md",
                 "techniques/sources/repos/mlx-affine-dispatch.md",
                 "techniques/metal/sources.md", "techniques/queries/README.md",
                 "techniques/templates/technique-card.md", "techniques/scripts/knowledge.py",
                 "techniques/examples/split_ffn_reference.py",
                 "techniques/examples/reference/contracts.py",
                 "techniques/examples/metal/kernels.metal",
                 "examples/requests/new-request.json", "tests/fixtures/database.json",
                 "profiles/apple-device.example.json", "assets/example.png", ".env.example"]
        self.assertEqual(self.ignored(paths), set())

    def test_all_knowledge_json_and_experiment_packages_stay_local(self):
        paths = ["techniques/new.json", "techniques/candidates/projects.json",
                 "techniques/labs/catalog.json", "techniques/references/sources.json",
                 "techniques/data/vocabulary.json", "techniques/data/publication-transforms.json",
                 "techniques/metal/evidence/results/curated-evidence.json",
                 "techniques/examples/paired-runs.json", "techniques/case-studies/data.jsonl",
                 "techniques/arrays/saved.npy", "techniques/arrays/saved.npz",
                 "techniques/review/history.md", "techniques/metal/archive/receipt.txt",
                 "techniques/labs/new_probe.py", "techniques/integrations/client.mm",
                 "techniques/tests/test_saved_evidence.py"]
        self.assertEqual(self.ignored(paths), set(paths))


if __name__ == "__main__":
    unittest.main()
