"""Extract pre-refactor encoder bodies and compare with native shared Qwen3.

Only development tooling: no Python or historical source enters the package.
Requires a built native library, MLX C++ headers and a Qwen3-4B checkpoint.
Timings exclude weight loading, tokenization and downstream image generation.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "bd5953a44d540e0ad1fdae8c73cc7f91086d1f2a"


def baseline_source():
    def show(path):
        return subprocess.check_output(
            ["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True)

    flux = show("native/models/flux2/flux_text.cpp")
    start = flux.index("    int n = int(t.ids.size());")
    end = flux.index("\n} // namespace tc")
    # Loading and optional LoRA setup are deliberately outside the timed
    # region; all original tensor operations below are extracted unchanged.
    flux = ("Tensor flux(const Tokens &t, const Weights &w, const Event &event, "
            "std::atomic<bool> &cancelled) {\n" + flux[start:end])
    z_image = show("native/models/z_image/z_image.cpp")
    start = z_image.index("Tensor linear_compat(")
    end = z_image.index("Tensor z_group_norm(")
    return ("namespace tc::baseline {\n"
            "constexpr int kTextHeads=32, kTextKVHeads=8, kHeadDim=128;\n"
            + flux + "\n" + z_image[start:end] + "\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--mlx-root", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--valid", type=int, default=47)
    args = parser.parse_args()
    mlx = args.mlx_root.resolve()
    library = ROOT / "build/native"
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    load_commands = subprocess.check_output(
        ["otool", "-l", str(library / "libturbocider.dylib")], text=True)
    minimum = next(line.split()[1] for line in load_commands.splitlines()
                   if line.strip().startswith("minos "))
    revision = subprocess.check_output(
        ["git", "rev-parse", BASELINE], cwd=ROOT, text=True).strip()
    print(f"baseline={revision} tokens={args.tokens} valid={args.valid}", flush=True)
    with tempfile.TemporaryDirectory(prefix="tc-qwen3-refactor-") as directory:
        work = Path(directory)
        (work / "baseline.hpp").write_text(baseline_source())
        executable = work / "probe"
        subprocess.run([
            "xcrun", "clang++", "-std=c++20", "-O2", "-isysroot", sdk,
            f"-mmacosx-version-min={minimum}", "-I" + str(work),
            "-I" + str(ROOT / "native/core"),
            "-isystem", str(mlx / "include"),
            str(ROOT / "tools/native/qwen3_refactor_probe.cpp"),
            "-L" + str(library), "-lturbocider", "-L" + str(mlx / "lib"), "-lmlx",
            "-Wl,-rpath," + str(library), "-Wl,-rpath," + str(mlx / "lib"),
            "-o", str(executable)], check=True)
        subprocess.run([str(executable), str(args.weights.resolve()),
                        str(args.tokens), str(args.valid)], check=True)


if __name__ == "__main__":
    main()
