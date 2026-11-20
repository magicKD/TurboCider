"""Build and run a dependency-free native memory fixture."""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def run_memory_test(name: str, *sources: str) -> int:
    compiler = subprocess.run(
        ["xcrun", "--find", "clang++"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    sdk = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    label = name.replace("_", "-")
    with tempfile.TemporaryDirectory(prefix=f"turbocider-{label}-") as raw:
        binary = Path(raw) / f"{label}-test"
        subprocess.run(
            [
                compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                "-isysroot", sdk,
                "-I", str(ROOT / "native/core"),
                "-I", str(ROOT / "native/runtime"),
                str(ROOT / f"tests/native/{name}_test.cpp"),
                *(str(ROOT / "native" / f"{source}.cpp") for source in sources),
                "-o", str(binary),
            ],
            check=True,
        )
        completed = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True
        )
    print(completed.stdout.strip())
    return 0
