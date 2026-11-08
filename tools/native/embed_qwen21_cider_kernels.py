#!/usr/bin/env python3
"""Embed the pinned Cider kernels in the native runtime."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "native/third_party/cider/kernels"
    lines = ["#pragma once", "#include <stdexcept>", "#include <string>",
             "namespace tc::qwen21 {", "inline std::string embedded_cider_kernel(const std::string &name) {"]
    for path in sorted(root.glob("*.metal")):
        source = path.read_text()
        assert ')CIDER_KERNEL"' not in source
        lines.append('if (name == "' + path.name + '") return R"CIDER_KERNEL(' + source + ')CIDER_KERNEL";')
    lines += ['throw std::invalid_argument("Unknown embedded Cider kernel: " + name);', "}", "}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
