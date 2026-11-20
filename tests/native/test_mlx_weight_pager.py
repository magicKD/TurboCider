#!/usr/bin/env python3
"""Build and run the sparse multi-artifact MLX weight pager fixture."""

from __future__ import annotations

from mlx_fixture_runner import run_mlx_directory_fixture


def main() -> None:
    run_mlx_directory_fixture(
        "mlx_weight_pager", "MLX weight pager",
        native=None,
        sources=(
            "native/runtime/streaming/mlx_weight_pager.cpp",
            "native/runtime/streaming/canonical_encoding.cpp",
            "native/runtime/streaming/source_lease.cpp",
            "native/runtime/memory_manifest.cpp",
            "native/runtime/memory_policy.cpp",
            "native/backends/ane_memory.cpp",
            "native/core/common.cpp",
        ),
    )


if __name__ == "__main__":
    main()
