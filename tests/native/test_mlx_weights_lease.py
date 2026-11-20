#!/usr/bin/env python3
"""Build and run the fd-backed MLX Weights lease fixture."""

from __future__ import annotations

from mlx_fixture_runner import native_test_directory, run_mlx_directory_fixture


NATIVE = native_test_directory()


def main() -> None:
    run_mlx_directory_fixture(
        "mlx_weights_lease", "MLX lease weights",
        native=NATIVE,
    )


if __name__ == "__main__":
    main()
