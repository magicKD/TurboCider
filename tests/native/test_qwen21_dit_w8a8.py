#!/usr/bin/env python3
"""Build and run the small, real-MLX staged Qwen21 attention fixture."""

import sys
import unittest

from mlx_fixture_runner import native_test_directory, run_mlx_fixture


NATIVE = native_test_directory()


def main(build_only: bool = False) -> None:
    run_mlx_fixture(
        "qwen21_dit_w8a8", "Qwen21 DiT W8A8", native=NATIVE,
        build_only=build_only, runtime_skip=True,
    )


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX")
class DiTW8A8Tests(unittest.TestCase):
    def test_true_activation_quantization_and_persistent_cache(self):
        main()


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-only"]:
        main(build_only=True)
    else:
        unittest.main()
