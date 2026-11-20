#!/usr/bin/env python3
"""Build and run the small, real-MLX staged Qwen21 attention fixture."""

import sys
import unittest

from mlx_fixture_runner import native_test_directory, run_mlx_fixture


NATIVE = native_test_directory()


def main(build_only: bool = False) -> None:
    run_mlx_fixture(
        "qwen21_staged_attention", "Qwen21 staged attention", native=NATIVE,
        build_only=build_only,
    )


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX")
class StagedAttentionTests(unittest.TestCase):
    def test_packed_storage_lifetime_and_bf16_projection(self):
        main()


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-only"]:
        main(build_only=True)
    else:
        unittest.main()
