#!/usr/bin/env python3
"""Build and run the small, real-MLX staged Qwen21 suffix fixture."""

import sys
import unittest

from mlx_fixture_runner import run_mlx_fixture


def main() -> None:
    run_mlx_fixture("qwen21_staged_quantization", "Qwen21 staged quantization")


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX")
class StagedQuantizationTests(unittest.TestCase):
    def test_packed_storage_lifetime_forward_and_bf16_join(self):
        main()


if __name__ == "__main__":
    unittest.main()
