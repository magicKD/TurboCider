#!/usr/bin/env python3
"""Build and run the fd-backed Qwen21 text layer streaming fixture."""

from __future__ import annotations

import unittest

from mlx_fixture_runner import native_test_directory, run_mlx_directory_fixture


NATIVE = native_test_directory()


def main() -> None:
    run_mlx_directory_fixture(
        "qwen21_text_streaming", "Qwen21 text streaming",
        native=NATIVE, create_fixture=True,
    )


class TextStreamingTests(unittest.TestCase):
    def test_exact_bf16_release_and_retry(self):
        main()


if __name__ == "__main__":
    unittest.main()
