from __future__ import annotations

import unittest

from turbocider.device import DeviceInfo, DeviceProfileCatalog


class DeviceProfileTests(unittest.TestCase):
    def test_validated_m4_max_profile_matches_without_private_identifiers(self):
        device = DeviceInfo(
            architecture="arm64",
            machine_model="Mac16,9",
            chip="Apple M4 Max",
            gpu_cores=40,
            memory_bytes=64 * 1024 ** 3,
            os_version="26.6.2",
            os_build="test",
        )
        report = DeviceProfileCatalog().match(
            ["apple-m4-max-40gpu-64gb"], device
        )
        self.assertTrue(report["matched"])
        public = report["device"]
        self.assertNotIn("serial", public)
        self.assertNotIn("uuid", public)
        self.assertEqual(len(public["fingerprint"]), 16)

    def test_different_gpu_configuration_does_not_match(self):
        device = DeviceInfo(
            architecture="arm64",
            machine_model="Mac16,9",
            chip="Apple M4 Max",
            gpu_cores=32,
            memory_bytes=64 * 1024 ** 3,
            os_version="26.6.2",
            os_build="test",
        )
        report = DeviceProfileCatalog().match(
            ["apple-m4-max-40gpu-64gb"], device
        )
        self.assertFalse(report["matched"])


if __name__ == "__main__":
    unittest.main()
