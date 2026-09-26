"""Run with the pinned Qwen3.5 reference environment (validation-only dependencies)."""
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:  # CPU-only TurboCider test env intentionally lacks torch.
    torch = None

if torch is not None:
    from tools.validation.qwen35_vision_parity import reference_thread_diagnostics


@unittest.skipUnless(torch is not None, "diagnostic requires pinned torch environment")
class ReferenceThreadDiagnosticsTests(unittest.TestCase):
    def test_equal_and_changed_outputs(self):
        expected = torch.ones(2, 3)
        def model(patches, grid):
            return SimpleNamespace(pooler_output=expected * torch.get_num_threads())
        original = torch.get_num_threads()
        reports = reference_thread_diagnostics(model, None, None, expected, [1, 2])
        self.assertTrue(reports[0]['exact_equal'])
        self.assertEqual(reports[0]['relative_rmse'], 0)
        self.assertFalse(reports[1]['exact_equal'])
        self.assertEqual(reports[1]['mismatch_count'], 6)
        self.assertEqual(reports[1]['relative_rmse'], 1)
        self.assertEqual(torch.get_num_threads(), original)

    def test_restores_threads_after_failure(self):
        original = torch.get_num_threads()
        def model(patches, grid):
            raise RuntimeError('fixture failure')
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            reference_thread_diagnostics(model, None, None, torch.ones(1), [1])
        self.assertEqual(torch.get_num_threads(), original)

    def test_invalid_count(self):
        original = torch.get_num_threads()
        with self.assertRaisesRegex(ValueError, 'positive'):
            reference_thread_diagnostics(None, None, None, torch.ones(1), [0])
        self.assertEqual(torch.get_num_threads(), original)


if __name__ == '__main__':
    unittest.main()
