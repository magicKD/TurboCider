"""Protected CPU contracts for tiling, plan evidence and process supervision."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


tiling = module("tiling_graphs", "labs/ane/tiling_graphs.py")
plans = module("compute_plan", "labs/ane/compute_plan.py")
supervisor = module("run_experiment", "scripts/run_experiment.py")
common = module("research_common", "labs/common.py")


class TilingContractTests(unittest.TestCase):
    def test_nondivisible_tail_and_zero_tile(self):
        self.assertEqual(tiling.spans(11, 4), [(0, 4), (4, 8), (8, 11)])
        self.assertEqual(tiling.spans(11, 0), [(0, 11)])
        self.assertEqual(tiling.spans(3, 8), [(0, 3)])

    def test_invalid_geometry_and_operation_budget(self):
        for args in ((True, 4), (4, True), (4, -1), (0, 4), (4, .5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tiling.spans(*args)
        for args in ((2049, 1, 1, 0, 0), (1, 4097, 1, 0, 0), (1, 128, 512, 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tiling.geometry(*args)
        self.assertEqual(tiling.geometry(1024, 4096, 12288, 2048, 4096)["projection_matmuls_before_passes"], 18)

    def test_reduction_must_finish_before_nonlinearity(self):
        x = [1., -2., 3.]
        wg, wu = [2., 1., -.5], [1., 2., 1.]
        gate = sum(a * b for a, b in zip(x, wg))
        up = sum(a * b for a, b in zip(x, wu))
        correct = common.silu(gate) * up
        partials = [(sum(x[k] * wg[k] for k in range(first, last)),
                     sum(x[k] * wu[k] for k in range(first, last))) for first, last in tiling.spans(3, 2)]
        self.assertAlmostEqual(common.silu(sum(g for g, _ in partials)) * sum(u for _, u in partials), correct)
        wrong = sum(common.silu(g) * u for g, u in partials)
        self.assertGreater(abs(wrong - correct), .01)


class PlanEvidenceTests(unittest.TestCase):
    def plan(self, weight=.25, unknown=False):
        inner = NS(operator_name="matmul", outputs=[NS(name="inner")], blocks=[])
        outer = NS(operator_name="cond", outputs=[NS(name="outer")], blocks=[NS(operations=[inner])])
        return NS(model_structure=NS(program=NS(functions={"main": NS(block=NS(operations=[outer]))})),
                  get_compute_device_usage_for_mlprogram_operation=lambda op: None if unknown else NS(
                      preferred_compute_device=NS(), supported_compute_devices=[NS()]),
                  get_estimated_cost_for_mlprogram_operation=lambda op: None if unknown else NS(weight=weight))

    def test_nested_plan_is_predicted_not_timing(self):
        report = plans.summarize(self.plan())
        self.assertEqual(report["operation_count"], {"value": 2, "source_kind": "derived"})
        self.assertEqual([item["locator"] for item in report["operations"]], ["main/0", "main/0/block-0/0"])
        self.assertTrue(all(item["source_kind"] == "predicted" for item in report["operations"]))
        self.assertIsNone(report["physical_operation_timing"])
        self.assertEqual(report["physical_ane_placement"], "unverified")

    def test_missing_cost_and_device_are_null_not_zero(self):
        report = plans.summarize(self.plan(unknown=True))
        for operation in report["operations"]:
            self.assertIsNone(operation["estimated_cost_weight"])
            self.assertIsNone(operation["preferred_device"])

    def test_invalid_cost_is_not_relabelled_measured(self):
        for value in (-1, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plans.summarize(self.plan(weight=value))


class SupervisorTests(unittest.TestCase):
    def test_success_and_failed_child_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            for code in (0, 3):
                output = Path(directory) / str(code)
                result = supervisor.run([sys.executable, "-c", f"print('owned'); raise SystemExit({code})"], output, 5)
                self.assertEqual(result["status"], "succeeded" if code == 0 else "failed")
                self.assertTrue(result["owned_process_exit_confirmed"])
                self.assertEqual((output / "stdout.log").read_text().strip(), "owned")
                self.assertFalse(result["production_qualified"])
                self.assertEqual(json.loads((output / "receipt.json").read_text()), result)

    def test_timeout_stops_only_owned_child_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = supervisor.run([sys.executable, "-c", "import time; time.sleep(5)"], Path(directory) / "timeout", 1)
            self.assertEqual(result["status"], "timed_out")
            self.assertTrue(result["owned_process_exit_confirmed"])

    def test_existing_output_and_bad_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                supervisor.run([sys.executable, "-V"], Path(directory), 5)
            for budget in (True, 0, 901, .5):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    supervisor.run([sys.executable, "-V"], Path(directory) / "new", budget)


if __name__ == "__main__":
    unittest.main()
