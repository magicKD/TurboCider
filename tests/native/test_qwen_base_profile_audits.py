"""Portable negative controls for sequence/view/observer artifact auditing."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/profiling/qwen_base"))
import sequence_audit
import vpipe_audit
import vpipe_view
from test_qwen_base_profiling import fixture


class ProfilingAuditTests(unittest.TestCase):
    def test_contiguous_gate_up_views_partition_the_original_bytes(self):
        header = {f"transformer_blocks.{i}.img_mlp.gate_up.weight":
                  dict(dtype="BF16", shape=[24576, 4096], data_offsets=[i * 24576 * 8192, (i + 1) * 24576 * 8192]) for i in range(32)}
        gate, up, index = vpipe_view.split_header(header)
        self.assertEqual((len(gate), len(up), len(index)), (32, 32, 64))
        for i in range(32):
            p = f"transformer_blocks.{i}.img_mlp."
            self.assertEqual(gate[p + "gate_layer.weight"]["data_offsets"][1], up[p + "proj.weight"]["data_offsets"][0])
            self.assertEqual(gate[p + "gate_layer.weight"]["shape"], [12288, 4096])
            self.assertNotIn(p + "gate_up.weight", index)
        header["transformer_blocks.0.img_mlp.gate_up.weight"]["dtype"] = "F16"
        with self.assertRaises(ValueError):
            vpipe_view.split_header(header)

    def test_sequence_geometry_predicts_chunks_and_requires_restore(self):
        operations = ["image.generate", "image.edit", "image.edit", "image.generate"]
        counts = [0, 1, 2, 0]
        requests = [dict(operation=op, seed=73, inputs=[{}] * count, output=f"{i}.png") for i, (op, count) in enumerate(zip(operations, counts))]
        metrics = [dict(operation=op, reference_tokens=count * 4096, text_tokens=32,
                        qwen_ffn_phases=dict(prefill=dict(actual_rows=1056 + count * 4096, steps_this_request=1,
                            completed_channel_blocks_this_request=32, runtime_calls_this_request=64 + count * 128),
                            decode=dict(actual_rows=1024, steps_this_request=39, completed_channel_blocks_this_request=1248,
                                        runtime_calls_this_request=1248))) for op, count in zip(operations, counts)]
        rows = sequence_audit.geometry(requests, metrics)
        self.assertEqual([r["predicted_runtime_calls"] for r in rows], [1312, 1440, 1568, 1312])
        for bad in (lambda r, m: r[3].update(seed=74), lambda r, m: m[2].update(reference_tokens=4096),
                    lambda r, m: r[2].update(inputs=[{}]), lambda r, m: m[1]["qwen_ffn_phases"]["prefill"].update(actual_rows=5120)):
            r, m = copy.deepcopy(requests), copy.deepcopy(metrics)
            bad(r, m)
            with self.assertRaises(ValueError):
                sequence_audit.geometry(r, m)

    def test_two_graph_observer_requires_actual_stable_hot_models(self):
        value, _ = fixture()
        # Reuse valid clock/calibration records, but give each measured
        # prefill both graph identities and each decode no ANE call.
        calibration = [copy.deepcopy(value["events"][0])]
        rows = calibration
        samples = []
        for i in range(4):
            cb = copy.deepcopy(value["events"][-1])
            cb["request"] = i
            rows.append(cb)
            phase = "decode" if i % 2 else "prefill"
            samples.append(dict(index=i, phase=phase, ane_predict_records=0 if i % 2 else 2))
            if not i % 2:
                for model in (1, 2):
                    event = copy.deepcopy(value["events"][3])
                    event.update(request=i, model=model)
                    rows.append(event)
        from collections import Counter
        value.update(events=rows, counts=dict(Counter(r["name"] for r in rows)))
        self.assertEqual(vpipe_audit.observer_summary(value, samples)["hot_runtime_model_ids"], [1, 2])
        next(r for r in value["events"] if r["name"] == "ane.evaluate_api" and r["request"] == 2 and r["model"] == 2)["model"] = 1
        with self.assertRaises(ValueError):
            vpipe_audit.observer_summary(value, samples)

    def test_raw_observer_checks_are_shared_without_weakening_base_single_graph(self):
        value, runs = fixture()
        self.assertTrue(vpipe_audit.trace(value)["traceEvents"])
        value["pending"] = 1
        with self.assertRaises(ValueError):
            vpipe_audit.trace(value)


if __name__ == "__main__":
    unittest.main()
