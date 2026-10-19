"""CPU-only matched full-RGBA and prefix arithmetic/replay gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import collect
import gpu_dual_request_audit as audit
from test_qwen_base_gpu_dual_runtime import receipts


class GpuDualRequestAuditTests(unittest.TestCase):
    def records(self,directory):
        values=[]
        for enabled in (False,True):
            root=Path(directory)/str(enabled);(root/"trial").mkdir(parents=True)
            for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],output=str(root/f"{i}.png"))))
            values.append(dict(root=root,images=[np.full((8,8,4),i,dtype=np.uint8) for i in range(3)],
                summary=dict(mode="stock",gpu_dual_swiglu=enabled,case="generate",scenario="repeat",runtime_prefix_snapshot=False,
                    runtime_text_prefix_snapshot=True,text_prefix_qkv_step0="gpu",environment=collect.environment("runtime","stock",6144,
                        private_qkv=True,fixed_program=True,ffn_activation="sigmoid-complement",gpu_dual_swiglu=enabled),
                    assets=dict.fromkeys(("library","manifest","reference","observer"),"same")),
                receipt=dict(hot_median_seconds=30.,prefix_receipts=[dict(recipe_sha256=("b" if enabled else "a")*64)])))
        return values

    def test_pair_has_exact_one_switch_source_and_full_rgba_gate(self):
        with tempfile.TemporaryDirectory() as d:
            off,on=self.records(d);audit.pair(off,on)
            for change in (lambda x:x["images"][2].__setitem__((0,0,3),9),lambda x:x["images"].pop(),
                lambda x:x["summary"]["environment"].update(TURBOCIDER_PRIVATE_ANE_CHANNELS="5120"),
                lambda x:x["summary"]["assets"].update(reference="different")):
                bad=copy.deepcopy(on);change(bad)
                with self.assertRaises(ValueError):audit.pair(off,bad)

    def test_changed_gpu_policy_cannot_reuse_identical_prefix_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            off,on=self.records(d);on["receipt"]["prefix_receipts"][0]["recipe_sha256"]="a"*64
            with self.assertRaises(ValueError):audit.pair(off,on)

    def test_replay_matches_cache_route_and_position_not_just_public_seed(self):
        with tempfile.TemporaryDirectory() as d:
            off,on=self.records(d)
            self.assertEqual(audit.replay(off,on),0)
            same=copy.deepcopy(off);self.assertEqual(audit.replay(off,same),3)
            same["images"][1][0,0,0]=9
            with self.assertRaises(ValueError):audit.replay(off,same)
            other=copy.deepcopy(off);other["summary"]["runtime_prefix_snapshot"]=True
            self.assertEqual(audit.replay(off,other),0)
            other=copy.deepcopy(off);other["summary"]["prompt"]="different workload"
            self.assertEqual(audit.replay(off,other),0)

    def test_session_validation_time_is_frozen_not_merely_monotonic(self):
        for enabled in (False,True):audit.validation_once(receipts(enabled),enabled)
        rows=receipts(True);rows[1]["qwen_gpu_dual_swiglu"]["validation_seconds_session_total"]+=.01
        with self.assertRaises(ValueError):audit.validation_once(rows,True)

    def test_raw_memory_recomputed_not_only_summary_or_digest_trusted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual=dict(complete=True,command_exit_code=0,correlation_id="owned",tree_peak_phys_footprint_bytes=123)
            (root/"report.json").write_text(json.dumps(actual));(root/"memory.jsonl").write_text("raw evidence placeholder")
            memory=dict(report="report.json",evidence="memory.jsonl",tool_sha256={},verified=copy.deepcopy(actual),correlation_id="owned")
            for name in ("report","evidence"):memory[name+"_sha256"]=audit.sha256_file(root/memory[name])
            summary=dict(memory=memory)
            with patch.object(audit,"verify_memory",return_value=actual) as verifier:
                self.assertEqual(audit.memory_contract(root,summary),actual)
                verifier.assert_called_once_with(root/"memory.jsonl")
                for change in (lambda x:x["memory"]["verified"].update(tree_peak_phys_footprint_bytes=1),
                               lambda x:x["memory"].update(correlation_id="unrelated")):
                    bad=copy.deepcopy(summary);change(bad)
                    with self.assertRaises(ValueError):audit.memory_contract(root,bad)
            with patch.object(audit,"verify_memory",return_value={**actual,"tree_peak_phys_footprint_bytes":321}):
                with self.assertRaises(ValueError):audit.memory_contract(root,summary)


if __name__=="__main__":unittest.main()
