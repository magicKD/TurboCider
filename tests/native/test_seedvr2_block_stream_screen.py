from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_block_stream_screen import weight_group,validate_weight_reads,validate_forward,compare_words,validate_lifecycle,SHARED_ROPE


class BlockStreamGuards(unittest.TestCase):
    def test_partial_read_errors_and_real_recovery_cannot_be_relabelled(self):
        keys=("pre_cancel_checked","mid_read_cancel_typed","after_block_cancel_checked","same_model_recovery_words_equal",
            "non_cancel_read_error_latched","duplicate_group_atomic","wrong_group_rejected","source_mutation_rejected",
            "source_replacement_rejected","unverified_source_rejected","fresh_source_recovery","zero_active_teardown")
        value=dict(schema="seedvr2-block-stream-lifecycle-v1",status="complete",qualified=False,mid_read_cancelled_payload_bytes=5120,
            completed_blocks=66,**{key:True for key in keys})
        validate_lifecycle(value)
        for change in [{key:False} for key in keys]+[{"mid_read_cancelled_payload_bytes":0},{"mid_read_cancelled_payload_bytes":True},
                                                   {"completed_blocks":32},{"status":"running"},{"qualified":True}]:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_lifecycle({**value,**change})
    def fixture(self):
        header={};position=0
        for name,size in [(SHARED_ROPE,42),("positive_conditioning",4)]+[(f"blocks.{i}.test.weight",4) for i in range(32)]:
            header[name]=dict(dtype="F16",shape=[size//2],data_offsets=[position,position+size]);position+=size
        reads=[dict(group=weight_group(name),offset=1024+record["data_offsets"][0],bytes=record["data_offsets"][1]-record["data_offsets"][0])
            for name,record in header.items()]
        reads.append(dict(reads[-1]))
        value=dict(schema="seedvr2-block-weight-parity-v1",status="complete",qualified=False,all_words_equal=True,
            zero_active_teardown=True,tensors=len(header),compared_compute_bytes=position+42,original_source_payload_bytes=position,reads=reads)
        return header,value

    def test_shared_rope_is_global_and_block_numbers_are_canonical(self):
        self.assertEqual(weight_group(SHARED_ROPE),-1)
        self.assertEqual(weight_group("positive_conditioning"),-1)
        self.assertEqual(weight_group("blocks.31.mlp.weight"),31)
        for name in ("blocks.01.x","blocks.32.x","blocks.-1.x","blocks.0","blocks..x"):
            with self.subTest(name=name),self.assertRaises(ValueError):weight_group(name)

    def test_every_payload_range_must_belong_to_the_actual_selected_group(self):
        header,value=self.fixture();gate=validate_weight_reads(value,1024,header)
        self.assertEqual(gate["tensors"],34);self.assertTrue(gate["all_reads_in_selected_groups"])
        for change in ({"group":0},{"offset":1000},{"bytes":9999},{"bytes":True}):
            bad=deepcopy(value);bad["reads"][0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_weight_reads(bad,1024,header)
        for change in ({"all_words_equal":False},{"zero_active_teardown":False},{"status":"running"},
                       {"tensors":33},{"compared_compute_bytes":1},{"original_source_payload_bytes":1},
                       {"reads":value["reads"][:-1]}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_weight_reads({**value,**change},1024,header)

    def forward(self):
        return dict(schema="seedvr2-block-stream-forward-v1",status="complete",route="stream",qualified=False,zero_active_teardown=True,
            frames_latent=3,height_latent=16,width_latent=16,total_streamed_blocks=64,peak_parameter_bytes=438862120,
            sampled_process_footprint_peak_bytes=2000000000,
            iterations=[dict(iteration=i,warm=i>0,blocks_completed=32,streamed_blocks=32,source_payload_bytes=1000,
                mlx_active_peak_bytes=1500000000,forward_seconds=2.) for i in range(2)])

    def test_all_blocks_and_actual_serial_io_are_mandatory(self):
        value=self.forward();validate_forward(value,"stream",[1,16,3,16,16],2,1000)
        for change in ({"blocks_completed":31},{"streamed_blocks":31},{"source_payload_bytes":999},
                       {"warm":False},{"forward_seconds":float("nan")},{"mlx_active_peak_bytes":0}):
            bad=deepcopy(value);bad["iterations"][-1].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_forward(bad,"stream",[1,16,3,16,16],2,1000)
        for change in ({"total_streamed_blocks":32},{"frames_latent":1},{"peak_parameter_bytes":600<<20},{"zero_active_teardown":False}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_forward({**value,**change},"stream",[1,16,3,16,16],2,1000)

    def test_output_word_gate_does_not_treat_signed_zero_as_equal(self):
        try:import numpy as np
        except ImportError:self.skipTest("NumPy validation interpreter required")
        a=np.array([0.,1.],dtype=np.float32);compare_words(a,a.copy())
        for b in (np.array([-0.,1.],dtype=np.float32),a.astype(np.float16),np.array([0.,np.nan],dtype=np.float32),a.reshape(1,2)):
            with self.subTest(candidate=b),self.assertRaises(ValueError):compare_words(a,b)


if __name__=="__main__":unittest.main()
