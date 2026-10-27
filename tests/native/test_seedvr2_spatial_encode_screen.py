#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
from seedvr2_spatial_encode_screen import validate_observation
from seedvr2_spatial_encode_fp32_compare import validate_reference

class EncoderScreenTests(unittest.TestCase):
    def observation(self,frames=25):
        return dict(schema="seedvr2-spatial-encode-observation-v1",route="spatial",source_frames=frames,height=1024,width=1024,
            finite=True,qualified=False,file_live_bytes_after_encode=0,host_live_bytes_after_encode=0,cleanup_failures=0,
            tile_convolutions=frames*64,global_normalizations=frames*4,file_written_bytes=1 if frames>1 else 0)

    def test_complete_frame_global_norm_and_file_ownership_gate(self):
        value=self.observation();validate_observation(value,"spatial",25,1024,1024,256<<20)
        for update in ({"global_normalizations":4},{"source_frames":5},{"host_live_bytes_after_encode":1},
                       {"cleanup_failures":1},{"file_written_bytes":0}):
            with self.assertRaises(ValueError):validate_observation({**value,**update},"spatial",25,1024,1024,256<<20)

    def test_image_has_no_causal_file_writes(self):
        value=self.observation(1);validate_observation(value,"spatial",1,1024,1024,256<<20)
        with self.assertRaises(ValueError):validate_observation({**value,"file_written_bytes":1},"spatial",1,1024,1024,256<<20)

    def test_fp32_reference_needs_actual_dtypes_audit_and_same_source(self):
        screen=dict(status="complete",checkpoint_sha256="a",input_sha256="b",source_shape=[1,3,25,1024,1024])
        reference=dict(status="complete",schema="seedvr2-vae-fp32-encoder-oracle-v1",parameter_dtypes=["torch.float32"],
            input_dtype="torch.float32",output_dtype="torch.float32",floating_dispatch_values_checked=1,
            checkpoint_sha256="a",input_sha256="b",input_shape=screen["source_shape"])
        validate_reference(screen,reference)
        for update in ({"parameter_dtypes":["torch.float16"]},{"output_dtype":"torch.float16"},
                       {"floating_dispatch_values_checked":0},{"input_sha256":"different"},{"status":"running"}):
            with self.assertRaises(ValueError):validate_reference(screen,{**reference,**update})

if __name__=="__main__":unittest.main()
