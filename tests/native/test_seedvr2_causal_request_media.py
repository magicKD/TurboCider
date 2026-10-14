from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_causal_request_media import validate_pair,load_capture,compare_raw,capture_frame,sha256


class MatchedCausalMedia(unittest.TestCase):
    def capture_pair(self,root):
        workload=dict(kind="video",frames=2,width=16,height=16,fps=24)
        captures=[];runs=[]
        for experimental in (False,True):
            folder=root/("candidate" if experimental else "reference");folder.mkdir()
            rows=[];rgb_hash=hashlib.sha256()
            for index in range(2):
                half=np.full((16,16,3),.5,dtype="<f2");rgb=np.full(half.shape,191,dtype="u1")
                if experimental and index==1:half[0,0,0]=.501;rgb[0,0,0]=192
                row=dict(index=index)
                for name,suffix,values in (("fp16","fp16",half),("rgb24","rgb24",rgb)):
                    path=folder/f"frame-{index}.{suffix}";path.write_bytes(values.tobytes())
                    row[name]=dict(path=path.name,bytes=path.stat().st_size,sha256=sha256(path))
                rgb_hash.update(rgb.tobytes());rows.append(row)
            shape=[1,16,2,2,2]
            tensors={name:dict(shape=shape,dtype="float16" if name=="condition" else "float32",
                bytes=256 if name=="condition" else 512,sha256="a"*64) for name in ("condition","velocity","latent")}
            capture=dict(schema="seedvr2-causal-request-pixels-v1",status="complete",fp16_layout="THWC",width=16,height=16,
                frame_count=2,fps=24,seed=42,source_sha256="source",frames=rows,predecode=tensors,
                experimental_causal_enabled=experimental,causal_manifest_sha256="bank" if experimental else None,
                checkpoint_sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1")
            path=folder/"capture.json";path.write_text(json.dumps(capture))
            run=dict(metrics=dict(seed=42,video_raw_capture=dict(manifest=str(path),sha256=sha256(path),frames=2,diagnostic_only=True),
                vae_causal_conv_manifest_sha256="bank" if experimental else None,
                video_preencode_pixel_hash_diagnostic=True,video_preencode_rgb24_sha256=rgb_hash.hexdigest()))
            captures.append(load_capture(run,workload,"source",experimental));runs.append(run)
        return captures,runs,workload

    def test_raw_capture_binds_every_frame_and_actual_preencoder_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            captures,_,workload=self.capture_pair(Path(folder))
            result=compare_raw(*captures,workload)
            self.assertTrue(result["predecode_bitwise_equal"])
            self.assertEqual(result["preencode_rgb24"]["maximum_error_8bit"],1)
            self.assertEqual(len(result["frames"]),2)
            captures[1]["expected_rgb24_sha256"]="wrong"
            with self.assertRaisesRegex(ValueError,"actual encoder"):compare_raw(*captures,workload)

    def test_changed_latent_missing_frame_stale_receipt_and_escaping_path_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            captures,runs,workload=self.capture_pair(Path(folder))
            captures[1]["predecode"]["latent"]["sha256"]="b"*64
            with self.assertRaisesRegex(ValueError,"same operands"):compare_raw(*captures,workload)
            captures[0]["frames"][0]["fp16"]["path"]="../outside"
            with self.assertRaisesRegex(ValueError,"escapes"):capture_frame(captures[0],0,16,16)
            path=Path(runs[1]["metrics"]["video_raw_capture"]["manifest"])
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError,"request-bound"):load_capture(runs[1],workload,"source",True)

    def pair(self):
        left=dict(status="complete",source_sha256="source",workload=dict(kind="video",frames=25),
            native_artifact_identity=dict(library_sha256="lib"),runs=[dict(metrics=dict(seed=42,execution="gpu",vae_causal_conv_experimental_enabled=False))])
        right=deepcopy(left);right["runs"][0]["metrics"].update(execution="gpu_vae_cpu_ne_experimental",vae_causal_conv_experimental_enabled=True)
        return left,right
    def test_same_source_library_seed_and_real_experiment_are_required(self):
        validate_pair(*self.pair())
        for key,bad in (("source_sha256","other"),("status","running"),("workload",dict(frames=5)),("native_artifact_identity",dict(library_sha256="other"))):
            left,right=self.pair();right[key]=bad
            with self.assertRaises(ValueError):validate_pair(left,right)
        left,right=self.pair();right["runs"][0]["metrics"]["seed"]=43
        with self.assertRaises(ValueError):validate_pair(left,right)


if __name__=="__main__":unittest.main()
