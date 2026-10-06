from pathlib import Path
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
with mock.patch.object(sys,"path",[str(ROOT/"tools/validation"),*sys.path]):
    from runtime_ane_quality_compare import load_tensor,pair,compare_generation,bind_execution


def write(path,values,dtype="F32"):
    values=np.asarray(values,dtype=np.float32)
    if dtype=="BF16":raw=(values.view(np.uint32)>>16).astype("<u2").tobytes()
    else:raw=values.astype("<f4" if dtype=="F32" else "<f2").tobytes()
    header=json.dumps({"tensor":dict(shape=list(values.shape),dtype=dtype,data_offsets=[0,len(raw)])}).encode()
    path.write_bytes(struct.pack("<Q",len(header))+header+raw)


class QualityTests(unittest.TestCase):
    def test_execution_binding_distinguishes_gpu_decline_and_actual_ane(self):
        from tests.native.test_runtime_ane_calibration import report
        with tempfile.TemporaryDirectory() as folder:
            gpu,ane=Path(folder)/"gpu.json",Path(folder)/"ane.json"
            common=dict(model="z-image-turbo",width=512,height=512,seed=42,steps=8,actual_denoise_steps=8,lora_strategy="none",
                lora_applied_projections=0,timings_seconds=dict(request_wall=1.,denoise=.8))
            baseline=dict(**common,runtime_backend="mlx_cpp_metal")
            calibration=report();calibration.update(status="gpu_only",selected_channels=0,proposed_channels=0,
                trial_passed=False,trial=None,predicted_layer_seconds=.016)
            candidate=dict(**baseline,hybrid=dict(runtime_failed=False,runtime_failures_session_total=0,
                runtime_calls_session_total=0,runtime_weight=dict(channel_calibration=calibration,executor_backend=None,
                    gpu_blocks_session_total=256,fallback_blocks_session_total=0,overflow_retries_session_total=0,
                    device_io_calls_session_total=0)))
            def store(path,row):path.write_text(json.dumps(dict(exit_code=0,binary_sha256="a"*64,stdout=json.dumps(row))))
            store(gpu,baseline);store(ane,candidate)
            self.assertFalse(bind_execution(gpu,ane,"z-image-turbo")["candidate_ane_executed"])
            candidate.update(runtime_backend="mlx_cpp_metal+private_ane_runtime_weight_experimental")
            candidate["hybrid"]["runtime_calls_session_total"]=1
            candidate["hybrid"]["runtime_weight"].update(channel_calibration=report(),executor_backend="private_ane",
                partition_axis="intermediate_channels",ane_channels=4096,gpu_channels=6144,io_path="gpu_iosurface",
                device_io_calls_session_total=1,data_path="w8a8_hadamard",headroom_scale=1,failure_reason="")
            store(ane,candidate)
            self.assertTrue(bind_execution(gpu,ane,"z-image-turbo")["candidate_ane_executed"])
            candidate["seed"]=43;store(ane,candidate)
            with self.assertRaises(ValueError):bind_execution(gpu,ane,"z-image-turbo")
            candidate["seed"]=42
            for change in ({"binary_sha256":None},{"binary_sha256":"not-a-hash"},{"exit_code":False},
                           {"artifacts_unchanged":False},{"adjacent_library_sha256":"b"*64},
                           {"adjacent_library_sha256":"b"*64,"artifacts_unchanged":True}):
                store(ane,candidate)
                raw=json.loads(ane.read_text());raw.update(change);ane.write_text(json.dumps(raw))
                with self.subTest(change=change),self.assertRaises(ValueError):bind_execution(gpu,ane,"z-image-turbo")
            for path,row in ((gpu,baseline),(ane,candidate)):
                store(path,row)
                raw=json.loads(path.read_text());raw.update(adjacent_library_sha256="b"*64,artifacts_unchanged=True)
                path.write_text(json.dumps(raw))
            self.assertTrue(bind_execution(gpu,ane,"z-image-turbo")["adjacent_runtime_identity_bound"])
            candidate["actual_denoise_steps"]=7;store(ane,candidate)
            with self.assertRaises(ValueError):bind_execution(gpu,ane,"z-image-turbo")
    def test_dtype_loader_and_finite_zero_energy_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/"a",Path(folder)/"b"
            for dtype in ("F32","F16","BF16"):
                write(a,[1.,2.,-1.],dtype);write(b,[1.,2.,-1.],dtype)
                self.assertTrue(pair(a,b)["exact"])
                self.assertEqual(load_tensor(a)[0].tolist(),[1.,2.,-1.])
            write(a,[0.,0.]);write(b,[0.,0.])
            with self.assertRaises(ValueError):pair(a,b)
            write(a,[1.,float("nan")])
            with self.assertRaises(ValueError):load_tensor(a)
            write(a,[1.,2.],"BF16");write(b,[1.,2.],"F16")
            with self.assertRaises(ValueError):pair(a,b)

    def test_matching_inputs_final_n1_and_failed_candidate_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/"gpu",Path(folder)/"ane";a.mkdir();b.mkdir()
            for name in ("qwen21_text","qwen21_initial","qwen21_latents"):
                for root in (a,b):write(root/(name+".safetensors"),[1.,2.,3.])
            result=compare_generation(a,b,"qwen-image-2.1")
            self.assertTrue(result["n1_final_latent_pass"]);self.assertFalse(result["qualification_passed"])
            write(b/"qwen21_latents.safetensors",[2.,4.,6.])
            self.assertFalse(compare_generation(a,b,"qwen-image-2.1")["n1_final_latent_pass"])
            write(b/"qwen21_initial.safetensors",[2.,4.,6.])
            with self.assertRaises(ValueError):compare_generation(a,b,"qwen-image-2.1")

    def test_trajectory_completeness_and_header_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/"gpu",Path(folder)/"ane";a.mkdir();b.mkdir()
            for name in ("z_conditioning","z_latent_initial","z_latent_final","z_latent_step_1"):
                for root in (a,b):write(root/(name+".safetensors"),[1.,2.])
            self.assertEqual(len(compare_generation(a,b,"z-image-turbo")["trajectory"]),1)
            write(a/"z_latent_step_2.safetensors",[1.,2.])
            with self.assertRaises(ValueError):compare_generation(a,b,"z-image-turbo")
            invalid=a/"invalid";invalid.write_bytes(struct.pack("<Q",(16<<20)+1))
            with self.assertRaises(ValueError):load_tensor(invalid)
            invalid.write_bytes(b"x")
            with self.assertRaises(ValueError):load_tensor(invalid)

    def test_receipts_require_all_executed_z_steps_before_publication(self):
        from tests.native.test_runtime_ane_calibration import report
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a,b=root/"gpu",root/"ane";a.mkdir();b.mkdir()
            for name in ("z_conditioning","z_latent_initial","z_latent_final","z_latent_step_1"):
                for path in (a,b):write(path/(name+".safetensors"),[1.,2.])
            common=dict(model="z-image-turbo",width=512,height=512,seed=42,steps=8,actual_denoise_steps=8,
                lora_strategy="none",lora_applied_projections=0,runtime_backend="mlx_cpp_metal",
                timings_seconds=dict(request_wall=1.,denoise=.8))
            calibration=report();calibration.update(status="gpu_only",selected_channels=0,proposed_channels=0,
                trial_passed=False,trial=None,predicted_layer_seconds=.016)
            candidate=dict(**common,hybrid=dict(runtime_failed=False,runtime_failures_session_total=0,
                runtime_calls_session_total=0,runtime_weight=dict(channel_calibration=calibration,executor_backend=None,
                    gpu_blocks_session_total=256,fallback_blocks_session_total=0,overflow_retries_session_total=0,
                    device_io_calls_session_total=0)))
            receipts=[root/"gpu.json",root/"ane.json"]
            for path,row in zip(receipts,(common,candidate)):
                path.write_text(json.dumps(dict(exit_code=0,binary_sha256="a"*64,stdout=json.dumps(row))))
            output=root/"result.json"
            result=subprocess.run([sys.executable,str(ROOT/"tools/validation/runtime_ane_quality_compare.py"),
                "--reference",str(a),"--candidate",str(b),"--model-id","z-image-turbo",
                "--reference-receipt",str(receipts[0]),"--candidate-receipt",str(receipts[1]),"--output",str(output)],
                capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("do not cover all executed steps",result.stderr)
            self.assertFalse(output.exists())


if __name__=="__main__":unittest.main(verbosity=2)
