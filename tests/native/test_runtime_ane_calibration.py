"""Native calibration raw evidence/timing scope; no SDK or hardware imports."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
with mock.patch.object(sys,"path",[str(ROOT/"tools/validation"),*sys.path]):
    from runtime_ane_calibration import channel_policy, validate_channel_calibration
    import runtime_ane_common as COMMON


def raw(layer):
    return [[.002+layer+i*.00001 for i in range(7)], [.002+4*layer+i*.00001 for i in range(7)]]


def report():
    points = []
    for share,g,a,b in ((.4,.013,.008,.013),(.8,.009,.02,.02)):
        points.append(dict(share=share,gpu=g,ane=a,both=b,prefetch=False,ane_calls=90,
                           raw_seconds=[raw(g),raw(a),raw(b)]))
    return dict(schema_version=1,enabled=True,cache_hit=False,trial_passed=True,complete=True,
        selected_channels=4096,proposed_channels=4096,bucket_rows=1056,actual_rows=1056,hidden=3840,width=10240,
        layer_count=32,warmups=2,repeats=7,status="accepted",reason="accepted independent FFN candidate",
        scope="complete FFN host spans; not E2E qualification or physical engine trace",
        input_recipe="synthetic-normal-key17-scale0.25; zero-padded ANE bucket",sampled_depths=[0,7,15,23,31],
        identity=dict(model_sha256="a"*64,adapter="",encoding="dense-bf16",precision="bf16",backend="private_ane",
            recipe="sylvester-dh-b128-b512-rne-norm-f16-v2-base",soc="Apple M4 Max",os_build="25G1",runtime_build="build1",
            metal_abi="metal1",graph_abi="prepared-channel-base-v1-b1056-l32",source_generation="1:2:3",
            rows=1056,hidden=3840,width=10240,tile_k=1024,tile_n=512,prefetch=False),
        baseline=dict(layer_seconds=.016,raw_seconds=raw(.016)),points=points,
        trial=dict(gpu_seconds=[.062,.061,.063,.064,.062,.061,.062],candidate_seconds=[.05,.049,.051,.05,.048,.051,.05],
            calls=36,fallbacks=0,retries=0,relative_l2=.01,cosine=.9998,completed=True,accepted=True),
        predicted_layer_seconds=.01339)


class CalibrationTests(unittest.TestCase):
    def test_adapter_costs_require_actual_dynamic_correction_traffic(self):
        value=report();value["lora"]=True;value["identity"]["adapter"]="actual-sha-strength-role";
        value["identity"]["recipe"]="sylvester-dh-b128-b512-rne-norm-f16-v2-lora"
        for point in value["points"]:point.update(correction_computations=90,correction_uploads=180)
        self.assertEqual(validate_channel_calibration(value,10240,3840),4096)
        for change in ({"correction_computations":0},{"correction_uploads":0},{"correction_computations":89}):
            bad=copy.deepcopy(value);bad["points"][0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_channel_calibration(bad,10240,3840)
        bad=copy.deepcopy(value);bad["identity"]["adapter"]=""
        with self.assertRaises(ValueError):validate_channel_calibration(bad,10240,3840)
    def test_accepted_native_samples_are_independently_recomputable(self):
        value=report()
        self.assertEqual(validate_channel_calibration(value,10240,3840),4096)
        value["cache_hit"]=True
        self.assertEqual(validate_channel_calibration(value,10240,3840),4096)

    def test_partial_mismatched_or_false_qualification_rejected(self):
        changes=[("schema_version",True),("selected_channels",513),("trial_passed",False),
            ("complete",False),("scope","E2E qualified"),("width",12288),("hidden",3840.),
            ("sampled_depths",[0,0,0,0,0]),("identity",None),("baseline",None),
            ("points",[]),("trial",None),("predicted_layer_seconds",.016)]
        for name,value in changes:
            bad=report();bad[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):validate_channel_calibration(bad,10240,3840)

    def test_raw_vectors_calls_and_runtime_numerical_gate_cannot_be_faked(self):
        for path,value in ((["baseline","layer_seconds"],.01),(["points",0,"gpu"],.01),
            (["points",1,"ane_calls"],89),(["points",0,"raw_seconds",0,1],[.01]*7),
            (["trial","calls"],35),(["trial","fallbacks"],1),(["trial","retries"],1),
            (["trial","relative_l2"],float("nan")),(["trial","relative_l2"],.031),
            (["trial","cosine"],None),(["trial","cosine"],.998),(["trial","candidate_seconds"],[.08]*7),
            (["trial","candidate_seconds"],[.01]*6),(["identity","recipe"],"comfy"),
            (["identity","model_sha256"],"unknown"),(["identity","prefetch"],True)):
            bad=report();target=bad
            for key in path[:-1]:target=target[key]
            target[path[-1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError):validate_channel_calibration(bad,10240,3840)

    def test_clean_decline_is_gpu_only_not_an_accepted_ane_trial(self):
        value=report();value.update(status="gpu_only",selected_channels=0,proposed_channels=0,trial_passed=False,
            trial=None,predicted_layer_seconds=.016)
        self.assertEqual(validate_channel_calibration(value,10240,3840),0)
        row=dict(runtime_backend="mlx_cpp_metal",timings_seconds={"request_wall":1.,"denoise":.8},
            hybrid=dict(runtime_failed=False,runtime_failures_session_total=0,runtime_calls_session_total=0,
                runtime_weight=dict(channel_calibration=value,executor_backend=None,gpu_blocks_session_total=256,
                    fallback_blocks_session_total=0,overflow_retries_session_total=0,device_io_calls_session_total=0)))
        COMMON.validate_results([row],"runtime",1,runtime_backend="private",expect_device_io=True,
            expected_data_path="w8a8_hadamard",channel_auto=True)
        with self.assertRaises(ValueError):COMMON.validate_results([row],"runtime",1,runtime_backend="private")
        for name in ("fallback_blocks_session_total","overflow_retries_session_total","device_io_calls_session_total"):
            bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"][name]=1
            with self.subTest(name=name),self.assertRaises(ValueError):
                COMMON.validate_results([bad],"runtime",1,runtime_backend="private",channel_auto=True)

    def test_adopted_runtime_must_match_actual_trial_and_physical_width(self):
        row=dict(runtime_backend="mlx_cpp_metal+private_ane_runtime_weight_experimental",
            timings_seconds={"request_wall":1.,"denoise":.8},
            hybrid=dict(runtime_failed=False,runtime_failures_session_total=0,runtime_calls_session_total=1,
                runtime_weight=dict(channel_calibration=report(),executor_backend="private_ane",io_path="gpu_iosurface",
                    data_path="w8a8_hadamard",partition_axis="intermediate_channels",ane_channels=4096,gpu_channels=6144,
                    fallback_blocks_session_total=0,failure_reason="",device_io_calls_session_total=1,
                    overflow_retries_session_total=0,headroom_scale=1)))
        COMMON.validate_results([row],"runtime",1,runtime_backend="private",expect_device_io=True,
            expected_data_path="w8a8_hadamard",channel_auto=True)
        for change in ({"ane_channels":5120},{"gpu_channels":4096},{"channel_calibration":None},
                       {"executor_backend":"public_coreml"},{"overflow_retries_session_total":1},
                       {"overflow_retries_session_total":None},{"headroom_scale":4},{"headroom_scale":True},
                       {"headroom_scale":float("nan")},{"headroom_scale":None}):
            bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                COMMON.validate_results([bad],"runtime",1,runtime_backend="private",channel_auto=True)

    def test_channel_cli_policy_does_not_accept_negative_unresolved_width(self):
        self.assertEqual(channel_policy("auto"),"auto")
        self.assertEqual(channel_policy("4096"),4096)
        for value in ("-1","-512","AUTO"," 512","1.5",""):
            with self.subTest(value=value),self.assertRaises(Exception):channel_policy(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
