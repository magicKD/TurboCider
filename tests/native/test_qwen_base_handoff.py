"""CPU-only controls for event/encoder timestamps and explicit fixed shares."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import handoff_collect as screen
import handoff_audit
import ffn_projection_collect as projection
from test_qwen_base_projection import fixture


def report_fixture(channels=6144):
    r=fixture();r.update(projection="descriptor",activation="sigmoid-complement",handoff_screen=True,
        ane_channels=channels,gpu_channels=12288-channels,gpu_complement_strategy="separate",gpu_packed_weight_bytes=0,gpu_packed_setup_seconds=0.)
    r["metrics"]["runtime_weight"].update(ane_channels=channels,gpu_channels=12288-channels,
        source_recipe="sylvester:w8-activation-sigmoid-complement-v1+fp32-partial-join-v1")
    for row in r["samples"]:
        if row["arm"]=="private-restaged":row["completion_deltas"]=dict(prediction_wait_seconds=.01,output_wait_seconds=.003,caller_wait_seconds=.014)
    r["gpu_complement_alone_samples"]=[dict(rep=i,warmup=i<0,input_variant=bool(i&1),seconds=.01) for i in range(-2,8)]
    return r


def raw_fixture(report):
    scopes=[1,2,3,4,5]+[6+i for i,s in enumerate(report["samples"]) if s["arm"]=="private-restaged"]
    rows=[]
    for j,request in enumerate([-4]+scopes):
        start=2_000_000+j*100_000
        rows.append(dict(request=request,begin_ns=start-1000,end_ns=start+10000,ok=True,
            gpu_begin_ns=start,gpu_end_ns=start+2000,encoder_sampled=True,resolved_bytes=16,
            encoder_begin_ticks=start,encoder_end_ticks=start+2000,
            events=[dict(event=1 if request==-4 else 2,value=1 if request==-4 else 8+2*(j-1),wait=True)]))
    rows.append(dict(request=-5,begin_ns=2_000_000,end_ns=22_100_000,ok=True,gpu_begin_ns=22_000_000,gpu_end_ns=22_002_000,
        encoder_sampled=True,resolved_bytes=16,encoder_begin_ticks=22_000_000,encoder_end_ticks=22_002_000,
        events=[dict(event=3,value=1,wait=True)]))
    return dict(handoff=dict(schema="tc-metal-event-encoder-timestamps-v1",drained=True,pending=0,dropped=0,uncommitted_event_buffers=0,
        calibration=dict(begin_host_before_ns=999_999,begin_host_after_ns=1_000_001,begin_gpu_ticks=1_000_000,
            end_host_before_ns=29_999_999,end_host_after_ns=30_000_001,end_gpu_ticks=30_000_000),records=rows))


class HandoffTests(unittest.TestCase):
    def test_share_is_explicit_and_legacy_default_still_rejects_other_share(self):
        for channels in (5120,6144):
            r=report_fixture(channels);result=screen.validate(r,8,channels)
            self.assertEqual(result["native_workload_predictions"],15)
            self.assertEqual(result["gpu_complement_alone_median_seconds"],.01)
        with self.assertRaises(ValueError):projection.validate(report_fixture(5120),8,"descriptor","sigmoid-complement")
        for channels in (True,0,8192,"6144"):
            with self.assertRaises(ValueError):projection.validate({},8,"descriptor",channels=channels)

    def test_counter_elapsed_is_not_cb_window_or_worker_wait(self):
        r=report_fixture();raw=raw_fixture(r);value=screen.handoff_analysis(raw,r,8)
        self.assertEqual(value["actual_workload_restore_encoders"],15)
        self.assertAlmostEqual(value["hot_encoder_gpu_median_seconds"],.000002)
        self.assertAlmostEqual(value["hot_output_wait_median_seconds"],.003)
        self.assertIn("must not be summed",value["scope"])

    def test_missing_dropped_clock_and_attribution_negative_controls(self):
        r=report_fixture();raw=raw_fixture(r)
        changes=(lambda x:x["handoff"].update(dropped=1),lambda x:x["handoff"].update(pending=False),
            lambda x:x["handoff"]["calibration"].update(end_gpu_ticks=1),
            lambda x:x["handoff"]["calibration"].update(end_host_after_ns=31_000_000),
            lambda x:x["handoff"]["records"].pop(),lambda x:x["handoff"]["records"].append(copy.deepcopy(x["handoff"]["records"][-1])),
            lambda x:x["handoff"]["records"][0].update(request=-5),
            lambda x:x["handoff"]["records"][-1].update(encoder_end_ticks=(1<<64)-1),
            lambda x:x["handoff"]["records"][-1].update(encoder_begin_ticks=100),
            lambda x:x["handoff"]["records"][-1].update(encoder_sampled=False),
            lambda x:x["handoff"]["records"][-1].update(begin_ns=9_000_000,end_ns=9_100_000),
            lambda x:x["handoff"]["records"][2]["events"][0].update(event=3),
            lambda x:x["handoff"]["records"][2]["events"][0].update(value=8),
            lambda x:x["handoff"]["records"][-1]["events"][0].update(wait=False))
        for change in changes:
            changed=copy.deepcopy(raw);change(changed)
            with self.assertRaises(ValueError):screen.handoff_analysis(changed,r,8)
        for field in ("prediction_wait_seconds","output_wait_seconds","caller_wait_seconds"):
            for invalid in (-1,float("nan"),float("inf"),True):
                changed=copy.deepcopy(r)
                next(s for s in changed["samples"] if s["arm"]=="private-restaged")["completion_deltas"][field]=invalid
                with self.assertRaises(ValueError):screen.handoff_analysis(raw,changed,8)

    def test_complement_alone_is_bounded_and_not_a_complete_image(self):
        r=report_fixture()
        for change in (lambda x:x.update(complete_request=True),lambda x:x.update(handoff_screen=False),
            lambda x:x["gpu_complement_alone_samples"].pop(),
            lambda x:x["gpu_complement_alone_samples"][0].update(seconds=float("nan")),
            lambda x:x["gpu_complement_alone_samples"].reverse()):
            changed=copy.deepcopy(r);change(changed)
            with self.assertRaises(ValueError):screen.validate(changed,8,6144)

    def test_packing_cost_and_strategy_cannot_be_hidden(self):
        r=report_fixture();r.update(gpu_complement_strategy="packed",gpu_packed_weight_bytes=2*2*6144*4096*2,gpu_packed_setup_seconds=.001)
        screen.validate(r,8,6144,"packed")
        with self.assertRaises(ValueError):screen.validate(r,8,6144)
        for key,value in (("gpu_packed_weight_bytes",0),("gpu_packed_setup_seconds",-1),("gpu_packed_setup_seconds",float("nan"))):
            changed=copy.deepcopy(r);changed[key]=value
            with self.assertRaises(ValueError):screen.validate(changed,8,6144,"packed")

    def test_audit_requires_actual_source_controls_and_same_recipe_replays(self):
        base=dict(channels=6144,gpu_complement="separate",array_sha256=dict.fromkeys(
            ("ffn_input","block_reference","alternate_reference","other_weight_reference","block_candidate"),"same"))
        replay=copy.deepcopy(base);packed=copy.deepcopy(base);packed["gpu_complement"]="packed";packed["array_sha256"]["block_candidate"]="different"
        handoff_audit.matched([base,replay,packed])
        with self.assertRaises(ValueError):handoff_audit.matched([])
        changed=copy.deepcopy(replay);changed["array_sha256"]["block_candidate"]="stale"
        with self.assertRaises(ValueError):handoff_audit.matched([base,changed])
        for key in ("ffn_input","block_reference","alternate_reference","other_weight_reference"):
            changed=copy.deepcopy(packed);changed["array_sha256"][key]="different"
            with self.assertRaises(ValueError):handoff_audit.matched([base,changed])


if __name__=="__main__":unittest.main()
