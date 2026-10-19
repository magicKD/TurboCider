"""CPU negative controls for raw activation diagnostics, not W8 performance."""
import copy
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import activation_op_collect as screen


def fixture():
    report=dict(status="complete",synthetic_only=True,model_weights_loaded=False,complete_request=False,image_export=False,
        dtype="float16-le",layout="channels_tokens",source_unchanged=True,identity_bits_exact=True,same_program_restoration_exact=True,
        programs=[],samples=[])
    def event(name,scope,model):
        return dict(name=name,request=scope,model=model,thread=1,queue=1,begin_ns=100,end_ns=120,
                    gpu_begin_ns=0,gpu_end_ns=0,ok=True)
    calibration=event("metal.command_buffer",-2,0);calibration.update(gpu_begin_ns=101,gpu_end_ns=119)
    events=[calibration]
    shapes=[(r,c,m) for r,c in ((32,128),(1024,6144)) for m in screen.MODES]
    for ordinal,(rows,channels,mode) in enumerate(shapes):
        report["programs"].append(dict(ordinal=ordinal,rows=rows,channels=channels,operator=mode,cache_key="key"+str(ordinal)))
        events.extend([event("ane.compile_api",ordinal*10,ordinal+1),event("ane.load_api",ordinal*10,ordinal+1)])
        for i,pattern in enumerate(screen.PATTERNS):
            scope=ordinal*10+i+1;events.append(event("ane.evaluate_api",scope,ordinal+1))
            report["samples"].append(dict(program=ordinal,scope=scope,operator=mode,pattern=pattern,rows=rows,channels=channels,
                quality=dict(elements=rows*channels)))
    observed=dict(enabled=True,drained=True,restored_imps=True,pending=0,dropped=0,events=events,
        counts=dict(Counter(x["name"] for x in events)))
    return report,observed


class ActivationOperatorTests(unittest.TestCase):
    def test_raw_diagnostics_do_not_claim_checkpoint_or_image_performance(self):
        report,observed=fixture();result=screen.validate(report,observed)
        self.assertEqual(result["actual_evaluations"],48)
        self.assertEqual(len(result["programs"]),12)
        for key in ("model_weights_loaded","complete_request","image_export"):
            changed=copy.deepcopy(report);changed[key]=True
            with self.assertRaises(ValueError):screen.validate(changed,observed)

    def test_identity_and_full_shape_samples_cannot_be_dropped_or_relabelled(self):
        report,observed=fixture()
        for change in (lambda r:r.update(identity_bits_exact=False),lambda r:r["samples"].pop(),
            lambda r:r["samples"][0].update(rows=16),lambda r:r["samples"][-1].update(pattern="wide"),
            lambda r:r["programs"][1].update(cache_key="key0")):
            changed=copy.deepcopy(report);change(changed)
            with self.assertRaises(ValueError):screen.validate(changed,observed)

    def test_actual_hot_compile_or_changed_model_is_not_allowed(self):
        for mode in ("compile","model"):
            report,observed=fixture()
            if mode=="compile":
                row=copy.deepcopy(observed["events"][-1]);row["name"]="ane.compile_api"
                observed["events"].append(row);observed["counts"]["ane.compile_api"]+=1
            else:observed["events"][-1]["model"]=100
            with self.assertRaises(ValueError):screen.validate(report,observed)


if __name__=="__main__":unittest.main()
