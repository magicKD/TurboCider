"""CPU-only original hidden-normalization, oracle and exact profiling contracts."""
from collections import Counter
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/profiling/qwen_base"))
import hidden_quant_collect as screen
import hidden_quant_audit as audit
from test_qwen_base_profiling import fixture


def report(rejected=()):
    programs=[];arrays=[];samples=[]
    for g,(m,c,rotate) in enumerate(screen.GROUPS):
        for p,policy in enumerate(screen.POLICIES):
            n=g*3+p;programs.append(dict(ordinal=n,rows=m,channels=c,rotate=rotate,policy=policy,
                status="compile-rejected" if n in rejected else "ready",error="explicit compile failure" if n in rejected else "",key=f"key{n}"))
        selected=[p for p in range(3) if g*3+p not in rejected]
        for i,pattern in enumerate(screen.PATTERNS):
            for p in selected:
                n=g*3+p;bad=rotate and pattern=="overflow"
                arrays.append(dict(ordinal=n,scope=n*1000+i+1,policy=screen.POLICIES[p],rotate=rotate,pattern=pattern,rows=m,channels=c,
                                   expected_rejection=bad,validation_flags=2 if bad else 0))
        for rep in range(-2,12):
            for j in range(len(selected)):
                p=selected[(j+rep+len(selected))%len(selected)];n=g*3+p
                samples.append(dict(ordinal=n,scope=n*1000+102+rep,policy=screen.POLICIES[p],rows=m,channels=c,rotate=rotate,
                                    rep=rep,warmup=rep<0,input_variant=bool(rep&1),validation_flags=0,seconds=.001+p*.0001))
    return dict(status="complete",synthetic_only=True,model_weights_loaded=False,complete_request=False,image_export=False,
                repeats=12,warmups=2,dtype="float16-le",layout="channels_tokens_packed_scale",source_unchanged=True,
                same_program_restoration_exact=True,overflow_negative_checked=True,programs=programs,arrays=arrays,samples=samples)


def observed(r):
    raw,_=fixture();base=next(e for e in raw["events"] if e["name"]=="ane.evaluate_api")
    raw["events"]=[e for e in raw["events"] if e["request"]==-2]
    for p in r["programs"]:
        n=p["ordinal"];raw["events"].append(dict(base,name="ane.compile_api",request=n*1000,model=n+1,ok=p["status"]=="ready"))
        if p["status"]!="ready":continue
        for phase in list(range(1,7))+list(range(100,114)):raw["events"].append(dict(base,request=n*1000+phase,model=n+1,ok=True))
    raw["counts"]=dict(Counter(e["name"] for e in raw["events"]))
    return raw


def packed_reference(x,rotate):
    _,_,codes,scale=audit.reference(x,rotate)
    return np.concatenate([(codes.astype(np.float64)/128).astype("<f2"),scale[None,:]],axis=0)


class HiddenQuantTests(unittest.TestCase):
    def test_native_exact_original_subgraph_constants_and_inverse_domain(self):
        with tempfile.TemporaryDirectory(prefix="tc-hidden-quant-") as directory:
            binary=Path(directory)/"test"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-I",str(ROOT/"native/core"),
                str(ROOT/"tests/native/qwen_hidden_quant_mil_test.cpp"),str(ROOT/"native/backends/private/ane_mil.cpp"),"-o",str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)

    def test_complete_counts_and_only_optional_setup_rejection(self):
        for rejected in ((),(2,8)):
            r=report(rejected);result=screen.validate(r,12)
            self.assertEqual(result["total_operator_calls"],(12-len(rejected))*20)
            screen.observer_contract(observed(r),r,12)
        for rejected in ((0,),(3,),(6,),(9,)):
            with self.assertRaises(ValueError):screen.validate(report(rejected),12)

    def test_no_missing_negatives_wrong_order_invalid_time_shape_or_fallback(self):
        for change in (lambda r:r["arrays"].pop(),lambda r:r["samples"].reverse(),
            lambda r:r["samples"][0].update(seconds=float("nan")),lambda r:r["samples"][0].update(validation_flags=2),
            lambda r:r["samples"][0].update(warmup=False),lambda r:r["samples"][0].update(rotate=True),
            lambda r:r["programs"][1].update(key="key0"),lambda r:r["arrays"][0].update(expected_rejection=True),
            lambda r:next(x for x in r["arrays"] if x["expected_rejection"]).update(validation_flags=0),
            lambda r:r.update(overflow_negative_checked=False)):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,12)
        for repeats in (True,0,10000):
            with self.assertRaises(ValueError):screen.validate(report(),repeats)

    def test_actual_scope_failures_stay_setup_only_and_export_remains_explicit(self):
        r=report((2,8));raw=observed(r);s=screen.observer_contract(raw,r,12)
        self.assertEqual(s["setup_compile_rejections"],2)
        self.assertEqual(sum(e["cat"]=="capability-setup-rejection" for e in audit.operator_trace(raw,r,12)["traceEvents"]),2)
        for change in (lambda v:v["events"][-1].update(ok=False),
            lambda v:next(e for e in v["events"] if not e["ok"]).update(name="ane.load_api"),
            lambda v:next(e for e in v["events"] if not e["ok"]).update(end_ns=0),
            lambda v:v["events"][-1].update(request=100000)):
            bad=copy.deepcopy(raw);change(bad);bad["counts"]=dict(Counter(e["name"] for e in bad["events"]))
            with self.assertRaises(ValueError):screen.observer_contract(bad,r,12)

    def test_independent_nominal_quantizer_zero_tiny_and_finite_overflow(self):
        for rotate in (False,True):
            for pattern in ("random-a","zero","tiny","wide"):
                x=audit.expected_input(512,32,pattern);p=packed_reference(x,rotate)
                self.assertTrue(audit.quality(x,p,rotate,pattern)["screening_passed"])
        x=audit.expected_input(512,32,"overflow")
        with self.assertRaises(ValueError):audit.quality(x,packed_reference(x,False),True,"overflow")
        p=np.zeros((513,32),dtype="<f2");p[-1]=np.inf
        self.assertTrue(audit.quality(x,p,True,"overflow")["expected_rejection"])
        np.testing.assert_array_equal(audit.expected_input(512,32,"random-a"),audit.expected_input(512,32,"random-a-restored"))

    def test_tiny_rotated_target_is_not_silently_relabelled_as_zero(self):
        x=audit.expected_input(512,32,"tiny");p=packed_reference(x,True)
        self.assertGreater(np.count_nonzero(p[:-1]),0)
        p[:-1]=0;q=audit.quality(x,p,True,"tiny")
        self.assertEqual(q["requantized_relative_l2"],1.)
        self.assertFalse(q["screening_passed"])

    def test_codes_scales_and_sparse_pointwise_error_gate(self):
        x=np.full((512,32),.125,dtype="<f2");p=packed_reference(x,False)
        for change in (lambda v:v.__setitem__((-1,0),0),lambda v:v.__setitem__((0,0),.001),lambda v:v.__setitem__((0,0),-1)):
            bad=p.copy();change(bad)
            with self.assertRaises(ValueError):audit.quality(x,bad,False,"random-a")
        bad=p.copy();bad[0,0]=120/128.;q=audit.quality(x,bad,False,"random-a")
        self.assertLess(q["requantized_relative_l2"],.02);self.assertEqual(q["pointwise_violations"],1);self.assertFalse(q["screening_passed"])

    def test_complete_packed_payload_size_hash_and_nonfinite_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"trial").mkdir();x=audit.expected_input(512,32,"random-a");p=packed_reference(x,False)
            path=root/"trial/packed.f16";path.write_bytes(p.tobytes());digest=hashlib.sha256(p.tobytes()).hexdigest()
            np.testing.assert_array_equal(p,audit.read(root,path.name,digest,513,32))
            for name,c in (("../packed.f16",513),(path.name,512)):
                with self.assertRaises(ValueError):audit.read(root,name,digest,c,32)
            p[-1]=np.inf;path.write_bytes(p.tobytes());digest=hashlib.sha256(p.tobytes()).hexdigest()
            with self.assertRaises(ValueError):audit.read(root,path.name,digest,513,32)
            self.assertTrue(np.isinf(audit.read(root,path.name,digest,513,32,True)[-1]).all())


if __name__=="__main__":unittest.main()
