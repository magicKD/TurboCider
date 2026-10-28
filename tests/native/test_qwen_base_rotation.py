"""CPU-only bounded H512 emitter/oracle/timing and setup-failure contracts."""
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
import rotation_collect as screen
import rotation_audit as audit
from test_qwen_base_profiling import fixture


def report(rejected=(),legacy=False):
    arms=screen.ARMS[:8] if legacy else screen.ARMS;blocks=screen.BLOCKS[:8] if legacy else screen.BLOCKS;k=len(arms)
    programs=[];arrays=[];samples=[]
    for g,(m,c) in enumerate(((32,512),(1024,6144))):
        for i,(name,block) in enumerate(zip(arms,blocks)):
            n=g*k+i;programs.append(dict(ordinal=n,arm=name,block=block,rows=m,channels=c,
                status="compile-rejected" if n in rejected else "ready",key=f"key{n}",error="explicit compile failure" if n in rejected else ""))
        ready=[i for i in range(k) if g*k+i not in rejected]
        for p,pattern in enumerate(screen.PATTERNS):
            for i in ready:arrays.append(dict(ordinal=g*k+i,scope=(g*k+i)*1000+p+1,arm=arms[i],pattern=pattern,rows=m,channels=c))
        for r in range(-2,12):
            for j in range(len(ready)):
                i=ready[(j+r+len(ready))%len(ready)];n=g*k+i
                samples.append(dict(ordinal=n,scope=n*1000+12+r,arm=arms[i],rows=m,channels=c,rep=r,warmup=r<0,input_variant=bool(r&1),seconds=.001+i*.0001))
    return dict(status="complete",synthetic_only=True,model_weights_loaded=False,complete_request=False,image_export=False,
                repeats=12,warmups=2,dtype="float16-le",layout="channels_tokens",source_unchanged=True,identity_bits_exact=True,
                same_program_restoration_exact=True,programs=programs,arrays=arrays,samples=samples)


def observed(r):
    raw,_=fixture();base=next(x for x in raw["events"] if x["name"]=="ane.evaluate_api")
    raw["events"]=[x for x in raw["events"] if x["request"]==-2]
    for p in r["programs"]:
        n=p["ordinal"]
        raw["events"].append(dict(base,name="ane.compile_api",request=n*1000,model=n+1,ok=p["status"]=="ready"))
        if p["status"]!="ready":continue
        for i in list(range(1,5))+list(range(10,24)):
            raw["events"].append(dict(base,request=n*1000+i,model=n+1,ok=True))
    raw["counts"]=dict(Counter(x["name"] for x in raw["events"]))
    return raw


class RotationTests(unittest.TestCase):
    def test_native_emitter_serialization_matches_original_and_factor_coefficients(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-rotation-") as directory:
            binary=Path(directory)/"rotation"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-I",str(ROOT/"native/core"),
                str(ROOT/"tests/native/qwen_rotation_mil_test.cpp"),str(ROOT/"native/backends/private/ane_mil.cpp"),"-o",str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)

    def test_complete_pattern_and_timing_contract_excludes_rejected_programs(self):
        for rejected in ((),(2,3,4,5)):
            r=report(rejected);value=screen.validate(r,12)
            self.assertEqual(value["total_operator_calls"],(20-len(rejected))*18)
            screen.observer_contract(observed(r),r,12)
        with self.assertRaises(ValueError):screen.validate(report((0,)),12)
        with self.assertRaises(ValueError):screen.validate(report((11,)),12)
        screen.validate(report(legacy=True),12,legacy=True)
        with self.assertRaises(ValueError):screen.validate(report(legacy=True),12)
        for repeats in (0,13,100000,True):
            with self.assertRaises(ValueError):screen.validate(report(),repeats)

    def test_missing_output_swapped_order_bad_times_and_wrong_shape_rejected(self):
        for change in (lambda r:r["arrays"].pop(),lambda r:r["samples"].reverse(),
            lambda r:r["samples"][5].update(seconds=float("nan")),lambda r:r["samples"][5].update(warmup=False),
            lambda r:r["samples"][5].update(input_variant=True),lambda r:r["samples"][5].update(rows=1024),
            lambda r:r["programs"][2].update(key="key1"),lambda r:r["arrays"][0].update(scope=0),
            lambda r:r.update(source_unchanged=False)):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,12)

    def test_only_declared_setup_compile_failure_can_be_observed_or_exported(self):
        r=report((2,3,4,5));raw=observed(r);value=screen.observer_contract(raw,r,12)
        self.assertEqual(value["setup_compile_rejections"],4)
        exported=audit.operator_trace(raw,r,12)
        self.assertEqual(sum(x["cat"]=="capability-setup-rejection" for x in exported["traceEvents"]),4)
        for change in (lambda x:x["events"][-1].update(ok=False),
            lambda x:next(e for e in x["events"] if e["ok"] is False).update(name="ane.load_api"),
            lambda x:next(e for e in x["events"] if e["ok"] is False).update(ok=None),
            lambda x:next(e for e in x["events"] if e["ok"] is False).update(end_ns=0)):
            bad=copy.deepcopy(raw);change(bad);bad["counts"]=dict(Counter(x["name"] for x in bad["events"]))
            with self.assertRaises(ValueError):screen.observer_contract(bad,r,12)

    def test_full_f64_butterfly_matches_independent_dense_signed_matrix(self):
        x=audit.expected_input(512,32,"random-a")
        out=np.arange(512)[:,None];inputs=np.arange(512)[None,:]
        parity=np.array([int(i).bit_count()&1 for i in (out&inputs).reshape(-1)]).reshape(512,512)
        matrix=np.where(parity,-1,1).astype(np.int64)*audit.signs().astype(np.int64)[None,:]
        # The fixed source grid is exactly integer *2^-18, even after FP16
        # rounding. Integer dense dot is an independent exact oracle and does
        # not inherit spurious FP flags from the platform BLAS implementation.
        scaled=np.rint(x.astype(np.float64)*262144).astype(np.int64)
        dense=(matrix@scaled).astype(np.float64)*(181./1073741824.)
        np.testing.assert_array_equal(audit.oracle(x),dense)
        basis=audit.expected_input(512,32,"signed-basis");y=audit.oracle(basis).astype("<f2")
        self.assertTrue(audit.quality(audit.oracle(basis),y)["screening_passed"])

    def test_full_payload_length_hash_paths_and_finite_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"trial").mkdir();p=root/"trial/source.f16"
            a=audit.expected_input(512,32,"random-a");p.write_bytes(a.tobytes());digest=hashlib.sha256(a.tobytes()).hexdigest()
            np.testing.assert_array_equal(a,audit.read_array(root,p.name,digest,512,32))
            for name,d,c,m in (("../source.f16",digest,512,32),(p.name,"0"*64,512,32),(p.name,digest,512,1024)):
                with self.assertRaises(ValueError):audit.read_array(root,name,d,c,m)
            p.write_bytes(np.full_like(a,np.nan).tobytes());digest=hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):audit.read_array(root,p.name,digest,512,32)

    def test_pointwise_gate_detects_sparse_error_hidden_by_relative_l2(self):
        x=np.ones((512,32));y=x.astype("<f2");y[-1,-1]=1.0625
        q=audit.quality(x,y)
        self.assertLess(q["relative_l2"],.002)
        self.assertEqual(q["pointwise_violations"],1)
        self.assertFalse(q["screening_passed"])
        np.testing.assert_array_equal(audit.expected_input(512,32,"random-a"),audit.expected_input(512,32,"random-a-restored"))


if __name__=="__main__":unittest.main()
