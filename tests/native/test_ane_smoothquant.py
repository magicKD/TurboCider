"""Foundation-only S1 contract tests; fake stat sources and tiny CPU inputs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_s1_sidecar", ROOT / "tools/coreml/create_smoothquant_s1_sidecar.py")
S1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S1)


def fingerprint(identity):
    def string(value):
        data = value.encode()
        return struct.pack(">Q",len(data)) + data
    return hashlib.sha256(b"H" + string("tc-ane-calibration-source-v1") + b"S" + string("identity") + string(identity)).hexdigest()


@unittest.skipUnless(sys.platform == "darwin", "requires Foundation")
class AneSmoothquantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="tc-s1-contract-cpu-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name).resolve()
        cls.binary = cls.directory / "s1-cpu"
        subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-fobjc-arc",
                        str(ROOT / "native/backends/ane_smoothquant.mm"),
                        str(ROOT / "tests/native/ane_smoothquant_test.mm"),"-framework","Foundation","-o",str(cls.binary)],
                       check=True,capture_output=True,text=True,timeout=60)
        cls.checkpoint = cls.directory / "fake-checkpoint.safetensors"
        cls.checkpoint.write_text("CPU stat identity only; never read as model weights")
        cls.capture = cls.directory / "capture"
        subprocess.run([str(cls.binary),"capture",str(cls.checkpoint),str(cls.capture)],
                       check=True,capture_output=True,text=True,timeout=15)
        cls.sidecar = S1.build_sidecar(cls.capture / "capture.json",cls.capture / "weight_stats.json")

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory(prefix="tc-s1-case-cpu-")
        self.addCleanup(self.temporary_case.cleanup)
        self.root = Path(self.temporary_case.name).resolve()
        self.path = self.root / "s1.json"
        S1.write_sidecar(self.path,self.sidecar)

    def run_load(self,path=None,mutation=None,checkpoint=None,success=True):
        args = [str(self.binary),"load",str(checkpoint or self.checkpoint),str(path or self.path)]
        if mutation:
            args.append(mutation)
        result = subprocess.run(args,capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0 if success else 1,result.stdout + result.stderr)
        return result.stdout.splitlines()

    def reject(self,mutate):
        value = copy.deepcopy(self.sidecar)
        mutate(value)
        self.path.write_text(json.dumps(value,allow_nan=True))
        self.run_load(success=False)

    def test_full_capture_and_candidate_digest(self):
        capture = json.loads((self.capture / "capture.json").read_text())
        self.assertEqual(capture["selection"],{"layers":list(range(32)),"steps":[0,3,5],"rows_per_point":8})
        self.assertEqual(capture["used"],{"input_bytes":12288,"statistics_bytes":4096})
        self.assertEqual(capture["budgets"],{"input_bytes":8*1024*1024,"statistics_bytes":2*1024*1024,
                         "total_bytes":24*1024*1024})
        self.assertEqual(self.sidecar["scope"],"full-model-s1")
        self.assertFalse(self.sidecar["quantization_qualified"])
        output = self.run_load()
        self.assertEqual(output[:4],[hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                   self.sidecar["calibration_sha256"],"32","0.5"])
        self.assertAlmostEqual(float(output[4]),2.,places=6)
        first = output[0]
        self.path.write_text(json.dumps(self.sidecar,separators=(",",":")))
        self.assertNotEqual(self.run_load()[0],first)

    def test_stat_and_adapter_fingerprints_match_canonical_oracle(self):
        info = self.checkpoint.stat()
        name = str(self.checkpoint.resolve())
        identity = f":{len(name.encode())}:{name}:{info.st_dev}:{info.st_ino}:{info.st_size}:" \
            f"{info.st_mtime_ns//10**9}:{info.st_mtime_ns%10**9}:{info.st_ctime_ns//10**9}:{info.st_ctime_ns%10**9}"
        output = subprocess.run([str(self.binary),"fingerprints",str(self.checkpoint)],
                                check=True,capture_output=True,text=True,timeout=5).stdout.splitlines()
        self.assertEqual(output,[fingerprint(identity),fingerprint("existing-bound-adapter-identity:0"),
                                 fingerprint("existing-bound-adapter-identity:1")])

    def test_optin_model_shape_recipe_lora_reference_bindings(self):
        for mutation in ("no-opt-in","model","hidden","steps","reference-count","reference-size","recipe",
                         "backend","adapter-order","adapter-strength","adapter-count","partial","alpha"):
            with self.subTest(mutation=mutation):
                self.run_load(mutation=mutation,success=False)
        changed = self.root / "different-source.safetensors"
        shutil.copyfile(self.checkpoint,changed)
        self.run_load(checkpoint=changed,success=False)

    def test_incomplete_and_qualification_claims_rejected(self):
        mutations = [lambda v:v.update(scope="partial-diagnostic"),lambda v:v.update(complete=False),
                     lambda v:v.update(experimental_only=False),lambda v:v.update(schema_version=1),
                     lambda v:v.update(runtime_applied=True),lambda v:v.update(quantization_qualified=True),
                     lambda v:v.update(performance_qualified=True),lambda v:v.update(pending=[]),
                     lambda v:v.pop("binding"),lambda v:v["binding"].pop("identity_kind"),
                     lambda v:v["binding"].update(actual_execution="unknown\x00hidden"),
                     lambda v:v["selection"]["layers"].pop(),lambda v:v["layers"].pop(),
                     lambda v:v["layers"].__setitem__(31,copy.deepcopy(v["layers"][0])),
                     lambda v:v["layers"][0]["contexts"].pop(),
                     lambda v:v["layers"][0].update(weight_source_fingerprint="0"*64),
                     lambda v:v["cpu_replay"].update(scope="real quantized weights passed"),
                     lambda v:v.update(extra_claim="qualified")]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.reject(mutation)

    def test_scales_types_and_row_context_rejected(self):
        for value in (0,-1,1/32,17,float("nan"),float("inf"),True,"1"):
            with self.subTest(scale=value):
                self.reject(lambda v:v["layers"][0]["s1"].__setitem__(0,value))
        for mutation in (lambda v:v["layers"][0]["s1"].pop(),
                         lambda v:v["layers"][0].update(hidden=8.5),
                         lambda v:v["binding"].update(width=True),
                         lambda v:v["selection"].update(rows_per_point=64),
                         lambda v:v["selection"].update(steps=[0,3,3]),
                         lambda v:v["layers"][0]["contexts"][1].update(step=0),
                         lambda v:v["layers"][0]["contexts"][0]["regions"][1].update(begin=3),
                         lambda v:v["layers"][0]["contexts"][0]["regions"][-1].update(end=16),
                         lambda v:v["layers"][0]["contexts"][0].update(phase="wrong")):
            self.reject(mutation)

    def test_duplicate_keys_escaped_keys_and_trailing_content_rejected(self):
        text = json.dumps(self.sidecar)
        for malformed in ('{"schema_version":2,' + text[1:],
                          text.replace('"hidden": 8','"hidden": 8, "\\u0068idden": 8',1),
                          text.replace('"model_id": "qwen-image-2.1"',
                                       '"model_id": "wrong", "model_id": "qwen-image-2.1"',1),
                          text + '{}'):
            self.path.write_text(malformed)
            self.run_load(success=False)

    def test_symlink_fifo_oversize_and_empty_files_rejected(self):
        symlink = self.root / "link.json"
        symlink.symlink_to(self.path)
        self.run_load(path=symlink,success=False)
        fifo = self.root / "pipe.json"
        os.mkfifo(fifo)
        self.run_load(path=fifo,success=False)
        self.run_load(path=self.root,success=False)
        self.path.write_bytes(b"")
        self.run_load(success=False)
        with self.path.open("wb") as destination:
            destination.truncate(4*1024*1024+1)
        self.run_load(success=False)

    def test_partial_diagnostic_export_remains_explicit(self):
        partial = copy.deepcopy(self.sidecar)
        partial["selection"]["layers"] = [0,15,31]
        partial["layers"] = [partial["layers"][i] for i in (0,15,31)]
        partial["scope"] = "partial-diagnostic"
        path = self.root / "partial.json"
        S1.write_sidecar(path,partial)
        self.run_load(path=path,success=False)


if __name__ == "__main__":
    unittest.main()
