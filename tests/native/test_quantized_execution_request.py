"""Native JSON/plan contract for the experimental bounded GGUF request."""
import copy
import ctypes as C
import json
import os
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]

@unittest.skipUnless(os.environ.get("TC_QUANTIZED_TEST_LIBRARY"),"explicit freshly built library required")
class QuantizedRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib=C.CDLL(str(Path(os.environ["TC_QUANTIZED_TEST_LIBRARY"]).resolve()))
        cls.lib.tc_plan_json.argtypes=[C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        cls.lib.tc_string_free.argtypes=[C.c_void_p]

    def request(self,config):
        return {"schema_version":2,"model":"z-image-turbo-gguf","operation":"image.generate",
            "inputs":[{"kind":"text","role":"prompt","text":"test"}],
            "outputs":[{"kind":"image","path":"test.png","width":512,"height":512,"audio":False}],
            "sampling":{"steps":4,"seed":42},"execution":{"policy":"gpu","quantized_execution":config}}

    def plan(self,request):
        value,error=C.c_void_p(),C.c_void_p()
        code=self.lib.tc_plan_json(json.dumps(request).encode(),C.byref(value),C.byref(error))
        def take(p):
            if not p.value:return None
            text=C.string_at(p).decode();self.lib.tc_string_free(p);return text
        raw,message=take(value),take(error)
        return code,json.loads(raw) if raw else None,message

    def test_defaults_and_explicit_lookahead(self):
        for p in (None,0,1,2):
            config={"schema_version":1,"enabled":True}
            if p is not None:config["prefetch_layers"]=p
            status,value,error=self.plan(self.request(config))
            self.assertEqual(status,0,error)
            self.assertFalse(value["executable"])
            self.assertEqual(value["quantized_execution"]["prefetch_layers"],1 if p is None else p)
            self.assertEqual(value["quantized_execution_qualification"],"experimental-unqualified")
        for mode in ("packed_resident", "packed_streamed"):
            status,value,error=self.plan(self.request({"schema_version":1,"enabled":True,
                "prefetch_layers":2,"source_residency":mode}))
            self.assertEqual(status,0,error)
            self.assertEqual(value["quantized_execution"]["source_residency"],mode)

    def test_profiles_have_separate_math_and_mode_identity(self):
        for profile in ("z-source-mixed-f16-v1","z-source-exact-f32-v1","z-source-native-affine-v1",
                        "z-mlx-compat-affine-v1","z-mlx-compat-f16-v1","z-mlx-compat-f32-v1"):
            config={"schema_version":1,"enabled":True,"precision_profile":profile}
            status,value,error=self.plan(self.request(config));self.assertEqual(status,0,error)
            self.assertEqual(value["quantized_execution"]["precision_profile"],profile)
            mode="bounded_packed" if "affine" in profile else "bounded_dequant"
            self.assertEqual(value["quantized_execution"]["mode"],mode)
            config["mode"]="bounded_dequant" if mode=="bounded_packed" else "bounded_packed"
            self.assertNotEqual(self.plan(self.request(config))[0],0)

    def test_disabled_is_not_implicitly_enabled(self):
        status,value,error=self.plan(self.request({"schema_version":1,"enabled":False}))
        self.assertEqual(status,0,error)
        self.assertFalse(value["quantized_execution"]["enabled"])
        for extra in ({"prefetch_layers":0},{"allow_requantization":False},{"mode":"bounded_dequant"}):
            config={"schema_version":1,"enabled":False,**extra}
            self.assertNotEqual(self.plan(self.request(config))[0],0)

    def test_raw_gpu_ready_is_not_cpu_decoded_ready(self):
        config={"schema_version":1,"enabled":True,"precision_profile":"z-raw-gpu-affine-f16-v1","source_residency":"packed_streamed"}
        request=self.request(config);request["execution"]["allow_approximation"]=True;request["parameters"]={"compile_gpu":True}
        status,value,error=self.plan(request);self.assertEqual(status,0,error)
        self.assertFalse(value["executable"])
        self.assertEqual(value["quantized_execution"]["mode"],"bounded_raw_packed")
        self.assertEqual(value["quantized_execution"]["decode_backend"],"cpu_io_gpu_affine")
        for field in ("mode","backend","residency","approximation","compile"):
            bad=copy.deepcopy(request)
            q=bad["execution"]["quantized_execution"]
            if field=="mode":q["mode"]="bounded_packed"
            if field=="backend":q["decode_backend"]="cpu_simd"
            if field=="residency":q["source_residency"]="packed_resident"
            if field=="approximation":bad["execution"]["allow_approximation"]=False
            if field=="compile":bad["parameters"]["compile_gpu"]=False
            with self.subTest(field=field):self.assertNotEqual(self.plan(bad)[0],0)

    def test_dense_bf16_is_explicit_approximate_streamed_compiled_candidate(self):
        config={"schema_version":1,"enabled":True,"precision_profile":"z-dense-bf16-v1",
                "source_residency":"packed_streamed"}
        request=self.request(config)
        request["execution"]["allow_approximation"]=True
        request["parameters"]={"compile_gpu":True}
        status,value,error=self.plan(request)
        self.assertEqual(status,0,error)
        self.assertFalse(value["executable"])
        self.assertEqual(value["quantized_execution"]["precision_profile"],"z-dense-bf16-v1")
        for field in ("authorization","compile","source","default_source","mode"):
            bad=copy.deepcopy(request)
            if field=="authorization":bad["execution"]["allow_approximation"]=False
            if field=="compile":bad["parameters"]["compile_gpu"]=False
            if field=="source":bad["execution"]["quantized_execution"]["source_residency"]="packed_resident"
            if field=="default_source":bad["execution"]["quantized_execution"].pop("source_residency")
            if field=="mode":bad["execution"]["quantized_execution"]["mode"]="bounded_packed"
            with self.subTest(field=field):self.assertNotEqual(self.plan(bad)[0],0)

    def test_unknown_wrong_type_and_unimplemented_options_rejected(self):
        cases=[{}, {"schema_version":1}, {"enabled":True},
            {"schema_version":1,"enabled":1}, {"schema_version":"1","enabled":True}]
        base={"schema_version":1,"enabled":True}
        for field,value in [("schema_version",2),("prefetch_layers",True),("prefetch_layers",1.5),
                ("prefetch_layers",-1),("prefetch_layers",3),("unknown",0),("allow_requantization",True),
                ("persistent_dense_layers",1),("ane_compute","w8a8"),("source_residency","unknown"),
                ("precision_profile","z-dense-bf16-v1"),("mode","packed_direct")]:
            cases.append({**base,field:value})
        for config in cases:
            with self.subTest(config=config):self.assertNotEqual(self.plan(self.request(config))[0],0)

    def test_incompatible_request_options_rejected(self):
        base=self.request({"schema_version":1,"enabled":True})
        for update in ({"residency":"resident"},{"memory_budget_bytes":1<<30},{"streaming_offload":False},
                       {"quantized_cache":"test"},{"ane_manifest":"test"},{"encoder_ane_manifest":"test"},
                       {"memory_constrained":{"enabled":True,"limit_bytes":16<<30}}):
            request=copy.deepcopy(base);request["execution"].update(update)
            with self.subTest(update=update):self.assertNotEqual(self.plan(request)[0],0)
        request=copy.deepcopy(base);request["parameters"]={"compile_gpu":True}
        self.assertNotEqual(self.plan(request)[0],0)

if __name__=="__main__":unittest.main()
