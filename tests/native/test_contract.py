import ctypes as C
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
lib = C.CDLL(str(ROOT / 'build/native/libturbocider.dylib'))
lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
lib.tc_string_free.argtypes = [C.c_void_p]
lib.tc_engine_create.argtypes = [C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_free.argtypes = [C.c_void_p]

def consume(p):
    if not p.value:return None
    value=C.string_at(p).decode();lib.tc_string_free(p);return value

def plan(r):
    out,err=C.c_void_p(),C.c_void_p()
    status=lib.tc_plan_json(json.dumps(r).encode(),C.byref(out),C.byref(err))
    a,b=consume(out),consume(err)
    return status,json.loads(a) if a else None,b

class ContractTests(unittest.TestCase):
    def test_flux(self):
        code,p,error=plan({'width':512,'height':512})
        self.assertEqual(code,0,error);self.assertTrue(p['executable'])
        self.assertEqual(p['decoded_shape'],[512,512,1])
        self.assertEqual([x['id'] for x in p['stages']],['text_encode','denoise','vae_decode','export'])
    def test_ltx_schedule_and_shape(self):
        code,p,error=plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11})
        self.assertEqual(code,0,error);self.assertFalse(p['executable'])
        self.assertEqual(p['decoded_shape'],[704,448,97]);self.assertEqual(p['validation'],'weights_pending')
        stages={s['id']:s for s in p['stages']}
        self.assertEqual(stages['av_stage1']['iterations'],8);self.assertEqual(stages['av_stage2']['iterations'],3)
        self.assertEqual(stages['mux']['dependencies'],['video_vae','audio_vae_vocoder'])
    def test_h3_execution_and_validation_are_separate(self):
        code,p,error=plan({'model':'minimax-h3-turbo','frames':22,'width':512,'height':512,'steps':4})
        self.assertEqual(code,0,error);self.assertFalse(p['executable']);self.assertEqual(p['weight_validation'],'pending')
    def test_invalid_requests(self):
        requests=[{'width':0},{'width':513},{'height':4096},{'width':'512'},{'width':True},{'steps':0},{'steps':51},{'seed':-1},{'frames':2},{'execution':'gpu_ane'},{'model':'flux2-9b'},{'dynamic_text':'yes'},{'schema_version':2},{'engine_env':{}},{'mode':'image_edit'},{'model':'ltx-2.5-distilled','steps':4},{'model':'ltx-2.5-distilled','steps':11,'frames':96}]
        for r in requests:
            with self.subTest(r=r):
                code,p,error=plan(r);self.assertNotEqual(code,0);self.assertIsNone(p);self.assertTrue(error)
    def test_invalid_json(self):
        for raw in [b'[]',b'null',b'nope',None]:
            out,err=C.c_void_p(),C.c_void_p();self.assertNotEqual(lib.tc_plan_json(raw,C.byref(out),C.byref(err)),0)
            self.assertIsNone(consume(out));self.assertTrue(consume(err))
    def test_model_missing(self):
        e,err=C.c_void_p(),C.c_void_p()
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(lib.tc_engine_create(d.encode(),C.byref(e),C.byref(err)),0)
            self.assertIsNone(e.value);self.assertTrue(consume(err))
    def test_multimodal_schema2(self):
        base={'schema_version':2,'model':'flux2-klein-4b','operation':'image.edit','inputs':[{'kind':'text','role':'prompt','text':'编辑图像'},{'kind':'image','role':'reference','path':'/tmp/reference.png'}],'outputs':[{'kind':'image','path':'/tmp/out.png','width':256,'height':256}]}
        code,p,error=plan(base);self.assertEqual(code,0,error)
        stages={s['id']:s for s in p['stages']};self.assertIn('image_encode',stages['denoise']['dependencies'])
        for change in [{'operation':'image.generate'},{'outputs':[{'kind':'video'}]},{'inputs':base['inputs']+[base['inputs'][0]]},{'execution':{'residency':'streamed'}},{'inputs':[{'kind':'image','role':'reference','path':'/tmp/ref','strength':1.1}]}]:
            self.assertNotEqual(plan({**base,**change})[0],0)
    def test_disabled_profile_and_hardware_guard(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'profile.json'
            path.write_text(json.dumps({'schema_version':1,'enabled':False}))
            self.assertEqual(plan({'profile':str(path)})[0],0)
            self.assertNotEqual(plan({'profile':str(path),'execution':'gpu_ane'})[0],0)
            path.write_text(json.dumps({'schema_version':1,'enabled':True,'match':{'gpu_name':'Wrong GPU','memory_bytes':1},'models':{}}))
            self.assertNotEqual(plan({'profile':str(path)})[0],0)
    def test_abi(self): self.assertEqual(lib.tc_abi_version(),1)

if __name__=='__main__':unittest.main(verbosity=2)
