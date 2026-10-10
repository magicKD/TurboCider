import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def module(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools/coreml'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sq=module('z_image_smoothquant');binding=module('qwen21_calibration_corrections')
class LoraCalibrationTest(unittest.TestCase):
 def fixture(self):
  r=np.random.default_rng(42)
  return tuple(r.normal(size=s).astype(np.float32) for s in ((8,4),(8,4),(4,8),(16,4),(16,8),(16,8)))
 def test_lora_reparameterization_and_threshold(self):
  g,u,d,x,dg,du=self.fixture();dg*=10;du*=8
  ng,nu,nd,s1,s2=sq.smooth_partial_ffn(g,u,d,[x],.5,.5,np,hidden_rows=16,corrections=[(dg,du)])
  def silu(a):return a/(1+np.exp(-np.clip(a,-80,80)))
  original=(silu(x@g.T+dg)*(x@u.T+du))@d.T
  changed=(silu((x/s1)@ng.T+dg)*((x/s1)@nu.T+du/s2))@nd.T
  np.testing.assert_allclose(changed,original,rtol=5e-5,atol=1e-4)
  scalar,_=sq.calibrate_hidden_a8(ng,nu,nd,s1,[x],.01,8,np,rows_per_sample=16,corrections=[(dg,du)],correction_s2=s2)
  wrong,_=sq.calibrate_hidden_a8(ng,nu,nd,s1,[x],.01,8,np,rows_per_sample=16)
  self.assertGreater(float(scalar),float(wrong)*2)
 def test_zero_delta_preserves_base_calibration(self):
  g,u,d,x,dg,du=self.fixture();z=np.zeros_like(dg)
  a=sq.smooth_partial_ffn(g,u,d,[x],.5,.5,np)
  b=sq.smooth_partial_ffn(g,u,d,[x],.5,.5,np,corrections=[(z,z)])
  for left,right in zip(a,b):np.testing.assert_array_equal(left,right)
 def test_malformed_delta_rejected(self):
  g,u,d,x,dg,du=self.fixture()
  for correction in ([],[(dg[:1],du)],[(dg*np.nan,du)]):
   with self.assertRaises(ValueError):sq.smooth_partial_ffn(g,u,d,[x],.5,.5,np,corrections=correction)
  with self.assertRaises(ValueError):sq.calibrate_hidden_a8(g,u,d,np.ones(4),[x],.01,8,np,corrections=[(dg,du)])
 def test_source_binding_and_tamper(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);inputs=root/'inputs';c=root/'corrections'
   (inputs/'block0').mkdir(parents=True);(c/'block0').mkdir(parents=True)
   x=inputs/'block0/step1.npy';g=c/'block0/step1.gate.npy';u=c/'block0/step1.up.npy'
   np.save(x,np.ones((16,4),np.float32));np.save(g,np.ones((16,8),np.float32));np.save(u,np.ones((16,8),np.float32))
   b=dict(schema_version=1,width=8,checkpoint_sha256='a'*64,adapter_identity='fixture-sha-strength',samples={'block0/step1.npy':dict(input_sha256=binding.digest(x),gate_sha256=binding.digest(g),up_sha256=binding.digest(u))})
   (c/'binding.json').write_text(json.dumps(b))
   pairs,_,_=binding.load_corrections(c,inputs,0,16,8,'a'*64,np);self.assertEqual(len(pairs),1)
   with self.assertRaises(ValueError):binding.load_corrections(c,inputs,0,16,8,'b'*64,np)
   np.save(x,np.zeros((16,4),np.float32))
   with self.assertRaises(ValueError):binding.load_corrections(c,inputs,0,16,8,'a'*64,np)
