"""Small random-weight Gemma4 parity fixture. No model downloads."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16');a=p.parse_args()
sys.path.insert(0,str(Path(a.reference)/'packages/ltx-core-mlx/src'))
# Load only this component; package __init__ eagerly imports unrelated full
# pipeline dependencies. No reference math is patched or replaced.
import types
base=Path(a.reference)/'packages/ltx-core-mlx/src'
for name in ['ltx_core_mlx','ltx_core_mlx.text_encoders','ltx_core_mlx.text_encoders.gemma','ltx_core_mlx.utils']:
 module=types.ModuleType(name);module.__path__=[str(base/Path(*name.split('.')))];sys.modules[name]=module
import mlx.core as mx
from mlx.utils import tree_flatten,tree_unflatten
from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextModel
from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig
c={'model_type':'gemma4_unified','text_config':{'model_type':'gemma4_unified_text','attention_k_eq_v':True,'hidden_size':32,'num_hidden_layers':4,'num_attention_heads':4,'head_dim':8,'global_head_dim':16,'num_key_value_heads':2,'num_global_key_value_heads':1,'intermediate_size':64,'vocab_size':128,'rms_norm_eps':1e-6,'sliding_window':3,'pad_token_id':0,'layer_types':['sliding_attention','full_attention']*2,'rope_parameters':{'sliding_attention':{'rope_type':'default','rope_theta':10000.},'full_attention':{'rope_type':'proportional','rope_theta':1000000.,'partial_rotary_factor':.25}}}}
mx.random.seed(37);model=Gemma4TextModel(Gemma4TextConfig.from_text_encoder_config(c));dtype=getattr(mx,a.dtype)
weights={k:v.astype(dtype) for k,v in tree_flatten(model.parameters())};model.update(tree_unflatten(list(weights.items())))
ids=mx.array([[0,0,7,13,19,31,11,49]]);mask=mx.array([[0,0,1,1,1,1,1,1]])
raw=model(ids,mask);raw[-1]=model.norm(raw[-1]);mx.eval(raw)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(c,indent=2));mx.save_safetensors(str(out/'weights.safetensors'),weights);mx.save_safetensors(str(out/'input.safetensors'),{'ids':ids,'mask':mask})
ref=out/'reference';ref.mkdir(exist_ok=True)
for i,x in enumerate(raw):mx.save_safetensors(str(ref/f'layer_{i}.safetensors'),{'tensor':x})
print(json.dumps({'layers':len(raw),'dtype':a.dtype,'weights':'synthetic random only'}))
