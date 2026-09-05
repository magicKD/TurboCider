"""Offline, unmodified flux2-engine comparison including output export wall time."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,json,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--mflux',required=True);p.add_argument('--model',required=True);p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--runs',type=int,default=5);p.add_argument('--manifest');p.add_argument('--bridge');a=p.parse_args()
sys.path.insert(0,str(Path(a.engine)/'src'))
from flux2_engine.config import EngineConfig,GenerationRequest
from flux2_engine.engine import Flux2Engine
r=json.loads(Path(a.request).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
config=EngineConfig(model_path=Path(a.model),mflux_root=Path(a.mflux),mode='mlx-ane' if a.manifest else 'mlx',persistent=True,precision='bf16',clear_mlx_cache_between_requests=False,ane_manifests=(Path(a.manifest),) if a.manifest else (),bridge_dir=Path(a.bridge) if a.bridge else None)
engine=Flux2Engine(config);load=time.perf_counter()-started
reports=[]
for i in range(a.runs):
 request=GenerationRequest(prompt=r['prompt'],width=r['width'],height=r['height'],steps=r['steps'],seed=r['seed'],dynamic_text_length=r.get('dynamic_text',True),output=out/f'{i}.png')
 started=time.perf_counter();result=engine.generate(request);wall=time.perf_counter()-started
 reports.append({'run':i,'request_wall_including_export':wall,'metrics':result.metrics.as_dict() if hasattr(result.metrics,'as_dict') else vars(result.metrics)})
 (out/'report.json').write_text(json.dumps({'engine':'flux2-engine','constructor_seconds':load,'config':{'precision':'bf16','clear_mlx_cache_between_requests':False,'mode':config.mode.value,'manifest':a.manifest},'request':r,'runs':reports},indent=2,default=str))
engine.close()
