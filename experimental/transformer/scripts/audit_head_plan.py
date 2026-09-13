import json,shutil
from pathlib import Path
from heads import head_model
from inspect_plan import inspect
import coremltools as ct
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'notes/raw/heads'
for m,h,count in [(64,256,1),(256,1024,4),(1024,1024,4),(1024,1024,8)]:
    cache=ROOT/'cache'/f'head_plan_{m}_{h}_{count}';cache.mkdir(parents=True,exist_ok=True);package=cache/'head.mlpackage';head_model(m,h,h//64,count).save(str(package));p=Path(ct.models.utils.compile_model(str(package)));shutil.move(str(p),cache/'head.mlmodelc')
    (out/f'{m}_{h}_{count}_plan.json').write_text(json.dumps(inspect(cache/'head.mlmodelc'),indent=2))
    size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file());shutil.rmtree(cache)
    with (out/'cleanup.jsonl').open('a') as log:log.write(json.dumps({'owned_path':str(cache),'logical_bytes_removed':size,'exists_after':cache.exists()})+'\n')
