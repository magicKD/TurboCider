"""Sequential AB/BA benchmark; never run competing GPU jobs concurrently."""
import argparse,json,os,statistics,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--engine',required=True);p.add_argument('--mflux',required=True);p.add_argument('--manifest',required=True);p.add_argument('--bridge',required=True);p.add_argument('--output',required=True);p.add_argument('--runs',type=int,default=7);p.add_argument('--modes',nargs='+',choices=['gpu','hybrid'],default=['gpu','hybrid']);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
summary={'method':'Sequential AB then BA; persistent session, first request excluded from warm statistics; wall includes PNG export. No tensor dumps. OS file/Core ML disk caches not flushed.','rounds':[]}
for mode in a.modes:
 request={'prompt':'A red fox sitting in a snowy forest, soft morning light, detailed photography.','width':512,'height':512,'steps':4,'seed':42,'dynamic_text':True,'residency':'resident'}
 if mode=='hybrid':request.update(execution='gpu_ane',allow_approximation=True,ane_manifest=a.manifest)
 rp=out/(mode+'.json');rp.write_text(json.dumps(request))
 for round_index,order in enumerate([['original','native'],['native','original']]):
  for engine in order:
   destination=out/f'{mode}-{round_index}-{engine}'
   cmd=[sys.executable,str(root/'tools/native'/('benchmark_native.py' if engine=='native' else 'benchmark_flux2_engine.py')),'--model',a.model,'--request',str(rp),'--output',str(destination),'--runs',str(a.runs)]
   if engine=='native':cmd+=['--library',str(root/'build/native/libturbocider.dylib')]
   else:
    cmd+=['--engine',a.engine,'--mflux',a.mflux]
    if mode=='hybrid':cmd+=['--manifest',a.manifest,'--bridge',a.bridge]
   with (out/f'{mode}-{round_index}-{engine}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
   report=json.loads((destination/'report.json').read_text());walls=[r['request_wall_including_export'] for r in report['runs']];warm=sorted(walls[1:])
   row={'mode':mode,'round':round_index,'engine':engine,'constructor_seconds':report['constructor_seconds'],'first_request_seconds':walls[0],'warm_median_seconds':statistics.median(warm),'warm_p95_nearest_rank_seconds':warm[max(0,__import__('math').ceil(len(warm)*.95)-1)],'warm_seconds':walls[1:]}
   summary['rounds'].append(row);(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(row),flush=True)
summary['comparison']=[]
for mode in a.modes:
 values={e:[v for row in summary['rounds'] if row['engine']==e and row['mode']==mode for v in row['warm_seconds']] for e in ['native','original']}
 n,o=[statistics.median(values[e]) for e in ['native','original']]
 summary['comparison'].append({'mode':mode,'native_warm_median_seconds':n,'original_warm_median_seconds':o,'native_overhead_percent':100*(n/o-1),'gate':'no more than 5% warm median regression','passed':n<=o*1.05})
(out/'summary.json').write_text(json.dumps(summary,indent=2))
if not all(x['passed'] for x in summary['comparison']):sys.exit(1)
