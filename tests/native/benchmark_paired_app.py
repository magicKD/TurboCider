"""ABBA crossover through two persistent App job-store processes; no concurrent inference."""
import argparse,json,statistics,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['baseline','candidate','model','manifest','output']:p.add_argument('--'+name,required=True)
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);reports={}
for mode in ['gpu_ane','gpu']:
 workers={};logs=[];sequence=[]
 try:
  for label in ['baseline','candidate']:
   directory=out/mode/label;directory.mkdir(parents=True,exist_ok=True)
   log=(directory/'stderr.log').open('w');logs.append(log)
   workers[label]=subprocess.Popen([str(Path(getattr(a,label)).resolve()),a.model,str(directory),mode,a.manifest,'--interactive'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
  def run(label,measured):
   child=workers[label];child.stdin.write('run\n');child.stdin.flush()
   while True:
    line=child.stdout.readline()
    if not line:raise RuntimeError(f'{label} worker exited: {child.poll()}')
    if line.startswith('RUN '):break
   sequence.append(dict(label=label,measured=measured,seconds=float(line.split()[1]),completed_at=time.monotonic()))
  for label in ['baseline','candidate','candidate','baseline']:run(label,False)
  for _ in range(4):
   for label in ['baseline','candidate','candidate','baseline']:run(label,True)
  for child in workers.values():child.stdin.write('quit\n');child.stdin.flush()
  for label,child in workers.items():
   assert child.wait(timeout=60)==0,label
  medians={label:statistics.median(x['seconds'] for x in sequence if x['label']==label and x['measured']) for label in workers}
  reports[mode]=dict(sequence=sequence,medians=medians,change_percent=(medians['candidate']/medians['baseline']-1)*100)
  # Both variants receive the same request, seed, dtype, partitions and output image.
  hashes=[]
  for label in workers:
   data=json.loads((out/mode/label/'report.json').read_text());hashes.append({x['png_sha256'] for x in data['runs']})
  assert hashes[0]==hashes[1] and len(hashes[0])==1
  print(json.dumps({mode:medians}),flush=True)
 finally:
  for child in workers.values():
   if child.poll() is None:child.terminate();child.wait(timeout=30)
  for log in logs:log.close()
result=dict(passed=all(r['change_percent']<=3 for r in reports.values()),scope='Same Swift NativeJobStore, two resident processes, sequential ABBA requests; two priming calls then eight measured calls per version; not a cold-start measurement',reports=reports)
(out/'report.json').write_text(json.dumps(result,indent=2))
assert result['passed'],result
