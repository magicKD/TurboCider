"""Exercise durable queue, cancellation and model reuse via local IPC."""
import argparse,json,os,socket,subprocess,time,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
state=out/'jobs';state.mkdir(exist_ok=True);sock='/tmp/tc-'+uuid.uuid4().hex[:10]+'.sock'
interrupted_id=str(uuid.uuid4());(state/(interrupted_id+'.json')).write_text(json.dumps({'schema_version':1,'id':interrupted_id,'state':'running','created_at':0,'request':{}}))
log=(out/'service.log').open('w');process=subprocess.Popen([str(root/'build/native/turbocider'),'serve',sock,str(state)],stdout=log,stderr=log)
def rpc(value):
 with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
  client.settimeout(10);client.connect(sock);client.sendall(json.dumps(value).encode()+b'\n');data=b''
  while b'\n' not in data:data+=client.recv(65536)
  result=json.loads(data)
  if not result['ok']:raise RuntimeError(result['error'])
  return result['result']
def await_state(id,terminal=True):
 deadline=time.monotonic()+90
 while time.monotonic()<deadline:
  value=rpc({'action':'status','id':id})
  if (terminal and value['state'] in ['succeeded','failed','cancelled','interrupted']) or (not terminal and value.get('progress')):return value
  time.sleep(.1)
 raise TimeoutError(id)
checks=[]
try:
 for _ in range(100):
  if Path(sock).exists():
   try:rpc({'action':'doctor'});break
   except (OSError,RuntimeError):pass
  if process.poll() is not None:raise RuntimeError('service exited')
  time.sleep(.1)
 assert rpc({'action':'status','id':interrupted_id})['state']=='interrupted';checks.append('restart_marks_incomplete_interrupted')
 assert len(rpc({'action':'models'})['models'])==3;checks.append('registered_models')
 request={'model':'flux2-klein-4b','prompt':'A red fox sitting in a snowy forest, soft morning light, detailed photography.','width':512,'height':512,'steps':4,'seed':42,'output':str(out/'first.png')}
 def submit(r):return rpc({'action':'submit','model_path':a.model,'request':r})['id']
 first=submit(request)
 await_state(first,False)
 conflict_file=out/'conflict.json';conflict_file.write_text(json.dumps({**request,'output':str(out/'conflict.png')}))
 conflict=subprocess.run([str(root/'build/native/turbocider'),'generate',a.model,str(conflict_file)],capture_output=True,text=True)
 assert conflict.returncode!=0 and 'busy' in conflict.stderr,conflict.stderr
 assert not (out/'conflict.png').exists();checks.append('cross_process_gpu_lease')
 queued=submit({**request,'output':str(out/'cancelled.png')});rpc({'action':'cancel','id':queued})
 assert await_state(queued)['state']=='cancelled';checks.append('queued_cancel')
 result=await_state(first);assert result['state']=='succeeded',result
 assert Path(request['output']).is_file();checks.append('actual_flux_generation')
 second=submit({**request,'output':str(out/'warm.png')});warm=await_state(second);assert warm['state']=='succeeded',warm
 assert warm['result']['prompt_cache_hit'];checks.append('persistent_model_and_prompt_reuse')
 third=submit({**request,'output':str(out/'active-cancelled.png')});await_state(third,False);rpc({'action':'cancel','id':third})
 assert await_state(third)['state']=='cancelled';assert not (out/'active-cancelled.png').exists();checks.append('active_cancel_without_output')
 page=rpc({'action':'jobs','limit':2});assert len(page['jobs'])==2 and page['next_offset']==2;checks.append('history_pagination')
 process.kill();process.wait(timeout=10)
 process=subprocess.Popen([str(root/'build/native/turbocider'),'serve',sock,str(state)],stdout=log,stderr=log)
 for _ in range(100):
  try:
   rpc({'action':'doctor'});break
  except (OSError,RuntimeError):time.sleep(.1)
 assert rpc({'action':'status','id':second})['state']=='succeeded';checks.append('crash_stale_socket_recovery')
 (out/'report.json').write_text(json.dumps({'passed':True,'checks':checks,'warm_result':warm['result']},indent=2))
 print(json.dumps({'passed':True,'checks':checks}))
finally:
 process.terminate()
 try:process.wait(timeout=30)
 except subprocess.TimeoutExpired:process.kill();process.wait()
 log.close()
