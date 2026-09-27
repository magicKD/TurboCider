#!/usr/bin/env python3
"""One-shot worker query correlation/cancellation with isolated C API doubles."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
request=json.loads((root/'tests/fixtures/streaming/worker-request-v1.json').read_text())
digest=hashlib.sha256(b'tc-worker-request-v1\n'+json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
wire={'protocol_version':1,'job_id':'00000000-0000-4000-8000-000000000001','request_id':'00000000-0000-4000-8000-000000000002','request_digest':digest,'model_installation_ref':'/tmp/model','native_request_v2':request}
with tempfile.TemporaryDirectory(prefix='tc-query-contract-') as tmp:
    work=Path(tmp);binary=work/'test';input=work/'input.json';input.write_text(json.dumps(wire))
    subprocess.run(['clang++','-std=c++20','-fobjc-arc','-Wall','-Wextra','-Werror','-I',str(root/'bindings/c/include'),str(root/'tests/native/worker_query_contract_test.mm'),'-framework','Foundation','-o',str(binary)],check=True)
    for mode in ['success','create_error','source_error','resolve_error','wrong_container','cancel']:
        output=subprocess.run([str(binary),str(input),mode],capture_output=True,text=True,timeout=5)
        expected=0 if mode=='success' else 2 if mode=='cancel' else 1
        assert output.returncode==expected,(mode,output.stderr)
        result=json.loads(output.stdout)
        for key in ['job_id','request_id','request_digest']:assert result[key]==wire[key],mode
        assert result['actual_container']=='cli_worker' and result['runtime_fingerprint']=='test-runtime'
        assert result['artifact'] is None and result['public_streaming_summary'] is None
        if mode=='success':assert result['status']=='resolved' and result['resolution_digest']=='resolution'
        else:assert result['status']==('cancelled' if mode=='cancel' else 'error') and result['resolution_digest'] is None
        print(mode,'PASS')
