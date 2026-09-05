"""Offline tokenizer parity. Does not load inference weights or fetch assets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--binary',required=True);p.add_argument('--report',required=True);a=p.parse_args()
from transformers import Qwen2TokenizerFast
t=Qwen2TokenizerFast.from_pretrained(str(Path(a.model)/'tokenizer'),local_files_only=True)
samples=['A red fox in snow.','一只猫坐在窗边。','Cafe\u0301 🦊 1234567890','first line\n\nsecond line','literal <|im_start|> and <think> tokens']
results=[]
for prompt in samples:
    native=json.loads(subprocess.check_output([a.binary,'tokenize',a.model,prompt]))['ids']
    text='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    reference=t.encode(text,add_special_tokens=False)
    results.append({'prompt':prompt,'identical':native==reference,'native_ids':native,'reference_ids':reference})
report={'passed':all(x['identical'] for x in results),'cases':results}
Path(a.report).write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps({'passed':report['passed'],'cases':len(results)}))
raise SystemExit(0 if report['passed'] else 1)
