"""Read-only, no-weight compile checks for reference video engines."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--h3',required=True);p.add_argument('--ltx',required=True);p.add_argument('--mlx',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    dev=Path(os.environ.get('DEVELOPER_DIR','/Applications/Xcode.app/Contents/Developer'))
    sdk=dev/'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
    cc=dev/'Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'
    cxx=cc.with_name('clang++')
    checks=[]
    h3='h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c h3_dit_schedule.c h3_dit.c h3_video_vae.c h3_taeh3.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c h3_vision_encoder.c h3_multimodal.c h3_metal.m h3_gpu.m h3_tokenizer.m h3_super.m h3_coreml.m'
    ltx='ltx.c ltx_conditioning.c ltx_connector.c ltx_transformer_io.c ltx_latent_stats.c ltx_rng.c ltx_safetensors.m ltx_weights.m ltx_gpu.m ltx_upsampler.m ltx_video_vae.m ltx_ane_mlp.m ltx_ane_kv.m ltx_mlx_upsampler.cpp ltx_mlx_video_vae.cpp'
    for model,root,names in [('h3',Path(a.h3),h3),('ltx',Path(a.ltx),ltx)]:
        for name in names.split():checks.append((model,root,name))
    def run(check):
        model,root,name=check
        cpp=name.endswith('.cpp')
        cmd=[str(cxx if cpp else cc),'-std=c++20' if cpp else '-std=c11','-fsyntax-only','-isysroot',str(sdk),'-I',str(root),'-D_DARWIN_C_SOURCE','-Werror=implicit-function-declaration']
        if name.endswith('.m'):cmd+=['-fobjc-arc']
        if cpp:cmd+=['-isystem',str(Path(a.mlx)/'include')]
        cmd+=[str(root/name)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        return {'model':model,'file':name,'status':'passed' if result.returncode==0 else 'failed','diagnostics':result.stderr[-6000:]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,checks))
    report={'scope':'syntax-only reference source; not new model executor parity; no model access/download','sdk':str(sdk),'checks':results,'passed':sum(x['status']=='passed' for x in results),'failed':sum(x['status']=='failed' for x in results)}
    Path(a.report).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
    for r in results:
        if r['status']=='failed':print(r['model'],r['file'],r['diagnostics'][:500])
if __name__=='__main__':main()
