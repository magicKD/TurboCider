#!/usr/bin/env python3
"""Fetch individually pinned Comfy Qwen21 assets directly from ModelScope.

No second hub cache, no renaming Turbo to base, and no publication until full
size/SHA256 verification. Existing BF16/GGUF assets remain untouched.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/validation'))
from download_qwen21_gguf import download_asset,local_size

REPOSITORY='Comfy-Org/Qwen-Image-2.1'
PINS={
    'turbo-bf16':('8c798285d642b81a1af415b70f2e444e809820d7','diffusion_models/qwen_image_2.1_turbo_bf16.safetensors',14230280616,'8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a'),
    'base-convrot':('5dc8a8352312860d67e07cc9bee310dc7ed3e55c','diffusion_models/qwen_image_2.1_int8_convrot.safetensors',7256783064,'cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d'),
    'turbo-convrot':('8c798285d642b81a1af415b70f2e444e809820d7','diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors',7256787829,'8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c'),
    'encoder-convrot':('5dc8a8352312860d67e07cc9bee310dc7ed3e55c','text_encoders/qwen3vl_8b_int8_convrot.safetensors',9350798360,'8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f'),
}


def prepare_variant_root(root,original,asset):
    if asset=='encoder-convrot':raise ValueError('encoder-only assets do not select a transformer variant')
    relative=PINS[asset][1];target=root.absolute();original=original.resolve(strict=True)
    if target.exists() or target.is_symlink() or target==original:raise ValueError('fresh separate variant root required; no model overwrite')
    source=original/relative
    if not source.is_file():raise ValueError('verified source must exist before variant publication')
    for name in ('processor','text_encoders','vae'):
        if not (original/name).is_dir():raise ValueError('original shared companion missing: '+name)
    target.mkdir(parents=True);(target/'diffusion_models').mkdir()
    (target/relative).symlink_to(source.resolve(strict=True))
    for name in ('processor','text_encoders','vae'):(target/name).symlink_to(original/name,target_is_directory=True)
    record=dict(schema='tc-qwen21-original-filename-variant-root-v1',asset=asset,checkpoint=relative,
        source=str(source.resolve(strict=True)),sha256=PINS[asset][3],companions='read-only shared path links, no copied payload',
        runtime_support_verified=False,performance_verified=False)
    (target/'variant-installation.json').write_text(json.dumps(record,indent=2)+'\n')
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',action='append',choices=PINS,required=True)
    p.add_argument('--root',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1');p.add_argument('--reserve-gib',type=float,default=4.)
    p.add_argument('--variant-root',type=Path,help='fresh separate root, keeping original Turbo/ConvRot leaf and shared companions');args=p.parse_args()
    if args.reserve_gib<2:p.error('at least2GiB reserve required')
    if args.variant_root and (len(args.asset)!=1 or args.asset[0]=='encoder-convrot'):p.error('variant root requires exactly one transformer asset')
    if args.variant_root and (args.variant_root.exists() or args.variant_root.is_symlink()):p.error('fresh variant root required')
    root=args.root.resolve(strict=True);assets=[];remaining=0
    for name in dict.fromkeys(args.asset):
        revision,path,size,digest=PINS[name];target=root/path
        partial=target.with_name(target.name+'.part');local_size(target);offset=local_size(partial)
        if offset>size:raise ValueError('oversize partial must be inspected, never overwritten')
        remaining+=0 if target.exists() else size-offset
        assets.append((name,dict(repository=REPOSITORY,revision=revision,filename=path,relative_path=path,bytes=size,sha256=digest)))
    if shutil.disk_usage(root).free<remaining+int(args.reserve_gib*(1<<30)):raise ValueError('insufficient space for selected assets and reserve')
    for name,asset in assets:
        result=download_asset(asset,root,endpoints=('https://modelscope.cn/models',))
        receipt=root/('modelscope-'+name+'-receipt.json')
        data=dict(schema='tc-qwen21-modelscope-pinned-asset-v1',asset=name,endpoint='https://modelscope.cn',result=result,
            runtime_support_verified=False,image_quality_verified=False,performance_verified=False)
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['result']['sha256']!=result['sha256']:raise ValueError('existing receipt identity differs')
        else:receipt.write_text(json.dumps(data,indent=2)+'\n')
    if args.variant_root:print(json.dumps(dict(variant_root=str(prepare_variant_root(args.variant_root,root,args.asset[0])))),flush=True)


if __name__=='__main__':main()
