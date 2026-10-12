"""Offline original FLUX.2 Klein 4B/9B single-block MLP export.

Dependencies: coremltools 8.3.0 and numpy 2.0.2 (validated on Python 3.11).
The runtime stays C++/Metal. Indexed9B sources are never flattened/duplicated.
"""
import argparse, fcntl, gc, hashlib, json, mmap, os, shutil, signal, struct, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import apply as apply_lora
from lora import load as load_loras
from lora import provenance as lora_provenance
from flux_checkpoint import Checkpoint, geometry

def atom(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)
def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):result.update(block)
    return result.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bucket',type=int,required=True);p.add_argument('--ane-mlp-width',type=int);p.add_argument('--lora',action='append',default=[]);p.add_argument('--lora-strength',action='append',default=[]);p.add_argument('--lora-role',default='transformer');a=p.parse_args()
    if not 64<=a.bucket<=8192:raise ValueError('bucket must be 64...8192')
    import numpy as np
    import coremltools as ct
    import coremltools.optimize.coreml as optimize
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if a.lora_strength and len(a.lora_strength) != len(a.lora): raise ValueError('--lora-strength must be repeated once per --lora')
    lora_strengths=[float(value) for value in a.lora_strength] or [1.0]*len(a.lora)
    lora_roles=[a.lora_role]*len(a.lora)
    lora_records=lora_provenance(a.lora,lora_strengths,lora_roles)
    lora_bundles=load_loras(a.lora,lora_strengths,lora_roles,np)
    config=json.loads((a.model/'transformer/config.json').read_text())
    dims=geometry(config);hidden=dims['hidden'];mlp_width=dims['mlp_width'];blocks=dims['single_layers']
    if a.ane_mlp_width is None:a.ane_mlp_width=mlp_width
    if not 0<a.ane_mlp_width<=mlp_width:raise ValueError('ane-mlp-width must fit the original MLP')
    reader=Checkpoint(a.model);checkpoint=reader.path
    output=a.output.absolute()
    if output.is_symlink():raise ValueError('symlink output unsupported')
    output.mkdir(parents=True,exist_ok=True)
    lock_path=output/'.export.lock'
    descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        provenance=reader.identity()
        identity={'owner':'turbocider.flux.coreml.v1',**provenance,'bucket':a.bucket,'variant':'int8_pc','coremltools':ct.__version__,'numpy':np.__version__,'recipe':1}
        if lora_records: identity.update(loras=lora_records,recipe=2)
        if a.ane_mlp_width!=mlp_width:identity.update(ane_mlp_width=a.ane_mlp_width,recipe=2)
        if hidden!=3072:identity.update(geometry=dims,recipe=3)
        marker=output/'.turbocider-export.json'
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text())!=identity:raise ValueError('export identity changed; select a new output directory')
        else:
            if any(x.name not in ['.export.lock','export.log'] for x in output.iterdir()):raise ValueError('refusing to adopt a nonempty export directory')
            atom(marker,identity)
        manifest=output/'manifest.json'
        with reader:
            def tensor(name,shape):
                return reader.tensor(name,shape,np)
            artifacts={};checksums={}
            for i in range(blocks):
                name=f'block{i}_mlp_branch.int8_pc.mlpackage';destination=output/name;receipt=output/f'block{i}.json'
                if destination.is_symlink() or receipt.is_symlink():raise ValueError('symlink artifact unsupported')
                if destination.exists() and receipt.exists():
                    saved=json.loads(receipt.read_text());valid=all((destination/k).is_file() and sha(destination/k)==v for k,v in saved.items())
                    if not valid:raise ValueError('existing export corrupted; delete it through resource management before rebuilding')
                    checksums[str(i)]=saved
                else:
                    if destination.exists():raise ValueError('incomplete artifact must be removed before rebuilding')
                    width=a.ane_mlp_width
                    if lora_bundles:
                        wide=tensor(f'single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight',(hidden*9,hidden))
                        projected=tensor(f'single_transformer_blocks.{i}.attn.to_out.weight',(hidden,hidden*4))
                        wide,_=apply_lora(f'single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight',wide,lora_bundles,np)
                        projected,_=apply_lora(f'single_transformer_blocks.{i}.attn.to_out.weight',projected,lora_bundles,np)
                        gate=wide[hidden*3:hidden*3+width];up=wide[hidden*6:hidden*6+width]
                        down=projected[:,hidden:hidden+width]
                        first=np.ascontiguousarray(np.concatenate((gate,up),axis=0)[:,:,None,None]);last=np.ascontiguousarray(down[:,:,None,None]);del wide,projected,gate,up,down
                    else:
                        gate=reader.tensor(f'single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight',(hidden*9,hidden),np,rows=(hidden*3,hidden*3+width))
                        up=reader.tensor(f'single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight',(hidden*9,hidden),np,rows=(hidden*6,hidden*6+width))
                        down=reader.tensor(f'single_transformer_blocks.{i}.attn.to_out.weight',(hidden,hidden*4),np,columns=(hidden,hidden+width))
                        first=np.ascontiguousarray(np.concatenate((gate,up),axis=0)[:,:,None,None]);last=np.ascontiguousarray(down[:,:,None,None]);del gate,up,down
                    # The ANE owns a contiguous prefix of the two MLP input
                    # projections and the matching output columns.  The native
                    # GPU graph computes the complementary suffix in parallel.
                    @mb.program(input_specs=[mb.TensorSpec(shape=(1,hidden,1,a.bucket),dtype=types.fp16)],opset_version=ct.target.macOS15)
                    def branch(x):
                        hidden=mb.conv(x=x,weight=first,pad_type='valid',name='projected')
                        gate,up=mb.split(x=hidden,num_splits=2,axis=1)
                        return mb.conv(x=mb.mul(x=mb.silu(x=gate),y=up),weight=last,pad_type='valid',name='y')
                    model=ct.convert(branch,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,compute_precision=ct.precision.FLOAT16,skip_model_load=True)
                    quantizer=optimize.OptimizationConfig(global_config=optimize.OpLinearQuantizerConfig(mode='linear_symmetric',dtype='int8',granularity='per_channel',block_size=32,weight_threshold=0))
                    compressed=optimize.linear_quantize_weights(model,quantizer)
                    with tempfile.TemporaryDirectory(prefix='.export-',dir=output) as temporary:
                        package=Path(temporary)/name;compressed.save(package);os.replace(package,destination)
                    saved={str(f.relative_to(destination)):sha(f) for f in sorted(destination.rglob('*')) if f.is_file()}
                    atom(receipt,saved);checksums[str(i)]=saved
                    del first,last,branch,model,compressed;gc.collect()
                artifacts[str(i)]={'int8_pc':name}
                atom(output/'progress.json',{'completed':i+1,'total':blocks});print(f'partition {i+1}/{blocks} ready',flush=True)
        reader.check()
        for bundle in lora_bundles:
            if bundle['role']=='transformer' and bundle['applied']==0: raise ValueError(f"LoRA did not match an exported FLUX weight: {bundle['path']}")
        source={**provenance,'blocks':list(range(blocks))}
        if lora_records: source['loras']=lora_records
        atom(manifest,{'schema_version':2,'source':source,'shape':{'K':hidden,'N':hidden,'mlp_width':mlp_width,'ane_mlp_start':0,'ane_mlp_end':a.ane_mlp_width,'buckets':[a.bucket]},'functions':{str(a.bucket):'main'},'artifacts':artifacts,'artifact_sha256':checksums,'export_identity':identity})
        print(json.dumps({'source_manifest':str(manifest)}),flush=True)
if __name__=='__main__':main()
