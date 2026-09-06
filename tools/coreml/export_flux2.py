"""Offline FLUX.2 Klein 4B single-block MLP export. No model loading or downloads.

Dependencies: coremltools 8.3.0 and numpy 2.0.2 (validated on Python 3.11).
The runtime stays C++/Metal; this optional build tool emits 20 INT8 partitions.
"""
import argparse, fcntl, gc, hashlib, json, mmap, os, shutil, signal, struct, tempfile
from pathlib import Path

def atom(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)
def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):result.update(block)
    return result.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bucket',type=int,required=True);p.add_argument('--ane-mlp-width',type=int,default=9216);a=p.parse_args()
    if not 64<=a.bucket<=8192:raise ValueError('bucket must be 64...8192')
    if not 0<a.ane_mlp_width<=9216:raise ValueError('ane-mlp-width must be 1...9216')
    import numpy as np
    import coremltools as ct
    import coremltools.optimize.coreml as optimize
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    checkpoint=(a.model/'transformer/diffusion_pytorch_model.safetensors').resolve(strict=True)
    config=json.loads((a.model/'transformer/config.json').read_text())
    if config.get('num_single_layers')!=20 or config.get('num_attention_heads')!=24 or config.get('attention_head_dim')!=128:raise ValueError('only official FLUX.2 Klein 4B supported')
    output=a.output.absolute()
    if output.is_symlink():raise ValueError('symlink output unsupported')
    output.mkdir(parents=True,exist_ok=True)
    lock_path=output/'.export.lock'
    descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={'owner':'turbocider.flux.coreml.v1','checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_sha256':sha(checkpoint),'bucket':a.bucket,'variant':'int8_pc','coremltools':ct.__version__,'numpy':np.__version__,'recipe':1}
        if a.ane_mlp_width!=9216:identity.update(ane_mlp_width=a.ane_mlp_width,recipe=2)
        marker=output/'.turbocider-export.json'
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text())!=identity:raise ValueError('export identity changed; select a new output directory')
        else:
            if any(x.name not in ['.export.lock','export.log'] for x in output.iterdir()):raise ValueError('refusing to adopt a nonempty export directory')
            atom(marker,identity)
        manifest=output/'manifest.json'
        with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as memory:
            header_size=struct.unpack('<Q',memory[:8])[0]
            if header_size>100_000_000 or header_size+8>len(memory):raise ValueError('invalid safetensors header')
            header=json.loads(memory[8:8+header_size]);offset=8+header_size
            def tensor(name,shape):
                meta=header[name]
                if meta['shape']!=list(shape) or meta['dtype'] not in ['BF16','F16']:raise ValueError('unexpected tensor shape/dtype: '+name)
                start,end=meta['data_offsets'];count=int(np.prod(shape))
                if start<0 or end-start!=count*2 or offset+end>len(memory):raise ValueError('invalid safetensors offsets')
                if meta['dtype']=='BF16':
                    words=np.frombuffer(memory,dtype='<u2',count=count,offset=offset+start).astype(np.uint32)<<16
                    return words.view(np.float32).astype(np.float16).reshape(shape)
                return np.frombuffer(memory,dtype='<f2',count=count,offset=offset+start).copy().reshape(shape)
            artifacts={};checksums={}
            for i in range(20):
                name=f'block{i}_mlp_branch.int8_pc.mlpackage';destination=output/name;receipt=output/f'block{i}.json'
                if destination.is_symlink() or receipt.is_symlink():raise ValueError('symlink artifact unsupported')
                if destination.exists() and receipt.exists():
                    saved=json.loads(receipt.read_text());valid=all((destination/k).is_file() and sha(destination/k)==v for k,v in saved.items())
                    if not valid:raise ValueError('existing export corrupted; delete it through resource management before rebuilding')
                    checksums[str(i)]=saved
                else:
                    if destination.exists():raise ValueError('incomplete artifact must be removed before rebuilding')
                    wide=tensor(f'single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight',(27648,3072))
                    projected=tensor(f'single_transformer_blocks.{i}.attn.to_out.weight',(3072,12288))
                    # The ANE owns a contiguous prefix of the two MLP input
                    # projections and the matching output columns.  The native
                    # GPU graph computes the complementary suffix in parallel.
                    width=a.ane_mlp_width
                    first=np.ascontiguousarray(np.concatenate((wide[9216:9216+width],wide[18432:18432+width]),axis=0)[:,:,None,None]);last=np.ascontiguousarray(projected[:,3072:3072+width,None,None]);del wide,projected
                    @mb.program(input_specs=[mb.TensorSpec(shape=(1,3072,1,a.bucket),dtype=types.fp16)],opset_version=ct.target.macOS15)
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
                atom(output/'progress.json',{'completed':i+1,'total':20});print(f'partition {i+1}/20 ready',flush=True)
        if sha(checkpoint)!=identity['checkpoint_sha256']:raise ValueError('checkpoint changed during export')
        atom(manifest,{'schema_version':1,'source':{'checkpoint':str(checkpoint),'checkpoint_bytes':identity['checkpoint_bytes'],'checkpoint_sha256':identity['checkpoint_sha256'],'blocks':list(range(20))},'shape':{'K':3072,'N':3072,'mlp_width':9216,'ane_mlp_start':0,'ane_mlp_end':a.ane_mlp_width,'buckets':[a.bucket]},'functions':{str(a.bucket):'main'},'artifacts':artifacts,'artifact_sha256':checksums,'export_identity':identity})
        print(json.dumps({'source_manifest':str(manifest)}),flush=True)
if __name__=='__main__':main()
