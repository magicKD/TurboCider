"""Bind native pre-SiLU LoRA calibration deltas to their exact real inputs."""
import hashlib
import json
from pathlib import Path


def digest(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('correction calibration requires regular files')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def load_corrections(directory, inputs, layer, rows, width, checkpoint_sha256, np):
    directory,inputs=Path(directory),Path(inputs)
    binding_path=directory/'binding.json'
    if binding_path.is_symlink() or binding_path.stat().st_size>4*1024*1024:
        raise ValueError('invalid correction binding file')
    binding=json.loads(binding_path.read_text())
    if (binding.get('schema_version')!=1 or binding.get('width')!=width or
        binding.get('checkpoint_sha256')!=checkpoint_sha256 or not binding.get('adapter_identity')):
        raise ValueError('LoRA correction checkpoint/width/adapter binding mismatch')
    samples=sorted((inputs/f'block{layer}').glob('*.npy'))
    if not 1<=len(samples)<=32:
        raise ValueError('correction calibration requires 1...32 unbatched input samples')
    pairs=[]; h=hashlib.sha256(bytes.fromhex(digest(binding_path)))
    for sample in samples:
        key=f'block{layer}/{sample.name}'; record=binding['samples'][key]
        if digest(sample)!=record['input_sha256']:
            raise ValueError('LoRA correction input digest mismatch')
        pair=[]
        for part in ('gate','up'):
            path=directory/f'block{layer}'/f'{sample.stem}.{part}.npy'
            if path.stat().st_size>rows*width*4+1024 or digest(path)!=record[f'{part}_sha256']:
                raise ValueError('LoRA correction content digest mismatch')
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if value.shape!=(rows,width) or value.dtype not in (np.float16,np.float32) or not np.isfinite(value).all():
                raise ValueError('invalid LoRA correction shape/dtype/values')
            h.update(bytes.fromhex(record[f'{part}_sha256']));pair.append(value)
        pairs.append(tuple(pair))
    return pairs,h.hexdigest(),binding['adapter_identity']
