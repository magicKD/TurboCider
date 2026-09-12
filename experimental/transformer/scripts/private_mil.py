"""Generate the same SwiGLU FFN shard as ANE-friendly conv MIL, with dynamic blobs."""
import struct
from pathlib import Path
import numpy as np

def private_mil(root,m,h,f,af,layers):
    for l in range(layers):
        d=Path(root)/f'layer_{l:02d}'
        up=np.fromfile(d/'upgate_ane.weights.fp16',dtype='float16').reshape(h,2*af)
        down=np.fromfile(d/'down.weights.fp16',dtype='float16').reshape(f,h)[f-af:]
        matrices={'up':up[:,:af].T.copy(),'gate':up[:,af:].T.copy(),'down':down.T.copy()}
        for name,w in matrices.items():
            blob=bytearray(128);blob[0]=1;blob[4]=2;struct.pack_into('<I',blob,64,0xdeadbeef);blob[68]=1;struct.pack_into('<I',blob,72,w.nbytes);struct.pack_into('<I',blob,80,128)
            (d/f'{name}.bin').write_bytes(blob+w.tobytes())
        s='program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]\n{\n'
        s+=f'func main<ios18>(tensor<fp16, [1, {h}, 1, {m}]> x) {{\n'
        for name,typ,val in [('pad_type','string','string("valid")'),('strides','tensor<int32, [2]>','tensor<int32, [2]>([1, 1])'),('pad','tensor<int32, [4]>','tensor<int32, [4]>([0, 0, 0, 0])'),('dilations','tensor<int32, [2]>','tensor<int32, [2]>([1, 1])'),('groups','int32','int32(1)')]:
            s+=f'{typ} c_{name} = const()[name = string("c_{name}"), val = {val}];\n'
        for name,w in matrices.items():
            n,k=w.shape;typ=f'tensor<fp16, [{n}, {k}, 1, 1]>'
            s+=f'{typ} W_{name} = const()[name = string("W_{name}"), val = {typ}(BLOBFILE(path = string("@model_path/weights/{name}.bin"), offset = uint64(64)))];\n'
        def conv(name,inp,n):return f'tensor<fp16, [1, {n}, 1, {m}]> {name} = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_{name}, x = {inp})[name = string("{name}")];\n'
        s+=conv('up','x',af)+conv('gate','x',af)
        typ=f'tensor<fp16, [1, {af}, 1, {m}]>'
        s+=f'{typ} sig = sigmoid(x = gate)[name = string("sig")];\n{typ} silu = mul(x = gate, y = sig)[name = string("silu")];\n{typ} act = mul(x = up, y = silu)[name = string("act")];\n'
        s+=conv('down','act',h)+'} -> (down);\n}\n';(d/'private.mil').write_text(s)
