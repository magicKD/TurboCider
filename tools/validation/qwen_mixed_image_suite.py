"""Matched final-image evaluation using real Qwen conditioning replay.

Both encoders use native tokenizer IDs and the same BF16 DiT/VAE pipeline.
The replay executable is test-only; image-stage timing excludes upstream
encoding and must not be advertised as integrated App e2e performance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BIN=ROOT/'build/native/z-image-conditioning-replay'
MODEL=ROOT/'models/Comfy-Org-z_image_turbo'
CASES=[
    ('fox','A cinematic red fox walking through fresh snow, soft morning light.'),
    ('portrait','A realistic photograph of a young adult woman with freckles and curly brown hair, holding a blue ceramic mug with both hands, sitting beside a window, soft natural light.'),
    ('objects','Studio product photograph: a red apple on the left, a blue ceramic cup in the center, and a yellow lemon on the right, three objects on a white table, plain pale gray background.'),
    ('architecture','A wide-angle architectural photograph of a small wooden cabin beside a mountain lake, pine forest, snow-covered peaks reflected in still water, sunrise, detailed natural textures.'),
    ('english_text','A straight-on photograph of a vintage cafe storefront. A large dark green sign clearly reads "TURBO CIDER" in white capital letters. Warm light through the windows, brick facade.'),
    ('chinese_text','一张中国街头茶馆的正面照片，木质招牌上清晰写着“春日茶馆”四个汉字，门前有两盆绿色植物，阳光柔和，写实摄影。'),
]


def invoke(args):
    r=subprocess.run(list(map(str,args)),text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr[-4000:])
    return json.loads(r.stdout)


def encode_all(out,precision):
    import mlx.core as mx
    from qwen_mixed_probe import header,prepare,encode
    filename='qwen_3_4b.safetensors' if precision=='bf16' else 'qwen_3_4b_fp4_mixed.safetensors'
    path=MODEL/'split_files/text_encoders'/filename
    _,formats=header(path)
    started=time.perf_counter()
    weights=mx.load(str(path));mx.eval(list(weights.values()))
    prepare(weights,formats,False,precision=='fp4')
    load=time.perf_counter()-started
    folder=out/precision
    folder.mkdir(parents=True,exist_ok=False)
    for name,prompt in CASES:
        t=invoke([BIN,'tokens',MODEL,prompt])
        mx.reset_peak_memory()
        started=time.perf_counter()
        value=encode(weights,formats,len(t['ids']),False,precision=='fp4',token_ids=t['ids'])
        elapsed=time.perf_counter()-started
        mx.save_safetensors(str(folder/f'{name}.safetensors'),
                            {'conditioning':value,'token_ids':mx.array(t['ids'],mx.int32)})
        (folder/f'{name}.json').write_text(json.dumps(dict(prompt=prompt,tokens=t,
            load_and_adapt_seconds=load,encode_seconds=elapsed,mlx_peak_bytes=mx.get_peak_memory()),indent=2))
        del value
    # A placeholder prevents replay from redundantly loading the BF16 encoder.
    # It is never used for numerical computation; replay validates exact IDs.
    layout=out/'layout'
    if not layout.exists():
        (layout/'text_encoder').mkdir(parents=True)
        mx.save_safetensors(str(layout/'text_encoder/model.safetensors'),{'replay_only':mx.zeros((1,))})
        (layout/'tokenizer').symlink_to(MODEL/'tokenizer',target_is_directory=True)
        (layout/'split_files').symlink_to(MODEL/'split_files',target_is_directory=True)


def compare(out):
    import numpy as np
    from PIL import Image,ImageDraw
    results=[]
    sheet=Image.new('RGB',(4*320,6*350),'white')
    draw=ImageDraw.Draw(sheet)
    for row,(name,prompt) in enumerate(CASES):
        for pair,seed in enumerate((42,123)):
            images=[Image.open(out/f'{name}-{seed}-{q}.png').convert('RGB') for q in ('bf16','fp4')]
            a,b=[np.array(i,dtype=np.float64).ravel() for i in images]
            mse=float(np.mean((a-b)**2))
            results.append(dict(case=name,prompt=prompt,seed=seed,
                pixel_cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
                pixel_rmse=float(np.sqrt(mse)),psnr_db=float(10*np.log10(255**2/mse)) if mse else None))
            for col,(q,im) in enumerate(zip(('BF16','FP4 mixed'),images)):
                x=(2*pair+col)*320;y=row*350
                sheet.paste(im.resize((320,320)),(x,y+30))
                draw.text((x+6,y+8),f'{name} | seed {seed} | {q}',fill='black')
    sheet.save(out/'comparison.png')
    (out/'quality.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print(json.dumps(results,indent=2,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encode-only',choices=('bf16','fp4'))
    p.add_argument('--compare-only',action='store_true')
    p.add_argument('--control-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.encode_only:return encode_all(out,a.encode_only)
    if a.compare_only:return compare(out)
    if not a.control_only and not a.resume and out.exists():raise ValueError('Use a new suite directory or --resume')
    out.mkdir(parents=True,exist_ok=True)
    for q in ('bf16','fp4'):
        if not (out/q).exists():
            subprocess.run([sys.executable,__file__,'--output',str(out),'--encode-only',q],check=True)
    # Native-vs-replay control is required before interpreting FP4 differences.
    name,prompt=CASES[0]
    control=out/'fox-native-control.png'
    if not control.exists():invoke([BIN,'generate',MODEL,prompt,42,512,'-',control])
    replay=out/'fox-42-bf16.png'
    if not replay.exists():invoke([BIN,'generate',out/'layout',prompt,42,512,out/'bf16/fox.safetensors',replay])
    from PIL import Image,ImageChops
    if ImageChops.difference(Image.open(control),Image.open(replay)).getbbox():
        raise ValueError('BF16 replay differs from native baseline; stop before comparison')
    (out/'control.json').write_text(json.dumps({'native_vs_bf16_replay_pixels_exact':True}))
    if a.control_only:return
    for row,(name,prompt) in enumerate(CASES):
        for seed in (42,123):
            # Alternate precision order by case to avoid a fixed timing order.
            for q in (('bf16','fp4') if row%2==0 else ('fp4','bf16')):
                image=out/f'{name}-{seed}-{q}.png'
                if image.exists():continue
                metrics=invoke([BIN,'generate',out/'layout',prompt,seed,512,out/q/f'{name}.safetensors',image])
                (out/f'{name}-{seed}-{q}.json').write_text(json.dumps(metrics,indent=2))
                print(name,seed,q,metrics,flush=True)
    compare(out)


if __name__=='__main__':main()
