"""Offline validation oracle. Not part of the native inference product."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import mlx.core as mx
    from mflux.models.common.config.config import Config
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
    from PIL import Image
    import numpy as np
    r = json.loads(Path(args.request).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(512 * 1024 * 1024)
    start = time.monotonic()
    model = Flux2Klein(model_path=args.model)
    config = Config(model_config=model.model_config, num_inference_steps=r['steps'], height=r['height'], width=r['width'], guidance=1.0, scheduler='flow_match_euler_discrete')
    def save(name, value):
        mx.eval(value)
        mx.save_safetensors(str(out / f'{name}.safetensors'), {'tensor': value})
    text, text_ids = Flux2PromptEncoder.encode_prompt(r['prompt'], model.tokenizers['qwen3'], model.text_encoder, padding='longest' if r.get('dynamic_text', True) else None)
    save('conditioning', text)
    # Stage release reflects the native budget policy, without altering math.
    del model.text_encoder
    mx.clear_cache()
    z, image_ids, h, w = Flux2LatentCreator.prepare_packed_latents(seed=r['seed'],height=r['height'],width=r['width'],batch_size=1)
    save('initial_latent', z)
    sigmas = config.scheduler.sigmas
    reference_latents=None
    reference_ids=None
    start_step=0
    from mflux.utils.image_util import ImageUtil
    from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers as Edit
    for index,asset in enumerate(r.get('inputs',[])):
        image=ImageUtil.load_image(asset['path']).convert('RGB')
        image=Edit.prepare_reference_image(image) if r['operation']=='image.edit' else ImageUtil.scale_to_dimensions(image,r['width'],r['height'])
        pixels=ImageUtil.to_array(image)
        save(f'input_image_{index}',pixels.transpose(0,2,3,1))
        encoded=model.vae.encode(pixels)
        encoded=Flux2LatentCreator.patchify_latents(encoded)
        encoded=Edit.bn_normalize_vae_encoded_latents(encoded,vae=model.vae)
        ids=Flux2LatentCreator.prepare_grid_ids(encoded,t_coord=10+10*index)
        encoded=Flux2LatentCreator.pack_latents(encoded)
        save(f'image_latent_{index}',encoded)
        if r['operation']=='image.transform':
            strength=asset.get('strength',.75)
            if strength>0:
                start_step=max(1,int(r['steps']*strength))
                sigma=sigmas[start_step]
                z=(1-sigma)*encoded+sigma*z
                save('conditioned_initial_latent',z)
        else:
            reference_latents=encoded if reference_latents is None else mx.concatenate([reference_latents,encoded],axis=1)
            reference_ids=ids if reference_ids is None else mx.concatenate([reference_ids,ids],axis=1)
    for i in range(start_step,r['steps']):
        print(json.dumps({'phase':'reference_denoise','step':i}), flush=True)
        model_input=z if reference_latents is None else mx.concatenate([z,reference_latents],axis=1)
        ids=image_ids if reference_ids is None else mx.concatenate([image_ids,reference_ids],axis=1)
        noise = model.transformer(hidden_states=model_input, encoder_hidden_states=text, timestep=config.scheduler.timesteps[i], img_ids=ids, txt_ids=text_ids, guidance=None)
        noise=noise[:,:z.shape[1]]
        save(f'noise_{i}',noise)
        z = config.scheduler.step(noise=noise,timestep=i,latents=z,sigmas=sigmas)
        save(f'latent_{i}',z)
    packed = z.reshape(1,h,w,128).transpose(0,3,1,2)
    x = model.vae.unpack_packed_latents(packed)
    def stage(name, value): save('vae_'+name,value.transpose(0,2,3,1))
    stage('unpack', x)
    x = model.vae.post_quant_conv(((x / model.vae.scaling_factor)+model.vae.shift_factor).transpose(0,2,3,1)).transpose(0,3,1,2)
    stage('post',x)
    x = model.vae.decoder.conv_in(x);stage('in',x)
    x = model.vae.decoder.mid_block.resnets[0](x);stage('mid0',x)
    x = model.vae.decoder.mid_block.attentions[0](x);stage('attention',x)
    x = model.vae.decoder.mid_block.resnets[1](x);stage('mid1',x)
    for i, block in enumerate(model.vae.decoder.up_blocks):
        x = block(x);stage(f'up{i}',x)
    x = model.vae.decoder.conv_norm_out(x);stage('norm',x)
    from mlx import nn
    pixels = model.vae.decoder.conv_out(nn.silu(x)).transpose(0,2,3,1)
    save('pixels_nhwc',pixels)
    array = np.asarray(mx.round(mx.clip(pixels.astype(mx.float32)/2+.5,0,1)*255).astype(mx.uint8))[0]
    Image.fromarray(array).save(out/'reference.png')
    report = {'reference':'mflux','seconds':time.monotonic()-start,'sigmas':sigmas.tolist(),'mlx_peak_bytes':mx.get_peak_memory(),'request':r,'model':args.model}
    (out/'reference.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
