"""Fail-closed summary of bounded real-input FFN replays, never E2E certification."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows)!=28:
        raise ValueError('incomplete replay: expected header and 3 x 9 completed points')
    header,*points=rows
    if (header.get('samples')!=9 or header.get('warmup_passes')!=1 or header.get('measured_passes')!=2
            or header.get('calibration_step')!=1 or type(header.get('lora')) is not bool):
        raise ValueError('invalid replay header')
    if header.get('width') not in (4096,6144) or header.get('recipe') not in ('hadamard','sq-hadamard','comfy'):
        raise ValueError('unknown channel/recipe')
    seen=set(); identities={}; steps=set()
    numeric=('gpu_seconds','hybrid_seconds','relative_rmse','cosine','headroom',
             'stage_seconds','stage_wait_seconds','pre_seconds','lora_input_ready_seconds',
             'a8_wait_seconds','async_ane_wait_seconds','post_join_seconds','wall_seconds')
    for p in points:
        key=(p.get('pass'),p.get('step'),p.get('layer'))
        if key in seen or any(type(v) is not int for v in key) or key[0] not in (0,1,2) or key[2] not in (0,15,31) or key[1]<1:
            raise ValueError('duplicate/invalid replay point')
        seen.add(key);steps.add(key[1])
        if type(p.get('held_out')) is not bool or p['held_out']!=(key[1]!=1):
            raise ValueError('calibration/holdout label mismatch')
        digest=p.get('input_sha256','')
        if len(digest)!=64 or set(digest)-set('0123456789abcdef'):
            raise ValueError('invalid input digest')
        source_key=key[1:]
        if identities.setdefault(source_key,digest)!=digest:
            raise ValueError('input changed across passes')
        if any(type(p.get(k)) not in (int,float) or not math.isfinite(p[k]) or p[k]<0 for k in numeric):
            raise ValueError('invalid numeric evidence')
        if min(p['gpu_seconds'],p['hybrid_seconds'])<=0 or type(p.get('calls')) is not int or p['calls']!=1:
            raise ValueError('no timed complete ANE path')
        for k in ('fallback_blocks','overflow_retries','scale_cache_hits','scale_cache_misses'):
            if type(p.get(k)) is not int or p[k]<0:
                raise ValueError('invalid path counters')
    if len(steps)!=3 or 1 not in steps or len(identities)!=9:
        raise ValueError('incomplete depths/steps')
    hot=[p for p in points if p['pass']>0]
    quality=all(p['relative_rmse']<=.03 and p['cosine']>=.999 for p in points)
    stable=all(p['fallback_blocks']==p['overflow_retries']==0 and p['headroom']==1 for p in points)
    # Compare sums of paired full-path medians at each depth/step. Exclude
    # warmup for timing, but NEVER discard its quality or retry failures.
    groups={key:[p for p in hot if (p['step'],p['layer'])==key] for key in identities}
    g=sum(statistics.median(p['gpu_seconds'] for p in group) for group in groups.values())
    h=sum(statistics.median(p['hybrid_seconds'] for p in group) for group in groups.values())
    layers={str(layer):sum(p['gpu_seconds'] for p in hot if p['layer']==layer)/
                        sum(p['hybrid_seconds'] for p in hot if p['layer']==layer) for layer in (0,15,31)}
    speed=g/h
    gain=speed>=1.05 and min(layers.values())>=1
    return dict(source=str(path),recipe=header['recipe'],width=header['width'],lora=header['lora'],
        launch_fence=header.get('launch_fence',False),
        setup_seconds=header['setup_seconds'],gpu_mean_ms=1000*g/9,hybrid_mean_ms=1000*h/9,
        speedup=speed,per_layer_speedup=layers,quality_passed=quality,stable_recipe=stable,
        worst_relative_rmse=max(p['relative_rmse'] for p in points),minimum_cosine=min(p['cosine'] for p in points),
        component_gain_passed=gain,eligible_for_next_e2e=quality and stable and gain,
        host_spans_mean_ms={k:1000*statistics.mean(p[k] for p in hot) for k in numeric if k.endswith('_seconds')
                            and k not in ('gpu_seconds','hybrid_seconds')},
        note='Host spans overlap and are not additive GPU/ANE engine times; no full-model export authorization.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replays',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reports=[summarize(path) for path in a.replays]
    result={'schema':'qwen21-real-ffn-screen-v1','scope':'bounded component screen; not E2E or physical INT8 arithmetic proof',
            'reports':reports,'full_model_export_qualified':False}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
