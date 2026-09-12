"""Rebuild tables and publication SVG figures from preserved raw measurements."""
import json,re,csv,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'notes/raw';OUT=ROOT/'notes'
rows=[]
for suite in ['screen','heads','confirm','bridge','sp_audit','triple_audit','startup']:
    for p in sorted((RAW/suite).glob('*.json')):
        d=json.loads(p.read_text())
        if not isinstance(d,dict) or 'stdout' not in d:continue
        parsed=[json.JSONDecoder().raw_decode(d['stdout'][m.start():])[0] for m in re.finditer(r'\{"(?:record_type|benchmark)"',d['stdout'])]
        if not parsed:continue
        z=parsed[-1];samples=parsed[0];z.update(suite=suite,file=str(p.relative_to(ROOT)),stem=p.stem,process_s=d['process_s'])
        z['gpu_samples']=samples.get('gpu_ms',[]);z['hetero_samples']=samples.get('hetero_ms',[])
        if not all(np.isfinite(v) for k,v in z.items() if isinstance(v,float)):raise ValueError(f'nonfinite {p}')
        z['quality_pass']=z['nrmse']<=.01 and z['cosine']>=.999
        if suite=='screen':key=re.sub('_L[123]_r0$','',p.stem)
        else:key=re.sub('_r[0-9]+$','',p.stem)
        z['group']=key;rows.append(z)
(OUT/'measurements.json').write_text(json.dumps(rows,indent=2))
keys=['suite','stem','M','H','F','heads','layers','gpu_wall_p50_ms','hetero_wall_p50_ms','gpu_wall_p95_ms','hetero_wall_p95_ms','speedup','nrmse','cosine','gpu_reference_nrmse','hetero_reference_nrmse','quality_pass','process_s']
with (OUT/'measurements.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
rng=np.random.default_rng(20260910);groups={}
for x in rows:
    if x['suite'] in ['confirm','bridge','sp_audit','triple_audit','startup']:groups.setdefault(x['suite']+'/'+x['group'],[]).append(x)
summary=[]
for name,rs in groups.items():
    ratios=[]
    for _ in range(1500):
        boot=[]
        for idx in rng.integers(0,len(rs),len(rs)):
            r=rs[idx];g=np.array(r['gpu_samples']);h=np.array(r['hetero_samples']);n=len(g)
            starts=rng.integers(0,n,(n+3)//4);ix=np.concatenate([(np.arange(4)+i)%n for i in starts])[:n]
            boot.append(np.median(g[ix])/np.median(h[ix]))
        ratios.append(np.median(boot))
    summary.append({'group':name,'rounds':len(rs),'M':rs[0]['M'],'H':rs[0]['H'],'layers':rs[0]['layers'],'gpu_ms':statistics.median(r['gpu_wall_p50_ms'] for r in rs),'hetero_ms':statistics.median(r['hetero_wall_p50_ms'] for r in rs),'speedup':statistics.median(r['speedup'] for r in rs),'ci95':np.quantile(ratios,[.025,.975]).tolist(),'max_nrmse':max(r['nrmse'] for r in rs),'max_ref_nrmse':max(r['hetero_reference_nrmse'] for r in rs),'min_cosine':min(r['cosine'] for r in rs),'quality_pass':all(r['quality_pass'] for r in rs)})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print('measurements',len(rows),'summary groups',len(summary))
for x in sorted(summary,key=lambda r:r['speedup'],reverse=True)[:8]:print(x)
if __name__=='__main__':
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.dpi':150})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');fig.savefig(figdir/(name+'.png'),bbox_inches='tight',dpi=150);plt.close(fig)
    shapes=['64x256','256x1024','1024x1024','256x2048','1024x2048'];plans=['tp25','tp50','tp75','sp50','triple','cpu','projection50']
    mat=np.full((len(shapes),len(plans)),np.nan)
    for z in rows:
        if z['suite']=='screen' and z['layers']==3:
            sh=f"{z['M']}x{z['H']}";pl=z['group'].split('_')[-1];mat[shapes.index(sh),plans.index(pl)]=z['speedup']
    fig,ax=plt.subplots(figsize=(10,3.7));im=ax.imshow(mat,cmap='RdYlGn',vmin=.4,vmax=1.6)
    ax.set_xticks(range(7),['TP 25%','TP 50%','TP 75%','Row 50%','G+C+A','G+C','Up-only 50%']);ax.set_yticks(range(5),shapes);ax.set_ylabel('Sequence × hidden');ax.set_title('Three independent blocks: screening speedup over matched GPU')
    for i in range(5):
        for j in range(7):ax.text(j,i,f'{mat[i,j]:.2f}×',ha='center',va='center',color='#172534')
    fig.colorbar(im,ax=ax,label='Speedup (higher is better)');fig.tight_layout();save(fig,'screening')
    fig,axs=plt.subplots(1,3,figsize=(11,3.5))
    for ax,(m,h) in zip(axs,[(256,1024),(1024,1024),(1024,2048)]):
        rs=sorted([z for z in summary if z['group'].startswith('confirm/') and z['M']==m and z['H']==h and 'public_steel' in z['group']],key=lambda x:x['layers'])
        ax.plot([z['layers'] for z in rs],[z['gpu_ms'] for z in rs],'o-',label='GPU fused attention',color='#1e40af');ax.plot([z['layers'] for z in rs],[z['hetero_ms'] for z in rs],'s-',label='GPU + public ANE',color='#166534');ax.set_title(f'S={m}, H={h}');ax.set_xticks([1,2,3]);ax.set_xlabel('Independent blocks');ax.grid(alpha=.2)
    axs[0].set_ylabel('Median wall latency (ms)');axs[-1].legend(fontsize=8);fig.tight_layout();save(fig,'layer_scaling')
    fig,ax=plt.subplots(figsize=(9,4));shapes2=[(256,1024),(1024,1024),(256,2048),(1024,2048)]
    for i,(name,label,col) in enumerate([('public','Public parallel','#166534'),('private_gpu_bridge','Private + GPU layout','#6b21a8'),('public_serial','Public serial branches','#64748b')]):
        rs=[]
        for m,h in shapes2:
            rs.append(next((z for z in summary if z['group']==f'bridge/{m}_{h}_d64_L3_{name}'),None))
        vals=[z['speedup'] if z else np.nan for z in rs];errs=np.array([[max(0,z['speedup']-z['ci95'][0]) if z else 0 for z in rs],[max(0,z['ci95'][1]-z['speedup']) if z else 0 for z in rs]])
        ax.bar(np.arange(4)+(i-1)*.24,vals,width=.23,label=label,color=col,yerr=errs,capsize=2)
    ax.axhline(1,color='black',lw=1,ls='--');ax.set_xticks(range(4),[f'{m}×{h}' for m,h in shapes2]);ax.set_ylabel('Speedup over paired GPU');ax.set_xlabel('Sequence × hidden, 3 blocks');ax.legend(fontsize=9);fig.tight_layout();save(fig,'bridge_ablation')
    fig,ax=plt.subplots(figsize=(9,3.8));hs=[(64,256),(256,1024),(1024,1024)]
    for i,(backend,frac) in enumerate([('cpu',.25),('cpu',.5),('ane',.25),('ane',.5)]):
        vals=[]
        for m,h in hs:
            count=max(1,int(h//64*frac));z=next(z for z in rows if z['suite']=='heads' and z['stem']==f'{m}_{h}_{count}_{backend}_L3');vals.append(z['speedup'])
        ax.bar(np.arange(3)+(i-1.5)*.19,vals,.18,label=f'{"CPU" if backend=="cpu" else "Core ML"} {frac:.0%}')
    ax.axhline(1,color='black',ls='--',lw=1);ax.set_xticks(range(3),[f'{m}×{h}' for m,h in hs]);ax.set_ylim(0,1.15);ax.set_ylabel('Speedup vs matched materialized GPU');ax.legend(ncol=4,fontsize=8);ax.set_title('Complete causal attention-head splitting, 3 blocks');fig.tight_layout();save(fig,'head_parallel')
    fig,ax=plt.subplots(figsize=(7,4))
    for suite,col in [('screen','#64748b'),('confirm','#166534'),('bridge','#6b21a8')]:
        rr=[z for z in rows if z['suite']==suite];ax.scatter([z['nrmse'] for z in rr],[z['speedup'] for z in rr],s=14,alpha=.55,label=suite,color=col)
    ax.axvline(.01,color='#991b1b',ls='--',label='1% relative L2 gate');ax.axhline(1,color='black',lw=.8);ax.set_xlabel('Relative L2 vs matched FP16 GPU');ax.set_ylabel('Block-stack speedup');ax.legend(fontsize=8);fig.tight_layout();save(fig,'quality_frontier')
