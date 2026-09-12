"""Fill the technical report with computed evidence; keep the narrative reviewable."""
import json,statistics,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];N=ROOT/'notes';summary=json.loads((N/'summary.json').read_text());rows=json.loads((N/'measurements.json').read_text());S={x['group']:x for x in summary}
def table(headers,rs):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
def num(x):return f'{x:.3f}'
def ci(x):return f"{x['speedup']:.3f}× [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
repl={}
repl['METHOD_TABLE']=table(['项目','设置'],[['主机','Apple M4 Pro；14 CPU cores (10P+4E)、20 GPU cores、48 GB'],['系统','macOS 26.6 (25G72)；AC 电源在会话内核查'],['GPU','MPS FP16 GEMM；MPSGraph SDPA / Steel fused causal attention'],['CPU','常驻 BNNS FP16 projection worker；head 实验 FP32 BLAS'],['ANE','public CPU_AND_NE；private MIL + IOSurface，两种布局桥接'],['模型','batch 1，1–3 独立层，F=4H，S=64/256/1024，H=256/1024/2048 的五个组合'],['计时','host wall；每进程 AB/BA 配对；筛选30对，确认40对×3进程'],['复核','数值误差、独立 FP32/FP64、MLComputePlan、串行分支消融']])
repl['MAIN_TABLE']=table(['S×H','层数','GPU ms','Public 并行 ms','速度比 [95% CI]'],[[f'{m}×{h}',l,num(z['gpu_ms']),num(z['hetero_ms']),ci(z)] for m,h in [(64,256),(256,1024),(1024,1024),(256,2048),(1024,2048)] for l in [1,2,3] for z in [S[f'confirm/{m}_{h}_L{l}_public_steel']]])
repl['TRIPLE_TABLE']=table(['S×H，3 层','CPU 中间通道','相对配对 GPU','相对同图 CPU=0 的速度比变化'],[[f'{m}×2048',cf,ci(z),f"{(z['speedup']/S[f'triple_audit/{m}_2048_cpu0']['speedup']-1)*100:+.2f}%"] for m in [256,1024] for cf in [0,128,256,512] for z in [S[f'triple_audit/{m}_2048_cpu{cf}']]])
repl['HEAD_PLAN']='编译计划复核发现：64×256、1 个 head 后缀的主要算子首选 CPU；256×1024 和 1024×1024 的已核查后缀主要算子首选 ANE。因此图中 “ANE” 是 Core ML 候选分支名称，小尺寸不能归因于 ANE。较大形状即便首选 ANE，仍没有端到端收益。原始计划见 `raw/heads/*_plan.json`。'
repl['SP_TABLE']=table(['S×H，3 层','FFN 分支行数','主要算子首选设备','相对 GPU 速度比'],[[f'{m}×2048',m//2,'CPU' if m==256 else 'ANE',ci(S[f'sp_audit/{m}_2048'])] for m in [256,1024]])
repl['API_TABLE']=table(['S×H，3 层','Public，确认轮','Private CPU 布局，确认轮','Public，bridge 轮','Private GPU 布局，bridge 轮'],[[f'{m}×{h}',num(S[f'confirm/{m}_{h}_L3_public_steel']['speedup'])+'×',num(S[f'confirm/{m}_{h}_L3_private_steel']['speedup'])+'×',num(S[f'bridge/{m}_{h}_d64_L3_public']['speedup'])+'×',num(S[f'bridge/{m}_{h}_d64_L3_private_gpu_bridge']['speedup'])+'×'] for m,h in [(256,1024),(1024,1024),(256,2048),(1024,2048)]])
repl['OVERLAP_TABLE']=table(['S×H，3 层','Public 并行/GPU','Public 串行/GPU','配对归一化的并行/串行收益'],[[f'{m}×{h}',ci(p),ci(s),f"{p['speedup']/s['speedup']:.3f}×"] for m,h in [(256,1024),(1024,1024),(256,2048),(1024,2048)] for p,s in [(S[f'bridge/{m}_{h}_d64_L3_public'],S[f'bridge/{m}_{h}_d64_L3_public_serial'])]])
startup=[]
for m,h in [(256,1024),(1024,2048)]:
    exp=json.loads((N/f'raw/startup/{m}_{h}_export.json').read_text())[0]
    stats={}
    for kind in ['public','private']:
        texts=[json.loads((N/f'raw/startup/{m}_{h}_{kind}_r{r}.json').read_text())['stderr'] for r in range(3)]
        def vals(key):return [float(re.search(re.escape(key)+r'=([0-9.]+)',s)[1]) for s in texts]
        if kind=='public':stats['load']=vals('public_model_load_ms');stats['first_pub']=vals('public_first_prediction_ms')
        else:stats['compile']=vals('compile_ms');stats['priv_load']=vals('load_ms');stats['first_private']=vals('private_first_prediction_with_layout_ms')
    startup.append([f'{m}×{h}',f"{exp['compile_s']*1000:.2f}",'/'.join(f'{v:.2f}' for v in stats['load']),num(statistics.median(stats['first_pub'])),num(statistics.median(stats['compile'])),num(statistics.median(stats['priv_load'])),num(statistics.median(stats['first_private']))])
repl['STARTUP_TABLE']=table(['S×H，单层','public compile ms（一次）','public load ms（r0/r1/r2）','public first predict ms（中位）','private compile ms（中位）','private load ms（中位）','private first+layout ms（中位）'],startup)
repl['QUALITY_TABLE']=table(['S×H','层数','最大 relative L2 vs GPU','最大 relative L2 vs FP32','最小 cosine vs GPU'],[[f'{m}×{h}',l,f"{z['max_nrmse']:.6f}",f"{z['max_ref_nrmse']:.6f}",f"{z['min_cosine']:.6f}"] for m,h in [(1024,1024),(1024,2048)] for l in [1,2,3] for z in [S[f'confirm/{m}_{h}_L{l}_public_steel']]])
repl['HEAD_DIM_TABLE']=table(['S=256,H=1024，3 层','head 数','attention backend','Public 并行/GPU','Private GPU 布局/GPU'],[[f'd={d}',1024//d,'Steel' if d==64 else 'SDPA',ci(S[f'bridge/256_1024_d{d}_L3_public']),ci(S[f'bridge/256_1024_d{d}_L3_private_gpu_bridge'])] for d in [32,64,128]])
counts=collections.Counter(z['suite'] for z in rows);pairs=sum(len(z['gpu_samples']) for z in rows)
repl['COUNT_TEXT']=f"最终归档 **{len(rows)} 组有效进程测量、{pairs:,} 对 GPU/异构 wall 样本**。分组数量："+'；'.join(f'{k}={v}' for k,v in counts.items())+'。另有 3 次 CPU head 异常诊断复测（90 对），记录在 raw/head_retest；原始失败点仍保留在主矩阵。启动 smoke 不混入主统计。'
cleanup=json.loads((N/'raw/final_cleanup.json').read_text()) if (N/'raw/final_cleanup.json').exists() else None
logs=[]
for p in (N/'raw').glob('*/cleanup.jsonl'):logs += [json.loads(s) for s in p.read_text().splitlines() if s]
logical=sum(x['logical_bytes_removed'] for x in logs)
if cleanup:
    last=sum(x['logical_bytes'] for x in cleanup['final_cleanup']);repl['CLEANUP_TEXT']=f"每个配置测量后删除其专属 Core ML package、compiled model 和权重。累计清理 {logical/2**30:.2f} GiB **逻辑工件字节**（包含多轮重新生成，不能等同于一次释放的物理空间）。最后又删除剩余 smoke 工件及本次新建的 5 个 executable cache，共 {last/2**20:.2f} MiB 逻辑数据；已检查这些目录不存在。private 临时 model 目录在卸载时删除。清理明细见 [final_cleanup.json](raw/final_cleanup.json) 与各 suite 的 cleanup.jsonl。保留用户原有共享 Core ML/其他应用缓存、源代码和原始数据。磁盘空间受 APFS 与其他进程影响，不以全盘空闲变化估算模型大小。"
else:repl['CLEANUP_TEXT']='缓存清理尚未完成。'
s=(N/'REPORT.template.md').read_text()
for k,v in repl.items():s=s.replace('@@'+k+'@@',v)
assert '@@' not in s
(N/'REPORT.md').write_text(s)
(N/'campaign_summary.json').write_text(json.dumps({'valid_processes':len(rows),'paired_samples':pairs,'suite_counts':dict(counts),'largest_confirmed_group':max(summary,key=lambda x:x['speedup']),'logical_artifacts_cleaned_bytes':logical,'cleanup_complete':cleanup is not None},indent=2))
print('report written',len(s),'characters')
