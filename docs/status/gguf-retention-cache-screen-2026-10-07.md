# GGUF resident retention / allocator cache 与整请求诊断

2026-10-07，Asia/Singapore，M4 Max 64GB / macOS26.6.2。接续
[内容代次缓存与 GPU-first 预取](raw-content-scale-and-prefetch-order-2026-10-07.md)。
本轮向整请求盈利推进，但没有完成全部双后端/Z/Qwen/base/LoRA/
512/1024/GGUF/ConvRot/视觉/严格 matched GPU 要求，整体目标保持 active。

## 实现与工具入口

CPU-direct GGUF 的显式 allocator cache hint 从仅 GPU-FP16 实验扩展到
native affine GPU/runtime consumer，继续要求 experimental CPU-direct
import、0...1GiB。default仍0；没有改变 W8/A8、模型、adapter或精度，
没有新增 full dense weights、raw常驻或 retained allocator bins。
既有 request cache scope 在结束时清理、恢复原 hint；hint不是硬 RAM cap。
原 source/ledger/finite/fallback 检查不放松。

新增 `runtime_ane_gguf.py`，screen显式支持 CPU-direct GGUF 真实
inference-time LoRA、affine/raw来源、packed retention和cache hint，
GPU/runtime两臂共用这些设置。只在验证原 CPU-direct GPU label、native
eager consumer、相同 source/import plan/recipe、真实allocation/read/
warm reuse后，映射共享validator的canonical GPU label；stdout原label
和回执不改写。raw还要求真实read、256MiB window/drain和retained
bank的递增流量；GPU/affine不能假称raw消费。frozen/Qwen/dense混用
这些选项拒绝，merged LoRA仍不允许。

screen亦允许已经实现的Public W8 prefetch和after-GPU顺序对照，验证
实际prefetch/placement receipt；没有放宽Public row/basis、GPU IO、
weight/LoRA执行、load/memory gate，也不改CLI/native默认route。

新增参数：`--gguf-import cpu_direct --gguf-ane-source affine|raw
--gguf-retain-packed --gguf-allocator-cache-bytes <0...1073741824>`。
显式cache环境 `TURBOCIDER_Z_GGUF_ALLOCATOR_CACHE_BYTES`；公共后端仍默认，
Private仍opt-in，沒有物理overlap/native INT8 MAC/FP32 ANE arithmetic声明。

## 两个严格窗口：正确拒绝，不能拼接

Q4、1024、8步、fox/seed42、真实distill patch strength1/238 applied
projections，Public rows1056、chunks1/fixed-async1、GPU blocks0,1,2、
prefetch0、affine来源、packed retention1、cache0。一冷三热，每臂独立
process，连续load+100ms process-tree memory采样。

| order | 实际完成arm | load samples / busy | accepted comparison trials |
| --- | --- | --- | ---: |
| GPU→runtime | GPU四次，runtime未开始 | 332 / 10 | 0 |
| runtime→GPU | runtime四次，GPU未开始 | 335 / 9 | 0 |

均因竞争推理CPU负载退出失败，summary仍incomplete。保留所有samples/
stderr/PNG/requests，不能移除busy samples、拼接被拒绝arms、结束外部
ComfyUI或降低阈值。没有正式speed ratio。

目录 `outputs/retained-gguf-q4-1024-lora-public-{forward3,reverse3}-20261007/`。
另对原stdout做来源/retention校验：两臂cold读取4,509,395,072 bytes，
warm三次 `reused_packed_bank=true`、request source read0/load0；真实
retention成立，但不把该检查当成正式load资格。runtime calls累计
232/464/696/928，非GPU-only。被拒绝的时间仅保留原始诊断，不比较倍率。

## 1GiB cache hint：同库双向诊断

新v1 Public库，GPU/runtime均retention1、native affine compute、同真实
LoRA/request/精度；一冷两热、独立进程串行，两次order。
prefetch0，GPU blocks0,1,2只在runtime设置，完整GPU基线不加bridge。
100ms process-tree memory采样，但**未请求continuous load资格**，
下表是诊断，不是正式加速/产品qualification。

| direction / arm | cold request s | 两次warm request s | warm median s |
| --- | ---: | --- | ---: |
| forward GPU | 47.433990 | 45.105217 / 45.147158 | 45.126187 |
| forward Public | 48.864783 | 42.519130 / 42.967701 | 42.743415 |
| reverse Public | 48.432872 | 42.610353 / 42.905821 | 42.758087 |
| reverse GPU | 47.667236 | 45.192454 / 45.257851 | 45.225152 |

双向热请求候选较短，冷请求候选仍较慢；不挑更好的方向，不改default。
这些GPU是当前native affine consumer，并未完成与全部GPU-only FP16/
compiled packed/dense decode recipes的最优baseline校准。cache0严格
窗口属于另一保留库/时刻，不能把与其差值全部归因于cache hint。
下一步需要same-binary cache0/1GiB paired controls和严格load窗口。

四个trial memory验证complete，无gap违规、swap-in/out bytes均0。
process-tree phys-footprint peaks依上表为18,325,950,976 /18,649,993,872 /
18,648,994,424 /18,543,727,128 bytes。scope包含load/cold/warm/exit，
不归因external services/driver，也不声称整个系统或特定请求RAM上限。

每臂三张PNG exact，两方向对应臂也exact。Public候选还与此前raw
protected 1024 LoRA PNG exact，沿用其有限agent视觉观察：整体非常
接近、毛发/雪粒/局部背景有小差异；不是多prompt/seed/用户批准。
原N1 fail、qualification_passed=false不改。238 applied projections、
232calls/request、retry/failure/fallback0、headroom1均保留；warm导入
read0/load0，实际hint1GiB，source/import身份跨arms一致。

目录 `outputs/retained-gguf-cache1g-q4-1024-lora-public-diagnostic{-reverse}-20261007/`。
forward/reverse summary SHA256：

```text
2b82b708d2901f33487743970c539df6c6c6025e1c2ff1fea0b2ea4eae8d1c2d
b480169376d84f779f12ec57a24f8e0600df56bdb70945c6258b24aa36d7e182
```

## 回归、provenance、仍待完成

90项host/tool tests通过（含新增3项GGUF evidence tests），15项Public
graph/Core ML/MLX/bank/identity tests通过，隔离coremltools9的5项Public
W8 exporter/actualdevice/recovery tests通过，3项Private native
channel/prepared MLX tests通过，无skip。不是全仓或全模型矩阵通过。

新Public library SHA256
`108fd63c94e5f9318c53768e15e3ae56fffcda8829d52edb42cae8641670a444`，
build ID `tc-runtime-build-v1-4de18cb31c18730b4802cb6a49b1640f0fc0f8ca2c157e2dfcabc586cc3fdbe9`。
新Private library SHA256
`2bbdcf18928b7cce24fb5566ba0dfaad07b21174df583397788de4469cc90650`，
build ID `tc-runtime-build-v1-86394c40a5c6682d41787aa53ae29577c580455d75704e9a4b1088c91aab4259`。
两份487 source inputs独立匹配/seal一致，Public实际库release guard通过。
严格cache0窗口仍用上一轮`0e694af1…`库，不称其匹配新guard；保留
构建含原未提交草稿，commit只独立stage本轮单行native变更/工具/记录。

继续Private整模型raw、Q8、其他尺寸/模型/adapter、多prompt视觉、
每operation盈利选择和GGUF/GPU更优baseline；完整目标不缩成这里一个
cell或一个热请求指标。外部负载只阻止正式资格，不阻止其余实现/分析。
