# Qwen base512：显式动态W的共享GPU complement编译

后续已完成[完整40步生成/编辑正反序验收](qwen-base512-compiled-complement-session-2026-10-19.md)：
组件信号没有成为稳健整图收益，原fixed FFN/QKV基线保留。下文仍是原组件实验。

2026-10-19，M4 Max/64GiB，macOS26.6.2/25G83，原BF16 base。仅Qwen Image 2.1
512²文生图/图生图，不涉及LoRA/student/Z Image。

新增并完成了GPU complement及完整混合FFN的profiling工程。共享compiled函数有
约0.5%的局部信号，受测输出全部逐位一致，但还没有attention/QKV组合、完整
DiT block或40步图片收益，因此未接入主native、App或默认。总体目标仍ACTIVE。

## 来自实际调用点的假设

`pipeline.cpp`的base无LoRA `ordinary_gpu`回调分别执行project_slice(gate/up)、
`silu(g)*u`、原pitch BF16×BF16→F32 down。silu本身已经是compiled函数，不能
称整条路线全是eager；纯GPU fallback的whole-compiled图也不是这个回调。

候选将X、完整原gate_up W和完整原down W显式传入一个共享mx::compile函数。
固定M1024/H4096/F12288、GPU份额6144，保留各BF16阶段和F32 partial，不cast/
拼接/复制W，不捕获每层权重、不建立逐层compiled函数或新ANE权重bank。准入
在调用compiled函数前检查shape/dtype，七项非法输入负控已实际拒绝。

原对照确实调用当前native Weights::project_slice和project_base_slice_fp32。
另逐位核对未compile的显式输入body与原native回调，防止换成较慢数学参考来
制造分母。不是把此前prototype的compiled runtime_ffn::channels当原pipeline。

## 范围与生命周期

使用既有真实base40请求首decode layer0的冻结FFN输入A/B，原请求为seed73、
512²、生成或一个标准1024参考的编辑。这里只重放这些数组，没有重新执行40步
生成图片。B是原捕获的扰动，不是新seed/prompt。W1来自实际layer1原BF16权重，
并不构成自然连续的第二个transformer block。

GPU-only两arm完成hidden与F32 partial；混合两arm调用同一个真实HybridFfn
owner，完整包含plan、逐层权重staging、输入准备、GPU/ANE执行、输出恢复、join
及最终owned BF16输出完成。输入在计时前ready，因此后者仍不是attention窗口
下的完整block，也没有one-hop QKV producer的并行干扰。

每arm两次warmup、轮转访问，交替A/B并周期换W0/W1。计时内不做CPU数组扫描，
保留末输出、A/B/A和W0/W1/W0恢复；权重和输入owner存活至真实完成/drain。
混合图固定C6144、sigmoid-complement、F32 join，不开权重cache或额外prefetch。

## 同窗结果

单位ms，中位数；不合并stock与timeline，也不把组件比例乘到完整40步：

| 边界 | 窗口 | 原回调 | 共享compiled | 减少 |
| --- | --- | ---: | ---: | ---: |
| GPU complement | 生成16 | 10.685042 | 10.630875 | 0.507% |
| GPU complement | 生成32 | 10.685355 | 10.630083 | 0.517% |
| GPU complement | 编辑32 | 10.683875 | 10.622751 | 0.572% |
| GPU complement | 生成timeline16 | 10.683646 | 10.628021 | 0.521% |
| 完整混合FFN | 生成16 pilot | 12.390501 | 12.353875 | 0.296% |
| 完整混合FFN | 生成32 | 12.405062 | 12.337146 | 0.547% |
| 完整混合FFN | 编辑32 | 12.404938 | 12.337167 | 0.546% |
| 完整混合FFN | 生成timeline16 | 12.391042 | 12.329646 | 0.495% |

这是约0.04–0.07ms级信号。GPU timeline每个sample仍有实际完成CB，合计72个，
不是“一次compile后只剩一个CB”的证据。未采dispatch/硬件counters，无法唯一
归因为融合、CPU构图、参数检查或GPU算术；不将mx::compile等同GPU永久replay。

## 完整数组、原权重与独立审计

八窗共384 measured调用（GPU192＋混合192）、100份完整数组；混合累计240次
实际workload预测。全部新旧GPU complement hidden/F32 partial、全部原/新
混合FFN输出均bits exact。原GPU输出还与冻结历史完整数组逐位一致，历史时间
不参与排名。同case独立repeat/timeline的全部数组一致。

独立auditor核对两层原gate_up/down四份完整BF16 payload，共576MiB，完整输入/
输出extent和finite、全部source/库/native ABI、原capture请求，并重算每个GPU
窗192个完整K6144的F64 down坐标见证，共768个。不是全CPU FFN oracle，也没有
32层、全部prompt或图片质量资格。混合路线对完整GPU FFN的局部relative-L2约
0.0046–0.0074；这来自既有W8近似，不是本次compile新增误差，也非图片误差。

只构造一份显式W compiled函数；受测X/W切换后trace count始终1。这个计数证明
graph body未重trace，不是Metal shader/library/pipeline编译次数。混合独立
timeline观测一个稳定ANE model、44个实际workload evaluate、hot compile/load0；
同一固定图实际服务W0/W1及两输入。这里不证明物理INT8 MAC或硬件overlap。

首混合pilot后仅收紧collector对incremental elements的校验，设备代码/库没变；
旧source.json已封存到`outputs/qwen-compiled-complement-hybrid-pilot-tools-v1-20261019.tar.gz`，
auditor按当前或该归档的精确bytes绑定，未放宽数值门槛。首个主机时间线负控遗漏
模拟校准事件，补齐fixture后通过；没有修改真实observer或删除设备事件。

## 内存、磁盘和交付

原始process-tree JSONL hash-chain已独立重算。GPU组件峰值约1.200–1.201GiB，
混合组件约1.235–1.236GiB，不是全模型内存。首GPU gen16窗记录系统swap-in
589824 bytes，另外七窗swap-in/out0；不把系统活动唯一归因于所测模型。
存在竞争推理CPU负载，全部qualification_passed=false，无独占/thermal资格。
全部自有设备窗串行，build/test/CPU数组审计只在设备窗外，不停止ComfyUI。

两种边界各有三个相同生成数组，经SHA及cmp后硬链接，所有原路径/bytes/hash
保留，编辑独立。回收约544MiB重复存储，不是推理速度/驻留内存收益。磁盘仍约
2.1GiB，后续大输出/native build之前须先预算，继续维持2GiB reserve。

- GPU工具：`compiled_complement.hpp`、`compiled_complement_{probe.mm,collect.py}`。
- 混合工具：`compiled_complement_hybrid_{probe.mm,collect.py}`。
- 独立审计：`compiled_complement_audit.py`；旧source封存：`seal_sources.py`。
- helpers：`build/qwen-compiled-complement-probe-v1-20261019`和
  `build/qwen-compiled-complement-hybrid-probe-v1-20261019`，分别对应不同CAPI签名。
- 审计：`outputs/qwen-compiled-complement-full-audit-v1-20261019/audit.json`，8窗complete。
- 177项profiler host测试通过：`outputs/qwen-compiled-complement-host-tests-v1-20261019.log`。
- 主native仍为`build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib`，661 inputs，
  SHA256 `7da7a7125bd7cb401344bf885c2425ec7eacc14d81b0563fd45317a94b4b260a`，未改源码/ABI。
- 工具/测试/本报告：`outputs/qwen-compiled-complement-tools-v1-20261019.tar.gz`；
  摘要：`outputs/qwen-compiled-complement-artifacts-v1-20261019.json`。

```sh
TURBOCIDER_NATIVE_OUT=build/FRESH-COMPILED-COMPLEMENT \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen-base-gpu-dual-v1-20261019 \
  bash tools/native/build_qwen_compiled_complement_probe.sh
.venv/bin/python -B tools/profiling/qwen_base/compiled_complement_collect.py \
  --library build/FRESH-COMPILED-COMPLEMENT/libqwen_compiled_complement_probe.dylib \
  --native-library build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib \
  --case generate --repeats 32 --output outputs/FRESH-COMPILED-COMPLEMENT
```

完整混合FFN改用对应`build_qwen_compiled_complement_hybrid_probe.sh`、
`compiled_complement_hybrid_collect.py`及其库；编辑改`--case edit`。timeline单独
进程；所有设备窗退出后，auditor使用重复的`--gpu-cohort ROOT HELPER`、
`--hybrid-cohort ROOT HELPER`，配native库/source archive、pilot tool archive与
fresh `--output`，独立重算全部数据，不用历史时间替代同窗强控制。

## 决策及下一道gate

不从此0.5%候选宣称新的整图纪录或默认采用。下一道gate应是带真实attention和
Private QKV lookahead的完整block/joint请求，保留一份显式runtime-W函数、原
BF16/F32算术、前缀配方/取消/失败drain和固定ANE model契约；先检查GPU小收益
是否转移成QKV weight wait。只有完整40步生成/编辑重复保留净收益和质量才采用。

完整40步仍保留此前fixed FFN6144＋Private QKV512＋one-hop lookahead强基线，
GPU-dual OFF；生成加text-prefix snapshot与首步GPU QKV，编辑用runtime-prefix
snapshot。先前约28.84s/36.22s只是原窗口的热请求数据，不是本轮新测量。
