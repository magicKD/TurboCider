# 22 · FP16 packed compute、逐 block N1 与较低内存速度目标（2026-10-02）

[目录](README.md) · [最快BF16目标](11-acceptance-profiles-and-feasibility.md#9-2026-10-02-用户补充fastest-bf16-low-memory-v1)

本阶段第一次在所测512²/portrait/seed42/4steps/cache-hit cell同时取得较低观测内存、
≤20% BF16 warm slowdown，以及同源全部block/最终latent N1通过。**不是48-case质量、
formal性能、whole-request cap、小容量实机或生产资格；目标仍active，默认路径未修改。**

## 1. 选择的是packed QMM，不是整模型dense展开

新显式`TURBOCIDER_Z_GGUF_COMPUTE` / runner `--packed-compute`，仅experimental
CPU-direct private GPU、compiled packed、allow_approximation；旧LoRA/ANE/guard/stream
冲突保持拒绝。缺省`native`完全保留。

最终所选`qmm_f16_down64` / `z-gpu-affine-qmm-f16-fp32-io-down64-v1`：

- main block大投影的输入显式转FP16，调用原packed QMM，结果恢复FP32。
  modulation小M=1投影、source浮点refiners、norm/residual/SiLU/attention保持原路线。
- gate/up分别恢复后才SiLU/相乘；down在FP32先除64、转FP16、QMM，再FP32乘64。
  不clamp、不再量化weights，不用A8，不宣传ANE或物理INT8。
- 不创建任何全矩阵dense权重，`dense_scope=none`、capacity upper=0。压缩weights
  不可变，因此恢复原whole-pass提交，不使用materialized-decode所需的逐main host wait。
- CPU-direct source仍由原verified lease/immutable bank/ledger拥有；metadata有限值与
  FP16重建保守upper在首次bank准备后验证，不以溢出值执行新的precision recipe。

新`TURBOCIDER_Z_GGUF_ALLOCATOR_CACHE_BYTES` / `--native-allocator-cache-bytes`仅接受
上述显式FP16实验，0…1GiB，捕获到model instance；缺省仍0。cache是单独资源/提示，
不藏入managed weights、更不是RAM cap。request RAII仍清理并恢复caller原cache limit。
本cell选择1GiB；256/512/1024MiB是分别记录的候选，未中途修改一个已冻结arm。

## 2. 新GPU组件与保留的负结果

`metal/affine_fp16.hpp`：g32 Q4/Q8直接FP32 `q*scale+bias`→一次FP16 RNE；没有整张
FP32中间矩阵，要求明确contiguous metadata、checked extent、源finite/range admission。
`affine_fp16_mpp.hpp`：half operands、FP32 destination MPP，tail stores有有效范围检查。
独立dynamic recipe按有效row maxabs/32752、minimum divisor归一化，在FP32恢复，不裁剪。

真实Metal18种Q4/Q8×row1/7/33×K32/64/128：scalar RNE（±0按合同canonicalize）、
same-order FP16 MM/QMM恢复、MPP同operand FP32对照、极大有限输入dynamic归一化、
非法shape/dtype/metadata/headroom通过。首次toy把u16 bit patterns作为数值构造array，
修正为真正float16 typed values；单元素metadata负例也修正为确实不同的shape，没有改门槛。

| 显式候选 | 真实Q8结果 | 决策 |
| --- | --- | --- |
| GPU decode→普通FP16 GEMM +FP32 glue | final relL2 .004161，通过final子门；热诊断约5.51s | 不因子门通过而称更快 |
| GPU decode→MPP FP32 destination | final relL2 .003953；热诊断约5.41s | 未取得速度目标 |
| packed FP16 QMM，原refiners | final relL2 .005181；后续全部block N1通过 | 本阶段选择 |
| packed QMM +普通FP16 refiner | nonfinite latent、无PNG | 保留失败，不clamp/推广 |
| packed QMM +dynamic FP16 MPP refiner | final relL2 .013935通过；中间block N1失败 | 即使局部较快也不推广 |

上表诊断带observer/dump且使用各自实际binary，不能混入最终同binary速度表。
materialized-decode每main eval限制dense liveness；packed QMM则没有此dense需求。
所有失败/旧候选保留独立recipe和raw artifacts，不覆盖成成功的最终参数。

## 3. 真正独立的逐block控制，而不是只检查最终latent

`--validate-source-blocks`仅diagnostic；timing/memory和环境强行开启均拒绝。
reference从相同initial noise/conditioning开始，独立逐step Euler演进，执行原同源
`z-mlx-compat-affine-v1`，不是把candidate input作为reference input。
每pass仅暂存34个**activation** outputs，candidate按同顺序比较后释放，不存dense weights。
source最终latent字节与已有独立native控制exact，证明没有使用不同source轨迹。

新`core/tensor_metrics.hpp`在CPU FP64累加relL2/cosine/max_norm_error，valid image和
valid caption rows单独取span、排除padding；零参考/nonfinite/geometry负例fail-closed。
每cell136个block comparisons +1个final，共137条原始数值，不平均掉失败层。

| source/candidate cell | 每block N1 | 最大block relL2 | final N1 |
| --- | --- | ---: | --- |
| Q8 packed QMM FP16，原refiners | 全136通过 | .00504139 | 通过 |
| Q4 packed QMM FP16，原refiners | 全136通过 | .00522185 | 通过 |
| Q8 dynamic refiner额外变换 | 失败 | .0180361 | 仍通过，但不足以接受 |

dynamic recipe在step2的layers28/29和step3多处relL2/max_norm_error超门，说明只看
final会误判。全部阈值仍是11的N1，没有为更快的候选放宽。

## 4. 同一实际binary的独立速度/内存screen

最终测量library=`899dee266ad8adad6e477cd42beb77f87fec6692c48d8288d5aecdae327e50e1`；
M4 Max 64GiB、macOS26.6.2、MLX0.32.0。原BF16 encoder/VAE/tokenizer内容相同，
GPU tuning与更快BF16一致，34 valid tokens / combined padded1088；cache-hit、每arm
4个热计时 +1保留warmup，另外新进程10ms memory（含warmup/kernel lifetime peak）。

| arm | 热全请求median | 观测process peak bytes | 对BF16 wall / memory |
| --- | ---: | ---: | --- |
| 更快纯GPU BF16 | 3.734826s | 23,010,829,984 | 1 / 1 |
| Q8 packed FP16 QMM +1GiB cache | 4.415047s | 12,244,900,104 | 1.18213 / .53214 |
| Q4 packed FP16 QMM +1GiB cache | 4.399699s | 9,765,032,280 | 1.17802 / .42437 |

两者在本cell的≤1.20/lower-memory经验目标通过，≤1.10优选门未通过，也不符合旧
`accelerated≤.95`门。Q8比stage21约5.33s显著缩短，但不称普遍加速或完整资格。
buffer10%时两者均observed-fit于16GB/16GiB，不能据此认证真实小机器。
Q4的9.765GB仍超过10GB扣余量9GB，也略超过10GiB扣余量9.664GB；不得说10GB已支持。
512MiB Q4 cache另作新候选/独立screen，不与本行混池。6/8GB仍需要streamed/tiles路径。
512MiB新候选实测median4.484822s、peak9,123,352,944bytes，wall比1.200811，**超过
1.20硬门，保留失败而不四舍五入成通过**。虽然可observed-fit于扣10%后的10GiB，
仍不满足本cell同时速度/容量目标，也超过10GB扣余量的9GB。
[512MiB负回执](validation/q4-fp16-qmm-cache512-bf16-screen-20261002.json)。

[Q8 screen](validation/q8-fp16-qmm-bf16-screen-20261002.json) ·
[Q4 screen](validation/q4-fp16-qmm-bf16-screen-20261002.json) ·
[逐block、成功/负结果与恢复](validation/gpu-fp16-compute-diagnostics-20261002.json)。
最高memory gap约15.5ms，swapout0；managed、allocator hint与process观测分账，
仍无driver/全组件连续时间upper，`whole_request_memory=unknown`。

## 5. 生命周期与未完成范围

最优Q8路径tensor180取消status2/无PNG、同engine重试status0且所有张量exact；
不同prompt A→B→A每次先释放重建bank，回到A全部张量/PNG exact。cache hint不改变
数字结果；普通构建实际generate门拒绝这些显式实验，不能从create-only误判初始化。
CPU/实际Metal tests、source-reference hash checks、CLI/screen污染负例和独立build保存证据。
最终host/CLI/identity/screen矩阵24项通过；实验和普通build通过，实际Metal18形状与
普通库8条初始化/generate实验gate通过。没有把GPU/模型opt-in skip算成功。

本阶段只覆盖512² portrait/seed42。1024²/长文本/多seeds、48-case媒体、人审、正式
ABBA/24样本/CI/p90/drift、cold与artifact verification一致分账、真实预算/required-site
closure仍须完成。cold first requests约8–10s，不冒充warm表现；BF16与GGUF首次原生
content-proof范围不同，不能据当前cold数字授予同策略cold资格。
产品planner/presets、GPU raw-GGUF bounded/tile供给、更低6/8GB目标、IQ/更多组件、
ConvRot旋转域及runtime ANE W8A8、M5静态backend接口/实际整数验证仍保留在完整目标中。
