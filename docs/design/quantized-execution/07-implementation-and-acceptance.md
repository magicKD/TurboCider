# 07 · 实施工作包与验收合同

[目录](README.md) · [本轮证据](08-experiments-and-delivery.md)

审阅补充：本页保留工作包编号；里程碑范围以 [09](09-release-scope-and-component-contracts.md)、
具体接入以 [10](10-execution-and-product-integration.md)、冻结验收政策以
[11](11-acceptance-profiles-and-feasibility.md) 为准。原“建议”不能替代正式 campaign profile。

## 1. 顺序与依赖

```text
R0: P0 + P4 contracts / bind first fixture and acceptance profile
  → R1: P1a + P1b(Q8/float subset) + P2(single/double slot) + P3a + P4 experimental
  → R2: P1b(Q4/Q5/K) + P3b encoder + mixed-component qualification
  → R3: P2(p=2 / streamed source / tiles) + P4 public qualification

C1 shared ConvRot rotation → C2 butterfly / bounded expansion
                           → C3 A8 oracle / calibration
                                      ├→ G1 M5 GPU primitive → G2 full model
                                      ├→ A1 static ANE (independent S1/S2)
                                      └→ A2 dynamic INT8 feasibility (+ small frozen control)
P1/P2 → A0 runtime FP16 direct slot staging; A0 + passed A2 → A3 integrated runtime
```

格式/低内存主线不等待M5或动态ANE。每个工作包单独提交、默认不变、可单独回滚；
不要把所有候选组合成无法归因的一个“最快模式”。以下新文件名是拟议，不代表已存在。

## 2. 逐文件工作包

| 包 | 主要位置 | 交付 / 退出条件 |
| --- | --- | --- |
| P0 | `native/backends/mlx.cpp`，模型profile，validation工具 | dtype/形状/格式/dispatch trace；保存matched基线；不改变numerics |
| P1a | `native/core/gguf.hpp`，拟议 `native/core/gguf_directory.*` | 完整checked目录、split/source lease、安全fixture；旧不支持type仍拒绝generate |
| P1b | 拟议 `native/core/gguf_decode.*` | Q8+浮点先纵切；再Q4/Q5/Q6/K；IQ4另批；CPU oracle/目标直填 |
| P2a | `native/runtime/streaming/{layout,slot_pool,io_executor,source_lease}.*` | 为packed/read/decode目标建立descriptor/live interval，不复制调度器 |
| P2b | `mlx_weight_pager.*` 或新typed decode pager | 无转换旧pager不被暗改；0/1/2前瞻、分片与GPU fence契约 |
| P2c | `native/backends/mlx.*`，Z compiled segments | 解码dtype明确、dense参数图复用、source表示不影响图选择 |
| P3a | `native/models/z_image/*` | 先Q8+浮点mixed DiT完整生成，再扩大Q4_K；不是全decoder做完才接模型 |
| P3b | `native/components/text/qwen3.*` | encoder GGUF source、embedding gather、hidden taps、阶段释放 |
| P3c | `native/models/{flux2,qwen21}`、UMT5/Gemma | 按02逐组件扩大，不复制tokenizer/attention语义 |
| P4 | `native/core/contracts.hpp`、request/profile parser、streaming public/resolved/catalog、model probe、CLI/API/App bridge | typed配置、冲突、source闭包、计划身份；先实验入口，再独立正式资格；不删除旧门禁来放行 |
| C1 | `native/models/z_image/ffn.hpp`、`z_image.cpp` | 本轮新增共享H的opt-in，组件实测见08 |
| C2 | `native/backends/mlx.cpp`、typed decode recipes | butterfly/有限逆旋转两arm；memory+质量+full-request gate |
| C3 | 拟议 `tools/validation/convrot_a8_*` | int32 reference、A8校准与outlier数据，先不接默认 |
| G1 | 拟议 `native/backends/metal_w8a8/*` | capability/MPP独立primitive；无M5时止于compile/oracle |
| G2 | `Weights` projection dispatch、model adapter | source codes不重复量化，M5全模型/低内存资格后接入 |
| A0 | `ane_runtime_quant.hpp`、typed source、exporter | 多格式直接填FP16槽；正确ConvRot变换，不删除门禁后直接用 |
| A1 | `tools/coreml/export_z_image.py`、校准/manifest | 原生旋转S1与derotate S2独立artifact；静态bank内存验证 |
| A2 | 独立runtime INT8小图export/probe | 同图WA/WB/WA、scale、placement及独立arithmetic证据、含staging速度；11规定有限搜索和退出 |
| A3 | `ane_runtime.*`、`ane_ffn.*`、`ane_scheduler.hpp` | versioned INT8 graph/slots、共享worker、完整GPU fallback，低内存全请求 |

每包都更新08或后续实际实验记录：实现状态、执行命令、binary/source身份、数据、
未完成项。不能仅用文档勾选或mock测试标记硬件交付。

每个 PR 附对应 R*/P*、新增支持与仍拒绝的组合、测试 ID/命令、profile/receipt
digest 和回滚边界。P4 的实验生成通过不等于 catalog 可发布；没有必要 fixture 的
部分保持 not_run，但不阻止已具备依赖的子包实施。

## 3. 测试分层

### L0 · 不需要模型/设备的格式与数学

- 每type边界值、随机codes、负/零scale、minimum offset、high bits、codebook。
- 故意混合Q4_K/Q6_K/浮点，文件名伪装，split缺片、重复、非法offset/overflow。
- CPU decoder与固定GGML oracle对齐，分别验证FP32值与目标16-bit舍入。
- 非连续stride、部分row/K slice、量化block tail、输出guard bytes、source替换。
- H256约定、正交/逆变换、signed -128、zero row、per-token/group尺度数学。
- INT32点积exact oracle；group scales必须在部分和正确位置应用。
- SwiGLU切分数学：K部分和不能提前SiLU；channel split down/bias/ConvRot组边界。
- 预算checked arithmetic、0/1/2前瞻槽数、异构field容量、超大层拒绝。

### L1 · Slot / lifetime / 故障仿真

随机I/O完成顺序、stale ticket、重复完成、两个reader队列、cancel每个状态、
短读/坏scale、decode异常、GPU fence失败、ANE partial chunk失败、drain失败。
独立fake-access log证明reader未结束不会复用；不是只看框架自己发的状态事件。
证明默认模式从不物化全模型dense，不仅仅“测试结束后缓存为空”。
必须包含 Qwen3 四层 lazy graph 与两槽的冲突、tap 输出独立生命周期、read buffer
在 GPU decode 完成前不可复用，以及 ANE 失败恢复不重复累加/不超 normal+recovery 计划。

### L2 · 真实设备算子/单层

同shape/dtype/tensor，交错ABBA，显式warmup和同步；同时保存wall/GPU time及
非重叠计时定义。测试CPUdecode/GPUdecode/direct、shared/unshared、dense/butterfly、
W8A16/W8A8、static/runtime、不同tile/chunk。包含staging/量化/fold/join。
禁止把只计GEMM的结果作为含解码FFN或整模型速度。

### L3 · 真实组件与完整生成

- encoder token IDs、逐层hidden/final conditioning；不只看能否生成文字。
- DiT不同step的FFN/attention/latent，最终decode图片或视频。
- cold首张和同进程warm分开，prompt cache hit/miss分开。
- encoder量化、DiT量化、两者量化三组消融，不能一次全改。
- checkpoint/LoRA/base切换、同shape不同源、512→1024→512、长短prompt切换。
- 每次实际backend/覆盖率/失败回退一致；fallback多不冒充有效ANE加速。

### L4 · 受限内存全生命周期

load/parse/pack→encoder→DiT→VAE→PNG/视频→retained session/release都采样。
记录独立进程树、driver/framework可观测项、swap/compression、峰值采样间隔与盲区。
在64GB机上限制软件预算只验证该预算计划，**不等价于16GB真实设备认证**。
无真实低容量设备时明确标budget-constrained-on-large-device。

## 4. 最小实验矩阵和逐步筛选

不要一开始跑所有组合。先筛选再扩大：

| 轴 | 最小覆盖 |
| --- | --- |
| 格式 | Q8_0、Q4_0、Q4_K_M实际mixed、Q5_K_M、Q6_K，随后IQ4 |
| 组件 | Z DiT，Qwen3 encoder，二者组合；其他架构单独推进 |
| 形状 | M=1/33/1056/4128及真实encoder短/长token；tail/K-block边界 |
| 内存 | 只能容纳投影tile、一层、两槽、三槽四档；实际bytes由tensor表计算 |
| 预取 | 0、1、2；每个都有managed capacity和实际峰值 |
| source | packed-resident、packed-streamed；OS cache warm/cold状态声明 |
| GPU | 现有packed、bounded dense、ConvRot共享H；M5候选另机 |
| ANE | off、runtime FP16、static W8A8；runtime W8A8只在A2通过后 |
| 输入质量 | 人像/手、文字/招牌、透明玻璃、重复几何、多物体、长prompt、不同seed |

全图质量首轮建议至少8类prompt×3 seeds，在512/1024两个shape中逐步覆盖；视频
另加运动/时序，不把静态图质量资格继承给视频。数量是campaign计划，不是本轮已跑。

## 5. 四个独立 gate

### Correctness

decoder/schema/lifetime必须严格通过。仅删除重复旋转的优化以同路线exact为目标。
kernel新增A8以相同量化数学oracle验证；浮点epilogue允许预先声明的舍入误差。
非法权重/scale、非有限计算结果、越界/缺tensor均为失败，不能归为“量化近似”。
attention mask 按算法定义使用的 -Inf 不在非法数据集合中。D0/D1/N1/N2 的具体
oracle、误差及舍入判定以 11 为准。

### Quality

先锁定source-quant与原BF16两个参考；layer relL2、cosine、max error、clip比例
帮助定位，但不能用一个局部阈值替代最终图像语义。保存图片、latent差异、固定
prompt/seed和盲测记录。允许门槛按模型/近似类别制定，但必须在看after之前冻结，
不能为某候选临时放宽。encoder变化尤其要检查语义而非只有PSNR。

### Memory

管理内存严格符合plan，0/1/2前瞻没有额外隐藏dense bank；source/driver未知上界
意味着不能宣称hard-cap认证。完整请求峰值与同预算baseline比较，不能只报MLX峰值。
swapout/压力异常按既有guard fail/inconclusive规则，不清除异常样本后宣布PASS。
新增ANE图若挤占slots导致整体更慢/超预算，应选择GPU，不为保留ANE而更改用户目标。

### Performance

筛选可从每 arm 8 个 warm 样本开始；Z 正式认证按 11 使用两个独立会话、每 arm
24 个 warm request 与 5 个 cold-process 样本。同 source/steps/seed/token/cache/
预算/库，只允许声明并已通过质量门的 profile 差异；A8 不伪称与 A16 完全相同精度。
正式计时关闭详细同步 profile，保存 median/p90、原始样本、漂移与配对置信区间。

推广门具体为 warm median 改善至少5%、比值95% CI上界<1、p90回归≤5%、cold
与load/prepare median回归各≤10%，详见11的计算和异常规则。这是冻结政策，
不是已经满足的成绩。省内存但更慢可保留为明确low-memory fallback，不贴“加速”
标签；不同容量resident基线只作背景，不作因果证明。

## 6. 性能 receipt 合同（拟议字段）

```text
identity: source hashes, per-tensor types, tokenizer/adapter/LoRA,
          OS/device/RAM, MLX/CoreML/shader/library hashes
plan: component stages, source residency, slots/fields, lookahead,
      M/K/N, tile/chunk/split, rotation/scale/dtype recipes
actual: selected kernels, fallback reasons/counts, compiled graph counts,
        observed placement and INT8 evidence (or unknown)
time: parse, IO, decode, repack, quantize, rotate, GEMM,
      stage, predict, join, exposed_wait, encoder, denoise, VAE, request
memory: unique backing capacity, reserved/retained, source pages,
        dense/ANE/scratch bytes, process-tree peaks, driver unknowns,
        sample interval, swap/compression deltas
quality: layer/caption/image/latent metrics, image paths/hashes, visual verdict
samples: all raw trials, warmups, order, exclusions declared in advance
```

failed/inconclusive/skipped分别保存；没有M5的run必须是not-run，不是mock-passed。
生成图片可以不同于BF16，但不能遗漏LoRA、mask、hidden tap或改变有效token数。

## 7. 发布、回滚与停止投入

- 默认路径不变，实验需显式选择；只有exact device/OS/model/shape/budget tuple
  通过资格才注册自动路线。旧历史M5/ANE配置不能为新INT8方式背书。
- mode/graph/decoder版本进入cache key；fallback不使用过期weight generation。
- 全部artifact保存source manifest，可重新编译；不修改用户checkpoint或reference仓库。
- 取消、安全drain、错误回退和低内存路线同样要在发行包验证。
- 若动态INT8不能真正lower、或桥接吞掉收益，停止A2/A3；继续A0和GPU主线。
- 若ConvRot逆旋转每步成本更大，保留在线H；若2层前瞻因内存压力更慢，保留1层。
- 不以“已经花了很多时间”为由推广负收益优化；所有负结果进入08/后续实验账本。
