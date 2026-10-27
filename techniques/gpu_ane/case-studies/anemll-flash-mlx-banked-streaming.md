# anemll-flash-mlx：稳定专家bank、I/O与失败发布

<!-- knowledge
id: technique-gpu-ane-case-studies-anemll-flash-mlx-banked-streaming
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [本轮摘要](../../MAINTENANCE.md) · [固定来源](../sources.md) · CPU数据 · 计划

2026-10-17核查。落实路线图X07/H04的驻留与消费者边界，不把MoE稀疏性直接迁移到dense DiT。固定commit `f86c0682a5dcd44eee1f1c090c588e4f209b62c9`、tree `09ffdde9f8e2c55168c7444b68aa850cf4b9f0ce`；官方归档完整30个Git blob在实验前后全核，13文件选段人工阅读，**不是全仓库全文审计**。main提交日期为2026-03-30，不宣称它是某发布版本或永久最新。[S1]

## 先给结论

- 这是**MLX GPU计算＋CPU路由/缓存管理＋文件专家加载**的案例。所读链没有Core ML/ANE调用；prompt中的“Apple Neural Engine”和Anemll项目名都不是ANE placement证据。[S1][S2][S3]
- 可迁移的是“逐层逻辑稳定的packed bank＋ID作为数据＋仅安装miss”。逻辑数组长期存在不证明物理buffer地址稳定、无copy或并发安全。[S3][S4]
- hit仍可能有`np.asarray(expert_indices)`的GPU→host需求、lookup/index/dispatch成本；C LRU不天然比Python快。8线程I/O完成门也不等于自动与GPU重叠的预取系统。[S2][S3]
- 原C＋原Python wrapper的CPU实验确认：**short-read时普通loader抛错，native slot-bank可返回`[-1]`却不抛错**。这是组件级合同，不是默认模型入口接受短文件或GPU故障复现；模型入口先检查完整layer文件大小。[S2][S3]本机数据

## 1. 驻留、模式与内存分层

下表描述固定ref，不是本机模型内存实测。统一内存下“GPU bank”指MLX计算可消费的数组，不表示独立显存。[S2][S3][S4]

| 模式/组件 | 保留什么 | 热路径仍做什么 |
| --- | --- | --- |
| `--resident` | checkpoint中的完整routed专家 | 原SwitchGLU索引计算；不能与不同K实验直接比速 |
| `--resident-flash` | Python bytes中的全部packed文件 | memoryview切片、K-slot安装及MLX计算，不是完整MLX bank常驻 |
| `--resident-pread-mlx` | 启动时加载的全专家MLX packed bank | expert ID直接索引；预加载逐层`mx.eval`后才复用CPU staging |
| `--resident-rebind` / `--resident-copy-k` | 全packed MLX store＋K-sized executor | 重绑selected K或复制selected K，成本不同 |
| `--slot-bank S` | 每个MoE层S个逻辑稳定专家slot | owner/LRU lookup、读miss并安装、expert IDs转slot IDs |
| `--slot-bank-native` | C中逐层owner/age，跨层共享CPU staging | C选victim/读miss，Python再安装到对应层MLX bank；不是把GPU kernel改成C |

`load_model_bundle`要求slot-bank至少容纳K，拒绝部分resident组合；可与完整packed bank组合，从MLX store补miss。uniform packed geometry限定affine 2/4bit；mixed-sidecar另从layout取各projection的bits/dtype/offset，传给executor。不能把uniform限制扩成“整个项目只支持2/4bit”，也不能据此授予任意混合checkpoint资格。[S2][S3][S5]

## 2. 一次真实消费者路径

普通Python管理路径如下；native模式将owner/victim/I/O交C，仍返回slot和miss_views给Python安装。箭头是依赖，**不表示物理并行**。[S2][S3][S4]

```text
router → top-k expert IDs/scores
       → np.asarray IDs → 查每层owner / 保护本次hit slot
       → 为miss保留victim → pread / resident source
       → 安装gate/up/down的packed weight、scale、bias（9数组）
       → 发布expert→slot映射 → slot IDs作为数据
       → gather_qmm(up / gate) → SwiGLU → gather_qmm(down)
       → score加权求和 → routed + sigmoid(shared_gate)*shared
```

Python `_install_slot_bank_experts`先copy/load，再更新owner、字典、GPU lookup与LRU。executor持有9个packed数组，scale/bias用`mx.view`将uint16位模式解释为BF16，不是每次把全部权重解成FP16矩阵。`compiled()`将bank权重等作为显式输入，不应概括成永远冻结初始专家。[S3][S4]

普通stream/slot-bank只支持`batch=1、seq=1`；生成函数逐prompt token prefill，不等同大batch GEMM。具体gather_qmm kernel由MLX/设备另选，本轮未追踪或执行该GPU消费者。[S3][S4]

full-hit也不表示无host边界：正常路径先将top-k取到NumPy。device-hit-lookup/direct-contiguous-hit是oracle诊断的强前提分支，分别假定所有ID有效或已按当前路由顺序放到`0..K-1`；源码直接增加full-hit计数，不能视为普通在线miss处理。[S3][S6]

## 3. C I/O：拆分资格、完成门与页缓存

loader创建8个worker，generation/条件变量发布一批`pread` task，调用者等8个worker全部完成后才汇总长度、提交成功owner、返回Python。这是批内I/O并发，**没有因此证明GPU计算与下一层I/O重叠**；同loader多调用者并发安全也未验证。[S2]

`cache_io_split`先clamp1..8；expert_size非16384整倍数直接用1，否则受页数限制，以页数商/余数划分chunk。所读C没有`F_NOCACHE`调用；仅包含`fcntl.h`不说明绕开OS页缓存。因此slot miss不等于物理SSD读，pread完成不等于GPU安装或完整token完成。[S2]

交接估算30720/245760字节经真实wrapper与capture合同否决。q4/group64正确布局为三矩阵weight各`D*F/2`字节，三组scale＋bias各`4*D*F/64`字节，总计`3*D*F*9/16`。保留早期未对齐run和失败日志，再增加真正对齐夹具，不抹去修订。[S2]本机数据

| D / F，q4、group64 | 原wrapper expert_size | 16KB整除 | request split=1/4/8对应源码有效split |
| --- | --- | --- | --- |
| 128 / 128 | 27,648 bytes | 否 | 1 / 1 / 1 |
| 256 / 512 | 221,184 bytes | 否 | 1 / 1 / 1 |
| 256 / 1024 | 442,368 bytes，27页 | 是 | 1 / 4 / 8 |

有效split由固定源函数和独立解释合同推导，**没有instrument每个pread task**。原始C在这些配置实际运行并核完整输出，不能把无task trace写成测得任务数，或从无计时的hot fixture推导split4更快。[S2]本机数据

## 4. 本机CPU执行：25用例与全字节检查

M4 Max、macOS26.6.2/build25G83、SDK26.5，既有Python3.11.16/NumPy2.0.2。按文件import官方`expert_io.py`，避免package `__init__`导入model/MLX；原builder/原C不改，仅在own进程把build globals指向本轮ignored输出。otool仅见libSystem依赖；记录import/dlopen拒绝钩子及无MLX/package导入。无GPU/ANE提交，故没有伪造GPU空闲准入记录。本机数据

每几何2个小layer文件，每层8个确定性byte-pattern专家。**不是有效数值权重，不验证量化数学、gather_qmm或生成质量。**18个往返配置`3几何×3split×2layer`各请求`[7,0,3]`；保存54份完整返回缓冲区，12,441,600字节全查。另7个边界用例合计25主用例；开发run不累计为独立性能样本。本机数据

| 用例 | 原始实现观察 | 限定 |
| --- | --- | --- |
| bank3：`[0,1,2] → [0,3] → [2,4] → [3,2,4]` | slots为`[0,1,2] / [0,1] / [2,0] / [1,2,0]`；miss为`[0,1,2] / [3] / [4] / []` | 本次hit slot受保护；不是LFU/LRU全局优劣测试 |
| layer0 `[0]`、layer1 `[1]`、再layer0 `[0]` | owner逐层独立；layer0仍hit，但借用的CPU旧view被layer1覆盖 | CPU共享staging不等于GPU各层bank错用；未跑GPU安装 |
| cold `[5,5]`、再同请求 | cold两slot/两miss；hot均指第一个owner slot | C不去重；正常top-k唯一，prefetch入口另去重，不宣称默认router浪费slot |
| split请求1/4×EOF/半专家，共4个short-read | 普通loader均抛错；native bank不抛错、slot`[-1]`、miss空、旧owner仍hit | 未将`-1`提交GPU，不预判GPU结果或访存安全 |
| 修复own文件后retry，包含于4个short-read | slot0返回真实miss和完整新专家字节 | 不证明异步回滚、取消/drain或并发恢复 |

memoryview/NumPy view是借用，不是快照；需要在staging复用前完成必要copy/读取。测试从不在close后读view。上层逐层安装与MLX评估的生命周期需另验，CPU hash不能授予zero-copy或异步owner资格。[S2][S3]本机数据

## 5. short-read为何穿过wrapper

C先选所有miss；每项只有`total == expert_size`才提交新owner。不完整时slot改`-1`、miss_flag改0，**最终仍返回最初非负miss_count**。Python只在`rc < 0`抛错，按miss_flags提取views，没有再次确认每个slot有效。[S2]

native消费者随后可能因`not miss_expert_ids`把失败计full-hit，并把slot数组转为MLX索引。这里只静态追到入口，未执行该GPU索引。CPU staging可已部分改写，但失败没有交付miss_view，**不能推导GPU旧bank被污染**。[S3]本机数据

同时保留限制：`load_model_bundle → validate_expert_directory`先核每层完整大小；实验直接调用组件注入截断文件，绕过上层预检。它揭示组件/运行期I/O失败合同，不能说默认启动接受短文件；真实运行期short-read、文件变更、EINTR的发生率未测。[S2][S3]

迁移建议而非已实施修复：GPU索引前要求所有slot有效，单列failed/miss/hit；I/O及GPU安装成功后才发布新owner/generation；失败保留可用旧generation或整请求拒绝，不以“无miss_views”推断成功。`require_complete_slots`只是本库教学合同，未接产品或修改上游，也不宣称它解决完整异步事务。合同与测试

C中worker_args的lifetime leak有源码注释；pthread_create失败分支的shutdown/join唤醒值得另查。两项只作静态候选，没有泄漏测量、创建失败注入或死锁复现。[S2]

## 6. 在线temporal、oracle与计时分母

`generate_with_stats`每token `model → mx.eval(logits)`后，temporal分支收集刚才各层expert IDs，再prefetch并eval所有bank。输入在线可得，但调用在当前token之后等待；不能称已有独立I/O/GPU overlap。刚用过的专家通常已驻留，额外lookup/touch/同步是否有益必须实测。[S3][S5]

oracle诊断从trace知道未来路由：all-hit每step在timer外prime/eval，one-step也在timer外预取下一step。因此是**已知未来且搬运成本排除的条件上界**，不是部署后的完整tok/s；parity不补回区外时间，direct/device-hit flags的前提也不能推广到在线miss。[S6]

正常decode timer包含sampling（logits转NumPy）、callback、model/eval与temporal调用；CLI streaming callback的decode/写出也可能在其中。首份logits属于逐token prefill，decode_tps不替代TTFT或含load的整请求。CLI末尾hit统计未在prefill后自动reset，累计所有已调用MoE层；full-hit率分母是**层调用**，不是“所有层同时命中的完整token”。[S3][S5]

`bench_slot_commit`将load/unpack/commit/forward分段fence，可作归因，但fence改变调度形态，GiB/s也不能仅凭名字视作物理SSD带宽。本轮只读未运行此MLX诊断。[S7]

## 7. PR #1：标题、改动、作者结果分开

2026-10-17官方API：[#1](https://github.com/Anemll/anemll-flash-mlx/pull/1)为open、merged=false、merged_at=null；非空merge预览SHA不是合并证据。base为本报告commit，head `3ff58277a8aa33d53b8ea527671a3afcd20b5e6b`。两次patch失败（HTTP/2 CANCEL exit92、重试HTTP500 exit22）；219255字节partial保留，不作完整patch。固定base/head compare API核2commit/7路径，另核head结果表Git blob。[S8][S9]

标题虽称fix native regression，实际只改ignore、CLAUDE、autoresearch收集/训练/验证脚本、npz和TSV，**没有修改`csrc/`或`flash_moe_mlx/` runtime**。新脚本/预测器权重未全文读或执行，PR正文不是已应用修复。[S8][S9]

作者报告M5 Max128GB上native38.23tok/s、81.34%hit，Python48.64tok/s、80.40%hit；建议该模型bank176/split4、更长warmup。120/500token改变缓存历史，不能视为同工作集只换一flag；本机M4未复现速度。QC还注明resident与packed-bank的K不同，不能把101.6/94.5仅归因bank设计。[S8][S10]

head四行TSV保留LFU discard，以及“本token hidden→experts”训练predictor在N→N+1验证中incremental hit仅作者报告3.35%、未达其50%gate、不要集成的负结果。需验证**所需预测时刻**，不能以任意router拟合准确率声称隐藏未来I/O；本轮未重训/评估/接入。[S9]

## 8. 向dense视频消费者迁移：仅列未运行研究

以下由固定实现推导，均`not-run`，不改变默认后端。[S2][S3][S4]

1. 先量真实逐层/逐step访问；dense每次用全权重，没有MoE稀疏命中红利。借bank的有界驻留/双缓冲机制，不继承作者tok/s。
2. 固定weight/LoRA/quant/layout身份、slot generation；记录lookup、I/O、位模式/格式、安装、consumer deadline、整块完成及失败回收。
3. single/double bank和lookahead同时比较完整请求、TTFT、P50/P95、峰值内存、共享带宽与同模型质量，不从oracle区外搬运推导真实overlap。
4. 将来运行MLX/GPU必须先经五次低于5%的空闲准入、缺计数拒绝；本轮CPU免guard不意味着未来GPU免guard。没有执行上游模型、PR parent/head或物理SSD带宽benchmark。[实验规则](../../MAINTENANCE.md)

## 固定源码入口

- [S1：README/许可](https://github.com/Anemll/anemll-flash-mlx/tree/f86c0682a5dcd44eee1f1c090c588e4f209b62c9)
- [S2：C I/O/slot-bank](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/csrc/expert_io.c)；[Python wrapper/geometry/sidecar](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/expert_io.py)
- [S3：model bank/消费者/统计/generation](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/model.py)
- [S4：packed executor/gather_qmm](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/upstream_switch_layers.py)
- [S5：CLI模式与调用](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/scripts/run_qwen35.py)
- [S6：oracle诊断185–300行](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/tools/diagnostics/bench_slot_bank_oracle_hits.py#L185)
- [S7：stage-fenced commit诊断](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/tools/diagnostics/bench_slot_commit.py)；真实文件为[bench_capture.py](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/tools/diagnostics/bench_capture.py)，README/migration中的neo名称不能照抄执行。
- [S8：PR官方API](https://api.github.com/repos/Anemll/anemll-flash-mlx/pulls/1)
- [S9：固定base/head比较](https://github.com/Anemll/anemll-flash-mlx/compare/f86c0682a5dcd44eee1f1c090c588e4f209b62c9...3ff58277a8aa33d53b8ea527671a3afcd20b5e6b)；[固定head结果表](https://github.com/Anemll/anemll-flash-mlx/blob/3ff58277a8aa33d53b8ea527671a3afcd20b5e6b/autoresearch_results.tsv)
- [S10：作者QC报告](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/docs/qc-report.md)

本库分发原创说明、身份/观察数据和原创客户端，不分发上游源文件、npz、checkpoint或dylib。MIT源码、完整日志、54份actual与构建物只留ignored `outputs/gpu-ane-flash-moe-round30-20261017`；[来源](../sources.md)记录逐文件Git/SHA和确切阅读范围。
