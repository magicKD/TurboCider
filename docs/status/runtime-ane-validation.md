# GPU/ANE 实验复现与维护边界

路线选择和已测速度以[当前加速结论](acceleration.md)为准。本文只维护统一
测试入口，不将工具支持等同于真实模型的性能或质量验收。

## 保留什么

- 普通 GPU 和现有冻结 base Core ML 路径：保留默认策略与已验证最快方案。
- `lora_fused`：独立的 base-only 冻结图，接收 runtime LoRA 激活修正；
  融合 base FFN 算子，不融合 adapter 权重。
- `runtime`：可选 runtime-weight 图、token-row 切分、自动 whole-block
  GPU/hybrid 调度、SIMD staging、失败 GPU 重算和调用计数。
- 离线 exporter、单层 probe、整模型对照与共图切换测试：保留为开发工具，
  不引入生成时 Python/Core ML SDK 依赖。
- 历史负结果：保留文档与本地产物，不将 `lora_suffix` 等不完整计算推广为
  正确的 runtime LoRA 路线；不把历史 merged 实验推荐给新请求。

整理保留已验证的原生计算、默认路由、模型权重和 Core ML 编译产物；
未完成整模型验证的候选撤出运行路径，独立有用的正确性回归继续保留。
不删除仍有复现价值的代码来缩短文件列表。已有暂存移除和对应本地文件保持不动。

## 一个整模型测试入口

从仓库根目录运行，模型与 manifest 路径由调用者提供。以下是命令模板，
不是附带这些权重/图的承诺；output 必须是尚不存在的目录。

运行环境使用项目 `.venv/bin/python3`（Python 3.11 或更新）；共用流式
SHA256 校验依赖 `hashlib.file_digest`，不要因系统旧 Python 缺少该接口
而删除或弱化证据完整性检查。

```sh
.venv/bin/python3 tools/validation/runtime_ane_model_screen.py \
  --model models/Comfy-Org-Qwen-Image-2.1 \
  --model-id qwen-image-2.1 --size 512 --steps 40 \
  --runtime-manifest outputs/runtime-ane/qwen21-c288-t1024-exp/manifest.json \
  --routes gpu,runtime,runtime,gpu --chunks auto --warm-repeats 2 \
  --output outputs/runtime-ane/qwen-base-abba
```

每个 trial 是独立 resident CLI batch，一次冷请求后运行指定数量的热请求。
`--routes` 允许重复，用于 ABBA；比较同一路线的全部热样本，不只挑最快值。
`--timeout` 是**每条路线整个 batch**的秒数，不是单张图时限。1024²/40 步
需要相应增加，例如 `--timeout 1200`。

工具的 `--chunks` 默认已统一为 `auto`，与原生 runtime 一致；旧版工具
默认是 `1`，复现旧固定分区实验必须显式传 `--chunks 1`。`0` 是保留 split
边界的消融，不能替代 `--routes gpu`。`--profile` 仅用于诊断，其状态写入
summary，不能与未开启 profile 的速度混报。

新构建的稳定层可异步提交 GPU head。`async_ane_wait_seconds_session_total`
包含与 GPU 重叠的等待，不是暴露开销；老 GPU/join 计数只累计同步采样
子集，不能据其下降宣称 kernel 提速。新计数必须完整、累计不倒退且不能
在同一 session 中忽然消失；旧 receipt 仍兼容。见[异步 head 计时边界](runtime-ane-async-join.md)。

`pre_ffn_seconds_session_total` 也不是纯 attention 时间：`HybridFfn::stage_weights`
开始计时后，`run()` 等待其输入就绪；若前面的 GPU block/残差仍为 lazy，
这次等待可以包含那些上游工作。只有用于 on/off 决策的 **measured block**
才在计时前先求值输入依赖（Qwen `transformer.cpp` 与 Z `z_runtime_block`）；
稳定 `HybridUntimed` 有意不增加该 fence。不能用 pre-FFN/整请求之比推导
attention 占比，也不能把所有 pre-FFN 当作可消除的串行开销。纯 attention
或硬件 overlap 的归因需要单独的诊断/timeline，不与 profile-off 跑分混用。

添加 `--frozen-manifest MANIFEST --routes gpu,runtime,frozen` 可对照冻结图；
manifest 必须与模型、行数、精度和模式匹配。普通冻结 base 图、
`lora_fused` 图和 runtime v1/v2 图不能互换。

## Runtime LoRA 与 1–3 图编辑

LoRA 在上述命令增加 `--lora ADAPTER --lora-strength 1`。Qwen 六步实验使用
Viggle v0.2.1 r256 原始 adapter、`--steps 6 --size 512`；runtime 必须提供
带 `--lora-inputs` 导出的 v2 图，冻结路线必须提供完整 `lora_fused` base-only
图。没有合并/写回 base 权重的步骤。

Qwen 512²、6 步 LoRA 可另外显式传 `--qwen-lora-fp16`，测试已有的
`TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1` 低秩乘法近似。该参数统一作用于
本次 screen 的**所有路线**，包括 GPU 对照；默认关闭，外部遗留环境变量
仍会清理。base/残差 dtype 与 Core ML 图不变，不省略 SiLU 或合并权重。
summary 的 `lora.rank_matmul_dtype` 记录 fp32/fp16，并核对每次 native
结果的精度选择描述。marker 缺失或不匹配会留下 incomplete 与原始结果，
不能仅凭 BF16 `runtime_precision` 推断低秩乘法精度。该参数本身不构成
速度/画质验收，也不会连带开启 Metal norm-RoPE 或其他近似。

Qwen 编辑在相同命令增加以下参数，按 `<image1>`、`<image2>`、`<image3>`
顺序传入参考图；一图或两图就省略后续 `--reference`。

```sh
--reference inputs/subject-a.png \
--reference inputs/subject-b.png \
--reference inputs/subject-c.png \
--reference-size 512 \
--prompt 'Compose the subjects from <image1>, <image2> and <image3> on a wooden table.'
```

- 输出 `--size` 与参考图 `--reference-size` 分开记录；后者默认 1024。
  缩参考图会改变工作负载，不能只给 ANE 缩图后声称同条件加速。
- 工具核对参考图顺序、字节 SHA256、实际 `image.edit` 操作、reference
  resize 和 cold/warm/跨路线的 `reference_tokens` 一致性。
- 前后哈希能检测修改，但不是不可变文件 lease；token 一致也不代表参考
  身份被正确保留，仍须看图验收。
- Qwen LoRA 编辑只暴露原生允许的 512 输出、512/1024 参考尺寸；冻结编辑
  仅暴露 512 输出。需要的 ref512/full-ref 诊断门禁按路线最小化设置。
- 上述编辑工具已有 mock/契约测试，并完成[真实三图 runtime LoRA ABBA
  对照](runtime-ane-qwen-edit.md)；一图/两图和更广质量覆盖仍待验收。
  既有冻结 `lora_fused` 三图结果是另一组历史实验。

## 读结果，避免误报

`summary.json` 写入请求参数、库 SHA、参考凭据、每路冷/热时间与最后一次
累计计数；完整 CLI JSONL、stderr、请求及 PNG 都留在 output 内。
新增 `status=incomplete/complete`：任一路报错时保留已完成的 trial，但不
标记整组完成。旧报告没有这个字段，须核对实际 trial 是否覆盖全部 routes。
`complete` 仅表示请求/遥测校验完成，不表示画质或速度达标。

看速度时使用 `request_wall` 热样本，包含 VAE/PNG、不含冷请求；不是 shell
启动到退出的总时间。比较前检查 steps、seed、prompt、尺寸、参考输入、
LoRA、resident 策略与 build 完全匹配。检查实际 hybrid/prediction 增量，
不能把“已退回全 GPU”的 runtime 标签当作 ANE 收益。

预检遇到疑似其他推理会等待，最多约两分钟，不杀进程。它是启发式检查，
不是设备独占证明。benchmark 期间不并行启动其他推理、微图测试或编译。

验证工具的遥测校验、竞争预检、环境清理与流式文件哈希由
`tools/validation/runtime_ane_common.py` 统一维护；新工具直接引用该模块。
它只依赖标准库，导入不启动进程、不等待、不加载 MLX/Core ML；原 screen
继续导出共享函数供已有本地分析脚本使用。编排与计算行为保持不变。
screen、共图切换、内存采样 wrapper 的文件哈希共用 `sha256_file`；
原 `sha256` 名称保留为别名。前后字节身份检查不等同于不可变 artifact lease，
不因代码去重省略校验，也不将工具内存开销下降记作模型提速。

质量以肉眼看主体、构图、参考身份、编辑完成和伪影；不要求逐像素相等。
共图切换的首尾 base 相同是状态隔离测试，另用
`tools/validation/runtime_lora_shared_graph_switch.py`，不能代替视觉检查。

### 共图切换工具的边界

该工具覆盖 resident 512² 文生图：Qwen 6 步或 Z 8 步，`lora_fused` 或
`runtime`；另支持 **Qwen runtime、六步、1–3 张 ref512 的编辑**。
两份请求必须引用同一完整 base-only manifest。
除 `output`、`loras`、`lora_strategy` 外，请求字段应相同（省略的
`schema_version` 按 1 处理），尤其是非空 prompt 和显式固定 seed。
不匹配、不支持的编辑配置、缺失参考文件或无效 timeout，在导入 MLX/
创建结果前拒绝。冻结图编辑仍不在该工具范围内，其原固定调用数门禁不变。

```sh
.venv/bin/python3 tools/validation/runtime_lora_shared_graph_switch.py \
  --model models/Comfy-Org-z_image_turbo \
  --base-request inputs/base-shared-graph.json \
  --adapter-request inputs/lora-shared-graph.json \
  --output outputs/runtime-ane/shared-graph-check
```

请求文件由调用者准备，包含本地 manifest/adapter 路径。合成 B 只修改临时
adapter，不改 base 权重；runtime 固定 chunks=1 并增加预热以隔离调度变化。
编辑时工具设置所需 ref512 诊断，新版 CLI 允许同 batch 的显式 runtime
base 请求使用该开关，不放宽 GPU/冻结图的 base 条件。参考图按顺序做
前后字节哈希；输出检查编辑操作、reference tokens 一致，并要求每个
runtime 请求都有非零 prediction 增量。短序列大 chunk 全走 GPU 不算
共图修正通过。manifest、运行库与原始 adapter 也做前后身份验证；
这些是字节身份检查，不是不可变 artifact lease。
`--help` 与工具契约只依赖 Python 标准库；真正运行仍需要 MLX 和原生 CLI。
速度对照使用前述 model screen，不用固定分区切换测试推断加速比。
三图 c1792 的真实五请求切换见[实测记录](runtime-ane-qwen-edit-chunks.md)，
1/2 图目前只有新增工具/请求契约覆盖，不外推为真实生成验收。

### 独立内存采样（optional）

整模型 screen 增加 `--sample-memory`，对**每个 trial、所有路线**统一调用
`tools/native/process_tree_sampler.py`；默认不启用，原直接 CLI 行为不变。
例如在前面的 ABBA 命令增加：

```sh
--sample-memory --memory-interval-ms 100 --memory-max-gap-ms 500
```

默认间隔 100 ms、最大间隙 500 ms；间隔须为正、最大间隙不小于间隔。
参数在创建输出之前验证，未开启采样不能传不同的采样时序。
每个 trial 保存 hash-chain JSONL 和独立 verifier 报告，再记录文件 SHA、
采样工具 SHA、correlation ID 与报告内容；采样/验证失败、错误关联或工具
在运行中改变均中止对照，保留原始证据与 incomplete summary。
不能在看到失败后放宽间隙门槛，把旧样本重新解释成合格。

采样窗口包括**加载、一冷多热、退出**；表中 request_wall 仍只取热请求。
不能把进程窗口峰值说成单张热请求峰值，也不能把采样/未采样结果直接混比。
sampler 和其 CLI 使用本工具新建的独立进程组；超时/中断先 TERM 再清理
剩余组成员，避免仅杀 sampler 留下推理。不会按进程名字终止其他任务。

系统 swap 不是模型独占内存，进程 footprint 也不覆盖全部 GPU/Core ML 服务。
`cpuAndNeuralEngine` 不证明物理 ANE 驻留；runtime FP16 staging 不等于
INT8 ANE 计算。

## 离线设备计划检查（不是物理执行证明）

沿用 `tools/validation/qwen21_ane_placement.py`，既支持原有冻结图
`artifacts[block].int8_pc` manifest，也支持 Qwen/Z 的 runtime-weight
`compiled_model` manifest。名称沿用历史接口，不只限 Qwen；只加载现有
编译图的 `MLComputePlan`，不调用 exporter、`compile_model` 或 prediction；
不据此保证 Core ML 内部没有设备计划准备/缓存工作。

```sh
.venv/bin/python3 -B tools/validation/qwen21_ane_placement.py \
  --manifest path/to/runtime-v2/manifest.json \
  --output outputs/runtime-ane/runtime-device-plan.json --quiet

.venv/bin/python3 -B tools/validation/qwen21_ane_placement.py \
  --manifest path/to/frozen/manifest-HASH.json --blocks 0 \
  --output outputs/runtime-ane/frozen-block0-device-plan.json --quiet
```

输出父目录必须已存在、文件必须不存在，并且不能写入被检查的编译图目录。
runtime 只有一个跨层共用图，不接受 `--blocks`；报告放在 `graphs.runtime`，
不假造逐层映射。冻结图继续使用 `blocks`。工具遍历函数与嵌套 block，按
算子报告 preferred/supported device；缺少偏好明确归为 unassigned。
`const`/`constexpr` 单独计数；按算子数量得到的汇总不是耗时、FLOPs或
ANE占比，重点检查 `matmul` 等实际计算节点，而不是常量总数。

工具前后哈希绑定 manifest 和选中的编译树；runtime 另校验全部 `files`
receipt，未覆盖文件、变化、symlink或越界路径均拒绝完成报告。冻结图身份
检查不替代原生 checkpoint/形状验收；前后哈希也不是不可变artifact lease。
报告即使complete，`observed_ane_residency`仍为unknown：不能用计划偏好
宣称真实ANE驻留、INT8硬件算术、GPU/ANE时间重叠或整请求提速。

帮助与host测试不导入Core ML SDK；实际计划检查需项目Core ML环境，且必须
与benchmark、编译和其他推理串行运行。不要为检查计划干扰正式计时。
三个runtime图及两模型冻结block0的当前计划结果见[组合记录](runtime-ane-qwen-qk.md#接续runtime设备计划观测)。

## 维护验证

```sh
make test-acceleration-contract
.venv/bin/python3 -B tests/repository/test_layout.py
make test-qwen21
make test
# 仅在没有其他性能测试时显式运行：
make test-runtime-ane
```

当前报告/预检测试 42 项，包含 1–3 参考顺序、哈希变化、native 门禁、错误
结果保留、跨路线 token 不一致、chunks 输入与 partial/complete 标记；
还覆盖串行/异步 FFN 时间有限性、累计性、总窗口边界与共享模块导入隔离；仓库布局 8 项检查 optional
边界、便携路径和产物不随源码分发。
可用 `--cli PATH` 显式选择隔离构建，必须是有执行权限的文件，旁边必须有
`libturbocider.dylib`；缺文件、目录或无执行权限均在创建输出目录前拒绝。
screen 记录该 CLI 和相邻库的 SHA；默认仍为仓库构建。跨构建 before/after
应交错运行并绑定各自身份，不能把不同构建的样本合成同一条路线的成绩。

screen与共图switch均支持显式 `--qwen-qk-norm-rope`，Qwen 512px；
screen另允许明确无LoRA/无reference的1024px base文生图候选。共用计划/
实际选择校验，screen会在全部所选路线使用相同GPU融合设置，
并记录在summary中；启用但缺任一标记时保留原始结果并拒绝完成报告。
默认清除环境继承的融合开关。runtime组合的性能/质量资格单独见
[Q/K融合实验](runtime-ane-qwen-qk.md)，不因工具支持就宣称更快。

另有 17 项共图切换工具 host 契约覆盖匹配请求、确定性输入、完整图、
未融合 LoRA、1–3 图编辑、参考哈希/缺失文件、真实调用/编辑 token 门禁、
无 MLX 帮助及写产物前拒绝错误，已加入同一契约测试入口。
这些 CPU/契约测试不构成新增模型速度或视觉质量证据。
另有 6 项采样 wrapper 契约（含主线程门禁、真实小进程超时/子进程清理），已加入
`make test-acceleration-contract`；原始 sampler/verifier 的 3 项 host 测试
通过 `make test-process-tree-sampler` 单独运行。
两模型历史 8d90 库的真实三路采样结果见[匹配性能与内存](runtime-ane-matched-memory.md)。
其中三项精度实验契约覆盖不支持请求在写产物前拒绝、所有路线同设置、
默认不继承外部近似、结果标记不匹配及 incomplete 保留；不取代真实看图。

另有10项离线设备计划host测试，覆盖无SDK导入/帮助、冻结接口、runtime
共图、嵌套算子、缺少偏好、命名空间constant、文件变化/receipt、便携路径、
symlink与不覆盖输出。已接入`make test-acceleration-contract`，当前共91项
（4 host + 10 placement + 42 screen + 6 memory + 17 switch + 12 CLI）。
1024²新增契约验证LoRA/编辑在读取artifact前拒绝，以及base三路同样启用/
关闭融合、清除继承环境、保留冻结1024专用门禁；不构成整模型after。
