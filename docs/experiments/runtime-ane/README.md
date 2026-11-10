# Runtime ANE 与共图 LoRA 实验归档

本目录集中保存五十篇历史记录：runtime-weight FFN 从单层原型到整模型集成、
完整 GPU block 调度、内存准入、QKV 投影、图几何、低秩精度与整请求匹配对照、量化来源与 base-only 接口，以及 Qwen/Z-Image 的
冻结 base 图、共图 LoRA 数学、异步执行与多 procedure 实验。其他接续记录仍从[状态索引](../../status/README.md)进入。

## 阅读顺序与范围

### 9 月 28 日：组件、集成与共图 LoRA

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [Runtime-weight 单层组件](runtime-ane-component-2026-09-28.md) | 真实 BF16 输入、权重 staging、分区与 FP16 headroom | 集成前组件结果，未包含 attention、encoder、VAE 或完整请求；当时不支持 LoRA，不能替代后续集成验证 |
| [两模型集成与性能筛选](runtime-ane-integration-2026-09-28.md) | Z/Qwen base、token-row 调度、多 chunk、Z GGUF Q8 与系统采样 | 早期 screen、修正后候选、同构建对照分别保留；Q8 不授予 Q4 资格，单次热请求不等于 ABBA，冻结图仍有更快的适用负载 |
| [Runtime graph-v2 LoRA 激活接口](runtime-ane-lora-2026-09-28.md) | gate/up 动态修正、hidden 输出供 GPU down-LoRA、base/A/B/base 切换 | 不合并 adapter；Qwen 整请求负收益、split GPU 消融及 swap/内存边界保留；合成第二文件只检查重绑定 |
| [Qwen unsplit GPU 回退](runtime-ane-unsplit-2026-09-28.md) | 自适应关闭时恢复完整 GPU block，分别缓存 full/split 图 | 显式零 chunk 仍是 split 消融；不把 Qwen 改动外推到 Z，跨构建改善不作同构建性能因果证明 |
| [Qwen 冻结 base 图与完整 runtime LoRA](qwen21-runtime-lora-fused-2026-09-28.md) | W8A8 base FFN、GPU 全部 adapter 投影、文生图与缩小参考图的编辑诊断 | 无稳健整请求优势；单层收益不推广到整图，画面差异和撤回的优化保留；替代 adapter 的六步 schedule 未获质量资格 |
| [Z-Image 冻结 base 图与完整 runtime LoRA](z-image-runtime-lora-fused-2026-09-28.md) | INT8 per-channel 权重、FP16 激活、共图切换与可选 direct-FP16 | 不是 W8A8；direct-FP16 有数值变化，单提示词筛选不保证跨输入收益，第二提示词冷图检查不是速度复验 |

### 9 月 29 日：调度、内存与同构建对照

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [完整 GPU block 探测](runtime-ane-full-block-probe-2026-09-29.md) | Qwen full/split/probe 边界、base 三路与 LoRA 文件校验成本 | LoRA 后续请求已全回 GPU，不能当持续 ANE 收益；独立 SHA 计时不是同一请求内的逐段 trace；Z 尚非完整 block 探测 |
| [稳定混合阶段减少计时同步](runtime-ane-periodic-sampling-2026-09-29.md) | Qwen untimed 生命周期、base 三路、共图切换与 LoRA 复测 | 固定 chunk 状态切换不作 auto 性能成绩；LoRA 负收益仍保留，优化没有作用于已关闭的混合路径；Z 沿用 stage-only 接口 |
| [请求级内存准入](runtime-ane-memory-admission-2026-09-29.md) | free/inactive 启发式、host scratch 新旧共存、拒绝回退、512²/1024² 采样与正反向对照 | free-only 失败和候选库单独留档；机会性保护不等于低内存或 driver/wired 硬预算资格；冻结窗口压缩及单提示词边界保留 |
| [两模型肖像与 Z c416 筛选](runtime-ane-portrait-and-z-chunk-2026-09-29.md) | Z 真实权重/合成输入组件对照，以及同库双模型 512² 肖像三路 | c416 组件更慢，未进行该候选整模型测试；512² 冻结图仍快；第二提示词不构成广泛画质资格 |
| [Qwen c1664 组件与整请求](runtime-ane-qwen-c1664-2026-09-29.md) | 真实权重/合成输入单层筛选、同库 1024² chunk ABBA | 组件正信号没有转为整请求收益，保留 c1792；GPU/冻结图未参加该次 chunk 对照，不借用其他实验的分母 |
| [Qwen 1024² 第二提示词](runtime-ane-qwen-1024-portrait-2026-09-29.md) | 同库肖像 GPU/runtime/冻结图正反向三路、独立内存流与代表图检查 | 重新测量该尺寸的三个分母；不拼接狐狸、512² 或 chunk 消融样本；冷请求后的系统压缩仍可能影响热态比较 |

### 转换与调度：撤回候选及保留回归

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [整数输出恢复转换](runtime-ane-output-convert.md) | FP16→BF16 整数转换微基准、完整位模式与 headroom 回归 | 只完成微基准和原版整模型，没有候选 after；撤回原因是端到端收益未验，不能写成已证明无收益 |
| [直接写入独占输出](runtime-ane-output-copy.md) | 固定分区 Qwen 三图 before/after、独立所有权和失败回收 | post-join 减少未转化为整请求收益，分配成本移动且窗口重叠；不是 Core ML 零复制，固定输入一致不等于广泛质量资格 |
| [大输出恢复并行](runtime-ane-output-restore-parallel.md) | Qwen/Z 匹配对照、真实共图切换、独立内存流和撤回后验证 | 局部 3.69× 未带来整请求收益；候选、原版和撤回后库分别记录，未在撤回后库复测整模型性能；合成第二 adapter 只检查状态隔离 |
| [Staging 转换任务粒度](runtime-ane-staging-partition.md) | 两模型真实权重/捕获输入的单层 ABBA，16 行任务与四组任务 | staging 全暴露的组件不含 attention，不能外推整模型退化；仅构建独立 probe，产品库未替换，短暂竞争进程边界保留 |
| [长序列起步分区](runtime-ane-initial-share.md) | 一个 chunk 与约四分之一 rows 起步、Qwen 三图配对和恢复 | 第一热请求更早使用多 chunk 不构成稳态收益；候选撤回，保留实测 1→3→1 调节回归；未覆盖一/两图或新 1024² 对照 |
| [完整 block 收益门槛](runtime-ane-block-margin.md) | 2%/5% 与 5%/8% 消融、两模型候选与恢复后检查 | 完整 block 门槛不同于 FFN 门槛和分区滞回；恢复版 Qwen base 不证明候选影响，GPU 采样发生少量系统 swap-in，不称整轮零换页 |

### QKV：从独立投影到完整 block 和整请求

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [Z fused QKV 组件筛选](runtime-ane-qkv-component.md) | 真实 QKV 权重、1024-row 合成输入、四种分区的 80 对测量 | 最佳分区含 staging 仍更慢；没有产品 block/norm/RoPE 或请求激活，未接产品路线 |
| [Qwen 长序列 Q 投影](runtime-ane-qwen-q-projection.md) | 单独 Q 权重、4096-row 合成输入和邻近 chunk 筛选 | 不是完整 QKV，不能把独立 Q 的时间与后续打包实验相加；竞争进程影响边界保留，1.336× 只作候选信号 |
| [三权重打包 QKV](runtime-ane-qwen-qkv-packed.md) | 计时前打包 Q/K/V、一次 GEMM 与三次 GEMM 两种 GPU 对照 | 打包分配未计时，32 层副本理论约 3 GiB；组件 1.4× 不授予产品 block、联合内存或整请求资格 |
| [Q/K/V 三源直填](runtime-ane-qwen-qkv-parts.md) | 三个 checkpoint 张量直接转换到单个 runtime 槽、失败恢复和组件对照 | 免永久 GPU 打包副本不等于请求内存合格；graph 估算不是 driver/wired 成本，仍未接产品 QKV |
| [Qwen 1024² 产品初筛](runtime-ane-qwen-qkv-product.md) | 显式 resident base QKV 路由、40 步同库四路和进程树采样 | 每路仅一热样本、未反向；1.013× 不构成稳定收益，5 步旧构建不能混入，FFN ANE 不与 QKV 并存 |
| [完整 block 自动启停](runtime-ane-qwen-qkv-auto.md) | 单 chunk on/off、5 步回退、修复后 40 步正反向与固定双 chunk | auto 不搜索分区，默认仍固定一 chunk；失败的旧库尝试不混池，40 步无明确收益，双 chunk 慢 12.96% 且未反向复测 |

### 共图 LoRA：数学、生命周期与计时边界

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [GPU 编译边界](runtime-ane-lora-boundary.md) | request-local gate/up 与 down 回调、失败恢复、固定分区和 auto 对照 | 局部窗口减少没有整请求收益；修复旧 BF16 scalar 测试读取且未放宽阈值，旧绝对误差通过不能继续当有效证据 |
| [Gate/up 分段与 alpha 类型](runtime-ane-lora-slices.md) | 分段修正的数学检查、alpha 转 FP32 后读取、非有限 scale 拒绝 | alpha 正确性修复与分段性能构建分开；原 Viggle 无 alpha，不能用该 bug 解释其既有差异；合成 B 仅用于隔离 |
| [V2 base hidden 处理](runtime-ane-v2-base.md) | 无 adapter 的 hidden 只检查、等价位模式判定、两版筛选与共图恢复 | 第一版只省副本无明确提速，最终版不是跨构建 ABBA；仍扫描真实 hidden，不把复制量等同内存流量，保留系统 swap-in |
| [V2 同构建三路对照](runtime-ane-v2-matched.md) | 两模型 GPU/runtime-v2/冻结图双向顺序和四热样本 | base 收益不授予 LoRA 性能，冻结图仍快；这是匹配对照补全，不把已保留优化重复算成新增加速 |
| [合并输入就绪屏障](runtime-ane-lora-ready.md) | 输入与 gate/up 修正同次 eval、固定分区、auto ABBA、base/A/B/base | 新 readiness 窗口含上游工作，旧 FFN 窗口不可直接相减；Qwen auto 仍慢，固定分区改善不外推到其他路线 |
| [稳定层异步 GPU head](runtime-ane-async-join.md) | untimed/profile-off 重叠、异常排空、完整输出 fence 和两模型回归 | async wait 可与 GPU 重叠，post-join 可含剩余 head；固定 chunk 共图不直接覆盖异步模式，旧构建基线不可混成新加速比 |

### 图几何、低秩精度与采样策略

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [图内 tile 与 token chunk](runtime-ane-tiles.md) | K/N tile、chunk 联合组件筛选、v1 base 与 v2 LoRA 整请求、后续 N256 负结果 | prediction 提速不等于并行 FFN 或整请求收益；v1/v2 分开，N256 只在组件失败，未做其整请求对照 |
| [Qwen 长序列 chunk](runtime-ane-qwen-chunks.md) | 同总分区与重新平衡的合成输入组件、c1792/c320 整请求交错对照 | 7.24% 请求省时不只来自 dispatch 减少；该组无 GPU/冻结图分母，不能借旧数据宣称胜出；系统 swap-in 保留 |
| [Qwen 三参考图大 chunk](runtime-ane-qwen-edit-chunks.md) | 短 batch 四配置、ref512 base 门禁修复、真实共图切换和长 resident ABBA | 旧库 0.977× 与新库预声明全热/较晚窗口分开；不挑实际 ANE 样本删除 GPU probe，合成 B 和六步 base 仅检查状态 |
| [LoRA FP16 低秩](runtime-ane-qwen-lora-rank.md) | 同库 GPU/auto/fixed1 交错、精度标记修复和独立内存采样 | 只改低秩 matmul，累加与 base dtype 保留；固定分区改善不能外推到 auto，单组相近图片不构成广泛质量资格 |
| [GPU-first 初始采样](runtime-ane-gpu-first.md) | LoRA 初始采样顺序、两模型候选/原版和恢复验证 | Qwen 成本后移、Z 小幅变化不足推广，恢复 hybrid-first；顺序筛选不是跨构建交错因果实验，恢复库与原版逐字节一致 |
| [零修正输入复用](runtime-ane-zero-input.md) | 微图状态隔离与原版整模型 before，撤回缓存状态和专用计数 | 未完成候选 after，收益未知而非已证无效；仍逐次清零，保留独立于缓存实现的失败恢复回归 |

### 同构建整请求对照与 Q/K 组合

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [Qwen 1024² v1](runtime-ane-qwen-1024.md) | bdc789 构建、c288 多 chunk 与 GPU/冻结图初筛 | 每路线单热样本，v1 无完整 LoRA 输入；不能把旧构建成绩外推到 v2 或稳定吞吐，采样不覆盖完整 driver/wired |
| [Qwen 1024² v2](runtime-ane-qwen-1024-v2.md) | e55 构建 c320，三路线正反向、独立进程树内存采样 | 四热样本全部纳入，方向间波动与系统 swap-in/compression 保留；staging 总工作不等于暴露等待，不能据此承诺双缓冲收益 |
| [8d90 匹配性能与内存](runtime-ane-matched-memory.md) | 两模型 512² base 同库 GPU/runtime/冻结图，12 份独立采样 | 无原生优化 before/after，不用旧库作当前分母；1.3125 MiB 系统 swap-in 未抹除，进程峰值不可相减归因为纯 Core ML 开销 |
| [Qwen 512² Q/K 组合](runtime-ane-qwen-qk.md) | 三图六步四配置、两模型 base 及离线设备计划 | 编辑的全热与较晚窗口分开，同样优化 GPU 后未扩大混合优势；base 单独验收，设备偏好和节点数不是物理驻留或性能证明 |
| [Qwen 1024² Q/K 组合](runtime-ane-qwen-qk-1024.md) | 55e5 构建、c1792 v2、同库六配置正反向 | base 文生图显式可选，不能外推 LoRA/编辑；不拼接旧库分母，不删探测或慢请求，保留 swap-in/compression 与单提示词质量边界 |
| [Z 完整 block 调度](runtime-ane-z-block.md) | Q4/BF16、第二提示词、固定分区共图状态和后续 875b 三路对照 | 旧 f20、bdc789 与 875b 分开；chunks=0 是 split GPU 消融，固定分区状态不等于 auto 性能；冻结 BF16 仍最快，Q4 收益不外推其他精度 |

### 量化来源、base-only 接口与编辑边界

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [Z Q4 与 padding](runtime-ane-q4.md) | 正确 Q4 来源、padding shape 修复、同库 Q4/BF16/冻结图初筛 | 错误全局量化文件不是性能分母；磁盘 BF16、加载 dtype、ANE FP16 staging 分开，Q4 收益仅相对自身 GPU |
| [Qwen 三图 LoRA](runtime-ane-qwen-edit.md) | bdc789 ABBA 负结果，58d6 新增串行时序诊断与 Z 回归 | profile 不混入正式对照，prefill 小幅收益不等于六步整请求收益；独立中位数不可相加，蓝壶透明材质未完成，无新加速结论 |
| [v1 base-only 筛选](runtime-ane-v1-base-2026-09-30.md) | Qwen v1/v2 双提示词、同库三路对照、Z 短序列组件 | 带 adapter 仍需 v2；Z 合法零等待导致的失败初次记录保留，修复驱动后另目录重跑，组件与长序列 GPU 对照分开 |
| [Z 1024² BF16 v1](runtime-ane-z-1024-v1-2026-09-30.md) | c352 相对 GPU 双提示词、c704 同分区组件与整请求 | 没有匹配冻结图分母，也不是 v1/v2 请求比较；c704 少调用只换得 0.38% 名义省时，继续保留 c352 |
| [Z 1024² Q8 v1](runtime-ane-z-gguf-q8-1024-2026-09-30.md) | 原生 Q8 GGUF 同库 GPU/auto 双提示词和固定分区探索 | 不与 BF16 跨 checkpoint 算倍速；肖像 runtime 正反向 PNG 不同、compression 均保留，固定分区批内相同不证明 auto 确定性 |
| [动态 INT8 / Hadamard](runtime-ane-w8a8-feasibility-2026-10-01.md) | M4 Pro 小型双动态操作数 Q/DQ、尺度恢复与 CPU 蝶形参考 | 数学恒等式不等于量化准确性；小图计划偏好 CPU，计时排除旋转，无物理 INT8 或生产提速证明；研究 manifest 不可进入 App runtime |

### 图生命周期与多 procedure 能力

| 记录 | 记录范围 | 保留的限制 |
| --- | --- | --- |
| [Compiled graph 私有快照](runtime-ane-artifact-lease.md) | 验证字节与延迟读取的所有权、旧/新库 ABBA、新库 512²/1024² 三路对照 | 小幅热态差异不能证明冷加载无成本；旧库分母不重标，自动分块差异与冻结窗口 compression 保留；私有目录清理不等于系统缓存清零 |
| [多固定行数 procedure](runtime-ane-multi-procedure-2026-10-19.md) | FP16 A/B/A、真实 symbol 映射与共享 pitch，W8 family V7 合成完整 FFN | V1 finite 审计失败不作最终证据；本页 V7 阶段未接入 Session，[后续 V9](../../status/qwen-base512-function-session-2026-10-19.md) 已完成显式接入与 31 张完整图片审计，但性能 gate 未过；C++ helper 须匹配对应库，不以小图证明吞吐或 INT8 MAC |

原文中的默认、开关、代码状态、成功测试和未完成项均属于记录时点。
本次归档保留所有原数字、命令、失败与撤回结论，没有重新执行模型实验。

## 区分执行路线

- 冻结 base 图把 checkpoint 权重编入图中。`lora_fused` 融合的是 base
  FFN 算子，adapter 仍在 GPU 动态计算；gate/up 修正在 SiLU 前输入，
  down-LoRA 使用图返回的 hidden。base artifact 不绑定 adapter 身份。
- Runtime-weight 图把 base 权重作为运行时输入；原始 v1 不处理 LoRA，
  后续 v2 才增加动态修正与 hidden 输出。图复用要求匹配模型与几何，
  不表示 Z/Qwen 可以共用同一个 H/F 不同的产物。
- `lora_suffix` 是不完整的诊断路径，不能当作完整 runtime LoRA 基线；
  adapter 绑定的 `lora_merged` 也不满足运行时切换而不合并权重的目标。

Core ML 的 CPU/Neural Engine 偏好与 host API 耗时不证明物理 ANE 驻留。
单层、denoise、完整请求，冷启动、预热、跨请求切换，以及不同构建、
模型、精度、尺寸、提示词、种子、rank 和 schedule 的结果分别解释。
MLX 峰值不包含全部 Core ML/OS 开销；一个窗口无新 swap 不等于完整内存资格。
base 前后 PNG 一致与合成第二 adapter 文件检查支持该次状态隔离，
不能替代第二个真实训练 adapter 的画质或 schedule 验证。

## 证据与现行入口

完整正文只维护在本目录，五十个旧 `docs/status/` 页面保留原章节跳转。
正文中的 `outputs/`、`results/`、`build/`、模型及回执路径沿用原记录；
本地忽略产物可能不随 checkout 提供，本次不复制、不重建，也不改写 JSON。
命令仍从仓库根目录运行，复验须先核对原记录的版本、产物和工作负载。

现行选择见 [GPU/ANE 加速维护入口](../../status/acceleration.md)，操作方式见
[GPU/Core ML 与 runtime LoRA 指南](../../status/runtime-lora-acceleration-2026-09-28.md)，
冻结 Qwen W8A8 的其他实验见[独立专题](../qwen21-w8a8/README.md)。
新测量遵循[统一归档规则](../README.md)，保留新旧证据各自的日期和结论。
