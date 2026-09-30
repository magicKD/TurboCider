# Runtime LoRA：按 gate/up 分段取修正与 alpha 类型修复

本轮只改 explicit Qwen runtime 路线的修正表达式，以及共享 LoRA 加载器
的 alpha 数值读取。没有合并 LoRA，没有改 Core ML artifact、staging
base 权重、SiLU、down-LoRA、调度门槛或默认 GPU/冻结图选择。

## gate/up 修正简化

Viggle v0.2.1 r256 的 gate/up 是两个分别训练的 adapter。原来对完整
gate+up 范围调用 `lora_delta_slice`，会把 gate 和 up 各自补零到全宽，
相加后再拆分并打包。现在分别请求 `[0,12288)` 与 `[12288,24576)`，
避免不必要的补零、拼接和全宽相加。

低秩乘法、FP32 累加、最终 dtype 舍入都保留。融合 gate/up adapter 和
重叠叠加 adapter 仍按范围求交；不是绕过非线性或只算 base。回调仍然
request-local，不复用上一 adapter 的捕获状态。

新增 96 组小模型组合（8 种绑定 × 3 种输出 dtype × 2 种 rank dtype ×
2 种行数）检查两种表达式逐元素相等、连续输出和 base 不变。覆盖独立、
融合、叠加、gate-only、up-only、缺省、零强度和负强度；这是合成数学
检查，不代表第二个训练 LoRA 的视觉资格。

## 固定分区消融

M4 Max 64 GB，Qwen BF16 + 原始 Viggle v0.2.1 r256、6 步、seed29，512²
输出、3 张有序 ref512，resident，c288/t1024 v2 图，`chunks=1`，无 profile。
每构建两个独立 trial，每 trial 一冷两热；下表取四个热请求中位数，
墙钟含 VAE/PNG。固定分区用于分离表达式成本，不是推荐的最快配置。

- before：`d2d47119c42b04b83d9aa66f16d78359fd7971afd98fc248a7681b2db13cebb5`。
- 分段修正构建：`4de01d515a4f1ae3f2972bc51acb46d6ffe3675fb393ae238bfb72808eeb2924`。
- 原始结果：`outputs/runtime-ane/qwen-edit3-lora-slices-before-fixed1/` 和
  `outputs/runtime-ane/qwen-edit3-lora-slices-after-fixed1/`，summary 均已 complete。

| 热请求指标中位数 | before | 分段修正 |
| --- | ---: | ---: |
| 整请求墙钟 | 15.935785 s | 16.060705 s |
| gate/up 串行累计 | 0.741310 s | 0.595274 s |
| post-join 累计 | 0.634571 s | 0.702965 s |
| hybrid FFN 累计 | 8.498089 s | 8.425219 s |

before 热墙钟：15.857304500、16.179490250、15.790778084、16.014265542 s。
分段修正热墙钟：16.035985542、16.085423500、16.018394042、16.134720833 s。

gate/up 窗口少约 **19.7%**，完整 FFN 少约 0.86%，但整请求名义慢约
0.78%。局部减少不是整模型加速，其他窗口的波动也不能全部归因于这项
表达式改动。各 trial 最终 576 次预测/混合 block（含冷请求），每个热
请求 192 个 hybrid block，无错误回退。

前后全部 12 张 PNG SHA256 相同：
`1e76ba5f78d7c3d67203b3ef2ba05c8ef008c5eab55f72b237bd7cf279749e1b`。
这说明本输入、固定分区下输出未变，不替代更广质量检查。

两边 MLX peak 都约 22.287 GB，不含 Core ML/驱动/文件缓存。before 两个
trial 的系统 swap-in 各增 4 页（每页 16 KiB）；分段修正 trial 无增量，
swap-out 均无增量。因此不称整轮零换页，也不将系统计数归因于本进程。
每路均做其他推理活动预检，编译/微图检查不与正式计时并行。

## alpha 数值读取：已复现的正确性问题

检查共享 `Weights::apply_loras` 时发现旧代码直接用 `item<float>()` 读取
alpha。该调用读取 tensor 的存储类型，不是数值转换：BF16/FP16/整数
scalar 可能被当作 FP32 位模式读取，导致实际 `strength * alpha / rank`
错误。新增数学回归在旧加载器上真实失败，错误为
`LoRA alpha dtype changed the mathematical scale`。

修复先 `astype(float32)` 再读取 scalar，并拒绝非有限 alpha 或有效
scale。测试使用 `X A^T B^T = 2`、rank=2、alpha=1、strength=-0.75，要求
完整 projection 与单独 correction 都精确等于 -0.75。F32、BF16、F16、
整数 metadata 均通过；NaN/Inf、非 scalar 和有效 scale 溢出均拒绝，
错误后保留原有 fail-closed 清理行为，base 权重不被合并。

Viggle 本文件没有 alpha 字段，原始文件不受此 bug 影响。本修复不能
解释既有 Viggle 的质量或性能差异。修复后的构建 SHA 为
`9b886bdecc1fd95db41a77da81928fc9da5df34999bc805ee2739214102c215e`；
不能把前面的 `4de01d5…` 固定分区测量改标为这个新构建。

该构建已通过 4 项 host、8 项微图/集成、acceleration contracts 和 Qwen
专项。LoRA / Q4-Q8 微图最坏 relative L2 保持 0.00533939 / 0.00658136，
未放宽原阈值。以下自动分区与共图回归使用此构建，不提升为新的最快方案。

## 当前构建：自动分区 ABBA

条件同上，但 `chunks=auto`，顺序 GPU → runtime → runtime → GPU，
每 trial 一冷两热。原始结果为
`outputs/runtime-ane/qwen-edit3-lora-slices-auto-abba/`，summary 已 complete。

| 路线 | 四个热请求墙钟（s） | 中位数 |
| --- | --- | ---: |
| GPU | 12.928880、12.892553、12.881927、12.891356 | 12.891954 s |
| runtime | 14.060345、13.712171、14.362398、14.119073 | 14.089709 s |

GPU/runtime 为 **0.915×**，runtime 慢约 **9.29%**。两次 runtime trial
热请求的 hybrid block/预测计数分别为 37/5 和 47/15；均无错误回退。
四个 trial 的系统 swap-in/out 增量均为零，不等于完整内存峰值已测清。
GPU PNG 与旧基线相同；肉眼查看末次 runtime/GPU 输出，壶、贴纸和桌面
主体接近，没有发现明显新退化，但两路均未满足蓝壶透明材质的编辑要求。
因此只报告本样本的相近画质，不宣称编辑能力全面合格。

## 同一 base 图的 LoRA 切换

`outputs/runtime-ane/qwen-runtime-lora-slices-switch/summary.json` 已完成。
同一进程、同一 v2 图、512²、6 步、固定一 chunk，顺序为预热 base →
base → 原始训练 A → 合成 B → base。B 只将 A 的 227 个低秩 B tensor
减半，临时创建；不是第二个训练适配器，也没有改写 base。

- Core ML 累计调用：192 / 384 / 576 / 768 / 960，图加载计时保持不变。
- 绑定 LoRA projection：0 / 0 / 227 / 227 / 0。
- 两个 adapter 输出不同；返回 base 后与先前热 base PNG 完全相同，SHA256：
  `ea6da56f997ce78c8a0f2169087c1a76aef3acd2ed4ed5ca7022533c3d301c47`。

这验证 adapter 状态不残留、图可复用；六步 base 只用于状态测试，不能作为
未蒸馏 base 的生成质量验收。此检查不与正式性能测试并行。

## 收尾验证

保留完整数学、已有 GPU/冻结图默认路由，以及显式 optional 的 runtime。
没有新增实验开关，没有删除模型、缓存、图片或历史负结果；不触碰原有
249 项暂存移除及其磁盘文件。没有新增 base 或 1024² 性能测试。

当前 `9b886bd…` 库的完整 `make test` 已退出 0，覆盖加速契约、8 项布局、
7 项独立性和共享运行时回归。加上此前同构建的 `make test-runtime-ane`、
`make test-qwen21`，本轮要求的检查均成功。缺少模型夹具、audit 专用库或
未显式 opt-in 的 GPU 测试仍报告 skip，不算作通过的真实推理验收。
Core ML 微图中的临时目录 ResourceWarning 仍存在，未造成测试失败。

README 与状态目录共 192 个本地链接目标存在；源码/工具及便携请求的
路径契约、`git diff --check` 通过。暂存 diff SHA256 保持
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`，
本次未 stage 或 commit。
