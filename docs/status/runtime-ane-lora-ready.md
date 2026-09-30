# Runtime LoRA：合并输入与 gate/up 修正的就绪屏障

接续 [v2 base 匹配对照](runtime-ane-v2-matched.md)。本轮针对真实 adapter
路径，不合并 LoRA，不修改 Core ML artifact、base 权重槽、SiLU、精度、
headroom 恢复或调度门槛。无 adapter 的 base 路径仍保留原来的输入屏障。

## 实现

原路径先 `eval(packed input)`，再构造并 `eval(gate/up corrections)`，
随后才把这三份输入交给 ANE。候选一次构造输入与两个修正张量，在同一个
`eval` 中等待它们；完整修正仍在 ANE SiLU **之前**，down-LoRA 仍消费
修正并恢复 headroom 后的 hidden。GPU head 与 ANE tail 的并行边界不变。

在取得任何 host 指针前完成就绪屏障，权重 staging 的借用与排空规则不变。
无可用 ANE 或零 chunk 时不执行这段 LoRA 提前准备，继续完整 GPU 路径。
stage/predict 失败仍执行原有 GPU 重算；未删除文件身份校验。

主要代码为 `native/backends/ane_ffn.cpp`。新增回归每次构造新的 lazy
输入，覆盖 base/base/A/负强度/base/base、错误 gate/up 形状、异常后排空
与恢复；保留输出/hidden 生命周期、溢出恢复和失效重算检查。

## 计时语义，不能把搬移窗口当收益

组合后的就绪窗口可能包含上游 attention，因此单独导出：

- `lora_input_ready_seconds_session_total`，profile 中对应
  `lora_input_ready_seconds`：输入与修正一起就绪的 host 窗口，属于
  `pre_ffn_seconds_session_total`，不是纯 LoRA kernel 时间。
- 旧 `lora_gate_up_seconds_session_total` 为兼容历史报告保留，在此候选
  路径为零，因为不再存在输入就绪后的第二段独立修正屏障。
- `hybrid_ffn_seconds_session_total` 从组合就绪后开始，不能直接与旧版
  该字段相减后宣称提速；`pre + FFN` 或完整请求才覆盖相同范围。

报告校验接受旧 receipt；新字段必须有限、非负、累计不倒退且不超过 pre。
新增 host 契约验证该窗口可以大于 FFN、但不可大于 pre。

## 固定一 chunk 的前后对照

- before 库：`b55419e636ada12e54dc197b7099811225b361d63d2c1d16e6e0560449104dba`。
- 候选库：`4d864b4439adb23cd53aa90dfe2884a3ced46027a4d24b4ac3e5ba2126ee6eac`。

M4 Max 64 GB、512²、resident、`chunks=1`、profile 关闭；每构建两个独立
trial，每 trial 一冷两热，统计四个热请求中位数，包含 VAE/PNG。串行
before/after，不是跨构建 ABBA。每 trial 检查竞争推理，构建/测试不与计时
并行。固定分区隔离调度变化，并非推荐的最快配置。

Qwen：Viggle v0.2.1 r256 原始 adapter、6 步、seed29，三张有序 ref512，
输出 512²，c288/K=N=1024 v2；原始 teapot/dragon 编辑请求不变。
Z：原始 distill patch adapter、strength1、8 步、狐狸 prompt/seed42，
c352/K1024/N512 v2。两边均为 `inference_time`，base-only 共图。

| 工作负载 | before 热样本（s） | 候选热样本（s） | 中位数变化 |
| --- | --- | --- | --- |
| Qwen 三图编辑 | 16.031798 / 16.108174 / 16.031525 / 16.275955 | 15.469974 / 15.548438 / 15.546266 / 15.501642 | 16.069986 → 15.523954，少约 3.4% |
| Z runtime LoRA | 7.867842 / 7.873671 / 7.892285 / 7.895828 | 7.885714 / 7.798388 / 7.836572 / 7.836656 | 7.882978 → 7.836614，少约 0.6% |

Qwen 本轮显示整请求改善；Z 的小幅变化不构成明确提速结论。不能把这组
固定分区收益当作已超过普通 GPU，更不能把 readiness 字段接近 attention
的数值理解为 LoRA 变慢了几十倍。

每个 Qwen trial 576 hybrid/predictions，每个 Z trial 768 hybrid、771
predictions（均含冷请求），无错误回退；八个 trial 的系统 swap-in/out
页数增量均为零。系统快照不等于完整进程/驱动内存峰值或物理 ANE placement。
两边 MLX allocator peak 分别约 Qwen 22.287 GB、Z 16.188 GB，前后基本
相同；它不计 Core ML、驱动或文件缓存，不能据此宣称完整内存峰值不变。

固定分区全部末次热 PNG 在前后分别相同：

- Qwen：`1e76ba5f78d7c3d67203b3ef2ba05c8ef008c5eab55f72b237bd7cf279749e1b`。
- Z：`f37cab791a458dfa83e70e449dc6b5ce7f40d5f0b728b1faea0df75c7a443c49`。

原始证据：`outputs/runtime-ane/{qwen-edit3,z}-lora-ready-{before,after}-fixed1/`，
四份 summary 均 complete，保留请求、JSONL、PNG、参考 SHA 和预检信息。

## 初步验证

候选构建、4 项 host、8 项微图/集成、51 项 acceleration contracts、Qwen
专项均通过。LoRA / Q4-Q8 最坏 relative L2 仍为 0.00533939 / 0.00658136，
未放宽阈值；原有临时目录 ResourceWarning 未使测试失败。
真实 auto GPU/runtime 对照及两模型共图切换在固定分区之后单独验证，
不能用这些合成数学测试取代真实图像或性能资格。

## 同构建自动分区 ABBA

仍为 `4d864b4…`，上述工作负载不变，改为 `chunks=auto`；每模型按
GPU → runtime → runtime → GPU，一冷两热，各路线四个热样本，profile
关闭。证据：`outputs/runtime-ane/{qwen-edit3,z}-lora-ready-auto-abba/`，
两份 summary 均 complete。

| 工作负载 / 路线 | 四个热请求（s） | 中位数 |
| --- | --- | ---: |
| Qwen 三图 / GPU | 12.929302 / 12.937660 / 12.898352 / 12.911562 | 12.920432 s |
| Qwen 三图 / runtime | 13.954281 / 13.741244 / 14.545995 / 14.540309 | 14.247295 s |
| Z LoRA / GPU | 8.627184 / 8.587724 / 8.572193 / 8.564511 | 8.579958 s |
| Z LoRA / runtime | 7.872398 / 7.796782 / 7.839379 / 7.801843 | 7.820611 s |

Qwen GPU/runtime **0.907×**，runtime 仍慢约 **10.3%**；保持编辑优先
GPU。两个 runtime trial 的热请求 hybrid 增量为 38/5、62/30，其余走
GPU，未发生错误回退。不把固定分区 3.4% 改善推导为 auto 也同等改善；
本轮没有旧构建 auto 的交错消融，不能从不同轮次数字宣称 auto 因果收益。

Z GPU/runtime **1.097×**，少约 **8.9%** 耗时；两个 trial 热请求 hybrid
增量为 248/248、256/256，无错误回退。收益只对应当前 distill patch、
8 步和狐狸样本，不等于本次改动贡献了全部 9.7% 加速比，也不代表超过
历史冻结 `lora_fused` 路线。本轮未做匹配冻结 LoRA 三路对照。

八个 trial 系统 swap-in/out 页数增量均为零。已肉眼查看两模型最后
GPU/runtime 输出：Qwen 两把壶、贴纸、桌面及暖光接近，但两路都未还原
蓝壶透明材质；Z 狐狸主体、姿态、雪景接近，局部毛发/枝叶不同，未见
明显新增崩坏。不将相近质量称为所有编辑指令完成。

## 同一 base 图切换回归

`outputs/runtime-ane/{qwen,z}-lora-ready-switch/summary.json`，同一候选
库、同一 v2 图，固定一 chunk；预热 base → base → 训练 A → 合成 B → base。
Qwen 图为 c320/K1024/N512，Z 为 c352/K1024/N512，与上一轮相同。

| 模型 | 累计 Core ML 调用 | 绑定投影 |
| --- | --- | --- |
| Qwen | 192 / 384 / 576 / 768 / 960 | 0 / 0 / 227 / 227 / 0 |
| Z | 259 / 515 / 771 / 1027 / 1283 | 0 / 0 / 238 / 238 / 0 |

图加载计时不变，返回 base 与先前热 base 相同；base/A/B 的 PNG SHA
分别与 `b55419e…` 对应状态测试一致。B 只将训练 A 的低秩 B tensor 减半，
不代表第二个训练适配器质量。Qwen 六步 base 只作状态检查，不作 base
质量验收。没有改写 base checkpoint、Core ML 图或 runtime base slots。

## 保留范围

保留合并 readiness 屏障及回归测试，作为 explicit runtime 的内部实现，
不增加开关。Qwen 固定分区有本轮整请求改善，Z 未见明显回退；不将它
宣传为所有工作负载提速。普通 GPU、冻结图和 runtime 的显式选择不变，
Qwen 六步/编辑仍优先 GPU。更广提示词、适配器、1024²、完整内存/overlap
及真实 ANE placement 等资格仍未完成。

## 正常 base 回归与收尾

全量测试结束后，使用保留 `4d864b4…` 库复测正常步数的 base，默认狐狸
prompt/seed42、512²、resident、auto、无 profile；每模型一冷两热：

| 模型 / v2 图 | 热请求（s） | 中位数 |
| --- | --- | ---: |
| Qwen，40 步，c320/K1024/N512 | 36.892960 / 37.088459 | 36.990710 s |
| Z，8 步，c352/K1024/N512 | 6.487921 / 6.433754 | 6.460837 s |

证据：`outputs/runtime-ane/{qwen,z}-base-lora-ready-regression/summary.json`，
均 complete。Qwen 3648 predictions/hybrid、192 GPU blocks；Z 688
predictions、685 hybrid、83 GPU blocks。无错误回退，系统 swap-in/out
页数无增量，新增 LoRA readiness 计数在两模型 base 上均为零。
末次热 PNG 与上一库分别相同，SHA 为 Qwen `194aac19…`、Z `8fb48cca…`。
此次耗时接近此前 base 回归；没有在新库重测 GPU/冻结图，不跨构建计算
新的 base 加速比，也不把这两个样本称为全输入无回退保证。

当前保留库 SHA：
`4d864b4439adb23cd53aa90dfe2884a3ced46027a4d24b4ac3e5ba2126ee6eac`。
原生构建、`make test-runtime-ane`、`make test-acceleration-contract`、
`make test-qwen21`、完整 `make test` 均成功退出；全量包含布局 8 项、
独立性 7 项。缺夹具/专用构建/GPU opt-in 的 skip 不算真实资格覆盖。
所有性能/状态测试均在构建和测试结束后串行运行，没有重建 Swift App。
没有 stage/commit 或删除模型、缓存、原始图片；249 项既有暂存移除及
暂存 diff SHA 与本轮前相同。
