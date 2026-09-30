# Runtime ANE：长序列起步分区消融

结论：约四分之一 rows 起步，确实让长 prefill 更早使用多个 chunks，
但本轮没有整请求收益，候选不采用。恢复原来一个 chunk 起步、随后按实测
调节的实现；保留行数边界、warmup 排除和分区回调节测试。

## 假设与候选

设计 §26 和 VPIPE `AneFeedForward::create` 从约半数 rows 起步，再按实测
分支速率调整。TurboCider 此前无论 rows 大小都从一个 chunk 起步。
[编辑诊断](runtime-ane-qwen-edit.md) 已观察到：4226-row prefill 的合适
分区约为 3–4 个 c288 chunks，但每个 layer/shape 只在每个请求出现一次，
原调度到第 5 个请求才实际用到该分区。

本轮试约四分之一，而不是直接照搬一半：此前 M4 Max
[组件测量](runtime-ane-component-2026-09-28.md) 中，1024 rows 分给 ANE
512 rows 比 256 rows 慢；完整 LoRA 又增加串行修正成本。这个起始比例
只是待测候选，不代表理论算力比例或所有设备的最优点。

候选首次 chunks：`max(1, floor(floor(rows / chunk) / 4))`，rows 不足一个
完整 ANE chunk 加一行 GPU 时仍完全 GPU。c288 下 1024 rows 起步仍为 1，
4226 rows 起步为 3。其后继续按实测速率选择分区；5% 分区滞回、2%/5%
整块 on/off 门槛、32 visits 周期重试、按 layer/rows 隔离全部保留。
固定 chunks 消融不变，不新增环境变量、图或精度模式。

LoRA 仍是 runtime 激活修正：gate/up 修正在 SiLU 前，down-LoRA 使用
修正后的 hidden。没有合入 base 权重、Core ML artifact 或临时 staging 槽。
普通 GPU、冻结 base 与冻结 `lora_fused` 路由不改。

候选 host 检查覆盖行数边界、初始 share 随 rows 增长、固定分区不变、
warmup 样本排除、根据实测缩回一个 chunk、层/shape 隔离。撤回后保留
边界测试，并改为验证真实保留路线从 1 → 3 → 1 的实测调节，而不是继续
断言已撤回候选的起始比例。

## 配对条件

M4 Max 64 GB；Qwen BF16 + Viggle v0.2.1 r256，6 步，512² 输出，3 张有序
ref512，seed29，resident，c288/t1024 v2 图，chunks=auto，profile 关闭。
每 trial 一冷两热；只统计热请求的 native request_wall，含 VAE/PNG。
参考文件、提示词、adapter、步数均匹配，不缩小输入或省略计算。

- before：`d2d47119c42b04b83d9aa66f16d78359fd7971afd98fc248a7681b2db13cebb5`，
  GPU → runtime → runtime → GPU。
- 候选：`de21f97d5eb2098caad4f89424b043e643aa0547b5c40e8f15faebfa9452b67e`，
  runtime → GPU → GPU → runtime。
- 原始结果：`outputs/runtime-ane/qwen-edit3-initial-share-before/`、
  `outputs/runtime-ane/qwen-edit3-initial-share-after/`。
- 每路预检其他推理，繁忙时等待、不终止他人进程。编译和其他推理测试
  不与正式计时并行；预检不等于全程独占设备。

before 已完成：GPU 四个热样本 12.900890834、12.952355542、12.929507541、
12.930522500 s，中位 12.930015 s；runtime 为 14.249341042、13.928092917、
14.182403416、13.778789291 s，中位 14.055248 s，GPU/runtime 约 0.920×。

候选构建、4 项 host、acceleration contracts、8 项微图/集成和 Qwen 专项
均通过；微图最坏 relative L2：LoRA 0.00533939、Q4/Q8 0.00658136。
阈值未放宽。

## 完成后的配对结果与决策

两个目录的 `summary.json` 均为 `complete`。每构建、每路线四个热请求，
不是把某个 trial 的最低时间作为最终结果。

| 整请求热中位 | before，一 chunk 起步 | 候选，约四分之一 |
| --- | ---: | ---: |
| GPU | 12.930015 s | 12.928609 s |
| runtime | 14.055248 s | 14.112098 s |
| GPU/runtime | 0.920× | 0.916× |

候选 GPU 热样本：12.928323000、12.959368625、12.928895333、12.915839542 s。
候选 runtime 热样本：13.881656208、14.026840167、14.197354834、14.268583500 s。
runtime 总体名义慢约 0.40%，对 GPU 慢约 9.15%；小样本差额不证明显著
性能回退，但足以拒绝把它称作经过验证的提速。

分开看每个 resident trial 中的请求位置：

| 同一 trial 的热请求位置（两个 trial 的中位） | before | 候选 |
| --- | ---: | ---: |
| 第一个热请求 | 14.215872 s | 14.039506 s |
| 第二个热请求 | 13.853441 s | 14.147712 s |

第一个热请求确实将每层 prefill 从 1 chunk 提前到 3 chunks；两 trial 的
runtime hybrid/ANE rows 增量从 42/12096、37/10656，变成 42/30528、47/31968。
但第二个热请求遇到 prefill 的 GPU probe，剩余 decode hybrid blocks 从
10/5 变为 10/15。初始分区、实际采样和后续自动关闭并不是简单的单调关系；
不能把第一热请求的局部收益外推为稳态提速，也不能把全部波动确定归因于
某一段同步或调度逻辑。

所有路线均无错误回退，实际 reference tokens=3072，LoRA 正常绑定。
八个 trial 前后的系统 swap-in/out 增量均为 0。runtime MLX peak 约
22.287 GB、GPU 约 24.972 GB，均不含 Core ML/驱动/文件缓存，不能据此
声称完整内存成本更低。

已查看候选最后 runtime 与 GPU 图片：左右茶壶、中央橙色贴纸、木桌暖光
和构图接近，没有明显新增画质崩坏；两路都未很好保留提示词要求的蓝壶
透明材质。GPU PNG SHA 在所有前后 trial 中一致；runtime 因自动分区
不同，PNG 不完全相同。这是本输入的视觉检查，不是多 LoRA/多提示词资格。

不采用候选，不新加开关，也不修改 GPU/冻结图推荐。后续应优先验证
LoRA 修正与 ANE 之间的串行依赖成本，而不是继续扩大起始 ANE 份额。
这次实验未覆盖一图/两图编辑或新的 1024² 配对，不宣称全目标完成。

## 恢复验证

恢复后原生库 SHA 为
`d2d47119c42b04b83d9aa66f16d78359fd7971afd98fc248a7681b2db13cebb5`，
与本轮 before **完全相同**，不是仍在运行被撤回的候选。
重新通过原生构建、acceleration contracts、4 项 host + 8 项微图/集成、
Qwen 专项、8 项布局和 7 项独立性检查；Core ML 临时目录 ResourceWarning
仍在，测试退出 0，数值阈值未放宽。本轮未重跑完整 `make test`，不把
上一轮全量结果当作这次新覆盖。

保留了边界和 1 → 3 → 1 分区调整回归。所有新增路径均为仓库相对路径，
文档链接目标检查和 `git diff --check` 通过。未删除模型/原始图片或提交
commit，原有 249 项暂存移除及暂存 diff SHA 保持不变。
