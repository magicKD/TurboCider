# Runtime LoRA：GPU 编译边界优化

接续[三图编辑串行成本诊断](runtime-ane-qwen-edit.md)。本轮只调整 GPU
编译边界，不导出新 Core ML 图，不合并 adapter 权重，也不改变调度策略、
精度或 SiLU/LoRA 顺序。普通 GPU、冻结 base、冻结 `lora_fused` 均不改。

## 实现与正确性

- Qwen gate/up 的 `contiguous` 移进 request-local 编译函数；外层仍检查
  shape、所有权及连续布局，不增加或删除必要同步。
- `HybridFfn::Adapter::down_and_add(hidden, base_down)` 返回修正后的 tail。
  Qwen 在同一编译函数内做 down-LoRA 和 base 相加；保留原有 delta 舍入、
  FP32 加法和最终 cast。Z 使用相同数学表达式适配这个内部接口，不增加
  新的编译/精度模式。
- callback 必须返回 base 的 shape/dtype，错误时拒绝该请求；Core ML
  出错时仍整段 GPU 重算，不消费部分成功或陈旧 hidden。
- base 权重 slots、Core ML v2 图、adapter SHA 校验和返回输出的拥有关系
  都没有变化。不同 adapter 的 GPU 闭包仍 request-local，避免捕获上一个
  adapter 的权重/强度。

新增微图检查融合前后数值相等、gate/up 连续布局、正/负强度、返回 base
一致，以及 malformed callback 拒绝后能继续正确运行。

期间发现并修复旧微图测试的三个 BF16 scalar 读取位置：MLX C++
`item<float>()` 是按 float 读取 storage，不自动转换 BF16。旧读取曾产生
接近零的乱码和负的“绝对差”，因此旧通过不能证明这几个绝对误差阈值。
本轮统一先用 FP32 计算这些指标，再读取 float；**没有放宽阈值**。
原有 FP32 relative-L2、布尔相等及真实 PNG/状态隔离证据不受这个读取问题
影响。修复后 8 项 Core ML/MLX 图/集成通过，LoRA worst relative L2 为
0.00533939；host/报告/CLI 契约测试也通过。

## 固定分区消融

M4 Max 64 GB，Qwen BF16 + Viggle v0.2.1 r256，6 步、seed29、三张有序
ref512、512² 输出、resident、固定 chunks=1、无 profile。每个 build 两个
独立 trial，每个 trial 一冷两热。所有热请求各有 192 hybrid block，
不混入 auto 关闭/改分区产生的调度差异。

- before 库 SHA：`58d6f005489ec8dc07811e170485288304b29296c58beed8b4a7d2512c09e562`。
- after 库 SHA：`7e80e38cdb6c8ace8f136b3e7bf4a4528b4dd21ab6f70f7392fcc18a524650c2`。
- 原始请求、参考 SHA、JSONL 和 PNG：
  `outputs/runtime-ane/qwen-edit3-lora-boundary-before-fixed1/`、
  `outputs/runtime-ane/qwen-edit3-lora-boundary-after-fixed1/`。

| 指标（四个热请求的中位） | before | after | 变化 |
| --- | ---: | ---: | ---: |
| 整请求 | 15.949750 s | 16.003950 s | 慢 0.34% |
| gate/up 串行累计 | 0.745217 s | 0.705060 s | 减少 5.4% |
| post-join 累计 | 0.725568 s | 0.650769 s | 减少 10.3% |
| hybrid FFN 窗口累计 | 8.577777 s | 8.473050 s | 减少 1.2% |

before 热样本：15.940550 / 15.958949 / 15.902638 / 15.972472 s。
after 热样本：16.020066 / 15.729361 / 15.987835 / 16.033568 s。
无错误回退；固定分区最后 PNG SHA 一致。

有局部窗口减少，但没有整请求提升；attention/pre-FFN 等其他窗口也在
波动，跨 build 小样本不能把全部差额归为单一变化。不能把局部 10.3% 写成
模型快 10.3%。固定 1 chunk 本身也不是推荐的最快配置，普通请求仍使用
原默认路由，explicit runtime 内仍保留 auto on/off。

## 自动分区与同图回归

同一 `7e80e38…`，相同三参考图、GPU → runtime → runtime → GPU、各一冷两热：
`outputs/runtime-ane/qwen-edit3-lora-boundary-auto-abba/`。

| 路线 | 热样本（秒） | 中位 |
| --- | --- | ---: |
| GPU | 12.891557 / 12.964432 / 12.945169 / 12.938797 | 12.941983 s |
| runtime auto | 13.907589 / 13.647781 / 14.820201 / 14.429316 | 14.168452 s |

GPU/runtime **0.913×**，慢约 **9.5%**，仍不推荐为 Qwen 六步编辑的最快方案。
两个 runtime 会话最后的热请求分别为 0/25 hybrid block，其余走 GPU；
首个会话第三请求完全没有 Core ML 调用增量，不能把它当作 ANE 提速。
三图的实际 reference tokens 均为 3072，LoRA 正常绑定，无错误回退。
已肉眼查看最后 GPU/runtime 图，三个主体、构图和暖光接近；两路都仍未
很好保留蓝壶的透明材质，不把数值/构图接近称作编辑指令全部完成。
preflight 每路执行，但不等于全程独占设备；不把不同会话的 auto 校准差异
归为编译边界的确定因果收益/回退。

Z v2 共图状态回归：`outputs/runtime-ane/z-runtime-lora-boundary-switch/`。
固定 chunks=1，warm base → base → 训练 A → 合成 B → base，Core ML 累计
调用 259/515/771/1027/1283，绑定投影 0/0/238/238/0；首尾 base 相同、A/B
不同，且三种 PNG SHA 都与此前 `bdc7892…` 相同，无需重载图。B 仅用于
状态隔离，不是第二个训练 adapter 的质量证据。
此回归开始前发现 ComfyUI 子进程活动，等待一次后运行，没有终止其他进程。

保留这次编译边界整合和更严格的测试，作为 explicit runtime 内部实现；
**不提升为新的整体最快方案，也不改普通 GPU/冻结图选择**。下一步应关注
自动调度校准稳定性、短 decode 的取舍及不可绕过的身份校验成本，而不是
仅继续强制增加 ANE chunks。全目标尚未完成。

## 验证与保留范围

`TURBOCIDER_NATIVE_ONLY=1 make build`、`make test-runtime-ane`、
`make test-acceleration-contract`、`make test-qwen21`、完整 `make test` 均
成功退出。布局 8 项、独立性 7 项通过；既有缺夹具/专用构建/GPU opt-in
skip 不算已覆盖。本轮未重建 Swift App，公开 C/JSON 契约没有变化。

原有 249 个暂存移除及对应本地文件均保持不动，没有删除模型/产物或提交
commit。没有为了基准取消哈希校验、降低步数/参考尺寸，或把 LoRA 合并到
checkpoint、Core ML artifact、runtime base 权重槽中。
