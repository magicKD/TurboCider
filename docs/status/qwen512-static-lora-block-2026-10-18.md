# Qwen512 静态 Private FFN + 完整 r128 LoRA：有局部信号，尚未接入整图

2026-10-18，Asia/Singapore，M4 Max/64GiB、macOS26.6.2/build25G83。
仅 Qwen Image 2.1 的文生图、单图编辑。未修改 Z Image、已安装 App 或默认配置。
优化目标保持 active；以下是局部实验，不是整请求提速发布。

## 为什么测试静态权重，而不是重新宣称共享内存

现有 Private runtime 已经支持 IOSurface/device I/O，但动态权重图仍需逐层换入
base 权重。[既有真实共享 FFN 案例](../../techniques/gpu_ane/case-studies/qwen-shared-ffn-consumer-cost.md)
有 base/no-LoRA 的局部收益，不能直接推广到六步 student。本轮把同一实际
block 的 base 前1024/2048个配对中间通道固化为 FP16 常量，测试省掉动态权重
换入后，是否还能覆盖真实 LoRA 修正、等待、GPU 补集与完整 down-LoRA 的成本。

不存在 adapter merge、少步数、少 token、跳 FFN 或不完整 suffix-only LoRA。
基准仍是原始普通 compiled GPU block，不借串行 Private 或分段 GPU 作分母。

## 实际计算边界与来源

原 BF16 Comfy checkpoint，原 Viggle v0.2.1 r128，SHA256
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`，
strength1/inference-time、joint BF16 A/B operands/F32 ranks。输出请求512²、
六步、seed73，普通 GPU 路由；观察器只在 step1/layer0重放，之后受控停止。
只完成首个完整 DiT step，没有 VAE decode 或图片导出；并非完整六步耗时。

真实解码 `M1024/H4096/F12288`，文生图及同蓝茶壶 ref512 编辑分别取实际上下文。
观察器借用原9个参数、原输出与原 compiled callable；前后核对输入/输出及
原 gate/up/down 全张量 hash，不允许修改或逃逸借用数据。

LoRA 原227个 terms 在 fused gate/up 后对应195个 resolved projection bindings；
不能把195误判为缺少32个 adapter，亦不能只数一个 marker 就证明数学完整。
GPU 在线算 gate/up 修正，Private 图在 SiLU 前加入；图同时返回 corrected hidden，
与 GPU 其余通道按原顺序拼接，仅对完整 hidden 执行一次 down-LoRA。
`separate/shared` 只改变 gate/up 输入 rank 是否在两支共享，不改变模型或舍入配方。

Private 常量为 base BF16 数值转 FP16，非量化或 BF16 逐位等价。稳定 SiLU 使用
`exp(-abs(g))`，图的单个 packed output 覆盖 down 和 hidden；GPU transfer
负责输入/修正转置与输出验证/恢复，仍有真实交接和转换成本，不宣称零流量。
1024/2048通道 blob 分别约24/48MiB，保留原 checkpoint，不生成全模型副本。

## 强控制与数值门槛

每进程四臂：原 compiled GPU、分段 GPU、串行 Private、并行 Private。
完整 attention/norm producer、FFN、modulation/residual consumer 均计入 wall；
编译/模型加载、poison、独立输出检查在外，setup 单列，不当免费冷启动。
每臂 first/warm3项及 measured12项，共60个完整 block 调用，循环换序；
每进程30次真实 Private 调用。不是多 prompt/seed/层的独立样本。

全部 GPU replay 必须与原 block exact；所有 staged/private first/warm/measured
完整 FFN relative L2≤0.03、cosine≥0.999，完整 block≤0.001/≥0.999。
每个 arm 前在计时外 poison 可复用 Private output，失败或不完整输出不得发布；
运行使用已完成的独立 MLX output，不跨 job 借用下一次会覆盖的 surface。
保留实际 FFN输入、GPU FFN/block reference、最后完整 Private FFN/block 数组，
不是只有 JSON finite 标志。局部门槛不授予最终图片或任意层的质量资格。

## 已观察到的2048通道结果

| 窗口 | 原 GPU中位 / ms | 并行 Private中位 / ms | 名义减少 |
| --- | ---: | ---: | ---: |
| 初版 v2 文生图、独立 ranks | 35.731 | 36.014 | −0.79% |
| 同版 v3 文生图、独立 ranks | 35.748 | 34.210 | 4.30% |
| 同版 v3 文生图、共享 ranks | 35.722 | 34.143 | 4.42% |
| 同版 v3 图生图、共享 ranks | 37.126 | 36.672 | 1.22% |
| 同版 v3 图生图、独立 ranks | 37.094 | 36.018 | 2.90% |
| 同版 v3 文生图、1024通道共享 ranks | 35.721 | 35.607 | 0.32% |
| 同版 v3 图生图、1024通道共享 ranks | 37.100 | 36.984 | 0.31% |

v3 新增共享 ranks 和轻量 host phase统计，两模式用同一 binary/library；
不能将 v2→v3 的正向窗口变化全归因于共享，v3的 independent 本身亦变快。
共享对文生图没有明显增益，对编辑的窗口反而较慢；均保留，不选择性报最佳。
编辑两项都未达到事前3%局部收益门槛。初版负结果也不改写成后续正结果。

其中 v3的2048通道两项文生图及1024通道编辑持续 CPU-load heuristic通过，
仍不证明物理 GPU 独占；其他窗口存在竞争负载，qualification保持false。
没有停止用户进程。全部7个 memory reports complete，系统swap-in/out0，
进程树峰值17,380,209,000–18,058,031,704 bytes；逐窗口原件保留，不把逻辑
blob 字节等同设备峰值，也不将本局部峰值与完整图片请求作减法比较。

1024通道约0.531–0.538秒、2048通道约0.944–0.989秒的单个 prefix setup
仍然存在，计入准备/构图/常量装载，并非完整32层 bank 成本。不得从热 block
数据推断首次整图也快。七窗口共420个完整 block 调用、210次真实 Private
调用；最大 block relative L2约0.00067865，最后完整 FFN约0.002756–0.002833。
候选并非 GPU 逐位等价模型，即使两个 rank-sharing recipe 的候选本身相同。

## 代码与运行状态

研究代码位于 `tools/native/qwen21_static_lora_{mil.hpp,block_probe.mm}`，
入口 `tools/validation/qwen_static_lora_block_screen.py`。没有注册新的产品后端。
原 GPU 观察器只允许 staged20 BaseOnly；新 helper 仅为显式 C++ 观察器加入
resident/pinned r128/six-step GPU generation 或单 ref512 edit，拒绝其他执行器、
内存约束与时间复用；普通请求无观察器时不进入此分支。

第一次尝试被旧 BaseOnly guard 在模型工作前拒绝，原失败目录及旧 binary
保留；没有改统计门槛绕过这次拒绝。新 native library：

```text
build/qwen512-static-lora-observer-20261018/libturbocider.dylib
SHA256 5a5b403055d3f32aaf6cfb684523eeb03eed6eda9eaa8adfd414d8bfa6354edd
build/qwen21-static-lora-block-v3-20261018/qwen21-static-lora-block-probe
SHA256 af600022cc23787331f88f3e80fc7a3203b3881ec3f7e308d699083cce16a509
```

新库 contract131项：130通过/1跳过；真实GPU生命周期1项通过。
诊断 options host1项（含17个新拒绝场景）及 block-screen host4项通过；
emitter6项常量布局/payload检查、12项无设备非法输入检查通过。
不是全仓或整个 Private 故障/取消/长期内存资格。

原始目录统一为 `outputs/qwen512-static-lora-block-*-20261018/`，包括失败、v2、
v3每份源码快照、binary/library hash、native/raw数组、stdout/stderr、load/memory。
旧研究库、旧源码快照、模型/adapter/参考及用户缓存均未删除。

## 普通路线的真实图片回归与独立核对

新库不安装观察器，按原 r128/joint/ref512/seed73、74跑11个完整文生图/编辑
请求，包括 snapshot填充、命中、新seed和关闭恢复。与先前稳定库的 r128
cohort全部11张 RGBA逐像素一致、除output路径外请求字段相同。
热条件生成中位7.484391秒，两项 prefix-hit编辑7.633443/7.645378秒；
这是回归 smoke，不以历史跨窗口差值宣称提速，亦非静态 Private 候选出图。
本批进程树peak31,794,092,832 bytes、memory complete、系统swap-in/out0，
持续竞争负载检查失败，qualification仍false。

`outputs/qwen512-static-lora-audit-20261018.{py,json}` 独立CPU核对全部648项
当前 native source hash、实际 loaded runtime ID、memory原件hash，重新从
保存的真实 BF16数组按FP64累加检查最后FFN/block误差并复算摘要。
v3同任务的两种share/三种策略实际input、GPU FFN/block reference完全一致；
2048通道 independent/shared的MIL、constants及候选FFN/block均逐位一致。
因此没有证据把窗口间速度差归给新的数值模型，也没有将共享 rank 宣称为
显著额外收益。CPU复核未重跑硬件，不授予物理 placement/overlap。

普通回归原件在 `outputs/qwen512-static-observer-normal-r128-smoke-20261018/`。
本轮未下载模型；磁盘约17GiB可用，owned构建/测量/采样均已正常结束。

复现局部实验时使用新目录，不能误把它叫完整图片 benchmark：

```sh
TURBOCIDER_NATIVE_OUT=build/new-static-lora-probe \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen512-static-lora-observer-20261018 \
  bash tools/native/build_qwen21_static_lora_block_probe.sh
.venv/bin/python -B tools/validation/qwen_static_lora_block_screen.py \
  --binary build/new-static-lora-probe/qwen21-static-lora-block-probe \
  --library build/qwen512-static-lora-observer-20261018/libturbocider.dylib \
  --share 2048 --shared-ranks --output outputs/new-static-lora-block
```

图生图额外指定 `--reference`；去掉 `--shared-ranks`保留独立rank控制。

## 决策与尚未完成

保留工具和原件，不接入正常请求，不导出整套32层静态bank。1024通道不盈利；
2048通道文生图有有限局部正信号，但编辑和较早窗口未达门槛，单层setup亦
不可忽略。当前证据不足以支付整套bank的准备、常驻、取消/失败和质量验证成本。
若继续该方向，应先从具体交接/补集算子取得可重复的更大收益，而非仅增加
静态图数量。更多层/seed、故障/长期行为和完整候选生成/编辑耗时仍未验证；
不把4.4%乘到整图或历史收益上，完整优化目标保持active。
