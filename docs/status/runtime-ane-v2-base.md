# Runtime v2：同一 LoRA 图上的 base 开销与 hidden 处理

接续已补齐同一保留构建上的 [GPU/v2/冻结图双向对照](runtime-ane-v2-matched.md)。
本页保留优化当时的 before/after；下文“尚未重测 GPU/冻结图”指当时阶段。

接续 [tile/chunk 筛选](runtime-ane-tiles.md)。上一轮的较快 base 数字属于
专用 **v1** 图；本轮直接使用支持 activation corrections / hidden 输出的
**v2** 图测 base，不能把两种接口的性能混为一谈。

图内只有 base FFN 算子，base 权重仍逐层传入；LoRA gate/up 修正在 SiLU
之前输入，hidden 返回 GPU 算 down-LoRA。无 adapter 时传零修正。没有把
LoRA 合入 checkpoint、Core ML artifact 或 runtime base 权重槽。

## 改动前的 v2 base

原生库 `9b886bdecc1fd95db41a77da81928fc9da5df34999bc805ee2739214102c215e`。
M4 Max 64 GB、512²、resident、狐狸雪景/seed42、auto、无 profile；每模型
独立 batch 一冷两热。Qwen 40 步，c320/K1024/N512；Z 8 步，c352/K1024/N512。

| 模型 | 两个热请求（s） | v2 base 中位数 | 上一轮同库 v1 中位数 |
| --- | --- | ---: | ---: |
| Qwen | 37.853355 / 37.900858 | 37.877106 s | 37.215519 s |
| Z | 6.669581 / 6.688260 | 6.678920 s | 6.332102 s |

证据：`outputs/runtime-ane/qwen-base-tile-v2-before/` 与
`outputs/runtime-ane/z-base-tile-v2-before/`，summary 均 complete。
v1/v2 不是同一时刻交错实验，不把这张表的差值全部归因于 hidden 复制。

## 候选：未使用的 hidden 只检查，不复制

之前 `RuntimeGraph::Impl::predict` 在 base 请求上仍把 hidden 转换、恢复
headroom 并写入常驻 `hidden_scratch`，尽管没有 down-LoRA 消费它。
候选让无 adapter 的 hidden 走 validate-only，仍访问真实 Core ML 输出，
检查 shape、strides、存储长度、NaN/Inf，以及 headroom 恢复后的目标 dtype
范围；只是没有目标写入。自检用一次性的局部 hidden 缓冲检查数值。

有 adapter 时仍完整复制 hidden；主输出 y 的处理不变。保持每次 base
launch 清零两个 correction 输入，不新增可能残留适配器状态的零值缓存。
不改变图、Core ML backings、SiLU、精度、headroom 重试、调度或 CLI 默认。

代码范围：

- `native/backends/ane_runtime_convert.hpp`：row restore 允许空 destination，
  原始有限性和恢复后有限性两种检查都保留。
- `native/backends/ane_runtime.mm`：无 LoRA 时不保存 hidden 副本；真实复制
  字节计数不再计入只读检查，`output_seconds` 仍包括检查时间。
- `tests/native/ane_runtime_convert_test.cpp`：每个 FP16 bit pattern 覆盖
  SIMD+scalar tail、BF16/FP16 目标、1/16/4096 缩放，验证空 destination
  与原 scalar 有副本路径的两种失败状态一致，源数据不变。
- `tests/native/ane_ffn_test.cpp`：连续 base、adapter/signed-adapter/base、
  精确复制字节数、hidden 溢出重试与输出生命周期检查。

减少的只是 host 副本；Core ML 仍计算/提供 hidden，CPU 仍扫描它。不能把
`copied_output_bytes` 下降宣传为同等比例的总内存流量或请求耗时下降。
IOSurface slots 的字节数不变；常驻 CPU hidden scratch 减少，但完整峰值
内存没有由此获得测量资格，admission 估计仍保守。

第一版候选构建 SHA：
`e5477ac65d99dea29c18626d786e69496802e8c15ffc2e1ee3e2c045c250e24d`。
构建、4 项 host、8 项微图/集成、acceleration contracts 和 Qwen 专项通过。
LoRA / Q4-Q8 微图最坏 relative L2 仍为 0.00533939 / 0.00658136，未放宽
阈值；原有 Core ML Python 临时目录 ResourceWarning 仍存在，测试成功。

## 整请求与共图回归

相同 v2 manifest、checkpoint、请求和调度下的 before/after 分开记录，
测试先检查竞争推理，构建与微图不并行进入正式计时。候选是否保留由完整
结果判断，不只依据副本计数。base/A/合成 B/base 是状态隔离检查；Qwen
六步 base 不代表未蒸馏 base 的质量资格。

### 第一版：仅省去副本，未证明提速

`outputs/runtime-ane/{qwen,z}-base-tile-v2-after/` 均 complete，构建为
`e5477ac…`。两边各一冷两热，参数与 before 一致，但不是跨构建 ABBA。

| 模型 | 两个热请求（s） | before → 第一版中位数 |
| --- | --- | ---: |
| Qwen | 37.964602 / 37.955034 | 37.877106 → 37.959818 s |
| Z | 6.676510 / 6.651055 | 6.678920 → 6.663782 s |

没有明确整请求收益。Qwen 累计复制字节 38,252,052,480 → 9,563,013,120，
但输出处理总计 5.121 → 5.114 s（均含冷请求）；只移除写入，没有消除转换
和目标有限性计算，不能宣称它已经带来速度提升。
Z 复制字节 6,798,049,280 → 1,859,911,680，输出处理 0.922 → 0.907 s。

前后实际计数一致：Qwen 3648 predictions/hybrid、192 GPU blocks；Z
688 predictions、685 hybrid、83 GPU blocks。无错误回退，四个 trial
系统 swap-in/out 增量均为零。两模型末次 PNG 在 before/第一版之间
分别完全相同，也与上一轮对应 v1 候选的末次 PNG 相同。

第一版新 tile 的真实共图切换也通过：

| 模型 | 累计调用（预热 base / base / A / 合成 B / base） | 绑定投影 |
| --- | --- | --- |
| Qwen | 192 / 384 / 576 / 768 / 960 | 0 / 0 / 227 / 227 / 0 |
| Z | 259 / 515 / 771 / 1027 / 1283 | 0 / 0 / 238 / 238 / 0 |

证据在 `outputs/runtime-ane/{qwen,z}-v2-hidden-switch/`。首尾热 base PNG
相同、训练 A 与合成 B 输出不同、加载计时不变。B 只缩放低秩 B tensor，
不是第二个训练 LoRA 的质量验收，也没有改写 base 权重。

### 第二版：保留等价失败判定的位模式检查

无目标副本、目标 BF16、有限缩放绝对值 ≤4096 时，最大有限 FP16 值乘以
该缩放仍远小于 BF16 的有限范围。因此恢复后有限性与输入 FP16 有限性
一致，可以只扫描所有 FP16 exponent 位，省去不被消费的恢复/舍入计算。
有目标副本、FP16 目标、超界或非有限缩放仍走原来的完整转换检查。

扩展逐位测试覆盖 0、正/负缩放、±4096、4097/8192、FP32 最大值、Inf
和 NaN；比较有副本 scalar oracle 的两种失败状态，不以更宽容差替代。
这不改变 Core ML 运算，也不跳过 hidden 扫描；实际整请求收益另行复测。

## 最终复测与保留决定

保留构建 SHA：
`b55419e636ada12e54dc197b7099811225b361d63d2c1d16e6e0560449104dba`。
相同 v2 图、请求参数与 auto 调度，每路一冷两热；原始结果为
`outputs/runtime-ane/{qwen,z}-base-tile-v2-fast/`，summary 均 complete。

| 模型 | 两个热请求（s） | before → 最终中位数 | 名义耗时减少 |
| --- | --- | ---: | ---: |
| Qwen | 36.933889 / 36.989016 | 37.877106 → 36.961452 s | 2.4% |
| Z | 6.480632 / 6.405626 | 6.678920 → 6.443129 s | 3.5% |

这是串行 before/after 筛选，不是跨构建 ABBA 或多提示词性能资格。
Qwen 输出处理累计 5.121 → 2.139 s，Z 0.922 → 0.404 s；这些计数包含
冷请求，不能直接与两个热请求中位数相减。实际 predictions/hybrid/GPU
计数与 before 一致，均无错误回退；copied bytes 与第一版相同，区别是
不再对没有消费者的 hidden 做逐元素恢复/舍入。

两模型末次 PNG 与 before/第一版分别完全相同：

- Qwen：`194aac1929165e0cc76be0d3dd8cb09201d07419d1dee6839fc9ca0120a6ff25`。
- Z：`8fb48cca8303d3d9b19ed8a1a05a426f5adab7fd861dab02e82e6c77c267dcec`。

它们也就是上一轮已肉眼查看的对应 v1 base 输出；没有将单一样本的相同
输出扩大为所有输入无损的保证。最终 Qwen trial 系统 swap-in 增加 4 页
（每页 16 KiB，共 64 KiB）、swap-out 不增；Z 两者无增量。仍不是完整
峰值内存或零换页认证，不把系统增量归因于某个模型进程。

最终库的两模型共图切换复测均通过，证据为
`outputs/runtime-ane/{qwen,z}-v2-hidden-fast-switch/summary.json`：
调用/绑定投影计数与第一版表格一致、加载计时不变、首尾热 base 相同，
训练 A 和合成 B 的两张 PNG SHA 也分别与第一版一致。完整 LoRA 路径
仍返回 hidden；新快速检查只作用于无 adapter 的 base 请求。

保留这项内部优化，不增加环境开关、不修改图，也不改变默认 GPU/冻结图
选择。当前新库未重新测 GPU/冻结图计时，不能与上一库数字混算新的
加速比。也未新增 LoRA 编辑速度、1024²、多训练 LoRA 画质或硬件 residency
资格；这项 base 优化不能解释为解决了 Qwen LoRA 仍慢的问题。

最终构建、4 项 host、8 项微图/集成、acceleration contracts 和 Qwen 专项
通过。没有放宽数值门槛，未重跑全量 `make test`；其余 native 代码没有
本轮功能改动。没有 stage/commit 或删除本地产物，249 项既有暂存移除
及暂存 diff SHA 保持不变。
