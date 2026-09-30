# Runtime FFN：稳定混合层的异步 GPU head

## 候选范围

针对设计稿的 exposed overhead / overlap 目标，继续沿用 VPIPE 的独立
ANE worker 和 GPU head / ANE tail 行切分；不复制其私有 graph emitter。
当前稳定层虽然不再做完整 block 的计时屏障，`HybridFfn::run` 仍会先
`eval(head)`，再 join ANE、复制输出、提交 down-LoRA 和最终 concat。

候选仅在 `HybridUntimed` 且 profile 关闭时改为：

```text
launch ANE → async_eval(GPU head) → join ANE
                                  ↓
                     复制独立输出 / 提交 down-LoRA
                                  ↓
                        eval(concat(head, tail))
```

当 ANE 先完成时，host 输出处理可与尚在运行的 GPU head 重叠；最终输出
fence 不移到请求计时之外。采样层、固定 chunks、profile 模式仍走原同步
路径，调度门槛/采样周期不变。保持完整 SiLU/LoRA 顺序，base slots 不含
adapter，Core ML artifact、headroom、dtype 和误差阈值均未改变。

出错后仍重算全部 ANE tail。取消或 down 回调异常时排空 ANE 与已提交的
GPU head；保留原始异常。输出仍复制到独立 MLX storage，不能借用可复用
scratch 给后续 lazy consumer。没有新的 CLI/环境开关。

## 遥测边界

- 新 `async_hybrid_blocks_session_total` 是 untimed hybrid 的子集。
- 新 `async_ane_wait_seconds_session_total` 是异步 GPU 提交后的 host join
  窗口，可能与 GPU 工作重叠，**不是暴露的 ANE 时间**。
- 老 `gpu_ffn_seconds_session_total` 和 `join_seconds_session_total` 只累计
  保留同步 head 边界的执行；不能把两个计数的下降当成 GPU/ANE 变快。
- `post_join_seconds_session_total` 仍是 join 后到最终输出就绪的窗口，
  异步路径中可能包含剩余 GPU head 等待；也不是纯复制/LoRA kernel 时间。
- 整请求 `request_wall` 与完整 FFN 窗口保留。报告校验兼容旧 receipt，
  新计数要求完整、有限、非负、不倒退，async 是 untimed 子集，wait 不超过
  FFN 累计时间。profile 使用同步路径，不能和正式无 profile 结果混比。

## 已完成的原版基线

M4 Max 64 GB、resident、512²、正常 base 步数、狐狸 prompt/seed42、
auto、profile 关闭。每模型两个独立 trial，每 trial 一冷两热；包含 VAE/PNG，
表中是四个热请求中位数，不是 shell 启动总时间。每 trial 预检竞争推理，
benchmark 不与构建/测试并行。

| 模型 / v2 图 | 原版四个热请求（s） | 中位数 |
| --- | --- | ---: |
| Qwen 40 步，c320/K1024/N512 | 36.965290 / 37.077612 / 37.048474 / 37.288228 | 37.063043 s |
| Z 8 步，c352/K1024/N512 | 6.477449 / 6.417415 / 6.462889 / 6.414247 | 6.440152 s |

原版库 `4d864b4439adb23cd53aa90dfe2884a3ced46027a4d24b4ac3e5ba2126ee6eac`。
每个 Qwen trial 3648 hybrid / 3264 untimed，Z 685 / 558；无错误回退，
系统 swap-in/out 页数无增量。末次热 PNG 与此前 base 回归分别相同。
证据：`outputs/runtime-ane/{qwen,z}-base-async-join-before/summary.json`，
两份均 complete。

## 候选正确性

候选库 `8d90f74ad94e63d4a1dfc1f4ce9cf5220f27b0f1491a3a2ddd2c25a722843296`。
构建、4 项 host、29 项报告契约、8 项共图工具契约、11 项 CLI 契约通过。
8 项 Core ML/MLX 微图集成与 Qwen 专项通过。LoRA / Q4-Q8 最坏 relative
L2 仍为 0.00533939 / 0.00658136，阈值未放宽。

新增测试用可控的 whole-block 采样进入稳定模式，覆盖 base/LoRA、
BF16/FP16/FP32、profile 隔离、异步计数、保存输出/hidden、取消、down
回调异常与完整 tail 重算；这些人工时间只测试调度契约，不代表性能。
首轮失败来自新增测试回调的归约提升 dtype，未按接口回到 base dtype；
修正测试 oracle/callback 后重跑通过，没有削弱产品 dtype 检查。

## Base before/after 筛选

候选保持同样工作负载、v2 图和采样次数；不是跨构建 ABBA，因此下表是
顺序 before/after 筛选，不能把小幅变化包装成跨工作负载的确定收益。

| 模型 | 候选四个热请求（s） | 原版 → 候选中位数 |
| --- | --- | --- |
| Qwen | 36.665325 / 36.646608 / 36.661301 / 36.863172 | 37.063043 → 36.663313 s，名义少约 1.1% |
| Z | 6.367167 / 6.313095 / 6.350808 / 6.311766 | 6.440152 → 6.331952 s，名义少约 1.7% |

Qwen 完整 FFN 热请求累计中位为 18.537736 → 18.279838 s，pre-FFN 为
16.202943 → 16.079305 s。Z FFN 为 3.032745 → 3.030690 s，几乎不变，
pre-FFN 为 3.137668 → 3.041261 s；不能把 Z 整请求差异全归为异步 join。
GPU 分支计数从全量改为采样子集，不使用其下降计算提速。

每个候选 Qwen trial 3648 hybrid / 3264 async，Z 685 hybrid / 558 async，
与各自原版 hybrid/untimed 数目一致，无错误回退。每个模型两张末次热
PNG 的 SHA 与原版分别相同：Qwen `194aac19…`、Z `8fb48cca…`。

内存边界：两模型 MLX allocator peak 分别约 20.606 / 16.029 GB，前后
基本相同；不包含 Core ML/驱动/OS。候选 Qwen 两个 trial 的系统 swap-in
增量为 172 / 28 页（16 KiB/页，总计 3.125 MiB），swap-out 均无增量；
Z 两个 trial 均无 swap-in/out 增量。系统数字不能归因到本进程，**本轮
不能称为所有 trial 零换页，也不是完整峰值内存验收**。

证据：`outputs/runtime-ane/{qwen,z}-base-async-join-after/summary.json`，
两份 complete。尚未在新库重测 base GPU/冻结图，不把旧构建加速比改标
为新库。

## 同构建真实 LoRA ABBA

仍为 `8d90f74…`，每模型 GPU → runtime → runtime → GPU，每 trial 一冷
两热，各路线四个热样本。Qwen 使用原始 Viggle v0.2.1 r256、6 步、seed29，
三张有序 ref512、输出 512²、c288/K=N1024 v2；Z 使用原始 distill patch、
strength1、8 步、狐狸/seed42、c352/K1024/N512 v2。均为 inference-time，
不合并权重，auto、profile 关闭。

| 模型 / 路线 | 四个热请求（s） | 中位数 |
| --- | --- | ---: |
| Qwen 三图 / GPU | 12.943522 / 12.940348 / 12.911640 / 12.914224 | 12.927286 s |
| Qwen 三图 / runtime | 14.540811 / 14.457723 / 14.014861 / 13.795172 | 14.236292 s |
| Z LoRA / GPU | 8.604200 / 8.623825 / 8.560678 / 8.564233 | 8.584217 s |
| Z LoRA / runtime | 7.767870 / 7.707956 / 7.814912 / 7.813777 | 7.790823 s |

Qwen GPU/runtime **0.908×**，runtime 仍慢约 **10.1%**；Z 为 **1.102×**，
runtime 少约 **9.2%** 耗时。此前 readiness 库为 0.907× / 1.097×，并非
跨构建交错消融，不能把这些小差异解释为本次改动确定带来的 LoRA 提速。
维持 Qwen 六步/编辑优先 GPU；没有匹配的冻结 LoRA 三路对照，不宣称
超过该路线。

Qwen 两个 runtime trial 的热请求 hybrid / async 数分别为 57/20、25/25
和 42/8、10/10；Z 四个热请求均为 248/248。Qwen 大部分层仍关闭混合，
此优化并未解决短 LoRA 请求的修正、同步、校验和 GPU block 边界成本。
八个 trial 无错误回退，系统 swap-in/out 页数均无增量。

已肉眼核对最后的 GPU/runtime 输出：Qwen 两把壶、龙贴纸、桌面和暖光
接近，两路都没有还原蓝壶透明材质；Z 狐狸姿态、雪景接近，毛发和枝叶
有局部差异，未见新增明显崩坏。GPU PNG 与此前分别相同，runtime PNG
因 auto 实际混合选择不同而不要求逐位相同；不称更广图像质量验收。
证据：`outputs/runtime-ane/{qwen-edit3,z}-async-join-auto-abba/summary.json`，
两份 complete，保留原始 PNG、请求、参考 SHA、JSONL 和预检快照。

## 共图切换、回归与保留决策

同一候选库、原有 v2 图、固定 chunks=1，分别运行预热 base → base →
训练 A → 合成 B → base：

| 模型 | 累计 Core ML 调用 | 绑定投影 |
| --- | --- | --- |
| Qwen | 192 / 384 / 576 / 768 / 960 | 0 / 0 / 227 / 227 / 0 |
| Z | 259 / 515 / 771 / 1027 / 1283 | 0 / 0 / 238 / 238 / 0 |

返回 base 相同，base/A/B 的 PNG 与此前 readiness 构建分别相同。证据为
`outputs/runtime-ane/{qwen,z}-async-join-switch/summary.json`。B 仅为训练
A 的低秩 B tensor 减半，不代表第二个训练适配器的质量；Qwen 六步 base
只作状态检查。固定分区切换不直接覆盖异步 steady 模式，该模式由前面的
真实 auto LoRA 对照与新增微图契约覆盖。

完整 `make test` 成功退出，包含 52 项加速契约（4 host + 29 报告 +
8 共图工具 + 11 CLI）、8 项布局、7 项独立性及其他模型/streaming 回归。
缺夹具、专用构建或 GPU opt-in 的 skip 不计为覆盖。新增报告校验也拒绝
同一 session 中异步计数突然消失；历史 receipt 仍接受。

保留这项同步顺序优化，仅作用于显式 runtime 的稳定、非 profile 层。
Qwen base 本轮 FFN/整请求均有小幅改善；Z 没有观察到整请求回退，但其
因果收益仍需进一步交错消融。数学、误差检查、输出所有权与失败排空均
保留；不新增开关，不改变 GPU/冻结图默认或已测最快 base 推荐。

未新增 base GPU/冻结图配对数据、1024²、真实 GGUF 或更多训练 LoRA
性能资格；物理 ANE residency、完整峰值内存与更广图像验收仍未完成。
没有删除模型/缓存/PNG，也没有 stage/commit，既有暂存改动保持不动。

最终复核：52 项加速契约再次通过；用最终校验器重验六份实际 base
before/after 与 LoRA ABBA 报告，均 complete 且遥测有效；230 个本地
文档链接目标存在，`git diff --check` 通过。249 项既有暂存移除及其本地
文件不变，暂存 diff SHA 仍为
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`。
所有本轮构建、推理和测试均已结束，未留下运行中的 benchmark。
