# Qwen512：有界异步双步调度，净收益不足以采用

2026-10-18，Asia/Singapore，M4 Max /64GiB。只涉及 Qwen Image 2.1 的512²
文生图和单ref512图生图；不涉及 Z Image，不改 App 或默认值。完整优化目标
保持 active，本页收尾一个默认关闭的调度实验。

## 结论

相对 direct-QKV＋typed LoRA-B/SwiGLU 强基线，五步完整热请求中位数仅减少
约0.63%（文生图）和0.06%（前缀命中编辑）。生成信号很弱，编辑基本没有
收益，不采用为默认、不宣称新的稳定提速，也不替换
[上一轮完整请求记录](qwen512-lora-swiglu-request-2026-10-18.md)。

64次成功完整请求和两次受控部分取消完成，16组完整原始数组/RGBA对照一致。
这是已测条件下的调度等价和生命周期证据，不是任意 prompt 的语义质量、设备
故障恢复或正式性能资格。更大质量/恢复矩阵已准备但未执行，不冒称通过。

## 机制与准入

新增默认关闭的 `TURBOCIDER_QWEN21_ASYNC_STEP_PAIR=1`，recipe 为
`bounded-async-step-pair-v1`。`native/models/qwen21/async_step_pair.hpp`
只在当前请求内拥有最多两步 pending 状态，不跨请求保留编译图。

每步先 `async_eval(state)`，再构建并单独 `async_eval(all(isfinite(state)))`，
保留原 state 求值边界和每一步独立 finite 运算。下一步可在先前结果尚未
host-ready 时构图；每批 drain 等待实际 state/check 依赖并检查全部 mask，
之后才逐步发出已完成 denoise 进度。五步最后的单步也显式 drain。不减少
采样更新、模型层、LoRA、参考或 VAE/PNG 工作，不融合 finite 与状态更新。

helper 在 Transformer/权重释放前析构，异常和取消退出仍等待持有的依赖。
等待异常另尝试框架 synchronize；没有把 timeout 视为完成。未验证任意 GPU
丢失、永久 hang 或错误 epilogue 下的恢复，不作这种安全保证。

只准入已 pin r128、strength1/inference-time、显式近似许可、resident512、
普通 GPU/MLX joint BF16 A/B/F32 ranks、原六步或作者五步：生成无参考，编辑
单ref512。普通/direct QKV、普通/fused FFN、显式 immutable prefix 可组合；
direct＋prefix 仍须独立 opt-in。packed QKV、ANE、callback、staged/streaming、
其他 pilot、step profiling/calibration 等仍拒绝。snapshot key 包含 async
recipe，切换策略需要重新 prime，不复用取消 bank。

成功六步的 submitted/checked/accepted 均6，completion_batches3、
evaluation_calls12；五步分别5/5/5、3、10，peak_pending_steps 均2。
回执 `qwen_async_step_pair` 及 plan 标签
`qwen21_bounded_async_step_pair_experimental` 是逻辑记录，不是 command
buffer/kernel 计数。实际两个 state＋mask 共262146 bytes，不是整个计算图、
driver 或请求的内存上限。

## 数值、取消与恢复

raw 窗口32次完整请求：任务×六/五步×普通/direct QKV×普通/fused FFN×同步/
异步，seed73、snapshot off。16对完整 initial/text/reference/final BF16
latent 和 RGBA 逐位一致；dump/readback 时间不进入性能分母。

真实模型取消驱动在同一 Session 完成八次完整请求和两次受控取消：生成在
提交一步、编辑在提交两步时取消，两者都实际观察到有结果尚未 ready。取消前
没有大于0的完成 denoise 进度，没有取消 PNG。原 Session 重试可用，编辑
retry miss 后重新填充、再次请求 hit，最终恢复同步策略。六组完整 RGBA 比较
exact；部分取消不计为完整成功请求，重试也不充当计时样本。

真实 GPU helper 另覆盖5/6步 state bits、+0/-0、先 Inf 后恢复有限值仍拒绝、
容量/顺序、奇数尾步和 scope-exit drain。末步有限值不能替代早先独立 mask。

32次 raw＋24次 pilot＋八次取消驱动成功请求共64次；raw/pilot 的56个实际
规范化请求及 PNG 都复现历史 v7 同策略质量矩阵。不是重新做七任务广泛矩阵。
准备的 restoration36 与 quality92/46对均未运行，微弱信号不足以支持继续
花费这批完整模型验证成本。

## 完整请求 pilot

四个独立进程各六次完整请求，首次 load/JIT/prime 单列，其后五次热请求取
中位数；固定 direct-QKV＋typed SwiGLU、作者五步、seed73/74。编辑两臂计分
均实际 prefix hit。计入原有限值检查、采样、VAE/PNG 和请求内准备；不计
后置 CPU 图片审计，不混入 dump、部分取消或独立 engine constructor。

| 五步路径 | 同步 / s | 异步双步 / s | 名义减少 |
| --- | ---: | ---: | ---: |
| 文生图 | 6.1757028750 | 6.1366947079 | 0.63164% |
| 图生图，实际 prefix hit | 6.2351345830 | 6.2313809579 | 0.06020% |

文生图顺序 off→on，编辑 on→off；每配置只有一个进程、五个热样本，没有
反序复测或置信区间。四窗有三窗检出外部推理 CPU 负载，生成 on 窗口没有
检出，但匹配 off 窗口失败，因此两对仍 `qualification_passed=false`。
CPU 安静也不证明 GPU 独占，不终止用户 ComfyUI。

完整 raw/pilot 的28个 async screen 请求累计 submitted/checked/accepted
均148、completion_batches84、host_ahead_observations64，不包含取消驱动的
三个成功 async retry。host-ahead 只是下一次构图时先前依赖尚未 host-ready，
不是物理 CPU/GPU 重叠、GPU/ANE 并行或节省时间的证明。

## 来源、资源与回归

CPU 审计核对652项 native/build 输入、当前树与 archive 一致，复查 raw 数组、
实际 request/PNG、取消进度/无导出/重试和完整计时分母。固定来源为：

```text
build/qwen512-async-step-pair-v8-20261018/libturbocider.dylib
SHA256 3bfdf446c8ba9f1e74e9c568d80fe23d5f8f08036daec25252da66a1211ac562
runtime tc-runtime-build-v1-eb8adff91ec5f083a273d47e5b91fc124bdc6f7673c799282604577c093a850f

outputs/qwen512-async-step-native-source-20261018.tar.gz
SHA256 829a55dc563b54cd8e3410265d900c6e3e5aabdd8dffd9a002003ee60fd3802b

build/qwen21-async-step-cancel-v2-20261018/qwen21-async-step-cancel-probe
SHA256 ac616602974c6636adbbc18693928a1b1c2bef672d37c8061af5f0434d11a2d7
```

审计为 `outputs/qwen512-async-step-pilot-audit-20261018.json`，取消驱动源另在
对应窗口的 `source.json`。后续更改 native 源码须新建库/归档，不能用新树
冒充 v8。固定原 adapter 未变：679604800 bytes，SHA256
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。

收尾重新运行独立 CPU 审计，输出
`outputs/qwen512-async-step-handoff-audit-20261018.json`，与首份审计内容一致。

六个模型窗口 memory evidence 都 complete，最大采样 gap 约110ms，阈值
500ms，退出码0，系统 swap-in/out 均0。raw/取消进程树峰值 physical footprint
分别31953591216 /32108469336 bytes，系统 compression 分别约2.96/3.49GB；
四个计时窗口峰值约19.08–19.52GB、compression0。系统压缩不能归因某个
进程，两个逻辑 latent 的字节数也不能代替这些资源记录。

v8 构建成功；C API141项为140通过、一个既有 Wan fixture 缺失跳过；真实 GPU
helper1、普通 lifecycle1、scheduler1、options host1通过。新增取消/analyzer
六项、audit五项，连同 async screen 四项共15项 portable host 测试通过；
这些合成负控检查数据损坏、提前进度、错误边界、alpha/尺寸、来源/分母等，
不能代替真实模型或设备验证。

收尾另复跑 Qwen 专项 C API57项、Qwen screen host172项全部通过（后者包含
上述 screen/cancel 测试，不重复相加）；知识库链接/索引和16项维护测试通过，
`git diff --check` 通过。没有在这次文档/host测试收尾中重跑完整模型或改 native。

原始窗口统一前缀 `outputs/qwen512-async-step-`：`raw-20261018`、
`cancel-20261018`、`gen-five-off-pilot-20261018`、`gen-five-on-pilot-20261018`、
`edit-five-prefix-on-pilot-20261018`、`edit-five-prefix-off-pilot-20261018`。
owned 模型/构建/审计任务均已结束；历史二进制、图像和权重保留，未下载或
清理用户文件，收尾时磁盘约9.8GiB可用。

## 下一步

保留默认关闭入口和有限负结果，不扩大当前两步上限，不授予生产资格。
研究三步/六步窗口须作为新实验验证取消延迟、ownership 和内存，不能悄悄
改变本轮合同。优先转向主要 DiT 算子或真实 GPU/ANE 交接成本；小幅 host
构图减少未改变主要 GPU 算术成本。完整优化目标仍未完成。
