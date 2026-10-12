# Qwen 2.1：真实 FFN、通道划分与 W8A8 本机筛选

2026-10-07，Apple M4 Pro / 48 GiB / macOS 26.6。接续当天 frozen FP16
对照。没有下载或复制源模型，没有导出 32 层新模型。使用现有 Qwen Image
2.1 BF16 与 Viggle v0.2.1 r128、strength 1。原始数值、哈希与清理记录见
[机器回执](2026-10-07-qwen-real-ffn-screen.json)。这是本机单提示词实验，
不是发布验收，也不能把原生 CLI 的数字套到目前的 App worker。

## 已实现

- 修复 `chunks=0` 时整块 GPU 编译绕过 FFN 观察点的问题。仅把选中的诊断
  观察点暴露出来，FFN 仍完整使用 GPU。普通请求没有额外采集或同步。
- 新增 `TURBOCIDER_ANE_CALIBRATION_REPLAY=1`：限定 Qwen512 无参考图、
  GPU 轨迹、关闭 DiT cache，保存层 0/15/31 的早、中、末期完整原始 BF16
  `[1,1024,4096]` 输入。每请求最多 9 个矩阵、72 MiB，逐个写盘；不会把
  稀疏的 8 行样本复制成所谓真实大矩阵。独立完成回执、模型/LoRA/seed/
  steps/prompt 绑定，base 和 LoRA 分开。
- `qwen21_real_ffn_probe.cpp` 回放生产 `HybridFfn`，复用 Qwen 的编译图、
  FP32 LoRA rank 累加、pre-SiLU gate/up 修正以及一次完整 hidden 的 down
  修正。交替三个层，计入权重 staging、A8、等待与最终 join；排除一轮
  warmup，保留两轮计时、反转 GPU/候选顺序。早期 step 1 校准，另两个
  step 检查误差。只有整个路径完成且无回退，才能作为 ANE 证据。
- Qwen base 的 GPU 通道补集现在也使用共享编译图，消除其与 LoRA/
  回放的 eager/compiled 不一致。自动校准的 GPU 配置 identity 更新为
  `shared-compiled-qwen-ffn-v2`，不能复用旧 eager 校准记录。
- 静态 SmoothQuant 导出可接收原生 LoRA 修正样本：S2 统计与 hidden A8
  搜索包含 gate/up 修正，up 修正同步除以 S2；绑定 checkpoint SHA、
  adapter identity、输入和 delta 哈希。原来只有 LoRA 轨迹输入、却仍
  按 base gate/up 计算 hidden 统计，不能称为完整的 LoRA hidden 校准。
- 新增缺失/重复输入、路径回退、冷启动重试、未参与校准样本误差、慢层
  等拒绝检查。报告不会因为只挑热样本或平均数而通过错误候选。

## 真实输入组件结果

同一 teapot prompt、seed 42。base 20 步采集 step 1/10/19；LoRA 六步采集
step 1/3/5。以下是九个深度/步骤点的配对完整 FFN 回放均值。

| 请求/方法 | ANE 通道 | GPU ms | 候选 ms | 倍率 | 最大相对 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| base，旋转 | 4096 | 41.89 | 40.88 | 1.025× | 0.00677 |
| base，旋转 | 6144 | 41.87 | 39.84 | 1.051× | 0.00819 |
| base，旋转＋提前提交 | 6144 | 约41.9 | 约27.9 | 1.502× | 0.00819 |
| base，S1＋旋转 | 6144 | — | — | 1.033× | 0.00900 |
| LoRA，旋转 | 4096 | 46.62 | 36.59 | 1.274× | 0.00686 |
| LoRA，旋转 | 6144 | 46.65 | 37.11 | 1.257× | 0.00829 |
| LoRA，旋转＋提前提交 | 4096 | — | — | 1.254× | 0.00686 |
| LoRA，S1＋旋转 | 4096 | — | — | 1.262× | 0.00696 |

这里的 S1＋旋转是用**对应 base/LoRA 的真实早期输入**分别计算 alpha=0.5
的 S1，再进入既有 Sylvester `DH/sqrt(n)` W8A8 路径；不是静态 SQ 的
同义词。没有观察到加 S1 的优势，因此不启用。

base 提前提交显著缩短 A8 host wait，表明队列提交先后关系确实影响
重叠。LoRA 已有 gate/up readiness 流量，强加同一同步策略反而稍慢。
计时中的 A8 wait、ANE wait、post-join 有重叠，不能相加为耗时，也
不能把 wait 全算成实际量化运算。上述单层窗口仍不包含完整 attention/
residual；必须继续 E2E。

## 静态 SQ W8A8：速度不能抵消误差

仅导出 layer 0。base 6144 通道、LoRA 4096 通道，真实 step 1 校准；
LoRA 使用原生 gate/up delta。fixed1024、fused-LoRA ABI、per-tensor
W8 gate/up/down、双 A8 boundary、alpha1/2=0.5、activation scale 8。
未参加校准的末期输入通过实际 Core ML/GPU 完整 FFN 路径：

| 请求 | 单层 GPU→候选 | 相对 RMSE | 决定 |
| --- | --- | ---: | --- |
| base step19 | 41.87→21.62 ms，1.937× | 0.24904 | 拒绝 |
| LoRA step5 | 46.60→36.23 ms，1.286× | 0.11159 | 拒绝 |

都超过本轮 0.03 误差门槛，未扩展导出。不能由此断言所有 SmoothQuant
都不可用：这是一个早期校准样本、该静态 per-tensor recipe 在未参与
校准步骤上的失败。下一次应先离线区分权重量化误差、静态 A8 阈值漂移
和 hidden 裁剪，再考虑更多校准样本、不同粒度或敏感层保留 BF16；
不应靠放宽误差门槛推进全模型。

沙箱外 MLComputePlan 两个图均为 22 个操作偏好 NE，29 个操作无设备
归属结果；这不是物理执行轨迹，更不能直接证明硬件 INT8 arithmetic。

Comfy H256 路径尝试在首个 weight staging 被拒绝并完整回退 GPU。
代码要求已有 ConvRot Q8 direct-code weights；当前原始 dense BF16
不能仅通过改 `data_path` 切到该 ABI。回退的零误差不是加速或质量证据。
没有为此下载/转换全模型，LoRA 的该组合没有继续跑。

## 两组 E2E

四个 fresh-process 原生请求，component_staged、512×512、同 prompt/seed，
关闭 PE、DiT cache 和采集。base 顺序 GPU→候选，LoRA 顺序候选→GPU。
组件阶段已使用过同形状程序，因此不声称 OS cache 冷启动；每次仍计入
本进程模型加载和 ANE setup。没有其它自有构建/测试与计时重叠。

| 请求 | GPU 总耗时 | 候选总耗时 | GPU 去噪 | 候选去噪 | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| base 20步 | 48.400 s | 43.044 s | 42.155 s | 35.268 s | 总耗时降低11.07%，去噪降低16.34% |
| LoRA 6步 | 20.664 s | 21.292 s | 14.183 s | 13.637 s | 去噪降低3.84%，总耗时增加3.04% |

base：6144 通道、request rows、fixed async、launch fence=1。LoRA：
4096 通道、request rows、fixed async、launch fence=0。其余为既有旋转
W8A8 GPU I/O、scale cache 开、prefetch/A8 lookahead/S1/deferred join 关。
两组分别 672/224 次真实调用，失败、回退、overflow retry 均为 0，
headroom=1。PSNR 对纯 GPU 分别 37.73/42.13 dB；人工看图构图相同、
纹理略有差异，仅一个样本，不能作为多提示词或语义画质验收。

## 当前瓶颈与后续优先级

1. **LoRA 的请求生命周期优先于进一步压低 GEMM。** 只节约0.545秒去噪，
   但每请求 ANE setup 1.187秒。先准备可取消、受内存预算约束的 Private
   graph 预加载，或复用经过模型/adapter/recipe 校验的 worker。现有 App
   每任务新 worker，没有这项跨请求收益，不能用组件热缓存作宣传。
2. **prefill 与稳态 decode 需要不同策略。** 该 prompt 实际29 tokens，
   首步为1053行，1024桶需两次 ANE；随后1024行一次即可。LoRA 首步
   GPU2.491s、候选2.979s；后续步中位2.327→2.101s。base 首步也变慢
   2.249→2.627s，后续2.097→1.721s。下一轮优先比“首步GPU、后续
   channel split”，而非让整请求永久用更大的桶。不要截断文本。
3. **base 后续重心是 attention/readiness 和 join。** 本轮 base staging
   13.10秒大部分已被隐藏，暴露 stage wait 仅0.232秒。为了消灭这个
   13秒统计而保存一份全模型 W8，会增加磁盘/内存，收益却不能按13秒算。
   pre-FFN span19.99秒包括尚未完成的 attention；post-join4.47秒还包括
   剩余 GPU 工作，不全是拷贝。后续需用完整 block 测试选择融合边界。
4. **保持设备/模型/LoRA 分开的策略。** base 提交栅栏有效、LoRA 无益；
   不能用统一环境开关推广所有请求。还未覆盖参考图、长 prompt、Z-image
   或其他强度。App 的当前实验 worker 也未采用本轮 fixed-async 配置；
   本轮没有替换 App 默认值或将 CLI 成绩冒充为 App 验收。

固定 FP16 的上一轮结果见 frozen-laptop 回执；public runtime INT8 I/O
在本机 SDK15.2 构建被合法阻止，代码需要 macOS26 SDK，未把回退当成
public ANE W8A8 实测。未来比较它必须连输入量化、权重 staging、拷贝、
同步、LoRA、输出还原一起计时。

[Splash #260](https://github.com/incoai/splash/pull/260) 的通道切分、共享事件
提交与一次校准值得参考，但其 LLM prefill 的形状/负载不能代替本机 DiT
验证。[vpipe](https://github.com/tgo-app-dev/vpipe) 的原生 Metal 路线也是
实现参考，不直接借用其模型或硬件跑分。
[Apple 的量化性能说明](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)
确认 M4 级 ANE 有更高 INT8 吞吐潜力，也明确 A8 在 CPU/部分 GPU 上
可能变慢；因此必须同时检查设备路线、完整路径与误差。

## 验证、文件与清理

- 原生库和 Swift App 编译通过；最终 native source hashes 与构建回执
  无差异。152项相关回归中149通过、3跳过；工具末次改动再跑9项、共享 SmoothQuant 12项通过。
  无 App UI 测试或发布验收。
- 新工具入口：`tools/native/build_qwen21_real_ffn_probe.sh`、
  `tools/native/qwen21_real_ffn_probe.cpp`、
  `tools/validation/qwen21_real_ffn_screen.py`。
- 设置 `TURBOCIDER_ANE_CALIBRATION_DIR`、`...REPLAY=1`、chunks=0/profile=0
  可采集；用 backend=off 避免为采集初始化 ANE。生成仍需显式 staged
  diagnostic。新采集只支持此处已验证的 Qwen512 无参考图，不推断别的形状。
- 原始结果位于 `outputs/qwen-real-ffn-20261007/`。每个最终 E2E 保存
  request、result、events、PNG、load observation；不要拿采集请求计时
  当性能数据。探针后续补了 build/capture identity 输出；最早组件二进制
  未按 arm 保存，机器回执明确此限制，没有追补虚构哈希。
- 已删除本轮无效静态源模型、编译缓存、衍生校准临时文件、输出大张量，
  以及未获 E2E 收益的4096/无效Comfy程序缓存，共回收约375 MiB。
  保留有效 base6144程序约6 MiB、原始 BF16样本144 MiB，以及小型回执/
  图片，总输出约161 MiB。原模型及其他实验缓存未动。
