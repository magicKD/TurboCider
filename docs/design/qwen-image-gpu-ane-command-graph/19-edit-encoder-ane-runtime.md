# Qwen Edit encoder 的 GPU／ANE runtime 候选

更新：2026-10-08。本轮不下载、复制或导出模型。新增的是现有 MLX 语言 encoder 的实验桥接，独立 Native Metal Graph encoder 尚未实现。默认仍使用 GPU，不能把 DiT 的历史 ANE 收益转授给 encoder。

本文保留第一版候选和预声明5%筛选的历史结论。后续修复GPU partial与compiled复用后，完整encoder出现约3.93%净收益，见[20](20-encoder-runtime-optimization.md)；后续8% encoder数值筛选不覆盖本轮5%失败。

## 1. 最新 Splash 值得参考的部分

本地 Splash HEAD 与本轮远端 `HEAD` 查询一致，固定为 `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`（2026-10-06）。参考实际源码：

- [AneFfn.cpp](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfn.cpp)：语言 prefill SwiGLU 通道划分、动态权重、GPU input rotate/pack/join、双 bank、设备事件和失败恢复。
- [AneFfnCalibration.cpp](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfnCalibration.cpp)：测量完整 split path，要求预测收益至少 5%，按行数检查、保存校准，并处理运行中失去收益的情况。
- [AneFfnStartup.hpp](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/engine/AneFfnStartup.hpp)：启动与内存准入的职责。

它的 ANE FFN 接口目前接受 affine Q4 或未旋转的量化 GGUF projection。不能直接接入本地 dense BF16 checkpoint，也不能把 upstream LLM 的收益当成图像 encoder 的实测收益。视觉塔仍走 GPU。`AneFfn::add` 在应用自有 command graph 中安排输入打包、GPU complement、下一层权重预取和 event wait/join；这比在 MLX 边界用 CPU 等待更容易控制并行。

最新接口还要求hidden为2560的整数倍；本地encoder H4096不符合这个直接调用条件。下文“几何相符”仅指TurboCider已有executor，不指Splash最新API。

本地 Qwen3-VL 8B 的语言 FFN 是 4096→12288→4096 SwiGLU，与现有 runtime executor 的几何相符。视觉 FFN 是 1152→4304→1152 GELU，且有 bias，不能借语言 SwiGLU 图直接替代。多图编辑是一次 36 层 causal prefill，另有 mRoPE、三份 DeepStack、retained conditioning 和独立参考图 VAE；没有 LLM decode 的重复摊薄机会。

## 2. 已接入的边界

实现入口：

- [text_encoder.hpp](../../../native/models/qwen21/text_encoder.hpp)／[text_encoder.cpp](../../../native/models/qwen21/text_encoder.cpp)：可选 `TextFfnHooks`。prepare 在 attention 前提交当前层权重 staging，execute 在 post-attention norm 后处理完整 FFN；原 attention、mRoPE、residual、DeepStack 和层释放顺序保留。
- [encoder_runtime.hpp](../../../native/models/qwen21/encoder_runtime.hpp)：复用 `HybridFfn` 与 PrivateW8Graph 的 device source、owner、finish/drain、整段 GPU 重算和内存准入。每层只保留当前 compiled GPU complement，避免累计 36 层 FP32 down constants。
- [encoder_runtime_policy.hpp](../../../native/models/qwen21/encoder_runtime_policy.hpp)／[pipeline.cpp](../../../native/models/qwen21/pipeline.cpp)：显式进程诊断入口、请求范围、held descriptor/source generation 和条件缓存失效。
- [results.mm](../../../native/platform/apple/results.mm)：encoder 与 DiT 分别报告实际后端；只有成功 hybrid FFN 数大于零才报告 encoder GPU/ANE，完整 fallback 不伪装成 ANE 加速。

实验范围：Qwen Image 2.1、image.edit、512² 输出、1–3 张图、reference_size512/1024、原 MLX DiT，`allow_approximation=true`。编码行数至少 1024 才创建 ANE 会话。冻结 `encoder_ane_manifest` 请求仍不支持，不能用新入口声称已有冻结 encoder 模型。encoder 是 base 权重；DiT LoRA 的正常 MLX 路径与 encoder 的 ANE 资格分开。

进程诊断入口是 `TURBOCIDER_QWEN21_ENCODER_RUNTIME_MANIFEST`，值为规范化绝对JSON路径。它要求同时显式设置 private backend/授权、W8A8、正的partial channels、`TURBOCIDER_RUNTIME_ANE_CHUNKS=1`、`TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC=1`和`TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN=1`，profiling关闭，不能同时开启per-prompt rows。进程首次读取后固定，不是App用户选项。只做独立encoder复现时用probe的`--ffn-screen`／`--encode-full`入口；本轮没有验证含该进程入口的完整Session编辑请求。

候选使用显式授权的 private API、W8A8 Hadamard representation、固定 descriptor bucket、固定异步通道分工、FP32 partial join。量化 INT8 representation／private program 执行证据不等于硬件 INT8 算术或所有算子 placement 证明。没有本轮 SmoothQuant profile，也没有自动 channel selection 的资格。

## 3. 变长行数与数值合同

每层独立 FFN 没有 token 间依赖，可把真实序列补到 descriptor bucket 的整数倍，分块执行，然后切回真实行数。attention 始终消费原序列、原 causal/key-valid mask；FFN padding 不进入 attention，也不截断 prompt。选择一个 1152 行形状：本轮提示词的 1071/2101/3131 行分别占 1/2/3 块（上一轮不同提示词是1074/2104/3134），避免为每个 prompt 编译新程序。这里是固定图分块处理变长输入，不是已经实现 Splash 多 function 动态行数程序。

GPU leading channels 使用原 BF16 gate/up/SiLU 值边界，down 用 FP32 partial；ANE trailing channels 做 W8A8 rotate/pack／SwiGLU／down，GPU restore 后加 FP32 partial，最终转回 BF16。完整失败时必须排空既有执行器再重算完整 GPU FFN，不能发布只含 GPU partial 的结果。

descriptor 和 text/tokenizer/VAE 使用同一 source lease generation 管理，已持有的源在设备完成后复核。进程 recipe 固定，阻止用变化的 runtime 配置复用旧 conditioning。program cache 只包含 geometry/recipe 程序，不能充当 checkpoint、校准或图像内容缓存。

## 4. 测量合同

[原生 probe](../../../tools/native/qwen21_ane_encoder_probe.mm)链接明确的新库并核对 runtime identity；[构建脚本](../../../tools/native/build_qwen21_ane_encoder_probe.sh)先运行纯 CPU 合同。`--shape-contract` 只解析 descriptor 并生成两种通道几何的 MIL，不创建设备或读取 checkpoint，提前拦截不兼容 tile。

真实组件使用既有 teapot/fox/puffin，完整 27 层视觉塔得到 merged/DeepStack，再用完整 layer0 attention 得到真实 post-attention-norm FFN 输入。捕获只执行一个语言 block，不能称作完整 encoder。输入不落盘、不复制 checkpoint；释放视觉权重后测试 3072／6144 ANE channels。

每个图数/通道数只有一次 warmup 和 AB、BA 两个配对。纯 GPU 对照是编译后的完整 FFN，并与原始 eager FFN做输出对照。hybrid 正式时间包含 prepare/stage、输入 ready fence、GPU complement、ANE、join、eval+synchronize；构造/program/self-test 开销另列，完整 encoder 时必须加回。预声明张量 RRMS≤5%，shape/dtype/finite 独立验证，不在看到结果后放宽门槛。

只有每个完整 path 通过质量及物理完成检查，并出现两种顺序均至少 5% 净收益，才考虑完整 encoder／请求。没有收益就结束候选，不追加全模型导出或长期扫描。原始失败、成功、build/source/binary identity 和系统观察保留在状态回执，缓存只清理本轮拥有的目录。

## 5. 后续优化顺序

1. 先根据实测分清 program prepare、每层 BF16→W8 转换、GPU complement、ANE 临界路径和 host join。`async_ane_host_wait_seconds` 是主机等待，不是纯 ANE kernel 时间，不能直接相加当作独占设备耗时。
2. 当前 GPU FP32 down partial 需要将权重转换到 FP32。这增加读取和 live bytes，也可能失去 BF16 GEMM 吞吐。优先实现 BF16 operands＋FP32 accumulator/output 的原生 partial GEMM；仍保留 BF16 gate/up/SiLU 边界，不能为速度偷偷换数值合同。
3. 借 Splash 的双 bank 和 next-layer W stage，但测量并行 staging 是否挤占当前 attention/GPU FFN 带宽。当前仅提交本层 stage；没有一层提前预取，也没有无主机等待的多层设备图。
4. 将当前 `mx::eval/synchronize` ready fence 与 host finish/join 换成拥有真实 buffer owner 的 device event handoff，先验收单 block 的故障解除、晚到执行、隔离与 whole-block replay。MLX 能产生 graph 不代表 TurboCider 已获得其 Metal command buffer 提交权。
5. Native encoder 需新增 Q32/KV8 causal GQA、mRoPE、VL norm、D72 vision attention、DeepStack 和 prompt assembly 的共同执行计划；DiT 非 causal attention 和后处理不能直接代用。先完整 GPU oracle，再 ANE channel executor；共享 arena/PSO/program管理，模型语义与资格独立。
6. 只在真实 encoder 输入上校准量化。可比较 encoder 专属 SmoothQuant+Hadamard，而不是复用 DiT profile。模型/recipe/source generation、校准域、通道范围和 bucket 进入校准 key；小图/短序列无收益时保留 GPU。
7. 实际完整编辑中的首次与重复 prompt/cache miss 有净收益后，再增加自动路由和 App 高级配置。无需在未获资格时向用户展示多个不稳定 backend 开关。

本轮 layer0 三图6144通道两顺序省时约16%、RRMS0.74%，但完整三图encoder **11.998→13.202s（慢10.03%）**，retained conditioning RRMS **5.754%**，未通过预声明5%筛选。保留显式诊断实现、默认GPU，不追加整图或产品开关。负结果、counter、源复核及构建边界见[状态记录](../../status/2026-10-08-qwen-ane-encoder.md)。
