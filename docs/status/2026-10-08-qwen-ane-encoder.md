# Qwen Edit encoder：Private ANE runtime 接入与负结果

2026-10-08，`dev-graph`，M4 Pro／48 GiB。**语言 encoder 已接入显式 GPU／ANE runtime 诊断桥，但完整三图 encoder 变慢、未通过预声明的5%误差筛选，默认继续使用 GPU。** 没有新增 App 开关、下载模型、复制 checkpoint 或全模型导出。本轮不追加去噪或最终图生成。

完整设计、最新 Splash 固定来源和后续顺序见[19 · ANE encoder](../design/qwen-image-gpu-ane-command-graph/19-edit-encoder-ane-runtime.md)，机器可读汇总见[JSON](2026-10-08-qwen-ane-encoder.json)。上一轮 GPU attention 融合结果独立保留在[GPU encoder 状态](2026-10-08-qwen-edit-encoder.md)。

## 接线与最新 upstream

本地 Splash 与本轮远端 HEAD 一致：`0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`，2026-10-06。其当前 ANE 路径是量化语言 prefill FFN 通道切分，视觉仍 GPU；应用拥有 graph 的 input pack、GPU complement、下一层双 bank stage、event wait/join。它支持 affine Q4／未旋转的量化 GGUF，不能直接为本地 dense BF16 encoder 提供性能资格。[源码](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfn.cpp)、[校准](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfnCalibration.cpp)。

本地语言 FFN 几何为4096→12288→4096。新 TextFfnHooks 保留 attention/mRoPE/DeepStack，在 attention 前 stage当前层 base 权重，FFN 用已有 PrivateW8Graph／HybridFfn，GPU算 leading channels、ANE算 trailing channels，FP32 restore/join 后转 BF16。显式 descriptor/program、source lease、进程 recipe 和 conditioning cache 管理已接入；小于1024行保持 GPU。固定1152行图分块处理变长序列，不截断，不等同于已实现 Splash 动态 function 程序。

视觉 GELU/bias FFN、attention、参考图 VAE和 DiT仍GPU；冻结 encoder manifest 的旧拒绝保留。当前桥仍有 MLX ready fence 和 host finish/join，没有完整自有 Metal command graph encoder。W8A8 representation/private执行证据不证明硬件 INT8 算术。

## 轻量真实 FFN 筛选

[组件回执](../../outputs/qwen21-ane-encoder-20261008/real-ffn-v2.json)和[监督回执](../../outputs/qwen21-ane-encoder-20261008/screen-supervisor-v2.json)固定新库 `c1abbbeb…`（574身份源）。使用本地 Qwen3-VL BF16、既有 teapot/fox/puffin；完整视觉塔和 layer0 attention产生真实 FFN输入。只捕获 layer0，不声称完整语言编码。提示词与上一轮略不同，实际序列1071/2101/3131行。

每图数/通道数一次 warmup＋AB/BA，两通道候选共18次成功FFN，无 fallback。正式时间包括stage/input fence/GPU/ANE/join/设备完成，构造另列。纯GPU对照为编译完整 FFN，输出与原 eager oracle逐位一致。

| ANE通道 | 图数 | AB省时 | BA省时 | FFN RRMS |
|---:|---:|---:|---:|---:|
| 3072 | 1 | 8.99% | 10.22% | 0.532% |
| 3072 | 2 | 2.97% | 5.18% | 0.527% |
| 3072 | 3 | 2.45% | 4.33% | 0.527% |
| 6144 | 1 | −0.73% | −0.85% | 0.748% |
| 6144 | 2 | 3.89% | 20.16% | 0.745% |
| 6144 | 3 | 16.04% | 16.26% | 0.740% |

6144通道的三图完整FFN约126.6→106.1ms。其程序构造/self-test首次约1.423s，3072约0.983s；组件测量不能忽略 encoder 的首次prepare。两图结果随顺序波动，不能授稳定性能资格。组件全组11.34s，MLX peak与各项counter见原始回执。

第一次描述文件误用K tile64，在创建ANE程序前报 `private W8A8 K tile too small`；真实视觉/input准备约5.44s，未运行ANE。失败[原始回执](../../outputs/qwen21-ane-encoder-20261008/screen-supervisor.json)保留。随后使用仓库现有生成器默认K2048/N1024，先做无设备 MIL几何检查，再运行有效组。没有把失败改成通过，也没有无界重试。

## 完整三图 encoder

组件通过后只追加一次完整pair：三个27层视觉塔、完整36层语言编码、mRoPE/三份DeepStack、45行retained conditioning；新进程各一次，hybrid→GPU，reference_size1024、cache cap256MiB，两路都逐层释放语言权重。使用组件已准备的私有程序cache，OS/driver/source cache没有重置，所以不是全冷首启性能资格。

| 包含的阶段 | GPU | GPU/ANE |
|---|---:|---:|
| source/tokenizer load | 0.139s | 0.127s |
| 三图视觉、resize/hash | 4.503s | 4.481s |
| prompt assembly | 0.127s | 0.112s |
| ANE程序准备/self-test | 0s | 0.264s |
| 完整36层语言、释放/drain | 7.223s | 8.212s |
| 全encoder | **11.998s** | **13.202s** |

语言阶段慢13.70%，全encoder慢10.03%。36个hybrid FFN、108次ANE evaluation完成，无fallback；最终184320个FP32读出的BF16 conditioning值全部finite，shape/dtype正确，RRMS **5.754%**，超过预声明5%门槛。不是逐位等价，也未测试最终编辑图的可接受性；不在看到结果后改成8%通过。

[原始监督回执](../../outputs/qwen21-ane-encoder-20261008/full-encoder-supervisor.json)状态保持fail；[独立配对审计](../../outputs/qwen21-ane-encoder-20261008/full-encoder-pair-audit.json)明确两个native arm均已完成和source复核，失败属于数值screen。保留两个小型conditioning输出（合计约1.41MiB），没有保存大规模FFN activation。两个arm各MLX peak约2.16GiB，不含全部IOSurface/driver/compiler分配，不是App容量认证。外部前后swap读数未增加，未记录thermal warning；只有两个成功设备组，无长期压力循环。

## 为什么单层快，完整变慢

单层配对重复使用已物化的 layer0 权重和已编译GPU complement，stage每次仍计时；完整prefill每层第一次使用新的权重、compiled complement和FP32 down常量。实际完整stage counter合计0.649s，程序prepare0.264s，host ANE overlap wait合计3.538s；这些跨度会重叠，不能简单相加或说全是ANE纯计算时间。

代码上仍有三项明显成本：BF16 dense源逐层量化/旋转、GPU complement把down权重转FP32，以及MLX input全队列ready fence＋host join。当前只把本层stage放到attention前，没有Splash的next-layer双bank预取和完整设备graph。它们是下一轮优先排查项；本轮没有逐项独立counter证明各自贡献多少，不能把所有变慢归因于某一个操作。

量化误差从layer0的0.74%累计到最终5.75%，也说明不能用DiT或零输入self-test代替encoder真实域校准。下一轮先采少数early/middle/late真实encoder输入，为encoder建立专属SmoothQuant/Hadamard profile，再用BF16 operands＋FP32 accumulator/output的原生partial GEMM减少upcast，验证单block/完整prefill；有收益才迁移设备event和跨层graph。完整质量/性能通过前不导出全模型、不自动启用。

## 最终代码与验证边界

测量使用 `c1abbbeb…`；最终整理另建 `4cdbf278…`，唯一原生变化是将encoder私有W8A8的precision回执与CoreML标签分开。性能记录属于测量构建，不给最终整理构建补记新设备成绩。源码接口已进入当前pipeline，但本轮完整验证是独立encoder入口，没有运行完整Session编辑请求或UI。

最终CPU结果序列化检查、App构建及owned清理记录见同名JSON和[验证回执](../../outputs/qwen21-ane-encoder-20261008/final-verification.json)。默认GPU、不新增发布开关；候选保留供下一轮局部优化。

最终库/CLI、Swift App与model-library helper构建通过；51项无设备policy检查、MIL几何检查，以及成功/全部fallback的实际结果标签检查通过。没有App UI/安装验收，也没有给最终报告标签整理构建重跑设备成绩。[owned清理](../../outputs/qwen21-ane-encoder-20261008/owned-cleanup.json)删除1149个文件、434.11MiB（含本轮未获资格的私有程序cache），保留库/CLI/App/probes/manifest/日志和小型conditioning输出，不清理其他模型或系统cache。
