# TurboCider Native 本机验证记录（早期纵切历史）

**当前实现已继续演进：请先阅读 [本次重构状态](rewrite-implementation-status.md) 与 [最终性能对比](flux-performance-comparison.md)。本文以下时间点早于图像输入、服务队列和共享 Core ML backing 实现。**

日期：2026-09-05。范围：新建原生 FLUX 纵切、三模型共同计划契约、Swift App/SDK/CLI、实验 Core ML 分区。未下载任何模型。原有 Python/Swift 系统与三个参考引擎未被替换或删除。

## 结论与完成状态

| 模块 | 实际状态 |
|---|---|
| FLUX.2-klein-4B text-to-image | 原生 Qwen3→DiT→Euler→VAE→PNG 已执行；两组真实测例各 22 组张量逐元素一致 |
| App | 最终打包 SwiftUI App 实际生成、取消后恢复、图片预览；可读取重启前记录 |
| CLI / Swift SDK / C ABI | 共用同一动态库；取消、恢复、缓存切换和任务持久化通过 |
| Metal | MLX C++ Metal 后端与自定义 BF16 Euler kernel；尚未拥有独立 Tensor allocator 或完整自有算子库 |
| GPU + Core ML | 20 个 single-block MLP INT8 子图真实运行；公开 cpuAndNeuralEngine，实际 ANE residency 未确认，实验路线 |
| H3 / LTX | 共同 recipe/shape/schedule/readiness 契约已加入；数学 executor 尚未迁入，真实模型验收待定 |
| 独立 daemon、统一驻留池、通用图执行器 | 尚未实现；当前为嵌入式 FLUX session，Recipe 为预检元数据 |

新实现位于 `native/`，构建/验证工具位于 `tools/native/`。它是可以真实运行和继续演进的第一阶段基础，不是全部目标已经完成的通用推理平台。

## 环境与依赖

Apple M4 Pro，48 GiB 物理内存，macOS 26.6 build 25G72；Metal recommended working set 40,200,896,512 bytes。MLX C++ 0.32.0，现有完整 Xcode 16 系列 compiler/SDK 15.1，arm64 构建。仅本机系统完成运行验证。

模型固定为本地 FLUX.2-klein-4B 原始权重。19 个配置/tokenizer/safetensors 文件的 SHA-256 及大小记录在 [依赖与模型身份](validation/dependencies-and-model.json)。参考是本地 mflux checkout，HEAD 与 dirty 状态同时记录；本地参考存在未提交修改，因此还保存了 [全部参考 Python 源文件 SHA-256](validation/reference-source-snapshot.json)。原生源码身份见 [源码快照](validation/native-source-snapshot.json)。开发 Python 只运行离线 oracle，强制 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE，不进入发行推理链路。

## 数值验证

| 测例 | 输入 | 结果 |
|---|---|---|
| English-512 | fox prompt，512×512，4 steps，seed 42，dynamic text | 22/22 张量完全一致，含最终 BF16 RGB 像素 |
| Chinese-fixed | 中文猫提示词，320×256，4 steps，seed 42，固定 512 text tokens | 22/22 张量完全一致，含最终 BF16 RGB 像素 |
| Tokenizer | 英语、中文、combining marks/emoji/数字、换行、special tokens | 5/5 完整 token IDs 一致 |
| Metal Euler | 4097 BF16 元素，4 个非平凡 dt，包含非整线程组尾部 | 与参考融合 step 逐元素一致 |

每个完整测例包括 conditioning、initial latent、4×noise、4×latent、VAE unpack/post/in/mid/attention/up/norm，以及 pixels。要求 shape 相同、全 finite、exact equality；`compare_tensors.py --require-exact` 任意失败返回非零，不使用自动宽松阈值。

证据：[512 parity](validation/flux-512-parity.json)、[中文固定桶 parity](validation/flux-chinese-fixed-parity.json)、[tokenizer](validation/tokenizer-parity.json)。为定位问题拆开的参考 VAE 与原始 decode 完全一致，见 [参考分解核验](validation/reference-decomposition-check.json)。完整原始张量和 PNG 保留在本机 `outputs/native-validation/`，未加入版本控制，也未上传。

这证明本机上述 fixture 的数值一致，不外推为全部 prompt/shape/MLX 版本/硬件的保证。768/1024/2048 及长文本极限尚未完成全矩阵验收。

### 实际发现并修复的问题

1. SiLU 中间 BF16 舍入不一致：使用相同融合语义，恢复 Qwen 与 DiT exact parity。
2. Euler scalar/乘积舍入顺序：Metal kernel 保留 BF16 乘积，再加 latent；非平凡 dt 的 kernel 对照覆盖此风险。
3. VAE BN eps 在 C++ 被提升为 FP32：显式 BF16 scalar 后，首个 unpack 差异及后续所有像素差异消失。
4. Swift serial DispatchQueue 切换 OS 线程后 MLX 报 `There is no Stream(cpu, 0) in current thread`：使用可跨线程流并以进程级执行锁串行保护，取消/恢复测试重新通过。
5. 导出是任务提交边界：export 开始事件之后再检查取消；文件原子提交后的完成事件不再把成功结果转成 cancelled。导出前回调取消有专门回归测试。

早期失败日志与中间输出仍保留在 outputs 中；不得引用早期 `parity-512-accepted.json` 的文件名作为最终验收依据。正式门禁证据是本目录 validation 下的两个 exact parity 报告。

## 生命周期与接口

无权重 contract suite 7 项通过，包含 17 组错误请求、错误 JSON、缺失模型、H3 unavailable、LTX 8+3 与实际解码 shape。错误 mode/未知字段不静默忽略。

真实 Swift 生命周期测试通过：取消在去噪和导出前边界均生效、同一 session 恢复、连续三次相同 PNG、prompt cache 命中、更换中文 prompt 失效、fixed 512 text、任务持久化。三次暖生成 MLX active bytes 均为 7,851,302,214，未见该样本内累计增长；此项不是长期压力或泄漏证明。见 [生命周期报告](validation/lifecycle.json)。App smoke 还检查了输出可由 NSImage 解码、重启记录恢复。

最终 App 的 GUI 验证：模型目录选择→生成→实际阶段事件→输出预览；点击取消后出现 cancelled 状态，随后重新生成成功。App 为 native SwiftUI，未依赖浏览器或 Python 服务。

## 性能与混合精度实验

同一 fox prompt、512×512、4 steps、seed 42、dynamic text。每条路线一个新进程，1 次首请求后 3 次同 prompt 暖请求；没有 tensor dump。此前已多次运行，所以 OS 文件缓存和 Core ML 编译缓存已热，**以下首请求不是清空系统缓存后的冷启动**。GPU 与 hybrid 顺序测量，无统计置信区间，不能作为跨设备或正式性能承诺。

| 路线 | 新会话首请求 | 三次暖请求中位数 | 最大 MLX allocator peak |
|---|---:|---:|---:|
| GPU BF16 | 6.397 s | 5.880 s | 9,900,577,880 bytes |
| GPU + Core ML INT8 MLP | 4.777 s | 3.483 s | 9,900,594,264 bytes（不含 Core ML） |

完整每次 text/hybrid setup/DiT/VAE/墙钟、内存、Core ML 调用及复制统计见 [性能原始结果](validation/performance.json)。权重惰性 materialization 可能落入对应阶段计时；未单独声称磁盘加载成本为零。MLX peak **不包括 Core ML、OS 或文件缓存**，不能据此比较两条路线总内存。

混合路线使用既有 schema 2、bucket 1088、K=N=3072 的 20 个 Core ML artifact，每张图调用 80 次。源 checkpoint path+size 验证，但旧 artifact 没有 source SHA，因此仍属于 provenance 不完整的实验。输出 backing 允许复用，交给 MLX 时存在独立复制，未宣称零拷贝。

与 GPU PNG 比较：8-bit RGB MAE 1.858、RMSE 5.460、最大差 187、PSNR 33.39 dB。人工查看没有明显破图，但没有预先设定的量化质量门禁或多 prompt 质量评估，**不能判为质量验收通过**。性能差异伴随 INT8 近似，不能宣传为相同数值路径加速。`auto` 保持 GPU；必须显式 allow_approximation 和 artifact manifest 才进入 hybrid。没有将 Core ML 允许 ANE 执行等同于证实 ANE 驻留。

## 打包与独立运行

`dist/TurboCiderNative.app` 和 `dist/native-cli/` 携带自身 dylib/metallib，通过本机 ad-hoc 签名校验。从 `/private/tmp` 调用打包 CLI 的 self-test 与真实 FLUX generate 成功，输出与最终 GPU 基线 PNG 一致。实际加载依赖见 [发行依赖检查](validation/packaged-dependencies.json)；无 libpython，也不需要 mflux 源码作为运行 cwd。

CLI/App 不附带模型；本轮未下载任何模型。当前仍依赖 Apple 系统框架及 bundled MLX 动态库，不是独立开发完成全部 Metal 算子的宣称。尚未做 Developer ID 签名、公证、App Store sandbox、升级迁移、跨机器安装验证。

## H3/LTX 无权重结果

34 个主要参考源文件静态语法检查：33 通过、1 失败。

- H3：18/19 通过；`h3_metal.m` 引用 `MTLGPUFamilyMetal4`，本机 SDK 15.1 不存在该声明。需要新 SDK，或在兼容构建中加编译期 feature guard；runtime @available 无法解决声明缺失。
- LTX：15/15 通过，包括 C/Objective-C、Core ML 分支、MLX C++ VAE/upsampler。

这不是新 H3/LTX executor 编译或推理通过；源码仍是参考资产，未用假权重或伪媒体代替测试。详见 [原始静态记录](validation/video-static-audit.json) 和 [迁移表及真实模型验收标准](video-model-acceptance.md)。

## 下一阶段的退出条件

1. 注册表化 ModelSession，PreparedPlan 真正绑定 executor；迁入 H3/LTX 原生数学子图，按逐层→单步→完整媒体门禁验证。
2. Storage/CompletionToken/WorkspacePlan 接管跨后端所有权与完成事件，再引入独立 Metal allocator 和热点替换。
3. 建立包含 Core ML footprint 的真实准入、模型驻留池、长期压力和异常恢复测试。
4. 抽出 daemon/XPC、CLI/SDK 客户端；同一 schema 覆盖提交、事件、取消、结果和进程重启。
5. 扩展图像编辑/参考输入、视频和音频 UI；每项只有真实模型验收后才显示可用。
6. 扩充 FLUX 精度和性能矩阵，并为 ANE 量化路线预先制定质量、驻留和端到端收益门禁。
