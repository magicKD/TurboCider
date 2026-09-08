# 验收证据的版本控制约定

此目录保留经审阅、被设计文档引用的验收摘要，不应整体加入 `.gitignore`。

应提交：逐张量误差统计、性能测量条件与样本摘要、测试通过/失败记录、依赖版本、模型及参考源码哈希。这些文件用于解释某个实现版本的正确性和性能结论；即使可重新生成，也需要保留当时的证据。

不应提交：模型权重、张量转储、生成图片/视频、编译缓存、完整运行日志、临时报告。统一放在仓库根目录 `outputs/`，已由 `.gitignore` 忽略。此目录内的 `raw/`、`local/` 和 `*.raw.json` 也被忽略，仅供临时整理。

新增摘要使用仓库相对路径或明确的路径占位符，避免携带本机用户名、绝对工作目录或私人输入。现有 JSON 报告中的本机工作目录、模型目录、编译诊断路径和 Xcode SDK 路径已归一化；数值、版本、哈希及失败记录保持不变。报告不是可直接执行的运行配置。

## 路径占位符

| 占位符 | 含义 |
|---|---|
| `${ISOLATED_PACKAGE_ROOT}` | 脱离源码的临时发行包验收目录 |
| `${APP_SUPPORT_ROOT}` | TurboCiderNative 的 Application Support 目录 |
| `${USER_CACHE_ROOT}` | 本机用户 Library/Caches 目录；各进程的可访问缓存抽样 |
| `${PROJECT_ROOT}` | TurboCider 仓库根目录；其下 `outputs/` 为原始本机验收产物 |
| `${FLUX_MODEL_ROOT}` | 本地 FLUX.2-klein-4B 模型根目录 |
| `${MFLUX_SOURCE_ROOT}` | 验收所用 mflux checkout 的 `src/` 目录 |
| `${COREML_PARTITION_ROOT}` | 本轮已有 FLUX Core ML 源分区及 compiled 子目录根目录 |
| `${H3_SOURCE_ROOT}` | 静态检查所用 h3.c-fork checkout 根目录 |
| `${MACOS_SDK_ROOT}` | 验收所用 macOS SDK 根目录 |

占位符是文档记号，不要求设置同名环境变量，也不会由报告自动展开。复现时将它们映射到自己的路径，并以报告中的版本和哈希确认对应源文件/模型身份。

`packaged-dependencies.json` 中 `/System/Library/…`、`/usr/lib/…` 是 macOS 标准动态库安装名，不是个人工作路径；保留原文以便审查发行包依赖。`@rpath` 等动态链接记号也保持原样。

本机启用的加速配置可命名为 `profiles/<device>.local.json`，不会进入版本控制；可移植的 `*.example.json` 应保留。

`.gitignore` 不会自动停止跟踪已经提交的文件；如未来误提交原始产物，需要另外从 Git 索引移除，并保留本地文件。

- `cpp-engine-refactor.json`：C++ 边界重构的完整版本对照、首次 Core ML 加载异常、ABBA App 性能和功能回归记录；不含本机绝对路径。

## 近似路径质量门禁

跨后端近似不要求逐 bit/逐像素相同，但不能只凭肉眼或单个相关性数字宣称通过。`tools/native/quality_gate.py` 提供离线、可复现的 RGB 配对门禁；默认策略是 correlation ≥ 0.99、cosine ≥ 0.995、MAE ≤ 5/255。默认值是当前图像实验的保守起点，不是所有模型和分辨率的普适真理；报告必须同时记录模型、尺寸、seed、prompt、dtype、参考实现和阈值。

LLaDA GPU/ANE benchmark 可用 `--require-quality` 启用该门禁；不传此参数时仍会记录 `quality_gate_passed`，但不会因为实验性近似结果而阻断性能探索。运行时不会加载参考图片做比较，因此正式推理不会隐式承担 benchmark 成本。

视频模型使用 `tools/native/video_quality_gate.py` 做离线帧级验收。除逐帧 RGB correlation/cosine/MAE 外，它还检查尺寸、帧数、帧率和 frame-to-frame motion energy；默认要求 mean correlation ≥ 0.99、最低单帧 correlation ≥ 0.95、mean cosine ≥ 0.995、mean MAE ≤ 5/255、最大运动能量相对误差 ≤ 15%。这同样不是运行时参考推理，也不代表所有视频模型都必须使用同一阈值；H3/LTX 的正式门禁还需多 prompt、seed、分辨率和音视频 mux 矩阵。

扩散纹理发生可接受偏移时，可额外构建 `build/native/vision-feature-distance`，用 macOS 公开 Vision feature print 记录感知距离。只有传入显式、同 OS/revision/workload 校准过的 `--max-vision-distance` 时，门禁才允许用感知距离替代对齐 RGB 阈值；默认模式保持不变。该指标不能单独验证 prompt 遵循、人物细节或音视频同步，设计边界见 `docs/design/vision-quality-diagnostics.md`。

当前 LTX GPU 与 GPU+ANE 的单个 704×448、97 帧配对证据见 `video-quality-gate-2026-09-08.json`：运动能量相对误差为 8.0%，但 mean RGB correlation 仅 0.850、mean MAE 21.59/255，因此没有通过质量门禁，GPU+ANE 不能自动启用。

- `vision-feature-print-2026-09-08.json`：同一 LTX 样本的公开 Vision revision 2 抽样诊断；只作感知证据补充，不构成自动放行阈值。

- `z-image-gguf-streaming-matrix-2026-09-08.json`：M4 Max 上 Q3_K_S、Q4_K_M、Q8_0 256² resident/streaming ABBA×2 矩阵；包含重复 warm、物理 footprint、预算语义和逐像素质量门禁。

- `h3-ssd-pinned-prefix-policy-2026-09-08.json`：H3 动态 pinned-prefix 的无模型权重策略测试；验证双 slot、activation reserve、预算上限和至少一个 streamed block 的 fail-closed 约束。真实 H3 权重 E2E 仍待补。

- `h3-ssd-pinned-prefix-dit-2026-09-08.json`：真实 62 GiB H3 Transformer 的 A/B/B/A DiT probe。16 GiB 预算选择 14 个 pinned block，四份最终 latent 字节完全一致；denoise 中位数约提升 1.140×，fresh 总时间基本持平。由于本机完整模型的 tokenizer/text encoder/VAE 符号链接已失效，该记录不包含完整 MP4 E2E。

- `transformer-heterogeneous-2026-09-08.json`：固定版本的 mac_transformer/ANE 证据、MLP/模型 E2E、Core ML startup 和异构采用结论。
- `private-ane-shipping-isolation-2026-09-08.json`：private `_ANE*` 源码隔离、shipping binary 字符串/依赖审计、portable package 和测试结果。
