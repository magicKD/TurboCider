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

- `transformer-heterogeneous-2026-09-08.json`：固定版本的 mac_transformer/ANE 证据、MLP/模型 E2E、Core ML startup 和异构采用结论。
- `private-ane-shipping-isolation-2026-09-08.json`：private `_ANE*` 源码隔离、shipping binary 字符串/依赖审计、portable package 和测试结果。
