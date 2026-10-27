# main 合并至 dev：整合与同机性能复核

2026-10-15，Apple M4 Max / 64 GiB，macOS 26.6.2（25G83），MLX 0.32.0，外接电源。

## 合并与验证

- 合并前 dev：`5d9d4e7`；拉取并快进后的 main：`2a4c2b5`。
- 合并提交：`c3e3e56`，两个父提交分别为上述 dev/main；未推送远端。
- 保留 dev 原有未推送提交和未跟踪的 `tools/validation/seedvr2_time_leaf.py`。
- 八处内容冲突：AccelerationView、App、JobStore、StudioState、Swift SDK、native request/results、App build script。
- 请求 schema 同时保留 Qwen 新字段和 SeedVR2 VAE 字段；消除 Swift residency 重复定义；保留 App-only 与 SeedVR2 定向构建入口。
- 处理文本冲突以外的语义问题：Real-ESRGAN 和 SeedVR2 都使用 `image.upscale`，历史复用/路径标签按模型区分；SeedVR2 继续使用自己的图像/视频工作区、源媒体类型、完整帧数/FPS、MP4 扩展名和空提示词。新增相应历史回归，避免误用辅助超分参数。

独立构建均为 `-O2`、private-ANE-enabled、无 test hooks/audit/实验 VAE，使用相同现有 MLX 与 SDK，没有下载模型、转换 checkpoint 或替换已安装 App。

| 构建 | native runtime identity | 源文件数 |
| --- | --- | ---: |
| before | `tc-runtime-build-v1-9ad3dd76b5b83b5757f6576873eab7e5c2b6ff8bda3e0a39e7242d3b237d1b2c` | 553 |
| after | `tc-runtime-build-v1-3f8f65b25b8967291eade737b89670a0549dbc9a99566829ad8a750fbf53b808` | 648 |

新库 648 份源摘要与合并树逐项一致。before/after 库、CLI、最终 Swift App/模型库助手均已编译；不覆盖历史 benchmark 二进制。完整 App 测试编译在相关目标和 App 就绪后主动收窄，随后以 SeedVR2 定向模式完整重建最终 App/助手/测试；没有声称运行全仓 `make test` 或全部 App 测试。

已通过：benchmark 证据/工具 241 项（另 1 项条件跳过）、repository 17 项、公共请求合同 129 项（1 跳过）、Qwen 专项 104 项（7 跳过）、ANE host 17+24 项及 worker 合同、合并评测工具 2 项、ComfyUI gateway 7 项。Swift Playground、画布尺寸、参考图准备、Qwen 工作流、Studio 与 SeedVR2（含视频导入和新增历史回归）通过；这些 App 检查不构成真实模型画质验收。

初次 Qwen 检查误用了另一 Python 的 MLX/旧 native 库和不存在的旧 SDK，失败日志保留；改用项目 `.venv`、当前 SDK 和本次新库后完整通过，未修改测试门槛。

日志与本地构建分别位于 `outputs/main-into-dev-20261015/` 和 `build/main-into-dev-20261015-{before,after}/`。

## main 新增优化的实际适用范围

| 改动 | 本机影响与边界 |
| --- | --- |
| Qwen 每请求 DiT cache：off/conservative/balanced/fast | 更完整的 DBCache 残差复用策略、失效/取消修复和实际 saved-block 回执；不是无损编译缓存。默认 off，需要允许近似；速度与图像变化必须单列。dev 已有诊断式 DBCache，不能称为全新算法。 |
| Qwen 默认高效编辑 encoder attention | FP32 vision D72→D80 零补并保持原缩放，语言 GQA/causal mask 优化；主要针对带参考图的长序列，不把 encoder 组件收益直接转述为整图去噪收益。 |
| Qwen 编辑 prefix snapshot、参考图尺寸与 LoRA 调度修复 | 利于重复编辑/启动流程；512-reference 是显式近似，普通 LoRA 与 Viggle 六步 student 分开；本轮 base 文生图不测这些编辑收益。 |
| Qwen M5 Pro 24 GiB layer streaming/紧凑 suffix/wired policy | 精确硬件门控；M4 Max/64 GiB 不启用，不能借用 M5 的历史加速或内存数字。 |
| 独立 Native Metal DiT/encoder、private runtime ANE、动态 INT8/SmoothQuant | 显式实验且请求域/硬件/来源受限，默认不替换已有 MLX executor；历史组件/单图结果不授本机默认资格。 |
| Z-Image LoRA workspace/可选 MPP、校准和 runtime 调度 | 主要面向 runtime LoRA/显式实验；不据源码变化推断 base GPU 或冻结 W8A8 一定更快。 |
| Z-Image catalog 默认 9→8 步 | 属于默认采样工作量变化；已有 benchmark 本来就是 8 步。本轮两版本都显式 8 步，不把少一步包装成 kernel 提速。 |
| fd reader 独立游标、ANE 资源退休/失败回收、App/API/ComfyUI | 主要是正确性、资源生命周期和使用体验收益，不自动等于热态生图加速。 |

相关既有证据：
[DiT cache](2026-09-30-qwen21-dit-cache.md)、
[M5 低内存路线](qwen21-m5pro-24gb-streaming-ane.md)、
[Native encoder](2026-10-08-qwen-native-encoder.md)、
[Native W1](2026-10-08-native-register-w1-base-e2e.md)。它们的硬件/任务各自独立，不能与本轮时间拼接成配对倍率。

main README 新强调的 Qwen 3.40× / Z-Image 3.15× 是既有对固定 Unsloth 后端的跨框架结果，不是 main 相比本次合并前 dev 又快了三倍。dev 已有的 GGUF 导入/读前取、fast encoder 默认和既有冻结分区优化也不重复计作这次合并的新收益。

## 本轮评测口径

`tools/validation/branch_merge_benchmark.py` 串行调用同一个 C ABI 观察器，对合并前后库分别测：

- 原 BF16 Qwen Image 2.1，512×512、40 步、seed 29；GPU 和原有 source-bound frozen W8A8（6144 ANE / 6144 GPU complement）。
- 原 BF16 Z-Image Turbo，512×512、8 步、seed 42、dynamic text；GPU 和原有 frozen W8A8（5120/5120）。
- 每个模型/路线按 ABBA 分支顺序，每版本 2 个独立进程，各 1 cold + 2 fresh warm，合计每路 4 fresh 样本；不以重复 prompt cache hit 代替 fresh 推理，不额外增加采样边界 eval。
- 计完整请求 wall（文本编码/去噪/VAE/PNG 导出）；constructor、cold、阶段时间和进程树物理峰值另记。OS 文件/driver cache 未清空，不称严格系统冷启动。
- 比较同路 before/after 的完整 RGBA；GPU/ANE 本身为近似，不与 GPU 声称逐像素相同或物理 ANE 驻留追踪资格。
- 独立进程树采样覆盖加载/cold/warm/退出，100 ms 周期、500 ms 最大间隙，完整性校验；不包含外部 Core ML 服务和 driver 的独立归因，不是整机硬预算。
- 新原始记录与历史 benchmark 分开，未重跑 ComfyUI/mflux/Unsloth，不能发布新的外部框架配对倍率。

复现（需保留本地两份构建和既有模型/分区）：

```sh
.venv/bin/python -B tools/validation/branch_merge_benchmark.py \
  --before build/main-into-dev-20261015-before/libturbocider.dylib \
  --after build/main-into-dev-20261015-after/libturbocider.dylib \
  --output outputs/new-main-dev-paired
```

## 默认/既有冻结路线：48 张完整图的配对结果

每个单元格为该版本/路线 4 个 fresh warm 的中位数；单位秒。耗时减少为负表示略慢。

| 模型/路线 | 旧 benchmark（10月12日，跨窗口） | 本轮 before | 本轮 after | 本轮耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| Qwen BF16 GPU，512²/40 | 43.309 | 43.317 | 43.292 | 0.058% |
| Qwen frozen W8A8，512²/40 | 31.810 | 31.458 | 31.478 | -0.066% |
| Z-Image BF16 GPU，512²/8 | 7.356 | 7.360 | 7.359 | 0.021% |
| Z-Image frozen W8A8，512²/8 | 5.955 | 5.924 | 5.820 | 1.746% |

**本机默认 base 路线没有明显新收益。** Qwen 两路和 Z GPU 均基本持平；Z frozen 本轮少约 0.103 秒（1.75%），但四个样本各自范围重叠，不能据此授予某个新 kernel 稳定提速或通用设备资格。旧 benchmark 的小幅差异只是跨窗口观察，main README 的外部框架倍率不转化为这次的分支收益。

全部 16 进程、48 PNG 完成；同模型同路线的 before/after 及正反序输出 **RGBA 完全相同**。两条混合路线均有原 checkpoint SHA 验证、实际完整工作和零失败：Z 每张 256 次，Qwen 每张 1248 次；没有把 GPU fallback 当混合。16 份进程树内存记录全部完整，swap-out 合计 0；不由此宣称整机无压缩/无内存压力。

混合并未改善首次延迟：本轮 Qwen frozen cold 为 before 56.79–56.92 / after 56.14–57.06 秒，Z frozen 为 before 26.45–27.51 / after 26.66–27.55 秒。cold 包含资源加载/准备但不含独立 constructor；未清 OS/driver cache，不能把先后进程的差值单独归因于优化。热态优势适用于连续请求，不自动适用于一次性生成。

历史时间来源：
[Qwen 同源 BF16 框架组](../design/validation/qwen21-mflux022-bf16-fp16-current-20261012.json)、
[Z 同源 Unsloth/mflux 组](../design/validation/z-unsloth-mflux-current-20261012.json)。
本轮全部原始样本、实际请求/结果、PNG digest、构建 manifest 与独立内存校验均在 `outputs/main-into-dev-20261015/paired/summary.json`，逐进程目录另保留 PNG 和 memory JSONL。

after 的全进程树物理峰值（加载+cold+warm+退出）为 Qwen GPU 27.66–28.60 GiB、frozen 33.56–33.61 GiB；Z GPU 29.20–29.50 GiB、frozen 29.90–29.93 GiB。样本/缓存状态会影响峰值；没有从这些数值推导新的内存档位或整机内存收益。

## 可选 Qwen DiT cache：有损加速确有收益

只使用 after，GPU、resident、512²/40 步/seed 29，无参考图/LoRA/ANE。两个提示词为狐狸和蓝色茶壶；每个提示词依次运行 off warmup（排除）、off、conservative、balanced、fast、off。每预设每提示词只一个样本，前后 off 控制取均值；不是另一轮 ABBA 或广泛画质认证。

这里所有计入比较的请求均命中**文本 conditioning 缓存**，但每次都实际运行 40 个采样步、VAE 和 PNG 导出；不是整图缓存。off 两次输出 RGBA 相同。不能把文本缓存收益计作 DiT 收益，也不能与 frozen ANE 倍率相乘。

| DiT cache | 两提示词完整请求中位/s | 相对同组 off 耗时减少 | RGB SSIM 范围 | 复用步数 |
| --- | ---: | ---: | ---: | ---: |
| off | 42.629（四个控制样本） | — | 1 | 0 |
| conservative | 31.119 | 26.97–27.04%（约1.37×） | 0.99218–0.99296 | 15 |
| balanced | 26.236 | 38.36–38.56%（约1.62–1.63×） | 0.98038–0.98129 | 21 |
| fast | 23.111 | 45.78–45.80%（约1.84×） | 0.95456–0.96570 | 25 |

每步仍计算 front 8 blocks，前 8 步/末步及连续复用上限受原有保护；实际 saved-middle-block 回执分别为 360/504/600。两组 off 控制为 42.627/42.631 和 42.627/42.638 秒，窗口内漂移很小。

六张缓存候选**都不是无损**：RGB MAE 为 conservative 0.787–1.243/255、balanced 1.631–2.327/255、fast 2.742–3.655/255；alpha 最大绝对变化均 2/255。SSIM 使用既有报告工具的 11×11、sigma1.5 Gaussian valid-interior RGB 定义。人工查看两提示词全部八张 off/预设图，主体、构图和茶壶把手/壶嘴接近，但毛发、釉面、高光和局部边缘有变化；两个简单提示词不能保证任意文字、编辑或 LoRA 的语义保持。

因此本机有实际价值的新增**可选**路线是 DiT cache：若接受近似可先试 conservative；更激进档位自行确认画质。没有修改 App/请求默认 off，也未替换已安装 App。M5 低内存路线、Native encoder/DiT、参考图和 LoRA 的完整性能均未借本轮取得新资格。

缓存原始记录在 `outputs/main-into-dev-20261015/cache/`，独立内存记录 `cache-memory.jsonl` / `cache-memory-report.json` 覆盖整个 12 请求窗口，校验完整、无 swap-out；不作各预设独立内存排名。可复现请求序列：

```sh
.venv/bin/python -B tools/validation/branch_merge_benchmark.py \
  --child cache --model qwen \
  --library /path/to/after/libturbocider.dylib \
  --output outputs/new-qwen-cache-study
```

本轮全部 60 PNG 测量结束后停止推理，不追加网格或外部框架复测。精确样本、图像指标、内存校验、库/原始摘要绑定及历史来源见[机器回执](../design/validation/main-dev-merge-performance-20261015.json)；历史 benchmark 表未改写。
