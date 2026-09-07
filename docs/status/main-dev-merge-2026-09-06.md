# main 合并到 dev 的兼容与验收记录

日期：2026-09-06；2026-09-07 LoRA-bound/分区复核

本文是合并历史记录；当前实现和性能结论见 [2026-09-07 当前状态](current-status-2026-09-07.md)。

## 结论

`main` 已通过 merge commit `2cee456` 合并到当前 `dev`，`git merge-base --is-ancestor main dev` 返回成功。合并保留了 `main` 的 C++20 core/runtime、MLX backend 和统一 `ModelModule` 架构，同时接入 `dev` 的 FLUX 4B/9B、MiniMax H3、LTX、FastMetal 和 Z-Image 六个模型执行器。

当前 native dylib、CLI、Swift App、集成测试和发行包均可编译链接。Apple M4 Max 实机能够识别 Metal/MLX，并已通过 Z-Image 的 App embedded-session 真实生成。该版本适合作为多模型 native integration 开发验收版本；尚未完成的性能与 parity 项仍保持 fail closed，不应标为全部模型生产就绪。

## 合并处理

- 以 `main` 的 `native/core`、`native/runtime`、C++ MLX backend 和 C ABI 生命周期为主干；Apple framework bridge 保留在 Objective-C++ platform 层。
- 将 H3、LTX 的 C/Metal/ANE runtime 快照编入同一个 `libturbocider.dylib`；运行时不从兄弟源码仓库加载实现。
- 将 FastMetal 作为受控外部 worker 模块注册，保留 profile、provenance 和 GPU+ANE 完整性门禁。
- 将 FLUX 4B/9B 与 Z-Image 接入统一 registry、plan、C ABI、Swift binding、CLI 和 App 模型目录。
- 修复 Z-Image Qwen3 输出维度：text encoder 的 `[1,T,2560]` 输出在截取有效 token 前去掉 batch 维，向 DiT 传递 `[T,2560]`。
- 将 Z-Image 采样日程对齐 ComfyUI 的固定 `shift=3.0` discrete-flow simple scheduler，包括 1000-entry sigma table 的离散取样；Euler 状态保持 FP32、仅在 DiT/VAE 入口转 BF16。增加可选共享初始噪声和逐步 latent dump，供同噪声 oracle 定位数值差异，不改变正常请求的默认随机噪声路径。
- 支持 ComfyUI split-files 与 Tongyi diffusers 组件目录；diffusers 多 shard、独立 Q/K/V、`all_x_embedder`/`all_final_layer` 与 VAE 命名在 C++ loader 内归一化，不依赖外部转换仓库。
- 官方 Z-Image distill patch LoRA 以独立文件运行时内存融合，238 个 projection 全部应用；同噪声对 ComfyUI 的最终 latent cosine 0.998929、PNG correlation 0.999156。
- 修复 FLUX 4B 的 M4 Max 6144-channel ANE 前缀分区：manifest 明确记录 `[0,6144)`，GPU 编译图并行补算 attention 与 `[6144,9216)` MLP 后缀；此前只计算前缀却未补尾部的错误不再存在。
- 离线导出器、Core ML 资源服务和 App 已共同支持 `ane_mlp_width`；新增 `apple-m4-max-64gb.example.json`。M4 Max 自动路径只接受 6144 前缀，M4 Pro 自动路径只接受已验证的完整 MLP，其他硬件继续 fail closed。
- FLUX 与 Z-Image 独立 LoRA 继续在加载时/运行时内存融合；base Core ML artifact 不含 LoRA delta，因此带 LoRA 的 App 请求安全切到 GPU，或仅接受同一 adapter 绑定的显式 LoRA-bound manifest；无 LoRA 的 Z-Image a4096 base 在 exact M4 Max 64 GB 上可自动 GPU+ANE，native 直接请求仍保持严格校验。
- FLUX/Z-Image 的 LoRA-bound Core ML 导出使用由适配器路径、大小、SHA-256、角色和强度派生的独立存储目录；资源服务现在先确定该目录，再把同一路径传给导出器、进度监控和 manifest 检查，避免合并后导出写入旧目录而 App 等待新目录。请求层现已允许匹配的 LoRA-bound artifact 进入显式 GPU+ANE；base artifact 仍回退 GPU。
- Z-Image 新增 32 分区 Core ML 导出、`output_scale` ABI 和 Core ML 资源编译；后续 GPU fusion 将 warm 从 45.22 s 降到 36.3685 s，并快于 stock ComfyUI 40.110 s。最终构建的 M4 Max base a4096 warm 中位数为 30.0014 s，相对优化 GPU 为 1.212×、warm 范围 29.9956–30.0072 s，按 1.2×端到端门槛进入 exact-device 自动策略；LoRA-bound 路线仍为显式 opt-in。
- 修复 App smoke 的多模型兼容：不再硬编码 256×256/4-step/PNG，而从模型 descriptor 读取操作、尺寸、步数、帧数、帧率、音频、驻留和输出媒体类型。

## 模型与 App 兼容矩阵

| 模型 | native 注册 | App 默认参数 | 独立 LoRA 请求 | 当前执行边界 |
|---|---|---|---|---|
| FLUX.2 Klein 4B | 已通过 | 已通过 | load-time in-memory bake | GPU；已验证 profile 时可 GPU+ANE |
| FLUX.2 Klein 9B | 已通过 | 已通过 | load-time in-memory bake | GPU-only |
| MiniMax H3 Turbo | 已通过 | 已通过 | 独立文件请求；首次生成 runtime cache | native Metal/MPS，可按 manifest 启用 ANE |
| LTX 2.5 Distilled | 已通过 | 已通过 | 独立文件请求；首次生成 runtime cache | public video-only；GPU+ANE 候选继续门禁 |
| FastMetal 1.3B QAD | 已通过 | 已通过 | 当前要求 provenance-verified premerged checkpoint | 显式 Python/FastVideo worker profile |
| Z-Image Turbo | 已通过 | 已通过 | 独立文件，运行时内存 delta | M4 Max 64 GB base a4096 可自动 GPU+ANE；LoRA-bound 分区显式 opt-in；其他机器 GPU |

## 验证结果

使用仓库现有 MLX 0.32.2 环境显式设置 `MLX_ROOT`：

```text
make test                         46 项 contract + 9 项 repository/boundary 检查通过；系统 Python 缺 NumPy 时仅跳过 1 项数值测试
Python/bin/python LoRA/分片测试    4 项 Core ML LoRA + 5 项 Z-Image shard 测试全部通过
make test-app                     通过；无 pasteboard service 时仅跳过系统剪贴板检查
make build                        native、CLI、Swift App、integration runners 编译链接通过
make package                      App/CLI 打包与 ad-hoc codesign 验证通过
turbocider models                 六个 executable model descriptors 正常返回
turbocider doctor（实机）          Apple M4 Max、64 GB、Metal GPU、MLX 0.32.2
全部 examples/requests plan        executable=true
git diff --check                  通过
```

FLUX 4B M4 Max 6144 前缀最终实机复核：

```text
GPU warm 中位数                       2.2663 s
GPU+ANE 6144 前缀 warm 中位数          1.6279 s
稳定复测加速比                        1.392×
对 GPU PNG cosine                     0.999840
对 GPU PNG correlation                0.999262
对 GPU PNG MAE                        1.285 / 255
```

这是 M4 Max 512²/4-step 的稳定 warm 复测；仍不替代更广尺寸/机器和交错 AB/BA 的 p50/p95 正式矩阵。重新构建后的自动选择测试通过 session reuse、自动/显式 PNG byte-identical、超桶/缺失/损坏 manifest 回退、近似 opt-in 和显式严格失败等门禁。

Z-Image Swift App embedded-session 实机 smoke：

```text
尺寸/步数          1024×1024，9 steps
request wall       47.445 s
denoise            44.078 s
text encode         1.495 s
VAE decode          1.025 s
MLX peak           25.63 GB
输出                可解码、非空 PNG
```

本轮合并后重新验证：`make build`、`make package`、`make test`、`make test-app` 均通过；全部请求样例可生成 `executable=true` 的计划；实机 `doctor/self-test` 识别 Apple M4 Max、64 GB、Metal 和 MLX 0.32.2。Z-Image embedded-session 端到端 request wall 为 45.632 s，在临时验证目录中生成可解码 PNG。

2026-09-07 复核：`turbocider-app-smoke models/Comfy-Org-z_image_turbo ... z-image-turbo` 真实运行成功，在临时验证目录中生成可解码 PNG，request wall `47.840 s`、denoise `43.917 s`，App job/persistence 校验通过；本次为纯 GPU 默认路径，未把 GPU+ANE 候选误报成默认加速。

Z-Image 7680-channel GPU+ANE 候选的常驻 warm request wall 约 47.52 s，旧 GPU warm 基线约 45.17 s；4096 LoRA-bound 旧矩阵约 1.21×。本轮 GPU fusion 后，base GPU warm 中位数为 36.3685 s，stock ComfyUI 为 40.110 s；最终构建的自动 4096 base 请求为 37.5838/29.9956/30.0072 s，warm 中位数 30.0014 s、相对优化 GPU 为 1.212×。App/API 因此仅对 exact M4 Max 64 GB 的 base a4096 自动启用；带 LoRA 时仍必须显式选择并绑定相同 LoRA identity。

4096-channel Core ML 路线使用 session-wide shared output backing，避免每个 block 各自保留同形状 FP16 buffer。最终构建的 Core ML 输出拷贝计数为 0，两次 warm 请求范围为 29.9956–30.0072 s，同 seed GPU↔ANE PNG correlation 0.999231；该 geometry 现在同时作为 exact-device base 自动路线和 LoRA-bound 显式候选。

## 仍然开放的风险

- LTX GPU+ANE 当前速度和 latent parity 尚未过门禁，不能作为默认路径，也尚未证明包含完整生命周期时稳定快于 mac-ltx。
- H3/LTX LoRA cache miss 现在调用 TurboCider 自带的 Python merge 工具并产生可清理的 merged artifact，不再从兄弟仓库发现脚本；仍未达到纯内存逐层融合目标。
- FastMetal 仍依赖显式外部 worker/profile，独立 LoRA runtime bake 未完成。
- FLUX 9B 尚缺标准尺寸、多轮 warm/resident parity 与性能矩阵。
- FLUX 4B M4 Max 6144 前缀最终 warm 中位数为 1.6279 s，相对 GPU 为 1.392×，并通过输出一致性复核；仍需更广尺寸/机器的交错 AB/BA 矩阵。
- Z-Image scheduler、共享噪声最终 latent/PNG 和官方独立 LoRA 图片 parity 已完成；优化 GPU 已快于 stock ComfyUI，base a4096 通过相对优化 GPU 的 1.2×重复 warm 门槛，仍缺逐 step oracle、LoRA-bound 优化后多轮 warm 和多机器矩阵。
- 当前包为本地 ad-hoc 签名，不是 Developer ID 公证发行包。

## 构建说明

正式受管环境由 `make setup` 创建仓库 `.venv`。本机尚未创建该环境，因此本轮使用已有且被 `.gitignore` 排除的 TurboCider `Python/` 环境，通过显式 `MLX_ROOT` 完成构建；这不改变源码或发行包的依赖边界。
