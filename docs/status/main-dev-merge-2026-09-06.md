# main 合并到 dev 的兼容与验收记录

日期：2026-09-06

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
- FLUX 与 Z-Image 独立 LoRA 继续在加载时/运行时内存融合；因为 base Core ML artifact 不含 LoRA delta，App 的显式或自动 GPU+ANE 请求安全切换到 GPU，native 直接请求仍保持严格校验。
- Z-Image 新增 32 分区 Core ML 导出、`output_scale` ABI 和 Core ML 资源编译；M4 Max GPU+ANE 目前仅为显式 opt-in 候选，自动模式保持 GPU，仍需逐 step parity、warm 矩阵和正式加速门禁，不能作为默认性能承诺。
- 修复 App smoke 的多模型兼容：不再硬编码 256×256/4-step/PNG，而从模型 descriptor 读取操作、尺寸、步数、帧数、帧率、音频、驻留和输出媒体类型。

## 模型与 App 兼容矩阵

| 模型 | native 注册 | App 默认参数 | 独立 LoRA 请求 | 当前执行边界 |
|---|---|---|---|---|
| FLUX.2 Klein 4B | 已通过 | 已通过 | load-time in-memory bake | GPU；已验证 profile 时可 GPU+ANE |
| FLUX.2 Klein 9B | 已通过 | 已通过 | load-time in-memory bake | GPU-only |
| MiniMax H3 Turbo | 已通过 | 已通过 | 独立文件请求；首次生成 runtime cache | native Metal/MPS，可按 manifest 启用 ANE |
| LTX 2.5 Distilled | 已通过 | 已通过 | 独立文件请求；首次生成 runtime cache | public video-only；GPU+ANE 候选继续门禁 |
| FastMetal 1.3B QAD | 已通过 | 已通过 | 当前要求 provenance-verified premerged checkpoint | 显式 Python/FastVideo worker profile |
| Z-Image Turbo | 已通过 | 已通过 | 独立文件，运行时内存 delta | GPU 默认；32 分区 GPU+ANE 候选已接入但继续 fail closed/opt-in |

## 验证结果

使用仓库现有 MLX 0.32.2 环境显式设置 `MLX_ROOT`：

```text
make test                         42 项 contract + 9 项 repository/boundary 检查通过
make test-app                     通过；无 pasteboard service 时仅跳过系统剪贴板检查
make build                        native、CLI、Swift App、integration runners 编译链接通过
make package                      App/CLI 打包与 ad-hoc codesign 验证通过
turbocider models                 六个 executable model descriptors 正常返回
turbocider doctor（实机）          Apple M4 Max、64 GB、Metal GPU、MLX 0.32.2
全部 examples/requests plan        executable=true
git diff --check                  通过
```

FLUX 4B M4 Max 6144 前缀实机复核：

```text
GPU warm 基线                         2.3479 s
GPU+ANE 6144 前缀 warm                1.6659 s
观察加速比                            1.409×
对 GPU latent cosine                 0.999660
对 GPU PNG correlation               0.998391
对 GPU PNG MAE                       1.484 / 255
```

这是本机候选路径的一组 warm 观察，不替代交错 AB/BA 的 p50/p95 正式矩阵。重新构建后的自动选择测试通过 session reuse、自动/显式 PNG byte-identical、超桶/缺失/损坏 manifest 回退、近似 opt-in 和显式严格失败等门禁。

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

本轮合并后重新验证：`make build`、`make package`、`make test`、`make test-app` 均通过；实机 `doctor/self-test` 识别 Apple M4 Max、64 GB、Metal 和 MLX 0.32.2。Z-Image embedded-session 端到端 request wall 为 45.632 s，生成 `/private/tmp/tc-app-z-smoke/app-generated.png`。

Z-Image 7680-channel GPU+ANE 候选的常驻 warm request wall 约 47.52 s，GPU warm 基线约 45.17 s；虽然输出 correlation 0.998571、cosine 0.999819，但当前没有加速。因此 App/API 自动策略已收紧为 GPU，只有显式 `gpu_ane + allow_approximation + manifest` 才会进入实验路径。

后续 4096-channel Core ML 复核使用 session-wide shared output backing，避免每个 block 各自保留同形状 FP16 buffer。真实双请求均成功，Core ML 输出拷贝计数为 0，PNG byte-identical；首次 request wall 约 44.69 s，第二次 warm/cache-hit 约 37.51 s。该优化改善了 backing/VAE 内存压力，但 GPU+ANE 仍未达到 1.3×，所以没有放开自动选择。

## 仍然开放的风险

- LTX GPU+ANE 当前速度和 latent parity 尚未过门禁，不能作为默认路径，也尚未证明包含完整生命周期时稳定快于 mac-ltx。
- H3/LTX LoRA cache miss 现在调用 TurboCider 自带的 Python merge 工具并产生可清理的 merged artifact，不再从兄弟仓库发现脚本；仍未达到纯内存逐层融合目标。
- FastMetal 仍依赖显式外部 worker/profile，独立 LoRA runtime bake 未完成。
- FLUX 9B 尚缺标准尺寸、多轮 warm/resident parity 与性能矩阵。
- FLUX 4B M4 Max 6144 前缀已达到单组 warm 1.409× 观察值并通过输出一致性复核；仍需多轮交错 AB/BA 后才可写成稳定性能承诺。
- Z-Image scheduler、共享噪声最终 latent/PNG 和官方独立 LoRA 图片 parity 已完成；32 分区 GPU+ANE 导出/ABI 已接入，但逐 step oracle、多轮 warm 数据和正式 GPU+ANE 性能门禁仍未完成。
- 当前包为本地 ad-hoc 签名，不是 Developer ID 公证发行包。

## 构建说明

正式受管环境由 `make setup` 创建仓库 `.venv`。本机尚未创建该环境，因此本轮使用已有且被 `.gitignore` 排除的 TurboCider `Python/` 环境，通过显式 `MLX_ROOT` 完成构建；这不改变源码或发行包的依赖边界。
