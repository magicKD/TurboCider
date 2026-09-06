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
- 修复 App smoke 的多模型兼容：不再硬编码 256×256/4-step/PNG，而从模型 descriptor 读取操作、尺寸、步数、帧数、帧率、音频、驻留和输出媒体类型。

## 模型与 App 兼容矩阵

| 模型 | native 注册 | App 默认参数 | 独立 LoRA 请求 | 当前执行边界 |
|---|---|---|---|---|
| FLUX.2 Klein 4B | 已通过 | 已通过 | load-time in-memory bake | GPU；已验证 profile 时可 GPU+ANE |
| FLUX.2 Klein 9B | 已通过 | 已通过 | load-time in-memory bake | GPU-only |
| MiniMax H3 Turbo | 已通过 | 已通过 | 独立文件请求；首次生成 runtime cache | native Metal/MPS，可按 manifest 启用 ANE |
| LTX 2.5 Distilled | 已通过 | 已通过 | 独立文件请求；首次生成 runtime cache | public video-only；GPU+ANE 候选继续门禁 |
| FastMetal 1.3B QAD | 已通过 | 已通过 | 当前要求 provenance-verified premerged checkpoint | 显式 Python/FastVideo worker profile |
| Z-Image Turbo | 已通过 | 已通过 | 独立文件，运行时内存 delta | GPU-only；GPU+ANE 继续门禁 |

## 验证结果

使用仓库现有 MLX 0.32.2 环境显式设置 `MLX_ROOT`：

```text
make test                         47 项通过
make test-app                     通过；无 pasteboard service 时仅跳过系统剪贴板检查
make build                        native、CLI、Swift App、integration runners 编译链接通过
make package                      App/CLI 打包与 ad-hoc codesign 验证通过
turbocider models                 六个 executable model descriptors 正常返回
turbocider doctor（实机）          Apple M4 Max、64 GB、Metal GPU、MLX 0.32.2
全部 examples/requests plan        executable=true
git diff --check                  通过
```

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

## 仍然开放的风险

- LTX GPU+ANE 当前速度和 latent parity 尚未过门禁，不能作为默认路径，也尚未证明包含完整生命周期时稳定快于 mac-ltx。
- H3/LTX LoRA cache miss 仍会调用 Python merge 工具并产生可清理的 merged artifact，未达到纯内存逐层融合目标。
- FastMetal 仍依赖显式外部 worker/profile，独立 LoRA runtime bake 未完成。
- FLUX 9B 尚缺标准尺寸、多轮 warm/resident parity 与性能矩阵。
- Z-Image 尚缺 ComfyUI oracle/scheduler parity、真实独立 LoRA 图片 parity、多轮 warm 数据和 GPU+ANE 分区。
- 当前包为本地 ad-hoc 签名，不是 Developer ID 公证发行包。

## 构建说明

正式受管环境由 `make setup` 创建仓库 `.venv`。本机尚未创建该环境，因此本轮使用已有且被 `.gitignore` 排除的 TurboCider `Python/` 环境，通过显式 `MLX_ROOT` 完成构建；这不改变源码或发行包的依赖边界。
