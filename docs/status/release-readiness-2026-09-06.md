# TurboCider 版本提交准备度

更新时间：2026-09-07（LoRA-bound GPU+ANE 复核）

当前性能、双格式 Z-Image 和 LoRA 证据以 [2026-09-07 当前状态](current-status-2026-09-07.md) 为准。

## 结论

当前工作树可以整理并提交一个**开发验收版本**，但不应把这次提交命名或描述为“全部模型生产版”。代码、测试、请求样例、配置样例和正式文档已经形成一个可审阅的纵切；LTX ANE、LTX I2V/音频、FLUX 9B 完整矩阵、FastMetal 独立 LoRA、H3/LTX 纯内存 LoRA，以及 Z-Image 的逐 step oracle和 LoRA-bound 多轮性能仍是明确的后续工作。Z-Image base/官方独立 LoRA 的同噪声最终 latent/PNG parity 已完成；优化 GPU 已快于 stock ComfyUI，M4 Max base a4096 也已通过相对优化 GPU 的 1.2×重复 warm 门槛。

建议提交主题：

```text
feat(native): integrate H3 FastMetal and LTX video runtime
```

当前集合包含两份有明确 provenance 的 native runtime 快照，若希望历史更易审阅，建议拆成三个 commit：

1. `chore(native): vendor H3 and LTX runtime snapshots`
2. `feat(native): integrate H3 FastMetal and LTX runtimes`
3. `docs(status): record model readiness and release boundaries`

`main` 合并提交已经存在；本轮新增的 M4 Max FLUX 6144 前缀收口适合单独提交，保持性能候选与未完成项的边界可审阅。

## 可以进入版本的内容

- `native/core`、`native/backends`、`native/media` 的实现和 ABI/生命周期修改；
- `native/models/fastmetal_module.mm`、`h3_session.mm`、`ltx_session.mm`；
- `native/models/h3_runtime`、`native/models/ltx_runtime` 中被构建脚本显式列出的源文件；
- `native/models/z_image`、`z_image_module.cpp` 及其 ComfyUI/独立 LoRA 契约；
- CLI、service、Swift/C binding 的契约修改；
- `tools/native` 的 build、benchmark、LoRA preparation 和 worker 工具；
- `tests/native`、`tests/test_fastmetal_worker.py` 以及请求/配置样例；
- `docs/status`、更新后的 `README`、`USAGE` 和设计索引；
- `native/THIRD_PARTY_NOTICES.md` 中声明的第三方来源和许可证。

## 不应进入版本的内容

`.gitignore` 已忽略下列本机生成或大文件目录：

- `.build/`、`build/`、`dist/`；
- `Python/`、`models/`、`outputs/`、`state/`；
- Python cache、编译对象、动态库、Core ML compiled artifacts；
- safetensors、ckpt、onnx、gguf 等权重。

本轮的 `notes/` 是本机研究记录，包含绝对路径和过程性细节；正式结论已收敛到 `docs/status/`，不应通过 `git add .` 原样加入版本。若需要保存研究记录，应先做路径归一化，再单独放入 `docs/design/validation/` 的摘要格式。

## 提交前必须确认的事项

### 代码边界

- `tools/native/build.sh` 的源文件清单只包含正式 `native/` runtime，不包含 `experimental/`、模型权重或本机路径。
- `tools/native/package.sh` 必须携带 H3/LTX Metal shader、LTX clean-exec finalizer/Video VAE helper、LoRA cache helper 和 FastMetal worker；helper 的 MLX rpath 改为 `@loader_path`。
- build/package 不再硬编码 MLX 版本或 macOS 15.0：构建从当前 `libmlx.dylib` 推导 deployment target，并将相同最低版本用于 C/C++/Objective-C++、Swift 和 App metadata；package 自动发现当前 MLX 许可证。
- AVFoundation 媒体探测已迁移到 `loadTracksWithMediaType:completionHandler:`，完整 native 构建无弃用或 deployment-target 警告。
- package 会先清理明确的生成目录 `dist/TurboCider.app` 和 `dist/cli`，避免历史 symlink 或旧签名污染新包；不会触碰源码、模型或 `outputs/`。
- H3/LTX 共享源的许可证和 provenance 已在 `native/THIRD_PARTY_NOTICES.md` 与 runtime README 中说明；FastVideo/TAEHV 的 Apache/第三方 NOTICE 仍应在发布包审查时补齐对应上游文本。
- 外部 `h3.c`、`ltx-mac` 工作树当前有用户修改；提交 TurboCider 时只提交本仓库，不要把兄弟仓库的修改混入。
- `git diff --check` 必须无输出。

### 功能门禁

- LTX public executor 只开放 `video.generate` + `audio=false`；I2V、音频和默认 GPU+ANE 继续 fail closed。
- FastMetal LoRA 继续要求 provenance-verified premerged manifest；不能把 `runtime_lora=false` 改成 true 来掩盖缺口。
- H3/LTX LoRA 的 runtime cache 是可清理临时 artifact，merge 实现已随 TurboCider 分发但仍依赖 Python 运行环境；不应写成已经完全不产生 merged 权重或已经纯内存融合。
- FLUX 9B 的 5.33 秒 smoke 不得写成正式 parity/性能验收。
- Z-Image 已有 Apple M4 Max 真实 1024×1024 base/官方独立 LoRA 出图和 App smoke；共享初始噪声下最终 latent/PNG 已通过 ComfyUI oracle。优化 GPU warm 中位数 36.3685 s，stock ComfyUI 为 40.110 s；最终构建的自动 base a4096 warm 中位数为 30.0014 s，相对优化 GPU 为 1.212×，可在 exact M4 Max 64 GB 自动启用；LoRA-bound 仍显式/fail closed。
- FLUX 4B M4 Max 自动路径只接受 `[0,6144)` ANE 前缀，GPU 必须补算 `[6144,9216)` 后缀；M4 Pro 继续只接受完整 MLP。最终 512²/4-step 复测为 GPU 2.2663 s、GPU+ANE 1.6279 s（1.392×）；仍需更广尺寸/机器和交错 AB/BA p50/p95 矩阵。

### 测试门禁

```sh
MLX_ROOT=/path/to/mlx tools/native/build.sh
MLX_ROOT=/path/to/mlx tools/native/package.sh
Python/bin/python -m pytest -q tests
make test
git diff --check
```

真实 GPU 环境还应执行：

```sh
build/native/turbocider models
build/native/turbocider doctor
Python/bin/python tools/native/benchmark_h3.py ...
```

LTX 需要按 fresh、conditioning-cache hit、loaded-model hot path 和 resident + clean VAE 四种口径分别记录；不能只保留一个“总秒数”。

## 当前版本的风险说明

1. **LTX 生命周期风险**：resident session 会令 Video VAE 变慢，model_load 为零不等于端到端更快。
2. **LTX ANE 数值风险**：当前 video latent 偏差约 cosine 0.945，不能作为默认精确路径。
3. **LoRA 磁盘风险**：H3/LTX 首次使用会生成临时 merged artifact，虽然可 prune，但还不是纯内存 merge。
4. **环境复现风险**：统一内存调度和 Core ML on-device compile 会造成明显 wall 波动；当前只有 M4 Max 64 GB 的实机证据。
5. **许可证发布风险**：FastMetal 依赖 FastVideo/TAEHV 的第三方许可证需要在最终发行包中逐项核对。
6. **Z-Image 验收风险**：base/官方独立 LoRA 的共享噪声最终输出 parity 已有实机证据；优化 GPU 已通过 stock ComfyUI 对照，base a4096 已形成重复 warm 统计并通过 1.2×门槛；逐 step oracle、LoRA-bound 优化后重复 warm 和多机器速度矩阵仍未完成。

## 推荐的下一版退出条件

- LTX 完成同入口、同输出边界的 AB/BA p50/p95；包含 fresh、cache-hit 和 warm resident；
- LTX GPU+ANE 通过 stage-boundary parity 和端到端速度门禁；
- H3/LTX loader 改为逐层内存 LoRA merge/requantize；
- FastMetal 独立 LoRA 完成真实 merged checkpoint 的 latent/媒体/性能验证；
- FLUX 9B 完成 128/512 标准尺寸 parity 与 warm/resident 矩阵；
- 补齐 FastVideo/TAEHV 许可证 NOTICE 后再制作 Developer ID 包。

本轮新增的退出条件：

- FLUX/Z-Image 带 LoRA 的 GPU+ANE 必须使用同一 LoRA-bound manifest；App 自动发现和 native SHA-256 校验均已覆盖；
- Z-Image 4096-channel base 候选已在最终构建形成重复 warm 数据并以 1.212×解除 exact M4 Max 64 GB 自动门禁；LoRA-bound 路线继续显式；
- H3/LTX 仍需后续逐层 native in-memory LoRA merge，当前 vendored Python/runtime cache 不应宣称为纯内存实现。
