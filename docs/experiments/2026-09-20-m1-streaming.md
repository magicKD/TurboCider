# M1 Pro streaming 实施与实验记录

状态：进行中，尚无本轮真实模型性能结论。本文保留用户要求的 Flux.2 Klein **4B** 和 Z-Image Turbo 范围，不以已有 Flux 9B / M5 结果替代。

## 实机基线

- Apple M1 Pro，8 核 CPU（6P+2E），16 GiB 统一内存。
- macOS 26.4.1，build 25E253。
- 起始代码：`feat/stream@85443b7`，工作树干净；未发现运行中的模型实验。
- 已 fetch `origin/dev@61c0849`。整合 INT8 streaming、variant selection、历史任务批量删除、图片选择；四处冲突涉及 App、Z-Image runtime 和权重流接口/实现。
- 项目首次配置 Python 3.11.15 及仓库锁定的 MLX/Core ML 依赖。不能沿用旧机器的编译产物和认证。

## 模型获取

下载目录：`/Users/chencanhui/models/TurboCider`。下载完成、内容校验和完整推理是不同状态，当前仅已启动下载。

| 模型 | 来源 | 固定 revision | 文件范围 |
|---|---|---|---|
| Flux.2 Klein 4B | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | Diffusers transformer、text encoder、tokenizer、VAE、scheduler/config |
| Z-Image Turbo | `Comfy-Org/z_image_turbo` | `08d04455279082882deaabc8d0d09fc914c071e1` | Comfy BF16 transformer、Qwen3 4B text encoder、VAE |

Z-Image tokenizer 单独来自 `Tongyi-MAI/Z-Image-Turbo`，下载时将其具体 revision 写入本地 provenance。每个下载先写 `.partial`，curl 成功后 rename；后续仍须检查大小、内容 digest 和模型结构。

## dev 整合要点

- legacy INT8 支持可变预取槽；exact BF16 仍只接受编译计划规定的 K1/K2，并按真实槽数分配。
- exact constructor 显式拒绝 ConvRot，避免 metadata 扩展后把未转换 INT8 当作 BF16 exact 权重加载。
- BF16 保留批量 read 路径和 worker cancellation；INT8 转换 read 同样传递取消信号。
- legacy 预取策略进入缓存配置键，exact 请求不调用 legacy 自动预取规划。
- App 保留 public streaming 控件，同时合入 dev 的设备/精度驻留限制和批量历史操作。
- 扩展真实 Metal tiny-fixture 回归，覆盖 K1/K2 的预算、每槽数据、分配数及 worker cancellation。运行结果待后续记录。

## 验收清单（尚未完成）

1. 合并后的 native / Swift App 构建、host / Metal / App 回归。
2. docs/design/streaming 56 R1–R6 的来源身份、options、全组件 lease、失败清理、App 事务和运行容器缺口；按真实代码逐项修复，不靠空 catalog 宣称完成。
3. Flux 4B 和 Z-Image 完整模型下载、校验、GPU baseline 与 streaming 真实出图。
4. 本机 Core ML artifact 导出与验证，GPU/ANE 分区探索；报告 cold setup、denoise、完整请求 wall、质量、process-tree memory 和 swap。
5. 相同 partition/reference 的迁移正确性，与纯 GPU 的性能收益分别测量；M5 专属优化不能直接标为 M1 已支持。
6. 非空 catalog 的 App→worker→verified result→有效新产物、取消和重试；完整发布证据和生产 record。
7. 所有实现完成后的默认路径回归与最终性能验证。当前不能宣称功能完备、发布可用或 ANE 有加速收益。

## 本轮运行句柄（临时，仅供继续工作时重新核查）

- 依赖安装：exec session `97499`，日志 `/tmp/tc-setup.log`。
- host 测试：exec session `2385`，日志 `/tmp/tc-host-tests.log`。
- 模型下载：exec session `70822`，脚本 `/tmp/tc-download-models.py`，日志 `/tmp/tc-model-download.log`。
- 完整构建：exec session `44723`，等待安装进程退出后执行，日志 `/tmp/tc-build.log`。

句柄/日志不是完成证明；下一轮先 poll 存活进程，不能仅凭日志静止启动重复下载或构建。

## 2026-09-20 首轮检查结果

- 合并提交 `7b886ab`；`git merge-base --is-ancestor origin/dev HEAD` 成功，包含本轮抓取的 `61c0849`。
- `make test-streaming-host PYTHON=python3`：exit 0。包含 layout/config 3,535 组合、descriptor、executor 14 组合、C bridge failure、actual receipt、source lease、public preset runtime/result、LTX metadata/model/snapshot、H3/Z-Image/Flux descriptor。这是合成 host 回归，不是模型出图或性能实验。
- repository `test_layout.py`、`test_independence.py`、`test_cpp_boundaries.py`：共 13 项通过。
- `git diff --cached --check`：通过。
- Metal fixture、完整 native/Swift 构建及 App 运行尚未完成。依赖安装、模型下载、等待依赖的构建句柄均已重新 poll 确认存活；没有重启重复作业。

## 第二轮：本机 native / Metal 验证与 App 结果绑定

依赖安装 exit 0，两个受管理 Python 环境的 `pip check` 通过。native dylib 与 CLI 编译/link 成功；完整构建仍在生成 Swift App / integration binaries（session `44723`），不能将 native 成功写成 App 验收完成。

新增 App 修复：非 LTX public 请求现在用 `resolveStreaming` 返回的 `exact_selector` 生成；native 仍重验最新 catalog 和 source。任务置为 succeeded 之前，Swift 严格解码并对照目标档、record/catalog/resolution、source/workload/runtime/device、授权与实际 layout、组件策略、container、memory scope 和 receipt 摘要；同时核对 model/operation/output/shape/seed/steps，拒绝 warmup 和伪造 boolean。此处消费 native verified result，不把 Swift 摘要检查称作独立 GPU receipt verifier。

`StreamingResolutionTests.swift` 已加入 build-app/test-app，独立编译运行 PASS：精确绑定、原始请求不变、各字段删除/篡改、数字/字符串冒充 bool、v2/v3 receipt。尚未覆盖 LTX worker-local resolution、新产物 staging/原子发布与全部 JobStore 错误恢复；这些仍是 R5 后续任务。

本轮验证：

| 检查 | 结果 | 范围 |
|---|---|---|
| `turbocider self-test` | PASS | M1 Pro Metal、BF16 Euler、layer norm、recipe/dependency rejection；不是完整模型 |
| `test_z_image_weight_stream.py` + `TURBOCIDER_TEST_GPU=1` | 4 PASS，6 SKIP | BF16 K1/K2 实际 GPU 值、复用/取消、损坏 metadata/截断拒绝；M5 专属 INT8/suffix 正确跳过 |
| `test_streaming_metal.py` | PASS | 真实 Metal K1/K2/K3、3 passes、输出相等；synthetic |
| `test_z_image_public_streaming.py` | PASS | source lease/layout、route rejection、失败清理、来源替换；host adapter |
| `test_contract.py` | 80 PASS，3 SKIP | 83 项 native API 合同；跳过原因保留在本机日志 |
| Swift streaming resolution regression | PASS | `/tmp/tc-resolution-tests.log`，独立 SDK fixture |

修复两个测试预期问题：原 exact 值检查对 13 个 tensor 求和，却只使用 1 个 tensor 的预期值；现断言 tensor 数为 13 并计算全部元素。旧 memory-constrained fixture 固定 48 GiB，会在本机 16 GiB 被正确拒绝；现使用不超过实机的 limit，分别验 plan-only / insufficient-budget rejection，并新增超过本机物理内存的拒绝检查。没有放松 runtime 的物理内存限制。

原始日志：`/tmp/tc-z-metal-tests.log`、`/tmp/tc-streaming-metal-tests.log`、`/tmp/tc-z-public-tests.log`、`/tmp/tc-contract-tests.log`。模型下载仍由 session `70822` 执行；没有真实 Flux/Z-Image 出图或 GPU/ANE 性能数据，M1 profile 仍明确关闭 M5 专属优化。

## 第三轮：Z-Image 全组件租约、App 启动与 Flux 4B 缺口

实现 56 R3 的来源/缓存接线：

- 公共 Z-Image SourceLease 从 transformer/text/VAE 三个文件扩展为四个，新增实际使用的 `tokenizer/tokenizer.json`。模板与配置逻辑目前在 runtime 中固定；没有将未使用的 tokenizer_config 声称为计算输入。
- 新的 `Tokenizer(fd, bytes)` 同步读取已授权 fd，不重开路径；probe 和 generate 都从本次租约构造 tokenizer，实际 conditioning 与 token 计数使用同一请求来源。
- public text encoder 与 VAE 经 `Weights::load_lease` 读取。默认路径仍使用原有 loader。
- 每次进入 public 或离开 public 到 legacy 时，先 synchronize 再清未绑定 source identity 的 conditioning/text/VAE 缓存。失败时保留 public-cache 标记，下一请求同步清理；没有在可能存在 pending 设备工作的 catch 中新增数组释放。
- adapter revision 更新为 `z-image-public-adapter-v3-all-component-lease`；组件策略更新为 `zimage-components-v2-all-sources-request-cache`。旧 calibration 不能直接充当新版本 evidence。

验证：

- 独立 `build/m1-component-lease` native 全量构建 exit 0，日志 `/tmp/tc-component-build.log`，避免覆盖在编译 App 的 `build/native`。
- `TURBOCIDER_TEST_NATIVE_DIR=.../build/m1-component-lease test_z_image_public_streaming.py` PASS。真实 fd 读取 tokenizer；替换路径后旧 fd 仍读原内容、旧 lease 拒绝、新 probe source identity 改变；invalid fd 和 truncated source 拒绝。transformer path replacement 继续拒绝。日志 `/tmp/tc-z-components-tests.log`。
- 相同新 dylib 的 `test_contract.py`：80 PASS、3 缺本地 LTX/Wan fixture 的 SKIP；日志 `/tmp/tc-component-contract-tests.log`。
- `DYLD_LIBRARY_PATH=.../build/m1-component-lease build/native/turbocider-studio-tests` PASS；日志 `/tmp/tc-component-studio-tests.log`。
- 原 App 构建的 Studio behavior / history batch-delete tests PASS。`TurboCiderNativeApp` 实际启动并存活约 2 分钟，启动日志无错误，随后主动关闭；这是启动 smoke，不是非空 catalog 的完整 GUI 生成验收。
- App 构建 session `44723` 仍在生成其余 integration binaries；完成后运行 `make test-app` 的等待任务为 session `81906`，日志 `/tmp/tc-app-suite.log`。不可提前宣称完整 suite PASS。

未完成：模型权重仍下载中（本轮约 4 GiB），未执行 public A→B→A 完整出图/质量回归；R1 内容身份分离、R4 drain/quarantine、R5 worker 与产物事务、R6 构建/container identity 和发布 evidence 均未因本改动完成。

### 下一项必须覆盖 Flux **4B**

代码复核确认：当前 `flux_streaming_descriptor.mm`、Flux public probe / exact run 以及 App publicStreamingModel 都只允许 9B。必须实现 4B，不能将文档里的 9B evidence 当作用户要求的 4B 完成。

固定官方 revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` 的 `transformer/config.json` 已单独获取（`/tmp/tc-flux4-transformer-config.json`）：24 heads × 128 = 3072 hidden；5 dual + 20 single blocks；joint_attention_dim=7680；MLP ratio=3。4B 是单个 `diffusion_pytorch_model.safetensors`，不能套用现有 9B 的两 shard index 要求或 hidden×3 的文本输入宽度假设。

下一轮优先扩展 metadata/source/descriptor/runtime 的 4B 实际几何与单文件支持、独立 fixture 和 App admission，然后在下载完成后测 M1 的 4B streaming。继续保留 9B 兼容回归。

## 第四轮：Flux 4B exact streaming 接入

已补齐前一节发现的 4B 缺口（开发能力，尚未发布认证）：

- Metadata 接受 4B 官方单个 `diffusion_pytorch_model.safetensors`；逐 tensor 校验 shape/dtype/range/完整性，不要求伪造 9B 的 index。
- 4B 固定 3072 hidden、24 heads、7680 context width、5 dual + 20 single。公共 probe/source/runtime/layout 身份区分 4B 与 9B；9B 保持原两 shard 布局。
- Exact adapter 使用实际 metadata 的 hidden/head 数，不再写死 4096/32；保留每 block 的真实 eval 完成边界、两个 pool、每池 K2，以及 D0/1、Q1/2。
- Exact 4B 不自动开启常驻 compiled graph；默认非 exact 4B 的 compiled 路径保持原行为。诊断 direct executor 同样支持 25 groups，可用于同布局后续对照。
- Candidate C API 和 App public model selector 接入 4B。生产 catalog 仍空，正常 public generate/resolve 继续明确拒绝未认证档位，没有绕过 gate。
- `build_app.sh` 现在遵守 native build 同一个 `TURBOCIDER_BUILD_OUTPUT_DIR`，修复独立 native 构建后 App 仍误链接 `build/native` 的目录不一致。

本机验证：

| 检查 | 结果 | 证据范围 |
|---|---|---|
| 4B/9B metadata descriptor suite | PASS | 单文件与两 shard；字节数 oracle、3072/7680、25 groups、D/Q、lease/path parity；missing/dtype/shape/overlap 必须以受控异常拒绝，不接受 crash 作为 PASS |
| Flux public adapter 4B + 9B | PASS | source closure、layout identity、错误路由/target、来源替换 |
| Flux candidate/public gate 4B + 9B | PASS | 空 catalog 拒绝 public；private candidate 到达对应 runtime；prepare selector 拒绝 |
| **M1 真 GPU 的 4B block smoke** | PASS | 合成零权重、实际 3072/24-head 形状、5+20 blocks × 2 passes=50 次、残差独立 oracle、2 pools/4 slots/50 fills；不是官方模型质量/性能证明 |
| 新 native API contracts | 80 PASS、3 缺 fixture 的 SKIP | 83 项，`/tmp/tc-flux4-contract-tests.log` |
| 新 CLI self-test | PASS | `/tmp/tc-flux4-selftest.json` |

构建目录 `build/m1-flux4`；native 构建 exit 0（`/tmp/tc-flux4-build.log`）。测试日志：`/tmp/tc-flux-descriptor-tests.log`、`/tmp/tc-flux4-public-tests.log`、`/tmp/tc-flux4-gate-tests.log`、`/tmp/tc-flux9-gate-tests.log`、`/tmp/tc-flux4-metal-tests.log`。真实 GPU fixture 入口为 `make test-flux4-streaming-metal`，可配 `TURBOCIDER_TEST_NATIVE_DIR` 选择构建。

前一轮完整 App build 与 `make test-app` 都已 exit 0，日志 `/tmp/tc-app-suite.log`，覆盖 streaming identity、ANE library、variant、Studio、history、model library、store、run insights、tensor cache。**该完整 App suite 对应新增 4B selector 之前的构建**；新的独立 App 构建 session `30718` 正在运行（`/tmp/tc-flux4-app-build.log`），其中已加入 4B Studio intent 回归，完成后须重跑。

下载 session `70822` 仍存活且持续增长，已开始 Flux transformer 权重下载；完整 Flux/Z-Image 文件尚未齐备。下一步：确认新 App/4B intent、完成官方模型校验后以 private exact 流程实跑，对比 eager resident / compiled resident / streamed，分别记录质量、全请求内存和耗时；再开展 ANE artifact 与 partition 实验。合成 block smoke 不作为模型出图、收益或 public qualification。

## 第五轮：内容验证基础与完整 App 回归

### 内容证明与请求文件绑定分开

新增 `SourceLease::capture_verified()`：只从该 lease 已打开的 fd 计算 SHA-256，使用 1 MiB 堆缓冲、`pread` 和取消检查；Apple 使用 CommonCrypto，其他平台保留 portable SHA 实现。全部文件完成后重新校验 held fd、named path 和 canonical target，再发布结果与缓存，避免读到变化中的文件仍发布证明。

`artifact_digest()` 使用独立 domain `tc-streaming-artifact-content-v1`，只编码排序后的 logical id、字节数、实际验证的 SHA-256；现有 `digest()` 保留请求 stat binding 语义。`capture()` 与 `open_and_verify()` 不因调用方提供了 `content_digest` 就授予内容证明；读取其 artifact digest 会明确失败。调用方提供给 verified capture 的摘要是待核对的期望值，不是可直接信任的证明。

增加有锁、最多 256 项的进程内原生摘要缓存，key 包含 device/inode/size/mtime/ctime。`capture_preverified()` 只消费仍匹配的缓存证明，缺失、被逐出或文件变化时返回 `artifact_verification_required`，不会隐式读取大权重。暴露本次验证读取字节数与缓存命中数用于检查成本。

测试覆盖：独立 Python canonical golden、不同安装副本内容身份一致而 binding 不同、logical role 区分、排序稳定、缓存零字节读取、同尺寸修改并恢复 mtime 后缓存失效、假摘要拒绝、metadata/replay 不授予证明、取消、短读、fd offset 保持、跨缓冲边界 SHA 对照，以及 **256 KiB worker stack** 上的实际文件摘要。

验证均 exit 0：source lease fixture（`/tmp/tc-source-content-tests.log`）、最终 fixture 的 ASan+UBSan（`/tmp/tc-source-content-sanitizer.log`）、完整 `make test-streaming-host PYTHON=python3`（`/tmp/tc-content-host-tests.log`，含 3535 layouts、executor/receipt/public resolver、LTX/H3/Z/Flux 4B/9B descriptor）及 repository 13 项测试（`/tmp/tc-content-repository-tests.log`）。本轮新增接口尚未用于已构建的 App；App suite 对应上一轮的完整 4B native/App 构建。

**边界仍未关闭**：这只是 R1/A1 的底层接口。当前缓存不能跨进程或重启保存证明；导入/下载 UI、Z/Flux public probe、portable descriptor/layout、catalog/Python/schema/receipt 迁移尚未接入。正常 public adapter 仍使用原 metadata capture，生产 catalog 仍为空；不能声称跨安装 public resolve 已通过，也没有重用旧认证证据。

### App 与下载验证

`build/m1-flux4` 的完整 App 构建 session `30718` 已 exit 0。随后从此目录运行九组 App suite（streaming resolution、ANE library、Studio variant、Studio behavior、history、model library、library store、run insights、tensor cache），全部 PASS，exit 0。新 Studio behavior 包含 4B public streaming intent；日志 `/tmp/tc-flux4-app-suite.log`。这次与前节旧构建的 App 测试已明确区分；仍不是 GUI 完整出图或 LTX worker 严格结果验收的证明。

对固定 revision 调用官方 Hugging Face API 的 `blobs=true`，取得 LFS SHA-256/普通 Git blob SHA-1 和字节数；对已完整落盘的 **18 个 Flux 文件** 全量验证，hash 前后核对 stat，全部匹配。日志 `/tmp/tc-download-verification.log`，逐文件结果 `/tmp/tc-download-verification.json`，原始官方清单 `/tmp/tc-flux-verified-manifest.json` 与 `/tmp/tc-z-verified-manifest.json`。

| 完整权重文件 | 字节数 | 官方 SHA-256 |
|---|---:|---|
| Flux text encoder shard 1 | 4,967,215,360 | `8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea` |
| Flux text encoder shard 2 | 3,077,766,632 | `82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07` |
| Flux VAE | 168,120,878 | `ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04` |

下载 session `70822` 持续运行，目录约 9.3 GiB；Flux transformer 与 Z BF16 transformer 正在下载。其余未完成文件未宣称通过校验。后续可重跑 `/tmp/tc-verify-downloads.py` 校验新完成文件，再执行真实模型推理。

## 第六轮：Flux 组件来源修复与新 App 启动

复核发现 Flux public probe 虽捕获 tokenizer 文件，却仍使用 engine 构造时的旧 tokenizer；run 也存在同样问题。现改为 probe 与 generate 各自从本次 lease 的 held fd 构建 tokenizer，解析前后重新校验来源，成功/失败都恢复请求绑定。新测试在同一 engine 内替换 tokenizer 的 special token，使实际 token 数改变：新 probe 必须匹配新文件，旧 fd 仍读到原内容，旧 snapshot/record/请求必须拒绝。

另外将原来仅 incoming public 才清理的组件缓存扩为 incoming/outgoing public 请求边界：run、prepare 和 load 在安全同步后清除 conditioning、encoder hybrid 和 VAE；失败时保留标记，下一次安全边界再清理。现有 text/VAE held-fd 加载保留。组件 revision 升为 `*-components-v2-all-sources-request-cache`，adapter 升为 `*-public-adapter-v3-all-component-lease`。修正 4B receipt runtime 的格式标签为 `diffusers-bf16-single-file`。

新 native 库与 CLI 已构建到 `build/m1-flux-components`，Flux 4B/9B public adapter 测试均 PASS（`/tmp/tc-flux-components-public-tests.log`），API contracts 80 PASS/3 缺 fixture SKIP（`/tmp/tc-flux-components-contract-tests.log`），本机 GPU self-test PASS（`/tmp/tc-flux-components-selftest.json`）。当前 build session `22293` 已进入后续完整 Swift 构建，尚未宣称整个 build exit 0。真实权重的 A→B→A 出图、失败恢复 owner 与缓存峰值仍待验，host fixture 不替代它们。

上一轮完整 `build/m1-flux4/TurboCiderNativeApp` 也已在桌面启动，90 秒保持存活，stdout/stderr 无报错，然后终止本次启动的进程。`/tmp/tc-flux4-app-launch.log`、`/tmp/tc-flux4-app-stderr.log` 为证据；这是 launch smoke，不是 UI 全流程验收，也不包含本节新 Flux native 修改。

## 第七轮：真实 Qwen3 权重的单 block ANE 工具链与桥接

Flux text encoder 已完整校验，因此无需等待 transformer 下载即可先验证 encoder artifact 工具链。首次在 App 的 Core ML Python 环境运行 `export_qwen3.py` 失败：无条件 `import mlx.core`，而该环境按 `coreml.lock.txt` 安装并不含 MLX。BF16/F16 路径只需要 NumPy；现将 MLX 导入移至实际 affine 量化解码分支。使用同一干净 Core ML 环境重跑，真实 BF16 的导出成功。量化源仍需要带 MLX 的构建环境，本次没有宣称覆盖它。

本次实验固定为 **Qwen3 layer 0、hidden=2560、MLP width=9728、ANE prefix=4864、64-token bucket、INT8 per-channel artifact、FP16 I/O、output_scale=1**。只导出一个 block，不将不完整 manifest 当作完整 encoder 路由。

导出命令（Python 为 `~/Library/Application Support/TurboCiderNative/toolchains/coreml/bin/python3`）：

```sh
python tools/coreml/export_qwen3.py \
  --model /Users/chencanhui/models/TurboCider/FLUX.2-klein-4B/text_encoder \
  --output /Users/chencanhui/models/TurboCider/experiments/m1-qwen3-block0-b64-w4864-int8 \
  --bucket 64 --ane-mlp-width 4864 --layer-count 1 --variant int8_pc
```

随后使用 native `coreml` 的 `compile` action 生成 managed-cache manifest（1 partition、0 cache hits），并通过 `tools/native/benchmark_coreml_ffn_bridge.py` 调用真实 C ABI：64 rows、seed=42、input_scale=0.25、1 次 warmup、5 次预测，额外以源 `.mlpackage` 的直接 Core ML predict 作桥接 oracle。export、compile、bridge 均 exit 0。

| 诊断项 | 结果 | 能证明的范围 |
|---|---:|---|
| native 与直接 Core ML 输出 | **bitwise equal**，max error=0，全部 finite | C ABI/FP16 backing/布局桥接正确；两侧是同一 INT8 artifact，不能证明相对原 BF16 的质量 |
| 5 次 native predict 中位数 | 2.780 ms | 单 block 诊断值；当时有 Swift 构建和下载，不作正式性能对照 |
| native create | 6.045 s | 包含来源校验与 model setup |
| manifest validation | 5.407 s | 冷启动成本显著，后续完整请求比较必须计入 |
| Core ML model load | 0.497 s | 与 manifest validation 分开报告 |
| compute plan preferred device | 两个 conv、split、silu、mul 共 5 个计算 op 偏好 Neural Engine | 静态 compute plan；不能替代运行时驻留/dispatch 观测 |
| observed ANE residency | **unknown** | 未取得运行时硬件证据，不宣称已证明所有计算驻留 ANE |

逐项原始结果、compute plan、导出来源、native library/exporter SHA-256 已封存到 [2026-09-20-m1-qwen3-block0-bridge.json](2026-09-20-m1-qwen3-block0-bridge.json)。本节没有 GPU baseline、原 BF16 数值对照、完整 encoder 或模型端到端收益，因此不能据此选择最佳 GPU/ANE 划分。下一步需要在空闲测量窗口完成数值 oracle、分宽度对照和完整请求验证。

当前权重目录约 11 GiB，下载 session `70822` 与新 App build session `22293` 继续运行；后者日志 `/tmp/tc-flux-components-build.log`。原目标仍未完成：两模型真实 streaming 出图/性能、ANE 最佳划分、R1 后续身份迁移、R2/R4/R5/R6 及产品认证仍须继续。

## 第八轮：R2 options 候选筛选与安装绑定验证

修复未安装查询拿不完整 workload 做完整相等比较的问题：新增独立 `PresetCandidateResolution` / `find_streaming_preset_candidate()`，仅按基础模型、操作、尺寸、步数、设备、容器等字段发现候选；继续检查 catalog、release、内存适配与校准。`resolve_streaming_preset()` 保留原完整 workload/source/runtime 比较，候选不构成执行权限。

同时发现实际请求的另一个失配：图片请求默认 fps=24，而 Z/Flux probe 的图片身份为 fps=0。公共基础 workload 构造现在将 image fps 归零；video 保留实际帧率。非空 catalog fixture 覆盖候选发现、缺组件/token 身份时 exact 仍拒绝、不同 token 数仍拒绝 exact、错误尺寸/容器/设备拒绝候选，以及图片/视频 fps 规则。

`tc_streaming_options_json` 对未绑定安装的匹配项返回 `candidate + artifact_verification_required`，不再返回误导性的 `available`。Swift 在提供模型目录且存在候选时才打开 metadata session，逐档调用原严格 resolver，核对返回 selector、catalog 与 container；只有成功项显示 available，所显示 record/校准/release 信息取自严格解析结果。空 catalog 不打开模型、不分配 GPU 权重。生成仍重新解析、冻结 exact selector 并验证实际结果。

App query key 纳入模型路径、prompt、动态文本、输入和加速策略等 draft 信息，排除返回的 streaming 状态以避免刷新循环。请求版本号与 key 防止旧结果覆盖新 draft，查询开始时清除旧可用快照。尚无已验证 options 时不按物理 RAM 推荐；首次使用保持 Off，即使有可用 streaming card 也不据此自动开启。显式用户选择保持。UI 区分“待验证”和“已校验本地模型”，不将技术错误码当成正常操作说明。

验证与范围：

- native host resolver suite PASS，`/tmp/tc-options-resolver-tests.log`；非空 catalog/不同 token 测试在纯 C++ 层执行。
- Swift resolution/options fixture PASS，覆盖严格结果替换候选信息、解析失败、catalog 变化、取消与既有结果校验；`/tmp/tc-options-resolution-new-native.log`。此处的非空 options 使用可控 resolver fixture，不冒充真实生产 catalog 验收。
- Studio behavior PASS，新增 prompt/path/token policy query identity、发布状态不触发循环、无记录不打开不存在的模型目录；`/tmp/tc-options-studio-new-native.log`。
- 新 native-only build 到 `build/m1-options`，session `55680` exit 0；83 API contracts 中 80 PASS/3 缺 fixture SKIP，GPU self-test PASS。日志 `/tmp/tc-options-native-build.log`、`/tmp/tc-options-contract-tests.log`、`/tmp/tc-options-selftest.json`。
- 最新 App 源码独立编译成功，binary 为 `build/m1-options/TurboCiderOptionsApp`；`/tmp/tc-options-app-build.log`。针对 Studio/options 的二进制在新库下重跑；本轮没有宣称重跑所有 App integration binaries。

前节后台完整 Swift build `22293` 实际以 exit 1 结束，原因是本轮更新源文件时编译器检测到 `TurboCiderNative.swift` 被修改。它不计为通过；上述新 native-only 构建与固定源码的 App/Studio 编译验证取代其本轮证据。临时把 Studio binary 放在 `/tmp` 的首次执行也因找不到同目录 `turbocider` helper 失败；移至 native 构建目录后通过，未把该测试布置错误当作产品通过。

生产 catalog 仍为空，portable identity/schema、真实非空发布包与 container 身份仍待完成。R5 的完整 worker 成功事务/原子产物发布和 R4 失败隔离仍未关闭。

### 下载复用尝试被内容校验拒绝

通过 pinned Comfy revision 的 HTTP Range 读取 Z Qwen3 原始 safetensors header，确认它与已校验 Flux text encoder 的 398 个 tensor 名称/shape/dtype 一致。尝试按该原始 header 拼接 Flux tensor 数据，得到完整 8,044,982,048 字节临时文件，但 SHA-256 为 `e37269b7ca1301ad72a92627ce95432ab5aad5f89143a06055886aad3419d12f`，**不匹配** Z 官方 `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a`。

因此未发布到 Z 模型路径，已删除本次临时重组文件，继续原官方下载。首段 216,288 字节 payload 相同也不能推导整文件相同；不将这次尝试记为下载节省或成功复用。失败记录 `/tmp/tc-z-qwen-reconstruction.json`。下载 session `70822` 仍在运行，完整两模型尚未齐备。


## 第九轮：LTX worker 结果验证与原子发布

原 worker 将 exit 0 和输出文件存在视为成功，可能接收空 JSON、旧文件或不可解码视频。现在每次创建同目录私有 staging，子进程只写 staging；父进程严格检查结果类型、model/operation/seed/steps/尺寸/输出路径。公共 streaming 还要求完整 verified summary、合法摘要、授权与实际 layout 一致，以及请求固定的 selector 字段匹配。

发布前检查普通非空文件、视频尺寸/时长/音频轨道，逐帧解码并核对帧数；流式计算 SHA-256，在读取前后核对文件身份。构造带 SHA/字节数的 worker_artifact receipt 后以同文件系统 rename 发布。错误或取消保留原产物并清理 staging。stdout 上限 8 MiB、events 上限 16 MiB，避免结果与诊断无限增长。

测试期间定位并修复一个真实退出等待问题：子进程已结束，协程仍阻塞于同步 waitUntilExit()；改用 terminationHandler 和异步 continuation 通知退出。取消保留 SIGTERM 后 5 秒 SIGKILL 的行为，异常清理也等待本次子进程退出后再删除临时目录。

新增 `tests/integration/LTXWorkerTests.swift`，用真正编码的 9 帧 H.264 视频和受控 worker 覆盖：成功发布、legacy 路径、空/错误 JSON、错误 seed/path/target/layout/preset、verified 类型错误、缺 summary、空/缺失/损坏文件、symlink、尺寸/帧数/音频不符、日志超限、发布失败、原文件保留及取消清理。它已接入 build_app.sh 和 test-app。

验证结果：固定源码的 worker 编译和测试 exit 0（`/tmp/tc-worker-build.log`、`/tmp/tc-worker-tests.log`）；StreamingResolution 与 Studio 回归 PASS（`/tmp/tc-worker-resolution-tests.log`、`/tmp/tc-worker-studio-tests.log`）；完整 App 编译 exit 0，产物 `build/m1-options/TurboCiderNativeApp`（`/tmp/tc-worker-regressions-build.log`）。本轮使用受控 worker，未宣称真实 LTX 推理成功。

这仍不是完整 R5 关闭：尚缺独立 worker resolution envelope、job/request hash 协议与持久化有界诊断；当前 summary 容器身份仍待 R6 修正。非 LTX 产物事务、进程树超时及 R4 GPU 失败隔离也仍待处理。

## 第十轮：真实 Qwen3 前缀与 BF16 GPU 数值对照

新增 `tools/native/compare_qwen3_partition.py`，读取原 checkpoint 的 gate/up/down，在显式 MLX GPU 上计算 BF16 MLP 前缀，并与 native Core ML C ABI 输出比较。两侧接收同一 BF16 舍入后的输入，Core ML 以 FP16 传输；按 native Qwen 调用方规则恢复 output_scale。使用 layer 0、64 rows、prefix 4864、seed 42、input_scale 0.25，与第七轮配置一致。额外导出并编译同形状 FP16 artifact 作控制组。

| artifact | 相对 L2 | cosine | 最大绝对误差 | 全部有限 |
|---|---:|---:|---:|---|
| INT8 per-channel | 0.029056662 | 0.999578075 | 0.044921875 | 是 |
| FP16 | 0.026692391 | 0.999644048 | 0.035156250 | 是 |

原始结果见 [INT8 oracle](2026-09-20-m1-qwen3-prefix-oracle.json) 和 [FP16 oracle](2026-09-20-m1-qwen3-fp16-prefix-oracle.json)。两个 native session 均验证 checkpoint SHA-256；运行时 ANE 驻留仍 unknown。FP16 本身相对 BF16 也有差异，不能把 INT8 的全部偏差归因于量化，亦不能把两项 L2 直接相减当作量化误差。

复现命令如下，MANIFEST 为各 JSON 中记录的已编译 manifest，REPORT 为新的结果路径：

```sh
.venv/bin/python tools/native/compare_qwen3_partition.py \
  --model /Users/chencanhui/models/TurboCider/FLUX.2-klein-4B/text_encoder \
  --manifest "$MANIFEST" --library build/m1-options/libturbocider.dylib \
  --output "$REPORT" --block 0 --rows 64 --seed 42 --input-scale 0.25
```

这是随机 hidden states 上单个真实权重前缀的误差测量，没有设置事后通过阈值，也不代表完整 encoder/image 质量通过。报告中的内部 quality_validation_passed=true 伴随 quality_validation_enabled=false，不是本次数值比较的通过证明。测量期间仍有构建与下载，不以这些调用耗时评判收益或最佳 GPU/ANE 划分。官方 transformer 下载仍在进行，两模型端到端 streaming、完整 encoder 对照及空闲窗口性能实验仍待完成。
