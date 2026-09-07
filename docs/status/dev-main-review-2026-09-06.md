# dev 对比 main 的实现与合并审查

审查日期：2026-09-06

审查范围：`dev@2cee456`（Merge main C++ runtime with dev model executors）对比 `main@410a605`（cpp arch）。以下结论针对这两个提交，不代表后续修复状态。

## 结论

**暂不建议将当前 dev 直接合入 main。** dev 能够编译、打包并完成真实 FLUX 推理，但新增 LoRA 实现存在已复现的正确性问题，LTX 能力门禁没有贯穿执行入口，默认测试也尚未全绿。

dev 已包含 main，Git 层面可以快进合并；阻碍合并的是实现和验证问题，而不是分支冲突。本次审查未修改产品源码，也未执行合并。

主要增量是 H3、LTX、FastMetal 执行器、FLUX 9B 与 LoRA 支持，以及相应的 App、服务、打包和请求契约修改。

## 待修复问题

### 1. [P1] FLUX LoRA 加载失败后，同一会话重试会错误成功

状态：未修复；已通过最小权重测试和真实 FLUX 同会话请求复现。

代码位置：

- `native/backends/mlx.cpp`：`Weights::apply_loras`，约第 95–140 行。
- `native/models/flux2/pipeline.cpp`：`Flux::select_loras`、`Flux::load`，以及 `Flux::run` 中约第 263–266 行的加载判断。
- `native/api/c_api.mm`：生成失败后的异常处理保留原 Session。

`apply_loras` 逐项原地修改权重，后续适配器或张量校验失败时没有回滚。LoRA 身份已经缓存，基础权重也已加载；相同请求重试时，`select_loras` 不清空权重，`transformer_cold` 为 false，因而跳过 LoRA 应用与校验。

复现结果：

```text
第一次请求：code=1，error="incomplete LoRA pair: z"
同会话相同请求重试：code=0，lora_fusion="load_time_baked"
```

第二次请求实际上没有成功应用该无效适配器，却生成了结果并报告 LoRA 已融合。另一个小型权重测试中，基础权重均值为 1，LoRA 中途失败后均值为 2，说明失败也可能留下部分修改的权重。

建议：采用事务式加载，或在 LoRA 加载失败、取消时清空相关权重和缓存状态，确保下一次请求重新加载并验证。增加“加载失败后相同请求重试仍失败”和“失败后有效请求不受污染”的行为测试。

### 2. [P1] LTX 的能力门禁没有贯穿执行入口

状态：未修复；源码确认，未做真实 LTX 推理验证。

代码位置：

- `native/runtime/plan.cpp`：将 LTX 音频或非 `video.generate` 请求标记为 `recipe.executable=false`。
- `native/platform/apple/ltx_session.mm`：`LtxNativeSession::generate`，约第 1500 行起。
- `native/api/c_api.mm`：`tc_engine_generate`。
- `services/turbociderd/service.mm`：`external_ltx_request`，约第 86 行；以及 submit 中的 executable 检查。

planner 将音频和图生视频标记为不可执行，但公共 Session 没有强制拒绝该标记，仍可进入对应实现；C API 也没有统一检查。服务对 `component_staged + audio=false` 的 LTX 请求绕过 executable 检查，分流条件没有限制 operation。

因此，文档声明的“未验收能力禁止执行”没有落实到所有入口。App 也仍展示 LTX 图生视频和音频选项，能力描述、UI 与执行行为不一致。

建议：在公共执行边界统一落实能力门禁；如保留实验入口，应显式区分正式能力与实验能力。服务分流不能覆盖能力校验，App 应根据实际可执行能力提供选项。补充 CLI、C API、App 请求和服务入口的门禁行为测试。

### 3. [P2] FLUX 文本编码器 LoRA 的原生键名被错误截断

状态：未修复；已通过小型权重测试复现。

代码位置：`native/backends/mlx.cpp`，`strip_lora_prefix`，约第 12–18 行。

原生文本编码器适配器键名 `model.layers.*` 被统一去掉 `model.`，变成 `layers.*`；基础权重仍使用 `model.layers.*`，因此无法匹配。复现错误：

```text
LoRA did not match any text_encoder weights
```

这不表示所有封装格式的文本 LoRA 都会失败，而是当前前缀归一化不兼容原生键名。

建议：优先尝试原始键名，并按组件和已支持的适配器格式处理前缀；增加原生文本编码器键名与封装前缀的数值行为测试。

### 4. 测试门禁：LTX 同步检查依赖兄弟仓库当前状态

状态：待处理；本机 `make test` 已复现失败。

`tests/native/test_contract.py` 的 `test_ltx_shared_hot_path_stays_synced_with_ltx_mac` 直接比较本仓库与兄弟 `ltx-mac` 工作树。以下两个文件不一致：

- `ltx_transformer_io.c`
- `ltx_shaders.metal`

该失败本身不能证明 dev 的 LTX 数学实现错误；它说明同步断言未满足，而且测试结果受仓库外工作树影响。没有兄弟仓库时该测试会跳过。

建议：检查差异的来源和必要性，对照固定提交或固定内容摘要验证 vendored runtime，而不是依赖兄弟仓库当前工作树。

## 实际验证结果

环境：Apple M4 Pro，48 GB 统一内存，macOS 26.6，MLX 0.32.0。

| 检查 | 结果与范围 |
|---|---|
| dev 原生库、CLI、Swift App 编译 | 通过 |
| dev 打包与签名验证 | 通过；本机包最低 macOS 为 26.2 |
| dev 原生及发行 CLI 自检 | 通过 |
| Swift 行为测试 | 通过 |
| FLUX 4B，128×128、4 步、GPU | main/dev 均成功，输出 PNG SHA-256 完全一致 |
| dev 服务，FLUX 512×512 | 生成、常驻复用、取消、跨进程 GPU 锁、历史分页和崩溃恢复通过 |
| main `make test` | 通过：仓库测试 7 项、契约测试 10 项 |
| dev `make test` | 仓库测试 8 项通过；契约测试运行 38 项，其中一个测试的两个子项失败，3 项跳过 |
| 完整 pytest | 未执行成功；当前环境缺少 pytest |
| H3、LTX、FastMetal、FLUX 9B 真实推理 | 本次未完成验收 |

FLUX 对照请求：提示词 `A red apple on a wooden table`，128×128，4 步，seed=42，execution=gpu。两份 PNG 的 SHA-256：

```text
77eb9d626552f5ef2f5a1b6b7314b198a2bae0415f3d58913c93d241be37faf5
```

此对照证明该请求下原有 FLUX GPU 路径的输出一致，不能替代完整尺寸、图像输入、ANE、LoRA 和性能回归矩阵。

直接 `make build` 最初被本机 Xcode 工具定位故障阻断，显式指定 SDK 和工具链后成功。沙箱内 GPU 不可见，沙箱外 Metal 自检通过。这两项应与源码缺陷区分。

构建使用的环境覆盖：

```sh
env \
  PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:/usr/local/bin:/usr/bin:/bin \
  SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  make build
```

## 已知运行边界

H3/LTX 首次 LoRA 合并仍依赖外部 `h3.c/tools` 合并脚本和 Python，发行包没有携带这些合并脚本。FastMetal 也依赖显式配置的外部 worker 环境。这些限制已在独立运行文档中说明，不应将“编译和发行 CLI 自检通过”解释为“所有模型和适配器均可独立运行”。

参考：[独立运行与外部依赖边界](independence-and-dependencies-2026-09-06.md)、[实现状态](implementation-status-2026-09-06.md)。

## 合并前条件

1. 修复上述三个实现问题，并通过失败恢复、键名映射和能力门禁行为测试。
2. 处理 LTX 同步断言，确保测试结果可复现。
3. 在配置完整的环境运行完整测试套件，复验构建、打包、FLUX 输出对照和服务行为。
4. 对尚未验收的模型能力维持明确门禁；实际开放哪些能力，就提供哪些能力的真实模型验证证据。

满足以上条件后，可考虑作为开发版本合入 main；本次证据不足以支持“全部模型已完成验收”的结论。

## 本机审查证据位置

以下是审查时生成的临时文件，不随仓库提交，也不保证长期保留。关键结果已记录在本文中。

- `/tmp/turbocider-dev-build-direct.log`：dev 构建。
- `/tmp/tc-dev-package.log`：打包与签名。
- `/tmp/turbocider-dev-test-fresh.log`：dev 测试。
- `/tmp/tc-main-review-tests.log`：main 测试。
- `/tmp/tc-lora-review.cpp`：小型 LoRA 权重复现程序。
- `/tmp/tc-review-retry.py`、`/tmp/tc-review-retry.log`：真实 FLUX 同会话失败重试复现。
- `/tmp/tc-dev-review-generate.json`、`/tmp/tc-main-review-generate.json`：FLUX 输出对照。
- `/tmp/tc-dev-review-service/report.json`：服务行为测试报告。
- `/tmp/tc-dev-packaged-selftest.json`：发行 CLI 自检。
