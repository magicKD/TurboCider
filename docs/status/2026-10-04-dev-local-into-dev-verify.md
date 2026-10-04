# dev-local 合并到 dev-verify 验收

2026-10-04，在 M4 Pro / 48 GiB 上完成本地合并和定向回归。

## 合并内容

- 目标合并前：`2b1e77a32b8181a63934183929ecdcb12c54c50b`。
- 来源 dev-local：`017145dfa4973cb8135d3f91f7d429b979241fe5`。
- 共同祖先：`975aa91e23b0d60c4494219930d54307fcf3b5ca`。
- 使用普通双亲 merge，保留两侧提交历史；dev-local 指针未改变。
- 无文本冲突。四个共同修改文件为 StudioState、Swift binding、Qwen pipeline 和 native contract tests，均自动合并后复审。
- 保留 dev-verify 的 M5 Pro 24 GiB 条件化 layer streaming、fd 权重读取和 W8A8 选项，也保留 dev-local 的 Playground、AI API、512 参考编码 / DiT 和分阶段驻留改进。

除下述测试脚本调整外，20 个仅目标分支修改的文件、120 个仅来源分支修改的文件，均逐字节匹配对应父提交。未手工修改产品代码。

合并验收修正了三处测试脚本：两个权重 / text streaming 测试尊重 `TURBOCIDER_TEST_NATIVE_DIR`，确保链接本次新库；仓库布局测试适配已经支持的自定义打包目录，替换失效的旧路径断言。

## 本次实际执行

| 检查 | 结果 |
| --- | --- |
| Native、CLI、Swift App、模型库 helper 全新构建 | 通过 |
| 原生源码构建清单和 App 源码哈希复核 | 435 个 native 输入、54 个 App / binding 输入与冻结构建一致 |
| Native / API / 小型 Metal 回归 | 187 个 unittest 方法：184 通过、3 跳过；另有独立 fd-backed MLX fixture 通过 |
| C++ 边界 / 驻留 / 硬件门控 | 2 个测试通过，含模拟 M5 正向和 M4 等负向门控 |
| 仓库布局、独立性、异步准备、构建身份 | 四组均通过 |
| Swift 回归 | 参考尺寸、Playground、Qwen 工作流、编辑画布、ANE 库登记、App API、模型库七组均通过 |
| 隔离打包与签名 | 成功；`codesign --verify --deep --strict` 通过 |
| 合并状态与空白检查 | 无未解决冲突；工作区和暂存区 `git diff --check` 通过 |

Native 回归覆盖请求契约、共享工作流、图片准备、Python 客户端、CLI 发现、RPC 校验、安装登记、会话复用，以及实际 tiny Metal 的 encoder streaming、DiT / LoRA 生命周期和 runtime activation 释放。三项跳过均因缺少非 Qwen 可选测试模型：Wan base、Gemma4 tokenizer、Gemma4 ConvRot checkpoint；没有为此下载模型。

## 一次完整模型回归

使用本机已有 Qwen-Image-2.1 base，以同一参考图、请求和 seed 对照此前 dev-local 验收输出：

- GPU 单图编辑，512×512 输出、512 参考编码（1024 reference tokens），25 步，seed 42。
- Balanced DiT cache；`component_staged` 驻留，encoder 完成后可释放。
- 请求耗时 **49.104 s**，去噪 **39.993 s**；缓存命中 **11 步**。
- 输出与 dev-local 基线 **逐像素一致**，RGBA MAE / 最大误差均为 0，PNG SHA256 也相同。
- 目视确认正常输出。M4 正确选择 GPU 路径，未进入 M5 专用 layer streaming。
- MLX allocator 峰值约 **15.81 GiB**，结果阶段 active 约 **0.55 MiB**；这些值不包含系统、文件缓存或 Core ML 内存。

仅执行一组完整模型请求，用于合并正确性回归，不据此推断性能提升。本轮没有重新执行完整 LoRA 大模型矩阵，没有 M5 实机硬件验收，也没有重新操作桌面 UI。对应逻辑由上述定向自动化测试覆盖，不将其称为全部功能验收。

## 构建和证据

本次 runtime identity：

`tc-runtime-build-v1-c79834f2a53ed61033738ca26312eca5c7a7584b205dcc0f642c12c12466169d`

本地原始记录位于 `outputs/merge-dev-local-20261004/`：父提交、内容比对、构建日志、冻结哈希、各组回归日志、完整模型请求 / 结果 / 图片与像素对比。初次失效的布局断言日志也保留。它们是本地验收产物，不随源码分发。

构建目录为 `build/merge-dev-local-20261004/`，隔离包为 `dist/merge-dev-local-20261004/TurboCider.app`。没有替换正在运行的既有 App，没有下载模型或新增 ANE / Core ML 导出。

本机 Xcode launcher 存在 CoreDevice 加载问题，因此构建和相关测试使用已安装的 Command Line Tools 与 macOS 15.2 SDK；部署版本沿用 MLX 依赖要求的 macOS 26.2。可用同样的本地工具链复现构建：

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk \
TURBOCIDER_BUILD_OUTPUT_DIR=build/merge-dev-local-20261004 \
TURBOCIDER_BUILD_PACKAGE_ONLY=1 bash tools/native/build.sh
```

Python native 回归使用 `TURBOCIDER_TEST_NATIVE_DIR=build/merge-dev-local-20261004` 选择新构建；Swift 七组也均重新编译并记录二进制哈希，未借用旧测试可执行文件。
