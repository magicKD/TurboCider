# TurboCider 独立部署与完整模型链路

2026-09-06。当前可执行交付为 FLUX.2 Klein 4B；H3/LTX 仍按单独契约等待实现与模型验收。本次独立性改造没有扩大其已实现范围。

## 独立性的具体边界

原始模型权重可以保留在任意用户指定目录，包括本项目之外。原模型没有移动、复制或下载。代码、构建依赖、转换工具链、Core ML 源分区和编译缓存由 TurboCider 自己管理，默认流程不读取其他 workspace 的实现、Python 环境或加速产物。

此前不独立的两处已移除：

1. `AccelerationDiscovery` 不再推导相邻 `mac_local_ai` 目录，也不再默认搜索外部模型目录里的 `coreml/`。自动适配查询 TurboCider 托管缓存，旧的外部 preferred 路径不能抢占托管结果。明确手动选择/环境变量覆盖仍是可选入口，不构成默认依赖。
2. 构建不再要求指定其他项目的 `MLX_ROOT`；离线转换不再借用其他项目的 `.deps/coreml`。本机 profile 已移除旧 Python 路径，App 已切换到重新导出的自有分区。

第三方库和系统工具有明确边界：原生发行包包含 MLX dylib/Metal 库；离线转换使用 TurboCider 托管的 Core ML Python 环境；开发需要完整 Xcode 与标准 CPython 3.11。源模型始终由调用者提供。

## 目录与归属

| 内容 | 默认位置 | 生命周期 |
|---|---|---|
| 原始 safetensors/config/tokenizer | 用户指定目录 | 只读，不自动复制或删除 |
| 构建依赖 | 仓库 `.venv/` | `make setup` 安装，由锁文件固定 |
| App/CLI 转换工具链 | `~/Library/Application Support/TurboCiderNative/toolchains/coreml/` | `make setup` 安装；发行包可直接调用，脱离源码仍可用 |
| 导出脚本 | `tools/coreml/export_flux2.py`；随 App/CLI 打包 | 本项目实现，不调用外部研究脚本 |
| Core ML 源分区 | `~/Library/Application Support/TurboCiderNative/coreml/flux2-klein-4b/m1088/` | 本机 safetensors 直接导出 |
| 稳定 compiled 缓存 | `~/Library/Application Support/TurboCiderNative/cache/coreml/` | 内容寻址、复用与清理 |
| Core ML 设备专用缓存 | 系统分配给 TurboCider 进程的用户缓存目录 | App/CLI 按已知范围提供清理预览 |

Application Support 是 TurboCider 的应用数据目录，适用于安装后的原生 Mac App；它不属于其他项目，也不会因为移动源码目录而丢失。开发者可用 profile 的相对目录把源分区与缓存放在仓库 `artifacts/` 下，该目录已忽略提交。

## 从干净 checkout 启动

```sh
make setup
make package
make test
```

`make setup` 调用 `tools/setup_dependencies.py`，安装两套被锁定的环境：项目构建环境与 App 自有离线转换环境。不会查找相邻仓库，不安装/下载模型。首次联网安装仅获取第三方依赖；已有 wheelhouse 时支持离线：

```sh
python3.11 tools/setup_dependencies.py --offline --wheelhouse /path/to/wheels
```

锁文件为 `tools/dependencies/build.lock.txt` 与 `coreml.lock.txt`，包含全部当前解析的传递依赖。`make build`、`make package` 自动从本项目 `.venv` 找到 C++ MLX，不要求设置环境变量。`MLX_ROOT` 保留为用户明确选择其他独立 MLX 构建的覆盖入口。

当前选用的官方 MLX/MLX Metal 0.32.0 wheel，其 Mach-O 最低系统版本实测为 **macOS 26.2**，因此构建和 App 声明同步为 26.2。之前标注的 15.0 不能用于这一依赖包。验收主机为 26.6；低版本 macOS 需要另外验证兼容 MLX 构建。

## 独立的模型处理链路

App 模型页：选择原始模型 → 导出 safetensors → 预编译 → 自动适配 → 加载/预热 → 生成。源包、编译结果、磁盘占用、删除预览均由同一 C ABI 资源服务管理。

CLI 例子，`request.json`：

```json
{
  "action": "export",
  "model_root": "/user/selected/FLUX.2-klein-4B"
}
```

```sh
dist/cli/turbocider coreml request.json
```

不指定 Python 时使用 App 自有 `toolchains/coreml/bin/python3`；不指定 profile 时使用当前默认 1088 桶和自有目录。要定制 bucket/路径，使用 `profiles/apple-m4-pro-48gb.example.json` 的 `coreml_export`。随后用 `action:"compile"` 与返回的 `source_manifest` 编译，并将返回的 compiled manifest 用于推理。详细协议见 [Core ML 资源管理](coreml-artifacts-and-storage.md)。

构建解析解释器使用隔离模式；转换子进程清除继承的 `PYTHONPATH`、`PYTHONHOME` 并禁用用户 site packages。显式配置依赖目录仍可覆盖，但默认不会被某个开发 workspace 的 Python 环境污染。

App 本机配置已迁移，原始模型路径保持原样，`manifest` 和 `sourceManifest` 指向自有目录。旧外部源包/编译包保留在其原位置，因为其他项目可能仍使用它们；TurboCider 当前链路不再读取它们。

## 可验证的隔离验收

除了静态路径检查，还完成了实际文件访问隔离：

- 将 `dist/cli` 与测试驱动复制到项目外临时目录。
- 拒绝读取 TurboCider 源码及其他 workspace，并禁用网络。
- 仅放行用户原始模型目录，及路径遍历所需的父目录元数据；不允许列出外部模型父项目的内容。
- 验证读取源码 Makefile 和外部研究脚本确实得到权限错误，排除“只是不主动用”的假隔离。
- 在该条件下重新运行包内导出器、复用 20 分区 compiled 缓存，并完成 GPU 和 GPU/ANE 生成及资源统计。

首次隔离导出曾因为模型祖先目录元数据也被禁止而失败；调整为仅放行路径解析所需元数据后通过。没有为任何外部工程代码添加放行。

256×256、4 步、seed 42 的混合输出与既有黄金 PNG SHA256 完全一致。512×512、4 步暖请求中位数：GPU **4.664 秒**，GPU/ANE **2.903 秒**，各 4 次请求、排除首个样本。它们是环境迁移回归观察，不是随机交错的跨实现性能研究；与此前约 4.68 / 2.90 秒相符。

`tests/repository/test_independence.py` 防止运行代码重新加入相邻工程探测；`tests/native/test_packaged_independence.py` 为复制后的发行包提供隔离验证驱动。参考对比工具可显式传入 oracle/其他引擎用于研究，正常构建与独立验收不需要它们。

完整 Studio 三操作、生命周期、服务及 Swift 交互回归均通过。[归一化验收证据](validation/standalone-project-validation.json) 包含隔离结果、数值校验、样本时间和发行包身份。
