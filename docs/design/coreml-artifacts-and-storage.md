# Core ML 模型、编译产物与磁盘管理

2026-09-09 边界更新：正式 App 和核心库已移除 Python 导出入口。
`export` 资源请求明确拒绝；请在开发环境运行
`tools/coreml/export_flux2.py` 或 `tools/coreml/export_z_image.py`，
使用显式 `--model`、`--output`、`--bucket`，带 LoRA 时同时传入
`--lora` / `--lora-strength`，并为不同融合身份选择不同输出目录。
导出器继续校验输出身份，App/native 继续校验 artifact 与请求 LoRA 身份。
App 通过选择源 manifest 使用原生编译和缓存管理，不选择 Python 环境。
以下旧导出 API / App 操作说明仅记录历史实现，不是当前支持的调用方式。

2026-09-06 实现与本机检查。当前支持 FLUX.2 Klein 4B 的 20 个 INT8 单流块 MLP 分区；没有宣称能把任意 safetensors 自动转换成完整 Core ML 网络。

最新独立性更新见 [独立部署说明](standalone-project.md)：本机已重新导出至 TurboCider 自有目录，不再使用下文记录的旧外部分区；下文的外部占用是当时的诊断证据。

## 为什么 App 原来显示 0

此前 App 只调用自身 `cache/coreml` 的计数，而自动适配实际加载了相邻项目的 compiled manifest。因此 0 表示“这个托管目录没有已编译条目”，并不表示机器上没有 Core ML 模型或缓存。

本机已有 FLUX 分区根目录占约 3.2 GiB，其中约 1.6 GiB 为 `.mlpackage` 源模型，另约 1.6 GiB 为 `.mlmodelc` 编译产物。外部 `mac_local_ai/models/coreml` 整体约 16 GiB，但其中还有其他模型/研究产物，不能全部称为 TurboCider 重复缓存。

在可访问的用户缓存目录内还发现多个进程的设备专用缓存。当前 `org.turbocider.native` 和旧 `turbocider-native` 的 e5bundlecache 各约 240 KiB；`macllm` 的同类目录约 2.3 GiB，Python 目录约 163 MiB。不能将其他工具或系统进程的缓存全部归因于 TurboCider；这也不是需要 root 权限的系统全盘审计。

## 三层存储与明确路径

| 层 | 内容 | 位置与管理 |
|---|---|---|
| Core ML 源模型 | 量化后的权重和 MIL 图，`.mlpackage` | 默认 Application Support/TurboCiderNative/coreml/flux2-klein-4b/m1088；可由配置或 App 覆盖 |
| 稳定编译产物 | `.mlmodelc`、内容身份与 compiled manifest | 默认 Application Support/TurboCiderNative/cache/coreml；可配置；兼容外部当前 manifest |
| 设备专用缓存 | 系统为设备进一步准备的资源 | 实测位于 `~/Library/Caches/<进程或 bundle ID>/com.apple.e5rt.e5bundlecache`；由 Core ML/OS 选择，不提供私有重定向接口 |

稳定 `.mlmodelc` 路径有利于复用 Core ML 后续准备；反复创建不同临时路径可能导致重复准备。[Apple Core ML 模型预测说明](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html)

新导出器使用 `skip_model_load=True` 生成源包，转换时不顺带加载完整 Core ML 模型。编译器将返回的临时产物复制到稳定内容地址并删除临时副本。运行阶段继续加载稳定 `.mlmodelc`，不在每次生成时转换 safetensors。

App 的统计现在分别列出当前源包、当前编译分区、托管编译目录和 TurboCider 进程缓存；同一路径/已包含子路径不重复累计。既显示文件大小，也在接口返回 allocated bytes；APFS 克隆/压缩/共享块意味着 allocated bytes 也不等于承诺可释放的独占物理空间。不可读取的条目返回错误，不能伪装成 0。

设备缓存清理范围限定为 `org.turbocider.native`、`TurboCiderNativeApp`、`turbocider`、`turbocider-native` 的已知缓存子目录。不删除其他应用、Python 共享目录或系统全局缓存。更换 macOS 后目录布局可能变化，当前实现不会猜测新的私有位置。

## 配置与离线构建

现有 profile 的 FLUX 项增加 `coreml_export`。推理配置解析器允许该构建字段，但正常推理不启动 Python；profile 的 `enabled` 控制运行策略，离线构建可读取尚未启用的配置。

```json
{
  "schema_version": 1,
  "enabled": false,
  "match": {"gpu_name": "Apple M4 Pro", "memory_bytes": 51539607552},
  "models": {
    "flux2-klein-4b": {
      "policy": "auto",
      "allow_approximation": true,
      "coreml_export": {
        "bucket": 1088,
        "variant": "int8_pc",
        "output_dir": "../artifacts/flux2/source",
        "cache_dir": "../artifacts/flux2/compiled",
        "python": "/path/to/python3",
        "python_path": "/optional/local/site-packages"
      }
    }
  }
}
```

`python_path` 可省略，此时使用所选 Python 已安装的环境。离线工具依赖见 `tools/coreml/requirements.txt`，实测 CPython 3.11、coremltools 8.3.0、NumPy 2.0.2。项目不自动安装依赖。App 可选择解释器、依赖目录、构建配置及输出/缓存目录；显式界面设置优先于配置文件。未提供构建配置时采用 1088 桶和默认托管目录。

转换读取 BF16/F16 safetensors，检查 FLUX 配置及所有相关张量的尺寸/偏移，逐块导出 `conv → SwiGLU → conv`，采用对称 per-channel INT8 权重量化，FP16 输入输出。GPU 保留 attention 等其余部分。

每个输出目录有导出身份、权重 SHA256、工具版本、桶和配方版本；单块原子发布并记录文件 SHA，可复用完整块；manifest 仅在 20 块全部完成并重新核对权重 SHA 后发布。不同身份拒绝覆盖同一目录。取消会停止并等待子进程退出，不发布半成品 manifest；已完成的块可供下次复用。源 safetensors 不修改。

## CLI 和 C/Swift 接口

CLI 统一入口：

```sh
build/native/turbocider coreml request.json
# 独立包对应 dist/cli/turbocider coreml request.json
```

所有请求为本地 JSON。路径以本机配置提供，不把机器用户名写入可提交示例。

查看当前模型和缓存：

```json
{"action":"inventory","manifest":"/path/to/compiled/manifest.json"}
```

compiled manifest 的 `source_manifest` 用于定位源包；旧格式没有时可显式传 `source_manifest`。可同时传 `profile`、`cache`、`storage`，查询与构建使用相同路径解析逻辑。

从权重导出：

```json
{"action":"export","model_root":"/path/to/FLUX.2-klein-4B","profile":"/path/to/device.local.json"}
```

解释器也可以由请求字段 `python`、`python_path` 提供。进度事件输出到 stderr，最终 JSON 到 stdout。详细构建诊断保存在输出目录 `export.log`。仅执行包内随附的导出脚本，不拼接 shell 命令。

编译现有源包：

```json
{"action":"compile","profile":"/path/to/device.local.json","source_manifest":"/path/to/source/manifest.json"}
```

返回的 `manifest` 是实际 compiled manifest，App 会保存该路径供后续推理使用。不是固定命名的占位文件。

清理先生成预览：

```json
{"action":"delete_artifacts","kind":"source","source_manifest":"/path/to/source/manifest.json"}
```

其他清理类型：

- `delete_artifacts`、`kind:"compiled"`、`manifest`：删除当前 manifest 引用的编译分区。
- `clear_compiled`、`cache`：清理有 TurboCider 所有权标记的托管编译缓存。
- `clear_runtime`：清理上述 TurboCider 进程的设备专用缓存。

预览返回 `entries`、`bytes`、`plan_token`；确认后重发相同请求，加 `"apply":true` 和原 `plan_token`。条目变化导致 token 失效，必须重新预览。仅返回预览不会删除任何内容。源/编译分区删除只针对 manifest 列出的目录及 manifest 自身，保留无关文件和原始权重。外部分区可能由其他工具共用，App 预览会明确列出路径。

C ABI 新增 `tc_coreml_resources_json`、`tc_coreml_resources_cancel`；Swift 为 `NativeEngine.coreMLResources` 和取消接口。不要求先创建 FLUX 模型会话，因此权重未加载也能管理磁盘。资源任务遵守本系统设备作业锁；API 调用者必须先卸载将被清理资源的会话，App 在应用删除前会完成卸载。

## App 页面

模型页新增“Core ML 模型与磁盘空间”：显示实际占用、分层明细、路径定位、当前源/编译分区位置；提供配置选择、离线导出、预编译，以及三类删除预览。清理不会因目录位于外部而静默忽略，也不会把外部目录自动搬进 App 制造另一份副本。

## 本次验收

- 本机 safetensors 完整导出 20 分区约 52 秒；编译约 2.6 秒。它们是一次构建观察，不是承诺的所有设备耗时。
- 新编译分区真实执行 256×256、4 步、seed 42，与原分区生成 PNG 完全一致。这验证当前配方/权重/桶，不宣称所有新桶的性能和质量都已验收。
- 20 个内容缓存全部复用命中。
- 外部源包/编译大小可见；删除预览、错误 token、预览后文件变化、safetensors 保护均经过测试。
- 本轮测试生成的源包与编译缓存已通过新接口清理，原始 safetensors、原分区和未知用户文件保留。
- 导出取消测试确认子进程退出且没有 manifest。

原始产物和诊断在本机 `outputs/coreml-storage-validation/`；可提交证据归一化个人路径。复现真实转换与编译后运行 `tests/native/test_coreml_resources.py --model "$MODEL" --legacy-manifest "$LEGACY_MANIFEST" --output "$OUTPUT"`，输出目录须含本轮 `export.json`、`compile-result.json`、`generated/`、`cache/`；测试会清理本轮生成的两类产物。

证据：[真实转换、推理与清理验收](validation/coreml-storage-validation.json)、[可访问用户进程缓存检查](validation/coreml-process-cache-audit.json)。App 真机界面已显示 3.4 GB（十进制，约 3.2 GiB），并验证 1.7 GB 源模型的路径删除预览和取消；未通过界面删除用户原有分区。
