# 2026-10-12 ComfyUI / Comfy Desktop API 接入验收

在 `dev-comfyui` 实现标准库 HTTP 网关、五个 ComfyUI 节点、安装器、一键服务启动器、工作流生成器、发布打包和自动验证工具。ComfyUI 只处理图像张量与编排，模型推理由 TurboCider 原生进程执行。

## 环境与边界

- Apple M4 Pro，48 GiB，macOS 26.6；原生 MLX 0.32.0。
- ComfyUI 官方源码：`e8665ca8e4089ab917567cdb491308e1397663d4`；frontend `1.55.14`；Python `3.12.7`；PyTorch `2.14.1`。
- Comfy Desktop `1.1.6`，新增 `TurboCider Local` 本地连接，指向 `127.0.0.1:8188`。
- TurboCider HTTP：`127.0.0.1:8765`，转发原生 Unix API。
- ComfyUI 使用 `--cpu --offline`；设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- 下载了官方 ComfyUI 软件和 Python 依赖；没有下载、复制或转换任何模型。ComfyUI 的 models 目录没有 safetensors/ckpt/pt/pth/gguf 权重。
- 只使用已注册 FLUX.2 Klein 4B、Z-Image Turbo、Qwen Image 2.1 与已有 LoRA；未更改模型注册库。
- 仅原生 `models` 宣告可执行并已注册的模型进入节点选择器。App 专有放大器、编辑器功能不自动成为原生 API executor。

## 真实 ComfyUI 队列

以下均经过 `/prompt` → TurboCider 节点 → HTTP → 原生服务 → 图像回传 → ComfyUI `SaveImage`。全部使用 512×512，seed 42，批量第二张 seed 43。耗时是单次工作流 wall time；期间存在构建/回归负载，不能作为性能基准。

| 工作流 | 秒 | 结果 |
|---|---:|---|
| flux-batch | 14.771 | success |
| flux-edit | 12.222 | success |
| flux-transform | 5.195 | success |
| z-base | 21.274 | success |
| z-lora-runtime | 27.504 | success |
| z-lora-merge | 24.45 | success |
| z-base-after-lora | 22.265 | success |
| qwen-base | 56.459 | success |
| qwen-turbo | 31.424 | success |
| qwen-edit | 29.47 | success |
| qwen-multi-edit | 34.486 | success |
| qwen-ordinary-lora | 65.66 | success |
| qwen-base-after-lora | 62.494 | success |
| qwen-ordinary-edit | 67.671 | success |

Z-Image 使用已有 distill patch adapter，分别测 `inference_time` 与 `in_memory_merge`。Qwen 测 Viggle v0.2.1 r128，strength 1 / 6 步，以及已有普通编辑 LoRA，strength 0.6 / 20 步。单图、双图编辑验证图像上传、参考顺序与输出解码；普通 LoRA 另测参考图编辑。

基础模型在 LoRA 前后使用相同提示词/seed：Z-Image、Qwen 两组均逐像素完全一致（最大差值 0），没有观察到 adapter 残留。PNG 文件哈希因 ComfyUI 元数据不同而不同，不应以文件字节差异判断模型残留。

目视检查：FLUX 编辑和 Qwen 单图编辑把蓝茶壶改成红色；Qwen 双参考图输出包含两只茶壶；生成结果可正常解码、显示、保存。此处是接入回归，不是所有提示词或实验性 Qwen 编辑质量的认证。

## Desktop、生命周期、发布包

- 在真实 Comfy Desktop 分别导入 FLUX 工作流和 Qwen Viggle LoRA 工作流，点击 Run，两次均看到 512×512 预览，Desktop 输出下载完成，队列回执 success。Qwen 使用最终标准 dist 运行库及一键启动器。
- `interrupt`、1 秒超时各实测取消一个正在运行的原生任务；batch_size=2 没有继续提交第二张。取消后 `Native Request` 成功生成并写入 ComfyUI 输出目录。
- 排队任务取消成功；不匹配 LoRA、未知字段、超范围 seed、不合法尺寸均明确报错。
- HTTP 网关重启后仍可从原生持久历史查询最初任务并读取原图。不维护第二份易失配的任务状态。
- 发布包单独启动成功；ComfyUI 对发布包 API 完成 FLUX 两张批量生成。标准 `dist/cli`、`dist/TurboCider.app` 和 `dist/comfyui` 已生成。
- 最终一键启动器同时启动原生 API、HTTP 与本地 ComfyUI；Desktop 原连接可以重连使用。

## 回归与构建

- `make test-comfyui`：7 个真实 HTTP fixture 测试 + 7 个实际 CPU tensor 节点测试通过。
- `make test-api`：已有 API、客户端、工作流、图片准备与安装发现检查通过。
- 仓库 layout 8 项通过；Python 编译与 `git diff --check` 通过。
- `make test` 首次在已有 GGUF/ANE 测试停止。随后 `make -i test` 跑完其余检查，保留失败记录，不能将其 exit 0 当成全绿。
- 尚有两项与本次 Python 接入无关的原有问题：`test_cli_ane_override.py` 仍预期拒绝当前运行库已允许的 GGUF runtime LoRA；`test_mlx_weight_pager.py` 的单独链接命令缺少 `tc::ane::observe_runtime_memory` 定义。
- `test_inventory.py` 首次与运行中的测试服务争用 GPU lease；停止服务后两项复验通过。
- 默认 Xcode 环境的 SDK 查询因 CoreDevice/Mercury `_XPCTypeBool` 符号缺失失败。显式 `DEVELOPER_DIR=/Library/Developer/CommandLineTools`，隔离完整构建并打包成功；没有修改系统 Xcode 设置。

## 使用与证据

[使用说明](../public/COMFYUI.md)。本机运行入口为 `outputs/comfyui-validation/start-local.command`；当前服务正在运行，不要重复启动。后续停止服务后可双击该文件重新启动，再打开 Desktop 的 `TurboCider Local`。

原始证据保存在被 Git 忽略的 `outputs/comfyui-validation/`：

- `e2e/summary.json`、各 case 的 prompt/history JSON 和 PNG；`contact-sheet.png`；
- `desktop-history.json`；`lifecycle/`；`negative-tests.json`；
- `packaged-e2e/`、`packaged-start.log`、`final-package.log`；
- `final-comfyui-tests.log`、`test-api.log`、`make-test-all.log`、`inventory-recheck.log`；
- `workflows/` 含可导入的本机模型/LoRA 示例。示例记录注册 ID，不包含权重。

视频原生请求可通过 Advanced Request 节点提交并取回文件，但本轮没有注册的视频模型用于真实生成验收。未宣称任意 ComfyUI KSampler/ControlNet/latent 工作流都能替换为远端模型；本方案以原生模型支持的操作和图像接口组合工作流。
