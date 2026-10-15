# 在 ComfyUI / Comfy Desktop 中使用 TurboCider

本教程适用于 `dev-comfyui` 分支及包含 `comfyui` 配套目录的发行包。模型和 LoRA 保留在 TurboCider 模型库中，由 TurboCider 加载与推理；ComfyUI 负责工作流、参考图、预览和保存，不需要安装任何模型权重。

## 1. App 里面有接口吗？

**有。TurboCider App 的「本地 API」页面已经集成原生服务的启动、停止、连接信息及任务结果查看。** 页面提供「启动服务」「复制 Socket 路径」「复制 AI 接入说明」「打开日志目录」等操作。

点击 App 的 **启动服务**会同时启动 **原生 API 和内置 HTTP 网关**，默认地址为 `http://127.0.0.1:8765`。页面可修改端口、复制 HTTP 地址、展开并复制 curl / Python 示例。App 网关是原生实现，不需要 Python 环境，也无需手动运行网关脚本。

ComfyUI 仍需安装一次 TurboCider 自定义节点；节点安装器和工作流工具位于同版本发行包的 `comfyui` 目录，或本分支源码中。原生 Unix socket 接口同时保留。

```mermaid
flowchart LR
    D[Comfy Desktop 或浏览器] -->|8188| C[ComfyUI 工作流和 TurboCider 节点]
    C -->|HTTP 8765| H[TurboCider HTTP 网关]
    H -->|Unix socket| T[TurboCider 原生 API]
    T --> M[已有模型和 LoRA · Metal 推理]
    T -->|结果图片| H
    H -->|IMAGE / MASK| C
```

两个地址的用途不同：

| 填写位置 | 默认地址 | 用途 |
|---|---|---|
| Comfy Desktop 的 Remote Connection | `http://127.0.0.1:8188` | 打开本机 ComfyUI 工作流服务 |
| TurboCider · Model (API) 节点的 `api_url` | `http://127.0.0.1:8765` | 请求 TurboCider 模型服务 |
| App 的 Socket | 以 App 页面显示为准 | 高级本机客户端直接调用原生 RPC |

本教程中的服务均运行在同一台 Mac。Comfy Cloud 无法访问这台 Mac 的 `127.0.0.1`；Desktop 应选择本地的 Remote Connection。

## 2. 准备工作

需要已经可运行的 TurboCider、已登记的本机模型，以及已安装依赖的 ComfyUI。ComfyUI 的目录应包含 `main.py` 和 `custom_nodes`。Comfy Desktop 是界面客户端；此方案仍需要一个本机 ComfyUI 服务。

- 在 TurboCider 模型库中登记已有模型和 LoRA，保留原来的文件位置。不必复制到 ComfyUI 的 `models` 目录。
- 使用本分支打包的 CLI，默认位置为 `dist/cli/turbocider`。节点安装器不会构建原生运行库，也不会安装 ComfyUI 或 Python 依赖。
- 下文命令默认在 **TurboCider 仓库根目录**执行。`/path/to/ComfyUI` 和 `/path/to/comfy/python` 是占位符，分别替换为实际 ComfyUI 源码目录及其 Python 可执行文件；含空格的路径保留引号。
- 使用发行包时，先进入配套 `comfyui` 目录再运行相同的 `tools/comfyui/...` 命令。启动器会寻找同级 `../cli/turbocider`，请保留发行包目录结构。

### 已完成本机验收的环境可以直接使用

本次验收已经配置了 Comfy Desktop 的 **TurboCider Local** 连接，地址为 `http://127.0.0.1:8188`，并生成了可导入工作流：

```text
outputs/comfyui-validation/workflows/
```

该目录包含 FLUX.2 Klein 4B、Z-Image Turbo、Qwen Image 2.1 基础工作流，以及本机已有的 Z-Image / Qwen LoRA 示例。文件名带模型登记 ID，因此不需要手工填写 ID。

如果本机服务已经运行，直接进入第 5 节。服务停止后，在仓库根目录执行：

```sh
./outputs/comfyui-validation/start-local.command
```

此脚本和上述工作流是本次验收生成的本机文件，不随源码分发。换一台机器时，按下面的通用步骤配置。

## 3. 安装 TurboCider 节点

对实际运行的 ComfyUI 安装一次：

```sh
python3 tools/comfyui/install.py "/path/to/ComfyUI"
```

安装结果位于 `ComfyUI/custom_nodes/turbocider/`。安装器只复制 Python 节点及客户端代码，不运行模型下载器，也不复制权重。安装或更新后，需要重启 **ComfyUI 服务进程**；仅关闭 Desktop 窗口不一定会重启服务。

## 4. 启动服务：选择一种方式

### 方式 A：独立后台运行，适合不打开 App 的场景

一次启动 TurboCider 原生 API、HTTP 网关和现有 ComfyUI：

```sh
python3 tools/comfyui/start.py \
  --comfyui "/path/to/ComfyUI" \
  --comfy-python "/path/to/comfy/python"
```

若 ComfyUI 使用自身 `.venv/bin/python`，可以省略 `--comfy-python`。启动器会在终端持续运行；等待 ComfyUI 就绪后，连接 `8188`。不要对已经运行的同一套服务重复启动。

这条命令让 ComfyUI 以 `--cpu --offline` 运行，并设置模型库的离线环境变量。**CPU 选项只作用于 ComfyUI，TurboCider 仍通过 Metal 执行模型推理。** 本教程使用的节点不要求 ComfyUI 加载 checkpoint、CLIP、VAE 或 LoRA。

如果已经有可用的 ComfyUI 服务，仅启动 TurboCider 两层服务：

```sh
python3 tools/comfyui/start.py
```

随后重启已安装节点的 ComfyUI，让模型下拉列表重新获取 API 目录。

独立方式由 CLI 持有原生服务，不需要打开 TurboCider App 的「本地 API」。退出 App 不会停止这套服务；在启动终端按 **Ctrl-C** 会停止该启动器创建的进程，生成结果保留。不要同时在 App 创作页执行另一份模型推理，以免争用 GPU 和内存。

### 方式 B：由 TurboCider App 持有原生服务

希望直接从 App 启停服务、查看任务和结果时，采用此方式。先用 Ctrl-C 停止方式 A，避免占用相同端口，然后：

1. 打开更新后的 TurboCider App，进入 **本地 API**。
2. 保持 **HTTP 端口**为 `8765`，或填写未被占用的端口。端口会在成功启动后保存。
3. 点击 **启动服务**。App 会释放创作页的模型会话，同时启动原生 API 与 HTTP 网关。
4. 等待显示 **HTTP 网关与原生服务已启动**，点击 **复制 HTTP 地址**。若端口冲突，App 会提示错误并停止本次创建的原生服务；停止旧服务或修改端口后重试。
5. 展开 **HTTP 调用方法**中的示例，可以直接复制服务检查、模型查询、参数校验、任务提交和结果下载命令。
6. 在终端启动已经安装节点的 ComfyUI；如果它已正常运行，刷新节点定义或重启该服务即可：

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TURBOCIDER_API_URL=http://127.0.0.1:8765 \
  "/path/to/comfy/python" "/path/to/ComfyUI/main.py" \
  --cpu --offline --listen 127.0.0.1 --port 8188
```

此方式**不再运行 `start.py` 或 `serve.py`**，因为 App 已经启动原生服务和 HTTP 网关。若修改端口，还需同步修改上面的 `TURBOCIDER_API_URL` 及 Model 节点的 `api_url`。上面的 `--offline` 适用于验收版本；较旧 ComfyUI 若不识别该选项，可使用其 `--disable-api-nodes` 选项。

执行工作流后，可在 App 的 **API 任务与结果**中刷新、预览图片、在 Finder 中定位，或导入为创作参考。API 历史与创作页历史分开保存。

保持 App 打开；点击 **停止服务**或退出 App，会关闭网关并停止它创建的原生 API。要回到 App 创作页生成，先等待或取消 API 任务，再停止服务。ComfyUI 有独立生命周期，在其终端用 Ctrl-C 停止。

## 5. 在 Comfy Desktop 打开并运行第一个工作流

1. 在 Comfy Desktop 添加 **Remote Connection**，名称可填 `TurboCider Local`，URL 填 `http://127.0.0.1:8188`；已有该连接时直接选择它。也可以用浏览器打开同一地址。
2. 保持 API 运行，在仓库根目录生成适配当前模型登记信息的工作流：

```sh
python3 tools/comfyui/workflows.py \
  --output-dir outputs/comfyui-workflows
```

3. 在 Desktop / ComfyUI 按 **Cmd-O**，打开生成目录内以 `flux2-klein-4b-` 开头的 JSON；如果没有此模型，选择本机已登记的 Z-Image 或 Qwen 工作流。
4. 检查 Model 节点的 `api_url` 为 `http://127.0.0.1:8765`，模型下拉框有实际登记项。
5. 点击 **Run / 运行**，等待 `Save Image` 显示图片。首次运行会加载模型，后续任务可以复用原生会话。

工作流生成器只读取现有登记信息；没有安装的模型不会生成示例，部分已登记的 Viggle / distill patch LoRA 会额外生成带 LoRA 的示例。普通 LoRA 可按第 7 节手动添加。

### 手动搭建最小工作流

双击画布，搜索 `TurboCider`，添加并连接：

```text
TurboCider · Model (API) 的 backend
    → TurboCider · Generate Image 的 backend

TurboCider · Generate Image 的 images
    → Save Image 的 images
```

在 Model 中选择已登记模型。在 Generate Image 中先使用：

```text
prompt: A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.
width: 512
height: 512
seed: 42（control_after_generate 选择 fixed，便于复现）
batch_size: 1
advanced: {}
timeout_seconds: 1800
```

`steps` 按下面的模型表设置。模型库新增或修改登记后，刷新节点定义；如果下拉框仍旧，重启 ComfyUI 后重新选择。

## 6. 三种已测试模型的设置

| 模型 / 组合 | 建议首次运行 | 能力和注意事项 |
|---|---|---|
| FLUX.2 Klein 4B | 512×512，4 步 | 文生图、参考图编辑、图生图 transform |
| Z-Image Turbo | 512×512，8 步 | 文生图；当前原生执行器不提供参考图编辑 |
| Qwen Image 2.1 基础模型 | 512×512，20 步 | 文生图、参考图编辑；普通用法可选 20–40 步 |
| Qwen Image 2.1 + 普通 LoRA | 512×512，20 步，按 LoRA 设置强度 | 普通 LoRA 不自动变成六步加速模型 |
| Qwen Image 2.1 + 已支持的 Viggle v0.2.1 LoRA | 512×512，**6 步**，强度 **1** | `advanced` 填 `{"allow_approximation":true}`，需匹配原生支持的适配器身份 |

`steps=0` 表示使用后端模型默认值，不会根据选择的 LoRA 自动改为 6 步。分辨率输入框的范围也不代表每个模型支持所有尺寸；最终由 TurboCider `plan` 校验模型约束。

这三类模型的生成、支持的编辑操作及已有 LoRA 已经过真实 ComfyUI 队列测试。FLUX 和 Qwen Viggle 工作流还在本机 Comfy Desktop 中完成了导入、运行、预览与保存。这里验证的是接入链路及具体样例，不代表每个提示词、模型组合或编辑质量都已覆盖。

## 7. 使用已有 LoRA

在 Model 与 Generate / Edit 之间插入 **TurboCider · LoRA (API)**：

```text
Model (API) → LoRA (API) → Generate Image → Save Image
```

1. 从 `lora` 下拉框选择已在 TurboCider 登记的适配器。它必须属于当前模型家族。
2. `strength` 先采用适配器要求值。普通 LoRA 本次测试使用过 `0.6`；Viggle 六步路线必须为 `1`。
3. `strategy` 首先选择 `auto`。FLUX 使用原生支持的内存合并；Z-Image 已验证 `inference_time` 和 `in_memory_merge` 两种方式。不同模型支持的策略不同，不必强行统一。
4. 对 Qwen Viggle，显式设置 6 步及 `{"allow_approximation":true}`；普通 Qwen LoRA 使用正常步数。

LoRA 文件由 TurboCider 读取。无需 ComfyUI 的 `Load LoRA` 节点，也无需往它的模型目录放权重。可以串联多个 API LoRA 节点表达有序适配器列表，但具体数量、组合及策略仍须通过原生校验；串联能力不代表所有组合都已测试。

要恢复基础模型，将 Model 直接接回生成节点。本次 Z-Image 与 Qwen 的测试中，使用 LoRA 前后的同 seed 基础模型结果逐像素一致，未观察到 LoRA 残留。

## 8. 参考图编辑、图生图和批量

### 一张参考图编辑

添加标准 **Load Image** 和 **TurboCider · Edit / Transform Image**：

```text
Model (API) ─────────────────→ Edit / Transform Image 的 backend
Load Image 的 IMAGE ────────→ Edit / Transform Image 的 images
Edit / Transform Image 的 images → Save Image
```

选择 Qwen Image 2.1 或 FLUX.2 Klein 4B，将 `operation` 设为 `image.edit`，提示词例如：

```text
Change the blue teapot to red, keep the table and lighting unchanged.
```

先用 512×512、`batch_size=1`，步数遵循上一节。如果用已支持的 Qwen Viggle r128 进行小尺寸参考图编码，可填写：

```json
{"allow_approximation":true,"qwen21_reference_size":512}
```

这个选项用于相应的 512×512、1–3 张参考图路线，不能推断其他 LoRA 都支持它。Qwen 原生实验性编辑路线仍有质量限制。

### 图生图 transform

选择 FLUX.2 Klein 4B，`operation=image.transform`，例如 `strength=0.65`。`strength` 用于 transform 的初始图像；选择 `image.edit` 时该控件不作为编辑强度传给后端。

### 多参考图与多张输出

- 将两张同尺寸图片通过标准 **Image Batch** 节点合并后接入 Edit 的 `images`，会成为**一次编辑任务的有序参考图**，顺序对应图 1、图 2。本次验证包含 Qwen 双参考图编辑。
- `batch_size=2` 表示生成两张输出，按顺序提交原生任务，seed 为当前值及下一值；达到 `2147483647` 后回绕。
- 输入 IMAGE batch 不表示逐图独立编辑。需要分别编辑每张图时，应在 ComfyUI 工作流中显式迭代。

输出是普通 `IMAGE`，可接入标准预览、保存、缩放、拼接、合成等节点。若输出含透明度，连接 `images` 与 `alpha_mask` 到 **Join Image with Alpha** 后再保存；直接保存 RGB 不保留透明通道。MASK 使用 ComfyUI 的反向 alpha 约定，不等于模型自动完成语义抠图。

## 9. 支持范围与高级参数

这套节点在**图像边界**衔接 ComfyUI。采样、文本编码、VAE 和权重管理由 TurboCider 完成，API 的 `backend` 不能连接到标准 `KSampler` 的 `MODEL` 输入。

因此不承诺任意 ComfyUI checkpoint 工作流、ControlNet、硬蒙版重绘或采样器替换直接兼容。App 专有功能也不会自动成为 API 模型；列表只显示已登记且原生服务声明可执行的模型。

`advanced` 接收原生 schema-1 参数组成的 JSON 对象，例如：

```json
{"residency":"component_staged","qwen21_dit_cache":"off"}
```

模型、提示词、尺寸、seed、输出位置、参考图和 LoRA 已由节点管理，不能在 `advanced` 中覆盖。未知字段或不支持的组合会报错，避免静默忽略参数。

**Native Request / Video** 节点用于高级原生请求，返回本地文件路径和任务回执，而不是通用 IMAGE 张量。视频仅在有已登记且可执行的模型时可用；本次本机验收没有覆盖真实视频生成。完整接口见[英文接口参考](COMFYUI.md)。

## 10. 结果、取消与停止

- `Save Image` 按 ComfyUI 的输出配置保存；默认在其 `output` 目录，生成的示例使用 `TurboCider/模型名称` 前缀。
- 独立启动的 TurboCider 默认将 API 原始输出保留在 `state/comfyui/http/outputs`，原生历史在 `state/comfyui/native`。App 方式可点击 **打开日志目录**，在打开的 `api-service` 目录下查看 `http/outputs`；原生历史位于同目录的 `jobs`。
- Generate / Edit 的 `jobs` 输出包含原生任务回执；ComfyUI 服务终端也会打印任务 ID。诊断问题时保留这个 ID。
- 点击 ComfyUI 的停止 / Interrupt 会请求取消当前原生任务；后端在安全边界完成取消，因此未必瞬间结束。超时也会请求取消，未开始的批量后续任务不再提交。
- 仅关闭窗口或断开 HTTP 连接不等于取消任务。提交回复丢失时，先查看任务历史，不要立即重复运行。
- 状态、参考图和输出不会自动删除。需要清理时，先停止任务和对应服务，再处理自己不再需要的状态目录。

## 11. 排查连接和常见问题

在没有配置可选 Token 的默认环境中，终端执行：

```sh
curl --fail http://127.0.0.1:8765/v1/health
curl --fail http://127.0.0.1:8765/v1/catalog
curl --fail http://127.0.0.1:8765/v1/jobs
```

分别检查原生服务连接、已登记模型与 LoRA、近期任务。也可以打开 `http://127.0.0.1:8188` 检查 ComfyUI 界面。

| 现象 | 处理方法 |
|---|---|
| Desktop 连不上 | Remote Connection 填 `8188`，确认本机 ComfyUI 服务运行；不要填 `8765` 或选择 Comfy Cloud |
| 搜不到 TurboCider 节点 | 检查安装目标是否为实际启动的 ComfyUI；重启服务并查看 custom node import 日志；确认 Python 是 ComfyUI 的依赖环境 |
| 节点提示 connection refused | 先检查 `8765/v1/health`；App 方式还需确认两个服务已启动，并核对页面实际 HTTP 地址与节点地址一致 |
| 模型下拉框为空或显示 unavailable | 先启动 API，再刷新节点定义或重启 ComfyUI；用 catalog 检查该服务使用的模型库中是否有可执行的登记项 |
| 修改端口后列表仍取旧服务 | 在启动 ComfyUI **之前**设置 `TURBOCIDER_API_URL`；同时更新 Model 的 `api_url`。高级用法可显式填写 `installation_id` / `lora_id` |
| 找不到模型文件 | 在 TurboCider 中修复已有文件的登记路径；节点不会自动下载替代模型 |
| LoRA 家族不匹配或策略不支持 | 使用与模型对应的适配器，先选 `auto`；不要跨模型家族混用 |
| Qwen Viggle 报步数、强度或 approximation 错误 | 核对已支持的适配器身份、6 步、强度 1、`allow_approximation=true`；普通 LoRA 不套用此设置 |
| Z-Image 编辑报 unsupported operation | 当前只提供文生图，参考图编辑改用支持该操作的 FLUX / Qwen |
| 首次生成慢或超时 | 先查看任务进度和服务日志；首次含模型加载。必要时增大 `timeout_seconds`，避免超时后立即重复提交 |
| 端口占用、GPU lease 或内存不足 | 检查是否重复启动了两套服务或 App 同时推理；在对应终端停止自己启动的旧实例，保留一个工作栈 |
| 图片生成了但 App 创作历史没有 | API 历史独立；App 方式在「本地 API」查看，独立方式在 ComfyUI 输出和 API 历史查看 |

独立 Python 网关支持可选 Bearer Token：在该网关与 ComfyUI 进程环境中设置相同的 `TURBOCIDER_API_TOKEN`，Model 节点的 `token_env` 保留该变量名。工作流中无需保存 Token 明文。此时诊断请求也必须带对应 Authorization 头。App 内置网关目前不启用 Token 鉴权，仅提供可信本机访问。

HTTP 网关当前仅监听本机 IPv4 loopback，拒绝带浏览器 Origin 的请求；上面的模型 HTTP 调用由 ComfyUI Python 进程发起。它不提供跨机器或公网部署教程。更多 API 路由、限制及验证命令见[接口参考](COMFYUI.md)。
