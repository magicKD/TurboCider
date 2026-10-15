# dev-verify：Qwen Image 2.1 GPU 与 App 验证（2026-09-30）

本轮先将本地 `dev` 合并到 `dev-verify`，合并提交为 `d3a2a87`。
唯一冲突位于 Z-Image 模型模块，保留 dev 的混合执行校验及正确的默认九步说明。
随后验证已有 Qwen Image 2.1 权重、已有 Viggle r128 六步 LoRA，修复原生内存生命周期和 App 交互问题。

## 环境与模型来源

- Apple M4 Pro，20 GPU 核心，48 GiB 统一内存；macOS 26.6，MLX 0.32.0。
- 原生 C++ / Metal，BF16 基础权重，执行选择为 `gpu`；参考图和文本编码也使用 GPU。
- 使用本地 `<Qwen-Image-2.1>` 中已有 DiT、Qwen3-VL 和 VAE，未下载其它模型或适配器权重。
- 本地缺少正确 tokenizer；只补齐同一模型官方 processor 的 tokenizer 元数据，固定 Qwen revision `0501a7e`。该文件不是额外模型权重。
- 使用已有 `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors`，SHA-256 为 `bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。454 个 A/B 张量匹配 227 个 Transformer 投影，强度为 1。
- 构建使用 `DEVELOPER_DIR=/Library/Developer/CommandLineTools`，避免本机 Xcode 命令行组件版本不匹配。未改系统工具链设置。

r128 是官方六步蒸馏适配器；它与基础模型的 40 步采样不同，不能把速度差说成无损内核加速。来源：[Viggle 模型说明](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)、[Qwen processor](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/main/processor)。

## 工作负载与测量口径

输入图片和输出画布均为 512×512。参考图遵循原生默认的 1024 编码尺寸，每张产生 4096 个参考 token；这不是把输出画布扩大到 1024。
固定种子 42，关闭 PE 和所有可选近似内核／ANE 诊断。LoRA 本身显式允许其六步近似采样。

每个工作负载在独立进程中连续执行三次；首轮包含文本／参考图编码，后两轮复用完整 conditioning。
检查真实步数、参考 token 数、GPU 执行回执、LoRA 投影覆盖、可解码非空 RGBA PNG，以及同种子三次 PNG 的 SHA-256 一致。

文生图使用 `resident`；编辑使用 `component_staged`，两者都由 GPU 执行神经网络。
这台 48 GiB Mac 的原生保守规划要求单参考图 resident 编辑至少 49 GiB（含系统余量），因此保留内存门槛，使用分阶段加载完成编辑验证。

系统文件缓存没有清空，桌面应用保持正常运行；部分样本与 Swift 编译重叠。时延是本轮端到端样本，不能当成隔离环境的稳定硬件上限。
MLX 峰值和 Darwin 进程 phys footprint 是不同口径；系统 VM／swap 计数不归因于单个进程。

| 工作负载 | 基础 40 步：首轮 / 后两轮中位数 | r128 六步：首轮 / 后两轮中位数 | 状态 |
| --- | --- | --- | --- |
| 文生图 | 115.05 / 111.60 秒 | 25.64 / 18.78 秒 | 已通过；内存修复前收集的 resident 基准 |
| 单参考编辑 | 142.24 / 126.65 秒 | 43.54 / 34.53 秒 | 已通过 |
| 两参考合成 | 174.48 / 158.82 秒 | 65.03 / 46.94 秒 | 已通过 |
| 三参考合成 | 206.72 / 190.53 秒 | 79.24 / 60.84 秒 | 已通过 |


| 工作负载 | 基础 MLX 峰值（GB） | r128 MLX 峰值（GB） | staged 完成后的 active（字节） |
| --- | --- | --- | --- |
| 文生图 resident | 20.620 | 21.971 | 常驻 DiT 约 14.9 GB；卸载另行检查 |
| 单参考编辑 | 18.938 | 20.663 | 983,562 |
| 两参考合成 | 21.462 | 24.597 | 1,704,458 |
| 三参考合成 | 24.252 | 28.359 | 2,359,818 |

GB 为十进制字节；三次 staged 请求的 idle active 相同。LoRA 的低步数改善速度，但本轮测得峰值高于基础模型，不能宣传为降低峰值内存。

## 原生修复与实机证据

识别并校验官方 r128 与既有 r256 的独立文件名／摘要，结果回执显示实际 rank，避免把 r128 标作 r256。

分阶段编辑现在也复用已完全求值的文本和参考 latents；缓存身份包含提示词、编码尺寸及有序图片内容摘要。
保留完整 conditioning，不保留 DiT 权重；取消后清理 staged 权重和适配器绑定。

MLX 0.32 的编译块内多输出 split 导致本轮真实模型请求结束后活跃分配逐轮累积，额外同步、allocator clear、engine free 和 compile cache clear 均不能释放。
将 Qwen RoPE 与 gate/up 内部 split 改为等价的单输出切片，保留编译、精度和算术顺序。
另在解码前销毁 request DiT，在清权重后同步并清 allocator cache，兑现分阶段释放。
这是针对本地实测的应用层绕行；上游相关记录：[MLX #3932](https://github.com/ml-explore/mlx/issues/3932)。

真实 r128 单图编辑三次实测，修复前 idle active 为 672.1 / 1410.3 / 2081.4 MB，engine free 后仍 2080.4 MB。
修复后每次为 983,562 字节，engine free 后 522 字节；三张 PNG 均与修复前同种子输出逐字节一致。
基础单图编辑三次 idle active 也均为 983,562 字节，MLX 峰值均约 18.94 GB。
不把 MLX 小 active 等同于整个进程仅占用这些内存。

新增真实小型 Transformer 回归覆盖 BF16、编译 RoPE、fused gate/up、runtime LoRA 和 0–3 refs。
24 轮重新分配权重、创建和销毁对象，验证 prefill／decode 输出一致、idle active 不累积，不依赖清 compile cache。

真实完整权重的六步 LoRA staged Session 已通过缓存与取消生命周期：重复 conditioning 命中；交换两张参考图、恢复其顺序、原路径覆盖图片并保留 mtime 均触发 miss；覆盖后的下一次重复重新命中。该覆盖测试未固定图片文件大小，不作为“大小与 mtime 同时不变”的独立证据。
在 denoise 第一步取消没有导出 PNG，取消后的 MLX active 为 1,704,458 字节；重试重新加载 DiT／VAE、复用完整 conditioning，PNG 与取消前相同条件的输出 SHA 一致。LoRA → 基础 → LoRA 的 prepare 覆盖为 227 → 0 → 227；显式卸载后 active 为 522 字节，再次 prepare 为 cache miss。

最终 resident 六步 LoRA 文生图 Session 也通过 prepare／warmup／缓存重复／取消无导出／适配器重新绑定／卸载检查，卸载 active 同为 522 字节。最终 `run-0.png` 的 SHA 与内存修复前的 resident r128 基准一致；本次热请求为 15.44 秒，原表保留原批次的 18.78 秒中位数，不将两种缓存预热和桌面负载下的样本合并。

## App 交互修复

- Qwen 的文生图／图片编辑入口按实际支持的操作显示，单图、结果编辑及历史复用保留 Qwen。
- 参考图显示编号与参与顺序，支持移除、重排和撤销；六步 LoRA 的三个输入限制直接反映在按钮和计数上。
- 已识别适配器提供 512×512／六步／强度 1／GPU 预设；错误尺寸、步数、角色、强度或多个适配器在提交前显示原因。
- 提交准备阶段锁定草稿入口及快捷键；LoRA 行按 UUID 绑定，避免删除后编辑错行。
- 小窗口中输入区滚动，主预览保留高度，生成按钮及画布信息常驻；透明棋盘作为图片背景。
- 标注读取在后台执行，按原图比例绘制；独立黑白蒙版和副本保留原图，支持撤销。蒙版作为有序视觉参考，不承诺逐像素保护。
- 超分后的结果查看、参数复用和配置导入清除残留的单独超分界面状态。
- 失败／中断的参数恢复使用同一个工作区恢复入口；详细状态区限制高度，长错误可滚动，恢复操作位于顶行，避免小窗口裁切。
- 模型库文件探测、迁移和 LoRA 目录扫描移到后台，避免文件提供器／系统权限等待冻结主线程；取消后不发布过期扫描结果。

实机已回放参考图重排／移除及撤销、仅编辑一张、椭圆标注／黑白蒙版及恢复、LoRA 开关与六步预设、40 步恢复提示，以及小窗口预览／滚动／固定生成按钮。
最终界面另已回放“单独超分 → 历史复用”、“单独超分 → 素材查看 → 编辑此图”和“单独超分 → 导入配置”，均正确恢复 Qwen 编辑工作区、GPU、六步 LoRA 和对应参考图。
这些界面检查使用隔离 App 与真实 native 输出构成的 UI-only 历史夹具；不计作 App transport 推理证据。最后一版在 980×752 点窗口中，结果图约 148×148 点，画布／种子／路线合并为一行，输入滚动到提示词及底部示例菜单时生成按钮仍完整可见。

同一个最小窗口另用 21 行合成错误验证状态区：恢复按钮与生成入口可见，滚动能读到最后一行，恢复后保持 Qwen／六步／r128／单参考配置。该合成错误仅为 UI 夹具。
随后在最终打包的真实 App 界面直接点击生成、取消和重试：采样阶段显示 Metal GPU · BF16，取消后状态为 cancelled 且无输出；草稿和原图保留。重试 34.81 秒完成单图编辑，实际六步、227 个投影、4096 参考 token、缓存命中、MLX active 983,562 字节，PNG SHA 与 native 矩阵相同条件的单图输出一致。最终结果图、历史缩略图和“图像权重已释放”状态均正确显示。

按用户后续要求，迭代验收优先使用已有六步 LoRA 的四个工作流；新增 `--lora-only` 快速验收入口。
真实 `StudioState → NativeJobStore → native worker` 的四组 LoRA 验收均通过，独立输出目录保留回执和实际生成图片。检查输入副本字节及顺序、GPU／BF16／六步／227 个 LoRA 投影、完整 denoise 进度、运行至成功状态、PNG 解码、任务保存／历史复用及卸载。
四个 App 首次请求分别为 24.12 / 40.81 / 57.72 / 84.12 秒；均采用分阶段加载和各自的新 conditioning，不能与上表复用 conditioning 的 native 热请求混为同一个测量。
原先启动的完整 App 批次在基础文生图和基础单图成功后，根据用户“后续以 LoRA 为主”的要求停止；不把该未完成批次记为八组 App 通过。

## 验证与质量边界

已完成完整 native 构建、Swift 构建及 `make test`、`make test-app`；最新 native 修复后的 `make test-qwen21` 通过 41 + 33 个 Python 检查及 Swift 合约回归，包含真实 Transformer 生命周期测试。可选 PE／外部证据组按其条件跳过，不计为实机通过。
最终 `make test`、`make test-app` 和 `make test-library` 均退出 0。`make test` 的 46 个 unittest 批次共 439 通过、21 按条件跳过，另有原生断言通过；跳过包含 15 个未启用 GPU opt-in、3 个缺本地 Gemma／Wan fixture、3 个缺指定 test-hook／release／audit 库。`test-app` 的 11 个执行目标和 `test-library` 的 4 个执行目标均成功。模型库下载合约的网络只访问微型本地夹具，未下载完整模型。真实 staged 与 resident Session 的取消／恢复／缓存及卸载检查通过。最后一轮预览布局调整后的 App 打包通过发布二进制门槛与严格 ad-hoc 签名校验，最低 macOS 为 26.2。

已打包 CLI 的 `doctor` 与 `self-test` 退出 0：识别 M4 Pro GPU，原生 Metal BF16 Euler／layer norm 等自测通过，运行时不依赖 Python。

人工检查目前确认：基础和六步文生图输出茶壶；单图将蓝色改为红色；双图基础结果包含可辨认茶壶和狐狸。
单图“哑光”要求没有严格实现，背景／桌面纹理也会变化。合成能运行、消耗全部参考 token，并不等于所有语义、位置和细节均被严格保留。
基础与六步的双图／三图均组合出可辨认的参考主体，三图包含海鹦。双图茶壶手柄／壶嘴方向会镜像；三图基础输出左侧壶嘴被裁切。六步与基础输出的姿态和材质也不同，均不能认证严格身份、哑光要求或逐像素背景保留。
本地 `outputs/qwen21-gpu-acceptance-20260930/comparison.png` 保存三个参考图与八种实际输出的对比。

另已人工检查四张真实 App LoRA 输出：文生图为木桌上的陶瓷茶壶，单图为红色茶壶，双图包含茶壶与狐狸，三图包含茶壶、狐狸与海鹦。App 的三图通用合成提示词还产生了第二只海鹦；单图仍有反光，未严格满足哑光要求。四组 transport 成功不认证主体数量、方向及背景细节完全符合指令；原始 acceptance 回执保持 `visual_quality_accepted=false`。

本轮不认证 2K、十张参考、任意 LoRA、GPU + ANE 或所有错误／异常退出组合，也不作“全部场景无 bug”的保证。
原始请求、每轮回执、图片、日志和 UI QA 笔记保存在被 Git 忽略的本地 `outputs/`，不提交权重、生成媒体或私有绝对模型路径。

## 后续迭代的快速验收

默认以现有六步 r128 LoRA 的文生图和 1–3 参考编辑为主，不为每个 UI 修复重复跑 40 步基础模型。
完整基础矩阵保留本轮实测结果；仅在基础采样、Transformer／VAE、精度或内存路线发生变化时再跑相关基础样本。

```sh
build/native/turbocider-qwen21-gpu-acceptance \
  "$MODEL" "$LORA" "$REF1" "$REF2" "$REF3" "$FRESH_OUTPUT" --lora-only
```

需使用新的输出目录。默认不传选项仍支持八组完整验收，`--quick` 保留原来的两个混合样本，`--lora-only` 固定六步 LoRA 的四组工作流。
这不是下载入口，模型与适配器均由本地路径传入。
