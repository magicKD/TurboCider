# dev → dev-verify 与 Tiny VAE App 验收

日期：2026-10-27，Apple M4 Pro / 48 GiB，macOS 26.6，MLX 0.32.0。
`dev-verify` 从 `2a4c2b5` 快进合并到 `dev` 的 `082db0f`，无冲突。
本次所有真实图片任务的输出均为 **512×512**，使用本机已有模型、纯 GPU / BF16。
原始请求、生成图片和逐帧观察保留在本地 `outputs/dev-verify-20261027/`，不提交媒体或权重。
可移植逐请求数据见 [JSON](2026-10-27-dev-verify-tiny-preview.json)。

## 中间预览的问题与修复

本次发现两个独立问题。

1. 原实现直接把 Euler 更新后的带噪 latent 交给 Tiny decoder。Tiny decoder
   是 VAE 的近似解码器，本身不负责扩散去噪，导致中间步骤出现明显噪声、彩块。
   [TAESD 官方预览示例](https://github.com/madebyollin/taesd/blob/main/examples/Previewing_During_Image_Generation.ipynb)
   先取得预测的干净样本，再做预览解码。对于本项目的 flow matching，改为
   `x0 = x_sigma - sigma * velocity`，仅在需要预览时计算 FP32 副本。
   原采样更新、dtype 和舍入路径保持原样；最后一步使用精确的最终 latent。
2. App 旧预览使用独立的 `NSImage` 自适应布局，在纵向 ScrollView 中缺少有界
   viewport，可能按宽度撑高、裁切图片并把预览标签挤出可见区域。本次旧 PNG
   实际已经是 512×512，不能把显示异常归咎于 PNG 尺寸。现在实时帧与最终图
   共用 `InteractiveMediaPreview`，统一适应窗口、缩放、拖动和透明背景；
   解码新帧时保留上一帧，取消的解码任务不能发布旧画面。画面按可用画布等比
   适应窗口；运行状态栏会影响可用显示面积，PNG 像素尺寸始终不变。

同时取消原生预览对 latent 的最长边 512 抽样限制，完整解码请求分辨率；
原生发布端和 App 接收端均检查预览尺寸与最终请求一致。仍使用一个原子替换
侧车 PNG，预览不进入最终结果库，任务结束或取消后清理。

依据 [TAESD 上游架构与权重说明](https://github.com/madebyollin/taesd)，
Z-Image 使用 `taef1`，Qwen Image 2.1 使用 `taeqi2_1`；对应 8× RGB 与
16× RGBA 路径。核对了现有网络结构及归一化，无需更换或重复下载权重。
Tiny 编码和最终解码仍是有损近似，原始 VAE 保持默认。

## 实际 App 操作

通过打包后的原生 App 选择模型、设置参数、提交生图、观察预览、取消/重试和
继续编辑。观察脚本只读取 App 生成的任务记录和侧车，未代替 GUI 提交任务。
测试提示词为红色陶瓷茶壶、木桌和暖色阳光，固定 seed `20261027`。
Qwen 使用已有 `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors`，
强度 1、运行时加载，回执确认应用 227 个 projection。

修复后，最终图仍选择原始 VAE，每 2 步预览：

| 模型 | 步数 | App 总耗时 | 采样耗时 | 观察到的预览步骤 | 结果 |
| --- | ---: | ---: | ---: | --- | --- |
| Z-Image Turbo | 8 | 22.36 s | 17.16 s | 2 / 4 / 6 / 8 | 每帧 512×512，最终 PNG 与关闭预览的基线完全相同 |
| Qwen Image 2.1 + Viggle Turbo | 6 | 24.53 s | 17.15 s | 2 / 4 / 6 | 每帧 512×512，最终 PNG 与关闭预览的基线完全相同 |

GUI 中确认 Z-Image 第 4/8 步、Qwen 第 2/6 步完整显示，标签和缩放控件均可见；
最终图也完整显示。Qwen 第 2 步仍较模糊，属于早期预测，后续步骤逐渐完善。

以相同模型的最终 Tiny 图为参考，比较修复前后相同步骤的预览 RGB MAE
（0–255 像素单位；这不是对原始 VAE 的画质认证）：

| 模型 / 步骤 | 修复前 | 修复后 |
| --- | ---: | ---: |
| Z-Image 2/8 | 58.39 | 17.39 |
| Z-Image 4/8 | 47.77 | 7.84 |
| Z-Image 6/8 | 28.47 | 3.65 |
| Qwen 2/6 | 74.49 | 18.81 |
| Qwen 4/6 | 49.36 | 8.77 |

两个模型的最后一步预览与各自最终 Tiny 解码 PNG 一致。原始 VAE 最终 PNG SHA-256：

- Z-Image：`02d80b5c1c10b6a04ab4e7334ca4550541c47061d381f7c7f97a9eb116574117`
- Qwen：`3eee94c5e9be7be8238a43da02a647233e812308ac17a7c5682e8713cc2005c7`

一次修复后 Z-Image 冷请求在打开本地权重时长时间等待，App 总耗时 181.31 s，
原生 request wall 22.35 s，最终成功且 SHA 相同。该异常样本完整保留，不用于
性能结论。首次完整测试与 App 推理并行，触发 GPU 租约拒绝；之后改为顺序执行。

## Tiny 最终解码与 Qwen 参考图编码

在预览修复前的同一合并构建中完成了以下 GUI 对照；这些编码/最终解码路径未被
此次预览修复修改。各行为为单次观测，包含不同缓存状态，不能解释成通用加速比例。

| 路径 | 原始 VAE | Tiny VAE | 观察 |
| --- | ---: | ---: | --- |
| Z-Image 最终解码 | 0.867 s | 0.127 s | App 请求 18.35 / 17.56 s；细节轻微差异 |
| Qwen 最终解码 | 1.074 s | 0.050 s | App 请求 22.11 / 20.17 s；细节轻微差异 |
| Qwen 编辑参考图编码，cache miss | 0.958 s | 0.093 s | App 请求 43.49 / 40.55 s |
| Qwen 编辑重复请求，cache hit | 接近 0 | 接近 0 | App 请求 24.19 / 24.23 s，无编码阶段净收益 |

Z-Image / Qwen 的 Tiny 最终图相对原始 VAE，RGB MAE 为 1.85 / 2.69，
PSNR 为 39.38 / 34.12 dB；同提示词主体和构图保持，Tiny 不是无损替代。

编辑测试对同一张红茶壶图片执行“改为蓝色，保留形状、光线和木桌”。原始及
Tiny 参考编码均得到蓝茶壶。输出 512×512；参考编码使用模型标准 1024 尺度，
回执为 4096 reference tokens。改变 encoder 会使参考/prompt 缓存失效，回执
正确标记 `tiny_vae_reference_encoding` 近似。Tiny 与原始冷编辑输出 MAE 3.30，
PSNR 34.02 dB。原始编码冷/热编辑也存在小量数值差异（MAE 0.45），因此没有
宣称编辑冷/热结果逐字节一致；生图预览开关的 SHA 对照与此分开验证。

两个模型均做过 GUI 预览取消并重试，取消不发布最终图，成功/取消后侧车数为 0。
修复后的原生 Metal 专项再次覆盖去噪估计、完整 512×512、cadence/末步、
错误尺寸拒绝、取消清理/恢复、关闭预览零工作、latent/velocity 不变。

## 下载及超分页面

仅通过 App 模型中心下载并校验以下两份 Tiny 权重，没有下载基础模型、LoRA、
超分模型或额外依赖模型；模型库安装数由 5 变为 7，LoRA 登记未变化。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| taef1.safetensors | 9,848,636 | `47a6c2bff850da04b267cab70fe3553fef57255eb9a8e76852baa0a87850e54d` |
| taeqi2_1.safetensors | 30,591,264 | `6578b31cc5e336233a2c842eba61b56425bbcac97c7088705f7c7db93291887e` |

超分页面现在在右侧设置直接切换 SeedVR2、Real-ESRGAN ×2/×4，显示配置状态，
图片/视频共用页面结构。GUI 检查了模型切换及源图保留；视频只提供 SeedVR2。
同时修正双媒体 SeedVR2 导致图片创作方式中出现“视频超分”的回归。
本机无 SeedVR2 权重，未下载，也未把选择器检查写成真实 SeedVR2 推理通过。
本次不运行更大尺寸超分；已有合成 ×2/×4、透明图、分块和取消测试已通过。

## 构建与回归

使用现有 Command Line Tools SDK，未下载编译工具或模型依赖。
Swift App/helper、51 个集成测试执行文件及原生库构建完成，打包为本地 ad-hoc
签名 `dist/TurboCider.app`，稳定包不含 Private ANE。
运行时身份：`tc-runtime-build-v1-f15aeaf1057fa579f8eb41f056602b8d83a73408290d811f32e3c827f4110e70`。

- `make test-app`、`make test-library` 通过。
- Tiny VAE App 协议/草稿、原生 Metal 预览专项及本地 HTTP 下载夹具通过。
- `make test` 前半段通过；在 MLX pager 夹具漏链接 `ane_memory.cpp` 处停止。
  补齐测试依赖后，从失败的 `test-streaming-pager` 起执行原 Makefile 全部剩余
  recipe，退出码 0。包括真实 Metal 合成 pager/slot、后续图像、视频和内存测试；
  现有条件跳过保留。完整测试项目分段完成，没有把中途失败计为通过。
- 回归中修正一个已有过期断言：GGUF runtime 已允许 transformer inference-time
  LoRA；继续拒绝内存融合和 text-encoder LoRA，未放宽实际运行时的功能边界。
- 修正上述独立 pager 夹具的漏链接依赖，直接验证新内存观察调用，而非跳过它。
- `git diff --check` 通过；所有媒体、权重和本机绝对模型路径留在忽略目录。

范围仅覆盖本机、这两个已有模型、指定 512×512 请求和上述测试；不是跨设备、
大分辨率、全部提示词或全部加速后端的性能/画质认证。
