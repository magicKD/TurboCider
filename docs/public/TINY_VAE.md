# Tiny VAE 与实时预览

支持 Qwen Image 2.1 的 TAEQI2.1（64 通道、16 倍缩放、RGBA）和 Z-Image
的 TAEF1（16 通道、8 倍缩放、RGB）。原生 C++/MLX Metal 推理，不启动 Python、
PyTorch 或上游仓库。原始 VAE 保持默认，Tiny 编码／最终解码是明确的近似选项。

在创作页或 Playground 的「Tiny VAE 与实时预览」展开项中：

- 选择对应的 `.safetensors` 文件；打包资源或模型目录旁的 `tiny-vae/` 会自动发现。
- Qwen 编辑可选择 Tiny 参考图编码；Z-Image 的现有文生图没有参考编码阶段。
- 最终图可独立选择原始 VAE 或 Tiny。RGBA 的 alpha 不会被丢弃。
- 开启实时预览，默认每 2 步，可配置 1–50 步；最后一步也会更新。
  预览最长边不超过 512 像素，是当前 latent 的近似可视化，不是最终结果，
  不会改变采样 latent。

预览只占一个原子替换的请求侧车 PNG，取消、失败、成功后清理；App 只读取当前
请求的确切侧车路径，绝不把预览作为最终结果发布。设置随草稿／历史请求保存。
未标定的公开流式内存档位和 memory-constrained 模式不能与 Tiny／预览混用。
Qwen 的同步 MLX 路线支持预览；实验性 native DiT／async step pair 不支持此选项，
会明确报错，不会悄悄忽略设置。

## App 内下载（推荐）

打开「模型中心 → Tiny VAE」，选择 Hugging Face 官方或镜像，下载 TAEF1 或
TAEQI2.1，随后点击「用于创作」。下载助手使用原生 Swift/Foundation，固定
官方版本、大小和 SHA256，校验后原子安装；支持进度、取消、重试与缓存复用。
镜像不接收登录凭据，不能通过元数据替换版本或权重。该操作不切换基础模型、
不覆盖原始 VAE。之后在创作页／Playground 展开 Tiny VAE 设置开启相应功能。

官方单文件可直接原生加载，无需转换：

- `madebyollin/taef1`，版本 `b1b2d00e9e440cfbf3dedb34266864da86016ceb`，
  `diffusion_pytorch_model.safetensors`，9,848,636 字节。
- `madebyollin/taeqi2_1`，版本 `5d62f11fee0068b4df7bd699ddab685e32174010`，
  `taeqi2_1.safetensors`，30,591,264 字节。

下载来源记录与权重保存在同一独立安装中。原生加载再次验证官方文件摘要，并在
内存中映射层名和 FP16→FP32，不执行 Python 或反序列化 pickle。

## 兼容的离线权重准备（开发可选，App 不需要这些依赖）

固定来源：madebyollin/taesd，提交 `fdbae2df989db95761f482efb8977e1182fb68d2`。
下载 `taef1_encoder.pth`、`taef1_decoder.pth`、`taeqi2_1_encoder.pth`、
`taeqi2_1_decoder.pth` 到同一个目录。可使用 HuggingFace 镜像／代理获取来源文件。
转换器会在反序列化前验证官方文件 SHA256，仅读取 weights-only 数据：

```sh
python tools/convert/tiny_vae.py --source SOURCE --architecture taef1 --output models/tiny-vae/taef1.safetensors
python tools/convert/tiny_vae.py --source SOURCE --architecture taeqi2_1 --output models/tiny-vae/taeqi2_1.safetensors
```

仅可选的离线转换需要 torch 和 safetensors。App 支持上述原始官方文件和转换后文件。
`tools/native/package.sh` 默认将 `models/tiny-vae` 中两份可选文件复制进资源目录，
可通过 `TURBOCIDER_TINY_VAE_DIR` 指定另一目录。权重不提交 Git。
保留上游 MIT 许可证，见 `native/licenses/TAESD-LICENSE.txt`。

## 请求示例

Schema 1 的 `tiny_vae`（Schema 2 放在 `parameters.tiny_vae`）：

```json
{
  "path": "/absolute/path/taeqi2_1.safetensors",
  "encoder": "original",
  "decoder": "original",
  "preview": true,
  "interval": 2
}
```

选择 Tiny 编码／最终解码时需 `allow_approximation=true`；仅预览不需要。
预览事件沿用进度回调，`phase="preview"`，附带 `preview_path` 和当前采样步数。

## 验证

```sh
make test-tiny-vae
make test-tiny-vae-metal
make test-tiny-vae-download
TURBOCIDER_APP_TARGETS=tiny-vae tools/native/build_app.sh
build/native/turbocider-tiny-vae-app-tests
python tools/validation/tiny_vae_parity.py --reference-root SOURCE --models models/tiny-vae --probe build/native/tiny-vae-probe --output outputs/tiny-parity
build/native/turbocider-tiny-vae-app-tests --gpu z-image-turbo MODEL models/tiny-vae/taef1.safetensors OUTPUT
build/native/turbocider-tiny-vae-app-tests --gpu qwen-image-2.1 QWEN_TURBO_MODEL models/tiny-vae/taeqi2_1.safetensors OUTPUT
build/native/turbocider-tiny-vae-app-tests --gpu-dmd2 qwen-image-2.1 QWEN_BASE_MODEL TINY.safetensors OUTPUT DMD2.safetensors
```

GPU 验收使用 512×512；Z-Image 为 4 步，Qwen 原生 Turbo checkpoint 为其作者指定的 8 步。
DMD2 验收单独使用基础 checkpoint 与固定两步 LoRA，不与 Turbo 八步权重混用。
检查真实 App 事件链、预览开关最终图片字节一致、Tiny 最终解码、采样中取消、
取消后恢复，以及 Qwen Tiny 参考图编码。不要以缺少 Metal 设备的跳过代替通过。

## 本机验收记录（2026-10-27）

- 实际 App 状态链：Z-Image 4 步、Qwen 2.1 原生 Turbo 8 步，均为 512×512。
  两种模型开／关预览的最终 PNG **逐字节相同**；每 2 步与每 3 步的预览、
  最后一步、预览时取消、取消后重试、Tiny 最终解码均已验证；Qwen 还验证了
  1024 参考图的 Tiny 编码与编辑生成。
- 官方 PyTorch 对照：12 个数值用例（含 512 解码、1024 Qwen 编码）和 8 个
  坏权重拒绝用例。最大相对 L2 误差约 9.30e-6，最大绝对误差约 3.04e-4，
  小于 1/8 个 8-bit 像素级的数值门限；这不是对 Tiny 相对原始 VAE 画质的保证。
- 原生预览测试：960 像素对应 latent 限制为 480×320 预览；验证不修改原 latent、
  cadence、取消／异常清理、关闭时零预览工作、拒绝覆盖已有侧车文件。
- 回归：Qwen 126 项（8 项条件跳过）、请求契约 141 项（1 项缺少 Wan fixture
  跳过）均通过；另有 Tiny 请求／App 草稿与协议测试、原生边界与发布检查。
- 发布包包含两份已验证 safetensors、MIT 声明，无 Python／pickle 文件及 Python
  运行库依赖，已通过 ad-hoc 签名校验。原生构建指纹与当前源代码匹配。
- 模型中心原生下载：实际从 HF 镜像下载并校验两份官方文件，验证 App 的下载助手、
  安装登记、选用后请求绑定及重复下载缓存复用；本机 HTTP fixture 另验证错误大小、
  SHA256、取消／清理与恢复。使用这些**下载得到的文件**重新通过了两模型全部 App
  生成／预览／Tiny 编解码验收，没有开发转换步骤。
- 基础版 Qwen Image 2.1：通过普通创作页 `StudioDraft.request` 构造 20 步请求，
  自动携带 Tiny 最终解码的近似许可；真实运行 `qwen_image_2.1_bf16.safetensors`，
  第 2、4…20 步均显示预览并生成有效结果，另存 `app-qwen-base20/acceptance.json`。

本机生成的报告位于 `outputs/tiny-vae-20261027/`：`parity-final/report.json`、
`hf-final/report.json`、`app-z-downloaded/acceptance.json`、
`app-qwen-downloaded/acceptance.json`、`app-qwen-base20/acceptance.json`；
可运行 App 位于 `release/TurboCider.app`。
这些权重与生成文件均未提交到 Git。
