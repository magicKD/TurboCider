# Qwen-Image-2.1 收敛与 App 集成报告

日期：2026-09-22

## 范围与交付

停止扩大实验范围，后续生成验收限定在 **512×512**。2048²/40 步实验在
第 19 步取消，没有完整输出，不计为质量通过。本次收尾不启动新的模型推理。
已有 1024² 编辑样例仅保留为历史证据；高分辨率纯规划测试不加载模型。

- 原生运行路径：C++/MLX/Metal，BF16 DiT、图像条件、RGBA VAE、请求内
  prefix KV cache、取消及非有限值检查；Python 仅供离线验证。
- App：Qwen 模型选择、1–10 张有序参考图、RGBA、画笔/椭圆标注、独立蒙版、
  任务保存与复用、提示词增强开关。模型选择和示例画布收敛为 512×512，
  GPU/40 步；不改变参考图编码尺寸，不强制覆盖用户保存的画布。
- PE-I2I FP32 保持实验性且须显式启用。2K 菜单标注未完成验证。
- 进度在去噪循环前发送 `0/steps`；内存规划补入参考图 prefix KV 与输出
  几何开销。规划值是估计，不是内存上限。
- Foundation 提示词解析实现移至 `native/platform/apple/`，模型层保留
  C++ 接口，满足既有代码边界检查。
- `make test-qwen21` 提供无模型推理的聚焦回归入口。

## 既有 512×512 正确性与性能证据

匹配比较：`results/qwen21/matched-refresh-mflux-512-report.json`，40 步、
seed 42，使用相同原生文本条件。该比较只覆盖 DiT 和解码 RGB，**不覆盖
独立文本编码对齐或端到端加载速度**。

| 指标 | 结果 |
| --- | --- |
| 调度最大绝对差 | 0 |
| latent relative RMSE | 0.00707334 |
| RGB RMSE | 0.00430514 |
| RGB correlation | 0.99989009 |
| 原生去噪时间 | 146.304 秒 |
| mflux 去噪时间 | 156.250 秒 |
| 去噪加速比 | 1.06798× |

### 用户等待时间（已有实测，非新增实验）

测试机器为 M4 Max 64GB。不启用 PE 的 512²/40 步 GPU 文生图，独立运行
`matched-refresh-gpu-512.log` 总计 155.366 秒；热会话样例总计约 153–156 秒。
短提示词 prepare 约 1.18 秒（并非纯 encoder 单独计时），缓存命中后文本
编码接近零；diffusion 约 146–149 秒，VAE 约 6.3–7.5 秒。热会话导出与
其他收尾差值约 0.02 秒，并非独立 PNG 编码计时。首次磁盘冷缓存、App
排队与界面显示延迟未单独验证，不能将此数值视为首次点击的延迟保证。

带参考图的既有 512² 编辑样例约 241–242 秒：image encode 约 9.3 秒，
text/visual conditioning 约 25.3 秒，diffusion 约 199 秒，VAE 约 7.5 秒。
启用 PE 的两次文生图样例总计约 14.6–16.6 分钟，包含额外提示词生成。
这些场景不可与无 PE 的纯文生图时间直接混用。

已有 512² 两步进度回归确认事件 `(0,2),(1,2),(2,2)`。十参考图单步结果
`results/qwen21/ten-reference-one-step-report.json` 确认 40960 reference tokens
及 512² RGBA 导出；这是输入数量边界验证，不是十个独立主体的质量验收。

## 验证记录

- `TURBOCIDER_NATIVE_ONLY=1 bash tools/native/build.sh`：通过；平台层移动后已重建。
- `make test-qwen21`：32 项 Python 测试通过，无跳过；原生提示词解析、
  采样和 Swift App 工作流测试通过，包含新增的 512 默认值/示例断言。
- `make test`：181 项 unittest 中 169 项通过、12 项跳过，另有原生辅助
  检查通过。跳过项为 1 项缺失 Wan 夹具、11 项需显式启用的 Metal 测试。
- PE conditioning CPU 检查通过，包括 0/1/2/10 图像布局及错误输入。
  PE generation 完成状态检查通过；缺少 tokenizer 参数的输入检查明确跳过。
- `git diff --check`：通过。
- `make test-app`：8 个 App 回归可执行程序全部通过。
- `bash tools/native/package.sh`：通过；`dist/TurboCider.app` 与 CLI 已重新打包，
  ad-hoc 签名验证通过。打包前旧发行目录保留在 `dist/pre-qwen21-512.Alb6WC/`。
- `dist/TurboCider.app/Contents/MacOS/turbocider self-test`：通过（5 项引擎检查）。
- `bash tools/native/build_app.sh`：完整 Swift App 与集成测试构建通过。

## 已知限制与结论边界

- Qwen3.5 BF16 视觉条件严格 parity 尚未通过；不得称 PE-I2I 已获质量验收。
- 蒙版和标注是视觉参考，不是强制逐像素保持的 inpainting。
- JPEG 解码存在小幅字节差异。
- GPU+ANE 路线的实际 ANE 硬件占用未获证明，正常使用默认 GPU。
- 未进行新的大范围主观质量验收；已有样例不能证明所有编辑都能保留主体。
- 2K 单步曾完成，但峰值约 58.0 GiB；40 步取消，不作生产性能承诺。

使用说明见 `docs/public/QWEN_IMAGE_21.md`。详细实验日志保留在本地
`notes/2026-09-21-qwen-image-21.md`；该 notes 路径被 Git 忽略，本报告作为
可纳入版本管理的收尾记录。模型权重和生成媒体不加入代码提交。
