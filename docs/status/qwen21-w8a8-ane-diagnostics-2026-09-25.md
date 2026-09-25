# Qwen-Image-2.1 512² W8A8/ANE 诊断

日期：2026-09-25。本文先记录基于实际 512² 生图第二步、缓存 prefix KV 的
block 0 FFN 输入 `[1024,4096]` 所做的离线实验；后文另列 32 层离线
检查及 512² 生图诊断。**这些结果均不是生产 Session 的 W8A8 验收**。
32 层校准输入位于忽略的 `results/qwen21/qwen21-w8a8-calibration-512-teapot/`；
文档中的本机模型产物、报告也不进入 Git。可用
`tools/native/qwen21_transformer_probe.cpp` 提取 FFN 输入，以
`tools/coreml/export_qwen3.py` 导出单层隔离模型，并用
`tools/validation/qwen21_a8_boundary_compare.py` 对同一输入比较
Core ML `CPU_ONLY` 和 `CPU_AND_NE`。计算设备策略不是物理 ANE 驻留证明。

## 已测结果

同层 W8A16（仅权重量化）相对 BF16 分支的 RMSE 为 `0.02064`，
Core ML prediction 中位数 `9.26 ms`。原 W8A8 的相对 BF16 RMSE 为
`0.91408`，prediction 中位数约 `8.34 ms`：单层调用略快，**输出不合格**。
即使 CPU_ONLY 相对 BF16 只有约 `0.09255` 的 RMSE，CPU_AND_NE 对
CPU_ONLY 的 RMSE 仍约 `0.91270`。该速度差不能外推到 GPU+ANE 的
32 层/5 步或 1024² 加速比。

进一步隔离同一实测输入（下表均为 `CPU_AND_NE` 相对 `CPU_ONLY` 的
relative RMSE；数值来自忽略目录下的对应 compare JSON）：

| 上投影 / hidden / down | 对比结果 |
| --- | ---: |
| W8，仅 input A8；无 hidden A8，完整分支 | 0.02165 |
| W8，hidden A8，W8 down，完整分支 | 0.91270 |
| W8，hidden A8，FP16 down，完整分支 | 0.91266 |
| FP16 上投影，hidden A8，W8 down，完整分支 | 0.04170 |
| W8，仅 input A8，在 down 前输出 SwiGLU | 0.06130 |
| W8，input/hidden A8，在 down 前输出 hidden | 0.90285 |
| FP16 上投影，input/hidden A8，在 down 前输出 hidden | 0.13686 |

后两组 hidden tap 不含 down，不能替代完整分支画质测试；但它们把当前
`CPU_AND_NE` 失真定位在 **W8 上投影与 hidden A8 的图组合中、进入 down
之前**。把 hidden A8 改成 uint8、把 down 切成两段、把上投影后 SwiGLU
暂转 FP32 均未改善：各自的单层对比结果约为 `0.91267`、`0.91267`、
`0.91270`；将 fused gate/up W8 卷积拆成两个独立 W8 卷积，结果也仍为
`0.91270`。把 hidden `dequantize` 换成显式 INT8 `cast × scale` 后仍为
`0.91271`。简单统一输出增益也不是修复：拟合增益约 `7.14×` 后仍剩
`0.599` relative RMSE。相应离线 JSON 为
`qwen21-a8-amplitude-compare-512.json`、
`qwen21-hidden-input-vs-both-tap-512.json`、
`qwen21-a8-projected-down-isolation-512.json` 等。

这些现象提示 Core ML 对该图的 ANE 路径与 CPU 路径不等价，尚不能在
没有内部中间值/设备轨迹的情况下断言具体是哪一个 ANE kernel 或编译 pass。
后续应先找到数值正确且能证明 ANE 驻留的**双 W8 + 双 A8**单层图；
再测未用于校准的提示词/编辑输入、至少 29/32 层覆盖率、512² 文生图和
1–3 张参考图编辑的实际视觉效果、GPU/ANE 同条件端到端性能；后续
诊断实验的进展与未通过项见下文。

## 修正权重量化粒度后的后续证据

上面严重的 CPU/ANE 差异并非所有 W8A8 图的必然结果。改变 **gate/up W8**
的压缩粒度，保留双 A8 和 W8 down、相同校准输入与 SmoothQuant：

| gate/up W8 布局 | 茶壶 block 0 CPU/ANE rRMSE | 狐狸 block 0 对 BF16 rRMSE | 真实 C ABI 单层调用中位数 |
| --- | ---: | ---: | ---: |
| per-channel（旧失败图） | 0.91270 | 未重新测试 | 8.34 ms（旧图） |
| per-block / 32 | 0.00979 | 0.07948 | 14.30 ms（茶壶） |
| per-tensor | 0.04204 | 0.08614 | 6.79 ms（茶壶），7.07 ms（狐狸） |

per-tensor 模型在校准茶壶的 hybrid FFN 输出相对 BF16 为 `0.03543`，
独立狐狸输入为 `0.03129`。狐狸输入来自另一真实 512²/40 步 GPU
生图的文本和初始噪声；通过原生探针复用 prefix KV，在第二步提取全部
32 层 FFN 输入，未加入茶壶校准目录。用茶壶尺度导出、编译全部
**32/32 层**，在狐狸输入上逐层检查，相对 BF16 ANE 分支 rRMSE
约 `0.067–0.296`；其中 block 3–7 明显较高。Core ML 源 `.mlpackage`
逐层进程中的约 12 ms 调用与 C ABI 已编译模型约 7 ms **不是同一
benchmark 边界**，不能以逐层源包测量否定或替代已编译速度。

独立诊断生成器在同一狐狸提示词、seed 42、512²/40 步下完成 GPU 与
GPU+ANE 两次请求：文本、初始噪声和 sigmas 逐项一致；后者执行
`39 × 32 = 1248` 次 Core ML FFN，记录 output copy `0`。GPU 去噪
逐步之和 `42.503 s`，混合 `34.838 s`，比例 `1.220×`；不同进程的
完整首次请求分别 `45.262 s` 和 `44.577 s`，混合还需模型加载，
**这不是重复 warm Session 提速证明**。两张 40 步狐狸输出均是清晰的
狐狸与雪景，目视姿态、主体与构图接近，毛发局部存在差异；RGB
correlation `0.98737`、RMSE `16.57/255`，latent rRMSE `0.08405`。
5 步输出也接近，但两张都明显模糊，不作为最终画质门槛。

实测文件保留在被忽略的 `results/qwen21/` 下：
`qwen21-w8a8-projected-tensor-heldout-fox-all32-report.json`、
`qwen21-w8a8-heldout-fox-layer0-compare.json`、
`w8a8-b0-projected-tensor-heldout-fox-report.json`、
`qwen21-w8a8-fox-{gpu,hybrid}-512-{5,40}.png`。但仅有一组未校准
提示词、一个 seed、单机和独立诊断调用，仍须测试重复 warm Session、
多 prompt/seed、生产级 1–3 张参考图编辑与实际设备驻留/重叠。GPU 后缀
已增加显式 W8A16 研究选项，但下面的对照尚不支持替换更快的 BF16
后缀；默认策略不得据此自动改成 W8A8。

## GPU 后缀 W8A16 与 BF16 同条件对照

诊断生成器显式 `--experimental-w8a8 --experimental-gpu-w8a16` 可把全部
32 层 GPU FFN 后缀的 gate/up 和 down 权重在加载时打包为 MLX affine
W8（64 通道组）；运行时两次 GPU 矩阵乘的激活保持 BF16，ANE 前缀
仍是 per-tensor gate/up W8、双 A8 的完整 32 层图。后缀输出显式转
回 BF16，避免量化矩阵乘返回不同 dtype 破坏下一步的 text/latent
精度约束。该开关只在独立诊断生成器可用，不进入生产 Session。

用同一未校准的狐狸提示词、seed 42、512²/40 步，与前文 BF16 后缀
W8A8 及纯 GPU 的保存张量逐项匹配（文本、初始噪声、sigmas 均一致）：

| 方案 | 去噪逐步时间之和 | 后续步中位数 | 相对纯 GPU 去噪 |
| --- | ---: | ---: | ---: |
| 纯 BF16 GPU | 42.503 s | 1.060 s | 1.000× |
| BF16 GPU 后缀 + W8A8 Core ML | 34.838 s | 0.844 s | 1.220× |
| **W8A16 GPU 后缀 + W8A8 Core ML** | 38.935 s | 0.956 s | **1.092×** |

W8A16 图对 BF16 后缀混合图的 RGB correlation 为 `0.99949`，
RMSE `0.00674`（`[0,1]`），latent rRMSE `0.01395`；对纯 GPU
correlation 为 `0.98903`。W8A16 图片仍为清晰的雪地狐狸，姿态和
主体接近；该样本的精度并非瓶颈，但 **W8A16 比 BF16 后缀慢约
11.8%（去噪耗时）**，因此保留 BF16 为目前最快的实验路线。首次完整
W8A16 请求为 `47.767 s`，包括加载与当次量化，不能当作 warm
Session 提速。后一方案执行了 `1248` 次 Core ML 预测、output copy
记录为 `0`；未证明物理 ANE 驻留。报告和图片位于忽略目录下的
`qwen21-w8a8-fox-w8a16-vs-{bf16,gpu}-512-40.json` 及
`qwen21-w8a8-fox-w8a16-512-40.png`。5 步配对也已运行，但两张
5 步图本身模糊，只作为输入一致/类型与时序 smoke，不作画质验收。

## 参考图编辑：质量与加速并非随参考图数量单调改善

用同一张茶壶、蓝色茶壶和橙色龙贴纸作最多三张参考图，对独立诊断
生成器进行了 512²/40 步、seed 42 的匹配 GPU 与 W8A8 混合请求。
CPU 比较器逐项核对文本 embedding、初始噪声、sigmas、图像槽位以及
每张参考图的 VAE latent；补充了缺失或不同参考张量必报错的测试。
三参考图还用**与生产编辑相同的 1024 像素参考缩放**重新跑了一组，
此前 1–3 图的历史组则直接使用原始 512/1024 参考尺寸，两种输入不可
相互充当同条件性能对照。

| 请求及参考尺寸 | GPU 后续步中位数 | 混合后续步中位数 | 去噪总加速比 | RGB correlation | 目视检查 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 图，原尺寸 | 1.106 s | 0.890 s | 1.213× | 0.99833 | 两图茶壶接近 |
| 2 图，原尺寸 | 1.290 s | 1.145 s | 1.112× | 0.99809 | 两图均只呈现第一把茶壶 |
| 3 图，原尺寸 | 1.350 s | 1.455 s | 0.944× | 0.99755 | 两图均只呈现第一把茶壶 |
| 3 图，生产同款缩放 | 1.626 s | 1.774 s | **0.936×** | 0.99708 | 两图均只呈现第一把茶壶 |

最后一组完整首次请求为 GPU `90.701 s`、混合 `103.468 s`；后者有
`1248` 次 Core ML FFN 调用、`14.825 s` 累计预测时间、`6.849 s`
加载时间，记录 output copy 为 `0`。匹配输入的 latent rRMSE 为
`0.08858`，RGB RMSE `0.02423`（以 `[0,1]` 为单位）。三图缩放对照
去噪和整次请求均比纯 GPU 慢，不能用单参考图的加速比代表多参考性能。
两条路径都漏掉了蓝茶壶和龙，**本例不能作为三主体编辑质量通过的
证据**；既有 1024² 纯 GPU 三参考示例曾呈现三个主体，但分辨率、
请求实现和输入不同，也不能替代本例验收。

配对报告在忽略目录 `results/qwen21/qwen21-w8a8-edit{1,2,3}-512-40-comparison.json`
及 `qwen21-w8a8-edit3-resized-512-40-comparison.json`；PNG 与相邻
`.safetensors` 供目视和输入复核。更长参考 prefix KV 与统一内存竞争
可能限制 FFN 并行收益，但当前计时不含足够的算子级轨迹，**不能据此
确认瓶颈**。诊断生成器现已对齐生产参考缩放；尚需重复 warm Session、
生产编辑质量门禁和其它 seed/prompt，并研究不增加质量损失的多参考
加速策略。

生产 Session 仍只允许完整 32 层 FP16 manifest；上述诊断产物不能用于
默认 CLI 或声称 W8A8 已合格。即使诊断图采用 FP16 权重，Session 仍须验证
`activation_precision=fp16`，防止研究用 A8 图误入生产路径。
1024² 的 Qwen W8A8 GPU/ANE 配对测试
尚未进行；Z-Image 的历史 1024² 速度不属于本模型。
