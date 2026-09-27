# Qwen-Image-2.1 Viggle runtime LoRA + base ANE 复用诊断（2026-09-27）

## 为什么不能直接将 LoRA 加到 ANE 的 FFN 输出

每层完整学生 FFN 是 `D_base(silu(G_base(x)+ΔG(x)) ×
(U_base(x)+ΔU(x))) + ΔD(...)`，而现有 1024-row W8A8 Core ML
图只给出前 6144 通道的 **base** FFN 最终输出。激活前的 LoRA
更新经过非线性，再参与 base down 和 LoRA down。只有 ANE 最终输出
并不足以准确恢复这些更新；将 `ΔD(x)` 直接加到 ANE 输出不是正确
算法。模型卡所需的 227 个 LoRA 投影仍在推理时绑定，不能通过
BF16 预合并后冒充原始 runtime LoRA 路线。

显式 `TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC=1` 的当前实验
让首步 prefill 和非 FFN 层照常执行完整 runtime LoRA，5 个 decode 步
的每层 FFN 使用未改动的 base W8A8 ANE 前 6144 通道，GPU 后
6144 通道使用 runtime LoRA 的 gate/up/out 投影切片。改动中的
`Weights::project_slice` 以适配器输出区间和输入列交集选取低秩矩阵，
保留后缀 GPU 的 LoRA；未包含 **ANE 前 6144 通道 gate/up 和
down LoRA**，故是显式有损路径而不是等价的学生模型。

只准入固定 Viggle v0.2.1 r256、scale 1、512²／6 步、BF16 GPU
后缀 + base 6144-channel W8A8 checkpoint-SHA 绑定 manifest、0–3 张
原尺寸参考图（或下文单独显式准入的 1–3 张 512px 参考图）、明确允许近似。GPU/ANE 默认仍拒绝 LoRA，
该开关不能作为 CLI 推荐配置；既有纯 GPU 学生输出不变。

## 单次 warm 请求实验

本机 M4 Max，同 prompt/seed、各自完整六步预热的 resident Session；
仅一次配对，均包括 VAE/PNG，不计首次 Core ML 加载。比较器核对
text/noise/reference 输入张量逐字节相同、227/227 LoRA 投影，hybrid
请求内 160 次 Core ML 预测、零运行失败；实际 ANE 驻留未知。

| 工作负载 | 完整 runtime LoRA GPU → base ANE＋LoRA GPU 后缀 | 倍速 | RGB RMSE／相关系数 |
| --- | ---: | ---: | ---: |
| 狐狸文生图 | 8.240 → 8.027 s | 1.027× | 15.89/255／0.97236 |
| 一图编辑，红色茶壶 | 14.695 → 14.634 s | 1.004× | 8.29/255／0.99466 |

狐狸在雪地、松树构图仍符合主体提示，但前腿和尾部姿态变化明显，
不能称为完整 LoRA 图像质量等价。一图编辑中两张图的茶壶形状和
位置肉眼近似；不过精确 GPU 基线自身也没按提示变成哑光，不能用
接近基线来宣称编辑效果合格。两组样本都没有达到稳健的端到端收益
门槛；不要因单次 1.027× 将该路径自动打开。

追加同条件两次 warm 对照（**两边都**显式设置
`TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1` 和
`TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1`）：

| 工作负载 | 优化 GPU 两次 | 优化 base ANE＋LoRA 后缀两次 | 墙钟中位比 | 与相同开关 GPU 的 RGB |
| --- | ---: | ---: | ---: | ---: |
| 狐狸文生图 | 7.821／7.866 s | 6.357／6.447 s | 1.225× | RMSE 15.65/255，correlation 0.97315 |
| 一图茶壶编辑 | 14.136／14.147 s | 12.630／14.206 s | 1.054× | RMSE 8.31/255，correlation 0.99462 |

文生图的两次单独配对为 1.230× 和 1.220×，观察到明显收益；
编辑则为 1.119× 和 **0.996×**，第二次比纯 GPU 慢，不能声称
编辑已稳定提速。编辑两张 hybrid PNG 与彼此一致，但红色茶壶仍
是亮釉而非提示词要求的哑光。上述两组输入 tensors 同一负载内
逐字节一致，hybrid 每请求 160 次预测且零失败；这些不是
ABBA 顺序或多种子质量门禁。

同样启用 FP16 rank matmul 和融合 Q/K，补充原尺寸参考图编辑的
**单次** warm 配对；三图采用修正 prompt 后重新运行的匹配请求，
不包含第一次 prompt 不一致、被比较器拒绝的试跑：

| 工作负载 | GPU → base ANE＋LoRA 后缀 | 倍速 | RGB RMSE／相关系数 |
| --- | ---: | ---: | ---: |
| 两图编辑 | 21.055 → 21.229 s | 0.992× | 10.76/255／0.98946 |
| 三图编辑 | 28.883 → 28.955 s | 0.998× | 10.17/255／0.99166 |

两组均核对相同输入 tensors、227/227 个 runtime LoRA 投影、候选
每请求 160 次 Core ML 调用与零失败；视觉上主体大体保留，但还
没有多种子编辑成功率门禁。双／三图目前都略慢于优化 GPU，不能
将文生图约 1.22× 的收益外推到图像编辑。

## 两／三图编辑的后 16 层 reference-local 诊断

追加允许六步 Viggle LoRA 在 2–3 张 **1024px 原尺寸**参考图的首步
prefill 使用 `TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION=3`；仅让后 16 层
对第二、第三张参考图使用局部注意力。GPU 路线需显式允许近似；
hybrid 还须开启 `TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC=1`，
不会自动启用，也没有解决 ANE 前缀缺失 LoRA 校正的问题。
两边都保持 FP16 低秩乘法和融合 Q/K。独立 Session 完整六步预热后，
每边重复两次相同 prompt／seed 的 warm 请求；下表列出两次墙钟，
同组对比均验证 text/noise/有序 reference tensors 逐字节相同。

| 工作负载 | reference-local GPU | reference-local hybrid | GPU/hybrid 中位倍速 | 旧无 locality GPU／hybrid 各一次 |
| --- | ---: | ---: | ---: | ---: |
| 两图茶壶编辑 | 20.780／20.741 s | 19.725／19.707 s | 1.053× | 21.055／21.229 s |
| 三图茶壶＋贴纸编辑 | 27.814／27.876 s | 27.062／27.104 s | 1.028× | 28.883／28.955 s |

两／三图的 hybrid 相对旧同路线一次测量分别约 1.076×／1.070×；
同时纯 GPU 也受益，故两路之间只有 2.8–5.3% 的观察差距。
hybrid 每次请求仍是 160 次 Core ML 预测、零失败，227/227
LoRA 投影，未证实预测实际全部在 ANE 执行。完整六步预热后的
相同 seed 重复并非交错 ABBA，也不是多种子质量统计，不能把几
个百分点当作稳定胜出或默认 CLI 配置。

手工并排查看：两图样本的米壶左／蓝壶右、壶嘴、盖和把手均保留；
三图样本两壶与龙贴纸居中均保留，贴纸眼睛和白边仍可辨。
局部纹理、茶壶光泽／桌面阴影有可见变化；此处以主体和编辑意图
为准，不要求逐像素一致。两图 hybrid 对同配置 GPU 的 RGB RMSE
约 8.38/255、三图约 9.96/255；指标仅辅助排错，非质量门槛。
当前提示词的主体摆放符合预期，但不能由此推断文字、细小对象、
其它种子或严格材质编辑可靠。实验 PNG／逐请求报告和比较结果
只保存在机器的临时 benchmark 目录，不进入仓库。

## 六步 LoRA 缩至 512px 参考图：以观感换首步速度

新增 `TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC=1` 门禁，仅接受指定
Viggle v0.2.1 r256、scale 1、512² 输出、六步、`image.edit`、1–3 张
`qwen21_reference_size: 512` 的有序参考图和 `allow_approximation: true`。
GPU 保留全部 runtime LoRA；hybrid 须再显式设置
`TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC=1`，并使用已绑定 base
checkpoint 的 W8A8 manifest 与 BF16 GPU 后缀。两者均使用
`TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1` 和
`TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1`。缩图改变输入张量，
**不能把这里与原尺寸参考图的时间作严格同输入加速比**。此外 hybrid
仍缺 ANE 前 6144 通道的非线性 LoRA 校正。

本机 M4 Max，分别完整六步预热、相同 prompt／seed，各两次 resident
warm 请求（含张量 dump、VAE/PNG，不含初次模型／Core ML 加载）：

| 缩至 512px 的编辑 | GPU 两次 | hybrid 两次 | GPU/hybrid 配对倍速 | RGB RMSE／相关系数 |
| --- | ---: | ---: | ---: | ---: |
| 两图：米壶和蓝壶 | 10.954／10.940 s | 9.194／10.036 s | 1.191×／1.090×；中位 1.139× | 6.84/255／0.99556 |
| 三图：两壶和龙贴纸 | 12.540／12.523 s | 12.051／11.372 s | 1.041×／1.101×；中位 1.070× | 10.54/255／0.99065 |

比较器验证同一负载两路的 text／noise／全部有序参考张量逐字节相同；
两边每次报告 227 个 LoRA 投影，hybrid 每请求 160 次 Core ML 预测、
零运行失败。人工并排查看：两图都保住米壶在左、蓝壶在右及把手、
壶嘴；三图都保住两壶、居中橙色龙贴纸及其眼睛与白边。表面纹理、
局部阴影和物体比例有差异，但按**主体可辨、相对位置与编辑意图
保留**的宽松观感标准，这两个样本可以接受；像素 RMSE 仅用于
定位差异，不是合格阈值。原尺寸参考版细节和比例亦不同；小字、
小主体、身份细节和严格材质要求尚未验证。若 GPU 基线本身不执行
指令，例如哑光茶壶仍为亮釉，则 hybrid 与之接近也不构成编辑成功。

追加同一三图 prompt／参考图但 **seed 29** 的一次完整预热配对：
GPU `12.539 s`，hybrid `10.850 s`，同输入 **1.156×**；完整
text／noise／三张 reference tensors 逐字节一致，RGB RMSE
7.00/255、相关系数 0.99576。肉眼仍保留米壶、蓝壶和更小的
龙贴纸；贴纸眼睛与白边也可辨。**这是第二个 seed，不是第二种
编辑任务，且只测了一次，不构成稳定收益证明。** 仍需不同
编辑任务与交错 ABBA 测试，再决定能否自动推荐缩图或 hybrid。
对细节敏感的请求继续保留原尺寸参考图、完整 runtime LoRA GPU。

合入 `feat/stream` 并重建原生库后，相同三图缩图、seed 17 的
六步 LoRA 回归为 GPU `12.532 s`、hybrid `10.859 s`
（`1.154×`），同输入 text／noise／三张 reference tensors 逐字节
一致、227 个 LoRA 投影、请求内 160 次 Core ML 调用、零失败。
两路线各自的输出 PNG 与合并前对应路线逐字节一致。与原先
`12.540／12.051 s` 单次配对相比，hybrid 时间有较大运行波动，
不能把这一次 `1.154×` 当作稳定编辑收益；此回归仅证明本样本
合并后未见图片变化或速度退化。

不同编辑意图的反例：仍用这三张 512px 参考图和 seed 29，但明确要求
**将小龙贴纸贴在蓝壶正面**而非置于桌面。另一次完整预热的单请求
GPU `12.560 s`、hybrid `12.771 s`，GPU/hybrid 为 **0.983×**：
hybrid 慢约 1.7%，不能推广上一段的 1.156×。输入 text／noise／
有序参考 tensors 逐字节相同；RGB RMSE 8.87/255、相关系数
0.99303。肉眼两路都遵守新指令：米壶左、蓝壶右，贴纸确实在蓝壶
正面，贴纸眼睛与白边仍清晰。因此此例**质量可接受，但没有
速度收益**。这是不同编辑指令但沿用相同参考图／seed 的单次结果；
生产 CLI 仍应优先保留准确 GPU，不能对所有六步三图编辑自动切
suffix-only hybrid。尤其不要把生成相似、RMSE 小误当作编辑指令成功，
应逐图检查贴纸位置等可验证的语义约束。

## 下一步

需要双／三图更多种子及交错顺序重复测量；探索包含 ANE 前缀 LoRA
校正而不重复计算全部 base 前缀的中间量接口，或者测其它 FFN
分片大小、FP16 rank matmul 对照。若拿不到足够收益和观感，
保留纯 GPU LoRA，不应将当前 suffix-only 方案提升为默认。现有
Core ML 图是针对 base 的缓存复用，不包含 LoRA 权重，更不等于
为每个适配器重新编译 Core ML 模型。
