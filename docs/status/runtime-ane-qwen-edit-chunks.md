# Qwen 三参考图：大 chunk 与 runtime 共图编辑

2026-09-29。接续[1024² base chunk 对照](runtime-ane-qwen-chunks.md)，
本轮不把 base 的收益直接用于 LoRA 编辑结论。LoRA 从未合入 checkpoint、
Core ML artifact 或 runtime base slots；保留完整 SiLU 和低秩修正。

## 同库四配置正反向对照

M4 Max 64 GB，Qwen Image 2.1 BF16 base，Viggle v0.2.1 r256、6 步、
512² 输出、三张按原顺序处理的 ref512、seed29、相同茶壶/贴纸提示词。
FP32 低秩，resident，auto chunks，profile 关闭，全部开启 100 ms 独立
进程树采样。每 trial 预检竞争推理，没有与导出、构建或测试重叠。

库 SHA：`e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`。
顺序 GPU → c288 → c320 → c1792 → c1792 → c320 → c288 → GPU。
每个独立 resident batch 一冷两热，共八 trial、24 请求；中位数使用
每配置全部四个热请求，含 VAE/PNG，不含冷请求或模型加载。

| 配置 | 四个热请求（s） | 池化中位（s） | GPU/该配置 |
| --- | --- | ---: | ---: |
| GPU | 12.882070 / 12.874899 / 12.927249 / 12.879780 | 12.880925 | 1.000× |
| c288 K1024/N1024 v2 | 13.812059 / 13.741352 / 13.946059 / 13.791114 | 13.801587 | 0.933× |
| c320 K1024/N512 v2 | 13.985220 / 13.677776 / 13.807720 / 13.622875 | 13.742748 | 0.937× |
| c1792 K1024/N512 v2 | 12.794024 / 13.658101 / 12.819440 / 13.558134 | 13.188787 | 0.977× |

c1792 相对旧 c288 约少 4.4% 耗时，但整组仍比 GPU 慢约 2.4%，
不能推荐成 Qwen 六步编辑的普遍快路径。c288 同时有不同 N tile，
不能把 c288/c1792 的差值全部归因于 chunk 行数。

按上述 trial 顺序的冷请求：16.901121 / 17.755745 / 17.688368 /
16.145404 / 16.066487 / 17.581480 / 17.658035 / 16.335546 s。
冷请求不等于加载到退出总耗时，不单凭这组顺序宣布冷启动优势。

## 实际调用解释：不能只选有利请求

所有路线 reference tokens=3072、编辑操作/参考顺序/字节身份相同。
c1792 两个 trial 冷/热1/热2 的 hybrid blocks 和 prediction 增量均为
**32 / 32 / 0**，完整 GPU probes 为 **0 / 0 / 32**。短 decode 直接
走完整 GPU；长 prefill 每请求只访问每层一次，第三次请求进入 GPU
对照采样，不能称所有热请求都在使用 ANE。

两个实际执行 ANE 的热请求为 12.794/12.819 s，GPU 探测热请求为
13.658/13.558 s。这里保留全部样本，不只选前两者宣布胜过 GPU。
后续需要长 resident 序列的匹配实验，区分启动调参、周期探测及稳定分区，
并保留 GPU/ANE 完整 block 对照，不能仅删除采样来制造更快结果。

旧 c288 的热1/热2 hybrid 增量为 37/5、38/5，c320 为 37/5、32/0。
这些也反映调度选择，不代表错误 fallback；全组没有失败、fallback 或
overflow retry。没有更改算子、调度参数、误差阈值或 base 权重。

## 独立复核、内存与视觉

主 summary 和八份子 summary 均 complete。24 份原始结果通过 backend、
runtime LoRA、精度标记、时序与累计计数验证；所有配置的工作负载身份
匹配，运行库、manifest、runner 和工具身份重新核对。
八份内存原始流、报告、采样工具 SHA 与 correlation 重验通过：共
**3,530 样本**，最大间隙 **110.0195 ms**。没有新增系统 swap-in/out；
首个 GPU trial 有 278,528 bytes 系统 compression，其余为零。

进程树峰值 footprint（十进制 GB）：GPU 23.743–24.046，c288
23.938–23.996，c320 23.893–23.973，c1792 22.661–22.671。
这是含加载/冷/热/退出的进程窗口，不是独占 ANE、全部 driver/wired 内存，
也没有据此确立峰值差异的因果解释。`cpuAndNeuralEngine` 不证明物理 ANE 驻留。

已打开反向 GPU、c288、c1792 的**首个热请求** PNG，后者确实有 ANE
prediction：两把茶壶的位置/形状、居中的橙色贴纸、暖光木桌及细节接近，
没有明显新增崩坏。三路均未很好保留蓝壶透明质感；这是共同编辑限制，
不能说全部指令完美满足。只有一组参考/prompt，不构成广泛质量验收。

本地证据：`outputs/runtime-ane/qwen-edit3-chunks-matched/`，不随源码分发。

## 共图编辑校验缺口与修复

原诊断开关 `TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC=1` 要求每个
请求恰好带一份 LoRA。它作用于整个 batch，所以同图 base → A → B →
base 编辑会在无 LoRA 请求处被拒绝。新增 CLI 回归在 e55 库上复现
1/2/3 参考图 base 的三项失败；带 LoRA 的对应请求通过。

仅为显式 `runtime` + `gpu_ane` 的 base 请求补齐该门禁：仍限六步、
512²、1–3 张 ref512、resident、显式近似及匹配 manifest；普通 GPU/
冻结图的无 LoRA 请求不会因此获准使用此诊断开关。没有改 FFN/SiLU、
staging、调度、LoRA 计算或默认选路。

共图工具新增 runtime ref512 编辑输入、参考图有序哈希前后检查、实际
非零 prediction 增量门禁、编辑 token/操作一致性及 graph/library/adapter
身份前后验证。固定 chunks=1 隔离调度变化；生成/frozen 原有边界不变。
合成 B 仍是临时半幅 adapter，不是第二个训练 LoRA 的质量资格。

新库 SHA：`8e533d4daf9edd3e45889ea15463c9bff33be85479d0941cd1debebc670cdf9b`。
上表属于修复前 e55 库，不重标为新库性能。native-only 构建成功；
73 项加速契约（4 host + 37 screen + 6 内存 + 14 共图工具 + 12 CLI）及
Qwen 专项 38 + 7 + 30 项、三个 native/工作流入口通过。

## 新库真实共图编辑切换：通过

`outputs/runtime-ane/qwen-edit3-c1792-shared-switch/`：同一 c1792 v2 图、
固定一 chunk，base 预热 → base → 原始 Viggle A → 合成半幅 B → base，
五个请求全部成功。重新读取原始 stdout 与 PNG 检查：

- 实际 Core ML 累计调用 **32 / 64 / 96 / 128 / 160**，每请求确实
  执行 32 次 prediction，不是短序列完全走 GPU 的假阳性。
- adapter projection 绑定为 **0 / 0 / 227 / 227 / 0**；每次 reference
  tokens=3072；图 load_seconds 保持 0.052659333 s，没有重载。
- 返回 base PNG 与预热后 base 字节一致；A/B PNG 不同。原始 adapter、
  manifest、运行库和按顺序的参考图前后身份检查通过。
- headroom 始终为 1；无失败、fallback、overflow retry。沿用微图的
  SiLU 前 gate/up 修正与修正后 hidden 的 down-LoRA 测试，不融合权重。
- 已查看新库 A 的编辑 PNG，与本轮 e55 GPU/c1792 的主体、构图与色调
  接近，蓝壶透明材质仍未很好恢复；不把 B 作为第二个训练 LoRA 验收。

六步无 LoRA base 仅用于**状态隔离**，不代表普通 base 四十步的图像质量。
固定分区切换也不是 auto 的性能对照；该测试不带独立内存采样，不替代
前述 e55 八 trial 的计时/内存报告。1/2 参考图通过请求/工具契约，
本轮真实生成与切换只测试了三参考图，不扩大全场景支持声明。

复现（原始 adapter 与模型由用户准备，B 在临时目录生成并清理）：

```sh
.venv/bin/python3 -B tools/validation/runtime_lora_shared_graph_switch.py \
  --model models/Comfy-Org-Qwen-Image-2.1 \
  --base-request inputs/base-edit-runtime.json \
  --adapter-request inputs/lora-edit-runtime.json \
  --output outputs/runtime-ane/new-edit-switch
```

两份请求除 output/LoRA 绑定外完全一致：`image.edit`、六步、512²、
相同 prompt/seed、按顺序的三张参考、`qwen21_reference_size: 512`、
`resident`、`gpu_ane`、`allow_approximation: true`、`hybrid_mlp_mode: runtime`、
同一 base-only v2 manifest。工具设置所需 ref512 诊断和固定 chunks，
清除外部其他实验开关；不得将此固定分区结果当成 auto 加速成绩。

## 回归与未完成项

新库上的完整 `make test` 已正常结束（exit 0），日志保留在
`outputs/runtime-ane/qwen-edit-chunks-full-test.log`。覆盖请求、CLI、仓库边界、
其他模型和 streaming 回归；原有缺夹具/专用构建/GPU opt-in 的 skip
仍不算通过覆盖。未另跑 runtime 微图套件，沿用上一轮 9 项真实微图/集成，
本轮新执行的是上述三图真实切换，不能互相替代覆盖范围。

未完成：更长 resident 的周期复测与收益复现、1/2 图实际编辑/切换、第二个
训练 LoRA 和更多提示词、同新库 GPU/runtime/冻结图的完整 base 重测、
物理 ANE placement 与完整内存资格。继续保留现有 GPU/最快冻结 base、
可选 runtime 和所有错误/有限性门禁；未 stage/commit，既有暂存移除和
原始图片、参考、模型及缓存不动。

静态检查：七份维护文档的 118 个本地链接目标存在，变更源码/工具/测试
及文档无机器用户目录路径，`git diff --check` 通过。运行库保持上述 8e533d4
身份；暂存 diff SHA 保持
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`，
249 项既有暂存移除对应的本地文件仍在。

## 新库长 resident 对照：完成

所有构建/回归/切换结束后，已在新 8e533d4 库启动 GPU → runtime c1792 →
runtime c1792 → GPU，每 trial 一冷八热，共 36 请求，保持同一三图
workload、FP32 低秩、auto/profile-off 和独立采样。证据目录：
`outputs/runtime-ane/qwen-edit3-c1792-resident-abba/`。

预先固定统计窗口：全部第 1–8 个热请求，以及第 6–8 个热请求，分别
每路线 16/6 样本。后者用于观察较晚阶段，不代表所有层或长期周期均已
收敛。运行期间没有构建、导出或测试；结束后独立重验全部原始结果。

| 预声明窗口 | 每路线热样本数 | GPU 中位（s） | runtime 中位（s） | GPU/runtime |
| --- | ---: | ---: | ---: | ---: |
| 热 1–8，含调度探测 | 16 | 12.895099 | 12.826079 | 1.005× |
| 热 6–8，较晚窗口 | 6 | 12.909628 | 12.682185 | 1.018× |

全部热样本仅少 **0.54%** 耗时，较晚窗口少 **1.76%**；这是一组匹配
campaign 的小收益，不足以宣布普遍显著加速，也不升级默认路由。旧 e55
一冷两热的 0.977× 保留，不能把统计窗口变化归因于新库校验修复。

按 trial 顺序，八个热请求的原始墙钟（s）：

```text
GPU 0: 12.870685 12.893996 12.879598 12.896202 12.860176 12.896898 12.916249 12.887587
RT  1: 12.793626 13.724308 13.584590 12.807154 12.887829 12.585561 12.677392 12.623764
RT  2: 12.845003 13.650044 13.619724 12.934743 12.887278 12.709299 12.686977 12.703965
GPU 3: 12.886836 12.907953 12.868779 12.890543 12.908663 12.929476 12.905884 12.913371
```

两个 runtime trial 的冷/热1…热8 实际 prediction 增量分别为
`32/32/0/0/32/32/32/32/32` 与 `32/32/0/0/31/31/31/31/31`。
热2/3 各做 32 层完整 GPU 探测；热6–8 各有 32/31 个异步 hybrid
block，第二 trial 有一层选择 GPU。短 decode 始终 GPU；没有失败、
fallback 或 overflow retry，headroom=1。不能将异步 wait 累计时间解释
成未隐藏的 ANE 等待，也不能把未计时分支的零 GPU 计时当成免费计算。

四 trial 和 driver 均正常结束，driver 保存的预声明窗口、source/manifest/
library/tool 身份在整理前重验一致。36 条结果的工作负载、LoRA FP32
标记、reference tokens=3072、参考顺序/字节、计时及累计调用验证通过。
工具源码在本次整理后有去重改动；原 driver 哈希保留为**运行时版本**，
不重写成整理后的源码哈希，原生运行库仍为上述 8e533d4。

四份独立内存流/报告 SHA、工具身份及 correlation 重验通过：共
**4,801 样本**，最大间隙 **110.017792 ms**。峰值 footprint 为 GPU
24.037–24.044 GB、runtime 22.653–22.702 GB（十进制，含加载/冷/热/退出）。
trial2 有 **983,040 bytes 系统 swap-in**，全组无新增 swap-out/compression；
不能称零换页，不归因于单个进程，也不是全部 driver/ANE 内存资格。

已查看反向 GPU 和 runtime 的热8 PNG；runtime 该请求有 31 次实际
prediction。主体、构图、颜色与贴纸细节肉眼接近，无明显新增质量退化；
两路蓝壶都仍呈不透明陶瓷质感，未满足透明材质要求。只覆盖同一组
参考/prompt/seed，不替代更多编辑/adapter 的验收。

结论：保留 c1792、auto 调度及稳定层异步 head 为显式 runtime 路径的
可选能力；六步三图编辑一般仍优先 GPU，长 resident 可自行复测此配置。
本轮到此收尾，不继续启动新的图几何或性能搜索。
