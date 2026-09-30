# Qwen runtime LoRA：低秩 FP16 与自动调度的交错对照

2026-09-29，接续[整理记录](acceleration-cleanup.md)。本轮测已有的低秩
FP16 optional 路线，不改变 base 权重、Core ML 图、SiLU 顺序或默认调度。
新原生改动仅修复 hybrid 结果覆盖精度描述的问题；工具显式记录并验证该选项。

## 决策

- 同输入纯 GPU：FP32 → FP16 低秩乘法为 12.901 → 12.665 s，约 1.019×。
- runtime auto：13.842 → 14.016 s，没有稳定收益，不能推荐 FP16 为混合最快预设。
- 固定一个 ANE chunk：15.484 → 15.006 s，约 1.032×；局部组合有收益，
  但仍慢于 auto 和 GPU，不把固定分区升级为默认。
- 本三图编辑样本仍优先 GPU。FP16 保持显式近似；没有合并 LoRA，也没有
  为得到收益省略非线性、down-LoRA 或身份校验。

## 同构建与工作负载

库 SHA：`e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`。
M4 Max 64 GB；Qwen Image 2.1 BF16；原始 Viggle v0.2.1 r256，strength=1、
6 步、seed29，512² 输出、三张有序 ref512，reference tokens=3072，resident。
提示词为木桌上的左米色壶、右蓝壶、中间橙色龙贴纸。runtime 使用相同
v2 c288/K=N1024 图；不重新导出、编译或写入 adapter。profile 关闭。

FP16 仅控制 LoRA 的两次低秩 matmul，合回 base 输出仍走既有 FP32 累加
与 dtype 舍入。base/残差不改成 FP16。没有开启 Metal norm-RoPE、参考局部
attention、DBCache 或跨步 FFN 复用。各路线都按相同选项运行，并验证实际
native 精度选择描述，不只依赖进程环境或 BF16 base 标签。

所有 trial 一冷两热，整请求墙钟含 VAE/PNG、排除冷请求。每个 trial
统一开启 100 ms 独立进程内存采样，最大间隙门槛 500 ms；竞争推理预检
通过后才启动，无编译/测试与 benchmark 重叠。预检不是设备独占证明。

auto 按以下交错顺序运行，各单元一个独立 resident batch：

```text
FP32: GPU → runtime
FP16: runtime → GPU
FP16: GPU → runtime
FP32: runtime → GPU
```

## 自动路线：匹配 GPU 基准

| 低秩精度 / 路线 | 四个热请求（s） | 中位数 | 同精度 GPU/runtime |
| --- | --- | ---: | ---: |
| FP32 GPU | 12.875012 / 12.890776 / 12.924892 / 12.911260 | 12.901018 | — |
| FP32 runtime auto | 13.931329 / 13.752419 / 13.983594 / 13.707577 | 13.841874 | 0.932× |
| FP16 GPU | 12.667414 / 12.658796 / 12.661873 / 12.693585 | 12.664644 | — |
| FP16 runtime auto | 14.404040 / 14.200222 / 13.831200 / 13.631896 | 14.015711 | 0.904× |

FP32 runtime 比匹配 GPU 慢约 7.3%，FP16 约 10.7%。FP16 runtime
两个 trial 波动明显，不能从最快的 13.632 s 单点声称已获得混合收益。

每请求总共 192 个 block。FP32 两个 trial 的热请求 hybrid/prediction
增量均为 37、5；FP16 分别为 63、30 和 47、15，其余走 unsplit GPU。
无错误回退，关闭不划算层属于正常调度。两种精度实际混合工作量不同，
不能直接用累计 FFN 或 post-join 时间差解释单次 kernel 的变化。

FP16 的 GPU 改善、混合块选择增加与整请求未受益同时出现，说明需要
联合验证调度与计算；本次还没有隔离出每个决策变化的因果，不能把全部
退速归因给 ANE 或 FP16。既有短请求身份校验、串行修正/输出交接仍在；
本轮没有独立重测 SHA 开销，不从时间中减掉历史常量来虚构优化成绩。

## 固定分区消融：同样的混合工作量

随后单独运行 FP32 → FP16 → FP16 → FP32，每个 trial 同样一冷两热，
但固定 `--chunks 1`。每个热请求都为 192 个 hybrid block、192 次预测，
不是 auto 资格数据，也没有把 forced runtime 与 split-GPU 当普通 GPU 比。

| 低秩精度 | 四个热请求（s） | 中位数 |
| --- | --- | ---: |
| FP32 | 15.278396 / 15.511168 / 15.457649 / 15.528768 | 15.484408 |
| FP16 | 14.842053 / 15.115588 / 15.015722 / 14.996878 | 15.006300 |

名义加速 1.032×，耗时少约 3.1%；四个 FP16 热样本均快于四个 FP32 样本。
以下是累计计数按请求作差后取四样本中位（s）：

| 窗口 | FP32 | FP16 |
| --- | ---: | ---: |
| denoise | 14.285132 | 13.787008 |
| hybrid FFN | 8.125280 | 7.575472 |
| LoRA input ready | 5.681422 | 5.462792 |
| post-join | 0.841620 | 0.673832 |

ready 包含上游 attention，post-join 包含输出恢复、down-LoRA、拼接等，
不是纯低秩 kernel；窗口与总时间的差不能机械相加作硬件归因。这个固定
分区结果证明本工作量下的整体改善，不证明 Core ML 本身更快，也不改变
自动路线仍不如 GPU 的选择。

## 质量与内存

实际查看三张输入及六张末次热输出：FP32/FP16 各自的 GPU、auto runtime、
fixed1 runtime。米色壶在左、蓝壶在右、橙色龙贴纸居中，壶形、眼睛与白边
可辨，暖光木桌构图接近，没有新增明显黑图、色偏或结构崩坏。细小表面
纹理和贴纸局部有差异，PNG 并不逐位一致。提示词要求的透明材质仍未在
蓝壶上实现，各路线都有这一限制；不称全部编辑指令通过或广泛质量合格。

两种 GPU 精度各自的四张热 PNG 一致，固定分区亦各自一致；auto 每种精度
四张 PNG 均不同，分区/调度也不同。代表性固定分区 PNG SHA：

- FP32：`1e76ba5f78d7c3d67203b3ef2ba05c8ef008c5eab55f72b237bd7cf279749e1b`
- FP16：`d7cf6c4489ebc1019af972ab1be33485c4bd683785ab7a84d06bddbe46f533ce`

12 个 trial 的独立原始采样与 verifier 报告重新校验通过，共 5,537 个
样本，最大间隙 110.571 ms；关联、报告/原始文件/采样工具 SHA 均匹配。
所有 trial 无新增系统 swap-in/out。进程树 physical footprint 峰值范围
（GB=10⁹ bytes，含加载/冷/热/退出）：

| 路线 | FP32 低秩 | FP16 低秩 |
| --- | ---: | ---: |
| GPU | 23.034–23.386 | 23.023–23.745 |
| runtime auto | 23.952–23.959 | 23.748–23.759 |
| runtime fixed1 | 22.138–22.283 | 21.880–21.880 |

不把 whole-process 峰值当单张热图峰值或 ANE 独占内存；外部 Core ML
服务/driver/wired 归因与物理 ANE placement 仍未验证。

## 代码、复现和验证

- `pipeline.cpp` 将 FP16 LoRA 描述追加移到 GPU/frozen/runtime 的描述替换
  之后，并要求确实有 adapter，修复标记丢失和 base 空标记；未改数学。
- screen 新增 `--qwen-lora-fp16`，统一所有路线的 flag，summary 记录
  `lora.rank_matmul_dtype`；逐请求核对标记，不匹配保持 incomplete。
  这是验证工具参数，原生 CLI 仍使用已有环境变量，默认关闭。
- 三项 screen 契约覆盖参数、环境清理/匹配、结果拒绝；另有 Qwen 源码
  契约检查描述不再被后续路线覆盖。没有通过放松有限性检查获得速度。

具体命令模板见[复现指南](runtime-ane-validation.md)。所有已验证的原始
请求、PNG、遥测、进程快照与内存证据保留在忽略目录：

```text
outputs/runtime-ane/qwen-edit3-rank-precision-fp32-forward/
outputs/runtime-ane/qwen-edit3-rank-precision-fp16-reverse/
outputs/runtime-ane/qwen-edit3-rank-precision-fp16-forward/
outputs/runtime-ane/qwen-edit3-rank-precision-fp32-reverse/
outputs/runtime-ane/qwen-edit3-rank-fixed1-{0-fp32,1-fp16,2-fp16,3-fp32}/
```

八份 summary 全部 complete，36 条原始请求重新通过 backend、LoRA、
精度、参考 tokens 和计时校验；构建/adapter/参考内容与参数全组匹配。

本轮构建、63 项 acceleration contracts、Qwen 专项（38 + 7 + 30 项及
三个原生/工作流入口）、8 项 Core ML/MLX 微图与集成均通过。LoRA / Q4-Q8
最坏 relative L2 仍为 0.00533939 / 0.00658136；已有临时目录 warning
未导致失败。本轮未重跑完整 `make test`，不把此前整理轮的完整回归重标。
最后补查 8 项布局、7 项独立性、112 个维护文档本地链接和 `git diff --check`
均通过。全部 benchmark/测试句柄已正常结束；原生库在整轮测量期间不变。
既有 249 项暂存移除对应的本地文件仍在，暂存 diff SHA 保持
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`。

保留本轮报告修复与显式工具入口，不改变 GPU/冻结图/runtime 的默认配置。
没有删除缓存/权重/原始结果，没有 stage/commit；既有暂存内容保持不动。
本轮不涉及 Z 新跑分、base 三路新对照、1024²、一/两图或第二个训练 LoRA；
整体目标仍未完成。接下来应优先隔离短请求的分区收敛与串行边界成本，
而非仅凭这次固定分区收益强制所有请求走 FP16 hybrid。
