# Runtime ANE：图内 tile 与 token chunk 联合筛选

后续已补测新 tile 的 v2 base 与两模型共图切换，并优化无 adapter 时的
hidden 处理，见 [v2 base 记录](runtime-ane-v2-base.md)。本页保留当时构建
的配置筛选，不将后续构建的数字回填为同组结果。

接续[分段修正](runtime-ane-lora-slices.md)。本轮按设计稿的 tile search 方向，
使用现有公开 Core ML exporter 调整图内 GEMM 几何，不改 base 权重、LoRA
计算、SiLU lowering、调度器或原生库。`tile_k` 是 reduction 分块，`tile_n`
是输出 channel 分块；`rows` 才是一次 prediction 的 token chunk。
这些是三种独立参数，不能用更快的 prediction 直接推导整模型提速。

## 单层筛选条件

M4 Max 64 GB；真实 BF16 checkpoint；Qwen block 0 的 teapot step 1 捕获、
Z noise_refiner.0 的 lighthouse 捕获，均为 1024 rows。探针每个独立 trial
先排除两次 warmup，再做十次 GPU/并行交错测量，报告各窗口中位数。
跨 trial 的表格取各 trial 中位数的中位数，不冒充原始样本池的中位数。

探针是独立编译 MLX FFN，并非完整模型 GPU 基线。并行时 GPU 处理 head、
Core ML 处理一个 chunk 的 tail；包含输入转换、输出收集和拼接。含 staging
列将 CPU 权重转换计入：此探针没有 attention 可以隐藏 staging。

每次进程前复用推理竞争预检；导出/编译不与计时并行。所有完成 trial 的
系统 swap-in/out 增量为零，但没有完整进程/驱动内存峰值和物理 ANE residency
证据。`cpuAndNeuralEngine` 仍只表示调度策略。未新增生成质量数值门槛。

原始记录（本地产物，不随源码分发）：

- `outputs/runtime-ane/tile2048-component-abba.jsonl`：每模型 1024 → 2048 →
  2048 → 1024，chunk 固定 288。
- `outputs/runtime-ane/tile512-component-screen.jsonl`：每模型 (1024,1024) →
  (512,512) → (512,1024) → (1024,512) → (1024,1024)，chunk 固定 288。
- `outputs/runtime-ane/tile-n512-chunks-screen.jsonl`：每模型旧 c288 →
  c320/n512 → c352/n512 → c352/n512 → c320/n512 → 旧 c288。

每个文件均包含执行命令、进程快照、probe SHA、前后系统内存、exit code 和
原始 JSON/stdout/stderr。三组均 complete，全部 trial 返回 0，实际形状自检、
数值与失效后恢复检查均通过。三个 screen 属于不同测量组，不拼接计算收益。
独立 probe SHA256 为
`31149f0294cdff2c1cb0efc135ed701651632e543504a07cff153f451315cbfc`。

## 结果：大 tile 并不更快

第一组，c288，毫秒：

| 模型 / K=N | Core ML prediction | 并行 FFN | 并行含 staging |
| --- | ---: | ---: | ---: |
| Qwen / 1024 | 15.564 | 16.379 | 19.155 |
| Qwen / 2048 | 18.022 | 18.642 | 21.446 |
| Z / 1024 | 11.976 | 12.997 | 15.243 |
| Z / 2048 | 14.209 | 14.797 | 17.046 |

2048 数值检查通过，但两模型都慢；不采用。不能仅凭图节点更少推断更快。

第二组保持 c288，K=1024/N=512 的 prediction 为 Qwen **13.164 ms**、
Z **10.395 ms**，对应组内基线为 **15.587 / 11.970 ms**。
但并行 FFN 仍为 **16.436 / 13.011 ms**，基线 **16.399 / 13.035 ms**：
Core ML 更快时，GPU head 成了较长分支，收益没有转化为 FFN 墙钟下降。
K=512 的两种配置也未改善并行 FFN；仅含 staging 的轻微下降不足以说明
图执行更优，故继续测试 K=1024/N=512 加更大的 token chunk。

## 重新平衡：320 / 352 rows

第三组，毫秒；GPU 每行对应同一配置 trial 的完整 GPU FFN 中位数。

| 模型 / chunk / N tile | GPU FFN | 并行 FFN | 并行含 staging | relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Qwen / 288 / 1024 | 21.092 | 16.413 | 19.051 | 0.001799 |
| Qwen / 320 / 512 | 21.097 | 15.051 | 17.696 | 0.001893 |
| Qwen / 352 / 512 | 21.079 | 15.078 | 17.855 | 0.001978 |
| Z / 288 / 1024 | 16.508 | 13.028 | 15.177 | 0.001328 |
| Z / 320 / 512 | 16.511 | 11.999 | 14.087 | 0.001392 |
| Z / 352 / 512 | 16.518 | 11.928 | 14.019 | 0.001459 |

Qwen c320 相对旧 runtime 单层并行窗口下降约 **8.3%**，Z c352 约 **8.4%**。
这是选入整模型测试的依据，不是端到端成绩。320 与 352 的小差距尚不足以
宣称所有层的最优 shape；没有改变 exporter 默认值或产品路由。

## 整模型验证边界

原生库保持 `9b886bdecc1fd95db41a77da81928fc9da5df34999bc805ee2739214102c215e`。
使用同一库分别复测旧 c288 runtime、新候选 runtime、普通 GPU 和冻结图。
下方 base 成绩使用 **v1 图**，不能冒充带 hidden 输出的 v2 共图性能。
LoRA 必须另用 `--lora-inputs` 的 v2 图；base 单层筛选不能替代真实 LoRA
性能或图像编辑验收。不改变最快路线推荐。

### Qwen base，512² / 40 步

resident、狐狸雪景提示词、seed42；旧 runtime → GPU → 候选 runtime →
冻结图，每路独立 batch 一冷两热；不是跨 artifact 的 ABBA，属于整请求筛选。
原始证据为 `outputs/runtime-ane/qwen-base-tile-baseline/` 与
`outputs/runtime-ane/qwen-base-tile-c320-n512/`，两份 summary 均 complete。

| 路线 | 两个热请求（s） | 热中位数 |
| --- | --- | ---: |
| 旧 runtime c288/t1024 | 38.776686 / 38.758152 | 38.767419 s |
| GPU | 42.628014 / 42.631435 | 42.629725 s |
| 候选 runtime c320/k1024/n512 | 37.224077 / 37.206961 | 37.215519 s |
| 原有冻结 base 图 | 30.117992 / 30.138360 | 30.128176 s |

候选比旧 runtime 整请求名义少约 **4.0%**，GPU/候选约 **1.145×**。
原有冻结图仍更快，不替代它。新旧 runtime 各累计 3648 个 hybrid blocks/
predictions、192 个完整 GPU blocks（含冷请求），均无失败回退。
因此收益不是简单地由候选全部退回 GPU 产生的。

候选 trial 系统 swap-in 增加 12 页（16 KiB/页，共 192 KiB），swap-out
不增加；旧 runtime/GPU trial 均无增量。不能称整轮零换页，也不能将系统
增量完全归因于某个模型进程。完整内存和更广质量资格仍待验证。

### Z base，512² / 8 步

同一原生库、resident、狐狸雪景/seed42，一冷两热。顺序同上，证据为
`outputs/runtime-ane/z-base-tile-baseline/` 与
`outputs/runtime-ane/z-base-tile-c352-n512/`，两份 summary 均 complete。

| 路线 | 两个热请求（s） | 热中位数 |
| --- | --- | ---: |
| 旧 runtime c288/t1024 | 6.628155 / 6.581343 | 6.604749 s |
| GPU | 6.996408 / 6.994079 | 6.995243 s |
| 候选 runtime c352/k1024/n512 | 6.327657 / 6.336548 | 6.332102 s |
| 原有冻结 base 图 | 5.341155 / 5.336821 | 5.338988 s |

候选比旧 runtime 整请求名义少约 **4.1%**，GPU/候选约 **1.105×**；
冻结图仍最快。新旧 runtime 均累计 685 hybrid、83 GPU blocks、688 次
Core ML predictions（含冷请求），均无失败回退。各 trial 的系统
swap-in/out 无增量。没有把单层约 8% 的窗口下降报告为整请求收益。

两模型比较共同说明：较小 N tile 的预测收益，需要重新平衡 token rows
才能转化为完整 FFN 收益；再加上 attention、编码/VAE、同步与请求准备，
整请求收益会进一步缩小。本轮没有取消任何校验或跳过 SiLU/LoRA 运算。

### Qwen runtime LoRA 三参考图，512² / 6 步

实际使用 c320/K1024/N512 的 **v2 activation-input 图**，Viggle v0.2.1
r256 原始文件、strength1、seed29；三张有序参考图均走 ref512 策略，
实际 reference tokens=3072。提示词与前次 teapot/dragon 编辑一致。
GPU → runtime → runtime → GPU，每 trial 一冷两热，profile 关闭。
`outputs/runtime-ane/qwen-edit3-tile-c320-n512-abba/summary.json` complete。

| 路线 | 四个热请求（s） | 热中位数 |
| --- | --- | ---: |
| GPU | 12.882528 / 12.939353 / 12.915250 / 12.885725 | 12.900487 s |
| runtime v2 | 14.508751 / 14.366387 / 14.347975 / 13.970411 | 14.357181 s |

GPU/runtime **0.899×**，runtime 慢约 **11.3%**。两个 runtime trial
热请求的 hybrid block/prediction 增量为 52/20、43/10；没有错误回退。
四个 trial 系统 swap-in/out 无增量。前次 c288/t1024 的 14.090 s 是另一
组 ABBA，不能把跨组差值当成隔离 tile 的因果消融，但当前结果不足以
推荐新配置用于 LoRA 编辑。**编辑仍优先普通 GPU**。

base 的分支平衡不等于带 LoRA 的平衡。v2 还需要 gate/up correction、
hidden 输出和 down-LoRA 的依赖/传输；增大 chunk 同时增大这些数据量。
本轮只有整请求和累计计数，不将全部差距归因于其中某个操作。
没有融合 LoRA 到 checkpoint、Core ML 图或 runtime base 权重槽。

### 肉眼检查与保留决定

已查看两模型 base 的末次 GPU/runtime/frozen 图片：狐狸主体、姿态、
构图和色调相近，runtime 有局部毛发/背景差别，未见明显新崩坏；冻结图
细节差别更明显，但本例整体生成正常。查看编辑末次 runtime/GPU 图片，
两把壶、贴纸和暖色桌面接近；两路都未满足蓝壶透明材质要求，不宣称完整
编辑语义验收通过。这些不是多提示词、多适配器或所有层的视觉资格。

保留新 tile/chunk 作为 **BF16 base 的显式 optional 导出配置**，不改
exporter 默认参数、产品默认选择，也不替代最快冻结图。当前 v2 编辑
不提升为最快方案；新 v2 图的无 adapter base 速度、Z 真实 LoRA、
base/A/B/base 切换和 1024² 尚未在这组 tile 上验证，不能用旧图成绩代替。

本轮没有修改 native 源码或库。独立 probe 重建成功；新图在组件和真实
模型运行中完成自检，未放宽原有数值检查。原始结果保留在忽略目录，
没有删除模型、缓存或图片，没有 stage/commit，既有暂存区保持不变。
收尾 `make test-acceleration-contract`、8 项布局、7 项独立性检查均通过；
README、状态目录与 CLI 使用文档共 225 个本地链接目标存在，
`git diff --check` 通过。原生库 SHA 和 249 项既有暂存移除均保持不变。

可复现候选导出，不依赖机器绝对路径；输出目录必须尚不存在：

```sh
.venv/bin/python tools/coreml/export_runtime_ane.py \
  --kind swiglu --rows 320 --hidden 4096 --width 12288 \
  --tile-k 1024 --tile-n 512 --output outputs/runtime-ane/new-qwen-c320-n512
```

Z 对应 `--rows 352 --hidden 3840 --width 10240`。要支持 runtime LoRA，
在另一新目录增加 `--lora-inputs`，不传 adapter、不合并权重。调用仍用
`--hybrid-mode runtime --ane-manifest MANIFEST`，不要将 v1/v2、冻结 base
图与 `lora_fused` artifact 混用。

## Qwen 1024² c1792：N256 未通过组件门槛

在同一 c1792/K1024 v2 共图、Qwen block0 真实 BF16 checkpoint 权重和
**合成** 4096-row 激活下，只将图内 N tile 从 512 改为 256。两图各在独立
进程运行十次 GPU/并行 FFN 交错样本，按 N512→N256→N256→N512
顺序执行；每 trial 有独立 100 ms 进程树内存采样。两种配置都只有一个
1792-row ANE chunk，GPU 保留 2304 rows，不改变模型数学或 LoRA 输入协议。

| 图内 N tile | GPU FFN | Core ML predict | 并行 FFN | 并行含 staging | GPU 后 join |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 82.811 ms | 46.585 ms | 51.487 ms | 54.003 ms | 3.386 ms |
| 256 | 82.810 ms | 47.615 ms | 52.569 ms | 55.087 ms | 4.400 ms |

四 trial 均完成数值与内存验证，relative L2 均为 0.002160，实际每个
样本一次 prediction、零 overflow retry。全部采样窗口无 swap-in/out；
图、checkpoint、probe 与驱动的前后 SHA 一致。N256 并行窗口约慢
2.10%，而非一个值得开展整模型 ABBA 的候选；不把局部负结果说成
整请求退化，更不改变已有 N512 的显式可选路线。候选图和完整原始
记录分别在 `outputs/runtime-ane/qwen-c1792-k1024-n256-v2-tile-screen/`
与 `outputs/runtime-ane/qwen-c1792-n256-n512-tile-screen/`；驱动在同级
`qwen-c1792-n256-n512-tile-screen.py`。生产库仍为
`875b0d3dcf5c766feeced21d71f1a366acf0a1238bc3f0ed332098aa3b27b8cd`，
本次没有重编译或改产品路径。
