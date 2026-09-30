# Runtime ANE：整块收益门槛消融（未采用）

结论：将自动调度的保留/重新启用收益门槛从 **2%/5%** 提到 **5%/8%**，
没有改善 Qwen 六步 LoRA 编辑，Z base 的本轮配对结果也变慢。恢复原门槛，
不将候选推广为更快方案；普通 GPU、冻结 base 图和 LoRA 计算均不改。

## 测量窗口与代码决策

设计稿 §70 建议 FFN 至少约 5% 收益；§28 的 5% 又是分区变化的滞回。
当前 `RowScheduler` 的 on/off 输入是模型提供的 **完整 block** 时间，
包含 attention、staging 和 residual，不是孤立的 FFN kernel 时间。
例如 FFN 占块耗时一半时，FFN 快 5% 只让整块快约 2.5%（忽略其他开销）。
因此把“FFN 5%”直接改成“整块 5%”是更严格的筛选，并非同一验收标准。

参考 VPIPE 的 whole-section probe、周期重试和独立分区滞回，但不照搬其
LoRA staging merge。adapter 仍通过激活修正参与完整计算，没有合入
checkpoint、Core ML 图或 runtime base 权重槽。

候选仅修改两个 on/off 比例，不改变分区搜索、warmup、EMA、32 次访问的
周期 probe、固定 chunks、精度、图、输入或 GPU 编译边界。恢复后的代码
将比例命名为 `kKeepHybridBlockRatio` / `kReenableHybridBlockRatio`，明确
测量窗口，数值仍为原来的 `.98` / `.95`。

## 构建与条件

- before：`7e80e38cdb6c8ace8f136b3e7bf4a4528b4dd21ab6f70f7392fcc18a524650c2`。
- 被拒绝的候选：`82db24f85ac3ccb1d410ea3bae308488d6f229d327f149bbf36f51608405b76b`。
- M4 Max 64 GB，resident，512²，chunks=auto，无 profile。
- 每个 trial 一次冷请求、两次热请求；墙钟包含 VAE/PNG，不含冷请求。
- 各路运行前执行竞争推理预检，未并行编译/跑其他推理；预检不保证全程
  独占设备。before/候选的各路系统 swap-in/out 计数都无增量，不等于完整
  驱动内存归因；恢复版的单独观测见下文。

## Qwen：三参考图、Viggle v0.2.1 r256、六步

seed29，512 输出、三张有序 ref512，实际 reference tokens=3072。
两构建各自 GPU → runtime → runtime → GPU，分别汇总每路四个热样本。

| 整请求热中位 | before 2%/5% | 候选 5%/8% |
| --- | ---: | ---: |
| GPU | 12.946174 s | 12.955419 s |
| runtime | 14.010241 s | 14.199435 s |
| GPU/runtime | 0.924× | 0.912× |

候选 runtime 名义慢约 1.35%，没有支持提速的证据。before 两个 runtime
会话的最后热请求各 5 个 hybrid block；候选第一会话为 10 个。门槛更高
不保证独立自适应会话的调用数严格单调：初始 timing 会影响校准与分区。
不能只比较阈值或调用数来断言整请求变快。

原始请求、参考 SHA、逐请求 JSONL、进程快照和 PNG：

- `outputs/runtime-ane/qwen-edit3-meaningful-gain-before/`
- `outputs/runtime-ane/qwen-edit3-meaningful-gain-after/`

## Z base：八步，狐狸提示词，seed42

before 顺序 GPU → runtime → frozen；候选反向 frozen → runtime → GPU。
使用同一个 c288 runtime 图、同一个 c4/a5120/b1024 冻结 W8A8 图。

| 整请求热中位 | before 2%/5% | 候选 5%/8% |
| --- | ---: | ---: |
| GPU | 7.012546 s | 7.009279 s |
| runtime | 6.614837 s | 6.718944 s |
| frozen | 5.356718 s | 5.356458 s |
| GPU/runtime | 1.060× | 1.043× |
| GPU/frozen | 1.309× | 1.309× |

runtime 热样本 before 为 6.675271 / 6.554403 s，候选为
6.747865 / 6.690022 s；每个热请求 hybrid block 从 248 降到 208，其余
走完整 GPU，无错误回退。名义慢约 1.57%，而 GPU/frozen 基本稳定。
这是小样本负结果，不是对差额来源的精确因果证明；足以拒绝把候选称为
经过验证的优化。冻结图仍是该工作负载最快路线。

原始证据：`outputs/runtime-ane/z-base-meaningful-gain-before/`、
`outputs/runtime-ane/z-base-meaningful-gain-after/`。

## 正确性、视觉与保留范围

### 恢复原门槛后的验证

恢复版库 SHA：
`9cad19b5d70bfe8b6a15932f07a907ad594b9ba59b5fcd04b3a1f06de3e65e54`。
比例与实验前相同，仅保留命名常量/注释和 host 回归；没有新的调度开关。
重新通过原生构建、契约、8 项微图/集成和 Qwen 专项检查。
布局 8 项、独立性 7 项和文档链接检查通过；本轮未重新运行完整 `make test`，
不将上一次完整测试的范围冒充本轮新增覆盖。

Z 同一工作负载再次运行 GPU → runtime → frozen，各一冷两热：
**7.006139 / 6.575968 / 5.352697 s**，runtime **1.065×**、frozen
**1.309×**。这次恢复屏也支持不保留更严格门槛，但不把相对最初 before
约 0.6% 的差异宣称为新优化收益。
两个热请求均恢复为 248 hybrid / 8 GPU block；GPU、runtime、frozen 三路
最终 PNG 的 SHA256 分别与最初 before 的对应路线一致，无错误回退。
证据：`outputs/runtime-ane/z-base-block-margin-restored/`。

Qwen base 同构建、相同狐狸提示词/seed42、512²/40 步，GPU → runtime →
frozen，各一冷两热：

| 路线 | 热样本 | 中位 | GPU/该路线 |
| --- | --- | ---: | ---: |
| GPU | 42.763206 / 42.768142 s | 42.765674 s | 1.000× |
| runtime c288 | 38.698480 / 39.037207 s | 38.867844 s | 1.100× |
| frozen | 30.220773 / 30.293681 s | 30.257227 s | 1.413× |

runtime 两个热请求分别为 1248/1216 hybrid block，32/64 GPU block，
预测增量与 hybrid 数相同，无错误回退；不是已经退成 GPU 的标签收益。
证据：`outputs/runtime-ane/qwen-base-block-margin-restored/`。未在被拒绝的
5%/8% 构建上运行这组 Qwen base，不用它证明候选的因果影响，也不将
相对历史构建的速度差额宣称为本轮新优化。

已查看 Qwen base 三路最终图：狐狸主体、雪景和整体质量接近，冻结图有
可见姿态细节差异，未见明显崩坏；仅这一提示词，不代表广泛质量验收。
Qwen GPU batch 的系统 swap-in 从 2830 增至 2834（16 KiB 页，共 64 KiB），
swap-out 保持 63896；后续 runtime/frozen batch 无增量。系统 swap 使用量
保持 958.38 MiB。不能把整轮说成“零换页”，也不能将系统增量归因于 ANE。

### 候选检查与质量

候选原生构建、4 项 host、27 项报告/预检、8 项共图工具、11 项 CLI 以及
8 项 Core ML/MLX 图/集成检查通过。LoRA 微图 worst relative L2 仍为
0.00533939；正确性通过不代表性能改善。

GPU PNG 在两个构建间相同，Z frozen PNG 也相同。已查看候选 Qwen/Z 的
GPU/runtime 图：主体、构图、纹理接近，没有明显新增伪影；Qwen 两路仍
没有很好实现蓝壶透明材质，不把“与 GPU 接近”称为编辑指令完全达成。
runtime 自动分配不同 rows 后 PNG 字节可不同，不要求逐像素相等。

新增 host 回归覆盖原 2% 保留边界、5% 重新启用滞回、跨层/行数隔离、
关闭后周期重试，以及固定分区不受 on/off 策略影响。不新增产品开关，
不保留一条未经证明更快的备用调度实现。

下一步应区分短请求的校准阶段与稳定阶段，并优化实际暴露的串行成本。
单纯提高 on/off 百分比不能消除 LoRA 身份校验、激活修正与图边界成本。
`qwen21/pipeline.cpp` 的 `cached_edit_` / `cached_text_` 命中时会复用编码
结果，几乎没有编码工作供哈希线程重叠；因此将哈希提前到编码阶段也不能
直接视为热请求的提速方案。本轮没有跳过、缓存或异步化 adapter 内容校验。
全目标仍未完成，1024² 重复验收、更多编辑/训练 adapter、物理 ANE/重叠
和完整内存归因等边界继续以[维护入口](acceleration.md)为准。
