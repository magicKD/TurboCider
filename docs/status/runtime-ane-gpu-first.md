# Runtime ANE：LoRA 初始采样顺序消融

2026-09-29。本轮检验初始采样顺序是否改善短 LoRA 请求；不改变 FFN
计算、不合并 LoRA，不把单层或冷请求变化当整请求加速。

结论：**候选不采用，恢复 hybrid-first**。Qwen 三图热中位名义慢 1.37%，
Z LoRA 名义快 0.45%，不足以支持推广；保留请求状态隔离回归，不新增开关。

## 假设与候选范围

设计稿 §29/30 和 VPIPE `AneFeedForward::plan_block` 都先采样混合路径，
再测完整 GPU，并周期重试。现有 TurboCider 按 layer/rows 隔离，初始
顺序为 hybrid/hybrid/GPU/GPU；同一 adapter 的 resident 请求保留调度状态。

候选只对非空 adapter identity 改为 GPU/GPU/hybrid/hybrid。base 请求
仍为原顺序，即使使用支持 LoRA 输入的 v2 图；换 adapter 或返回 base
重建调度状态，同 adapter 不重置。固定 chunks、每路线首样本排除、
每个新分区 warmup、2%/5% 整块门槛、5% 分区滞回、32 visits 周期采样
和稳定层异步 head 均不变，没有新增环境变量。

长 prefill 的同一 layer/rows 在一个编辑请求中只访问一次，而 decode
多次访问。因此 GPU-first 可能只是把混合编译/探测成本从冷请求或第一次
热请求移到第二次热请求；必须分别看各位置，不能只挑一个最快值。

参考 VPIPE 的调度与 worker 组织，但不采用其 adapter staging merge：
gate/up 修正仍在 SiLU 前，down-LoRA 使用修正后的 hidden，base 权重槽
与图内没有 adapter 权重。此次候选也未改 GPU 或 Core ML 数值精度。

## 构建与配对条件

- 原版：`e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`。
- 候选：`e00222e3c0f5763b4bdbd90fcea70bf3bce485044723634a8e95147094d76187`。
- M4 Max 64 GB、resident、512²、auto chunks、profile 关闭，100 ms 独立采样。
- Qwen：Viggle v0.2.1 r256、6 步、seed29、三张有序 ref512、FP32 低秩，
  c288/t1024 v2 图。Z：distill patch LoRA、8 步、seed42、狐狸提示词，
  c352/K1024/N512 v2 图。
- 原版每模型 GPU → runtime → runtime → GPU，候选反向；每 trial
  一冷两热。每路线全部四个热请求中位数，native request_wall 含 VAE/PNG，
  不含加载和冷请求。不是跨构建交错 ABBA 的严格因果实验。
- 原版基线在[收尾记录](acceleration-cleanup.md#本次整理)完成。候选构建
  和测试全部结束后才启动 benchmark，各 trial 预检竞争推理、繁忙时
  等待，不终止其他进程；预检不代表全程独占设备。

原始证据：

```text
outputs/runtime-ane/qwen-edit3-gpu-first-{before,after}/
outputs/runtime-ane/z-lora-gpu-first-{before,after}/
```

## 验证边界

候选通过构建、66 项加速契约、8 项 Core ML/MLX 微图与集成、Qwen
专项（38 + 7 + 30 项与三个原生/工作流入口）。host 覆盖两种顺序、
首样本排除、保留/关闭/重试、分区 warmup、层/shape 隔离和固定模式。
新增真实 wrapper 覆盖同一 v2 图 base → A → 同 A → B → base，GPU/
混合路径都实际计算并检查输出，返回 base 一致；其中人为提供的调度
成本只检查状态，绝不作为实际加速数据。

## 整请求结果

两模型 before/after 的四份 summary 全部 complete，每个构建、每条路线
有四个热样本。下表只在同一列内计算 GPU/runtime：

| 工作负载与路线 | 原版 hybrid-first | 候选 GPU-first |
| --- | ---: | ---: |
| Qwen 三图 GPU | 12.912022 s | 12.876224 s |
| Qwen 三图 runtime | 14.001179 s | 14.192626 s |
| Qwen GPU/runtime | 0.922× | 0.907× |
| Z LoRA GPU | 8.584159 s | 8.581255 s |
| Z LoRA runtime | 7.766461 s | 7.731253 s |
| Z GPU/runtime | 1.105× | 1.110× |

Qwen 候选热样本为 14.328695 / 14.514621 / 13.790954 / 14.056557 s，
Z 为 7.756497 / 7.706009 / 7.776173 / 7.686828 s。Z 的微小名义改善
不证明稳定提速；Qwen 的结果不支持保留候选，更不能替代普通 GPU。
这是一次顺序筛选，不把跨构建小样本差额称为统计显著因果结果。

### 成本是否只是后移

每个 resident trial 的同一热请求位置，取两个 trial 中位数：

| runtime 请求位置 | 原版 | 候选 |
| --- | ---: | ---: |
| Qwen 热 1 | 14.090598 s | 14.059824 s |
| Qwen 热 2 | 13.856097 s | 14.285589 s |
| Z 热 1 | 7.776194 s | 7.766335 s |
| Z 热 2 | 7.762969 s | 7.696418 s |

Qwen 热 2 名义慢约 3.10%。原版两个 runtime 会话的 cold/热1/热2 hybrid
block 数为 100/42/10、100/47/15；候选为 70/27/57、66/10/42。
长 prefill 到第三次请求才第一次走候选 hybrid，探测成本确实后移；
这是解释整请求变化的调度证据，不是每个耗时差的精确分项归因。
Z 原版两个会话均为 189/248/248，候选均为 192/256/256，后续更多
block 留在混合路径；初始采样会影响后续开关决定，不是等价地重排四次调用。

冷请求也没有一致优势：Qwen 原版 18.045637/18.180100 s，候选
20.752250/17.437750 s；Z 原版 8.846033/8.842848 s，候选
11.348927/8.805113 s。首 trial 额外冷成本不能忽略，也不据后一 trial
更快就宣布改善。以上均不含模型加载，不能当作冷启动总耗时。

## 质量与证据复核

重新校验 16 份原始 JSONL、48 次请求的 backend、LoRA 绑定、失败和
累计计数、精度、计时；两构建的 workload、adapter SHA、参考顺序/SHA/
tokens 一致。16 trial 的独立内存原始数据、报告、工具 SHA 和关联
全部匹配并重新通过 verifier：共 5,689 样本、最大间隙 110.020 ms，
无新增系统 swap-in/out。外部服务、driver/wired 归因与物理 ANE placement
仍未完成，不能宣称完整内存资格。

查看 Qwen 候选 GPU、两个 runtime 末次热输出及原版 runtime 代表图，
米色壶左、蓝壶右、橙色贴纸居中和木桌构图接近，无新增明显崩坏。
蓝壶透明材质仍未实现，与已有记录相同，不把共同失败称作编辑成功。
Z 候选 GPU/runtime 狐狸主体、姿态与雪景接近，毛发和树枝局部有差异，
未见新黑图或明显色偏。这只是两个样本的肉眼检查，不是广泛质量资格。

## 最终保留范围

撤回 scheduler 的 GPU-first 参数和 `begin_request` 的候选选择，恢复
原生实现；固定图、runtime 图、LoRA 数学和已有加速均不改。新增 host
回归改为验证保留顺序下的 warmup、关闭和周期重试；真实 wrapper 继续
验证 base/A/同 A/B/base 的调度重置与输出隔离，不断言已撤回候选。

不继续单纯调整初始顺序、起始 share 或全局门槛；结合此前
[起始分区](runtime-ane-initial-share.md)和[门槛消融](runtime-ane-block-margin.md)，
后续更值得从短请求中暴露的 LoRA 修正/拼接开销或长序列工作量入手，
仍需独立实测。双缓冲是否有价值取决于暴露 staging，而不是理论带宽。

恢复后构建成功，库 SHA 与原版逐字节一致：
`e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`。
再次通过 66 项加速契约、8 项 Core ML/MLX 微图与集成、Qwen 专项
（38 + 7 + 30 项与三个原生/工作流入口）。LoRA/Q4-Q8 最坏 relative L2
仍为 0.00533939/0.00658136，未放宽阈值；新增请求状态隔离回归通过。
既有临时目录 ResourceWarning 未导致失败。本轮没有另跑完整 `make test`，
上一整理轮的完整回归不重标为本轮执行，也未重跑 base/frozen 或恢复版
整模型 screen。保留库一致，历史 base/frozen 跑分仍按各自构建标注。

8 项布局、7 项独立性检查通过。五份维护文档的 98 个本地链接目标存在
且无机器路径，`git diff --check`
通过。未 stage/commit，没有删除原始产物、模型或缓存；249 项暂存移除
对应本地文件均仍存在，暂存 diff SHA 仍为
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`。
基线、候选、构建与测试进程均已终止，没有遗留本轮推理。
更广质量、1024² 重复对照、量化覆盖、完整内存/placement 等目标仍未完成。
