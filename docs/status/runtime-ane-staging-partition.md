# Runtime ANE：转换任务粒度消融

接续 [v2 base 优化](runtime-ane-v2-base.md)。设计中的 staging/attention
重叠仍是优化目标，但不能把更少的 CPU 任务直接等同于更快的请求。

## 候选与边界

参考 VPIPE 的 `AneWorker::parallel_for`：其独立转换池最多四个线程，按
连续区间处理数据。当前实现的 Core ML worker 与 GCD 转换队列分离，
`Surface::fill_rows` 将大矩阵拆为 16 行任务。候选只将这些任务归并为
最多四组连续区间后调用 `dispatch_apply`，小矩阵串行路径不变。

这不是完整复刻 VPIPE 的固定线程池；本次只检验 GCD 任务粒度。未改变
SIMD 数学、矩阵布局、图、权重、LoRA、headroom 或调度策略。候选同时
影响权重和 activation 输入转换，不能把差异全部归为权重 staging。

## 两模型真实单层 ABBA

M4 Max 64 GB，真实 BF16 权重和捕获 activation，1024 rows，ANE 一个
chunk。Qwen c320/K1024/N512，Z c352/K1024/N512，均为 **v2** 图。
Qwen 为 teapot block0/step1，Z 为 lighthouse noise_refiner.0。
每模型顺序为原版 → 四组 → 四组 → 原版，每 trial 排除两次 warmup，
随后十次 GPU/并行交错测量。表格取两个 trial 中位数的中位数，单位 ms。

| 模型 / 转换策略 | GPU FFN | 并行 FFN | 并行含 staging | staging | Core ML prediction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen / 原版 | 21.085 | 15.043 | 17.687 | 2.646 | 13.511 |
| Qwen / 四组 | 21.089 | 15.009 | 18.036 | 3.012 | 13.398 |
| Z / 原版 | 16.512 | 12.164 | 14.244 | 2.078 | 11.157 |
| Z / 四组 | 16.512 | 12.029 | 14.419 | 2.371 | 11.072 |

该组件工具没有 attention，staging 全部暴露，不能据此宣称整模型回退
约 2%。但当前候选没有证明总体组件优势：较小的并行窗口变化不足以抵消
更慢的转换。**撤回候选，不增加产品开关，不替代已有转换实现。**

数值与失效恢复检查全部通过，relative L2 均保持 Qwen 0.00189305、
Z 0.00145874；Z 两次 warmup headroom 重试后为 16，Qwen 无重试。
这里使用 base 输入，不是 adapter 性能或图像质量资格。

每次启动前检查竞争推理。第二个 Z trial 后发现短暂 ComfyUI 进程，后续
trial 自动等待，随后该进程已退出；没有终止他人进程。预检并非独占设备，
不能排除中途干扰，因此 Z 小幅差异只作筛选，不作为精确因果结论。
各 trial 系统 swap-in/out 页数增量均为零，不代表完整进程峰值内存验收。

证据：`outputs/runtime-ane/staging-partition-component-abba.jsonl`（complete），
包含命令、预检快照、系统内存、二进制 SHA、stdout/stderr 和退出码。
独立 probe SHA：

- 原版：`68087887e983922c629a4ffb242eb56085e1e5b0324dbfca97279e3d7e0f1cc4`。
- 四组：`fbb47676a93e4e44ea3b50e4f0c742feba3db93353ea29733c92437fc11e7e16`。

候选仅构建到独立 probe，没有替换产品原生库。源码已恢复原转换分支，
产品库仍为 `b55419e636ada12e54dc197b7099811225b361d63d2c1d16e6e0560449104dba`。
没有新增缓存、合并适配器或删除实验依据。接续已完成同一保留构建上的
[GPU/runtime v2/冻结图匹配对照](runtime-ane-v2-matched.md)，没有凭单层
staging 推导端到端收益。
