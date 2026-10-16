# 异构切分：先证明图合法，再寻找关键路径收益

[专题导航](README.md) · [共享交接](memory-and-handoff.md) · [vpipe](vpipe-runtime-weights.md)

本页是代数推导与运行时设计综合，不把线程同时运行当成物理 GPU/ANE 同时执行。参考实现和来源见 [项目案例](hybrid-projects-and-stages.md) 与 [固定来源](sources-and-coverage.md)。

## 六种不同的“协同”

| 方案 | 合法条件与合并 | 主要成本 | 典型入口 |
| --- | --- | --- | --- |
| FFN 按 row/token 切 | 每 row 独立，拼接输出 | 两边都需完整权重；row bucket | vpipe `AneFeedForward` |
| SwiGLU 按 intermediate F 切 | gate/up 配对、down 对应切片，partial 相加 | 输出归约、精度顺序、LoRA | mac_transformer、本库真实 Qwen FFN |
| 按 GEMM K 切 | `AB = Σ A_i B_i` | partial 归约；不一定跨设备 | ccv 图内 K-split |
| 按模型 branch 切 | branches 仅依赖已就绪共同输入 | join 与共享输入准备 | FLUX.2 parallel block |
| 按 stage 切 | 前阶段完成才可消费 | encoder/decoder 或 prefill/decode handoff | whisper.cpp、ANE 研究路径 |
| 多请求/微批流水 | 独立请求或合法流式 chunk，有界状态 | 排队、backpressure、尾延迟 | ANEMLL/音频运行时的部署思路 |

图内 K-split 不等于 GPU/ANE 两路分工；按阶段使用两设备也不等于单请求并行。流水吞吐与单请求 latency 分别报告。

## SwiGLU 的正确切法

采用 `X[M,H]`、`Wg/Wu[H,F]`、`Wd[F,H]`：

```text
F = F_GPU ∪ F_ANE，二者不交且覆盖全部 intermediate channels
Hg = silu(X Wg[:,F_GPU] + bg[F_GPU]) * (X Wu[:,F_GPU] + bu[F_GPU])
Ha = silu(X Wg[:,F_ANE] + bg[F_ANE]) * (X Wu[:,F_ANE] + bu[F_ANE])
Y  = Hg Wd[F_GPU,:] + Ha Wd[F_ANE,:] + bd
```

`bd`、residual、外层 modulation gate 只加/施加一次。若按 row 分，每个 row 则完整算自己的 bias/residual。改变归约顺序是实数代数等价，不保证浮点逐位相同。不能把 hidden input 随意切成两半再各自做 SiLU：一般 `silu(a+b) != silu(a)+silu(b)`。

LoRA 也必须保持原函数：gate/up 的 adapter 输出先进入非线性；down 的低秩项须覆盖所有 intermediate channels。将量化 base 反量化后合并 LoRA 并再次量化是新数值配方，不是纯调度。详见既有 [LoRA 合同](../gpu-ane/lora.md)。

## Parallel block 与 sequential block

普通 Transformer：`A = attention(norm(X)); Z = X+A; Y = Z+FFN(norm(Z))`。FFN 依赖 attention 的结果，不能写成两支的 `max(Tattn,Tffn)`。

FLUX.2 一类 parallel block 可先得到共同 `Z`，分别求 attention `A` 和 MLP `B`，再 `concat(A,B) Wo`。将 `Wo` 按输入行分解后可写成 `A Wo_A + B Wo_B`；这时 branch 级放置有代数依据。还须检查真正模型是否有共同 projection、norm、位置编码、modulation 和 bias。模型名字相似不保证 block 图相同。

## 成本模型只用于提出候选

```text
T_hybrid = exposed_prepare
         + max(T_GPU_concurrent, T_ANE_concurrent)
         + exposed_restore_and_join
         + newly_exposed_synchronization
```

只在确实独立且提交已发生的窗口里使用 `max`。测 concurrent 的分支时间，不拿 isolated 结果相加/取最大来推断；共享 DRAM、CPU staging、driver、cache 和功耗都会改变两支成本。

忽略固定开销的粗略平衡点 `f_ANE ≈ rate_ANE/(rate_GPU+rate_ANE)` 只能用作搜索起点。真实搜索允许 `share=0`，再考虑 bucket、tile、带宽与 p95 deadline。若 CPU 第三支成为 straggler，应减小或移除它，不以“三设备全忙”为目标。

## 一个实用控制器

1. 对相同形状建立优化后的 unsplit GPU 基线，避免把额外中途 commit 当成免费的操作。
2. 测 serial split：包含 pack、ANE、GPU complement、restore/join；检查数值。
3. 用持久 worker 提交 ANE，GPU 只计算 complement；完成后 join。观测等待落在哪一支。
4. 扫少量合法 share、tile、bucket；先固定候选，再做 ABBA/反序和未用于调参的输入确认。
5. 与真实 unsplit GPU A/B；完整请求若不盈利则保留 GPU，即便 ANE isolated TOPS 很高。

vpipe 还区分 `kSplit/kProbe/kGpu`：probe 保留 split 点但不做 staging，用于识别 mid-block drain 的固定损失。这比只根据 GPU/ANE rows/ms 调 share 更完整，见 [源码走查](vpipe-runtime-weights.md)。

## 最小复现与否决

```sh
python3 -B techniques/labs/run.py parallel-branches
python3 -B techniques/labs/run.py segmented-ffn
python3 -B techniques/labs/run.py optional-devices
python3 techniques/examples/split_ffn_reference.py
```

这些是 CPU 数学/调度演示，不是设备 benchmark。负控包括 gate/up 错配、down 漏片、bias 重复、sequential 假并行、CPU 分支晚到、LoRA 缺支。设备实验另外检查真实输出、完成事件和 [内存所有权](memory-and-handoff.md)。

重开条件：GPU 基线变化、激活 token 分布变化、checkpoint/LoRA 改变、图 tile/SDK/SoC 更新。历史失败不代表该拓扑永远无效。
