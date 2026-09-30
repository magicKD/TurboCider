# Runtime v2（8d90 构建）：三路匹配对照与独立内存采样

接续[异步 GPU head](runtime-ane-async-join.md)与[代码整理](acceleration-cleanup.md)。
本轮不改原生计算，补齐当前构建 GPU / runtime / 冻结图的性能和进程内存
证据；不再用历史 b554 的 base 数据代表当前库。

本页“当前”指本次实验的 `8d90f74…` 构建。随后撤出零输入候选并重建的
状态见[整理记录](acceleration-cleanup.md)，没有将本页数字改标为重建库成绩。

## 条件与复现

原生库 SHA：
`8d90f74ad94e63d4a1dfc1f4ce9cf5220f27b0f1491a3a2ddd2c25a722843296`。
M4 Max 64 GB，resident，512²，狐狸雪景 prompt / seed42，无 LoRA。
Qwen 40 步、v2 c320/K1024/N512；Z 8 步、v2 c352/K1024/N512。
使用原有匹配冻结 base 图，没有重新导出或更改精度、checkpoint、分区。
v2 仍是支持完整 runtime LoRA 修正的同一图，base 提供零修正。

每模型 GPU → runtime → frozen → frozen → runtime → GPU，每个 trial
独立 CLI batch 一冷两热，各路线四个热样本。先 Z 后 Qwen，全部串行；
每 trial 预检竞争推理，不与测试/构建并行。不是设备独占或热稳态证明。
时间是 native `request_wall`，含 VAE/PNG、不含冷请求。

两模型的 `runtime_ane_model_screen.py` 均使用：

```sh
--size 512 --routes gpu,runtime,frozen,frozen,runtime,gpu \
--chunks auto --warm-repeats 2 --sample-memory
```

具体 model / runtime / frozen manifest 位于各 trial 的原始请求；模板见
[复现指南](runtime-ane-validation.md)。采样固定 100 ms / 最大间隙 500 ms，
各路线同样开启，没有在结果出来后放宽门槛。profile 关闭。
证据目录：`outputs/runtime-ane/{qwen,z}-async-matched-memory/`。

## 同构建性能

| 模型 / 路线 | 四个热请求（s） | 中位数 | GPU/该路线 |
| --- | --- | ---: | ---: |
| Qwen GPU | 42.659559 / 42.681161 / 42.633083 / 42.624513 | 42.646321 s | 1.000× |
| Qwen runtime v2 | 36.579693 / 36.657189 / 36.559576 / 36.625764 | 36.602728 s | **1.165×** |
| Qwen frozen | 30.160227 / 30.234445 / 30.183856 / 30.258052 | 30.209151 s | **1.412×** |
| Z GPU | 6.990049 / 6.988854 / 6.991111 / 6.990495 | 6.990272 s | 1.000× |
| Z runtime v2 | 6.317371 / 6.319577 / 6.367638 / 6.271066 | 6.318474 s | **1.106×** |
| Z frozen | 5.342010 / 5.333367 / 5.343329 / 5.335394 | 5.338702 s | **1.309×** |

两个代表性 base 工作负载的 runtime 都超过 5% E2E 收益，但冻结图仍最快。
本轮没有原生优化 before/after，不把与旧库的差异直接归因于某个改动。
也没有重测 LoRA/编辑；其当前决策仍是 Qwen 六步优先 GPU、已测 Z
distill patch 可选 runtime，不能把上表 base 收益套给 LoRA。

Qwen runtime 每 trial 累计 3648 predictions / hybrid blocks、3264 async；
Z 为 688 predictions（含首次 headroom 重试）、685 hybrid / 558 async。
各条路线无错误回退。每路两张末次热 PNG 相同，且与此前 b554 匹配实验
相应路线的 PNG SHA 相同；runtime 也与异步 head 实验的 base PNG 相同。

重新查看两模型 GPU/runtime/frozen 六张图：主体、姿态和雪景构图接近，
runtime 有毛发/眼部等局部差异；冻结图尾巴与背景枝叶变化更明显，但未见
黑图、明显色偏或纹理崩坏。仅覆盖这个 prompt/seed，不是广泛质量资格。

## 独立内存结果与限制

12 份原始 hash-chain JSONL、12 份独立 verifier 报告全部重新校验通过，
共 8,143 个样本，最大实际间隙 110.936 ms，小于预设 500 ms。
关联 ID、原始文件/报告/采样工具 SHA 已绑定在 summary；36 个原始请求的
推理遥测也重新校验通过。

下表是两个独立 trial 的进程树峰值范围，**GB = 10⁹ bytes**。采样范围含
模型加载、冷请求、两次热请求和退出，不是每张热图的峰值，也不是 ANE
单独占用。RSS、physical footprint、MLX allocator peak 不可相加。

| 模型 / 路线 | peak RSS（GB） | peak physical footprint（GB） |
| --- | ---: | ---: |
| Qwen GPU | 15.027–15.247 | 19.039–19.419 |
| Qwen runtime | 15.503–15.504 | 18.861–19.982 |
| Qwen frozen | 15.729–15.731 | 24.707–25.032 |
| Z GPU | 12.751 | 22.947–22.948 |
| Z runtime | 13.104–13.105 | 23.041–23.042 |
| Z frozen | 13.417–13.418 | 23.071 |

Qwen runtime 在这些完整进程窗口内的 footprint 峰值低于冻结路线，说明
速度不是唯一取舍；但两次 runtime 自身也有明显变化，不能把跨路线峰值
相减当作纯 Core ML 开销，或据此缩减内存 admission 的安全余量。

- runtime 四个 trial 均无系统 swap-in/out 增量；冻结图四个 trial 也无。
- Z GPU 两个 trial、Qwen 首个 GPU trial 无 swap 增量；Qwen 最后 GPU
  trial 有 **1,376,256 bytes（1.3125 MiB）swap-in**，无 swap-out。
- 所有 trial 的系统 compression 增量为零，但部分存在 decompression，
  最大约 37.9 MB（Qwen 反向 runtime）。系统此前已有 swap 占用，不能
  称整轮“零换页”，也不能把系统计数归因给特定模型进程。
- 采样不覆盖全部外部 Core ML 服务、GPU driver/wired memory；零 wired
  字段不是无 wired 开销。未验证物理 ANE residency 或低内存设备压力行为。

因此本轮提供了更强的进程峰值/系统换页证据，**不等于完整内存验收**。

## 对下一步优化的影响

四个热请求的累计时间差中位数（s）：

| 窗口 | Qwen runtime | Z runtime |
| --- | ---: | ---: |
| 权重 staging 总工作时间 | 3.425320 | 0.578041 |
| 输入就绪后显式等待 staging | 0.000293 | 0.000081 |
| pre-FFN（含上游 attention 等） | 16.026231 | 3.037544 |
| 完整 hybrid FFN | 18.245852 | 3.027243 |
| post-join（可含剩余 GPU head） | 0.595137 | 0.059177 |

显式 staging 等待已很小：此处优先增加双缓冲并不能直接消除很多已测等待，
反而增加 slot 内存。该推断不排除 CPU/GPU 带宽竞争，也不是硬件 timeline
的 70% overlap 验收。async ANE wait 与 GPU 重叠，不当作 exposed 时间。

本轮代码检查发现 v2 base 每次 launch 都清零两块 LoRA 修正输入，当时
提出安全复用已清零状态。后续候选只完成微图及原版 before，没有 after，
已在整理时撤出运行路径；保留状态隔离测试，见[零输入记录](runtime-ane-zero-input.md)。
不能用 microbench 猜测 E2E 收益；本页仍只记录 `8d90f74…` 的实测结果。

## 工具与验证范围

新增 optional `--sample-memory` 与独立 `runtime_ane_memory.py`。采样器和
CLI 属于新建进程组；超时/中断清理本组并保留原异常及原始证据，不终止
其他任务。非主线程调用在启动进程前拒绝，避免无法安装信号清理后留下子进程。
原直接 CLI 模式不变；sampler 原实现和原生库均未修改。

60 项加速契约（4 host + 31 报告 + 6 wrapper + 8 共图工具 + 11 CLI）及
3 项 sampler/verifier host 测试通过。包括超时子进程清理、参数/完成标记
及旧路径；实际 Mac 小进程 smoke 与上述完整两模型采样也通过。
本轮未重跑完整 `make test` 或 runtime 原生微图；本轮修改仅在工具/测试，
不将上一轮完整回归重标为新执行。
没有新模型构建、LoRA 合并、缓存删除或 stage/commit。
最终 8 项布局、7 项独立性、81 个本地文档链接和 `git diff --check` 通过；
原生库及暂存 diff SHA 不变，249 项既有暂存移除对应的本地文件仍在。
本轮所有采样、推理与测试进程均已正常结束。

仍未完成：更广 LoRA/编辑/1024²、Q4/Q8 范围、物理 placement、完整
driver/wired/压力验收及 artifact lease；不将持续优化目标标记为全部完成。
