# 2026-10-16 本机对照：FP16、W8、LUT4、W8A8 与手写反向

[ANE 目录](README.md) · [全部可分发样本](experiments-20261016.json) · [来源审计](source-audit.md)

结论：**压缩率不是加速比，允许 ANE 不是选中 ANE，loss 下降不是 ANE 训练证明。** 本轮新增4组有效压缩协议、3组训练机制协议；保留1组初版客户端错误。没有修改产品后端、权重、系统安全/电源设置，没有下载模型或清理缓存。

## 环境、范围和独立 reference

M4 Max、64GiB、macOS26.6.2/build25G83、SDK26.5；Python3.11.16、NumPy2.0.2、coremltools8.3.0，MLX 已安装0.32.0。该 SDK/依赖描述来自本机环境读取，不宣称所有进程的底层动态库均获得硬件 attestation。工具链没有完整 Xcode，`xcrun --find xctrace` 返回找不到工具；本轮没有物理 ANE timeline/整数 MAC counter。

全是**原创合成小图**，不是 Qwen 激活或完整生成质量实验。权重固定为 FP16 源，输入确定生成，独立 CPU FP64 `einsum(optimize=False)` 计算完整输出；不把某个设备的输出当唯一 reference。原始 package、编译目录、下载源码和每次 runner 快照放在 ignored `outputs/ane-survey-20261016/`，本轮约46MiB，磁盘仍约45GiB可用，无需删除任何模型/缓存。

MLX臂沿用新进程的默认设备策略；事后独立只读检查的 `default_device` 与默认stream均为GPU0，但旧四组协议未逐run现场保存这两个属性，不回填为旧run的设备attestation。表中GPU是预期API后端，严格物理归因仍未验证；后续正式设备比较应显式固定并记录device/stream。

公开 JSON 保留 source/report SHA、输入/权重/输出 SHA、所有 warmup/timed 数值与 host wall、计划摘要及派生配对统计；不发布用户数据、模型权重、地址或绝对机器路径。[curator](../scripts/curate_ane_survey.py)只核对既有数据与源码快照，不重新执行硬件或 FP64 oracle，不是硬件证明。

## 1. 压缩图与计时边界

图是 L 个同宽 `Conv1×1→ReLU`，输入 `[1,C,1,R]`。同源权重构造 FP16、per-channel W8、scalar LUT4、静态校准 W8A8；第五臂是相同公式的 compiled MLX GPU FP16。

- W8 用 `linear_quantize_weights`；LUT4 用原创、确定性24轮 scalar Lloyd（16 centroids，不依赖 sklearn），不是最佳可达模型压缩配方。
- W8A8 先用4个独立 calibration seeds 的合成输入，再量化权重；确认输入种子与 calibration 分离，幅度也明确不同（0.35 vs 0.45）。这不代表真实 activation calibration。
- 保存 MIL 中 `constexpr_blockwise_shift_scale` / `constexpr_lut_to_dense`、Conv/ReLU 及 Q/DQ 数量；不能只信 arm 名称。
- 每臂 first1+warmup2，再10或20个 timed calls；轮换并反转五臂顺序，每臂在各位置各出现2或4次。每个 trial 的五臂共享同一输入，逐次全输出检查。
- 边界为 **ready host FP16 ndarray → 全图执行 → 独立 owned host 输出**。Core ML predict/返回数组复制与 MLX 输入构建、`mx.eval`、输出复制均计入。模型导出/压缩、compile/load、权重准备在外，单列。
- compute plan 在首次正式 prediction 前读取；calibration/plan/load 可能改变缓存。first 不是真正 cold start；没有清系统缓存。

固定 relative L2 gates：FP16/MLX0.015、W8 0.06、W8A8 0.08、LUT4 0.35。**LUT 的宽松门槛仅观察合成压缩行为，不是模型质量允许35%误差**；不能称无损或替代真实任务 gate。没有根据运行结果放松门槛。

## 2. 全部 warm 中位数（ms）

| 图/加载政策 | FP16 | W8 | LUT4 | W8A8 | compiled MLX GPU | 计划首选（非const） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C256/R256/L4、seed37、CPU_AND_NE | 0.1819 | 0.1845 | 0.1838 | 0.2308 | 0.2464 | CPU |
| 同源小图、seed37、CPU_ONLY | 0.1833 | 0.1791 | 0.1792 | 0.2298 | 0.2464 | CPU |
| C512/R512/L8、seed41、CPU_AND_NE | 0.7234 | 0.6908 | 0.6840 | 0.6498 | 1.1643 | ANE |
| 同大图、独立seed43、CPU_AND_NE | 0.8487 | 0.8221 | 0.8184 | 0.7514 | 1.2311 | ANE |

CPU_ONLY 只约束四个 Core ML arm，第五臂始终是 MLX GPU。小图控制是新进程同源重新导出/编译，不冒充严格复用同一个 compiled bundle 的政策 A/B；实际 plan 与数值相同提供辅助证据。

plan 中 const 的 preferred device 缺失，保持 unknown，未填0。大图 plan 首选 ANE 不等于所有物理执行已经 trace 证明。

### 比率的统计口径必须分开

大图 `median(FP16 wall)/median(W8A8 wall)` 为约1.113×/1.130×；逐 trial `FP16/W8A8` 的配对中位则为：

| run | paired 中位 | 探索性 bootstrap 95%区间 |
| --- | ---: | --- |
| seed41 | 1.0799× | [1.0450, 1.1542] |
| seed43 | 1.2066× | [1.0827, 1.3239] |

小图 paired 为0.7831×，即 W8A8 较慢。W8/LUT4 在大图的 paired 区间仍跨1，不授予稳定性能胜出。两种子的计时有明显波动，bootstrap 只是同进程有序20组探索样本，不是独立长时显著性或跨机器保证。

表中 Core ML 在该 **CPU-origin** 边界快于 MLX，不证明 GPU-resident 模型也有同样优势，更不是对所有 Metal kernel 的最优性声明。也不能把 Core ML/MLX 比率全归因于整数硬件：布局、融合、accumulation、IO 和框架固定税仍不同。

## 3. 压缩与误差

大图 seed41 的 package：FP16 4,204,960 bytes，W8 2,117,523，LUT4 1,061,282，W8A8 2,123,783。接近2×/4×体积缩减，却只有远小于2×的热态变化；这不是矛盾。

| arm | 小图最大 relative L2 | 大图seed41 | 大图seed43 |
| --- | ---: | ---: | ---: |
| FP16 Core ML | 0.003865 | 0.003606 | 0.003525 |
| W8 | 0.015422 | 0.020706 | 0.020423 |
| LUT4 | 0.202729 | 0.259604 | 0.272317 |
| W8A8 | 0.033591 | 0.045476 | 0.045441 |
| MLX FP16 | 0.000430 | 0.000587 | 0.000578 |

全部360次显式输出通过各自预声明的合成门槛（Core ML288、MLX72）；校准内部调用不计入这360次。不同 arm 的误差不同，不能把性能表包装成完全相同有限精度算法。未另做 decompressed fake-quant 对照，因此尚未将“量化误差”与“后端计算误差”完全分离；这不妨碍报告总误差，但限制归因。

大图压缩本身约0.09s W8、约1.7s LUT；compile约0.018–0.030s、load约0.044–0.083s，远大于一次warm预测。一次性任务不能只引用0.65ms。没有测每臂独立系统峰值、能耗、长期温度、request 或真实模型质量。

## 4. 训练机制：37次完整调用，而非“训练已上 ANE”

[training_probe.py](../labs/ane/training_probe.py)把同形状 runtime-input matmul复用于 `XW`、`GWᵀ`、`XᵀG`；12个 host FP32 SGD update，loss scale64。每步 forward/dX/dW 及最终 forward 共37次，三组总111次完整输出。GPU trainer、Adam 和 LLM 没有在此实验比较。

| 模式/shape | 编译数 | initial loss → final loss | 全调用最大 relative L2 | 计划 |
| --- | ---: | --- | ---: | --- |
| CPU数学控制/32² | 0 | 2.939639 → 0.205436 | 0.000221 | 不涉及设备计划 |
| 公共Core ML/32² | 1 | 2.939639 → 0.205434 | 0.000221 | 首选CPU |
| 公共Core ML/256² | 1 | 212.094051 → 14.562638 | 0.002597 | 首选CPU，支持列表另含ANE |

12个 master generations 全部不同，未重编译，最终新权重再次前向验收。直接 FP16 的 `1e-9` 归零，先放大65536再以FP32恢复得到约1.00044e-9；这是 underflow 数学控制，不是完整训练稳定性证明。

[标准库 training_math](../labs/ane/training_math.py)还用中心有限差分核对小型线性层 dW，最大绝对差约3.65e-12；Adam 小梯度反例说明 loss scale 不自动抵消 epsilon。上述证明梯度公式/参数更新学习机制，**没有取得 ANE 训练加速、物理梯度执行、长期收敛或设备 optimizer 资格**。

## 5. 保留的客户端失败与复现

初版 compression runner 将 MIL `op_type` 写为不存在的 `type`，四个 Core ML arm 在编译前 setup 失败；MLX 有13次输出。原 report 当时写 finished，但 capsule 显式降为 nonqualifying，不产生 Core ML 速度比。修正字段后重新用新目录跑，不覆盖旧结果。这个失败不是 Core ML/ANE 不支持压缩。

从仓库根目录运行，使用已含 NumPy/coremltools/MLX 的环境；不自动安装：

```sh
python3 -B techniques/labs/ane/training_math.py
python3 -B -m unittest discover -s techniques/tests -p test_ane_survey.py -v
.venv/bin/python -B techniques/labs/ane/compression_survey.py --device \
  --output-dir outputs/my-ane-compression-small
.venv/bin/python -B techniques/labs/ane/compression_survey.py --device \
  --channels 512 --rows 512 --layers 8 --rounds 20 --seed 41 \
  --output-dir outputs/my-ane-compression-large
.venv/bin/python -B techniques/labs/ane/training_probe.py --cpu \
  --output-dir outputs/my-training-cpu
.venv/bin/python -B techniques/labs/ane/training_probe.py --device --size 256 \
  --output-dir outputs/my-training-coreml
```

脚本拒绝已有 output；设备工作独立串行执行。复现完整 capsule 还需大图seed43、小图CPU_ONLY和32² Core ML三项控制，再调用 curator 指向包含这些新目录的任务根。`--units cpu` 是 CPU_ONLY 控制；只有 `--device` 才执行 Core ML/MLX，纯标准库测试不会自动跑设备。

采用决定：收录四种压缩表达、计划/执行分离、动态参数和负控方法；不晋升产品默认。实际模型需真实 calibration/holdout、fake quant、强 GPU-resident 分母、完整 request/质量/峰值及冷热/长期验证；这些作为迁移门槛，不是本轮已完成实测。

## 本地学习入口与复查

- [ane/compression_survey.py](../labs/ane/compression_survey.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [ane/training_probe.py](../labs/ane/training_probe.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py ane-training-math`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
