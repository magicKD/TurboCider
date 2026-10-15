# 2026-10-08 · Native W1 完整并行 FFN 组件

本轮完成独立 Native Metal Graph 路线的 W1 FFN：原生 W8／A8 staging、精确 BaseOnly private Program、GPU 通道补集、permit 后 restore、FP32 join 和集合完成域。**这是完整 FFN 组件，不是已经接入 32 层 DiT 的整图并行路线。** 实际设计与接口见 [15](../design/qwen-image-gpu-ane-command-graph/15-native-w1-implementation.md)。现有完整 native GPU Session 的 E2E 仍以 [前一组回执](2026-10-08-native-session-and-e2e.md) 为准，不授予本构建。

## 构建与审查

当前 native runtime build ID：

```text
tc-runtime-build-v1-fdce2b4d788f4093c0beb8e1026ad02932367077d428ea1c22adbdd793c9cc1a
```

原生库、CLI、辅助原生 targets 与构建身份验证通过；561 项 native source/build inputs 与最终 manifest 匹配。5 项 native request route tests 通过；runtime contract 共 129 项，3 项跳过，其余通过。private 新源仅进入显式 private 构建，未改变请求默认策略。

首次构建期间头文件变化导致 source identity 检查拒绝，已按要求重新编译，未绕过校验。最终脚本完成原生阶段后进入额外 App 编译；为减少笔记本负载，主动终止了该脚本的 App 子阶段，整体退出 143。**不将此写成整个 App build 通过**，未启动或替换安装的 App。下次原生验证使用 `TURBOCIDER_NATIVE_ONLY=1`。

独立审查修正了剩余亚毫秒被向下取整为 `wait(0)` 的问题，改用共享绝对 deadline／ceil wait，并检查返回后的真实截止时间。两次独立只读复核未发现当前可信 builder 组合下的提前复用或释放问题。公开的内部 plan 仍不能认证任意 shader 的语义，见 15 的接口前提。

## CPU／宿主验证

| 检查 | 结果 |
|---|---|
| W8／A8／restore 布局、offset、pitch、跨阶段 alias、status mask | 82 checks PASS |
| 共享 MSL kernel body、参数重排与 buffer binding | 149 checks PASS |
| producer permit／ANE submit／read permit 协议 | PASS；不可伪造、同 state 消费、共享 deadline、迟到回调不恢复 |
| FP32 partial join、单次 BF16 边界、生产／tiny 几何 | PASS |
| producer／ANE＋GPU＋consumer 完成聚合 | PASS；未知提交不算物理完成 |
| 新 Objective-C++ 严格宿主语法、独立 fixture link | PASS |
| 完整 GPU block／FFN 默认路线的 CPU 合同回归 | PASS；旧 prepare 保留三 PSO，不引入 join PSO |

原 `ane_gpu.mm` 的 `fastMathEnabled` SDK 弃用警告仍存在；本轮未改变其旧路线数学。新模块严格编译通过，独立链接对该已知弃用项使用 `-Wno-deprecated-declarations`。

## 一组小型硬件组件结果

只运行 **一次 dense GPU＋一次 hybrid**，M=32、H=128、F=1024、G=A=512，生成 BF16 输入与权重，seed=2026100801，fused gate/up 与 pitched down。独立 fixture 不链接 MLX、不加载 checkpoint、不生成图片。先 dense、后 hybrid；其余测试没有提交设备任务。

| 数值／合同 | 结果 |
|---|---:|
| 最终 BF16 relative RMS | 0.02475652297（约 2.48%） |
| 最大绝对误差 | 0.0004119873047 |
| 最大误差／参考峰值 | 0.02109375（约 2.11%） |
| cosine | 0.9996935111 |
| finite 输出 | 4096／4096 |
| producer、ANE、GPU branch、restore consumer | 各有真实物理完成证据 |
| 源输入／权重 hash、padding／guard | 未改变 |
| 原始 Surface 分配 | 9 个，327680 bytes；导入重复收费 0 |
| 同一 fixture ledger storage／peak committed | 1388432／1404812 bytes |
| 完成销毁后 storage／reservation／pending／unknown | 全部归零 |
| alias 错误 plan | 在新 generation／提交前拒绝，原成功状态不变 |

该组件的明确筛选门槛是 RRMS≤8% 与 max_abs/reference_peak≤8%，通过。门槛只属于这组生成数据，不能外推真实 checkpoint、prompt 或最终图片；finite、来源、同步与资源规则没有放宽。

原始时间记录：dense wall 2.811709 ms，hybrid wall 1.430167 ms，其中 producer 0.668708 ms、ANE submit 0.202250 ms、branch join 0.324166 ms、consumer 0.194375 ms。两套 operator 的合计 prepare 0.911852042 s；整个进程约 0.95 s，`time -l` peak footprint 103481968 bytes。fixture 账本不是完整进程／compiler／driver 内存认证。

**没有性能资格。** 每臂只有一个样本，运行顺序固定，PSO 首次使用不同，系统/driver caches 未清空；合计 prepare 也不是同构请求准备时间。该时间只能用于接线诊断，不能报告真实 Qwen 的加速比，也不能证明设备物理重叠。

## 证据与清理

原始输出、测试日志、27 项组件 source hashes、测量 binary SHA256、runtime manifest 验证及清理清单在仓库 `outputs/native-w1-20261008/`。机器可读摘要见 [对应 JSON](2026-10-08-native-w1-parallel-ffn.json)。

删除本轮独占的 `.o`／`.a`、module cache、未测量 helper/test binaries、tiny ANE 编译目录和临时 CPU artifacts：**681 个文件、254981200 bytes（约 243.17 MiB）**。保留当前 native library／CLI、测量 fixture binary、runtime manifest、MSL 资源和必要日志。未删除模型、旧测量证据或系统缓存；没有下载模型、复制 checkpoint，也没有新增真实模型／E2E 请求。

## 下一步

先扩展现有 native phase manifest，给 bank、Surface、status、FP32 partials、Program 与不可观测 guard 登记统一预算；再接 NativeDenoiser 的稳态逐层 W1、整步事务和集合 drain。随后只用已有真实 FFN 输入比较完整 staging/同步路径，证实收益后才安排下一组 base E2E。LoRA、编辑 encoder、Z-image adapter、App 入口和 W2／W3 尚未完成，本轮不对这些能力作发布声明。
