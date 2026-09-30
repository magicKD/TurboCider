# 零修正输入复用候选：未验证收益，撤出运行路径

## 收尾决策

候选尝试在连续 base 请求中复用已经清零的 v2 gate/up 修正 IOSurface，
避免每层重复清零；不改变权重、LoRA 或激活计算。候选已通过真实微图，
但没有运行候选库的整模型 after，**不能判断端到端收益**。
按本次整理要求，移除运行路径中的缓存状态及专用流量计数，恢复每次 base
launch 清零。没有增加用户开关，也没有把它记作已证明无效的优化。

候选库 SHA 为
`cc422e7b7e48648b2499e52a04b93e1ad7217005975a28ad0cbf67c29eb20538`；
现有同库三路性能结论仍属于
`8d90f74ad94e63d4a1dfc1f4ce9cf5220f27b0f1491a3a2ddd2c25a722843296`，
见[匹配对照](runtime-ane-matched-memory.md)。不能将 before 数据写成候选成绩。

## 已有原版基线

M4 Max 64 GB，512²、resident、seed42、同一 fox prompt、auto chunks，
profile 关闭，100 ms 独立内存采样。每模型两个 runtime trial，各一冷两热；
热请求含 VAE/PNG，不含冷请求。这是顺序筛选基线，不是跨构建 ABBA。

| 模型 | 四个热请求（秒） | 中位数 |
| --- | --- | ---: |
| Qwen base，40 步 | 36.587991 / 36.825766 / 36.565593 / 36.563615 | 36.576792 s |
| Z base，8 步 | 6.361563 / 6.339210 / 6.315591 / 6.371458 | 6.350387 s |

两组 summary 均 complete，原始证据保留于本地忽略目录：

- `outputs/runtime-ane/qwen-base-zero-input-before/`
- `outputs/runtime-ane/z-base-zero-input-before/`

此前采样记录显示第一轮 Qwen 有 3.375 MiB 系统 swap-in，其余 trial 无
swap-in/out 增量；不能称全组零换页。没有对应 after，也没有新的 GPU 或
冻结路线对照，不更新当前最快推荐。

## 保留有用的回归

`tests/native/ane_ffn_test.cpp` 的 `correction_input_isolation_tests` 保留
CPU-only / CPU+NE 策略、scalar/SIMD 的四种组合，检查连续 base、变更层权重、
多 chunk、正负 adapter、第二 chunk 的 gate/up 分别失败、非有限 base 输入、
重新 self-test、headroom 溢出重试之后，base 均与独立重置图逐位相同。
移除依赖候选缓存/流量的断言，测试关注正确性而非实现细节。

候选撤出前已通过 4 项 host 与 8 项真实 Core ML/MLX 测试；这不代表
整模型质量或速度通过。撤出后的本次验证单独记在[整理记录](acceleration-cleanup.md)。
未来重启候选需重新实现并完成匹配前后测量、共享图 base/LoRA 切换及失败
恢复验证，不能仅凭节省理论写入字节数决定保留。
