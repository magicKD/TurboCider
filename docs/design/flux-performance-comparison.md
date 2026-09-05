# FLUX.2 原生框架与原始 flux2-engine 性能对比

2026-09-05。Apple M4 Pro / 48 GiB / macOS 26.6，MLX 0.32.0。模型为本地官方 FLUX.2-klein-4B，未下载模型。比较的是未修改的外部 `mac_image_generation/flux2-engine` 与新的 C ABI 推理实现。

## 结果

同一英文 fox prompt、512×512、4 steps、seed 42、动态文本28 tokens，生成图1024 tokens。每个引擎两次新进程，每进程7次请求；排除各进程首请求后，共12次暖样本。AB/BA 交换运行顺序，推理作业严格串行。GPU 与混合各自在同精度/切分路线内比较。

| 路线 | 原始暖中位数 | TurboCider 暖中位数 | TurboCider 相对变化 |
|---|---:|---:|---:|
| GPU BF16 | 6.073 s | 4.758 s | 快 21.66% |
| GPU + 20 MLP Core ML 分区 | 2.870 s | 2.896 s | 慢 0.89% |

本次预设工程门槛为暖中位数回归不超过5%，两条路线均通过。混合路线不是被报告为更快，它与原始引擎基本相当。结果只适用于本机该负载，不是全部尺寸/芯片保证；12个样本无法可靠估计长期尾延迟或给出强统计置信结论。

混合最终两轮明细：

| 顺序 | 引擎 | 首请求 | 暖中位数 | 样本 p95（nearest-rank） |
|---|---|---:|---:|---:|
| A→B | 原始 | 4.494 s | 2.864 s | 2.977 s |
| A→B | 原生 | 4.997 s | 2.889 s | 2.901 s |
| B→A | 原生 | 3.550 s | 2.898 s | 2.914 s |
| B→A | 原始 | 3.947 s | 2.876 s | 3.209 s |

构造器单独计时。以上首请求不包含构造器，也不是清空 OS 文件缓存后的冷启动。OS/Core ML 磁盘缓存已有运行历史，没有清缓存，没有通过删除系统缓存制造“冷”数字。请求墙钟包含原生 JSON 边界、执行和 PNG 导出；原始引擎使用 public generate API，包括导出。原生测试驱动虽为 Python ctypes，但推理为同一个 C++ 动态库，不调用 Python 模型实现。性能运行关闭 tensor dumps 与原生逐块回调。

## 同条件与算法限制

- 原始 `precision=bf16`、persistent session、`clear_mlx_cache_between_requests=false`。原生 resident、相同 prompt conditioning 缓存，两者重复请求缓存均可命中。
- 混合均使用同一个本地 schema2 manifest、1088 rows、20 single block INT8 per-channel MLP、FP16 I/O，GPU 执行 attention，公开 Core ML `CPUAndNeuralEngine` 执行 MLP。
- 四步完整去噪、同权重、同图像尺寸，不通过减层、裁剪 prompt 或换更低精度取得 GPU速度。
- 原始默认 GPU 使用完整 DiT 的 MLX compile；原生 GPU 保留可逐块取消/观测的执行边界，数值以原始 mflux eager 子图为 oracle。两种融合范围的浮点舍入不同，因此不把相同 seed 当成默认 compiled GPU 最终像素必须逐值相同的保证。
- 原生的 English512、Chinese-fixed、transform、reference、多参考与 strength1 的中间 tensor fixtures 逐值对齐参考。PNG 导出另行检验，不能由中间张量正确推定导出正确。

## 已定位并修复的 overhead

1. **Core ML 输出复制。** 早期每次 MLP 后创建独立 MLX 输出，512负载约多复制534MB/请求。改成 MLX 持有共享 output backing，消费者每块完成后才复写，Core ML 拒绝 backing 时才复制并记录。最终测量中 fallback复制为0。
2. **GPU attention 子图重建。** 原引擎预编译混合 attention 子图；原生最初每块重复构建该图。加入按 shape/dtype 缓存的 MLX compile，权重作为显式输入，避免缓存捕获上一个模型 Session。MLP仍与GPU异步重叠，join边界不变。
3. **会话驻留。** 重复prompt保留conditioning、DiT/VAE及Core ML会话；换prompt先释放大权重再编码。编译/加载/预热不进入每一个去噪step。
4. **框架边界。** JSON只在请求/结果解析，不逐token传递；热路径为native张量调用。持久任务进度节流到约1秒写盘。跨进程锁只在请求准入/完成时获取/释放。

在子图缓存前的两轮混合测量中，中位数回归4.15%；缓存后复测为0.89%。由于温度/系统负载未严格受控，不把全部差值都归因于单个优化。报告保留优化前各轮，避免挑选最快一次。

还发现 PNG 导出的归一化精度差异：参考先用 VAE dtype 做 `/2 + .5`，再转FP32量化，旧原生提前转FP32，导致最大1个8-bit灰度级差异。已修正导出顺序，见最终图片核验记录；这不改变推理步骤或中间张量。

## 证据与复现

- [GPU及优化前各轮](validation/rewrite-performance-before-graph-cache.json)
- [最终混合各轮](validation/rewrite-hybrid-performance.json)
- [服务队列/取消/跨进程及崩溃恢复](validation/rewrite-service.json)
- [生命周期与缓存](validation/rewrite-lifecycle.json)
- [多参考图数值](validation/flux-multi-reference-parity.json)
- [图生图强度边界](validation/flux-transform-strength1-parity.json)

完整每次计时与 Core ML 指标位于本机 `outputs/rewrite-validation/final-comparison/` 与 `final-compiled-comparison/`；后者为优化后混合。工具 `tools/native/benchmark_comparison.py` 可复现AB/BA；`--modes gpu hybrid`默认两条，`--modes hybrid`仅混合。门槛不通过脚本退出非零。

原始flux2-engine的GenerationRequest没有图像输入，因此图生图/参考编辑通过mflux oracle验正确性，不能伪造对应的原始引擎图像性能对比。新shape、不同prompt连续请求、长会话、多参考大图、内存压力和其他机器仍需各自建立性能基线。

混合legacy artifact仅有checkpoint path+size，缺少源SHA，实际ANE驻留未测定。MLX峰值不包含Core ML/OS/file cache；不据此声称混合总内存更低。当前提供显式GPU/ANE配置和已验证子图边界，尚非通用自动分区编译器。
