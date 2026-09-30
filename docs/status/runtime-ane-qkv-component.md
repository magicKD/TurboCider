# Z-Image fused QKV 的 runtime-weight MatMul 组件筛选

设计稿的 Phase 2 将 QKV 定义为**更严格的机会性 ANE tier**：注意力必须等
完整 QKV，只有整段并行投影短于 GPU-only 才值得启用。本次先用现有
`MatMul` runtime-weight 图和 `ane-runtime-probe` 验证 Z-Image 的 1024-row
短序列，不修改产品后端或默认路由。

## 输入、图与计时边界

- Z BF16 checkpoint 的真实 `noise_refiner.0.attention.qkv.weight`，形状
  `[11520,3840]`。probe 现在记录 `weight_source`，八个 trial 均实际报告
  这个键，而非从 `real_weights=true` 猜测使用了哪一个权重。
- 1024×3840 BF16 **合成**输入，固定随机种子与 normal×0.5；不是从
  真实请求捕获的 QKV 激活。Graph 为固定行数、K1024/N512、权重通过
  输入槽提供，不含 checkpoint 常量。`cpuAndNeuralEngine` 不证明物理位置。
- `c256×2` 和 `c512×1` 都给 ANE 512 rows、GPU 512 rows；
  `c768×1` 为 768/256；`c896×1` 为 896/128。
- 顺序 256→512→768→896→896→768→512→256；八个独立进程，
  各排除两次 warmup 后交错测量十次 GPU-only/并行投影，总共 80 对。
  每次启动前检查竞争推理，原始逐样本、进程快照与前后系统内存留档。
  不把不同配置 GPU head 几何改变误说成单纯 chunk 次数的差异。

下面是每配置**池化 20 个实测样本**的中位数，单位毫秒。GPU-only 为
probe 自己编译的 `x @ Wᵀ`，并非 Z 产品中融合 QKV 准备、norm/RoPE 的
完整 block，也没有包括后续 attention。`并行`含输入转换、输出收集和
GPU/ANE join；`含 staging` 额外计入转换权重的暴露成本，此 probe 没有
可供隐藏 staging 的 attention 窗口。

| ANE chunk × 次数 | GPU-only | 并行投影 | 并行含 staging | prediction | GPU 完成后 join |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256×2 | 6.284 | 9.615 | 10.482 | 6.735 | 5.305 |
| 512×1 | 6.281 | 7.867 | 8.690 | 5.038 | 3.438 |
| 768×1 | 6.280 | 10.134 | 10.992 | 6.471 | 7.005 |
| 896×1 | 6.282 | 11.588 | 12.452 | 7.451 | 9.084 |

最好的 512-row 分区在**不计 staging**时仍比 GPU-only 慢约 25.3%，
计入 staging 慢约 38.4%；扩大 ANE 行数更慢。prediction 单独约
5.038 ms 不能代表完整 QKV 的 critical path：GPU head/绑定、输出处理和
等待使并行投影回升到 7.867 ms。八 trial 全部退出 0、无 overflow retry，
relative L2 为 0.001112–0.001468，最小 cosine 0.99999899。
这些是数值有效的**负性能结果**；不为此增加产品 QKV 分支或默认开关。
不外推到 Qwen 长序列、别的形状、GPU kernel 或真实完整 block。

为能核对权重选择，独立 probe 新增 Z fused QKV 的 checkpoint key 绑定
和 `weight_source` receipt，显式 `w` bundle 仍优先；临时小型 BF16
safetensors 的两个分支有集成回归。`make test-runtime-ane` 退出 0
（5 host、11 Core ML/MLX 集成）；`make test-acceleration-contract` 亦退出 0。
产品库 `28ff6aaf…` 未重建。
最终 probe SHA256：
`fe5492e1154ac2b4f1104e675961f224a7cc63a2d55cd861003acfec621f31fc`；
checkpoint SHA256：
`2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6`。
原始证据及完整前后 artifact/hash 为本地被忽略的
`outputs/runtime-ane/z-qkv-source-verified-screen.jsonl`；状态 `complete`。
前两次无权重来源 receipt 的探索保留在同目录其他文件，不作为本表分母。
这里只保存系统内存前后快照，没有独立进程树/driver-wired 或物理 ANE
采样，不宣称内存资格。下一项可检验的是不同模型或真实长序列 QKV，
但须先做同样的组件级 GPU-only 对照，不直接接入产品路径。
