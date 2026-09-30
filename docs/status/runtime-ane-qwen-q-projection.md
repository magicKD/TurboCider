# Qwen 长序列 runtime-weight Q 投影组件筛选

Z fused QKV 的 1024-row 分区全为负结果；Qwen 1024² prefill 的 Q/K/V
权重则是分开的 `[4096,4096]`。依照 Phase 2 的严格门槛，先独立筛选
Q 投影，**尚未接入产品 QKV tier**，也不修改默认路线。

## 范围与身份

- 真实 BF16 checkpoint `transformer_blocks.0.attn.to_q.weight`；safetensors
  header 声明 `[4096,4096]`。显式 `w` bundle 优先，其次 Qwen 的 Q 键，
  最后原有 Z fused QKV 键；小型 BF16 集成回归验证这几个分支。
- 输入是固定种子的 **4096×4096 BF16 合成激活**，不是模型请求捕获的
  attention norm 输出。每个 MatMul 图的权重均为 runtime 输入槽，输出
  包含 GPU head 与 Core ML tail 的拼接；不含 K/V、Q/K norm、RoPE、
  后续 attention 或产品中的 MLX 图编译/调用成本。产品默认使用普通分离的
  Q/K/V；实验性融合 Metal QKV 不能用此 GPU-only 分母代替。
- `cpuAndNeuralEngine` 是所请求的 Core ML policy；物理 ANE 驻留未知。
  图固定单个 chunk、K1024/N512；先屏蔽 staging 记录并行窗口，再单独
  报告 staging 暴露总耗时（这里没有 attention 窗口隐藏 staging）。
- 每组两个独立进程，每进程排除两个 warmup、交替顺序测量十对，表中每行
  池化 20 对的中位数。两次 screen 各自有前后 SHA、逐样本 JSONL、
  竞争推理预检和系统内存前后快照；不可把不同配置的 GPU head 几何视作相同。

| ANE rows / GPU rows | GPU-only ms | 并行 ms | 含 staging ms | GPU / 含 staging |
| --- | ---: | ---: | ---: | ---: |
| 512 / 3584 | 9.371 | 8.645 | 9.005 | 1.041× |
| 768 / 3328 | 9.368 | 8.172 | 8.529 | 1.098× |
| 1024 / 3072，首组 | 9.359 | 7.547 | 7.906 | 1.184× |
| 1024 / 3072，邻近组 | 9.373 | 7.576 | 7.917 | 1.184× |
| 1280 / 2816 | 9.369 | 7.068 | 7.405 | 1.265× |
| 1536 / 2560 | 9.364 | 6.662 | 7.011 | 1.336× |
| 2048 / 2048 | 9.370 | 8.329 | 8.690 | 1.078× |

1536 的两个独立 trial 的含 staging 中位数分别为 6.928 和 7.645 ms：
其间预检发现另一 ComfyUI 推理忙碌，等待其空闲后继续。预检不是逐时段
隔离证明，因此 **1.336× 只作候选信号**；不因此宣布最优 chunk。
1280/1024 的两个试次更接近。14/14 trial 正常退出，140 对样本，
overflow retry 均为零；relative L2 为 0.000566–0.001134。
这说明单独的长序列 Q 投影有组件级正收益，但不足以推断整个 QKV critical
path、完整 block、1024² 图片质量或整请求收益。

## 可复核证据与下一门槛

首次与邻近行数 screen 分别保存在被忽略的
`outputs/runtime-ane/qwen-q-projection-screen.jsonl`、
`outputs/runtime-ane/qwen-q-nearby-screen.jsonl`；同目录有各自的 `.py`
入口和六个离线 MatMul 图。两份证据的 `summary.status=complete`。
第一份记录的 probe SHA256 为
`c16c0473d250607c485b0954bca2d950f401f3b11dc189e88a2175fcf5bca6f8`；
checkpoint 为
`89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc`。
`make test-runtime-ane` 的 5 项 host 与 11 项 Core ML/MLX 集成测试通过；
原 sandbox 运行因系统禁止 Core ML 临时目录与 Metal 访问而失败，授权
运行后通过。没有重新构建产品库或声称物理内存资格。

接续的三权重打包 QKV 组件对照另见
[QKV 组合投影筛选](runtime-ane-qwen-qkv-packed.md)。此处 Q 的独立成绩
仍不可与那组不同宽度/不同 GPU baseline 的时间直接相加。
