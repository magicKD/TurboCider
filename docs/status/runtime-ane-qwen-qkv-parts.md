# Qwen Q/K/V 三源直填 runtime MatMul 权重槽

前一轮[打包 QKV 组件](runtime-ane-qwen-qkv-packed.md)在计时前生成一份
`[12288,4096]` GPU 权重。产品若保留32层副本，额外理论值约 3 GiB。
参考仓库旁 `../references/vpipe/generative-models/shared/ane-ffn.h` 的多源单槽
布局，本轮让 runtime MatMul 的一个 FP16 输入槽直接按 Q→K→V 顺序
转换三个独立的 `[4096,4096]` BF16 checkpoint 张量。无需先构造 GPU
打包矩阵；槽只容纳当前层权重。旧的单源 MatMul/FFN API 未改变。

新接口 `RuntimeGraph::stage_matmul_parts` 在写入前检查非空、每段存储
边界与列数、总行数刚好覆盖完整槽。输入仍借用到 stage 完成；失败清除
staged 标志，不能凭之前完整槽执行。数值/顺序、短缓冲、缺段、失败后
拒绝旧权重与恢复路径均有小型 Core ML/MLX 集成回归。

使用同一 Qwen BF16 block-0 Q/K/V 权重和同一 checkpoint-independent
MatMul 图：合成 4096-row BF16 输入，ANE 1536 rows、GPU 2560 rows，
K1024/N512。GPU-only 由三次独立 GEMM 与 concat 构成；并行计时须
等三投影全部输出拼接完成。两个独立进程，各排除两次 warmup，再交替测
十对。20 对池化中位数：

| GPU-only | 并行投影 | 并行含 staging | GPU / 含 staging |
| ---: | ---: | ---: | ---: |
| 27.955 ms | 18.851 ms | 19.808 ms | 1.411× |

staging 本身中位数 0.945 ms；一个 graph 的槽分配回执 150,994,944
bytes、估计总额 432,013,312 bytes。这只是 graph 的估算，不是请求级
driver/wired 压力证明。两次 trial 退出 0、20 对无 overflow retry，
relative L2 0.000985、cosine 0.99999946；原始样本、进程预检、前后
系统内存及 hash 在被忽略的
`outputs/runtime-ane/qwen-qkv-parts-screen.jsonl`，`summary.status=complete`。
受测 probe SHA256 为
`b906becbaccf7f2de881365cd28bedd22def6b4025fe6cbc8caded2dd55e4c89`；
checkpoint SHA256 为
`89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc`。

`make test-runtime-ane` 的 5 host、12 Core ML/MLX 集成项通过，
`make test-acceleration-contract` 通过。产品库仍为 `28ff6aaf…`，
没有接入产品 QKV 或改变默认路由。单个 QKV graph 若与现有 FFN graph
并存，还需**联合**内存准入与 ANE 串行调度；更重要的是 Q/K norm、
RoPE、attention 和整个 block 的产品级 GPU-only 对照。只有完整窗口
仍为正、失败可回退且请求级内存合格，才可启用 QKV tier。物理 ANE
驻留仍未知。
