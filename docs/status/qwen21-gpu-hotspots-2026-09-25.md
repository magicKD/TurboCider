# Qwen Image 2.1：512² 纯 GPU 去噪热点诊断

Apple M4 Max／BF16 Qwen Image 2.1，512×512 文生图、seed 42、5 步。
同一默认提示词在单机 resident Session 中 load-only prepare、2 步 warmup，
再测一次 5 步请求。两条诊断均需显式环境变量，默认 GPU 路线保持不变：

- `TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS=1`：在每个纯 GPU block 前后
  同步，逐层输出 JSON 行；不用于混合 FFN 路径。
- `TURBOCIDER_QWEN21_PROFILE_GPU_OPS=1`：仅第 0 层关闭整个 block 的
  `mx::compile`，在算子组边界同步并逐组输出 JSON 行；其余 block
  仍编译。不能当作真实未插桩内核的耗时。

两个开关互斥，不能同时使用；它们都不会自动改变正常请求的执行策略。

| 运行方式 | 5 步去噪 | 请求墙钟 | 解码层的观测 |
| --- | ---: | ---: | --- |
| 默认未插桩 GPU | 5.388 s | 6.017 s | — |
| 每层同步 | 5.485 s | 6.098 s | 大多数层约 33–34 ms／层；偶发约 35–36 ms |
| 仅第 0 层逐算子同步 | 5.448 s | 6.045 s | 下表 |

第 0 层逐算子诊断对 5 次 decode 和 3 次 prefill 求均值；分母包含
warmup、测量和取消前的请求，不能把这八次当成八个独立性能实验：

| 第 0 层阶段 | decode 平均 | prefill 平均 |
| --- | ---: | ---: |
| attention input norm／AdaLN | 0.387 ms | 0.553 ms |
| Q/K/V 投影 | 8.237 ms | 10.250 ms |
| Q/K norm＋RoPE | 3.535 ms | 3.971 ms |
| attention／KV 拼接 | 2.060 ms | 1.696 ms |
| attention output 投影 | 2.781 ms | 3.032 ms |
| attention 残差＋FFN 输入 norm | 0.473 ms | 0.768 ms |
| FFN gate/up＋SwiGLU | 14.841 ms | 17.121 ms |
| FFN down＋残差 | 7.743 ms | 10.403 ms |

诊断状态下 FFN 两段合计约 22.58 ms、第 0 层分段合计约 40.06 ms，
它提示优先考察 FFN gate/up、down 的真实 GPU kernel／精度策略，
而非只继续压 RoPE。注意分段诊断关闭单层编译并强制 GPU 同步，
而默认每层约 33–34 ms；这些数值既不是精确生产算子占比，
也不能乘以 32 推算真实总耗时。先前 QKV 权重合并及 Z-Image MPP
FFN 投影在本模型的成对 5 步对照中均未稳定提速，不能直接推广。

三条运行的输出 PNG SHA-256 都是
`a8606387e51ea09b42730a4a4f23666afcbad4009c890c457364fb0e0399b021`；
这是单个 seed 的诊断等价性，不等于多种编辑任务的画质验收。
原始请求报告／图片分别在被 Git 忽略的
`results/qwen21/session-gpu-profile-{control,blocks,ops}-512-5/`。
当前没有新的纯 GPU 默认提速；1.1–1.2× 目标仍未达到。
