# FLUX4B：当前完整运行预检，配对性能仍待补

2026-10-12，M4 Max64GiB，512²/4步、seed42，无额外LoRA。
当前只比较4B，不用9B或9月旧4B速度代替。

| 路线 | 已完成的证据 | 性能口径 |
| --- | --- | --- |
| Native GPU packed Q4/g64 | 一个进程cold/两fresh/repeat，4PNG，独立完整进程树内存，100实际DiT block events/request | constructor0.128s；cold3.959s；两fresh中位3.723s；repeat3.440s。尚非完整配对排名 |
| mflux0.22，上游同一整包 | 单cold完整PNG，387/905/266实际保存参数覆盖、4步callback | cold3.998s；不与native两fresh中位拼倍率 |
| mlx-serve26.10.1，同一整包 | HTTP200返回512² RGB PNG，视觉正常、独占服务终态退出 | 兼容性预检，无正式性能分母 |
| Unsloth Desktop / sd-server Metal | 上游支持FLUX，但本机暂无这条4B完整配对cohort | 尚未比较；不能由Qwen3.40×或mflux预检推算 |

Native来源是`mflux-community/flux2-klein-4b-mflux-q4`，revision
`794cd159538149ad9830848508c31f0ea7088e58`，不是原BF16或本地重做量化。
Native派生布局完整保留DiT/encoder压缩payload，包括量化embedding和
已存储的F32 RoPE常量；VAE逆转卷积轴序后逐tensor核对bitpattern。
**VAE有8个Q4/g64 attention线性层，卷积/norm为BF16**，不能标整VAE BF16。
配置来自固定原4B repo/revision，不复用9B或猜测架构。387 DiT /
905 encoder /266 VAE source tensors完整保留；原文件、失败receipts均不改写。

Native两fresh阶段中位text encoder0.275s、DiT3.174s、VAE0.263s：
当前主要耗时在packed DiT，继续优先筛选编译量化GPU图/一次预解码
候选，再补source-bound GPU/ANE；尚无本4B整包混合路线速度结果。
新实现对g32/64/128和Q4/Q8的6个GPU小样例，embedding逐行解码、
整矩阵/切片投影及FP16/BF16解码精确核验；Qwen3原dense路径12个
旧代码/当前代码resident/submitted parity case通过。

Native本次tree physical peak4,208,315,944bytes，compression/swap均0，
只代表这一进程窗口，不与mflux allocator peak5.76GB混成内存胜率。
原生`gpu_graph`字段在这次预检反映自动compile请求标志，而packed
source实际compiled-single events为0；本页只称packed eager，不称
已执行compiled或strict BF16。后续计时应以实际branch events为准。

[Native完整预检](../design/validation/flux4-native-saved-q4-precheck-20261012.json)、
[参考兼容预检](../design/validation/flux4-mflux-saved-q4-precheck-20261012.json)、
[Unsloth覆盖边界](OPEN_SOURCE_COMPARISONS.md#unsloth支持与本机覆盖不是一回事)。
本页明确尚无当前配对排名，不再将其列作自动扩展任务。按用户最新
收敛要求，现有实测统一进入[性能报告](PERFORMANCE_REPORT.md)，
历史与预检分开，未测倍率不填写。
