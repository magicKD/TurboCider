# FLUX4B：当前Q4来源的预解码与GPU/ANE完整结果

2026-10-13整理，M4 Max64GiB，512²/4步、seed42，无额外LoRA。
当前只比较4B，不用9B或9月旧4B速度代替。

## 新候选：只预解码single投影，不重跑原packed或参考

现有hybrid GPU图要求dense投影，不直接接受Q4 UInt32。新视图只将
20个single-block的两种投影（40个）用MLX0.32.3官方g64/Q4解码成BF16，
删除对应80个scales/biases tensor；其它267个DiT tensor逐位保留。
encoder、VAE和tokenizer只读链接原native视图，原模型不改写。
派生307tensor/5.707GB，20个ANE图绑定其canonical checkpoint SHA；
514个native source inputs匹配既有库，没有放宽来源或geometry guard。

| 同一新派生视图，完整PNG | 四fresh中位 | cold，正/反 | repeat中位 | tree physical peak |
| --- | ---: | --- | ---: | --- |
| compiled GPU | **3.483532s** | 3.818 /3.562s | 3.200s | 7.72–8.03GB |
| **GPU/ANE W8/F16，6144/9216前缀** | **2.574438s** | 9.081 /5.362s | 2.301s | 8.06–8.25GB |

GPU→hybrid→hybrid→GPU，4进程/16完整PNG/4complete memory，warmup0。
每请求100个真实DiT block events；GPU实际80个compiled-single events，
混合实际80个runtime calls，无fallback/失败。四份最大内存gap均小于
111ms，swap-out/压缩均0。混合对自身GPU为**1.353×，省26.1%**。
fresh DiT约GPU2.93s/混合2.03s；encoder约0.28s、VAE约0.26s，
本轮收益主要发生在DiT，不是重扫SHA、跳过encoder或省略导出。

32项完整原尺寸/原通道PNG比较在计时结束后执行：8对正反序同路线全部exact；
新GPU与旧packed输出在本组4个请求/有限prompt上exact，不推论全场景。
混合对GPU的RGB MAE最高**2.573/255**、SSIM最低**0.98059**，完整图
为正常雪林狐狸等输出；不是bit-exact混合或广泛语义/画质资格。
图是per-channel INT8权重、FP16输入与计算，**不是W8A8激活量化**；
CPU+NE compute-units设置和成功calls也不等于物理ANE占用/INT8 MAC trace。
预解码增加模型存储与驻留；冷态更贵，推荐为opt-in热态配置。

[完整来源/编译/runtime/媒体/内存/误差证据](../design/validation/flux4-derived-hybrid-current-20261013.json)。
最优配置已保存至[best-local-configs](best-local-configs-20261011.json)。
原packed与外部数据仍属于各自旧窗口，没有用本轮派生结果制造新的
同窗口外部排名；原BF16 mixed1.628s也不替代本轮结果。

仅按“TC取现有最优、外部已测不重跑”的记录对照：

| 已保存参考，512²/4步 | 参考两fresh中位 | TC派生混合四fresh中位 | 记录差距 |
| --- | ---: | ---: | --- |
| mflux0.22，原savedQ4/g64 | 3.810622s | 2.574438s | **1.480×，省32.4%** |
| mlx-serve26.10.1，原savedQ4/g64 | 4.304634s | 2.574438s | **1.672×，省40.2%** |

原系数11个source SHA与派生链逐一匹配；但TC投影预解码/混合改变
表示与数学，计时窗口和样本数也不同。这个表是已保存记录差距，
**不是新同窗口配对倍率或全局最优**，不加入主报告的paired优势榜。
Comfy本地原saved-index尚无完整任务分母，不由上述参考推算。

## 保留的原packed与兼容参考窗口（2026-10-12）

| 路线 | 已完成的证据 | 性能口径 |
| --- | --- | --- |
| Native GPU packed Q4/g64 | 一个进程cold/两fresh/repeat，4PNG，独立完整进程树内存，100实际DiT block events/request | constructor0.128s；cold3.959s；两fresh中位3.723s；repeat3.440s。尚非完整配对排名 |
| mflux0.22，上游同一整包 | 单cold完整PNG，387/905/266实际保存参数覆盖、4步callback | cold3.998s；不与native两fresh中位拼倍率 |
| mlx-serve26.10.1，同一整包 | HTTP200返回512² RGB PNG，视觉正常、独占服务终态退出 | 兼容性预检，无正式性能分母 |
| Unsloth Desktop / sd-server Metal | 上游支持FLUX，但本机暂无这条4B完整配对cohort | 尚未比较；不能由Qwen3.40×或mflux预检推算 |

随后只补已兼容的两个参考热态，不重跑TC：mflux0.22 saved-Q4两fresh
中位 **3.810622s**（cache0）；mlx-serve26.10.1原整包SSE/PNG两fresh
中位 **4.304634s**。两独立进程、8完整512² PNG、两complete memory；
constructor/readiness、cold、repeat新seed独立保留。相对已有TC Q4
两fresh3.723469s的差距很小/有限，不宣称新同窗口配对或稳定全局领先，
也不拿历史BF16 mixed1.628s除当前Q4参考。
[完整参考热态与原事件/媒体/内存复核](../design/validation/flux4-reference-warm-current-20261012.json)。
收集器最初错误地筛选`denoise`；本版本实际SSE阶段名为`Generating`，
四步1–4全部存在、媒体与memory早已complete/exit0。新独立auditor只修正
事件提取，原failed summary不改写、不降任何门槛、不重新推理。
这不是恢复失败模型或inconclusive memory；原验收原因与SHA链均保留。

Native来源是`mflux-community/flux2-klein-4b-mflux-q4`，revision
`794cd159538149ad9830848508c31f0ea7088e58`，不是原BF16或本地重做量化。
Native派生布局完整保留DiT/encoder压缩payload，包括量化embedding和
已存储的F32 RoPE常量；VAE逆转卷积轴序后逐tensor核对bitpattern。
**VAE有8个Q4/g64 attention线性层，卷积/norm为BF16**，不能标整VAE BF16。
配置来自固定原4B repo/revision，不复用9B或猜测架构。387 DiT /
905 encoder /266 VAE source tensors完整保留；原文件、失败receipts均不改写。

Native两fresh阶段中位text encoder0.275s、DiT3.174s、VAE0.263s：
当时主要耗时在packed DiT。当前已补上述一次预解码/来源绑定混合
候选；直接packed UInt32 hybrid仍未实现，不把新派生结果标成它。
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
本页有新派生视图内部GPU/混合配对，但没有新同窗口外部排名。按用户最新
收敛要求，现有实测统一进入[性能报告](PERFORMANCE_REPORT.md)，
历史与预检分开，未测倍率不填写。
