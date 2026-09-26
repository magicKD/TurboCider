# Qwen Image 2.1：纯 GPU 最后一步 FFN 复用实验

512² BF16 Comfy checkpoint、Apple M4 Max。显式设置
`TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN=1`，请求须为 `execution=gpu`、
`allow_approximation=true` 且至少 3 步；编辑目前只允许 1–3 张
256px 缩放参考图。默认纯 GPU 与 W8A8 混合路线均不改变。

请求内仍使用原有 prefix KV cache。倒数第二步的 32 层 FFN 计算
照常运行，并保存每层 `gate/up → SwiGLU → down` 输出；最后一步
各层重算 attention 和当前步调制，在残差更新中复用上一步的
FFN 输出，因此这**不是等价 kernel fusion**，而是一步的时间近似。
每次请求独立持有缓存；条件变化会清空它；准备、2 步 warmup 和
取消重试不会跨请求借用结果。`plan.algorithm_approximations` 显式
标记 `qwen21_gpu_reuse_final_ffn`，实际生成的
`acceleration_selection` 也标记近似。默认关闭。

各路线独立进程 `prepare`、2 步 warmup、然后相同条件连续生成两轮；
墙钟包括 VAE、PNG 导出和两路线相同的验证张量转储 I/O，
不含加载与准备。1/2/3 参考图统一采用 256px 缩放；
对照与候选共享模型、提示词、seed、输入文件与缓存命中条件。

| 512² 场景 | GPU 墙钟两轮 | FFN 复用墙钟两轮 | 两轮合并墙钟比 | 两轮合并去噪比 | RGB correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 文生图，5 步，狐狸/42 | 5.973/6.008 s | 5.344/5.294 s | **1.126×** | 约 **1.140×** | 0.99867 |
| 1 参考编辑，5 步，茶壶/42 | 6.297/6.314 s | 5.608/5.599 s | **1.125×** | 约 **1.133×** | 0.99980 |
| 2 参考编辑，5 步，茶壶/17 | 6.700/6.702 s | 6.005/6.047 s | **1.112×** | 约 **1.127×** | 0.99952 |
| 3 参考编辑，5 步，茶壶和龙贴纸/17 | 7.086/7.072 s | 6.406/6.385 s | **1.107×** | 约 **1.117×** | 0.99952 |
| 文生图，40 步，同狐狸/42 | 43.128/43.175 s | 42.505/42.476 s | **1.016×** | 约 **1.016×** | 0.99984 |
| 2 参考编辑，40 步，同茶壶/17 | 44.718/44.743 s | 44.070/44.146 s | **1.014×** | 约 **1.015×** | 0.99990 |

全部六组均转储并核对 `text`、`initial`、以及编辑的每张
`reference` 张量逐字节相同；各轮均命中相同条件缓存。每条路线的两轮 PNG
一致。默认关闭新开关时，双参考 5/40 步 GPU PNG SHA 与修改前
保存的对应 GPU 输出一致，因此这项实验没有改变默认 GPU 出图。

5 步文生图的两张图都偏软；1/2/3 参考编辑的主体、顺序、壶嘴与
把手以及三参考中的龙贴纸在本样本中肉眼接近，细节并非逐像素
一致。40 步狐狸、双壶均清晰，对照和候选目视非常接近；狐狸和
双参考 RGB RMSE 分别为 `0.00445`、`0.00418`（归一化像素）。
一个提示词/seed、部分 5 步图偏软，不能宣称普遍图像编辑质量。
40 步只省去一个末步 FFN，约 1.5% 收益符合机制；若要在长请求
达到 1.1×，还需其它 GPU 优化并重新做画质验收。

曾筛选复用末两步的更激进版本：5 步狐狸暖请求
`5.973/6.008 s → 4.608/4.612 s`（约 1.299×），去噪约
1.34×；转储输入相同，但候选狐狸比基线明显更模糊、毛发呈颗粒状
（RGB RMSE `0.03626`，correlation `0.98966`）。因肉眼质量
下降过大，**撤回该实现**，不开放这个档位。原始图片和对照报告在
`results/qwen21/session-gpu-reuse-final2-512-5*/`。

原始逐轮 JSON/PNG 和匹配报告位于忽略目录
`results/qwen21/session-gpu-reuse-{control,final,edit1-control,edit1-final,edit2-control,edit2-final,edit3-control,edit3-final}-512-{5,40}*/`
及 `results/qwen21/session-gpu-reuse-*-comparison.json`。
`qwen21_compare_sessions.py --candidate-execution gpu` 检查候选
`plan` 带有近似标记，`--verify-dumps` 可额外检验文本/噪声张量。

另用正式 `turbocider generate` 命令（非 Session 基准工具）在显式开关下
分别完成 512²／5 步文生图与双参考编辑；两次均导出有效 PNG，
返回 `actual_denoise_steps=5`、`execution=gpu`，且计划和运行结果
均标记最后一步 FFN 复用。编辑图保留了左侧米色、右侧蓝色两只茶壶。
本次为独立冷进程 CLI 功能冒烟，不以其墙钟替代上面的常驻 Session
配对加速比；本地产物在忽略目录
`results/qwen21/cli-gpu-reuse-{final,edit2}-512-5*`。
