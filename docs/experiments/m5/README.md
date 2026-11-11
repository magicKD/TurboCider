# M5 Pro / 24 GiB 实验归档

本目录保存 Apple M5 Pro / 24 GB 上的实验：2026-09-16 与 2026-09-20 的
GPU/ANE、Z-Image streaming 记录，以及 2026-11-11 的 LTX 2.5 发布验证。
历史正文中的机型白名单、默认策略、
客户端行为和“当前”均属于原构建的历史事实，不是当前版本的支持承诺。

## 记录与边界

| 记录 | 证据范围 | 保留的限制 |
| --- | --- | --- |
| [GPU / ANE 优化](m5-ane-adaptation.md) | 2026-09-16 FLUX 自动混合、Z-Image BF16 后缀读取、编译及分区选择 | 各行是独立实验；Z-Image resident 与部分 1024² 混合路线较慢，不能合并成统一加速结论 |
| [INT8 ConvRot 流式加载与预读](z-image-int8-streaming-2026-09-20.md) | 2026-09-20 固定 512² 工作负载、6/8 GiB 预算下的预读深度，以及后续构建和客户端回归 | 6 GiB 两层预读约快 6.9%；预算和 MLX 峰值不代表进程内存上限，ANE 请求偏好不证明物理驻留 |
| [通用 streaming 接入与对照](z-image-generic-int8-streaming-2026-09-20.md) | 原实现与通用框架的冻结 library、ABBA 和引擎生命周期对照 | 私有 manual layout；60 张正式输出在执行模式内与旧实现一致，连续复用引擎的 ANE 延迟尚未对齐，不能宣称稳定提速 |

原报告中的设备、精度、分区、提示词、样本数、失败或跳过项继续绑定各自
实验。BF16、INT8、GPU、混合路线和不同引擎生命周期分别比较；单设备结果
不外推到其他 M5 配置。上述三份历史报告没有在本次 LTX 验证中重新测试。

## 复现资料与相关入口

正文保留原命令、来源提交和已知验证限制。`local-experiments/`、`outputs/`
中的原始日志、请求、图片、哈希索引与冻结 library 可能仅在原实验机器可用；
已退役的产物仍按原文标注，归档没有复制或重建它们。

[Streaming 设计目录](../../design/streaming/README.md)保留通用框架合同；
其他设备的记录见 [M1 Pro](../2026-09-20-m1-streaming.md) 和
[M4 Pro 集成历史](../2026-09-27-streaming-integration-history.md)。
当前运行方式以[用户指南](../../public/README.md)为准。

三份完整正文只维护在此目录，原设计／状态路径保留章节跳转。新增实验遵循
[归档规则](../README.md)，不要把历史的本地证据要求当作提交原始大文件的依据。

## LTX 2.5 加速与桌面发布验证 · 2026-11-11

设备实测为 Apple M5 Pro，15 核 CPU、16 核 GPU、24 GB 内存，macOS
26.5.1（25F80）。请求使用原始 INT8 ConvRot 权重、FP32 scales、BF16
边界和 RNE127 激活量化；没有再次量化模型。PrunaVAED 本身是裁剪解码器，
需要允许近似；其 TensorOps 卷积采用严格 FP32 累加。

当前 M5 默认优化包括 BF16 输出融合、旋转与量化融合、符合条件的视频
AdaLN 融合、stage workspace 复用、第二阶段 Q/K RMSNorm/RoPE/layout
融合，以及 connector 命令批处理。严格内存 hooks、预算不足和不符合几何条件
时保留对应回退路径。Q/K 融合默认要求至少 2048 行，第一阶段没有默认开启。
PrunaVAED 融合 norm/SiLU，并对符合条件的卷积使用 TensorOps 和优化输入读取；
扩张投影、未对齐的向量读取保留标量路径。Sol、空间 K/V pooling 和其他稀疏
模式仍是显式实验，默认关闭。

### 已有单元 profile

以下结果来自独立测量批次，不能相乘，也不能外推为完整视频的加速比。
connector 使用真实权重、单 block、四组平衡 ABBA/BAAB；DiT 使用真实权重和
连接后的文本、合成 BF16 hidden states，排除首轮后测量。PrunaVAED 使用真实
卷积系数和实际几何的固定合成输入，每个算子四组平衡对照。

| 范围 | 原路径 | 优化路径 | 加速比 |
| --- | ---: | ---: | ---: |
| connector video block 0 | 26.126 ms | 24.886 ms | 1.050× |
| connector audio block 0 | 12.187 ms | 10.595 ms | 1.150× |
| DiT 第二阶段完整 block 16，workspace + Q/K 组合对照 | 102.422 ms | 98.865 ms | 1.036× |
| 上一行中的 video self attention，含投影 | 32.487 ms | 30.055 ms | 1.081× |
| PrunaVAED 1024→1024 中间卷积，优化输入读取 | 16.031 ms | 13.981 ms | 1.147× |
| PrunaVAED 64→48 输出卷积，优化输入读取 | 31.928 ms | 19.319 ms | 1.653× |

DiT 这一批性能测量采用串行音视频分支；fast AV 路径另有正确性检查，不能
把表中的比例当成默认 fast AV 全流程收益。VAE 的 128/256/384/512 通道残差
卷积分别约为 1.641×、1.625×、1.594×、1.588×；1024→4096 扩张投影没有
启用新输入读取策略，其计时差不计为优化收益。上述候选与对应原路径的输出
逐字节一致；这一结论不代表与上游不同后端、不同累加顺序的实现逐字节相等。

完整采样、对照边界、源码与二进制哈希见
[便携测试证据](ltx25-acceleration-2026-11-11.json)。原始逐次日志和媒体留在
忽略的 `outputs/ltx25-m5-optimization-20261110/`，没有提交模型或生成视频。

### 当前完整音视频请求

固定提示词为雪地狐狸及环境音，704×448、97 帧、24 FPS、11 步、seed 42，
`c_metal`、GPU、`streamed`、12 GiB denoiser 预算、`ltx_fast_av=true`、
PrunaVAED、动态文本编码。12 GiB 是 denoiser 工作集目标，不是进程内存上限；
本轮没有新增完整进程峰值内存测量。

总共六次串行请求，每次启动新 CLI 进程，使用最终签名包中的 CLI/library。
先运行一次优化版本，使用新的 conditioning 目录记录提示词缓存未命中；
再完整预热基线一次，然后按 ABBA 测量，每组两个正式样本。基线只设置
`TURBOCIDER_PRUNAVAED_OPTIMIZED_INPUT_LOADS=0`，候选设置为 `1`；两者都
保留其他已验证优化。计时期间没有并发构建、单测或 Agent 推理任务，笔记本
仍处于正常交互使用状态。没有强制清空 OS 文件缓存或 shader 缓存。

| 阶段，秒 | 首次提示词缓存未命中 | 当前暖缓存中位数 |
| --- | ---: | ---: |
| encoder 权重加载 | 0.587 | 0 |
| encoder 编码 | 3.526 | 0 |
| connector | 5.203 | 0 |
| DiT 权重加载 | 2.278 | 1.398 |
| DiT 第一阶段，8 步 | 16.334 | 14.006 |
| latent upsample | 0.719 | 0.573 |
| DiT 第二阶段，3 步 | 14.628 | 15.303 |
| 视频 VAE 权重加载 | 0.128 | 0.095 |
| 视频 VAE 计算 | 2.620 | 1.917 |
| 音频解码，含 VAE/vocoder/BWE | 0.962 | 0.968 |
| 视频编码与导出 | 0.321 | 0.238 |
| 音频 mux | 0.094 | 0.042 |
| exec 交接及相关开销 | 3.736 | 0.196 |
| 原生完整请求 | **51.458** | **34.969** |
| 完整 CLI 进程 | **51.856** | **35.002** |

各行包括嵌套计时，不能将全部行相加。暖请求的音频 VAE、vocoder、BWE
分别约为 0.043、0.441、0.484 秒。实际逐步计时的均值为第一阶段
**1.748 秒/步**、第二阶段 **5.071 秒/步**，包含流式读取、调度和 GPU 同步，
不是纯 kernel 时间；原始每一步数组均保留在便携 JSON 中。

当前主要瓶颈是 DiT：两阶段共 **29.309 秒，占原生请求约 83.8%**。
视频 VAE 计算约占 5.5%。其余五次请求都报告 connected conditioning
cache hit，encoder 与 connector 的执行耗时均为 0；同一兼容提示词可以
直接进入后续 DiT 流程。首次与暖请求还包含模型加载、首次执行和 exec
交接差异，不把 51.9→35.0 秒全部归因于提示词缓存。

本轮单独比较最新 VAE 输入读取优化：VAE 计算从 **2.772→1.917 秒**，
约 **1.446×**，耗时减少约 30.9%。原生请求中位数为 36.286→34.969 秒，
完整进程为 36.329→35.002 秒，整机观测减少约 3.6%。但基线两个正式样本
相差约 2.36 秒，大于组间约 1.32 秒差值，尚不能据此宣称稳定的整机加速。
DiT 在这个开关对照中没有代码差异，其计时差也不算作 VAE 优化收益。
本轮没有重跑“所有历史优化关闭”的完整模型基线。

### 正确性与桌面产物

六份 MP4 都完成整段解码检查：97 帧、704×448、24 FPS，48 kHz 双声道
音频，194000 个音频帧，音视频时长均为 4.041667 秒，视频 PTS 严格递增，
音频非静音，原生结果报告 clipped samples 为 0。六次输出解码后的 BGRA
视频和 PCM16 音频各自逐字节一致，覆盖首次缓存未命中、缓存命中和 VAE
开关；这只是固定请求的输出一致性检查，不是跨提示词感知质量评估。

原生及最终打包 library 各运行 152 项契约测试，149 通过、3 跳过；5 个
实际打包 DiT block 场景与 V22 输出一致，覆盖两阶段、图生视频 prefix、
fast AV 与严格内存 hooks。PrunaVAED 通过 30 类通道/tail/strided/offset
场景及 450 个独立 FP64 边界见证。另运行 W8A8 融合/量化/workspace、
connector 缓存 CPU 测试、`make test-ltx-optimization`、完整 App 编译和
`make test-app`。

`make test` 的前半部分已通过；中途发现本地增量构建遗漏标准符号导出选项，
修正后按原 Makefile 的 43 条剩余命令完整补跑并通过。该记录没有冒充为一次
从头至尾成功退出的 `make test`，可选硬件/测试 hooks 的跳过项仍保留在日志中。
构建目录选择、桥接文件平台归属和防止运行中覆盖打包的修复均包含在提交中。

最终更新 `dist/TurboCider.app`、`dist/prunavaed/TurboCider.app` 和
`dist/qwen-time-condition-cache-auto/TurboCider.app`，保留旧包备份。正常退出旧
客户端后重新启动标准入口，并检查新进程映射的 App 与三个 dylib 的 inode
和 SHA-256。菜单“关于 TurboCider”可查看构建时间与位置。归档为
`dist/TurboCider-macOS-arm64-20261111-v23.zip`。本轮没有通过 GUI 再运行一次
生成任务；完整推理验证使用同包 CLI，桌面请求转换和任务保存使用 App 测试。

复现时，从便携 JSON 的 `request` 字段创建请求文件，并将
`<PRUNAVAED_CHECKPOINT>` 替换为本机 checkpoint：

```sh
.venv/bin/python3 tools/native/benchmark_ltx_optimization.py \
  --cli dist/cli/turbocider --model /path/to/LTX-2.5-INT8 \
  --request /path/to/request.json \
  --baseline-env TURBOCIDER_PRUNAVAED_OPTIMIZED_INPUT_LOADS=0 \
  --candidate-env TURBOCIDER_PRUNAVAED_OPTIMIZED_INPUT_LOADS=1 \
  --candidate-first-warmup --skip-latent-dump --rounds 1 \
  --output outputs/new-ltx-e2e-check
```
