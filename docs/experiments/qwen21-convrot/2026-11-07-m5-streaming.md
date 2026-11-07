# M5 Pro 上 Qwen Image 2.1 Turbo ConvRot 流式加载与 Runtime ANE

Date: 2026-11-07
Status: measured

## 结论与产品默认

App 保持 **GPU TensorOps W8A8 + streaming**。M5 及更新芯片在满足 Metal TensorOps
运行条件时，编码器与 DiT 的 ConvRot 线性层使用原始有符号 INT8 权重与动态 INT8 激活。
较早芯片保留浮点激活路径。当前发布包不包含 Private ANE 后端。

本次优化把流式编码器的整张词表改成通过已认证的文件描述符按 token 读取。五次完整
512×512 Turbo8 生成的文本、初始噪声、最终 latent 与图片像素均与改动前完全一致。
编码器固定源权重从 622,945,864 字节降到 8,192 字节；短提示词的受管理容量从
1,009,582,080 字节降到 386,793,472 字节。进程峰值约 4.2 GiB，整段请求的物理峰值
没有随编码器常驻权重等量下降。

Runtime ANE 可以在本机的显式诊断构建中与双 INT8、W8A8 和 streaming 共同执行。
修复了构造阶段使用常驻模型内存准入、尚未进入流式准入就可能拒绝图的问题，同时
计入 capability 自测缓冲并保留预算、系统余量、固定几何和后端限制。

两种配对顺序没有给出稳定的端到端加速。因此不启用 App 的 ANE 默认路线。

## 复现环境与版本

| 项目 | 值 |
| --- | --- |
| 主机 | Apple M5 Pro，24 GiB，macOS 26.5.1，arm64 |
| 依赖 | 项目 `.venv`，MLX 0.32.0；计时统一设置 `MLX_ENABLE_TF32=0` |
| Git | main 更新到 `db1f6d9`，合并提交 `22b6336`，无冲突 |
| 实现 | 稀疏词表 `32a1733`；流式 ANE 构造准入与配对工具 `180af38` |
| 模型 | ModelScope 下载的原始 Turbo INT8 ConvRot DiT、Qwen3-VL-8B INT8 ConvRot、原始 BF16 VAE |
| 推理 | 512×512、官方固定 8 步/CFG1、seed 73、无 LoRA、无 DiT 缓存 |
| 调度 | 0 个常驻 Transformer 层，2 个原始权重缓冲，1 个读取线程，预取距离 1 |
| 阶段 | 编码器、DiT、原始 VAE 依次加载与释放；每层完成读者后才复用缓冲 |
| 采样 | ABBA 与 BAAB；每个进程 1 次预热、短/长提示词各 2 次计时，共 40 次生成 |
| 内存 | 独立进程树采样，包含加载、认证、预热、生成与退出；间隔 100 ms，最大允许间隔 500 ms |
| 控制条件 | 暂停本任务的编译和其他测试以计时；没有锁定芯片频率或控制温度，观察到明显顺序漂移 |

完整模型 SHA、两种构建的 runtime ID、容量、逐批计时、质量与内存摘要见
[小型证据文件](2026-11-07-m5-streaming-evidence.json)。

## 默认常驻量与优化边界

| 范围 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 常驻 Transformer 层 | 0 | 0 |
| 原始流式缓冲数 | 2 | 2 |
| 编码器固定源权重 | 594.09 MiB | 8 KiB + 按提示词临时读取的 token 行 |
| 编码器权重/缓冲容量，短提示词 | 0.940 GiB | 0.360 GiB |
| DiT 固定源权重 | 约 259 MiB | 约 259 MiB |
| DiT 权重/缓冲容量 | 0.660 GiB | 0.660 GiB |
| 完整进程物理峰值，对照与候选 | 4.195 GiB | 4.229 GiB |
| 新增系统 swap-out | 0 | 0 |

4 GiB 是受管理权重的预算上限。实际容量按源布局与提示词长度计算。激活、临时计算、
图像解码、App 界面及外部服务另有占用。上表两个阶段的容量不相加为常驻峰值。
MLX 分配器峰值、进程物理内存、文件缓存也具有不同的统计范围。

稀疏读取保留原始 I8 码与 F32 行尺度，先逆 Comfy H256，再执行原来的 BF16 舍入。
重复或乱序 token 行不改变结果；读取前后校验原始文件代际，取消及越界 token 被拒绝。
浮点激活路径另做了一次 256×256 完整生成的前后比较，文本、噪声、latent 与像素
同样逐位一致。该测试在本机强制选择浮点路径，较早芯片的实际硬件仍未在本机运行。

## ANE 的完整生成比较

表格为每种提示词各 4 次测量的请求耗时中位数；每个批次的首个请求用于预热与中间张量
核验，预热写盘开销不计入计时样本。A 为 GPU，B 为 GPU + Private ANE。

| 顺序 / 提示词 | GPU W8A8 | GPU + ANE | GPU / 混合 |
| --- | ---: | ---: | ---: |
| ABBA / 短 | 8.965 s | 8.414 s | 1.065× |
| ABBA / 长 | 8.919 s | 8.981 s | 0.993× |
| BAAB / 短 | 8.154 s | 9.187 s | 0.888× |
| BAAB / 长 | 8.216 s | 10.174 s | 0.808× |

ABBA 的首尾 GPU 批次分别约 7.3–7.7 s 和 10.3–10.5 s。BAAB 的末尾 ANE 批次也有
明显变慢。因此表中的小幅提升不能作为稳定收益或跨模型结论。

全部候选均检查真实运行回执：每次 224 个成功 ANE FFN 分区、0 个 fallback、1 个固定
program，并验证 IOSurface 输入输出、原始 I8 源、F32 分区合并、1 个 GPU prefill 步与
7 个 decode 步。两轮共有 4,480 个成功 ANE 分区。编码器仍使用 GPU W8A8；ANE
诊断只分担 DiT decode 的 4,096 / 12,288 个 FFN 中间通道。

质量阈值固定为相关性 ≥ 0.99、cosine ≥ 0.995、MAE ≤ 5/255。所有图像比较通过，ANE
候选的最低相关性约 0.9990，最大 MAE 约 1.102/255。文本与初始噪声逐位一致；ANE
输出保留其量化及浮点边界带来的数值差异。此处基准是同一双 INT8 ConvRot 的 GPU
W8A8 路线，A8 相对浮点激活的质量影响属于独立比较。

所有批次均通过进程树采样完整性检查且新增 swap-out 为 0。进程物理峰值约
4.0–4.2 GiB；独立 ANE 服务及驱动分配的归因不完整，不能据此推导整机 ANE 内存。

## FFN 通道分配与预览组合

在同一组真实 DiT 第 0 层权重上，对 256 / 1,024 行与 3,072 / 4,096 / 6,144 个 ANE
通道各做 11 次组件测量，包含每次权重 staging：

| 行数 | 3,072 通道 | 4,096 通道 | 6,144 通道 |
| --- | ---: | ---: | ---: |
| 256 | 0.617× | 0.535× | 0.494× |
| 1,024 | 0.990× | 1.003× | 0.824× |

这些是单 FFN 的 GPU/混合耗时比。1,024 行的 4,096 通道分配接近持平，扩大通道数
没有带来组件收益。代码中的权重搬运、激活转换和合并会增加开销；本次未对所有硬件
带宽与温度影响分别归因。

另一次完整 Private ANE + streaming + W8A8 + Tiny VAE 预览生成，确实收到第
2、4、6、8 步预览；与不启用预览的同一 ANE 请求相比，文本、初始噪声、最终 latent
和原始 VAE 最终图都逐位一致。默认发布 App 的 GPU 路线还有独立桌面与生命周期验收。

GPU 的自定义 primitive 使用 I8×I8→I32 和 F32 epilogue。ANE 图使用 I8 权重与激活
接口，尺度、非线性及输出合并包含浮点运算。实际 ANE 调用与图输入类型已核验；没有
使用硬件指令计数器独立证明 ANE 编译器对每个 MatMul 的物理整数实现。

## 复现命令与原始证据

在仓库根目录执行。`$MODEL` 指向已下载的原始双 INT8 ConvRot 安装目录。

```bash
export PATH="$PWD/.venv/bin:$PATH"
TURBOCIDER_BUILD_PACKAGE_ONLY=1 TURBOCIDER_NATIVE_ONLY=1 \
  TURBOCIDER_BUILD_OUTPUT_DIR="$PWD/build/m5-main-optimized" tools/native/build.sh
TURBOCIDER_BUILD_PACKAGE_ONLY=1 TURBOCIDER_NATIVE_ONLY=1 \
  TURBOCIDER_ENABLE_PRIVATE_ANE=1 \
  TURBOCIDER_BUILD_OUTPUT_DIR="$PWD/build/m5-main-private" tools/native/build.sh

MLX_ENABLE_TF32=0 .venv/bin/python3 tools/benchmarks/qwen21_convrot_stream_ane_benchmark.py \
  --cli build/m5-main-private/turbocider --model "$MODEL" \
  --output build/convrot-ane-abba-new --repeats 2 --order ABBA
MLX_ENABLE_TF32=0 .venv/bin/python3 tools/benchmarks/qwen21_convrot_stream_ane_benchmark.py \
  --cli build/m5-main-private/turbocider --model "$MODEL" \
  --output build/convrot-ane-baab-new --repeats 2 --order BAAB
```

原始证据位于以下 **本机忽略目录，未提交到仓库**：

- `build/setup-logs/m5-main-sparse-comparison.json`
- `build/setup-logs/m5-main-sparse-float-a-comparison.json`
- `build/setup-logs/m5-main-stream-ane-abba/` 与 `m5-main-stream-ane-baab/`
- `build/setup-logs/m5-main-ane-ffn/sweep-report.json`
- `build/setup-logs/m5-main-ane-tiny-preview/report.json`

## 验证范围

Native 专项测试覆盖真实 Metal 的整数 CPU oracle、编译/非编译路径、源闭包、内存
准入、缓冲退休、取消、稀疏行读取及浮点/W8A8 路径的流式/常驻数值一致性。App 的
运行详情补充 ConvRot 专用的读取次数与分阶段容量，避免把 4 GiB 预算当作实际常驻量。

本轮 Qwen 专项运行 142 项，其中 6 项可选项跳过，其余通过；仓库边界 52 项通过。选定当前构建目录的
`make test`、App / model-library / API 官方目标和额外 5 个 host 目标均通过。原生测试
统一使用 `MLX_ENABLE_TF32=0`；VDN 的 FP32 oracle 子进程也显式固定此设置，避免用
MLX 的默认约简精度 GEMM 判断 FP32 求解误差。库存与 VDN 测试修正为尊重所选构建目录。

打包产物的库、辅助进程与 Tiny VAE 资产还完成独立生命周期验收：原始 VAE 生成、
预览开关的最终图逐位一致、2/4/6/8 步预览、在第 2 步预览后取消并且不发布最终图、
随后 Tiny 最终解码重试成功，以及重启状态对象后的历史持久化。完整进程树物理峰值
100 ms 采样峰值 3.875 GiB，新增 swap-out 为 0，模型库辅助进程纳入采样角色，回执完整性通过。
根进程生命周期峰值约 4.250 GiB；两种测量分别保留在小型证据文件中。

相应本机原始日志为 `build/setup-logs/m5-main-optimized-contract-tests-serial.log`、
`m5-main-optimized-app-tests.log` 与 `m5-main-packaged-app-acceptance-verified/`。
桌面 Computer Use 当前遇到系统锁屏，等待用户手动解锁；此次桌面点击验收尚未完成。

当前结论适用于这台主机、512×512、Turbo 固定 8 步、已记录的提示词与 seed。较大
画布、其他芯片、其他模型、LoRA 和编码器 ANE 没有由这组结果完成性能资格验证。

Apple 的 [Metal tensor 说明](https://developer.apple.com/videos/play/wwdc2026/330/)
将 M5 GPU 内的 Neural Accelerators 与 GPU tensor 运算联系起来；这条路线与独立 ANE
分区的调度分别核验。MLX 的浮点精度策略可参考其
[默认 TF32 与精度讨论](https://github.com/ml-explore/mlx/issues/3860)。
