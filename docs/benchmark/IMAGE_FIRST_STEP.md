# 图像首步耗时：测量工具与本机观察

开发工具 `tools/validation/image_first_step.py` 通过公开 C ABI 测量 Qwen Image 2.1
文生图／编辑，以及 Z-Image Turbo 文生图。Z-Image 当前原生执行器没有图片编辑入口，
工具拒绝为它构造编辑耗时。App 不依赖此 Python 工具。

## 口径

- 构造 engine 单独计时；请求总耗时包含编码、采样、解码、导出。
- `time_to_denoise_seconds`：进入第零个 denoise 边界之前的时间。
- `first_step_seconds`：denoise 0→1；后续步中位数来自 1→2 及以后。
- 同一 engine 默认四次请求：首次、相同提示词、改提示词、重复新提示词。
- `--prepare none|load|warmup` 独立记录准备成本。预热成本不计入后续请求，但保留在报告中。
- 首次指新 engine 的首次请求，不代表清空 OS 页缓存或 Metal 驱动缓存。
  报告保存构建身份、系统、完整请求、进度边界、原生指标和 PNG SHA256。
- 事件边界内同时可能包含惰性求值、编译、真实前缀计算等，不能直接称为编译耗时。

```sh
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id z-image-turbo --residency resident --output outputs/startup-z
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id qwen-image-2.1 --residency component_staged --steps 20 --output outputs/startup-qwen
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id qwen-image-2.1 --residency component_staged --steps 20 --reference INPUT.png --output outputs/startup-edit
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id qwen-image-2.1 --residency component_staged --steps 2 --lora ADAPTER.safetensors --sampler dmd2 --output outputs/startup-dmd2
```

为避免系统负载干扰，请串行运行推理，并在独立进程对照中重复采样。
这是启动诊断工具，不是外部框架配对性能评测。

## 2026-10-27 本机观察

Apple M4 Max、64 GiB、macOS 26.6.2、MLX 0.32.0；512×512、seed 42、纯 GPU、
BF16 基础组件、原始 VAE。Z-Image 常驻、8 步；Qwen 分组件驻留、基础 20 步
或 DMD2 2 步；编辑使用一张标准 1024 编码的参考图。
未使用 PE、ANE、DiT 近似缓存或 Tiny VAE。
以下每行是一次观察，不是置信区间或中位性能保证；单位为秒。

| 路线／请求 | 采样前 | 首步 | 后续步中位数 | 完整请求 |
|---|---:|---:|---:|---:|
| Z-Image 首次 | 3.151 | 0.873 | 0.844 | 10.309 |
| Z-Image 重复 | 0.001 | 0.878 | 0.844 | 7.053 |
| Z-Image 改提示词 | 4.326 | 3.485 | 0.844 | 14.023 |
| Z-Image 重复新提示词 | 0.001 | 0.850 | 0.843 | 6.996 |
| Qwen 文生图首次 | 4.808 | 1.138 | 1.048 | 26.525 |
| Qwen 文生图重复 | 0.367 | 1.129 | 1.048 | 22.007 |
| Qwen 编辑首次 | 6.246 | 5.953 | 1.228 | 36.419 |
| Qwen 编辑重复 | 0.674 | 1.241 | 1.227 | 25.999 |
| Qwen 编辑改提示词 | 4.709 | 5.953 | 1.228 | 34.856 |
| Qwen 编辑重复新提示词 | 0.741 | 1.245 | 1.227 | 25.936 |
| DMD2 文生图首次 | 4.839 | 1.163 | 1.130 | 7.816 |
| DMD2 文生图重复 | 0.437 | 1.155 | 1.081 | 3.288 |
| DMD2 编辑首次 | 6.343 | 6.096 | 1.270 | 14.509 |
| DMD2 编辑重复 | 0.980 | 1.274 | 1.254 | 4.125 |

DMD2 两路各做了首次与重复两次请求，均报告实际 2 步、128 个 LoRA 投影。
查看结果图确认文生图生成红色茶壶、编辑改为蓝色；这只是有限冒烟验证，
不是与 20 步基础模型等画质的证明，不能把减少步数包装成无损算子加速。

Qwen 文生图首步只略慢；编辑首次和改提示词时明显更慢，重复条件后恢复至常规步时。
Z-Image 改提示词出现首步尖峰，重复新提示词后消失。下面补充独立观测与缓存关闭
对照，区分实际计算、同步编译相关 API 和尚不能精确归因的 GPU 提交／等待。

本机原始报告位于忽略的 `outputs/image-startup-20261027/`：
`baseline-z`、`baseline-qwen-t2i`、`baseline-qwen-edit`、`dmd2-t2i`、`dmd2-edit`。
模型、媒体和包含本机路径的原始报告不提交 Git。

## 独立 Metal API 观测

`tools/profiling/image_startup/metal_trace.mm` 是仅用于开发的观察器，**不链接或打包进
App／运行库**。它拦截同步 library 编译／加载、function 创建／特化和 compute pipeline
创建，按单调时钟与实际 denoise 回调窗口求交。嵌套或重叠调用取时间区间并集，不相加。
它测的是 **API 主机耗时**，包含缓存查询和驱动工作，不是纯编译器时间；不覆盖
异步 API、驱动内部延迟编译、GPU 执行或 MLX 图追踪。

这与所用 MLX v0.32.0 的同步 library／function／pipeline 建立路径对应；
其 `get_library` 和 `get_kernel` 已有进程内缓存，见
[固定版本的 Metal 后端源码](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp)。
不能因为调用了 `mlx::clear_cache`，就假设编译好的 Metal kernel 也全部失效。

观测不重写编译选项或返回对象，使用 MRC 保持 `new` 返回值的原始所有权；
实际小型 GPU 测试验证数值、错误透传与对象释放。记录只保存源代码长度／非密码学
摘要，不保存源代码或提示词。文件必须是新路径；PID、时钟、安装失败、空捕获、
日志截断均会拒绝作为有效观测。macOS framework Python 可能二次 exec，需用独立
启动器选择真实解释器后再注入；直接注入 venv launcher 得到的空捕获不可信。

```sh
bash tools/profiling/image_startup/build_metal_trace.sh
python tools/profiling/image_startup/run_metal_trace.py --trace outputs/trace-z/metal.jsonl -- --model MODEL_ROOT --model-id z-image-turbo --output outputs/trace-z
python tools/profiling/image_startup/run_metal_trace.py --trace outputs/trace-qwen/metal.jsonl -- --model MODEL_ROOT --model-id qwen-image-2.1 --steps 20 --residency component_staged --reference INPUT.png --output outputs/trace-qwen
TURBOCIDER_TEST_METAL_TRACE=1 python -B tests/native/test_metal_compile_trace.py
```

同机独立进程、相同尺寸和提示词，**带观测器**的结果如下。不能将观测器开销直接从
另一进程的未插桩时间中相减；驱动缓存未清空，首次请求不等于首次驱动编译。

| 路线／请求 | 首步秒数 | 首步内同步 Metal build API 并集秒数 |
|---|---:|---:|
| Z-Image 首次 | 1.674 | 0.950 |
| Z-Image 重复 | 0.853 | 0 |
| Z-Image 改提示词 | 1.835 | 0 |
| Qwen 文生图首次 | 1.132 | 0.00654 |
| Qwen 文生图重复 | 1.128 | 0 |
| Qwen 编辑首次 | 6.198 | 0.38281 |
| Qwen 编辑重复 | 1.241 | 0 |
| Qwen 编辑改提示词 | 5.954 | 0.000497 |

结论：Qwen 编辑新增条件的约 6 秒首步不是主要耗在这些同步编译调用中；其前缀
计算在缓存 miss 时必须重新完成。Z-Image 首次确有可观测的建库成本，但换提示词
之后的尖峰在此样本中没有新的同步 Metal build API 调用，不能全部归为 compile overhead。

独立的未注入 Z-Image 进程用 `--sample-run 2` 在换提示词后的首步采样 1 秒。
该次首步 2.944 秒，采样窗口主线程 278 个样本中 277 个在 MLX eval 的等待链；
另见 Metal command queue 的 IOGPU 提交等待。它支持“主要卡在执行／提交等待”而非
主线程图追踪的判断，但不能据此区分 GPU 算术、页驻留或驱动调度各占多少。
采样也不是无开销基准，不能把窗口样本比例外推为整步的精确耗时占比。

## 缓存究竟省了什么

Qwen 的默认路径已保留一个有界编辑前缀 K/V 快照；512 方图、1–3 张标准参考图、
兼容 GPU 配方满足条件时自动使用，预算不超过 8 GiB 与物理内存的 1/8。
App 的常规非流式路径复用 engine，因此可以利用该缓存；进程退出／卸载后不保留。
提示词、参考图、adapter、几何或采样身份变化时不能当作同一条件命中。

对同一参考图与提示词，用 `TURBOCIDER_QWEN21_PREFIX_SNAPSHOT=0` 关闭快照后：

| 重复编辑请求 | 快照 | 首步秒数 | 完整请求秒数 |
|---|---|---:|---:|
| Qwen 基础 20 步 | 默认开启／命中 | 1.241 | 25.999 |
| Qwen 基础 20 步 | 关闭 | 5.951 | 30.347 |
| DMD2 2 步 | 默认开启／命中 | 1.274 | 4.125 |
| DMD2 2 步 | 关闭 | 6.090 | 8.589 |

这些是同机本地对照观察，不是随机化 ABBA 或跨框架速度资格；完整请求还包含其他
阶段噪声。约 4–5 秒的重复编辑收益来自已有前缀复用，不能称为新加入的持久化编译缓存。
本例基础模型开关快照的最终 PNG 逐字节一致；DMD2 则不逐像素一致，RGBA 8-bit
通道平均绝对差 0.251、RMSE 0.626、最大差 18。缓存命中的目标行 GEMM 与完整
前缀 GEMM 可有不同 BF16 舍入，不承诺所有输入下图像一致。DMD2 已要求明确近似许可。
若需固定旧路径的像素行为，可关闭快照；不能用一个样本代替整体画质资格。

显式预热也不是免费：Z-Image `--prepare warmup --runs 2` 的准备成本 10.240 秒，
随后两次完整请求 7.024／7.002 秒，首步 0.851／0.852 秒。准备加首个请求约
17.264 秒（engine 构造另计），比未预热首次约 10.3 秒更长；只有将可见成本放在
用户主动的准备阶段才有意义，不建议为漂亮的首步数字强制增加一次完整采样。

因此保留已有进程内编译缓存、条件缓存与有界前缀缓存；本轮没有新增不透明的磁盘
Metal 缓存或自动隐藏预热。首个全新编辑条件的实际前缀工作不能仅靠复用旧 prompt
消除。Tiny 编解码与 DMD2 是明确改变质量／步数档位的加速选项，不是无损编译优化。

## DMD2 与 Tiny VAE 的 App 组合验收

用 App 下载助手取得的官方 TAEQI2.1 与固定 DMD2 文件，真实 `NativeJobStore` 链通过：
基线生成、两步末步预览、取消、取消后恢复、Tiny 最终解码、Tiny 参考图编码编辑。
文生图开／关预览的最终 PNG 逐字节相同；Tiny 编码编辑观察到 `tiny_vae_encode` 事件。
本机 Tiny 最终解码阶段 0.031 秒，对比同一链普通解码 0.550 秒；暖态完整文生图
约 2.776 秒（含预览），这不是相对原始 VAE 的画质等价证明。

```sh
TURBOCIDER_APP_TARGETS=tiny-vae tools/native/build_app.sh
build/native/turbocider-tiny-vae-app-tests --gpu-dmd2 qwen-image-2.1 MODEL_ROOT TINY.safetensors OUTPUT_DIR DMD2.safetensors
```

新增原始证据目录：`traced-z-v2`、`traced-qwen-t2i`、`traced-qwen-edit`、
`qwen-edit-snapshot-off`、`dmd2-edit-snapshot-off`、`z-host-sample`、
`z-explicit-warmup`、`dmd2-tiny-app`。`traced-z` 是 Python launcher exec 丢失注入的
无效尝试，已排除，不得用于“没有编译”的判断。
本轮发布包在 `outputs/image-startup-20261027/release/TurboCider.app`，包含官方 Tiny
权重与许可声明、不包含 Python 或本观察器；本机 ad-hoc 签名已通过校验。
