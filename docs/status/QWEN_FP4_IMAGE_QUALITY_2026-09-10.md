# FP4 mixed 文本最终出图与 GPU/ANE 加速复查

日期：2026-09-10。M4 Max Mac Studio，64 GiB。MLX 0.32.2。

## 最终出图画质

这里的 FP4 指 **Comfy Qwen3-4B FP4 mixed 文本编码器**，不是 NVFP4 图像
DiT。图像 DiT 与 VAE 均保持 BF16，512×512、9 步、GPU、无 LoRA/ANE。
6 类提示词 × seeds 42/123，共 12 对、24 张比较图片，另有一张原生控制图。

图片与原始指标：`outputs/qwen-mixed-e2e-20260910/`，其中
`comparison.png` 为总览，`quality.json` 为每对图的差异。
总览每行从左到右：BF16 seed 42、FP4 seed 42、BF16 seed 123、FP4 seed 123。

### 隔离方法与控制

- 使用原生 `Tokenizer::z_image_prompt` 的真实模板和 token IDs，动态长度。
- 两种文本权重分别运行完整 Qwen conditioning；FP4 路径沿用已验证的
  NVFP4/MXFP8 无损重打包，取第 35 block、最终 RMSNorm 前的隐藏状态。
- 将真实 conditioning 与 token IDs 保存后，交给同一原生 DiT/VAE 流程
  生成最终 PNG。回放入口检查 token IDs、shape、BF16 dtype 与有限性。
- 测试入口直接编译原 `z_image.cpp`，只替换 conditioning 调用，未改变
  去噪、噪声、调度器、VAE 或图片导出算法。它不链接到 App。
- BF16 conditioning 回放生成的狐狸图，与该入口的原生文本编码生成图
  **逐像素一致**，也与先前原生 C ABI BF16 狐狸基线逐像素一致。
- 这是最终图片的完整链路质量验证，不是只比较文本张量；但编码与图像
  阶段通过文件回放分离，图片阶段计时**不代表集成 App 的端到端延迟**。
  没有用“跳过文本加载”的时间宣称 FP4 e2e 加速。

### 看图结论

所有比较图都保持可辨识的目标场景，未见全图崩坏、大片噪点或明显普遍失真。
这是本次人工查看结果，不是盲评或广泛模型质量保证。

| 场景 | seed 42 / 123 像素 cosine | 人工查看 |
|---|---|---|
| 雪地狐狸 | 0.995681 / 0.999399 | 最接近的场景之一；seed 42 脸部和腿部位置变化，seed 123 很接近 |
| 人像、双手持杯 | 0.988258 / 0.991737 | 保持人物和杯子主题；眼神、五官、手部位置及饰品细节有变化，不能称为无损 |
| 苹果、蓝杯、柠檬 | 0.998623 / 0.998684 | 两种精度均保持三件物体和左中右顺序；轮廓、间距及阴影略变 |
| 山湖木屋 | 0.978829 / 0.992082 | 木屋位置/结构、树木和倒影变化可见；两种图都保持自然场景 |
| 英文咖啡店招牌 | 0.918813 / 0.973655 | 两种精度、两个 seed 都写出 TURBO CIDER；店面框架和布局变化较大，不是文字乱码 |
| 中文茶馆招牌 | 0.967301 / 0.977350 | 两种精度、两个 seed 都写出“春日茶馆”；招牌、门框、植物形状和位置改变 |

12 对像素 cosine 均值 0.981701、中位数 0.989997。
这不是“98% 画质”，也不能从英文店面的较低 cosine 推导文本生成失败。
同 seed 下的几何位移会显著拉低像素相似度，可能仍然是可接受的画面。

FP4 mixed 可以作为这台机器上的省文本内存候选，尤其是允许细节变化的普通
摄影场景。若要求尽可能复现 BF16 的脸部、构图或细节，仍选择 BF16。
此轮没有同提示词的 Q8/Q4 图片，因此不把旧的单狐狸结果扩展为新的总体排名。
也未测 1024² 的 FP4 文本图片、多 seed 统计、LoRA 或视频。

此前同口径文本测试：FP4 mixed 128-token MLX 峰值 3.595 GiB、编码
99.65 ms，BF16 为 7.914 GiB、156.88 ms。这里只引用独立文本阶段测量；
本轮图像阶段峰值约 14.93 GiB，没有由此推导完整 App 的新内存保证。

## 为什么上轮 GPU/ANE 看起来没有 1.2×？

必须拆分去噪和 VAE，不能只看整图时间。以下均为暖请求中位数。

| 记录 / 路线 | 去噪 s | VAE s | 原生 request wall s |
|---|---:|---:|---:|
| 9 月 8 日 INT8 GPU | 39.811 | 0.850 | 40.729 |
| 9 月 8 日 INT8 GPU/ANE | 31.678 | 0.853 | 32.598 |
| 上轮 9 月 10 日 INT8 GPU | 39.858 | 0.958 | 40.887 |
| 上轮 9 月 10 日 INT8 GPU/ANE | 33.413 | 6.277 | 39.791 |

上轮去噪实际仍有 **39.858 / 33.413 ≈ 1.193×**；但 VAE 增加约 5.32 s，
抵消了大部分去噪收益。因此不能说“ANE 本身不再加速”。
本项目另一份 9 月 7 日历史性能记录为 GPU 36.369 s、hybrid 30.001 s，
约 1.212×，也不是任意精度/机器状态下的保证。

本轮使用同一个 INT8 checkpoint 和 SHA 匹配的 4096/10240 FFN 分区，
1024²、9 steps、seed 42。VAE 路径仍在 MLX GPU；ANE 只分担 FFN，
未因开启 GPU/ANE 而换成另一个 VAE 算法。

### 捕获的内存压力证据

新诊断入口不使用前轮 25 ms polling 线程，只在请求、去噪、VAE 边界采样。
不限制缓存时仍复现了 VAE 变慢，因而不能简单归咎于旧采样线程。

`ane-diagnostic/report.json` 中三次 VAE 为 4.941 / 4.116 / 3.374 s。
VAE 后 MLX 空闲缓存约 **30.9 GiB**。

VAE 阶段的主机 VM 计数增量（计数是页操作，不是唯一驻留字节）：

| 请求 | compressions | decompressions | swapins | swapouts |
|---|---:|---:|---:|---:|
| 冷请求 | 2,356,877 | 466,043 | 37 | 75,320 |
| 暖请求 1 | 529,888 | 1,871,965 | 5,320 | 0 |
| 暖请求 2 | 0 | 1,647,819 | 112 | 0 |

这些是系统级计数，不能说所有换页都由 TurboCider 独自造成。只读环境检查
同时看到另一个 Python 进程约 12 GiB RSS，系统已有约 3.1 GiB swap；未终止
或修改该进程。结合 VAE 期间的压缩/换页与后续缓存上限对照，说明内存压力
是本轮不能忽略的因素，不能只归因为数学算子速度变化。

代码层面，`Request::allocator_cache_bytes` 有默认值，FLUX 和 LLaDA
会调用 `mx::set_cache_limit`，但当前 **Z-Image 路径未应用该上限**。
空闲缓存不是模型必要权重；它与 Core ML 额外分区及其他进程共同消耗统一内存。

### 只限制空闲缓存后的对照：恢复 1.20×

随后只在隔离进程内调用 `mx.set_cache_limit(1 << 30)`；两条路线都使用
同样的 1 GiB 上限，不改权重、分区、精度、分辨率、步数或模型算法。
每配置仍为 1 冷 + 2 暖，先 hybrid 后 GPU，与上轮路线顺序相反。

| 路线（均限制空闲缓存为 1 GiB） | 暖去噪 s | 暖 VAE s | 暖 request wall s | 自进程 physical footprint 历史峰值 GiB |
|---|---:|---:|---:|---:|
| INT8 GPU | 39.990 | 1.690 | 41.784 | 18.152 |
| INT8 GPU/ANE 配置 | 33.004 | 1.708 | 34.815 | 20.075 |

整图比值 `41.783750354 / 34.8151748745 = 1.200159×`。
GPU 暖 wall 两次 41.780752 / 41.786749 s；hybrid 两次
34.820282 / 34.810068 s。原始记录为 `gpu-cache1g/report.json` 和
`ane-cache1g/report.json`。

限制缓存后三次 hybrid 的 VAE 均未观察到系统 compression/swapout 增量；
仍有少量系统级解压/换入，不能声称全机完全没有任何交换活动。
选定的 GPU、hybrid 暖输出分别与其未限制缓存的对应图片逐像素一致。
空闲缓存上限不改变模型数值，实测也未改变这些图片。

结论：上轮 VAE 延迟异常与巨大空闲缓存下的系统压缩/换页同时出现；
限制缓存后 VAE 稳定、整图恢复约 1.20×，支持**内存压力抵消去噪收益**
这一诊断，而不是“ANE 数学计算不再有效”。降低空闲缓存后的 VAE 约
1.7 s，仍不等于历史记录约 0.85 s，不能忽略重新分配开销和当前系统负载。
1 GiB 是本轮诊断参数，没有做不同缓存上限的全面调优。

上述物理占用是当前进程计账，包含其设备分配，不包含所有外部 Core ML
服务进程；不与 MLX/RSS 相加，也不是最低 RAM 保证。前一报告约 39 GiB
的读数是该轮真实高水位，但包含大量可回收空闲缓存，不能当作模型必要常驻量。

建议的后续工程改动是让 Z-Image 同样应用已存在的 allocator cache 预算，
并针对不同内存、分辨率和多任务场景验证。**本次没有修改生产缓存策略**。

## 复现工具与限制

- `tools/validation/qwen_mixed_image_suite.py`：真实提示词、编码、BF16
  控制、12 对最终图片及总览；目录重用必须显式 `--resume`。
- `tools/validation/z_image_conditioning_replay.cpp`：测试专用回放可执行文件，
  不进入 App。生成阶段遵守 TurboCider GPU 进程锁。
- `tools/validation/diagnose_z_image_ane.py`：阶段耗时、MLX 活跃/空闲缓存、
  自进程物理占用和系统 VM 计数；`--cache-mib` 仅改变测试进程的空闲缓存。
- 数据目录 `outputs/qwen-mixed-e2e-20260910/`。保留全部图片和逐次记录。

没有修改 App 生产加载逻辑、生产缓存策略或重新打包 App。

测试回放入口可使用与当前运行库一致的 macOS deployment target 编译，
链接本地 `libturbocider` 与 `libmlx`；例如本次目标为 26.2：

```sh
xcrun clang++ -std=c++20 -O2 -mmacosx-version-min=26.2 \
  -I native/core -isystem Python/lib/python3.13/site-packages/mlx/include \
  tools/validation/z_image_conditioning_replay.cpp \
  -L build/native -lturbocider -L Python/lib/python3.13/site-packages/mlx/lib -lmlx \
  -Wl,-rpath,"$PWD/build/native" \
  -Wl,-rpath,"$PWD/Python/lib/python3.13/site-packages/mlx/lib" \
  -o build/native/z-image-conditioning-replay
Python/bin/python3 tools/validation/qwen_mixed_image_suite.py \
  --output outputs/qwen-mixed-image-rerun
```
