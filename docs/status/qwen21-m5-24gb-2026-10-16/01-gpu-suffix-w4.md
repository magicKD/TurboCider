# Qwen-Image-2.1：GPU FFN 后缀 W4 缓存实测报告

发布日期：2026-10-16。单项实验：2026-10-02。当前整合验证：2026-10-15 至 2026-10-16。

在 Apple M5 Pro、24 GiB 统一内存、无 LoRA、text/DiT 逐层加载、组件分阶段释放和 ANE 开启的固定路径上，GPU FFN 后缀 W4 单项将狐狸样本整次耗时中位数从 **52.175150 秒降至 43.031015 秒，减少 17.53%**；同套件 W8 为 49.075479 秒，减少 5.94%。这是 2026-10-02 同库、每配置两次的实测。W4 会改变生成图的结构和材质，属于有损优化。

当前 PR 将 W4 后缀与 [attention W4](02-attention-w4.md)、[Q/K norm + RoPE 融合](03-qk-norm-rope.md)、[DBCache](04-dbcache.md) 兼容接入；四项组合的正式三轮中位数为 **53.828678 → 25.784862 秒，减少 52.098%，速度比 2.088×**。该数值属于组合，不能归因于后缀 W4，也不能由各单项百分比相乘推得。

## 优化范围与生效条件

模型有 32 个 Transformer block，隐藏宽度 4096，FFN 宽度 12288。原有混合图把 FFN `[0,6144)` 交给 ANE，GPU 计算 `[6144,12288)`。本优化仅压缩 GPU 后半段的 gate/up 和 down 权重，保留原 ANE W8A8 前缀、切分边界、BF16 首步 prefill 和前缀 KV 语义；它不是整模型四位量化。

当前自动策略严格匹配 `Apple M5 Pro` 和 `24 × 2^30` bytes 的设备配置，并要求 512×512、20–40 步、`image.generate`、`gpu_ane`、`component_staged`、`qwen21_w8a8=true`、`allow_approximation=true`、已验证 ANE manifest、空 LoRA、无输入图、无 prompt enhancer。只接受原标准 32 层、1024 rows、6144 ANE 前缀的 W8A8 图；其他精度、混合切分、GPU FFN fallback 或不兼容诊断不会进入此优化。硬件准入见 [设备策略](../../../native/runtime/device_optimizations.hpp)，请求准入与开关见 [memory_policy.hpp](../../../native/models/qwen21/memory_policy.hpp)，图约束见 [hybrid.cpp](../../../native/models/qwen21/hybrid.cpp)。本报告不外推到其他 M5 型号或内存容量。

合格请求默认启用 suffix W4、attention W4 和 Q/K 融合。DBCache 仍由请求显式选择 `qwen21_dit_cache="fast"`；它不会因前三项默认开启而自动启用。`TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS=0` 可关闭自动选择；`TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS` 接受 `0`、`4`、`8`，其中非零覆盖仍须通过相同准入。

## 计算精度、缓存所有权和有效原因

每层首次进入 hybrid decode 时，从 BF16 源层提取紧凑后缀切片，转成 FP16，以 affine、group size 64 打包。gate/up 和 down 各拥有 packed weight、scale、bias，共六个数组；packed weight 为 `uint32`，scale/bias 为 FP16。后缀输入、两次 `quantized_matmul` 及局部激活路径使用 FP16，结果在与 ANE 分区合并前转回 BF16。因此观察到的图像变化同时包含权重量化和局部算术精度变化，不能全部归因于四位权重。

六数组求值后独立拥有压缩缓冲区，后续步骤直接用 QMM，不恢复整块 BF16 权重。编译函数把输入与六数组作为显式参数，避免捕获源层或上一请求的缓存。缓存归当前请求所有，text/DiT 的逐层加载继续生效，去噪完成后在加载 VAE 前释放；取消路径等待已提交计算完成后清理。DBCache 前 8 步完整计算，足以建立全部 32 层缓存，之后跳过后部 block 时不破坏所有权或切分约定。实现见 [量化 helper](../../../native/models/qwen21/staged_suffix_quantization.hpp)、[hybrid 接入](../../../native/models/qwen21/hybrid.cpp)、[接口与指标](../../../native/models/qwen21/hybrid.hpp)、[请求生命周期](../../../native/models/qwen21/pipeline.cpp)。

设后缀元素数 `N = 32 × 3 × 6144 × 4096 = 2,415,919,104`，每 64 个权重的 FP16 scale/bias 合计 4 bytes。紧凑缓存大小如下；GiB 均按 `2^30 bytes` 换算。

| 后缀表示 | 缓存 bytes | GiB | 相对 BF16 减少 | 证据性质 |
| --- | ---: | ---: | ---: | --- |
| BF16 | 4,831,838,208 | 4.500000 | — | 由形状推导，旧基线未上报此字段 |
| W8 affine g64 | 2,566,914,048 | 2.390625 | 46.875% | 理论值与运行收据一致 |
| W4 affine g64 | 1,358,954,496 | 1.265625 | 71.875% | 理论值与运行收据一致 |

更小的权重表示减少后续 decode 的权重搬运需求，并允许在逐层释放源权重后重复利用请求内缓存；这是预期机制，速度是否改善仍须以实测为准。BF16 checkpoint 文件不变，首步仍需读取 BF16，首次打包也会暂时共存源权重、切片、FP16 中间值和压缩数组。缓存字节数不包括算子工作区、激活、attention、Core ML 或 VAE；不能用缓存减少比例代替进程内存或端到端收益。

## 历史单项 A/B：2026-10-02

套件 ID 为 `qwen21-quantization-20261002/hybrid-suffix-quantization`。六格共用同一 native 库，按 baseline → W8 → W4 → W4 → W8 → baseline 顺序运行；baseline 是同库原 BF16 后缀，未叠加 attention 量化、Q/K 融合或 DBCache。每格为新进程的首次完整请求，包含 PNG 导出，不包括 engine 构造；文件缓存未清理。固定提示为 `A red fox standing in fresh snow, soft morning light.`，seed 42、512×512、40 步。MLX 版本为 0.32.0；allocator cache 限额 512 MiB、去噪 wired 限额 6 GiB 保持一致。

| 实际运行 ID | 后缀 | 整次秒数 | 去噪秒数 | 首次打包累计秒数 |
| --- | --- | ---: | ---: | ---: |
| `00-baseline` | BF16 | 52.216932 | 42.098451 | 未上报 |
| `01-q8` | W8 | 48.995537 | 38.804713 | 1.330481 |
| `02-q4` | W4 | 42.787986 | 32.420196 | 1.337913 |
| `03-q4` | W4 | 43.274043 | 32.924072 | 1.344525 |
| `04-q8` | W8 | 49.155420 | 38.871766 | 1.384164 |
| `05-baseline` | BF16 | 52.133369 | 40.866560 | 未上报 |

| 每配置 n=2，中位数 | BF16 baseline | W8 | W4 |
| --- | ---: | ---: | ---: |
| 整次耗时（秒） | 52.175150 | 49.075479 | 43.031015 |
| 整次耗时减少 | — | 5.94% | 17.53% |
| 去噪耗时（秒） | 41.482506 | 38.838239 | 32.672134 |
| 去噪耗时减少 | — | 6.37% | 21.24% |
| 首次打包累计（秒） | 未上报 | 1.357322 | 1.341219 |
| MLX 全请求峰值（bytes） | 6,083,208,433 | 5,829,385,061 | 5,829,385,082 |
| MLX 全请求峰值（GiB） | 5.665429 | 5.429038 | 5.429038 |
| 进程 lifetime max footprint（GiB） | 6.554910 | 4.770105 | 4.778818 |

所有六格都完成 40 步，首步 BF16 GPU，后 39 步各执行 32 次 Core ML 调用，即每格 `39 × 32 = 1248` 次，零失败；W4/W8 每格均缓存 32 层。单项加速没有减少采样步数或跳过 block。`initial_pack_seconds` 包含首次 decode 的读取、切片、cast、quantize、eval，已计入去噪和整次时间，不能再次相加或从整次时间扣除后宣称实际收益。

W4 缓存减少 71.875%，但 MLX 全请求峰值仅减少约 4.17%；W8 与 W4 的峰值几乎相同。本实验没有分阶段 profiler，不能确定峰值来自哪个阶段。MLX 峰值与进程 footprint 是不同口径，不应相加。两次采样、未清文件缓存、未严格控制温度和后台负载，仅支持这个固定场景的观测，不能称为冷盘结果或跨提示稳定比例。

## 图像质量与额外样本

各配置自己的两次狐狸 PNG 逐字节一致，但候选与 baseline 不同。以下 RGB 差异按 0–255 尺度从原 PNG 重新计算。

| 狐狸，相对同套件 baseline | W8 | W4 |
| --- | ---: | ---: |
| RGB RMSE | 2.24120902 | 20.33441582 |
| RGB MAE | 0.75523376 | 5.78821055 |
| RGB correlation | 0.99900707 | 0.91405915 |
| RGB 最大绝对差 / alpha 最大差 | 53 / 1 | 222 / 2 |

历史目视记录中，W8 保留狐狸主体姿态与构图，毛发和雪纹略变；W4 后躯抬高、后腿更露出且更直、尾巴更低更横，属于姿态和轮廓变化。不能把 W4 的损失描述成仅有细节纹理，也不能仅凭像素距离决定通用画质。

文字与物体复核各为 baseline → W8 → W4 一轮，没有收尾 baseline；这些 n=1 样本不与狐狸中位数合并。

| 单次样本 | BF16 整次秒数 | W8 整次秒数 | W4 整次秒数 | W8 RGB RMSE | W4 RGB RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid-suffix-text`，seed 13 | 51.672637 | 48.930124 | 43.244446 | 64.65898318 | 57.59424321 |
| `hybrid-suffix-objects`，seed 17 | 52.297584 | 48.566110 | 41.205394 | 1.74071035 | 8.02731104 |

文字提示为 `A clean white poster with the exact words "BLUE SKY" in large dark-blue capital letters, centered, no other text, flat graphic design.`。三图短语均正确可读，但 baseline 的衬线斜体在 W8 中变为直立无衬线，W4 变为更粗的无衬线斜体；字体未在提示中限定，语义合格并不等于设计一致。物体提示为 `A red apple on the left, a blue ceramic cup in the center, and a yellow lemon on the right, arranged on a wooden table, soft studio lighting.`。三图保留颜色、数量和左右关系；W4 杯内壁变为大面积亮白，材质发生变化。三个固定提示不足以保证长文本、中文、其他种子或编辑任务的质量。

## 当前 PR 的组合结果与兼容验证

正式运行比较上游 `2a4c2b5b03b8dba4c53f0f8e8f37d0dfa1e6a441` 与当前实现 `242befe`，保持同一狐狸提示、seed 42、512×512、40 步、无 LoRA、逐层加载、分阶段释放和 ANE。组合为 suffix W4、attention W4、Q/K 融合、DBCache threshold 0.25 / max consecutive 4 / front blocks 8 / warmup 8。各格为新进程首次请求且包含 PNG 导出；smoke、screen 和生命周期请求不计入正式中位数。

| 实际运行 ID，按执行顺序 | 配置 | 整次秒数 | 去噪秒数 | DB 命中步数 | Core ML 调用 |
| --- | --- | ---: | ---: | ---: | ---: |
| `1792079228692-upstream-fox` | 上游 | 54.713867 | 44.067167 | 0 | 1248 |
| `1792079283508-fast-fox` | 四项组合 | 26.400967 | 14.828719 | 25 | 648 |
| `1792079310343-fast-fox` | 四项组合 | 25.351826 | 15.043620 | 25 | 648 |
| `1792079335789-upstream-fox` | 上游 | 53.828678 | 43.608358 | 0 | 1248 |
| `1792079389774-upstream-fox` | 上游 | 53.724513 | 43.501677 | 0 | 1248 |
| `1792079443624-fast-fox` | 四项组合 | 25.784862 | 15.320972 | 25 | 648 |

整次中位数为 53.828677625 与 25.784862292 秒，降幅 `1 − 25.784862292 / 53.828677625 = 52.098%`。组合三格均收据确认 suffix W4 缓存 32 层、1,358,954,496 bytes，attention W4 缓存 32 层、1,207,959,552 bytes；全请求 MLX 峰值中位数为 5,829,385,454 bytes，上游为 6,083,208,370 bytes。所有 Core ML 调用零失败；DB 命中仍执行前 8 层，因此调用预算为 `32 × 39 − 24 × 25 = 648`。

组合的三张狐狸 PNG 逐字节一致，对上游 RGB RMSE 为 22.60006163、MAE 为 8.11170832；这包含全部有损变化。本轮没有重复执行当前源码的独立 suffix 消融，所以不能把历史 17.53% 直接用作当前组合内 suffix 的边际收益。新进程也不等于冷文件缓存；六格小样本没有置信区间或热稳态保证。

## 所有权与取消恢复验证

2026-10-02 的同 engine 五项验证覆盖 W4 加载取消、W4 去噪取消、W4 重试、切 W8、再切回 W4。取消均不写 PNG；成功图分别与正式 W4/W8 结果逐字节一致，缓存大小随位宽正确切换。五次结束 active MLX 为 163,840–165,784 bytes。测试将调用者 wired 设为 256 MiB，四个进入去噪的请求均观察到 6 GiB，所有返回均恢复 256 MiB；加载取消未进入去噪，不补造对应观测。

2026-10-15 至 16 的组合生命周期验证继续复用一个 engine，结果为 10/10 通过：

| 场景组 | 结果 |
| --- | --- |
| text 第 4 层取消 → fast 重试 40 步 | 取消无 PNG；重试与正式 fast PNG 一致 |
| denoise 第 2 步取消 → fast 重试 40 步 | 同上 |
| denoise 第 12 步取消 → fast 重试 40 步 | 同上，覆盖 DB 可能已命中的阶段 |
| 关闭默认优化与 DB → 再开启 fast | 两次均与各自正式参考 PNG 一致 |
| DB max8 的 denoise 第 12 步取消 → max8 重试 40 步 | 取消无 PNG；重试与 max8 参考 PNG 一致 |

共 4 次取消、6 次成功，成功 fast 请求为 25 hits / 648 calls，关闭项为 0 / 1248，max8 重试为 28 / 576，均符合 `32 × 39 − 24 × hits`，且无 Core ML 失败。成功量化请求各有 32 层后缀缓存与精确字节收据。最大空闲 active MLX 为 169,252 bytes，低于 2 MiB；engine 释放后为 5,412 bytes。所有返回恢复调用者 256 MiB wired，9 个进入去噪的场景均观察到 6 GiB，测试结束恢复并核对原值 0。取消本身没有完整推理指标，因此不为取消请求推断 DB hits 或完整调用数。

当前 [真实 MLX C++ 夹具](../../../tests/native/qwen21_staged_quantization_test.cpp) 通过 36 次重复前向，验证 W4/W8 六数组 dtype、形状、紧凑 backing 与 bytes、源释放后可用、替换缓存不保留源层、显式参数编译和 BF16 合并。[静态测试](../../../tests/native/test_qwen21_staged_quantization.py) 覆盖接入约束。夹具曾在异步 command buffer 尚未 drain 时过早检查源释放计数；增加同步后验证通过，生产量化数值流程未为测试修改。小型夹具证明实现约束，不代替全模型画质或速度测试。

所有 ANE 证据限于 `cpuAndNeuralEngine` 策略、正数 runtime 调用和零失败；报告未测量实际物理 ANE 驻留或利用率。

## 便携复现

先在本仓库准备可用 native 库、Qwen-Image-2.1 BF16 checkpoint 与匹配该 checkpoint 的标准 ANE W8A8 manifest。以下在仓库根目录运行，三个路径变量按本机资源填写。脚本仅生成普通请求并调用已提交的 [benchmark_native.py](../../../tools/native/benchmark_native.py)，每格一个新进程，清除继承的 Qwen 诊断变量。当前源码下的同库 suffix 消融可复核实现，不保证重现历史二进制的逐位结果或秒数。

```sh
export MODEL_DIR="models/Qwen-Image-2.1-BF16"
export ANE_MANIFEST="artifacts/qwen21-w8a8/manifest.json"
export LIBRARY="build/native/libturbocider.dylib"
export RUN_DIR="results/qwen21-suffix-w4-repro"
.venv/bin/python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

out = Path(os.environ['RUN_DIR'])
out.mkdir(parents=True, exist_ok=False)
request = {
    'model': 'qwen-image-2.1', 'operation': 'image.generate',
    'prompt': 'A red fox standing in fresh snow, soft morning light.',
    'width': 512, 'height': 512, 'steps': 40, 'seed': 42,
    'frames': 1, 'audio': False, 'dynamic_text': True,
    'execution': 'gpu_ane', 'residency': 'component_staged',
    'loras': [], 'qwen21_w8a8': True, 'allow_approximation': True,
    'qwen21_dit_cache': 'off',
}
path = out / 'request.json'
path.write_text(json.dumps(request, indent=2))
prefix = 'TURBOCIDER_QWEN21_'
clean = {k: v for k, v in os.environ.items() if not k.startswith(prefix)}
for index, bits in enumerate((0, 8, 4, 4, 8, 0)):
    env = dict(clean)
    env.update({prefix + 'STAGED_FAST_DEFAULTS': '0',
                prefix + 'STAGED_SUFFIX_BITS': str(bits),
                prefix + 'STAGED_ATTN_BITS': '0',
                prefix + 'METAL_QK_NORM_ROPE': '0'})
    subprocess.run([
        sys.executable, 'tools/native/benchmark_native.py',
        '--library', os.environ['LIBRARY'], '--model', os.environ['MODEL_DIR'],
        '--model-id', 'qwen-image-2.1', '--request', str(path),
        '--ane-manifest', os.environ['ANE_MANIFEST'],
        '--output', str(out / f'{index:02d}-suffix-{bits}'),
        '--runs', '1', '--record-vm',
    ], env=env, check=True)
PY
```

独立 W4 应收据确认 `qwen21_staged_suffix_quantization.bits=4`、`group_size=64`、`activation_dtype="float16"`、`cached_layers=32`、`cache_bytes=1358954496`、`scope="request_owned_decode_only"`、40 步、1248 次调用且零失败，并确认没有 attention 量化、Q/K 融合或 DB 命中。W8 的对应缓存字节为 2,566,914,048；0 位应无 suffix 量化收据。三个配置必须保留 `layer-streamed text/DiT`、`component_staged`、空 LoRA 和 `mlx_cpp_metal+coreml`。

复核组合时用独立输出目录，将请求改为 `qwen21_dit_cache="fast"`，把 `STAGED_FAST_DEFAULTS` 改为 `1`、`STAGED_SUFFIX_BITS=4`、`STAGED_ATTN_BITS=4`、`METAL_QK_NORM_ROPE=1`；正式 A/B 仍须分别使用上游与当前库，依次执行 upstream、fast、fast、upstream、upstream、fast，每格 `--runs 1`。DB max8 的诊断参数与请求覆盖规则见 [DBCache 报告](04-dbcache.md)。性能测量不要混入 tensor dump、预热、取消或同 engine 文本缓存复用。

## 证据身份与留存范围

本报告重新读取原始 JSON、identity 和 PNG，核心数值已内嵌。原始报告、日志和 PNG 本地留存，未上传；下列 ID 是核对标识，不是下载链接。SHA-256 标识对应记录的字节内容，不替代原始文件公开或独立复现。

| 记录 ID | SHA-256 |
| --- | --- |
| 历史单项共用 native 库 | `f17fd4f8440fb40db71871887d74458d4f4edfe21e0003cd93d8626ad326796b` |
| `hybrid-suffix-quantization/identity.json` | `3f3f329acb904b19dbc107408db1657dd34c5b9747b2b34cf392b89f651cbaff` |
| `hybrid-suffix-quantization/runs.json` | `4c6c20f6651e95fef45f2e9df447dca1783d4b72c7043502669f2ef2b47c7148` |
| `00-baseline/report.json` | `d3f7eb5c377a8a0faf7a0097a3e0a05bb7f1f5a05aab0e2cc17f02012829e0c8` |
| `01-q8/report.json` | `62c694eb10e07cc65bc43743af11d70c734b078a9da58d073a714fa2b5b6eb45` |
| `02-q4/report.json` | `d7c5488d0b75776979033c23c6c88b87f02608d704991d747cfcb6dcf73088d6` |
| `03-q4/report.json` | `0e7ca0e199818e178266ad5b0de223e57e9aad03210a3d73d58650df509600d6` |
| `04-q8/report.json` | `450ca1c7b99e5866fa1c67e7079c99e83d8ac9dab2292a607f0755dd036bf2b5` |
| `05-baseline/report.json` | `e1732dc8e9f5512feade9c33d7e76dd0a09ace37a19babc17f41de57f47779d7` |
| `lifecycle-staged-quantization/report.json` | `643217ab7a89a11deb911470e8a6c081a64a065d4dc3c866442b3559f8da7cde` |
| 当前上游对照 native 库 | `394f27d1e4b653867846b8ae4484e8f66710071a68c574a18d96805847ca0f53` |
| 当前 PR 实测 native 库 | `36305cc4999e6dca2ec0ecaa3c81263d24a49592602b2b6514fadc0704c43d53` |
| `qwen21-m5-24gb-integration-20261015/summary.json` | `93a76ee9f601a4a47f7c6f20176e971f5427fff0142c9ffb95cbb4f9490007bf` |
| `lifecycle-combined/report.json` | `97dd6119be61fa6ad37ad92c07fd85a28028232c0741ee9017cdda156cdf0907` |
