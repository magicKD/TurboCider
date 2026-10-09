# Qwen LoRA B：融合 epilogue 的小幅整请求收益

2026-10-09收尾；组件和模型运行日期2026-10-08，Asia/Singapore，
M4 Max64GB/macOS26.6.2。接续
[BF16 ranks与B候选](local512-student-ffn-reuse-2026-10-08.md)。用户允许
delta近似，本轮以真实模型性能/图像为主要判断，不强求逐位相同。
完整目标尚未完成，混合路线仍慢于匹配纯GPU。

## 实现与误差边界

`dense_gpu_lora_b.hpp` 读取原BF16/FP16 B的物理row interval，保留
原F32 A-ranks直到小输入转换。一个Metal B dispatch完成projection、
F32 scale、可选F32 base add和一次请求输出cast。不compact W或复制
完整base；rank narrowing计入组件时间，不声称连A投影也被融合。

stacked adapters保留原顺序：projection逐adapter narrowing；delta API
先F32 sum，再最终一次cast。只有单个覆盖整个slice的eligible adapter
才直接输出requested dtype。原alpha、正/负strength及partial fused
QKV都保留，不merge/requantize模型或adapter。

`TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE=1` 默认关闭，要求明确
approximate resident512²六步LoRA GPU或Private fixed-channel runtime；
排除FP16 A-ranks、BF16 A-operand和时间reuse组合。原F32 A-ranks、
rank>=64/M>=128、row-contiguous BF16 B才选新consumer。base有效flag
忽略、非法值拒绝；request snapshot、drain和identity隔离沿用原契约。

432 actual Metal cases对独立typed-rank/F32 epilogue oracle最大误差0；
108实际Weights cases覆盖stacked±alpha、shared/sliced/full、cast和
guards，最大delta relL2约1.542%，低于5%组件预算。这不是whole-model
latent/画质通过。有限值、shape/source/memory检查和full GPU fallback
不放宽。两个早期scalar/buffer compile错误及修复后日志均保留。

## 真layer0组件：对compiled控制更快

原BF16 A/B、合成activation、原F32 ranks，三warmup/15换序hot；
host operator spans，非GPU timestamp/整模型。选定16×128：

| M / projection | original compiled ms | fused ms | delta relL2 |
| --- | ---: | ---: | ---: |
| 1056 / gate7168 | .561708 | .448458 | .002579 |
| 1056 / gate7168 + base | .715375 | .449250 | .000214 |
| 3137 / out4096 | .879666 | .619500 | .002374 |
| 3137 / gate7168 + base | 1.495667 | 1.035208 | .000213 |

无base时真实delta误差约0.24–0.26%，并非最终生成误差。完整eager/
compiled/six-tiles全部样本保留，未以慢eager分母替代compiled控制。
probe observer errors0、最大gap约.22s、binary unchanged；独立probe
用absolute rpath链接保留student-reuse Private库，无adjacent dylib，
不冒称本轮final phase库或loaded-image trace。

## 实模型：四臂、单/双参考图

原BF16 Qwen、Viggle r256、512²六步/seed29、三个fresh prompts、
conditioning miss；各独立process一冷两热，相同encoder source retention。
encoder全部GPU；hybrid Private5120、shared gate/up ranks，两臂均
关闭W-code cache、down split和student reuse。100ms memory sampling，
无strict load资格。

| route | 单图 warm request s | 双图 warm request s |
| --- | ---: | ---: |
| GPU B off | 10.941355 | 13.173294 |
| GPU B on | 10.831854 | 12.925057 |
| hybrid B off | 11.809791 | 14.477782 |
| hybrid B on | 11.569649 | 14.341818 |

GPU耗时下降1.001%/1.884%；hybrid下降2.033%/.939%，但hybrid on仍
比GPU on慢6.81%/10.96%。两workload顺序相反，不是同workload双向
重复，不宣称小幅差异具有统计稳定性，不promote默认。

每请求227个adapter bindings、192个成功channel blocks、384 shared
rank arrays，单/双图224/256 actual calls；failure/fallback/retry0，
headroom1。8个memory报告complete、swap-in/out0，peak约38.2–41.1GB。
scope是进程树load+cold+warm+exit，不归因外部服务/driver；零swap不
证明无compression pressure或物理overlap/native INT8 MAC。

目录 `outputs/local512-qwen-edit{1,2}-b-epilogue-v1-diagnostic-20261008/`。
component receipt为 `outputs/local512-qwen-b-epilogue-v1-component-20261008.json`。

有限whole/detail抽查壶形、布局与光照很接近，无新明显块纹，釉面/
高光略变；不是逐张多seed验收。自动visual manifests仍pending。
10月9日phase实验的GPU/all两臂三张PNG均与对应B-on臂exact，进一步
验证未改变这些控制输出；不能以此把旧B-off/on当逐位等价。

新核是小幅GPU性能候选，不是ANE盈利证明。最终Private/Public回归、
制品身份与清理和phase实现一并收尾，见
[本轮机器证据](../design/validation/local512-qwen-prefill-parallel-20261009.json)。
Private/Public按operation杂糅、更多scene/seed、最佳GPU baseline、
GGUF真实复用与完整base/LoRA矩阵仍待接续。
