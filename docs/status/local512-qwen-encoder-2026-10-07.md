# 512²：Qwen 编码器混合执行、真实 LoRA 编辑与冷成本

记录日期为 **2026-10-07 UTC**；实验目录的 `20261008` 使用
Asia/Singapore 本地日期。M4 Max 64GB / macOS26.6.2。只读本地
Qwen Image 2.1、Z-Image 与原 LoRA；没有下载模型或改写模型、参考工程。
当前目标仍 active：512² base/真实 LoRA、Qwen 1–2 参考图编辑、DiT 与
encoder 加速；本记录不是全部模型/格式/后端的完成声明。

## 本轮实现

Qwen3-VL 的 36 层语言 FFN 现在可以借用共用 `ane::HybridFfn`，
沿用 Public/Private executor、stager、channel/row scheduler、独立
GPU complement、完整 GPU 重算与 drain。权重在 attention 前 staging；
原 causal/padded-key mask、FP32 normalization、mRoPE、DeepStack 在
FFN 后注入的顺序不变。视觉塔、attention 与 DiT 默认仍在 GPU。

入口是显式 `encoder_ane_manifest`；要求 resident、512²、授权
approximation、无 prompt enhancement，编辑只接受 1–2 参考图。
可以只加速 encoder 而让 DiT 保持 `execution=gpu`。没有用 encoder
开关隐式启用 DiT ANE；当前 encoder 不提供 `PRIVATE_ANE_CHANNELS=auto`
的 calibration workload，试验必须选固定 channel share。

`ConditioningCache` 把生成与编辑缓存绑定到同一 encoder identity。
canonical manifest 路径、manifest digest 或 executor policy 改变时，
两种条件缓存一起失效，避免先切换 T2I 后错误复用旧编辑条件。
公共 executor identity 补上 `RUNTIME_ANE_CHUNKS/PROFILE`，不同调度
配置不再误用旧 executor/cache。命中缓存不携带新的 encoder metrics。

冷尝试的 metrics 保留，即使 calls=0；但实际执行、backend、GPU graph
与 precision 标签按调用事实记录。Private W8 不再被标成 Core ML FP16，
plan 与顶层结果保持一致。encoder metrics 记录实际 36 层、释放边界，
且与 DiT counters 分开。selection 在 DiT route 描述覆盖后再追加，
避免组合路线丢失 encoder 说明。memory headroom 使用不溢出的减法。

## Splash 与设计文档的对应

只读 `../splash` HEAD `134807b80bd6f64b1533cb1306358c042bb750d5`
没有 ANE FFN；实际 #260 参考实现位于
`../references/splash-ffn/runtime/ops/AneFfn.hpp` 与 `ane/Handoff.hpp`。
本轮复核了 intermediate-channel split、两套 W staging、shared-event
handoff、超时/失败丢弃整段结果与 GPU 重算；也复核了当前 Splash 的
`runtime/metal/CommandGraph.hpp` 参数所有权。

两份 `../notes/` 设计文档中的 channel-unit512、H128/H512、runtime
weight-source、Private/Public 分层与统一 memory/fallback 合约继续适用。
不把其 LLM prefill 收益外推成 diffusion/encoder 收益，也不把 W8A8
representation 或 host async counters 当作 native INT8 MAC/物理并发证明。

## Base：真实 1–2 参考图，encoder 仍较慢

同一 v3 Private library，每格两 arm 独立进程串行；512²、40 步、seed29、
参考图缩放512、DiT GPU。encoder 使用 rows512/Fa5120、W8A8 channel、
GPU IOSurface、FP32 partial join、fixed async；prefetch/lookahead 关闭。
wrapper 保留连续 host-load 与 binary/library identity；**非严格 E2E
qualification，也没有 GPU/ANE physical trace**。

| 参考图 | GPU text / request s | encoder text / request s | actual encoder calls |
| --- | ---: | ---: | ---: |
| 1 | 1.094386 / 47.164967 | 2.623633 / 48.718082 | 36 |
| 2 | 1.484109 / 50.653335 | 3.255879 / 52.387182 | 72 |

候选 runtime failure/fallback 均0。第一次每个 gate/up/down scale
都未命中：108 misses、0 hits；执行器 load 约0.78–0.80s，prediction
分别0.253/0.486s。pre-FFN 包含 input/attention 等待，不能当作纯
weight stage 或纯 GPU kernel time。原失败的 v2/gate receipt 保留，
一参考图使用 `gpu-v3/` 与 `generation-receipt-v3.json`，不混用 v2 arm。

## 通道份额与 bucket：局部热收益不足以覆盖冷成本

新增 `tools/native/qwen21_encoder_runtime_probe.cpp`：加载真实本地
encoder/vision 权重，只执行编码组件，不加载 DiT、不输出 activation。
一个真实参考图、302 language rows、原 mRoPE/DeepStack；同 binary，
每个配置三次串行换序热样本全量保留。热测保留同一 source owners 和
executor，**不同于生产 request-local encoder 生命周期**。

| bucket / Fa | GPU hot median s | hybrid hot median s | hybrid first incl. setup s | conditioning rel L2 |
| --- | ---: | ---: | ---: | ---: |
| 512 / 5120 | 0.358410 | 0.376189 | 1.618555 | 4.072% |
| 512 / 1536 | 0.356372 | 0.368666 | 0.924316 | 3.254% |
| 512 / 3072 | 0.357048 | 0.343037 | 1.114287 | 3.586% |
| 320 / 3072 | 0.358681 | 0.343522 | 0.802333 | 3.586% |

3072 的热组件较短；320 bucket 把 setup-inclusive first 从1.114降至
0.802s，但仍比对应 GPU first0.367s 慢。没有跨配置拼 GPU 分母。
每配置144实际调用、fallback0；108 scale misses +324 hits 清楚地
区分首轮和三轮 owner-retained 热复用。不将小样本热收益推广为默认。
生产冷请求的尺度代次和图初始化仍需解决，简单加深预取不是已证实收益。

## 真实 LoRA：1–2 参考图兼容，仍不盈利

同一 v4 Private library，真实本地 Viggle v0.2.1 r256、strength1、
inference-time、6步、seed29。显式 `QWEN21_LORA_REF512_DIAGNOSTIC=1`，
GPU 和候选都用相同 reference resize/adapter/GPU DiT；候选 encoder
选 rows320/Fa3072，其余控制不变。不是 base-vs-LoRA 比较。

| 参考图 | GPU text / request s | encoder text / request s | calls / applied LoRA projections |
| --- | ---: | ---: | ---: |
| 1 | 0.794874 / 12.136152 | 2.111241 / 12.819890 | 36 / 227 |
| 2 | 1.408437 / 13.865925 | 2.207806 / 14.625359 | 72 / 227 |

两臂均227真实 adapter projections、实际6步；候选失败/回退0。
这些是单次冷请求诊断，denoise的小幅波动不归因于 encoder。
因此保持 encoder 显式实验；不新增 encoder+DiT 同开“最快默认”。

四对 base/LoRA PNG 的 whole、center、top-left、bottom-right 均已逐张
agent 目视：壶形、构图、把手、光照非常接近，有釉面纹理/高光/木纹
小差异，没有显眼的新结构性破坏。蓝壶两臂都不透明；不声明透明材质
保真。review 标为 `agent_reviewed_user_acceptance_pending`，
`qualification_passed=false`；不是多 prompt/seed 或用户认可。

## ConvRot GPU / GGUF 提前解码 / 混合 backend 决策

继续保留已有 `convrot_rotation.hpp` register H256 与 Private Comfy
A8 register stager。它们保持旧 butterfly 的舍入/排列；不能据此
消除与 dense-H control 的历史 trajectory 差异，见
[ConvRot A8 kernel](convrot-register-a8-staging-2026-10-06.md)。

GGUF 容器与 native 消费格式不混为一谈。已有
`native/runtime/streaming/affine_dense_window.hpp` 是有界同步研究窗口，
不是已自动接入的异步 ahead-decode。原真实权重测量在 M≈1056 每次
重解码不盈利；应按 `decode / 实际复用次数 + dense GEMM < packed QMM`
判断，同一矩阵被 eviction 后的下次执行不能算 cache hit。512优先时
先保留 packed source/内容代次尺度缓存，不制造 full dense sidecar。
详细数字见 [consumer 对照](convrot-register-dense-window-2026-10-06.md)。

Private/Public 继续共用 model/source/scheduler，但一次 operation 只选
一个 ANE executor，再与 GPU complement 并行。现有 auto 是 capability
fallback，不是已完成的 per-layer profitability 三选一路由。短/敏感
operation 优先 GPU；不同时常驻 Private W8 与 Public FP16 权重副本。
Qwen encoder 的首次单次36层不能直接复用 DiT 的 channel share：这里
图 load 与每层首次 scale 代次成本更显著，需独立 profiling/calibration。

## 回归、构建与磁盘

重建/回归入口（Core ML 模板需使用本地 exporter，不能以空 JSON 代替）：

```sh
TURBOCIDER_ENABLE_PRIVATE_ANE=1 TURBOCIDER_BUILD_LIB_ONLY=1 \
TURBOCIDER_BUILD_OUTPUT_DIR=build/encoder-private bash tools/native/build.sh
TURBOCIDER_NATIVE_OUT=build/encoder-private \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/encoder-private \
bash tools/native/build_qwen21_encoder_probe.sh
# request-gate tests 另需同目录的 native CLI；见现有 build.sh 的 CLI link 行。
TURBOCIDER_TEST_QWEN_ENCODER_ANE=1 TURBOCIDER_TEST_RUNTIME_ANE=1 \
TURBOCIDER_TEST_GPU=1 TURBOCIDER_NATIVE_LIBRARY_DIR=build/encoder-private \
.venv/bin/python -m unittest -v tests.native.test_qwen21_encoder_ane tests.native.test_ane_runtime
```

session test 还需显式 `TURBOCIDER_TEST_QWEN_ENCODER_SESSION=1` 与两个
真实 template manifest 的 `TURBOCIDER_TEST_ENCODER_MANIFEST_A/B`，并采用
上文 Private fixed-channel 控制；不要让普通 tests 自动加载大模型。

Private、Public 选定 suite 各16项通过，无skip；含 actual Public
Core ML/MLX 微图、causal/padding/visual DeepStack、short-row GPU exact
control、完整 FFN fallback/取消/ownership 与新 cache/receipt/request gate。
额外1项真实 Private session C API 测试通过：GPU edit → ANE T2I →
ANE edit →同条件cache hit → GPU edit → GPU cache hit →另一manifest→
原manifest，实际 cold calls36、cache hit calls0；malformed 与错误
geometry 模板拒绝。prepare-only 不执行 denoise、不导出媒体。
这不是全仓 `make test` 或完整所有模型矩阵通过。

v4 两库各488 source input hashes 与生成 manifest 匹配。Public actual
release-binary guard通过，Private不用于稳定发行。库 SHA256：

```text
Private 6bd4d56d15baf40ba4e3b2f842eb570a68f19b9d5bf6a8334f92622b7ae12bbe
Public  34a421cb9d04697708c945e5b6a07e2be4b2b154ac2361176b90799f3314c7ec
```

构建包含先前 ConvRot working-tree drafts，提交不夹带它们；不是
clean staged-tree build。证据 hash、路径、全部组件/媒体 receipt 见
[机器记录](../design/validation/local512-qwen-encoder-20261007.json)。
连续负载拒绝的 Z/Qwen base512 strict screens 保留，尚无新有效比值，
不拼接被拒绝的 arm，不结束外部 ComfyUI 或降低 gate。

前次接续清理约1.6GiB已验证 module-cache；本轮在所有 jobs 完成后，
仅清理 v1/v2/v3 Private 与 v4 Private/Public 的1077个 `.o`，逻辑
大小83,304,424 bytes（约79MiB）。这些可以重建；保留 dylib、CLI、
probe、manifest、log、PNG、模型及适配器。五份新 module-cache 为空。

下一步优先：独立 encoder profitability/短序列 decline、可预算的图复用
与 identity 隔离、512 base 严格窗口、实际 DiT+encoder 两参考图/LoRA
组合、GGUF 消费窗口真正复用与 ConvRot 数值/视觉修复。总目标仍 active。
