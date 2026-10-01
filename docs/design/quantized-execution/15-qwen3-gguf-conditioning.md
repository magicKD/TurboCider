# 15 · Qwen3 GGUF conditioning 与 Z 组合执行（2026-10-01）

[目录](README.md) · [组件合同](09-release-scope-and-component-contracts.md) · [验收](11-acceptance-profiles-and-feasibility.md)

本阶段实现 R2/P3b 的 **Qwen3-4B GGUF 有界 conditioning 与 Z 实验接入**，真实 Q8、
mixed Q4_K/Q6_K 已运行。不是完整 R2/产品/媒体/内存/加速资格；生产 catalog、发行
`build/native` 和默认 encoder 不变。下面分别保留实现正确性与原 BF16 画质参考的差异，
不因“成功出图”将后者重标为通过。

## 1. 实际模型与 source 接入

`Qwen3ConditioningState` 从原 resident consumer 拆出共享计算，持有 activation/RoPE/mask/
hidden taps，不持有全模型 Weights。resident 原 eval interval 与 hybrid/fallback 不变。
新 GGUF consumer 在每层 residual 完成后真实 `mx::eval`，再报告 reader complete，
不能将原来的四层 lazy graph 放进两槽。显式 defer 或 interval≠1 返回配置冲突。

`qwen3_gguf.*` 使用 verified SourceLease 同时绑定 GGUF、原 config、原 tokenizer JSON：

- standard dense Qwen3、H=2560、heads32/KV8/head128、theta1e6/epsilon1e-6；F/V/层数与
  绑定 config、GGUF metadata 双向核对。不是按文件名猜模型，不支持 MoE/Next/VL。
- 全部层/embedding/output norm/可选 head 做 required-key、shape、type/range 验证；实际
  只执行 block0…34，取 block34 post-residual，不 final norm，不执行 block35/lm_head。
- 标准 GGUF 名字映射为 HF Qwen3 consumer 名字；`qk_layout=hf-half-split-v1`，没有自行
  添加 Llama permutation。数学、causal/padding mask、GQA 和 RoPE 均复用同一个实现。
- 量化字段使用既有 CPU decoder，FP32重建后一次 RNE 写 BF16；原浮点字段保留 dtype。
  residual/RoPE FP32，conditioning BF16，profile=`qwen3-z-source-mixed-v1`。
- 每个实际 token ID 对照 GGUF vocabulary 中的 UTF-8 token strings：本地 tokenizer
  151,669 个 ID 全部匹配，GGUF有151,936个词表行，未用 padding词表行不展开。
  GGUF chat template 不覆盖图像 prompt template。冲突直接拒绝。
- parser 为 metadata 记录 checked payload offset/length；token-array reader 只读 held fd。
  config JSON ≤1 MiB、tokenizer/vocabulary 各≤32 MiB，重复 JSON key、坏类型/短读拒绝。

`GgufWeightPager` 增加显式 `gguf-packed-gather-source-v1` resident materialization：目标是
immutable U8 byte backing，不是伪装的 dense embedding。`gather_rows` 仅解码请求 rows，
保持重复/顺序，输出独立 backing、同 ledger计费；不把完整 packed embedding绑定成浮点矩阵。
拒绝非法 ID、过长 gather、descriptor未声明source、短buffer/错误recipe。

复用 StageExecutor/原队列/slot tracker：p=0/1、1/2 slots、单层 group、一个 fill worker。
没有第二个 scheduler、全模型 dense dictionary或额外 ready bank。content ticket检查仍有效；
norm的小副本和全部raw/target backing都计入 managed weights。fill workers持有组件自有
cancel token；owner同步检查engine取消，drain失败保留/quarantine而不提前释放。

## 2. 实验入口与生命周期

仅 `TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1` 编译支持模型入口；普通构建真实
generate返回 `qe_capability_unqualified`，无PNG、无新encoder执行。GGUF wrapper是lazy
初始化，单独 engine-create 成功不是门禁失效，也不是generate授权。

显式选择：

```sh
TURBOCIDER_Z_QWEN3_GGUF=models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf \
TURBOCIDER_Z_QWEN3_GGUF_CONFIG=models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json \
TURBOCIDER_QWEN3_GGUF_PREFETCH=1 \
.venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib --model models/z-image-runtime-gguf-q8 \
  --output outputs/new-qwen3-z-run --prefetch 1 --precision z-mlx-compat-affine-v1 \
  --steps 4 --size 512 --seed 42 --dump
```

encoder `TURBOCIDER_QWEN3_GGUF_PREFETCH` 与此工具的 DiT `--prefetch` 是两个明确
组件设置，不混报槽数。`TURBOCIDER_QWEN3_GGUF_WEIGHT_LIMIT_BYTES` 默认8 GiB，
仅managed encoder source/slots/gather ceiling，**不是 whole-request RAM guard**。
现有 whole-request constrained/public 请求、encoder ANE、encoder LoRA仍拒绝；正式
per-component request schema/qualification尚未交付。不能用环境变量授予生产资格。

新 encoder 使用同lease构造的 tokenizer；conditioning cache identity含source/config/
tokenizer内容、profile、prefetch、managed ceiling，连同原prompt/dynamic/LoRA生命周期。
切回原 encoder不复用GGUF conditioning。模型可只绑定GGUF encoder，不要求存在被忽略的
dense encoder。encoder drain/release后才进入DiT；新run result明确报告encoder backend/
每层eager提交/FP32 residual和profile，不把它标为ANE或compiled encoder。

## 3. 真实 fixtures 与数字

固定官方仓库提交 `bc640142c66e1fdd12af0bd68f40445458f3869b`，download完成后完整SHA-256
对照LFS身份；没有覆盖本地原模型。fixture目录均398 tensors：

| fixture | bytes | SHA-256 | types |
| --- | ---: | --- | --- |
| Qwen3-4B-Q8_0.gguf | 4,280,404,704 | `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300` | F32/Q8_0 |
| Qwen3-4B-Q4_K_M.gguf | 2,497,280,256 | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` | F32=145/Q4_K=216/Q6_K=37 |

这是固定测试数据，不是“任何Qwen/Unsloth GGUF都已支持”。weight文件不提交，license/
来源保持原样。需复取时使用上述固定commit的 Hugging Face resolve endpoint，而非main。

两种文件 × 短21 tokens/长273 tokens × p=0/1，共8次真实conditioning，单/双槽
四组全部逐位exact。每次35 fills/35 compute，目标各槽201,916,416 bytes。观测managed峰值：

| source | packed capacity | 单槽峰值（短） | 双槽峰值（短） |
| --- | ---: | ---: | ---: |
| Q8 | 4,167,187,456 | 4,369,174,528 | 4,571,046,912 |
| Q4_K_M | 2,427,681,792 | 2,629,668,864 | 2,831,541,248 |

未使用 block35/head 的payload不驻留，embedding仅packed一份，gather output另计。
long gather峰值更大，也保存在receipt。managed容量不是metadata/activations/allocator/
framework/OS整体上界，whole_request_certified=false，不声称真实低容量设备资格。

Q8 decode总active约0.49–0.58s；Q4_K/Q6_K约7.1–7.8s。当前K decoder仍是scalar，
p=1只能隐藏部分等待，不能突破持续decode吞吐；这是下一阶段优化的实际瓶颈。
这些是组件诊断，不是正式cold/warm跨arm性能campaign，含原BF16参考的elapsed更不比较速度。

## 4. 正确性、源量化损失与完整生成

同token原BF16参考的最终conditioning差异（不是同源O1实现误差）：

| source | 短 relL2 | 长 relL2 |
| --- | ---: | ---: |
| Q8 | 0.006312 | 0.013902 |
| Q4_K_M | 0.036081 | 0.128188 |

原BF16是另一个存储表示；这些数值单列源量化/目标舍入差异，不将Q4源损失混成D1
scheduler错误，也不据Q8局部指标推导媒体资格。K类型的独立source-quant model golden/
完整O1数值campaign仍需完成，不能只靠两种slot对照自己证明所有模型语义。

实际完整 Z-Image Turbo，Q8 DiT固定，512²/seed42/4 steps/default portrait：

- Q8 encoder p0/p1：conditioning、全部step/final latent、decoded和PNG exact；各35层encoder、
  120个DiT block执行。第5层encoder取消：status2、无PNG，同engine重试status0完成。
- 对原BF16 encoder，Q8 final latent relL2=0.031152/cos=0.999517；**0.03局部门失败**，
  没有放宽阈值或开启默认。这个different-source对照不替代同源O1，也没有人工媒体门。
- Q4_K_M encoder+Q8 DiT成功完成，但vs原BF16 final latent relL2=0.326160/cos=0.946611，
  差异显著，不能取得画质/默认资格。下一步需同源reference和实际图像审核，不隐藏此负结果。
- 不启用GGUF encoder时，最新状态重构后的原路径与此前原BF16-encoder/Q8-DiT golden
  initial/所有step/final latent、decoded/PNG exact；默认数学未改变。

便携回执：

- [8次组件、原BF16差异与5负例](validation/qwen3-gguf-component-20261001.json)
- [完整请求、取消重试与逐tensor比较](validation/qwen3-gguf-z-generation-20261001.json)

raw目录 `outputs/quantized-execution-qwen3/` 保留媒体、conditioning、全部samples/
token IDs、日志和失败记录。源码/库/probe身份和raw receipt SHA存于回执，不登记qualification。

## 5. 已执行的工程验证与复现

- 新状态相对固定 `3030a77` 原函数、resident wrapper、逐层提交：12组GPU exact；覆盖
  FP32/non-FP32 residual、1/33/129 rows、causal/padding、GQA切换、多tap，不由被测函数自证。
- 真实pager K1/K2/两pass、stale ticket、GPU矩阵、packed embedding gather重复/顺序、
  非法ID、逃逸view账本claim与最终释放通过。
- 8个component正例、config/tokenizer冲突、1-byte预算、p2、lazy interval共5负例通过。
- 固定GGML CPU/真实Z Q8/Q4 decoder24项、既有Qwen sweep/Z/旧GGUF等81项通过；
  ANE/Core ML/MLX真实集成12项通过；prefill policy4项通过。
- 最终独立 decoder ASan/UBSan 安全程序通过，涵盖allocation-free、alias/unaligned、
  overflow及cancel；它不替代完整异步encoder故障矩阵。
- 最初prefill probe有旧MLX ABI链接错误；隔离重建后4项重新通过。不是将loader失败计为pass。
  pytest缺依赖已安装固定8.3.5，未修改生产requirements或下载替代dense模型。
- 最终实验/普通两个隔离lib完整构建、syntax/Python compile/bash syntax/diff检查通过。
  普通build的实际generate拒绝已确认，发行`build/native`未替换。

```sh
TURBOCIDER_BUILD_OUTPUT_DIR=build/quantized-execution TURBOCIDER_BUILD_LIB_ONLY=1 \
  TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1 TURBOCIDER_BUILD_PYTHON=.venv/bin/python bash tools/native/build.sh
bash tools/native/build_qwen3_source_probes.sh
TURBOCIDER_TEST_GPU=1 TURBOCIDER_NATIVE_LIBRARY_DIR=build/quantized-execution \
  .venv/bin/python -m unittest discover -s tests/native -p test_qwen3_state.py -v
TURBOCIDER_TEST_GPU=1 TURBOCIDER_NATIVE_LIBRARY_DIR=build/quantized-execution \
  .venv/bin/python -m unittest discover -s tests/native -p test_gguf_weight_pager.py -v
TURBOCIDER_NATIVE_PROBE_DIR=build/quantized-execution \
  .venv/bin/python -m pytest tests/native/test_qwen3_prefill_policy.py -q
.venv/bin/python tools/validation/run_qwen3_gguf_validation.py \
  --q8 models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf --q4k models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
  --config models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json \
  --tokenizer models/Tongyi-MAI-Z-Image-Turbo/tokenizer/tokenizer.json \
  --reference models/Tongyi-MAI-Z-Image-Turbo/text_encoder --output outputs/new-qwen3-validation
```

## 6. 剩余范围

接下来优先优化 K decoder / 短token packed路径，补同源 source-quant/O1、完整质量与
组件消融；正式per-component schema/plan/source closure及whole-envelope尚待交付。
p=2、packed-streamed、tiles、量化refiners、其他encoder/IQ、ConvRot模型/ANE并行和
M5/static/runtime W8A8仍继续按原目标推进。R2功能实验通过不意味着R0–R5全部完成。
本阶段没有清理模型/缓存/历史证据；新增两份fixture共约6.78 GB，磁盘仍有约10 GiB余量。
