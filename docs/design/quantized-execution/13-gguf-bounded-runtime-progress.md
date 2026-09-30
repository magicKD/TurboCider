# 13 · GGUF 有界 GPU 执行：实验实现与真实模型证据

[目录](README.md) · [基础模块](12-implementation-progress.md) · [验收合同](11-acceptance-profiles-and-feasibility.md)

本阶段已将 GGUF source、CPU materialization、固定容量 MLX slots 和既有
StageExecutor 接到真实 Z-Image Turbo。状态为明确选择的**实验实现**，不是 R0/R1
全部认证完成，也不是 W8A8/ANE 交付。生产默认、旧格式拒绝及 catalog 不变。

## 1. 实际接入与生命周期

- GgufWeightPager 使用原生 verified SourceLease，相同 fd 完成目录和 payload 读取，
  校验内容身份/type/shape/range/generation；文件读取设置 F_NOCACHE。
- source只缓存一份压缩权重；每次请求的payload/capacity独立计费，不保存另一套
  全模型 affine/dense 权重。materialized slots精确为1/2，没有无界ready队列。
- 复用owner/fill queue/ticket/reader/cancel/drain；worker只用提前捕获的CPU spans，
  不创建MLX arrays或操作stream。GPU每个main block真实eval完成后归还reader。
- 物理backing的StorageLease放入array Data owner/deleter；逃逸view继续持有同一
  claim，pager/slot删除不会提前使账本归零。工作集按16KiB上界和实际allocator容量验证。
- 原本浮点的fixed/refiner参数使用source backing别名，不展开第二份refiner bank；
  量化refiner当前拒绝。其源浮点内存仍计费，不能推广到任意GGUF文件。
- encoder完成后释放权重，再加载GGUF；denoiser source/pool drain并释放后才加载VAE。
  请求内allocator cache限额0；不以删字典或清空logical ticket自证物理释放。

唯一managed ledger覆盖GGUF source/slots，但尚没有encoder/VAE/activations/driver/
control的完整envelope。whole-request guard因此返回qe_envelope_unknown，结果明确
whole_request_bounded_certified=false，不能称16GB真实硬件或RAM hard-cap认证。

Q8实测payload=7,224,676,608 bytes，source capacity=7,225,299,200 bytes，其中原
浮点source=1,459,147,008 bytes。dense-BF16/FP16每槽361,922,560 bytes，affine每槽
203,685,888 bytes；两槽相应翻倍。4 steps×30 main blocks的fills/computes均为120。

## 2. 新发现：浮点 loader 语义

真实小切片probe确认当前MLX把GGUF BF16浮点矩阵加载为FP16，而源F32 norm/bias
保持F32。Q8 codes/scales/biases与本实现布局变换逐字节一致，FP32 source weight
oracle对照max error=0。不能仅凭文件dtype猜当前算术。

因此直接按BF16 source执行与旧importer有不同的dtype promotion，尤其影响timestep/
refiners。匹配eager边界未消除差异；修复importer语义的兼容profile则取得整图exact。
probe为tools/validation/gguf_mlx_weight_probe.cpp，输入为真实checkpoint有限切片。
被检查的浮点key包括cap embedder、noise QKV/norm、main bias和pad token。

| profile | main量化权重表示 | 文件BF16浮点参数 | mode |
| --- | --- | --- | --- |
| z-source-mixed-v1 | BF16 dense，一次RNE | 原BF16 | bounded_dequant |
| z-source-mixed-f16-v1 | FP16 dense | 原BF16 | bounded_dequant |
| z-source-exact-f32-v1 | FP32 dense | 原BF16 | bounded_dequant |
| z-source-native-affine-v1 | 原Q4/Q8 affine | 原BF16 | bounded_packed |
| z-mlx-compat-affine-v1 | 同原MLX Q4/Q8算子 | 当前importer的FP16 | bounded_packed |
| z-mlx-compat-f16-v1 | FP16 dense | 同上 | bounded_dequant |
| z-mlx-compat-f32-v1 | FP32 dense | 同上 | bounded_dequant |

compat的BF16→FP16在同尺寸缓存backing内转换一次，不保存第二份完整浮点权重；
原文件和量化codes/scales不改。它是显式importer representation，不声称保持原BF16
source精度，overflow/nonfinite失败而不clamp。source profiles需要真正source语义的
golden，compat profiles对照旧MLX；两者不能继承数值资格。

## 3. 执行入口与发布隔离

```sh
TURBOCIDER_BUILD_OUTPUT_DIR=build/quantized-execution \
TURBOCIDER_BUILD_LIB_ONLY=1 TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1 \
TURBOCIDER_BUILD_PYTHON=.venv/bin/python bash tools/native/build.sh

.venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib \
  --model models/z-image-runtime-gguf-q8 \
  --output outputs/new-gguf-run \
  --prefetch -1 0 1 --precision z-mlx-compat-affine-v1 \
  --steps 4 --size 1024 --seed 1234 --dump
```

输出目录必须不存在。-1是原native packed控制，0/1是bounded。dump和20ms parent
memory sampling是诊断，不用于正式速度资格；工具保留request、加载库hash、所有
结果/事件/VM样本和媒体，并持有GPU diagnostic lock。

schema2 execution.quantized_execution已接原生严格parser/plan。enabled/version必填，
空对象/未知key/错误类型/不支持p=2/ANE/requantization/LoRA/legacy budget等拒绝。
manual K/G/P/D/Q须一致。普通构建generate返回qe_capability_unqualified，只有明确
实验构建执行；没有用户可注入的qualification manifest或新catalog记录。

## 4. 实际通过和失败

| 对照 | 输入 | 结果 |
| --- | --- | --- |
| Q8 source-mixed K1/K2 | 512,seed42,4 steps | 每步latent/decoded/PNG exact；仅同profile调度证明 |
| Q8 source-mixed vs旧packed | 同上 | final relL2=0.05850、cos=0.99830；旧-reference N1失败 |
| Q8 source-f16 vs旧packed | 1024,seed1234,4 steps | relL2=0.09211，N1失败 |
| Q8 source-affine vs旧packed | 同上 | relL2=0.09343，匹配eager仍失败，促使检查loader dtype |
| Q8 mlx-compat-affine 原packed/K1/K2 | 同上 | 所有逐步latent、decoded和PNG字节exact；120 fills/computes |
| Q4 mlx-compat-affine 原packed/K1/K2 | 512,seed20260930,4 steps | 同样exact，覆盖另一种量化及文件dtype |
| Q8 mlx-compat-f16 vs旧packed | 1024,seed1234,4 steps | relL2=0.006639、cos=0.999978；该输入final-latent子门通过 |
| Q8 block5取消后同引擎重试 | 512,4 steps,K2 | 首次status2/cancel且无PNG；第二次成功、120 fills |

原始目录为outputs/quantized-execution-20260930/gguf-r1-*。比较工具
tools/validation/compare_gguf_execution_outputs.py不依赖MLX/safetensors包，保存
exact/relL2/cosine；其N1 verdict仅final latent，不是全部层/媒体或模型资格。

compat affine K1 exposed wait约6.28s、K2约0.22s，显示prepare stall被部分隐藏；但
完整denoise约21.25s、原packed约19.96s，**不是已获加速**。单轮compat dense-FP16
约19.62s也不支持推广门，且cold/loading/cache状态不同。禁止把冷热墙钟差当加速比。
暂未注册accelerated，仍需同预算/retention/cache的正式配对campaign。

## 5. 已执行的工程检查

可携带的[诊断摘要](validation/gguf-bounded-runtime-diagnostics-v1.json)保留加载库hash、
raw-report digest、shape/seed/steps、actual slots/fills、图片/tensor身份、内存/时间及
通过或失败的局部数值门；不包含大图片/模型，不是qualification manifest。

- CPU decoder/affine pack固定GGML oracle和真实切片、finite FP16穷举、target/stride/
  slice/guard及非法part检查24项通过。affine signed zero按既有D0合同canonicalize，
  非零值不放宽容差。CPU SIMD扩展到Q8 FP16/FP32输出及原浮点same-dtype copy。
- 真实Metal pager K1/K2、两次pass、GPU输出、stale ticket、逃逸view账本claim及释放通过。
- 原生request正负例5项、旧GGUF门禁4项、独立ASan/UBSan decoder程序通过。
- 普通非实验构建已在真实generate入口验证拒绝：status1、qe_capability_unqualified，
  约0.17s返回，无PNG/新权重执行；原始记录为gguf-r1-release-rejection/report.json。
- 既有10个Z模块72项通过；独立目录完整实验dylib构建及语法检查通过。
- NativeGGUF wrapper透传inner session的quarantine，不能在drain不明时绕过进程隔离。

当前发行dylib/CLI/App未替换，默认用户不执行新路径。

## 6. 尚未完成

关键代码位置：native/runtime/streaming/gguf_weight_pager.*、native/models/z_image/
gguf_execution.*和z_image.cpp、native/core/quantized_execution.*、native/core/
gguf_affine.*。request/results bridge保留显式实验资格和managed/whole-request区别。

R0完整encoder/VAE/tokenizer/BF16-reference/噪声campaign binding未完成；R1的48-case
媒体、全部局部数值、full envelope/真实小容量硬件、正式性能资格未取得。本阶段
只是实验纵切及所列工程/输入验证，不标记R1发布完成。

后续仍须完成：同预算dense oracle、K/mixed真实文件、Qwen3 adapter、唯一whole-request
账本/envelope、量化refiner、p=2、streamed source/tiles、ConvRot、GPU/ANE并行、M5
和static/runtime W8A8。完整线程目标保持active，不因本阶段已出图而删减。
