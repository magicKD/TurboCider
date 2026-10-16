# 可选产品与上游接入：与自包含核心分开

[总目录](../README.md) · [自包含 ANE 样本](../labs/ane.md) · [混合 FFN 设计](../designs/hybrid-ffn.md)

本目录的 Public W8A8 脚本保留原来的完整 TurboCider 原生集成能力，没有用较小教学边界冒充或替代它。它需要产品`native/`、`tools/`、`tests/native/`，**不是学习核心的运行依赖**，也不依赖 Splash/vpipe/Sana checkout。只复制`techniques/`时，使用本地 CPU/Metal/Core ML 样本，不运行这些可选接入。

[public_w8_product.py](public_w8_product.py)导出完整 normalized SwiGLU/W8A8 图，调用产品两个原生 fixture，测试真实 GPU stage→CoreML→restore、修正输入、bank/source、late failure 等合同。仍不授予实际模型 LoRA、端到端性能或硬件 ANE 驻留资格。

```sh
python -B techniques/integrations/public_w8_product.py --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/new-product-w8-integration.json
```

使用已准备好NumPy/CoreML等依赖的Python环境；脚本不安装包或下载模型。原始报告写新的ignored路径，源码输入前后校验SHA。历史结果见[上一轮验收](../review/labs-validation.md)，当前教学源码结果另见[自包含验收](../review/self-contained-validation.md)。

## MFA / MPS GEMM研究客户端

[mfa_gemm_bench.swift](mfa_gemm_bench.swift)是可选的原始Swift客户端，需要另外固定版本的`philipturner/metal-flash-attention`之`FlashAttention`模块；目前参考版本为`8671cddc38f19a6eadb804dee6a3ca2954b8bf32`。在上游的隔离副本中添加executable target使用，不修改参考checkout，也不将它加入产品依赖。

它比较合成FP16输入的MPS与MFA单个驻留GEMM，含ABBA顺序、每对输入generation、完整command完成、padding初始化、逐轮NaN输出poison、尾部guard、有限值检查和每arm独立的抽样double oracle。它不是完整FFN、真实权重、模型质量或端到端性能验证；MPS也不等于产品优化后的compiled MLX基线。

2026-10-15已有尝试在上游动态Metal shader编译时报`illegal string literal in 'asm'`；另一次上游测试缺少XCTest。尚无有效的数值/性能报告。本次整理仅通过Swift语法解析，未重新构建上游或运行GPU；不能据此宣称兼容性或加速。研究状态继续在[工作台](../review/research-campaign-20261015.md)保留。

## Qwen真实FFN捕获与可移植fixture

[capture_qwen_ffn.py](capture_qwen_ffn.py)调用现有native诊断hook：仅base、512px、20steps、seed42、cache off，强制GPU FFN；保存3层×3步的9个原始BF16张量，最多约72 MiB。校验CLI构建的native source hashes，保留二进制hash和checkpoint stat identity；不拿任意旧binary冒充当前源码。

产品request仍要求显式公共runtime manifest，即使chunks=0；它必须是完整H4096/F12288的SwiGLU，而不是6144-channel子图。本轮先后发现这两项请求/geometry拒绝，未绕过验证器。正确入口是：

```sh
.venv/bin/python -B techniques/integrations/capture_qwen_ffn.py \
  --native-dir "$TC_NATIVE_DIR" --model-root "$TC_QWEN_ROOT" \
  --ane-manifest "$TC_PUBLIC_RUNTIME_MANIFEST" --output-dir outputs/my-new-qwen-capture
.venv/bin/python -B techniques/integrations/qwen_ffn_fixture.py \
  --checkpoint "$TC_QWEN_CHECKPOINT" --capture-root outputs/my-new-qwen-capture \
  --layer 0 --output-dir outputs/my-new-qwen-ffn-fixture
```

两个输出目录必须是新的。实际模型、原始prompt/图片/张量只留本地ignored目录。捕获是带instrumentation的请求，不作普通wall分数；不会安装/替换App、下载模型、修改默认路由或清理系统缓存。

[qwen_ffn_fixture.py](qwen_ffn_fixture.py)严格读取原checkpoint选定BF16 tensor与其capture，导出FP16数组、原BF16 compiled MLX replica输出及hash。BF16→FP16是新的近似表示，不自动继承原模型质量；Replica也不替代完整请求oracle。生成fixture后核心[tiling lab](../labs/ane.md)无需读取产品或兄弟源码。

本轮结果：[真实Qwen FFN证据](../review/qwen-ffn-tiling-20261015.md)。GPU捕获成功不等于新候选通过；数值通过也不等于消费者更快或请求画质合格。

## 有界attention原配方replay

[capture_qwen_attention.py](capture_qwen_attention.py)只编译[独立小客户端](capture_qwen_attention.cpp)，显式复用与当前source/MLX assets匹配的已有public native library。不会自动重建全项目或替换App；不匹配时拒绝，不能用旧ABI勉强运行。

```sh
.venv/bin/python -B techniques/integrations/capture_qwen_attention.py \
  --native-dir "$TC_NATIVE_DIR" --model-root "$TC_QWEN_ROOT" --seed 42 \
  --output-dir outputs/my-new-attention-capture
```

ordinary GPU observer在actual完整step0、step1 layer0提供借用参数、真实prefix cache和原callable输出。客户端按原公式重放、暴露额外QKV/O，核对完整block/prefix/输入存储后才写BF16副本；它不是未改变compiled kernel内部的直接capture。两phase合计128MiB cap，停在VAE decode/export之前；普通路线可能预加载已有VAE权重。source、loaded runtime、文件hash与统计生成前后复核，私有/实验环境不继承到子进程。

核心benchmark只需要这些显式raw文件，不需要产品或兄弟源码；[全部结果](../review/qwen-attention-20261016.md)与[步骤](../labs/gpu.md)包含独立seed确认、noop和empty-row拒绝。原始QKV与prompt/session本地数据不入库。

## 可选 ANEForge 上游执行

[aneforge_probe.py](aneforge_probe.py)只接受仓库ignored `outputs/` 下调用方显式准备的源码副本。需要预先核对[固定source与安全边界](../sources/repos/aneforge-rewrite-search.md)；它不会 clone、install、auto-build、运行上游pytest/conftest、下载模型或修改已有checkout。Python3.11/NumPy可运行CPU合同；device模式另需macOS/CLT、MLX与预构建shim。

先设 `TC_ANEFORGE_CHECKOUT` 为审核过的固定checkout路径，再使用新的任务目录；不把下列build命令用于原checkout：

```sh
mkdir outputs/my-new-aneforge
mkdir outputs/my-new-aneforge/source
cp -R "$TC_ANEFORGE_CHECKOUT/aneforge" outputs/my-new-aneforge/source/aneforge
cp "$TC_ANEFORGE_CHECKOUT/LICENSE" "$TC_ANEFORGE_CHECKOUT/NOTICE" "$TC_ANEFORGE_CHECKOUT/AGENTS.md" outputs/my-new-aneforge/source/
sh outputs/my-new-aneforge/source/aneforge/_lib/build.sh
python3 -B techniques/scripts/run_experiment.py --timeout 120 \
  --output-dir outputs/my-new-aneforge/cpu-launch -- .venv/bin/python -B \
  techniques/integrations/aneforge_probe.py --source-root outputs/my-new-aneforge/source \
  --mode cpu --output-dir outputs/my-new-aneforge/cpu
python3 -B techniques/scripts/run_experiment.py --timeout 120 \
  --output-dir outputs/my-new-aneforge/ffn-launch -- .venv/bin/python -B \
  techniques/integrations/aneforge_probe.py --source-root outputs/my-new-aneforge/source \
  --mode device --case ffn --candidate exp-silu --shape 128 512 1024 --seed 29 \
  --gpu-output-conversion compiled --gpu-compute fp32 --output-dir outputs/my-new-aneforge/ffn
```

脚本为自己子进程指定dylib/cache与禁止自动构建，保留breaker；一次失败后不能立即换新进程绕过15s退避。不启用profiling、fork-safety例外、experimental async、多op共享或系统权限变更。只测合成有界输入；完整GPU/ANE输入copy、同步与FP32输出恢复计入host-fed wall。borrowed view只在owner活着时测试，不做真正UAF。

`matmul --candidate int8`与`reduction --candidate reduce-matmul`可复核其他图；`activation/projection-silu --candidate exp-silu`用于保持独立oracle的手动子图缩减，不是假称自动最小化工具。CPU控制流mock与设备数值/性能分开。当前[15份记录](../review/aneforge-20261016.md)中的所有失败、强GPU对照和独立seed都保留；不改变产品默认。核心的[CPU学习实现](../labs/search_contracts.py)完全不需要该上游副本。

## 真实Qwen private-prefix完整FFN消费者

[aneforge_qwen_ffn.py](aneforge_qwen_ffn.py)读取明确的[原BF16 checkpoint/捕获/fixture](qwen_ffn_fixture.py)，不以FP16文件倒推原始权重。它复用已审核的ANEForge隔离copy/shim，只发射小比例constant prefix图；完整GPU基线、serial/hybrid和完整BF16输出恢复均保留，不修改产品/App。

```sh
python3 -B techniques/scripts/run_experiment.py --timeout 180 \
  --output-dir outputs/my-new-real-private-launch -- .venv/bin/python -B \
  techniques/integrations/aneforge_qwen_ffn.py --source-root "$TC_ANEFORGE_COPY" \
  --fixture "$TC_REAL_FIXTURE" --checkpoint "$TC_QWEN_CHECKPOINT" --capture-root "$TC_FFN_CAPTURE" \
  --point layer-0-step-19 --share 2048 --repeats 24 --output-dir outputs/my-new-real-private
```

显式numpy/MLX依赖与upstream副本仅属于此可选集成，不是techniques核心必需品。先核对512MiB admission与常量图存储余量；保留失败breaker、owned进程timeout与全样本，不清系统cache/旧实验。raw checkpoint、激活、时间戳/机器路径不发布。

边界是ready原BF16 GPU input→ready完整BF16 GPU output：GPU→host FP16、Model feed/execute/read/cast、GPU remainder、restore/join均计入，CPU验证在边界外。不声明零拷贝/physical overlap；全输出对BF16 replica与独立full/prefix FP64样本分别过固定门槛。API线程重叠不是物理counter。本轮[所有比较与held-out确认](../review/qwen-private-ffn-20261016.md)有局部warm收益，但完整请求/LoRA/内存/冷启动等资格仍缺。

## 真实Qwen实际shared端口与五臂完整消费者

[qwen_shared_ffn.py](qwen_shared_ffn.py)与[独立native](qwen_shared_ffn.mm)、[版本检查ABI](qwen_shared_abi.h)、[完整compute](qwen_shared_compute.h)不链接产品库或导入ANEForge实现；需要已准备的真实fixture/checkpoint/capture、明确成功的旧private-prefix run之MIL/blob，以及已有NumPy/MLX和MLX C++ headers/lib。原BF16 source、native paired-prefix、完整GPU remainder和BF16 output均严格保留；不下载模型、不安装或重建依赖/产品。

默认只准备显式owned资产；`--device`才编译/执行，`--mode inspect`只compile/load读实际port，不推理。第一次新建graph用[严格clone](../sources/official/clonefile.md)，需256MiB free且不fallback普通copy。完整复现示例：

```sh
python3 -B techniques/scripts/run_experiment.py --timeout 300 \
  --output-dir outputs/my-new-qwen-shared-launch -- .venv/bin/python -B \
  techniques/integrations/qwen_shared_ffn.py --device --mode benchmark \
  --fixture "$TC_REAL_FIXTURE" --checkpoint "$TC_QWEN_CHECKPOINT" --capture-root "$TC_FFN_CAPTURE" \
  --prior-run "$TC_QUALIFIED_PRIVATE_RUN" --mlx-root "$TC_INSTALLED_MLX" \
  --point layer-0-step-19 --point layer-0-step-10 --repeats 30 --output-dir outputs/my-new-qwen-shared
```

已有完整验收的本脚本run可显式加`--prepared-run outputs/my-new-qwen-shared`、使用另一个全新output-dir，走只读prepared/load流程：不新建常量clone、不调用compile selector、不fallback编译，128MiB门槛；必须明确`--mode benchmark`，不能隐式把inspect变成模型执行。load仍可有大量内部工作，本轮650.99ms和32.11ms都保留，不叫“免费setup”。

[全部560次](../review/qwen-shared-ffn-20261016.md)包括同client五臂copy/shared、两个实际point反序重复与264个完整actual-part exact join。当前约1.15×是本layer0 warm完整FFN相对compiled GPU的组件观察，不是整请求/全层/LoRA或物理ANE并行资格。原件和per-run源码不改，未执行的存储拒绝也保留；当前CLI额外参数guard单测不为旧run回填源码身份。

## 独立工作负载与完整ordinary block重放

[capture_qwen_holdout.py](capture_qwen_holdout.py)要求seed和prompt均不同于明确的baseline capture，复用同一source-qualified runtime。它仍使用原9张量hook与267MiB admission；本轮在启动native之前拒绝，没有新bulk capture或参考图片。

[qwen_block_shared.py](qwen_block_shared.py)与[小型native客户端](qwen_block_shared.mm)复用显式成功的private constant bank，在ordinary GPU request的step1/layer0 observer内运行五臂完整block。需要已有产品library、原模型、baseline capture和installed MLX headers/lib；不会重建产品、导入ANEForge实现或下载依赖。核心检索/CPU合同不依赖这些资产。

```sh
.venv/bin/python -B techniques/integrations/qwen_block_shared.py --help
```

执行必须显式传`--device --native-dir ... --model-root ... --bank-run ... --baseline-capture ... --output-dir outputs/NEW`，输出目录必须全新。默认owned TMPDIR；`--temp-context inherited`是显式上下文对照，不是已证明修复。不同存储模式各自admission和后置128MiB reserve保持不变，不能为让实验通过而降低门槛或删旧数据。

borrowed observer参数/原callable不逃逸，只有owned CPU BF16副本允许导出；受控停在step1/layer0，不进行VAE decode/image export，候选不回灌请求。`--lossless-archive`保留完整BF16字节，生成本地忽略的`.tcbf16z`；仅合成fixture完成独立decode测试，真实模型archive写出本轮被reserve拒绝。[全部失败与部分执行证据](../review/qwen-block-context-20261016.md)不作为合格性能或完整请求结果。
