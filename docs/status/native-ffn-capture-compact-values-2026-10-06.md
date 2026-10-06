# 真实 FFN 回放与紧凑数值舍入：单点通过，整模型仍未达标

2026-10-06，Asia/Singapore，M4 Max 64GB。接续
[共享 GPU/runtime 边界](native-z-runtime-boundary-controls-2026-10-06.md)。
完整双后端、Z/Qwen四格 base≥1.2×、LoRA加速与正式媒体/内存/device
trace 目标仍 active。本轮没有提升默认、放宽 N1 或宣称整体加速成功。

## 原始 BF16 输入，不借用 eager/F16 capture

新增 experimental-only、默认关闭的 bounded dense runtime capture：

```sh
TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_DIR=<fresh-directory>
TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_BLOCK=2
TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_LIMIT=8
```

block明确指定0..31，limit为1..32、默认1，整个process累计封顶而非每
次请求重置。只允许resident dense base 的 GPU split control 或实际
runtime请求；GGUF/ConvRot/LoRA、ordinary build和已有目标拒绝。
保留原BF16、shape和物理padded rows，finite检查后先写partial再rename；
metadata包含recipe/block/sample/runtime build。默认不复制任何输入。
capture不是性能配置，也不是自动calibration。原eager/F16 capture不变。

`native-z-ffn-capture-v1-private`同库，fox/seed42/8步/512：ordinary GPU、
GPU-only split+capture、Private FP16+capture三次串行生成。两个capture
目录各8份BF16 `[1,1056,3840]`、序号0..7、共约62MiB；没有lossy F16
中转。GPU control与ordinary GPU全部latent/PNG逐位一致；Private FP16
仍原relL2=0.0934619628、cosine=0.9956320612，capture没有掩盖误差。
目录 `outputs/native-z-ffn-capture-v1-z512-20261006/` 保留完整证据。

## 实权重、同输入的 FFN 回放

新 `build_z_image_runtime_ffn_error_probe.sh` 构建独立工具。它按capture
block读取原checkpoint的三片BF16矩阵，使用native GPU kernel与实际
Private FP16+zero-correction hidden ABI，对最后352行做冻结输入对照。
绑定checkpoint/capture前后hash及capture/replay build；不是整模型验证。
GPU software FP16模型明确逐stage舍入，不冒称物理ANE内部算术。

block2=layers.0，GPU trajectory的8个实际输入，原K1024/N512：

| 分离项 | 8步范围/事实 |
| --- | --- |
| GPU tail与完整GPU对应行 | 每步relL2=0 |
| manual BF16 hidden与fused GPU | relL2约1.0e-8..8.7e-8 |
| software FP16 gate/up再BF16 | relL2约0.00155..0.00159 |
| native hidden+GPU down | relL2约0.00272..0.00361 |
| native完整FFN output | relL2约0.00281..0.00351 |
| software FP16 projections+BF16激活+GPU down | relL2约0.00183..0.00256 |

回放headroom依次4/4/16/16/16/16/64/64，3次overflow recovery；它是
单层冻结回放的实际scale轨迹，不是假装整个32层生成的scale轨迹。
gate max166..548，up max149..528。结果支持继续调查中间激活/舍入，
不能独立宣称唯一根因或以局部FFN误差替代最后latent门槛。
证据 `outputs/native-z-ffn-replay-v1-gpu-capture-k1024-20261006/receipt.json`。

## 新舍入算法与独立实际驱动 oracle

原185个MIL算子的linear-bin数值BF16-RNE emitter保持不变。
新增compact emitter：五级精确二进制幂normalization、显式floor/
fraction/整数奇偶ties-to-even、逆序restore，45个算子；没有近似log2
决定边界。45/185是host生成的算子数，不是吞吐或整模型倍率。

先尝试normalized值+8再-8的magic-add捷径。实际driver在63,480个
finite-half encoding上有54,776个数值错误、另1个负零符号差异；失败
回执保留，未用于运行配方。这与没有保留所需中间舍入一致，不推断
实际硬件accumulation dtype或所有compiler优化的唯一机制。

compact-floor的16次实际driver调用覆盖全部63,480个可由FP16承载的
BF16-RNE有限编码：numeric mismatch=0，raw-bit mismatch=1，唯一
`-0`→`+0`。另8个BF16-rounded值超FP16 carrier的encoding明确排除，
运行图必须另行guard。旧strict-bit模式仍返回失败；新的numeric模式
明确canonical-zero政策，不是修改旧oracle。FP16 cast正控制256个
样本零错误；两种native BF16 cast仍被该tuple编译器拒绝。

回执：`outputs/ane-bf16-compact-{magic-v1,floor-v2}-20261006.json`。
小点通过不证明大图composition、FP32 ANE arithmetic、原生INT8 MAC
或物理GPU/ANE overlap。

## 默认关闭的 Private FP16 实验配方

```sh
TURBOCIDER_ANE_BACKEND=private
TURBOCIDER_ALLOW_PRIVATE_ANE=1
TURBOCIDER_PRIVATE_ANE_DATA_PATH=fp16
TURBOCIDER_PRIVATE_ANE_GPU_IO=1
TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES=1
```

只接受experimental build、明确Private、base-only SwiGLU、原dense
BF16 weights及BF16 input/output。不接LoRA、Public/auto fallback、
MatMul或W8 path；默认/public/frozen/既有LoRA算术不改。
实际receipt recipe为
`fp16-swiglu-compact-bf16-values-canonical-zero-guarded-v1`，configuration
identity包含开关，MIL/constants与runtime build隔离旧cache。
gate/up/sigmoid/SiLU/hidden/down六个数值边界使用compact emitter，
注意显式sigmoid边界，而非只round最后SiLU。

每边界先检测finite/carrier范围，unsafe token附加不可由物理FP16
承载的输出sentinel；原IO finite validator在publish前拒绝，原up-only
headroom recovery或完整operation GPU重算保持。不裁剪/隐藏非有限，
两种host/device launch和原dense source类型均有guard。新workspace在
estimate计入16个F16 slice+64MiB allowance，非RAM硬上限/物理scratch。
quality工具 `--fp16-bf16-values` 要求双execution receipts和实际对应
base-only Private row recipe，不接受默认policy、假recipe或GPU decline。

v1组合图因guard与round helper重复SSA变量名，实际编译失败；两尺寸
生成完整GPU回退、runtime_failed=true/calls0，quality工具正确拒绝。
原失败目录 `outputs/native-z-fp16-values-v1-z{512,1024}-base-20261006/`
保留。修复变量namespace后，小完整图compile/execute通过；增加host
symbol唯一性断言，不能只用文本生成成功证明可运行。
composition v1/v2回执保留；v2的去掉carrier-output消融仍被compiler拒绝，
不据此删除生产安全检查或认定其它未独立验证的唯一原因。
`build_private_ane_fp16_value_compile_probe.sh` 可重建该小图消融工具。
final回执 `outputs/ane-fp16-value-composition-final-20261006.json` 使用
显式清零input surfaces，结果与v2相同；这里只验证composition/capability，
不把零输入当非零数值或模型资格oracle。

## v2最终生成：不能推荐这个配方

同一新Private v2 CLI/相邻library，fox/seed42/8步，独立GPU/runtime
进程、512/1024串行，原352-row/K1024/N512模板、chunks1/fixed-async1。
没有与owned build/test重叠。每候选实际Private calls259、retries3、
headroom64、rows90112，零runtime failure/fallback，recipe/GPU IO匹配。
slot262963200、estimate811597824 bytes；不是process-footprint资格。

| 尺寸 | 旧FP16 relL2 / cosine | compact values relL2 / cosine | 原N1 |
| --- | --- | --- | --- |
| 512 | 0.0934619628 / 0.9956320612 | 0.1353334216 / 0.9908479484 | fail，变差 |
| 1024 | 0.0779761958 / 0.9969570683 | 0.0684830628 / 0.9976525138 | fail，仍超标 |

原relL2≤0.03、cosine≥0.999不变。普通GPU输出与此前对应GPU dump
一致，所有`qualification_passed=false`。diagnostic denoise约21.85/
32.74s，含dump/观测、无hot/reverse/load资格；不计算正式倍率，也不
把45算子局部简化宣称为≥1.2×。此配方不适合作默认推荐。
完整证据 `outputs/native-z-fp16-values-v2-z{512,1024}-base-20261006/`。

## 构建、回归与下一项因果检查

Private v2完整native-only、Public v2 ordinary library-only build成功，
各479个source inputs独立匹配current native tree；包含原用户dirty草稿，
不是staged-only制品。Private library SHA256
`ffa96553d504285c0fc8d5dd00a108660af4fe695821998fa2b219e131ee8571`，
build ID `tc-runtime-build-v1-ece027a77a1ee5ea4407c90aee012191a8fdfcc50b45eba51c953f1cbdc132c7`；
Public SHA256 `30e26506a794b783827a73ce4110936f5e02dd959e0e4c87b02f1aa31b1b4470`，
ID `tc-runtime-build-v1-efec9fefaab2723a66433094446a3a4908c7d45473533392b9b8d18d5e02af9c`。
Public library未发现四种private client/request/event/surface strings。

121项host/tool/capture初始化/source guards通过；15项Public graph/
CoreML/MLX/receipt及ordinary-capture/control gate通过；2项实际Private
channel/MLX校准、原LoRA/ONE down、F32 partial及late-failure重算回归
通过。不是全仓/Swift/全部设备资格。

下一个可区分的假设是当前row尾部offload包含caption tokens：两分辨率
offload352行中都包含相同caption尾部，而图像占比分别明显不同，最终
误差却都较大。应做same-library、固定同rows/share、仅交换row placement
的对照，让caption FFN保留GPU，再看完整trajectory；这目前只是待验证
的假设，未接入/未宣称能通过。不能靠改阈值、减少steps或更换有利prompt
达成原四格目标。本轮未重试已被外部ComfyUI污染的正式速度矩阵。
