# Public W8A8 完整 FFN 与共享 GPU 桥接

2026-10-07，Asia/Singapore，M4 Max / macOS26.6.2。接续
[Public INT8 IO 正控制](public-int8-io-qdq-control-2026-10-07.md)。
本轮不止MatMul探针：新增完整Public W8A8 SwiGLU Executor，经共同
factory/HybridFfn接入实际Z-Image生成。完整目标仍active：尚未完成
Z/Qwen、512/1024、base/真实LoRA、GGUF/ConvRot全组合的matched GPU
加速与视觉/内存/trace验证，没有扩大单格证据的适用范围。

## 一个 GPU 实现，两种合法 executor

原Private里的Metal-only资源移到 `ane_gpu.hpp/.mm`：

- 同一Surface owner、FP16/I8 IOSurface、physical pitch/page accounting；
- 同一raw GGUF Q4_0/Q4_K/Q8_0/Q6_K、affine Q4/Q8、dense decoder；
- 同一Sylvester H128/H512和Comfy H256、RNE normalized scales/codes；
- 同一bounded26 pipelines、4MiB weak-generation scale cache、单seed表；
- 同一独立W/A staging queue/event、GPU upload与FP32 restore/finite guards。

MSL arithmetic未改，Private头保留compatibility include/type aliases。
Private client、MIL/cache/driver selectors仍在conditional private source，
公共模块没有private selector/framework loader。GPU timeout safety domain
与Private driver completion safety domain均检查，未取消超时后的禁用。
Private真实shared-event、各编码CPU oracle、Comfy/direct、4224-row/
多chunk/LoRA恢复4项hardware regression已通过。

Public可以直接将这些原IOSurfaces包装成CVPixelBuffer/MLMultiArray，
不建立第二个GPU stager或checkpoint FP16副本。新upload-only接口不
弱化既有transfer的nonempty-download约束。

## Public 的完整 normalized graph

`export_runtime_w8a8_ffn.py` 用隔离coremltools9.0生成macOS26模板。
x/wg/wu/wd是真正INT8 runtime features；tx/sg/su为FP16 scales。
headroom为动态FP16 scalar，不因overflow重新编译每层模型。图执行
gate/up、scale restore、可选LoRA gate/up correction、SiLU、固定模型
无关grouped Hadamard、hidden A8和down normalized projection。
输出packed y包括normalized down、hidden scale及可选corrected hidden。

仅Hadamard是矩阵const；没有checkpoint或adapter矩阵嵌入。两种basis
独立declared，Comfy direct W不再次旋转。初始Python sign生成写错了
seed/index组合，接入前对照native unsigned64 recipe修正；独立scalar
sign/H4-kron tests覆盖，导出及实际测试使用修正版本。

`PublicW8Graph` 是共同Executor接口，沿用FfnWeight、DeviceWeightRegion、
GraphShape和HybridFfn row scheduler；factory识别explicit W8模板并
选择Public。默认legacy FP16模板不改，Private仍要求授权。当前Public
W8模板是明确row/basis ABI，不静默接受Private channel-auto或group-A8
策略；自动按operation三选一路由尚未完成。

Public持有两套W bank、一套A8 slot；next-W prefetch有source身份和
双consumer/producer reuse fence。LoRA无weight merge，GPU低秩rank/
correction路径沿用原接口；hidden回原model dtype，FP32 base partial
只在GPU恢复，不声称ANE FP32 arithmetic。

## 同步与失败：没有隐藏的队列环

Public prediction在持久worker同步执行。leading GPU upload先signal
ready，public shared-event listener在CPU上等待ready，CoreML API完成
后CPU释放done，GPU epilogue再消费输出。不能在同一Metal queue的
done-wait之后再enqueue ready-wait，否则形成环。

Public成功时使用指定output backing；若CoreML返回另一个合法backing，
先bounded/scoped copy到owned y再释放done。失败/ObjC异常先设置sticky
CPU failure flag再释放GPU wait，不读取未计算scratch；whole-operation
失败由共用HybridFfn完整GPU重算。listener deadline、producer/consumer
owners和future bank draining保留。API/event ordering不是物理GPU/ANE
overlap或原生INT8 MAC证明。

模板使用既有ArtifactLease机制：全部compiled file digest验证后复制到
不可变私有snapshot，model/worker释放前保留lease。same-length篡改在
prediction前拒绝；这里“私有snapshot”指进程独占目录，不是私有API。

## 实际公共接口回归

隔离9.0下4项 `test_public_w8_ffn` 全通过，无skip：geometry/recipe
guards、dynamic-code/scales/headroom/LoRA graph结构、独立unsigned64
Sylvester与H4^4常量，以及真正GPU→CoreML→GPU executor integration。
integration包含：base/A/base、三chunk、F32 restore与原BF16 hidden
逐位边界、two-bank future A/B identity、lease/mismatch/discard、headroom
retry与复用、input/output alias、second/last-chunk nonfinite、短weight
拒绝、padding/guards和健康refill。public二进制无四种private class
strings；corrupt compiled file失败测试实际执行。

独立小bucket回执：
`outputs/public-w8-shared-executor-{selftest,integration}-m33-h128-f512-lora-v1-20261007.json`。
移植初次Private编译因const lambda中的release方法失败，标记const后
4 hardware tests通过；不是将失败编译计为通过。

工具适配保留严格backend/actual-call/IO计数/recipe绑定，允许明确
Public W8 GPU IO，不把它归类为Private ANE。legacy Public FP16不能
靠gpu label冒充压缩接口。81项相关host回归通过，包含新拒绝controls。
正式load/memory gates和历史N1阈值未改。
最终工具/identity/release相关host suite扩大为92项，全部通过，无skip；
Public W8 v2的4项suite再独立重编译/实际预测复跑通过。重复run不重复
计作不同test，也不把synthetic correction ABI测试冒充整模型真实LoRA。

## 第一格真实生成：Z512 base

同Public v1 CLI/dylib、GPU/runtime独立进程串行，原BF16 checkpoint，
fox/seed42、8步、resident、352-row模板、chunks1/fixed-async1。
未与owned build/test重叠；两个encoder均GPU，conditioning/initial
latent逐位一致。Public真正256次CoreML calls/device IO，retry0、
fallback0、failure0、headroom1，pipeline4、scale-cache96 entries。
不是GPU-only decline或只是成功编译。

完整latent N1：relL2 0.1617834966、cosine 0.9868842929，旧N1 fail保留。
RGB SSIM 0.9329545115，仅数值补充。目视原尺寸whole和三张同坐标detail
crops：狐狸姿态、脸部、颜色和整体构图很接近，但耳部/胸毛纹理、背景
虚化和右侧细枝有可见改变；未见明显新棋盘格/色块/肢体破裂。本记录
是单prompt/seed agent观察，不是用户审批、多样本视觉资格或发行通过。
自动visual manifest保持pending，qualification_passed=false。

诊断raw denoise：GPU7.612076s、Public6.589978s；request-wall分别
11.081505s/8.891485s。**不计算正式加速比**：带dumps、无hot repeats/
reverse/strict load/memory审计，cold model/OS cache状态未隔离。
GPU先运行、runtime后运行；不能将这一次差异当matched hot加速。

目录 `outputs/native-public-w8-z512-base-v1-20261007/` 保留所有请求、
PNG、8步dumps、execution/quality/visual receipts。最初候选误写
execution=hybrid，在schema处失败；原失败保留。request-v2使用正确
gpu_ane，generation-v2-receipt是实际成功运行，native schema未放宽。

## 构建身份与下一步

该生成使用Public v1 experimental/native-only build：486 source inputs
匹配，library SHA256
`1d96fc1d993dd4dee12505033816d3163de9a541359e2c117194d048f9c3298a`，
build ID `tc-runtime-build-v1-7a96e86534c5e952f1cec1d8959fbba57d206d2bcfd2af777199f6f4cbda8dd2`。
随后v2增加ObjC预测异常的failure/signal清理，不改变math；没有把v1
PNG重新标为v2制品。Private v2完整native-only构建成功、486 inputs
及manifest seal匹配，library SHA256
`290c8a1f07181eadea1b5c92fc7379d5e2a5e23b8fb3818846691d218a51df61`，
ID `tc-runtime-build-v1-010432fa7ae58d423bc2d163f6c896c3ad91ec704dac5cdc04b2dd6a93171b35`。
普通Public v2 library-only成功，486 inputs/seal匹配、actual release
guard通过；library SHA256
`f7cdd3598ade93a4a4559093e9095cc07163aae8f70500f8f51a01dce3fca911`，
ID `tc-runtime-build-v1-f5eddbdae7423259af304faf6373b51ecd6554b32cd51405baa1cbc81169f4d0`。
最终库13项legacy Public graph/CoreML/MLX integration、3项Private
prepared/channel/MLX、2项Private factory-on/off和missing-producer timeout
回归通过，无skip；先前4项Private hardware和4项新Public W8测试也
通过。不是完整make test/全部private hardware/Swift验收。

所有构建包含用户原ConvRot未提交草稿，但selective commits不夹带它们。
models/adapters/references/design只读。既有fastMathEnabled deprecated
warnings保留，没改fast math。尚未做全仓/Swift/所有Mac支持验收。

后续保持完整目标：补Public v2普通库/Private集成回归，实Z/Qwen
512/1024 base与真实LoRA、GGUF/ConvRot控制，再做multi-prompt/seed视觉
和matched GPU cold/hot/reverse/load/memory。根据实际完整traffic成本
选择row/channel/backend，仍需物理trace；不能仅凭256次call或本格
diagnostic时间将任何固定share升为默认profitable策略。
