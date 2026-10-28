# 实际共享 backing：证明外部交接，不泛称系统零拷贝

<!-- knowledge
id: technique-gpu-ane-shared-surface
type: method
stage: ffn
operator: layout
backend: metal, mlx, private-ane
device: gpu, ane
symptom: wrong-layout, lifetime, stale-cache
semantics: bitwise, numerically-equivalent
representation: fp16, bf16, iosurface
boundary: gpu-ane, producer-consumer
api: public, private
-->

[总目录](../README.md) · [共享内存合同](shared-memory.md) · [完整FFN边界](private-ffn-boundary.md) · [本机全部记录](../gpu_ane/case-studies/surface-strides-and-stale-output.md)

## 1. 要解决的接缝

GPU→host数组→private模型→host数组→GPU即使处于统一内存，也有显式feature copy。[真实Qwen copy-bridge](../gpu_ane/case-studies/qwen-private-ffn-copy-bridge.md)已把这些成本计入，但没有证明shared backing。本页先用原创有界图建立可以迁移的**外部接口证据**，不以小图代替完整FFN。

[官方API/SDK来源](../sources/official/shared-memory.md)说明Metal no-copy包装的页对齐、单VM region和生命周期要求；private `_ANE*` 则另做运行时selector/ABI检查。两者是不同稳定性等级，不能把公共Metal合同扩张为Apple对私有ANE入口的保证。

## 2. 五项证据，必须同时成立

1. 从实际`NetworkStatusList`读取唯一input/output的dtype、NCHW、row/plane/batch字节stride；未知offset/depth/layout拒绝，不猜紧密布局。
2. 每个owned IOSurface保留至少一页尾guard，allocation/page对齐检查后用`newBufferWithBytesNoCopy`包装。失败即拒绝，不偷偷换成copy。
3. 从实际`_ANERequest.inputArray/outputArray`取`ioSurface`，比较输入/输出与目标surface的IOSurfaceID；再比较Metal `contents`与实际返回surface的base address。初始、每call前后、最终都检查，只发布相等bool，不发布ID/地址。
4. GPU producer逐元素写入代际相关输入；private同步evaluate后，GPU consumer从实际output mapping读完整结果。所有逻辑input/ANE output/consumer输出都有独立标准库全量hash，不能仅相信alias标记或finite。
5. producer与consumer的owned command都成功完成，验证结束且unload成功后才释放。evaluate失败/未知GPU完成/unload失败时不复用mapping，保留到child退出；不做真正UAF。

`external_feature_copy_count=0`只适用于所记录执行边界内的shared arm。计时外初始化/poison/CPU数值检查仍访问内存；driver、编译图和runtime内部copy行为始终`unknown`。相同surface和正确GPU消费者**不是**物理ANE放置、跨设备并发、INT8 MAC或整个系统零拷贝证据。

## 3. Padding与有限错误负控

本机2026-10-16、M4 Max/macOS26.6.2/SDK26.5的两个明确布局：

| 逻辑FP16 NCHW | 紧密plane bytes | 实际row/plane bytes | 实际batch bytes |
| --- | ---: | ---: | ---: |
| `[1,32,1,33]` | 66 | 128 | 4096 |
| `[1,64,1,65]` | 130 | 192 | 12288 |

不能从一个布局外推固定256-byte对齐或所有private图的NCHW映射。这里H=1，row和plane相同；不证明多行/多batch/非零offset可用。

故意让GPU consumer按紧密plane读取，所有值仍finite，却分别产生1011/1056、3990–3991/4160个错误。另直接消费上一代实际输出，与下一代期望比对，全部元素拒绝。两项负控都不进入正式计时/ratio；它们说明“finite”“API返回成功”“拥有一个指针”都不足以证明交接正确。

## 4. 相同边界的copy控制，不偷换GPU分母

shared与copy均测**GPU producer encode→GPU consumer完成**，编译/load、pipeline prepare、poison和验证在外。copy arm GPU先写独立shared Metal buffer，再CPU copy到input surface，evaluate后CPU copy到另一Metal buffer供同一consumer；两次copy及lock/unlock计入wall。两arm同generation，顺序交替；保留first1/warmup2/全部measured8或24组。

小图copy量只有8KiB/24KiB，移除copy不保证主导延迟：两次Metal提交/完成和private调用仍存在。[全部实测](../gpu_ane/case-studies/surface-strides-and-stale-output.md)没有稳定性能晋升，也没测optimized GPU-only分母，所以不能写“GPU+ANE快了多少”。host事件只表示同步边界，不是GPU/ANE硬件时钟。

## 5. MLX实际adoption与generation：Copy不等于快照

[固定源码导读](../sources/repos/mlx-external-memory.md)与[四份完整协议/首轮失败](../metal/case-studies/external-memory-adoption-and-fresh-views.md)继续验证MLX端。两个seed下，raw-pointer constructor和strict `make_buffer`均实际采用整块owned region；padded view与8MiB FP16逻辑输入、完整新consumer、尾guard、view存活和最后deleter得到独立全量见证。8MiB case明确是合成geometry，不是Qwen激活或FFN性能。

这里有两个容易混淆的状态：`mx::copy`在固定0.32.0实现仍`copy_shared_buffer`，新consumer可看到外部新generation；已求值的旧consumer不会自动失效。另以实际不同backing的FP16→FP32 materialization建立快照控制；读取它仍得到旧值。不能用“新array对象”“copy名字”或“又调用eval”替代独立物化和generation检查。

所测MLX导入buffer为untracked hazard，实验显式排空MLX→外部Metal producer完成→构造全新MLX consumer并排空；没有测试无同步竞争或宣称自动跨queue coherence。只导入page合法的完整region，再从contiguous flat创建已核对stride的view；直接导入短logical span、任意CPU指针塞Buffer或对非连续输入flatten后仍假定alias都不成立为通用方案。

## 实现、复现与下一步

CPU合同/独立全量witness、原创native客户端、显式CLI、、离线反例均在本目录，无产品/兄弟源码依赖。默认仅准备合成MIL，`--device`才实际执行；不联网、安装、下载模型、申请PMU或降低系统保护。

接真实Qwen仍须重新读取真实二维MIL的physical映射，GPU pack/cast、restore/join都计入完整BF16 FFN边界。这里的`_ANEClient`与此前e5rt入口不同，须重新过full/prefix source与独立FP64门槛，不能直接继承约1.12×的copy-bridge结果。更多层/LoRA/请求质量、冷启动/内存/长时、失败/取消和物理overlap继续独立验证。

上述MLX合成adoption已在2026-10-16实际验证；它本身不能把两个独立probe拼成“完整Qwen已shared”。后续[真实FFN实验](../gpu_ane/case-studies/qwen-shared-ffn-consumer-cost.md)另对实际二维port、GPU pack、private prefix、MLX restore/join完成了完整source/alias/数值验证，两种证据保持独立。合成probe默认仅准备源码，显式设备复现需要已有MLX C++ headers/lib assets：

新独立witness/native//反例测试均随本目录分发，不安装或重建MLX。

重开条件：OS/SDK/private ABI、shape/dtype/stride、owner/generation、实际MLX或模型消费者改变。当前资格只属于上述有界合成接缝，不是产品默认。
