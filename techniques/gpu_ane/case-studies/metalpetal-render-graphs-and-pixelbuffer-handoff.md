# MetalPetal：渲染图融合与真实PixelBuffer交接

<!-- knowledge
id: technique-gpu-ane-case-studies-metalpetal-render-graphs-and-pixelbuffer-handoff
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [本轮摘要](../../MAINTENANCE.md) · [来源/行范围](../sources.md#metalpetal-fusion) · 完整数据 · 预声明计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17。本轮实际编译MetalPetal的完整Objective-C核心，执行原始filter、原始嵌入MSL和IOSurface-backed CVPixelBuffer输入/输出。它是GPU图像处理框架，不是AI推理或ANE引擎；研究价值在于模型前后处理也必须遵守的中间材质、颜色、完成门和外部buffer所有权合同。

最重要的结果不是速度：同一亮度`+0.25 → −0.25`图，BGRA8中间纹理的未融合路径会裁剪，开启图优化后的合并矩阵则为identity；白色从191变成255，最大差64码值。两条路径各自完整匹配对应的materialization-aware CPU参考，并不表示优化前后等价。

## 1. 来源、实际构建与独立fork身份

固定原仓库`MetalPetal/MetalPetal`的commit `f9b78897bd4214bb097f352a1bde0a4f4a1e2ddb`、tree `4a983da6590b81b548cbe2c7f62e893c9b49ecc1`，commit日期2023-02-21。完整584个Git blob（含symlink目标文本）在准备和数据整理时核验，23个选定文件的实际阅读范围另存[来源](../sources.md#metalpetal-fusion)，不是全仓库人工审计。许可为MIT；原文件/归档/二进制仅留ignored `outputs/gpu-ane-metalpetal-round34-20261017`，不随Wiki分发。

[原Package.swift](https://github.com/MetalPetal/MetalPetal/blob/f9b78897bd4214bb097f352a1bde0a4f4a1e2ddb/Package.swift)声明无外部package依赖。本库CMake直接构建原`Sources/MetalPetalObjectiveC`的`.m/.mm`，用原`MTISwiftPMBuiltinLibrarySupport.mm`注册的MSL source library，再由原`MTILibrarySource`/context在运行时创建Metal library。不是自写替代shader，也不要求离线metal编译器。原核心保持C++14；自有Objective-C++ probe因使用filesystem单独设C++17。没有执行Swift overlay、SwiftPM package tests、示例视频App或codec。

另核[issue #400](https://github.com/MetalPetal/MetalPetal/issues/400)与独立`aldealabs/raster`的`master`元数据：固定commit `cdba5d0b44fb4ffa851e45e50ee7939cb6c62fce`、tree `e8cb1e93ed73fb2c9278ea30336b0e4ce201b03b`，日期2026-08-15。issue作者`rdev`的association为`NONE`；其独立延续、1.26兼容线/2.x新线等表述是作者声明，不是原维护方移交或背书。本轮没有阅读/构建fork实现，不能据此授予HDR、兼容性或性能资格。最初请求`main`返回HTTP422，随后`master`成功；错误分支URL不等于项目不存在。

## 2. 从recipe到texture：哪些优化改变实际执行

原README把`MTIImage`描述为不可变recipe；mutable filter的`outputImage`生成依赖图，context在render时解析。不要把recipe不可变误读成外部CVPixelBuffer内容已经复制成不可变快照。原`MTICVPixelBufferPromise`持有输入buffer，但retain不阻止其他owner提前覆盖同一帧。

原默认`enablesRenderGraphOptimization=NO`。本轮分别显式OFF/ON。`MTIRenderGraphOptimization.m`只在中间结果transient、依赖唯一等条件满足时尝试改写；`MTIRenderPipelineKernel.m:582–616`的color-matrix合并还检查kernel、输出descriptor、geometry/viewport等条件，再concat矩阵。persistent中间结果或额外消费者会改变可改写性；本轮没有对所有gate做动态sweep。

本轮实际原shader的核心表达式是采样值乘矩阵再加bias；filter未指定输出format时走context working format。即使两个节点在实数域可合并，节点之间的UNorm写回并非实数identity。CPU原optimizer观察到两个filter加source的promise数`3 → 2`，identity仍`1 → 1`；这是软件图观察，不是hardware drawcall/encoder trace，更不是已测速度或带宽收益。

原`MTIImageRenderingContext.mm`按promise/cache状态解析，persistent结果保留，transient结果在最后依赖消费后释放其render-target引用。原`MTITexturePool.mm`有texture validity/refcount；普通pool按descriptor复用，heap pool还按size/resourceOptions寻找可复用heap、使用tracked hazard并在纹理归还时makeAliasable。复用不承诺清零，不得读取未写区域。`idleResourceCount/Size`只是当时闲置池快照，不是峰值分配、进程RSS或节省内存实测。

## 3. 主矩阵和全像素结果

设备为Apple M4 Max。输入/输出均由本库拥有的真实IOSurface-backed BGRA8 CVPixelBuffer构造，alpha全部255，输入/输出`sRGB:NO`；这定义的是码值算术，不是通用物理线性光或HDR流程。尺寸17×13与65×31、seed17/29；先identity验证通道/方向，再跑原invert两次及原brightness两次。

主矩阵为`2 formats × 2 optimizer settings × 2 sizes × 2 seeds × 3 graphs = 48 renders`，共53664逻辑像素/214656个BGRA字节。独立新进程再48份，全部直接byte equality；pilot另3份/663像素/2652字节，不混入主数。所有实际输出完整匹配独立CPU参考，最大码值误差0；参考不用原matrix concat helper。

| 图/中间format | 优化OFF | 优化ON | 本轮观察 |
| --- | --- | --- | --- |
| identity，两个format | 原输入 | 原输入 | 全字节相同，作为layout/方向控制 |
| invert两次，两个format | 原输入 | 合并后原输入 | 全字节相同，非普遍融合正确性证明 |
| brightness两次，BGRA8Unorm | RGB为`min(input_code,191)` | RGB为`input_code` | 每组最大差64；白色191→255 |
| brightness两次，RGBA16Float | 原输入 | 原输入 | 仅这些输入和最终BGRA8输出下相同 |

以单通道码值`c`理解BGRA8负例：第一pass计算`c/255 + 0.25`，写UNorm时量化并把超1值裁剪；第二pass再减0.25，已经丢失的高亮不能恢复。合并矩阵的净bias为0，中间UNorm写回消失，结果直接保持输入。独立参考逐pass执行F32算术、对应UNorm/F16 materialization，最后转换BGRA8；不是只在实数域验`(+a) + (−a) = 0`。

四组BGRA8 brightness输入的OFF/ON不同字节位置分别是174、174、1520、1499（顺序17×13 seed17/29、65×31 seed17/29），alpha不变；不是不同像素数。RGBA16Float可以保留本组超过1的中间值，最终量化后回到输入，但不证明任意半精度图、非opaque alpha、非整数bias、HDR或不同采样下无误差。也不声称所有fusion都有错：这里证实的是这组明确materialization合同不能仅凭矩阵可合并就判语义保持。

## 4. scheduled不是completed，buffer也不是自动快照

原`MTIContext+Rendering.m:90–110`的`renderImage:toCVPixelBuffer:`返回BOOL，只检查创建task是否非nil。实际startTask路径提交command buffer后调用`waitUntilScheduled`，而非等待GPU完成。原`MTIRenderTask.m`另提供`waitUntilCompleted`及status/error。不能仅凭BOOL YES就让CPU读取/复写buffer或把它交还外部帧池。

本轮probe采用的所有权顺序是：初始化并保持source不变 → 保持MTIImage/source/destination → 原startTask → 显式`waitUntilCompleted` → 检查status=4且error=nil → lock/read逻辑行 → release。原始record均满足此完成门。source初始化的全部字节（含自有padding）前后未变；output只按实际row pitch读取每行`width×4`逻辑字节，绝不导出padding。实际pitch不是简单的`width×4`假设，完整值保存在每份record。

实际bridge class为`MTICVMetalIOSurfaceBridge`，原bridge通过texture descriptor/IOSurface API创建texture；这补强了真实CV→Metal→CV路径身份，但不证明底层物理zero-copy、跨API backing adoption、无隐藏copy或并发consumer安全。所选原promise对非IOSurface输入另有CI fallback，本轮未运行；也未运行NV12/YUV、不同alpha/sRGB、Core ML消费或异步frame reuse。

我们没有提前读buffer、故意早释放、注入GPU错误或制造并发覆盖。因此“原接口只等scheduled”是源码事实及调用者合同，不是已复现的上游race或failure/drain缺陷。不要用正常完成的一次owner链替代取消/超时/故障退役证明。

## 5. 准入、失败保留和有界收尾

pilot/primary/confirm均经guard准入，连续五次一秒间隔占用低于5%，15次观测全0%；各实际GPU/native子进程terminal0。粗粒度preflight仍不排除执行期间短时外部活动。本轮不采性能排名，不拿supervisor elapsed或promise数当推理延迟。

三类失败保留在owned receipt，不覆盖成成功：自有CMake初次漏link系统framework导致return1，补MetalPerformanceShaders/SpriteKit/VideoToolbox后核心链接成功；自有probe在C++14下用filesystem导致return2，只改probe为17后成功；全部GPU和独立确认完成后，首次证据capture在末尾磁盘reserve检查return1，像素检查此前已全部完成。

最后一次失败时可用空间低于原定10GiB大动作储备。保留原计划及失败capture，另在计划预声明metadata-only收尾：新增curated JSON总量≤2MiB、隔离CPU snapshot≤64MiB、剩余空间≥2GiB，不再GPU/build，不清用户文件或全局cache；原10GiB门不降低。成功capture记录available=9256747008字节，明确`large_action_reserve_currently_satisfied=false`。CPU/文档检查通过不表示该reserve恢复，更不授权继续设备实验。

## 6. 对异构推理设计可复用的启示

- 图融合资格必须包含中间format、量化/裁剪、alpha、颜色transfer与采样语义；“代数可合并”和“硬件command更少”不够。若消费者依赖8-bit中间结果，应明确保留materialization或重新定义/验证完整图合同。
- cache key/immutable recipe不能代替外部帧代际所有权。更换输入、frame pool reuse或并行消费者时，要同时检查谁保留存储、何时可复写，以及逻辑像素和padding的有效范围。
- 数据交接要绑定真实完成事件；scheduled、BOOL成功、idle pool计数分别只是提交、创建和局部资源状态证据。

这些是本轮源码和小图支持的设计约束，不是MetalPetal/Raster的普遍质量排名。没有运行模型、ANE、M5 TensorOps、视频编解码、PTS/EOF/flush、能耗、峰值内存、物理overlap或故障drain；25项可移植测试及隔离检查见[验证（过程原件已移出）](../../MAINTENANCE.md)。
