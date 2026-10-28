# stable-diffusion.cpp：Metal参数驻留、分段执行与失败边界

<!-- knowledge
id: technique-gpu-ane-case-studies-sdcpp-metal-residency-and-segments
type: case-study
stage: pipeline
operator: layout, scheduling
backend: coreml, metal
device: gpu, ane
symptom: wrong-layout, lifetime
semantics: contract-only
representation: iosurface, pixelbuffer, fp16
boundary: gpu-ane, producer-consumer
api: public
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [引擎地图](../inference-engines.md) · [共享内存](../memory-and-handoff.md) · [来源](../sources.md#sdcpp-residency) · CPU执行

2026-10-17，固定`leejet/stable-diffusion.cpp@2f886889e6e8b78738d6b87f7191f6018557c551`，tree `03d8fe24e70b04824acd504c47567c593d4df889`。父提交指定的是fork `leejet/ggml@4bf5f6000653b7881d00963cd6ddb665ccd62a8d`，tree `5c1292254d8c5405ca1d5d16c352495f731bb6a0`。**本地GGML checkout为另一commit，不能拿它解释父项目**；本轮从固定Git对象读取GGML，25个选定文件/范围核blob及全文hash，不切换或修改共享checkout。

这是CPU/Metal/磁盘之间的模型阶段与驻留案例，不是GPU+ANE并行实现。实际执行仅原始CPU规划函数的合成预算观察，没有构建/运行整个引擎、读取模型、提交GPU或调用ANE；源码机制和本机资格分开。

## 1. 先分开“在哪算”和“权重放哪里”

`--backend`指定模块图执行入口，`--params-backend`指定参数驻留位置。`diffusion`、`te`、`vae`可不同；`disk`只是驻留模式，不是执行设备。`SD_METAL`只接通GGML Metal构建，不证明每个op都在GPU、没有CPU fallback，或支持独立ANE。

`diffusion_engine.cpp:949–1008`里，auto-fit仅当`auto_fit && params_backend_spec.empty()`时启用；显式params assignment会关闭它，即使auto-fit字段仍为true。原始规划将元数据转换后的tensor type计入权重估计，初始compute reserve为DiT/TE各2GiB、VAE1GiB；它不是按某个分辨率/帧数实测的workspace。

一个简化的生成阶段链可以这样读：

```text
checkpoint元数据 → 参数后端（Metal / CPU RAM / disk）
    ↓ 需要时load、stage、LoRA、pin
TE runner → conditioning → 多次denoiser图 → VAE decode
    ↓ 每张图可monolithic或fixed segments，unsupported op可进入CPU scheduler
segment结束：同步 → 保存仍活的cut/cache → unpin，不必立刻evict
runner阶段结束：释放workspace/cache、清prefetch、驱逐可重载的compute副本
```

`runner_end`不等于整个context/model源数据全部释放：同后端永久参数、CPU源参数、磁盘源和可重载GPU副本有不同寿命。降低阶段峰值和提高跨sampling-step复用也可能冲突。

## 2. auto-fit按阶段优先级，而不是三份独立设备预算

`backend_fit.cpp:26–71,195–316,418–478`优先处理DiT→TE→VAE。先试模块compute devices，再RAM，再其它GPU，最后保持默认disk。较低优先模块成为永久GPU居民前，要为高优先offloaded模块的完整权重cache和compute reserve留空间；低优先待stage模块还留最大单tensor空间。

这解释了看似反直觉的选择：即使小TE/VAE单独能放进GPU，也可能一起选择CPU源参数，以便反复执行的DiT复用缓存。显式compute backend不因完整权重超过预算就自动变CPU；后续分段仍可能在GPU执行。auto-fit也不会自行创造多GPU layer/row计算配置。

CPU observer逐字提取并执行原始`select_main_device`、`compute_plan`、`params_backend_name`及其POD类型。合成组件为DiT8GiB、TE2GiB、VAE1GiB，reserve为2/2/1GiB、staging各1GiB：

| 合成预算/显式runtime | 原始策略参数选择（DiT/TE/VAE） | 说明 |
| --- | --- | --- |
| GPU16、RAM16GiB | GPU/GPU/GPU | 能连同headroom一起容纳 |
| GPU6、RAM16GiB | CPU/CPU/CPU | 给高优先DiT留cache空间；不是三模块改成CPU计算 |
| GPU6、RAM0GiB | disk/disk/disk | 没有显式disk assignment也有此兜底 |
| GPU0=6、GPU1=14、RAM0，runtime固定GPU0 | GPU1/GPU1/GPU1 | 其它GPU作参数storage；compute仍由显式runtime约束 |
| 同上但RAM16GiB | CPU/CPU/CPU | RAM先于其它GPUstorage |
| 无GPU、显式CPU runtime、RAM4GiB | disk/CPU/CPU | CPU计算也可有disk参数源 |
| 无GPU、没有显式runtime | plan.valid=false | 外层再处理默认后端，不冒充计划已成功 |
| 两GPU各10GiB、显式两设备、RAM16GiB | CPU/CPU/CPU | 构建真实split前会保守按每设备全权重计 |

这些GiB是输入fixture，**不是本机可用内存或多GPU实测**。八种actual输出与独立Python策略一致，未执行设备枚举、metadata loader、模型/设备buffer分配或文件IO；observer仍有普通std::vector等CPU堆分配，数据中no-alloc资格仅指前述设备/模型范围。

文档差异需保留：此ref `docs/performance.md`有“disk never selected implicitly”的无条件句，而`docs/backend.md:268`明确限定为auto-fit之外，且实际planner及`derive_backend_specs:461–463`会输出disk placement。这里以固定源码和原始CPU函数观察为依据，不继承过宽的性能文案；“CPU offload不降速”也不是通用结论。

## 3. Apple统一内存不让CPU参数与Metal参数自动成为同一份

固定GGML的`ggml-metal.cpp:275–279,672–684`中，Metal shared buffer的GGML `is_host`仍false；device caps为async=true、host_buffer=false、buffer_from_host_ptr=true、events=true。这些分类不是在否认底层物理共享内存，而是在说明GGML可用的buffer ABI。

`ggml-metal-device.m:867–876,1635–1690`默认对统一内存设备使用shared，且存在环境覆盖；shared allocation有自己的host allocation，再用`newBufferWithBytesNoCopy`包装。它不等于任意CPU源权重自动alias到该allocation。同步shared `set_tensor:1850–1853`是memcpy；异步host上传走另一条路径：

```text
CPU源weight bytes
  → newBufferWithBytes（复制内容到新的Shared staging buffer）
  → blit到最终Metal tensor backing
  → command completion后才能结束相关owner义务
```

因此所读RAM→Metal异步分支存在copy和临时对象，即便两边都在同一DRAM。mapped/host-pointer import另有路径，未沿完整loader验证，不能反推所有mmap场景都一定如此，也不能把`WithBytesNoCopy`出现一处就写成全链路零拷贝。真正peak应包含CPU源、最终GPU副本、prefetch副本、上传temporary、workspace/cache及驱动开销。

Metal `get_memory:1076–1083`报告的是recommended working-set减current Metal allocation，不是独立物理VRAM剩余。auto-fit另查询可用RAM；在Apple UMA上不能将两份逻辑budget当成两套可相加的物理容量。这里没有观察实际pressure、RSS、compression或驱动复制。

## 4. 第二个backend不是第二个传输queue

`ModelManager::prefetch_backend_for`在相同device上创建第二个backend context，`populate_prefetch_block`按alignment/max-size拆staging块，host源用tensor_set_async，非host源用tensor_copy_async，最后记录event。`activate_prefetched_params`先event synchronize（无event则同步transfer backend），再swap tensor的buffer/data/extra，最后将staging块转入compute缓存。clear/destructor也先同步再free。

但固定GGML `ggml-metal-context.m:103–112`没有为每个backend新建queue，而从device取queue；`ggml-metal-device.m:723,980–982`返回同一device持有的queue。SD runner又在`enqueue_next`之后才调用当前segment的`execute_segment`，因此源码上的提交顺序示例是：

```text
upload(next) → record transfer event → compute(current)
```

Apple [MTLCommandQueue](https://developer.apple.com/documentation/metal/mtlcommandqueue)说明queue维护有序command buffers，也允许多个host线程同时编码。**async API、两backend、两host活动窗口都不足以证明copy与当前compute物理重叠**；本路径未提供独立传输queue的证据。不能由此倒推GPU内部一定绝无任何重叠，但也不能直接把next-segment预取的理想`max(copy,compute)`计入性能模型。需要实际queue/device trace与完整消费者A/B。

`SegmentWeightPipeline`还会跳过没有参数的segment、排除当前segment共用参数、检查missing bytes与async支持、保护当前和queued参数。Unsupported/Failed、容量失败会清理并关闭该pipeline的后续预取；同步准备路径仍保留。layer-split multi-device路径禁用此单设备分段/预取；row-split buffer也不能走主设备async upload。不是所有后端/多设备模式都共享一种streaming策略。

## 5. monolithic/segment决策算的是新增成本

`GGMLRunner::execute_graph:833–1028`仅在单非CPU runtime、有manager、切分开关允许、有效cut plan且多segment、monolithic放不下时选择segments。monolithic预判额外留128MiB planning headroom；运行期还有独立512MiB scratch reserve。

将`P=pending allocation`、`R=current runner已保留workspace/cache`、`W=missing weights`、`A=本设备已驻留weights`、`O=其它runner已注册runtime bytes`，所读单设备Metal容量分支要求：

```text
P + W + 512MiB ≤ live available device bytes（再由total−tracked residency收紧）
R + P + W     ≤ managed limit − A − O          （设置正limit时）
```

live检查不把已存在R再当新分配扣一遍；managed预算则必须包含它。managed limit不含所有外部/driver allocation，不是physical硬上限。source对0/0设备内存报告保留unknown，不等于0可用；本库GPU空闲准入仍独立要求可读计数、缺失拒绝，不能混用两种策略。

运行时先重用或释放过大idle workspace，向manager请求容量，再prepare/pin权重。LoRA准备可能执行图并回收idle workspace，因此要重新核scheduler/measurement、再prepare、再做容量检查；实际allocate后还会复查。驱逐先尝试其它runner的idle workspace，再从反向segment优先序选可重载参数，最后用全局last-use epoch；pinned和queued-prefetch参数受到保护。

`release_compute_backend_params`这里只减pin_count，不立即free；未pin副本可跨sampling步复用，内存压力或runner_end才驱逐。固定cut plan和跨cut活跃值不是“按层随便free”：每段先bind仍活cut，执行后capture未来仍需的cut/cache，最终output/callback读完再prune future集合。

## 6. CPU fallback与失败不能藏在“Metal backend”标题里

`ComputeWorkspace::needs_scheduler`只要发现任一节点不受runtime backend支持，或有extra backend，就启用GGML scheduler；非CPU runtime会附CPU backend作为fallback。Metal此ref没有专用host-buffer type，scheduler可退到CPU default buffer。实际compute summary也单列CPU workspace，但该summary排除weights/cache，不是完整request peak。

scheduler allocation会重写graph sources；测量使用图副本，实际graph只split一次。无scheduler的gallocr按layout signature复用reservation；segment结束同步但保留capacity，runner_end再释放。日志与eval callback也有限制：scheduler路径会警告并忽略该eval callback，不能再把callback-based观测当完整算子轨迹。

segment完成/清理要先同步再解除pins、restore bindings和free graph context。`ComputeWorkspace::segment_end`先将active置false，防止外层cleanup对失败提交再试一次；捕获同步异常并报告失败，而非悄悄把状态当成功。GGML Metal context也有has_error锁存，后续graph compute会拒绝直至重建backend。源码清理合同不等于本机fault/drain测试，本轮未故意触发GPU错误。

## 7. VAE OOM重试不是一般设备错误恢复

`decode_first_stage:2881–2905`使用局部tiling params副本；decoded非空直接返回。仅当last_compute_status为ALLOC_FAILED时才调用retry helper：支持的video、5D、多于一latent frame时先尝试temporal；之后空间尺寸取半，初次还用256/scale限制，最低4 latent pixels，并要求至少一个轴真正变小。写回tile_size单位是**image pixels**，清除relative overrides。到最小值后停止，不无限重试。

执行失败不通过tile变小重试；encode没有这个自动OOM循环。不同VAE的temporal支持默认false、可按model/direction覆写；空间/时间切分也可能改变上下文、边缘和质量，不能称数学无损。独立CPU见证只覆盖这些策略条件，不是原VAE运行或真实OOM。

## 8. attention开关仍有实际shape、dtype与消费者成本

`ggml_ext_attention_ext:624–811`并非开flag就盲目发FA：重排/连续化Q/K/V，K/V转F16，对per-key padding mask展开query维到Lq，再由`ggml_backend_supports_op`检查具体out；不支持或mask条件不满足就回manual matmul→softmax→matmul，可能重新出现显式score矩阵。`used_flash_attn`才反映本次构图是否成功选择该分支。

`kv_scale=c`把K、V先乘c、logit scale除c、最后输出再除c；精确算术中分别抵消K和V缩放，但F16 cast和归约误差仍要验证。CPU小例证实漏掉最后inverse scale会保留错误的输出倍率，不是完整attention数值验证。SageAttention入口在所读pipeline明确检查patched GGML/CUDA支持，不因Metal可用就可照搬CUDA快路径；fallback也不能当ANE。

## 9. 证据、失败与复测

实际CPU执行为八种原始plan例子；native仅声明libc++/libSystem动态依赖，没有GGML/Metal/ANE。第一次本库提取器误把默认参数`{}`当函数body，host编译失败；修正提取规则后在新cpu-source-v2目录编译成功，旧include/receipt/stderr保留并有回归测试。这是本库工具错误，不是上游编译失败，也没有通过改上游源码绕开。

25项标准库测试另覆盖独立预算、retry、缩放与来源边界。它们不是完整scheduler、prefetch、Metal错误恢复、VAE质量或真实生成速度测试：

原始参数源码、官方元数据、Apple正文、逐字CPU include与MIT notice、build/run日志只留ignored `outputs/gpu-ane-sdcpp-round18-20261017`；分发[来源JSON](../sources.md#sdcpp-residency)和保存身份/hash/全部plan及失败。没有上游源码/二进制进轻量Wiki，没有提交PR或改参考子模块。

下一步需同模型/seed/精度/LoRA、同Metal基线，比较resident/CPU-source/disk-source与prefetch开关，完整TE→denoise→VAE计时并看cut/cache/peak、真实copy及fallback；VAE分开OOM/执行失败和图像质量。所有设备运行先过连续5次一秒GPU占用均低于5%的guard，缺计数拒绝，忙时等待；本轮没有设备进程。

固定源码：[auto-fit](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/core/backend_fit.cpp#L206)、[runner执行](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/core/ggml_runner.cpp#L833)、[prefetch](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/model_manager_prefetch.cpp)、[Metal async上传](https://github.com/leejet/ggml/blob/4bf5f6000653b7881d00963cd6ddb665ccd62a8d/src/ggml-metal/ggml-metal-context.m#L307)、[同设备queue](https://github.com/leejet/ggml/blob/4bf5f6000653b7881d00963cd6ddb665ccd62a8d/src/ggml-metal/ggml-metal-device.m#L980)。范围/隔离验证见[第十八轮](../../MAINTENANCE.md)。
