# Qwen encoder：旋转 W8A8、执行成本和继续优化的路径

2026-10-08，接续 [19](19-edit-encoder-ane-runtime.md)。本轮只验证 encoder，使用已有 BF16 checkpoint 和 teapot／fox／puffin，两个有限设备实验组，不运行 DiT／VAE 或下载模型。新候选的完整三图 encoder 有小幅净收益；仍保留诊断入口与 GPU 默认。

## 1. Splash 到底使用什么量化

固定上游 commit `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`：

- [AneFfn.cpp](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfn.cpp)：随机符号 Hadamard 旋转，seed=20260930，input block128／hidden block512，weight per-row 和 activation per-token INT8。
- [ane_ffn.metal](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/metal/kernels/prefill/ane_ffn.metal)：寄存器／SIMD butterfly、旋转、scale、pack、restore/join。
- [AneFfnCalibration.cpp](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfnCalibration.cpp)：完整路径收益与行数校准，不能以单个 matmul 的成绩替代完整路径。

这是**旋转 W8A8**，不是本轮使用 SmoothQuant profile 的实现。按上游行／列约定对输入及权重应用同一个正交变换，点积在未量化时不变；随机符号与 Hadamard 把离群值分散到 block，再取 INT8 codes。这里的 runtime 表示动态权重输入、执行图及程序复用，并不等于 runtime SmoothQuant。

中间 hidden 在 ANE 用 grouped 1×1 conv 做 H512 旋转，再按 token 取 peak／量化。归一化 scale／载体及非线性使用 FP16，GPU 以 FP32 恢复与相加。因此可以称作算法／数据表示上的 W8A8；源码中的 `dequantize -> fp16 matmul` 是否被编译成硬件 INT8 指令，需要额外设备证据，不能由名称或 INT8 Surface 独立证明。

上游还有两个重要适配边界：

1. 最新 `AneFfn::unsupported` 要求 hidden 是 **2560 的整数倍**，并接受 affine Q4／未旋转的量化 GGUF source。Qwen3-VL H4096、dense BF16 不能直接传入这个 API。我们的 H4096/K2048 程序和 dense staging 是本地适配，不能宣称直接移植上游后端。
2. Splash 的 SiLU 用 `g/2*(1+tanh(g/2))`，用于减少其 sigmoid LUT 在小 gate 值上的误差。本地当前 W8 程序用 exp／division，已避开 sigmoid LUT；换成 tanh 必须先做真实 gate 值的误差及时间筛选，不能从上游数字推断本地必然获益。

## 2. 本轮实际改了什么

[EncoderRuntime](../../../native/models/qwen21/encoder_runtime.hpp) 保留完整 GPU fallback、source ownership、private executor drain、BF16 模型边界和 FP32 channel join。新路径复用 [dense_gpu_projection.hpp](../../../native/backends/dense_gpu_projection.hpp) 的现有 MPP range GEMM，不新建模型权重副本。

### 2.1 BF16 operands，FP32 partial

旧 down partial 将 hidden 和 down weight 都转成 FP32，再做 FP32 matmul。它增加转换、访存与临时 live bytes，也可能降低低精度矩阵吞吐。新路径直接读原始 BF16 source，保留物理 leading dimension 和 channel range，以 FP32 累积并输出 partial，join 后才转 BF16：

```cpp
// a = {x, gateWeight, upWeight, downWeight}; all original BF16.
auto gate = mx::matmul(a[0], mx::transpose(slice_axis(a[1], 0, 0, gpuChannels)));
auto up = mx::matmul(a[0], mx::transpose(slice_axis(a[2], 0, 0, gpuChannels)));
auto hidden = silu(gate) * up;
auto partial = dense_gpu::projection_range(
    hidden, a[3], 0, 4096, 0, gpuChannels, 32, true);
return std::vector<Tensor>{partial, hidden};
```

gate/up/SiLU 的 BF16 边界没有改。FP32 output 不意味着必须扩大权重到 FP32。

### 2.2 36 层共用一个 dynamic-weight compiled FFN

旧路径每层构造 compiled closure 并捕获该层权重；释放上一层 closure 能控制内存，但会重复 tracing。新 compiled callable 只捕获固定 channel count，权重是动态参数。一次完整 encoder 实测 callable 构造计数 **36→1**，调用仍是36次；这不是整个 encoder 合并成一个 Metal command buffer，也不表示只编译一个 PSO。

进程固定的诊断变量 `TURBOCIDER_QWEN21_ENCODER_RUNTIME_GPU=mpp|legacy` 支持同构建对照。显式 encoder runtime 候选默认 `mpp`；整个 App 的默认仍是 GPU，未设置 encoder runtime manifest 不会进入此路径。recipe 纳入进程固定校验和结果 source recipe，缓存不得跨 recipe 使用。

### 2.3 去掉重复的全 stream drain

新路径在导入前保留 `mx::eval(x)`，取消其后的额外 `mx::synchronize()`，完成／销毁时仍排空。
[MLX 0.32.0 transforms.cpp](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/transforms.cpp) 中 eval 对已经 scheduled 的输出调用 wait，对新输出等待 eval_impl 的依赖事件；input readiness 不依赖“图已创建”的假设。

这只是减少一个冗余全 stream 等待。`HybridFfn::run_channels` 仍会 eval padded input，仍有 CPU finish/join；不能称作 Splash 的设备驱动多层图。后续应由 padded/import dependency 边界承担唯一 input fence，先验证 owner、offset、异常和 cancellation，不直接删掉所有 eval。

### 2.4 已有寄存器 Hadamard staging

候选显式使用 `TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE=1`，复用已有 H128／H512 寄存器 staging。旧路径用0。它只改变 W staging；不是 SmoothQuant、新精度或 W2。当前对比同时改变 GPU partial、compiled callable、drain 和 staging，**不能把收益全部归因于某一个内核**。

## 3. 本轮证据与口径

完整数据、build/source/binary 身份和负结果见 [状态](../../status/2026-10-08-qwen-encoder-followup.md)。

真实 layer0 FFN 使用完整视觉特征与真实 post-attention-norm 输入，1／2／3参考图为1071／2101／3131行。每个候选／shape一次 warmup、AB和BA两个配对，hybrid计时包含 stage、input ready、两分支和join。6144通道新路径两顺序省时：

| 参考图 | AB | BA | FFN RRMS |
|---|---:|---:|---:|
| 1 | 20.87% | 19.48% | 0.748% |
| 2 | 27.30% | 24.17% | 0.745% |
| 3 | 22.37% | 22.38% | 0.740% |

完整 encoder 包含视觉27层×3、完整语言36层、mRoPE、DeepStack、retained pre-final-norm conditioning；两侧都逐层释放 source。fresh process 的 GPU→legacy→mpp-register 各1个样本：

| 路径 | 视觉 | 语言 | ANE准备 | encoder总时间 |
|---|---:|---:|---:|---:|
| GPU | 4.546s | 7.327s | 0 | 12.144s |
| legacy | 4.555s | 6.938s | 0.277s | 12.048s |
| mpp-register | 4.550s | 6.587s | 0.249s | 11.666s |

新路径语言少10.09%，总时间少3.93%，相对本轮legacy总时间少3.17%。36个 hybrid FFN、108次 ANE evaluation，没有fallback；所有源和reference复核通过，没有新增swap或系统thermal warning。MLX peak约2.16GiB，主要受视觉阶段影响；不包含IOSurface／driver，不据此声称完整进程只有2.16GiB或peak已降低。

新旧 hybrid 的 retained conditioning **逐值／字节一致**；相对 GPU 的 RRMS都是5.754%。本轮在实验前按此前用户允许的范围声明8% encoder数值筛选，已通过；上一轮5%失败保持原结论。5.754% RRMS不是像素误差，也不自动保证所有编辑图片质量。

没有达到本轮完整路径至少5%收益筛选线，且每arm只有一个样本、固定顺序、OS/source/driver cache未清，程序缓存由组件阶段准备。legacy本轮已经接近GPU，不能把历史13.202s与本轮11.666s直接组合成同构建收益。**暂不增加自动路由／App选项或授予发布资格**。这是小幅encoder净收益的实验结果；没有最终编辑图／完整Session验证。

## 4. 为什么完整收益比 FFN 小

### 4.1 覆盖率和一次性成本

此路径只覆盖语言 FFN，视觉塔、语言attention／norm／RoPE／residual不变。视觉独占约37%的encoder总时间；即使语言部分少10%，整体收益也会被摊薄，再扣除约0.249s的准备成本。

从本次数字估算，若准备成本完全隐藏，整体最多由3.93%提高到约5.99%，其他部分不变。它是算术上限，不是已证明的预热收益。编译缓存命中并不等于 session／IOSurface／绑定／self-test 的准备开销为零。

### 4.2 权重 source 和并行调度并不相同

本地从大 BF16 checkpoint 按层懒加载，每层进入 ANE 还要旋转、取scale和写codes；GPU与staging共享带宽。Splash 可以从更紧凑的Q4/GGUF读取，并自行安排当前GPU工作与下一层staging。上游的设备、source、矩阵形状与端到端任务也不同，不能把其加速比直接套在M4 Pro的BF16 encoder。

本次完整新路径 counter：input_ready2.599s、async_ane_host_wait3.359s、stage1.127s、post_join0.059s。input_ready包含上游attention/norm；ANE host wait与GPU工作重叠；stage也可能受到queue排队和带宽争用。**这些数字不能相加，也不能把3.359s称为纯ANE算子时间**。新stage counter高于legacy0.622s，不能据组件staging变快就声称完整层staging也变快。

### 4.3 分块和 CPU 边界

1152 bucket 的3131行输入需要补到3456、每层3个evaluation，36层共108次。padding浪费约10.38%；CPU逐层等待仍存在。单层暖机不会覆盖36层source更换、graph准备、不同层activation与整条链的误差累积。

### 4.4 不能直接开启现有 prefetch

`HybridFfn::run_channels` 当前顺序是launch ANE、`maybe_prefetch`、channel_gpu。`maybe_prefetch`会`mx::eval(next weights)`。encoder source尚未materialize时，这会在当前GPU complement提交前等待下一层加载／评估，可能反而让GPU空闲。Splash的graph先排当前GPU complement，再排下一层weight stage，最后event wait/join。这处顺序需要改造；本轮没有接一个懒加载provider来虚报预取已完成。

## 5. 下一步按收益／成本排序

### A. 先优化当前桥的准备、bucket和唯一input fence

1. 将 immutable compiled program／PSO与 request source bindings／scratch分开：程序可由同device/OS/ABI/shape/recipe复用，源owner每个请求重新绑定；完成／隔离之前不得重新绑定或销毁。
2. 先测同worker第二次cache miss encoder；区分精确conditioning cache hit（应完全跳过encoder）和program复用，不能用cache hit伪装设备加速。
3. 1664行bucket对3131行是2个evaluation、补到3328，padding6.29%、108→72次调用，**仅为待筛选候选**。保持1152旧bucket作为对照，不因为调用数减少就默认更快。CPU先验MIL/预算，再做一次真实full-path筛选；不为每个prompt生成新程序。最多保留少数几何的有界LRU，支持行数不足时GPU。
4. 移除 encoder 层的前置 eval，让run_channels的padded/import dependency证明唯一readiness；验证无padding／padding、取消／异常时的owner和GPU replay。没有跨queue完成证据不能先导入lazy data。
5. 由GPU command completion／shared event驱动下一层prepared BF16 source staging。双bank受共享内存账本限制，最多提前一层；slot busy、源变更、超时或缺少下一层producer时跳过prefetch，仍正确stage当前层。

这些工作比立即增加新的量化格式更接近当前临界路径。本轮没有对A的新增候选执行设备测试。

### B. 建立 encoder 专用量化校准

只需要encoder误差可控，也必须测最终retained conditioning，不能只看layer0。记录少量真实输入的逐通道absmax／二阶统计和关键层scalar误差，流式累计，不落盘36份全序列activation。

- 独立36层profile；H4096、F12288、对应channel range。已有32层DiT SmoothQuant profile不能复用。
- 先保留当前旋转W8A8，再比較encoder SmoothQuant、SmoothQuant+rotation。若`x'=x/s`，gate/up必须使用`W'=W*s`，随后在同一basis旋转；down输入及其scale单独校准。顺序、basis、头寸scale和BF16边界必须进recipe key。
- 量化只处理线性输入／权重；不能把任意rotation穿过SiLU／norm。hidden与down必须配对变换，保持点积关系。
- 在含DeepStack的早期层与最后conditioning层检查误差；必要时少量敏感层GPU，测量质量改善与覆盖率下降，不预先承诺固定哪些层有效。
- 对照exp/div和tanh SiLU的真实gate值；只有数值更好、完整路径不变慢才采用。较小误差不等于更少设备时间。
- profile key至少含模型/source generation、36层几何、数据域、bucket、channel范围、basis/seed、精度边界和编译ABI。当前DiT LoRA不等于encoder LoRA；只在实际修改encoder时扩展对应校准与invalidate。

未来实验重新预声明数值门槛。可以把8%作为明确实验范围，同时追求≤5%；不得在看结果后改门槛或把本轮8%通过改写成上一轮5%通过。

### C. 真正拥有 encoder 的 Metal graph

复用 [16](16-splash-native-backend-spec.md) 的资源管理和现有Native DiT primitives，但先交付完整GPU encoder oracle：Q32/KV8 GQA、causal有效key、head norm、mRoPE、DeepStack、retained行、BF16 residual与视觉D72/bias/GELU语义。DiT的非causal attention不能直接替代。

语言block的目标设备链：

```text
input norm -> Q/K/V -> q/k norm+mRoPE -> causal GQA -> o+residual
 -> post norm -> input rotate/pack -> event_ready
 -> GPU gate/up/SiLU/down complement
 -> next-layer W rotate/scale/pack into alternate bank
 -> wait(event_done) -> restore/FP32 join -> BF16 residual -> DeepStack
```

ANE worker只等待有真实GPU producer的ready，按绑定bank执行并signal done。token/tag绑定request、layer、bank、generation；late completion只完成旧owner，不能推进新请求。先用两个完整block验收异步owner、注入失败和整段GPU replay，再增加多层segment，CPU只在有界segment验收。运行中无法证明drain时隔离资源／worker，不能复用可能仍被设备读取的Surface。

一套program/PSO/source/arena/binding/cache manager服务GPU-only与hybrid，模型计划语义共用。先base encoder有收益，再考虑实际encoder adapter；避免另外复制一套模型仓库。N encoder目前未实现，本文的设备链不是已交付API。
