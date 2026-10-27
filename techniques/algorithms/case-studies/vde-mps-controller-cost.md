# VDE在MPS上的净成本：少调用、标量读回与观察器重入

<!-- knowledge
id: technique-algorithms-case-studies-vde-mps-controller-cost
type: case-study
stage: dit
operator: attention, sampling, layout
backend: metal, cpu-reference
device: gpu, cpu
symptom: launch-bound, stale-cache, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [VDE源码/CPU反例](../methods/vde-input-adaptive-velocity.md) · [来源 S69](../sources.md#s69) · [配套实验](../../examples/README.md)

2026-10-17，第25轮。实际MPS共同去噪窗口里，VDE从50次完整网络调用降到16次，但配对中位收益只有约1.67–1.80倍，不是按调用数相除得到的3.125倍。保留相同近似轨迹、在已验证的唯一时间日程下去掉GPU重复时间检查的变体，相对VDE原版约1.41–1.55倍，端点逐字节相同。这个结果说明控制器的host依赖值得单独核查，**不是预训练FLUX、完整媒体生成或任意Apple模型的速度承诺**。

## 同一窗口，五个方法

预登记计划、worker和launcher固定50次Euler更新、`stable_step=6/interval=5`。四个近似臂完整求值均为索引`0…6,11,16,21,26,31,36,41,46,49`，共16次；进入后期history的完整锚点11次，估计34次。这不是物理allocation/copy计数；没有为了某个方法另减步数/分辨率。

| 臂 | 估计时用什么 | 来源及边界 |
| --- | --- | --- |
| full | 每步真实网络输出 | 本库共同循环＋实际Diffusers模型/FlowMatch |
| replay | 最近完整输出 | 本库简单重放强对照，不冒充某作者cache实现 |
| linear | 最近两次完整输出，按真实`t`线性外推 | 本库全输出外推对照，不是TaylorSeer完整实现 |
| vde | 原作者完整模块中的分解/外推helper | 公式及全tensor归约保持不变；非原公共pipeline计时 |
| vde-no-item | 同分解/重建；两系数共享一次weight，去掉重复时间host检查 | 本库受限变体，不修改原作者函数或安装库 |

每次请求在计时内重新生成/设置scheduler的50步日程、重置begin index和局部history，执行全部网络/估计/Euler，再调用`torch.mps.synchronize()`结束窗口。权重、packed latent、embedding、position IDs和guidance已驻留；准入、CPU参考、设备准备、CPU读回/检查、journal写入在窗外。没有只测估计hit而丢掉刷新成本。

这是**共同的完整latent去噪循环**，不是原`FluxPipeline.__call__`或`vde_flux_call`完整公共API调用时间；其prompt准备、progress、hook等额外边界未计入。另在窗外实际运行stock/未改作者完整pipeline：CPU与MPS各8次端点均和共同循环逐字节相同，证明所测recipe对齐，不把端点对齐转授为公共pipeline性能。

## 实际环境和工作量

本轮单独采集：M4 Max、64 GiB、macOS26.6.2/build25G83，Python3.13.12、Torch2.14.0声明git`08187d9e…`、Diffusers0.39.0固定`a3608b5…`。GPU worker核对MPS Python wrapper字节；独立CPU控制核对`_device.py`安装字节，六个Diffusers选定文件沿S68匹配。Host SDK26.5不是Torch构建SDK；这不是完整wheel/native二进制闭包，也不借用另一线程Torch2.14.1的身份。

实际`FluxTransformer2DModel`，2个双流＋2个单流block、2个attention heads、FP32、batch1，guidance embedding开启。`width64`为inner64/head32/483,472参数，`width128`为inner128/head64/1,843,472参数；各有64/256个image token、16个text token。112项参数manifest及六份输入/条件/位置银行在全部CPU/MPS臂前后匹配。改变width也改变整网宽度/参数量，不能孤立归因于head/kernel。

全部是本次构造的随机网络、合成latent/embedding；没有预训练权重、真实activation、CLIP/T5、VAE或图片/视频。因此没有本轮图像质量，也没有CPU/GPU速度比较。

正式数据：四cohort×10个平衡trial×5臂＝200个窗口。五次轮转再五次反序轮转，每臂在各位置出现两次；20个cold/40个warm排除，cold是进程次序相关首个计算请求，不是系统shader-cache冷启动。正式完成后才运行20个观察器诊断和8个原pipeline对齐，避免把hook成本放进正式窗口。

GPU初始与各phase/trial均用连续三次Device≤5%准入；native共115窗口，61通过、54暂缓，另有外层新鲜准入及不含argv的进程快照。暂缓可包含自己的工作冷却，不当作54次外部竞争，也没有证明连续独占。只限自己子进程，未停止其它进程。

## 配对成本：调用减少不是完整收益

下表全为每个trial的`full_time / candidate_time`再求中位数，**不是中位时间相除**。IQR是样本四分位范围，不是置信区间；全样本保留在公开JSON。

| cohort | replay加速 | linear加速 | VDE加速 | 去guard加速 |
| --- | ---: | ---: | ---: | ---: |
| width64，64 token | 2.727× | 2.720× | 1.668× | 2.582× |
| width64，256 token | 2.845× | 2.776× | 1.776× | 2.669× |
| width128，64 token | 2.718× | 2.663× | 1.803× | 2.600× |
| width128，256 token | 2.666× | 2.585× | 1.750× | 2.522× |

四组full窗口中位约113.6–118.5 ms；原VDE约66.3–70.2 ms，去guard约44.1–47.5 ms。没有将表中配对比值改写成这些中位时间的商。

最接近受控语义的比较是去guard/VDE，其配对中位及IQR分别为：1.549［1.434,1.595］、1.413［1.306,1.492］、1.502［1.369,1.555］、1.442［1.342,1.544］。所有冷/热/正式/诊断观察的两个变体均共享同一个MPS端点payload，CPU端点也相同。这个采样范围支持“当前配方的额外host guard很贵”，不支持所有真实模型都有同倍率收益。

原VDE对相同调用次数的replay/linear配对比均小于1。replay/linear与VDE改变近似函数，**没有相同媒体质量预算**，不能只凭更快就建议替换。随机端点relative-L2反而是linear约2.46–3.78%、replay约3.47–5.31%、VDE约4.96–6.80%；这组负结果说明“input-adaptive”不是任意函数的误差优势定理，更不能把latent百分比当图像损失。VDE论文的真实模型质量结论不由这些随机控制推翻或复现。

## 去掉的是什么：bool检查也是设备依赖

作者predictor在两锚点条件下先比较`dt==0`，再对这个**布尔tensor**调用`.item()`，让Python选择返回/外推。共同循环的34个估计步各执行一次这段检查，即34个源码级逻辑查询；不是读取整个旧速度，也不是读取34份完整latent。

所读固定源码链：`Scalar.mm::_local_scalar_dense_mps`建CPU输出并调用`mps_copy_(..., false)`；`Copy.mm`同dtype路径进`copy_and_sync`；`MPSStream`把`non_blocking=false`映射到当前stream的`COMMIT_AND_WAIT`。这为host依赖提供具体实现依据，但不是本轮逐GPU命令观察、物理copy字节数/次数或全设备同步测量。

去guard变体只在host端先确认**本计划全部50个时间严格下降且首估计已有两锚点**后使用；两系数共用一次设备weight，再分别lerp。保留当前latent norm/方向重建及所有刷新。任意重复时间、自定义回调、动态改日程或history缺失时不能无条件去掉检查；应恢复guard或拒绝。它是受限合同下的算法实现变体，不是上游修复/PR或产品默认开关。

删除host依赖也会改变命令合并/提交机会，并省掉布尔比较、Python分支等开销。因此不能用总墙钟差直接当“复制一个scalar的耗时”，更不能据此算出单条GPU指令成本。本轮没有profile/capture/ISA或物理kernel分支证明。`torch.norm`/sum的所读dispatch也只给条件，不把API次数当真实dispatch数；`fallback=0`同样不证明所有op纯GPU。

### 为什么68个hook entry不是68次同步

观察器只在正式窗后临时包裹`torch.Tensor.item`，显式`finally`恢复；VDE记录68个MPS bool/标量形状entry，其它四臂为0。这个数字比34个源码逻辑查询多，不能直接统计成68个读回/同步。

独立CPU校准在同一固定Torch上执行34个明确的`.item()`逻辑请求，完整保存call ID与重入depth：

| 默认设备状态 | 逻辑调用 | hook entry | depth模式 |
| --- | ---: | ---: | --- |
| unset | 34 | 34 | 每次0 |
| `set_default_device("cpu")` | 34 | 68 | 每次0、1 |
| reset为None | 34 | 34 | 每次0 |

已读`DeviceContext.__torch_function__`会将函数再次转发；这个CPU控制复现了观察器重入。MPS工作进程也启用了该默认设备mode，但原MPS记录没有嵌套depth，故这里只把68保留为真实hook entry，并以源码日程解释34个逻辑检查，**不补造68次物理copy，亦不授予精确C++内部同步次数**。正式窗口没有安装该hook；CPU校准更不是MPS计时。

## 数值、归档和首次排除

标准库整理器重放全部663行journal、所有准入/窗口/顺序/参数输入、288次GPU端点与20份同recipe CPU参考；40个content/shape payload包含输入及CPU/GPU值，去重没有丢掉任何观察。固定门槛`atol=rtol=1e-4`、relative-L2≤0.001不变；最差max-abs`6.4522e-6`、scaled`0.06326`、relative-L2`3.6748e-6`。这些都是**同recipe backend parity**，不是对full的近似质量通过，也不是独立重新实现完整transformer数学。

首次完整运行的去guard客户端对alpha/beta各算了一遍相同weight。它数值通过，但不符合预定共享单weight配方；保留原client快照、完整receipt/native与SHA，整体排除其200个正式窗口。修正本库客户端后按同一个plan、同一个数值门槛重新跑全部五臂/四cohort；本页只用第二次run，不选择性混入首次的其它臂，也没有改上游或为过门槛改模型/阈值。

前两条仅标准库/公开证据，不导入Torch、执行IORegistry或GPU。真实复跑另需显式`--run --gpu`、新输出目录及新鲜准入；归档整理器严格绑定本次第二个cohort，另起实验需新记录，不覆盖旧证据。

## 对Apple迁移的建议

先画出每个估计步的host依赖，再计算跳层省算。固定唯一日程能把某些值依赖证明搬到setup，但proof要绑定scheduler/分辨率/回调与history；不要默认所有tensor标量都可删。驻留history、归约精度、完整请求重置/刷新也应进成本账。

这组小型FP32模型中，控制器/提交边界很重要；生产FLUX更宽更深、FP16/BF16/量化与long-context可能改变占比。真正选择VDE、直接外推或缓存，仍须在实际模型、相同媒体质量门槛下复测完整生成。上一轮的batch耦合/FP16溢出反例仍有效，本轮batch1/FP32不能替它们作安全背书。

重开条件：Torch/DeviceContext/MPS wrapper/native源码或OS/SoC、Diffusers/VDE、dtype/模型宽度/序列长度、默认设备mode、独立流/同步边界、sigma/回调/锚点合同、真实媒体质量预算改变。
