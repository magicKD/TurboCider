# PMetal 深读：GPU 草拟、ANE 验证与真正生效的 GPU backend

<!-- knowledge
id: technique-gpu-ane-case-studies-pmetal-speculative-and-dispatch
type: case-study
stage: ffn
operator: swiglu, gemm
backend: cpu-reference, mlx, coreml
device: cpu, gpu, ane
symptom: straggler, precision-loss
semantics: numerically-equivalent
representation: fp16, fp32
boundary: gpu-ane, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [协同分类](../partition-and-scheduling.md) · [续研证据](../../MAINTENANCE.md)

固定 [`Epistates/pmetal@a225c0e2568f882573b05caeb42d94fed208fdd3`](https://github.com/Epistates/pmetal/tree/a225c0e2568f882573b05caeb42d94fed208fdd3)。本轮读 `ane_worker.rs`、`dflash_drafter.rs`、`ane/lm.rs` 的构建/生成、`extend.rs` 的实际cache更新、GPU dispatch/backend；不安装依赖、不运行模型，不迁移README的性能倍率。

## 1. 先排除两个名称陷阱

`ane/inference_hybrid.rs` 的hybrid指Qwen3.5混合GDN/full-attention架构，该文件实际是CPU Accelerate GEMV decode，不是GPU+ANE并行分担。`KernelDispatch` 则是Metal3/Metal4 GPU路由，也不是ANE分区器。

本轮真正找到的异构实现位于 `AneEngine { lm: AneLm, drafter: DFlashDrafter }`：小DFlash模型在GPU草拟，大target在ANE验证，CPU负责embedding、采样/状态衔接。这是**算法角色分工**，不同于同一FFN的row/F切分。

## 2. 常驻模型线程与请求状态

`ane_worker` 用一个长期ModelThread保存按checkpoint路径索引的engine，串行处理请求，token事件与caller的继续/停止反馈通过channel交握。context容量或draft路径不匹配会先移除旧engine，再构建新模型；panic时清掉该线程缓存的models，避免后续请求继续用半写KV。

这些保护比“模型可运行”更接近服务生命周期，但路径key仍假设checkpoint内容未原地变化；不能拿它当内容寻址身份。callback停止与panic清理也不等于所有private runtime故障都已有完整drain/恢复资格，需要故障注入单独测试。

## 3. ANE target 把多个layer放进同一program

`AneLm` 的所读入口只接受Qwen3-shaped dense模型；不能将整个仓库支持的所有模型架构继承到这一ANE路径。默认width16、capacity2048、INT8常量权重、每program约1GiB权重预算，是该实现的选项，不是ANE规格。

`layer_bytes`估算决定每个program含多少层；逐层读权重构建后释放临时FP32数据，不长期保留整个模型的FP32影子。增加层融合减少dispatch，但影响编译、常驻与失败范围；width更大有利prefill吞吐，却可能让decode验证做多余工作。

extend图将历史cache与当前新token作为两个attention块进入同一个softmax。host在调用后只追加新K/V，不为每次append复制整个cache；K为channel-first、V为token-major，避免图中不利的输入transpose。

但“cache不全量复制”并非“host不参与”：所读`extend()`仍有CPU padding、RoPE、mask构建、输入FP16转换、每layer新KV写入和hidden输出转换；`step()`另有CPU embedding和LM-head交接。

## 4. GPU drafter 与 ANE verifier 的真实顺序

```text
ANE prompt/target输出 → host hidden taps → GPU draft+完成
    → host proposed tokens → ANE target验证一块
    → 接受前缀/截断逻辑KV位置 → host taps → 下一轮GPU draft
```

`DFlashDrafter::observe` 将taps保存在host `Vec<f32>`；`propose` 构建GPU array、调用draft、`eval`并把tokens读回。`AneLm::generate` 随后同步`step`验证。因此这是本轮已读的**交替依赖链**，不能宣称GPU与ANE在同一请求中持续物理并行，也不是零host-copy桥。

load时会预热多row与单row的draft路径，减少首请求的不同packed-matmul编译；冷启动与两份模型/target head等驻留代价必须单列。

## 5. 接受前缀、采样与KV回退

一个pass输入已决定的`next`及若干draft guesses。target sampler按位置依次决定token，只要与guess相等便接受；第一次不等就停止，**不继续采样错误前缀之后的logits**。这不是经典p/q acceptance-ratio实现：它直接以target sampler的决定为准，接受率/性能取决于draft与target的实际匹配。

target已经物理计算过整块，随后逻辑position回退到 `start + 1 + accepted`。`ExtendModel::truncate`不清空旧slot，而在下次mask中隐藏位置以后的cache；将来覆盖新slot。taps也只观察`kept+1`个有效context，不能把被拒绝branch的hidden永久喂给drafter。

这套截断不能机械套到有额外递归state的GDN/Mamba：只回退KV位置不一定恢复SSM/conv state。恰好所读ANE入口的模型限制防止了这种直接泛化。CPU前缀见证检查采样调用数、位置和first-mismatch；不证明真实token parity或量化质量。

## 6. GPU “选中Metal4”还不等于用了MPP GEMM

`KernelDispatch::backend_for_gemm` 检查构建、NAX和库、shape资格后可返回Metal4 backend；`preferred_backend`更只是默认偏好。实际 `Metal4Backend::gemm` 在该commit直接委派给`self.fallback.gemm`；量化GEMM才构造MPP dispatcher并检查`is_available()`。不能根据router名字或feature表统计全部算子的TensorOps使用率。

`encode_mpp_kernel`又仍使用传统commandBuffer/encoder，注释把Metal4生命周期迁移留作未来工作。**TensorOps shader、backend选择、Metal4 command API是不同实现层。** 返回command buffer已commit但未wait，消费者仍要建立完成依赖。

## 7. 怎样判断这条协同路线值不值得采用

对比target-only与GPU-draft/ANE-verify，冻结target checkpoint、精度、tokenization和sampler，报告accepted tokens/pass、draft/verify/handoff分项、首请求、TPOT和每请求完整输出。最优width、context bucket、draft量化不能从作者表直接继承。

若draft+host交接比节省的target passes还贵，或双模型常驻挤出关键cache，则退回target-only。该路线适合研究低带宽/高dispatch成本下的多token摊销，不是DiT同层并行的替代证明。
