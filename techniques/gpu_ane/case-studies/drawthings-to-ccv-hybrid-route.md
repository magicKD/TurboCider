# Draw Things→s4nnc→ccv：从模型选择到GPU/ANE rowwise GEMM

<!-- knowledge
id: technique-gpu-ane-case-studies-drawthings-to-ccv-hybrid-route
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [调度/成本模型](ccv-hybrid-cost-and-failure-boundaries.md) · [来源](../sources.md#drawthings-ccv)

2026-10-17。固定公共仓库Draw Things `c31f67eecfe2108e2573bc65ed45ce489a8c9b71`、s4nnc `ccb9971247b64ada98adf58c0028355b7b85fab0`、ccv `2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b`。本轮追的是**公开模型/运行库源代码**，没有运行Draw Things、s4nnc/ccv完整runtime、CoreML、GPU/ANE或checkpoint；不代表商店二进制、用户持久设置或所有生成入口。

## 1. 三个仓库的版本真的是一条链

Draw的[Package.swift](https://github.com/drawthingsai/draw-things-community/blob/c31f67eecfe2108e2573bc65ed45ce489a8c9b71/Package.swift)与Package.resolved同时固定上述ccv/s4nnc版本；[WORKSPACE.darwin](https://github.com/drawthingsai/draw-things-community/blob/c31f67eecfe2108e2573bc65ed45ce489a8c9b71/WORKSPACE.darwin#L133)固定同一s4nnc，其[deps.bzl](https://github.com/liuliu/s4nnc/blob/ccb9971247b64ada98adf58c0028355b7b85fab0/deps.bzl#L13)又指向同一ccv；s4nnc的SwiftPM依赖也一致。

这比随意把三个最新main拼在一起更强，但仍只是**声明/lockfile身份相符**；没有执行Bazel/SwiftPM resolve、编译app或验证发布产物。`_maybe`会尊重已存在的Bazel repository rule，外部override依然需要单独核查。旧本地ccv参考为66dc且是稀疏快照，不能拿它冒充这次依赖。

25个文件的明确人工阅读范围另记；30份单独下载原文件核Git blob。还从97,480,742bytes仓库归档中只解出348份Swift文本并核blob，**不把这些都算已读**；归档含生成/二进制资源，未选取解包。上游root AGENTS全文读取，MediaGenerationKit与SwiftSoup子树不进入。Draw的GPL3、s4nnc的BSD3与ccv的代码/文档/第三方许可分别保存，不把应用许可或第三方许可混为一份。

## 2. App层“允许混合”不是物理并发保证

固定[DeviceCapability](https://github.com/drawthingsai/draw-things-community/blob/c31f67eecfe2108e2573bc65ed45ce489a8c9b71/Libraries/DataModels/Sources/DeviceCapability.swift#L228)保留三态raw值，对应持久键`use_mfa_apple_neural_engine_v2`：

| 状态 | 所读生成入口对NNC flags的改变 | 不代表什么 |
| --- | --- | --- |
| disabled=0 | 插入disable ANE；此分支不改已有hybrid-disable位 | 不等于禁用GPU内NAX |
| ANE-only=1 | 移除disable ANE，插入disable hybrid | 不是所有算子强制ANE；资格不足仍可走GPU |
| ANE+GPU=2 | 移除上述两个disable位 | selector仍可选单设备或拒绝 |

[LocalImageGenerator选段](https://github.com/drawthingsai/draw-things-community/blob/c31f67eecfe2108e2573bc65ed45ce489a8c9b71/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift#L4055)先检查MFA总开关，再套用状态。初始状态的项目策略要求OS26+、Metal、非Intel、Apple9且**非Apple10**；满足时初始为混合，否则disabled。这不是“Apple10没有ANE”，而是该ref的默认性能策略，不继承为硬件规律或当前用户设置。

s4nnc的[DynamicGraph flags](https://github.com/liuliu/s4nnc/blob/ccb9971247b64ada98adf58c0028355b7b85fab0/nnc/DynamicGraph.swift#L823)分别映射GPU Neural Accelerators、独立ANE、ANE+GPU混合disable bits，并读写ccv全局flags；它不是每个Model独有的设备声明。多请求/线程怎样协调这些状态需额外审计，不在本轮伪造race复现。

## 3. 模型block→Dense→GEMM，而非attention固定GPU/MLP固定ANE

固定[Flux2.swift](https://github.com/drawthingsai/draw-things-community/blob/c31f67eecfe2108e2573bc65ed45ce489a8c9b71/Libraries/SwiftDiffusion/Sources/Models/Flux2.swift#L509)的single block将attention输出投影与SwiGLU MLP输出相加，再做modulation/residual。共享输入分支在代数上独立，但这不证明当前graph scheduler已经重叠它们，也不能把全block延迟直接写成两支耗时的max。

s4nnc [Dense](https://github.com/liuliu/s4nnc/blob/ccb9971247b64ada98adf58c0028355b7b85fab0/nnc/ModelAddons.swift#L515)把count/noBias/GEMM flags交给`ccv_cnnp_dense`；ccv [Dense build](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/ccv_cnnp_model_addons.c#L1441)创建`[output,input]`权重并设置GEMM的W转置。因此可达的是**每个GEMM内部按activation行M切分**，不是直接把整个attention与MLP分给两设备，也不是切SwiGLU intermediate F。

同一FFN的上投影、gate和下投影有不同N/K，可能各自选不同row split；完整consumer还包含swish/multiply、输出投影、dtype及join。局部GEMM支持异构不能替代完整FFN/请求验证。

## 4. 权重表示是关键门票：Q8P不等于rowwise I8

Draw的UNetFromNNC加载选段接受`.jit/.q6p/.q8p/.i8x/...`等codec，但列表出现某编码不证明所有权重最后就是该表示。s4nnc [Store](https://github.com/liuliu/s4nnc/blob/ccb9971247b64ada98adf58c0028355b7b85fab0/nnc/Store.swift#L1865)的rowwise JIT路径检查identifier、dtype、形状/容量及已有目标tensor，条件适合才保留CCV_QX rowwise；其它分支会解码到常规tensor。

`.q8p`与`.i8x`是不同codec bit；不能因名字都有8就给相同的GPU/ANE资格。所读上层offload回调还会改codec组合，因此真实采用需检查**最终张量subtype/layout**，不能只抄文件名、模型下载格式或load参数。

ccv [GEMM选路](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/cmd/blas/mps/ccv_nnc_gemm_mps.m#L395)的ANE资格包括：rowwise权重；activation非QX；非转置A、转置W；连续view与相同有效dtype；bias batch约束；合法共享权重batch；MFA/context支持及相关disable位均允许。H256还有K范围等条件。任意前提失败，不等于ANE硬件故障。

这条源路径把`supports_int8_ane`按Apple9+项目gate处理，而GPU NAX另查Apple10；两者必须分开。CoreML请求CPU_AND_NE允许CPU，并不是物理ANE-only证明。

## 5. 另一条旧整UNet CoreML通道不能混进来

同仓库的UNetFromCoreML使用固定旧SD模板、77 token/64×64 latent等门槛，检查控制注入、upcast、tiled diffusion和LoRA条件；CoreMLModelManager与DeviceCapability的相关初始support值为false。它与ccv内的按GEMM rowwise bridge不是同一个开关或模型表达。

不要从旧UNet模板、CoreML类存在或项目“使用ANE”的文案，推出Flux2/视频模型现在全部用这条通道；也不要因旧通道初始关闭，就断言ccv rowwise不可能使用ANE。当前[调度细节](ccv-hybrid-cost-and-failure-boundaries.md)解释的是后者。

## 采用前怎样取证

先固定三个实际依赖ref，再记录生成配置、全局flags、最终weight subtype、GEMM M/N/K/stride/bias和chooser返回；分开统计GPU-only、ANE-only、hybrid、资格拒绝与运行失败。随后检查buffer实际backing、精度/LoRA一致性和完整请求输出，再取device trace及匹配timing。本轮只跑了原始CPU selector的**假设成本**重放，不安装app依赖、不调用设备，不给任何物理并发/倍率资格。
