# 独立后端进入真实请求：Session、来源与内存阶段

更新：2026-10-08。本章记录实际代码合同；性能／质量资格以对应构建的状态回执为准。它是 [10 的新后端路线](10-native-metal-graph-backend.md) 和 [12 的完整去噪图](12-native-denoiser-runtime.md) 的请求接线，当前源码已包含 private W1S 的稳态完整分支与账本。上一构建 `4fd1e980…` 在未开启请求内尺度缓存时完成一组 native GPU／register W1S 的 512² base 完整请求，去噪省时 13.60%、请求省时 9.97%，两路 phase 收尾成功；这是单组收益证据，尚非性能或完整请求内存认证，见[上一构建状态](../../status/2026-10-08-native-register-w1-base-e2e.md)。当前源码新增请求内 compact row-scale 复用，GPU tiny 实现证据以[尺度缓存状态](../../status/2026-10-08-native-request-scale-cache.md)为准；后续真实 FFN 精确复用通过但收益不稳定，未追加 E2E，见[真实缓存筛选](../../status/2026-10-08-native-scale-cache-real-screen.md)。旧收益不转授新源码或缓存。[较早组件状态](../../status/2026-10-08-native-w1-session-components.md) 的 held-out 收益不稳定结果保留原构建范围。原始草案见 14，实际实现见 15，完整 Splash 风格后端规格见 [16](16-splash-native-backend-spec.md)。

## 1. 用户提出的“新后端”落实在哪里

`native-metal-graph` 是显式的独立 executor identity，不是另一个 MLX fusion 开关。MLX 负责已有权重读取、文本编码、初始 noise 和 VAE；在 native begin 到最终 latent 之间，由应用自己的 Metal device／queue／PSO／CommandGraph 执行全部 32 层 DiT 和 Euler。

```text
现有 Request／make_plan／Session
  → 持有来源 fd，执行现有文本 encoder
  → 物化同一本地 BF16 DiT，产生同一 noise／sigma schedule
  → 明确的 MLX synchronize 边界
  → NativeSession：来源、phase Context、Backend、NativeDenoiser、Run
      → 输入／文本／时间调制
      → 首步完整 prefix + target，随后 target-only
      → 全部 32 层、output、BF16 Euler、finite／事务验收
  → 设备物理完成后复制最终 128 KiB latent 回 MLX
  → 销毁 native 图、buffer wrappers，阶段账本归零
  → 现有 VAE、PNG 导出、RunResult
```

native 去噪热路径不调用 MLX `matmul`、`fast::metal_kernel`、`compile` 或 `eval`。MLX Data 作为权重 backing 被保留不等于由 MLX 调度这些算子。最终小型复制是明确的阶段边界，不进行每层 activation 的 CPU 往返。

当前用控制性桥接读取原始权重，没有另存全模型 checkpoint，也没有同时建立两套完整权重。后续直接原生 loader 必须仍沿用同一 SourceLease／预算合同，不能改变这一点。

## 2. 请求域与默认兼容

实际字段定义于 [contracts.hpp](../../../native/core/contracts.hpp)，验证在 [native_route.hpp](../../../native/models/qwen21/native_route.hpp)，JSON 解析在 [request.mm](../../../native/platform/apple/request.mm)：

```json
{
  "model": "qwen-image-2.1",
  "operation": "image.generate",
  "prompt": "A blue teapot on a wooden table",
  "output": "result.png",
  "width": 512,
  "height": 512,
  "steps": 20,
  "frames": 1,
  "audio": false,
  "seed": 42,
  "execution": "gpu",
  "residency": "component_staged",
  "allow_approximation": true,
  "qwen21_dit_cache": "off",
  "qwen21_denoiser_backend": "native-metal-graph"
}
```

schema 2 放在 `execution.qwen21_denoiser_backend`；不放在 `parameters` 或 schema 2 顶层。省略字段选择 `mlx`，现有默认行为保持。

实验域：本地 Qwen21 dense BF16、512²、20 步、batch=1、base 生图。GPU 使用上述配置；private W1S 替换以下字段。示例显式选择上一构建测量的 register 实验变体，尺度缓存保持关闭：

```json
{
  "execution": "gpu_ane",
  "qwen21_w8a8": true,
  "qwen21_native_w8_staging": "register",
  "qwen21_native_scale_cache": false,
  "qwen21_native_ane_cache": "/absolute/owned/native-ane-program-cache"
}
```

这是完整 GPU 请求的字段替换示例，schema 2 对应字段在 `execution` 对象中。W1S 需 private 构建、`allow_approximation=true` 和明确、非根、无 dot traversal/NUL、长度有界的绝对 cache 路径；不能借旧 `ane_manifest` 或 `hybrid_mlp_mode` 选择该 executor。public 构建在准备前拒绝 W1S，native GPU 仍可用。

`qwen21_native_w8_staging` 仅取 `generic`／`register`，省略时保持 generic。schema 1 位于顶层，schema 2 位于 `execution` 对象；未知值／错误类型／错误层级拒绝。此字段只允许显式 private native W1S：即使显式写 `"generic"`，用于 GPU-only、MLX 或其他模型也拒绝。Request → NativeSession → NativeDenoiser → NativeHybridFfn → PreparedAneStages 传递同一个选择，成功回执报告实际 `weight_staging`，不能只复述请求或静默回退。当前只是 CLI／JSON 实验接口，App 默认与高级配置没有接入。

新增 `qwen21_native_scale_cache` 是**严格 boolean**，默认 `false`。只有 native private W1S 可以显式提供此字段；显式 `false` 也不能传给 GPU-only／MLX／其他模型，public 构建与 LoRA 请求继续拒绝。schema 1 顶层／schema 2 `execution` 的位置与 staging 一致，数字、字符串、null、错误层级均失败。`true` 启用当前请求的按层 compact FP16 W scales；与 `generic`／`register` staging 独立，不改其默认值，不创建磁盘缓存或跨请求复用。

register 只替换 W staging 的 BF16 Sylvester H128／H512 scale/code kernel，32-thread dispatch、额外 4 个 prepare PSO；A8／restore、权重来源、Surface 和分配标签不变。仅开启 register 时下述 W1S 38 sites／105 backings 不变；额外开启尺度缓存使用下述 39／137 计划；额外编译仍属于未认证 framework guard，不能从 buffer 数未增推断零冷启动成本。

接受 resident 或 component_staged，但不继承 MLX 逐层 streaming 估计。GPU 初始估计为 `22 GiB + canvas`；W1S 另加 2 GiB 保守路线估计，不沿用旧 frozen Core ML 的 10 GiB 项。此 planner heuristic 与后述 phase 的 1 GiB framework guard 不同，均非实测或认证上界。native DiT 期间持有完整 dense 权重。

LoRA、编辑、GGUF／ConvRot、MLX compile、VSA、近似 DiT cache、PE、完整请求 constrained memory、streaming 均不属于当前请求域。W8A8 仅上述 private W1S base recipe 可选；native GPU 不接受 W8A8/cache 字段。混入其他 executor 的诊断／capture 选项会明确失败；显式请求不会暗中回退或忽略参数。待后续按各自合同与实际验证扩域。

`allow_approximation=true` 是本实验 executor 的明确许可，不代表可以放宽有限值、语义、alias、source、生命周期或预算校验；也不声明内核与所有 BF16 输入逐位等价。

`prepare(false)` 仅沿用权重／来源准备语义，不提交 native 图、不声明 PSO warmup 或完整请求收益。完整 warmup 才实际执行原生去噪与 VAE。

## 3. 来源、可见性与寿命

[pipeline.cpp](../../../native/models/qwen21/pipeline.cpp) 捕获 DiT 的 held-fd SourceLease，以及文本 encoder／Tokenizer／VAE 的 conditioning lease。权重通过 `Weights::load_lease` 使用这些实际描述符，Tokenizer 使用重复 fd 读取。native 请求不复用来自旧普通加载路径、无 lease provenance 的 dense DiT 或文本缓存。

当前 lease revision 是文件描述符身份、stat／header 的受控身份，不把它描述为本轮新计算的 14／17 GiB 内容 SHA。原始文件不复制，source owner 保留至物理 drain；准备后、每步前后、最终输出与 VAE 后重新核对。来源发生变化使请求失败，不能发布或缓存混合版本结果。

import 只保留已物化、stride／offset／device 合法的 MLX Data owner，不隐式 `eval`、dequantize 或转换完整权重。native 开始前一次显式同步保证另一个队列的生产操作已完成。NativeSession 成员逆序销毁使 Run／model／Backend 先释放，phase 和来源最后释放。

尺度缓存的来源 key 使用 FFN 导入时真实 MLX `data_shared_ptr()` **control-block identity**，并包含 native handle、view offset／extent／shape／pitch／dtype、source revision／generation 以及完整 channel／basis／seed／block／norm／staging／recipe 身份。仅相同裸地址、stat、层号或新创建的 array wrapper 不能命中；gate/up 共享 fused backing 但 offset 不同。Data identity 证明 allocation 寿命身份，不独自证明数据不可变；当前可信 base Session 的持有来源、只读权重和每步 revalidate 合同共同建立复用前提。

每个 Run 有一个有界 cache，layer slot 内一份 gate/up/down entry 持有源 MatrixViews。冷路径为 scales→codes→gather compact；命中为 scatter 回当前 bank→codes，A8 每次重算。只有 controller 的 producer、ANE、GPU 补集、restore／join／finite 全部实际成功并物理完成后，Pending entry 才 commit 为 Valid；producer flags 通过或 event signal 均不够。controller 执行失败使全部 entry Retired，source／cache owners 保留至整个完成域 drain；不能借缓存恢复失败 Run。

最终复制要求完成状态成功、物理完成、run 的 final latent 可发布。复制后 latent 是新的小型 MLX backing；它归属显式 framework guard，不能虚构为已拥有 native StorageLease 的 buffer。

## 4. 一套账本，阶段证据与全请求资格分开

[MemoryBridge](../../../native/backends/metal_graph/memory_bridge.hpp) 把原生 buffer reserve／actual-capacity commit 接入既有 `MemoryExecutionContext`／`MemoryReservation`／`StorageLease`，不实现第二套计费器。

它按 exact label／site／class 预留，提交真实 Metal allocation capacity、domain、handle 和 generation；foreign／伪造 claim、错误线程和不匹配 import 被拒绝。多个 view 不重复计费同一 backing。close 清除 Context 指针，已创建 claim 自己保留现有 ledger shared state；晚到析构不解引用已经销毁的 Context。

[native_memory_plan.hpp/.cpp](../../../native/models/qwen21/native_memory_plan.hpp) 给 NativeSession 建立 **native DiT phase**：

- baseline 在 text、noise、schedule、完整 DiT 产生／物化、跨队列同步和 MLX cache clear 后测量；这些 retained sources 已包含在 baseline，不给匿名 MLX Data 装上假的单数组 claim。
- GPU 的 22 类 site／89 个 backing 按实际 shape、arena 及逐 backing 16 KiB capacity upper 编译。W1S 增 16 sites／16 backings，总计 38／105；9 个 Surface、2 个 FP32 partial、signs／zero／flags 共用同一 hooks。生产几何新增 capacity upper 122,798,080 B（117.109375 MiB），Surface 用真实 64-byte pitch；不是全模型 W8 常驻或 32×20 份状态。
- 尺度缓存默认关闭，W1S 仍是 38 sites／105 backings。开启时追加 `tc-native-hybrid-weight-scale-cache` 一个 site、32 个 layer backings，变为 **39 sites／137 backings**。每层 `(2×6144+4096)×2=32768 B`，总 payload **1 MiB**；每 backing 的 upper 为 `align16KiB(payload+16KiB)=49152 B`，总增量 **1.5 MiB**。实际只申请 payload，额外 16 KiB 是显式保守 capacity headroom，不是多分配或已测的驱动上界；实际 capacity 仍由同一 hooks charge，超上界失败。
- 默认 process budget 为物理内存减 4 GiB system reserve，另保守预留 1 GiB **未认证** framework guard；同时检查当时系统可用内存。这个 guard 是准入估计，不是测得的 framework 上界。
- `Request.memory_budget_bytes` 仍是原来的 denoiser hint，通过原 ResidencyPolicy 校验；不能偷偷改成整个 process cap。元数据 API 的独立显式 phase process cap 与该 legacy 字段不同。
- 如果已有 parent constrained-memory Context，明确拒绝嵌套。当前 Qwen 尚无完整请求认证，不能为了开启选项补一个虚假 capability record。
- 全部 buffer／wrapper 物理 drain 并释放后，才进入 release epoch、finish_success；storage／reservation／cache／pending 必须为零。来源 baseline owner 可继续由外层 Session 持有。

报告单列：

```json
{
  "native_denoiser_memory_phase": {
    "scope": "native_denoise_phase",
    "full_request_certified": false,
    "prepare_seconds": 0.0,
    "memory_admission": {},
    "memory_trace": []
  }
}
```

`prepare_seconds` 是 native 导入、PSO、begin 的 host 时间；不混入 denoise 后宣称稳态更快。既有 `memory.mlx_peak_bytes` 仍只属于 MLX allocator，不能与 phase process peak 混成同口径“显存”。阶段成功不认证 encoder、VAE、文件缓存、整个请求或某个小内存机器。

## 5. 失败与取消

每 step 只在完整图完成、finite 验收后提交 prefix／latent。NativeDenoiser 的失效 run 不因晚到 completion 恢复；状态查询／drain 只提供物理结束证据。

Session 异常尝试最多一秒 native drain。如果不完整，保留整个 NativeSession／import／source／phase owner，置 sticky quarantine，禁止 unload 或下一次请求。析构不是设备完成证明；绕过 Session 协议的析构仍不能释放可能被设备访问的全部执行 owner。

bridge close 不等于 buffer 物理结束。即使 caller 已退出，claim 仍保留 ledger；迟到释放不恢复发布资格。当前 W1S 聚合 producer、ANE、GPU complement、restore／join 与最终 GPU graph，不能用单个 GPU Ticket 冒充全部完成。未知物理状态保留完整执行 owner；当前不提供深窗口无缝 GPU replay。

成功路径在 finish 销毁 Run 前复制独立 `native_denoiser_hybrid` 收据，验收 608 FFN／19 steady steps 后才报告实际 W1 executor。producer／submit_ane／branch_join／consumer／total 是 host spans；另列 captured Metal ticket 的 `producer_gpu`／`gpu_complement` first-start／last-end spans 和 `timed_ffns`。后者只累加两段均有正数时间的成功 FFN，含 command-buffer 间隙，覆盖数不得超过成功数；不是 kernel busy time，不与 host spans 相加，也不证明 ANE overlap 或给出纯 ANE device 时长。`prepare(false)` 无执行完成计数，不从 requested 字段推断 ANE 已运行。

实际 hybrid 报告新增 `weight_scale_cache={enabled,entries,payload_bytes,hits,misses,scope}`。一个 entry／lookup 表示一层三份 W scales，不按三矩阵各计一次；只累计完整成功 FFN，后续整步失败仍可留下已完成分支诊断，不当作提交步数。首步全 GPU、后续 19×32 FFN 的成功完整运行**预计** 32 misses／576 hits；这是当前固定循环的预期计数，尚非此缓存的真实模型测量。

## 6. 历史请求证据与当前缓存验证

上一构建 `4fd1e980…` 的 native GPU／register W1S 请求均未开启尺度缓存，使用本地 BF16、512²、20 步、seed=42、blue teapot、component_staged、近似 DiT cache off。native 去噪／request wall 为 **41.616892042／48.602237417 s**，hybrid 为 **35.956603708／43.758467125 s**；相对省时 **13.60094%／9.96615%**。两图目视接近；hybrid 实际回执为 register、608 successful FFNs、19 completed hybrid steps，设备 span 覆盖 608 次。两路 native phase 均 succeeded，storage／reserved／cached／pending／unknown 归零，无 taint／quarantine；observed source baseline 仍由外层持有，不应要求整个 process ledger 数字为零。

4 个原始来源文件的 size／inode／device／mtime／ctime 前后未变；这是 stat 身份检查，不是重读大模型的内容 hash。外部采样的 swap used、Swapins／Swapouts 未增加；内置 phase 报告自身的 swap observation 不可用，不能冒充它已经提供该证明。原始完整回执及状态见[上一构建 Base E2E](../../status/2026-10-08-native-register-w1-base-e2e.md)。每路线只有 1 样本，固定 native→hybrid，各自新进程，系统 source／driver cache 未清；这些结果不授予 App、LoRA、编辑、Z-image、其他设备或完整请求内存资格。

后续顺序：

1. 保持 CPU manifest／MemoryBridge、strict v1／v2 请求、来源、alias 和失败隔离合同；Request ABI 变化仍需全库重建，不混链历史 runtime。
2. 保留上一构建原始 receipt／图片／二进制身份，历史 GPU 和组件结果独立引用；其 13.60%／9.97% 结果不授予当前尺度缓存收益。后续有限验证针对剩余顺序、冷热和代表性输入问题，不把此组测量改写成稳定倍率。
3. 当前请求内 scale reuse 已接线，先核对 CPU 来源／状态／预算合同与小型 GPU cold／hit 对照，结果见[尺度缓存状态](../../status/2026-10-08-native-request-scale-cache.md)。后续[真实缓存筛选](../../status/2026-10-08-native-scale-cache-real-screen.md)已通过真实 importer／精确复用／物理释放，但完整路径收益随顺序反转，默认关闭且未追加 E2E；不追加长时间扫参或全模型衍生导出。
4. Base B1 资格成立后，LoRA 与 W1R／B2 分别推进；之后按独立语义合同扩展编辑／encoder／Z-image。当前 W1S 的逐层 host admission 未消除，硬件 INT8 算术／真实 ANE placement 未验证。

App 默认与高级入口需等相应资格再改；新 JSON 实验入口不是已发布的用户配置。每次实验后只清理本轮拥有、可重建的 object／archive／module cache，不删除模型、既有 App、历史测量 binary 或未知系统缓存。
