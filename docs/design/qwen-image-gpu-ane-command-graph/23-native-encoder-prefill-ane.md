# Native encoder 与无 KV cache 编辑首步的 GPU / ANE FFN

2026-10-08，承接 [22：完整 Metal encoder](22-native-edit-encoder.md)。本轮实现 W1 有界并行执行，仍不是 Splash 的完整 W2 多层设备图。性能结果与构建身份见 [本轮状态](../../status/2026-10-08-qwen-native-prefill-ane.md)。

## 参考来源与具体落实

参考 [Splash PR 260](https://github.com/incoai/splash/pull/260)，并以本地 upstream `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7` 的 `runtime/ops/AneFfn.cpp` 交叉核对。采用的思想是：prefill 的 FFN 按中间通道拆分、两设备计算互补部分、FP32 合并、准备资源复用、设备提交后准备下一层源权重。没有链接 Splash runtime，也不迁移它的设备性能数字或强制 event 信号的失败处理。

TurboCider 的差异：H=4096 不符合该 upstream 实现的 H%2560 要求；这里沿用已有的 H128 输入旋转、H512 隐藏旋转及 private ANE 程序生成器。GPU 从原始 BF16 checkpoint 计算通道前半部分，ANE 使用后半部分的即时旋转 W8/A8 表示。FP16 surface/缩放和非线性不等于整个计算都是 INT8；没有硬件 INT8 placement 证书。

## 共享有界行适配器

实现位置：

- `native/backends/metal_graph/native_prefill_ffn.hpp/.mm`：一个物理 slot、一个固定 bucket、复用 scratch、GPU 补零/裁剪、多个 chunk 的同一绝对 deadline。
- `native_ane_stage_contract.hpp`：Qwen H4096/F12288；bucket 为 128..4096、128 对齐；C 仅允许 3072/6144/8192。允许形状不等于授予性能资格。已有 tiny H128/F1024/M32 用于测试。
- `native_hybrid_ffn.mm`：原来的 producer → ANE submit → GPU complement → restore/join 完成协议；新增提交两分支后的 host source preparation 回调。

| 调用点 | 本次真实有效行数 | 固定 bucket | chunk 数 | 每次额外计算行数 |
|---|---:|---:|---:|---:|
| 单图1024-reference语言 encoder | 1066 | 1152 | 1 | 86 |
| 512² edit，1024-reference，首步 | 5148 | 2688 | 2 | 228 |

有效行数依 prompt 实际 token 数变化；不得硬编码为表内数值。只给 FFN 补零，attention、mRoPE、DeepStack 和 image/text token 排列保持实际形状。A8 按 token 缩放；补零行在量化前写入精确零，结果只按有效行复制回模型。`ushort` 拷贝保持 BF16 位模式，padding 不作为 conditioning 发布。

ANE emitter 的独立上限为4224行；本适配器更保守地限制4096。不能仅放宽 A8 输入的 shape 检查：否则尾块会越界读取。较长 prefill 按 token 维分块，FFN 没有跨 token 依赖。

每个 chunk 的 GPU down partial 与 ANE restore partial 都为 FP32，合并后只在模型输出处舍入一次到 BF16。仍会对每个 chunk 重做 W8 bank staging；尚未实现同一层跨 chunk 的 bank code 复用。

## 完成与资源合同

1. 校验 source、output、完整 pitch、logical extent 和 checkpoint dtype；拒绝 output 与 input/checkpoint 共用 backing。
2. 记录前图与补零，写入 W8 bank、A8 input；完成 producer 并校验 status。
3. 提交 ANE 后再提交 GPU complement。
4. 允许 host 同时准备下一层只读 checkpoint 窗口。encoder 最多当前+下一层，不持有完整 W8 权重副本。
5. 两分支真实完成后执行 restore、FP32 join、裁剪与后图。
6. 全部 chunk 成功后才返回输出。超时、非有限值、源准备失败永久 retire 本 slot；不伪造 event 完成、不复活结果。

源准备回调的异常先被保存，待两分支和 consumer 真实完成再退役。测试注入 next-source 失败，验证 retired 且 all_work_physically_complete。未完成的旧 slot 不可复用；已有 ticket/handoff owner 保留必要的 live resources。

## 完整 Metal encoder 接入

`NativeEncoder::language` 的普通 GPU 路径保持两层一图；显式 ANE 候选每层拆成前图、并行 FFN 和后图。视觉27层继续走完整 Native Metal GPU，不宣称视觉已在 ANE 上执行。

- 默认开关仍为 `TURBOCIDER_QWEN21_ENCODER_BACKEND=mlx`。
- Metal GPU：设置为 `metal`。
- Native encoder W1：额外设置 `TURBOCIDER_QWEN21_NATIVE_ENCODER_ANE_CACHE` 为明确拥有的绝对缓存目录。
- 当前 ANE candidate 限定有效行1..1152、H4096/F12288、C6144；其它形状拒绝显式候选，不静默丢 token。
- Session 内 NativeEncoder 保存 operator/program/scratch；每次重置计数、使用 fresh source 权重，不复用 KV 或完整 conditioning 来假装 encoder 加速。

前图包含 native attention、norm 与 FFN input；后图保留 residual、DeepStack 和最后层输出。ANE 执行时准备下一层源权重。首次准备计入 phase；复用结果独立记录。`dispatches` 是 encoder 显式记录的 model dispatch 计数，不包含内部 W8/A8、finite、pad/crop dispatch，ANE calls/chunks 另外报告。

## 编辑首步接入

`Session::run` 使用已有 `Transformer::set_prefill_mlp` 边界，执行真正的无 KV cache 的 step0 FFN。attention、modulation、norm、residual、Euler 与 VAE 仍沿用 MLX；这是一个 native FFN island，不是完整 native edit DiT 后端。

设置 `TURBOCIDER_QWEN21_NATIVE_PREFILL_ANE_CACHE` 的显式候选要求：512² output、单参考图、`component_staged`/支持的 GPU edit、allow_approximation、BaseOnly、普通 MLX DiT、无其它 FFN/QKV 诊断、无 resident prefix/snapshot hit、无近似 DiT cache。

首步32层全部执行后报告 `native_prefill_execution`。step1 开始清除该 callback，后续维持既有 GPU/KV 路径。不能在 LoRA 请求中直接使用原始 gate/up/down 而省略 adapter：当前候选明确拒绝 LoRA，普通 GPU LoRA 路径保持可用。

冷启动负结果后，prefill operator 已整理为 `NativePrefill` Session owner，可通过 `Session::prewarm_native_prefill(absolute_cache)` 预热；重复调用不重建。输出分别报告 prewarm、request prepare 与 `preparation_reused`。API 不生成或缓存 conditioning/KV；尚未接 App UI 自动预热。随后才是双 bank 预取，当前不是 W2。

## 验证与后续门槛

使用本地 BF16 checkpoint、teapot reference，零下载、零完整模型导出。先 generated 短行/尾块/多块/变权重，再真实 encoder layer0 与编辑step0 layer0的完整 FFN。C3072/C6144 各一 warm、AB/BA 两顺序。预声明 FFN RRMS≤5%、完整 FFN 平均省时>5%；通过后才进行完整 phase。

完整 encoder 要执行视觉27层＋语言36层，conditioning RRMS≤8%，同时比较首次与准备复用。编辑 E2E 仅单步 base 诊断，检查32层/64chunks、图像输出、首步/请求耗时；不是可用画质认证。单样本、未清系统/driver cache，不外推稳定性能。

后续按完整路径证据决定：

1. 已加入 Session 复用和显式预热；继续区分首次全成本、系统编译缓存和用户等待时间。
2. 同一层多 chunk 复用完成且源绑定的 W8 bank，避免重复量化；不得只靠指针相等判断来源不变。
3. 真正的 W2 bank0/bank1，未来 W staging 与当前 ANE 重叠；加入资源冲突图、不同 generation、取消和失败注入。
4. 针对 encoder 与 prefill 各自校准，在真实 base/LoRA FFN 输入上比较 SmoothQuant 与旋转；不能搬用 DiT steady-state 校准。
5. 最后才扩大到 held-out prompt/reference、LoRA 修正与 App 高级配置。自动策略维持 GPU，避免仅凭组件成绩打开全局 ANE。
