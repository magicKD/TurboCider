# 09 · 首发范围、组件映射与执行精度合同

[目录](README.md) · [执行接口](10-execution-and-product-integration.md) · [验收配置](11-acceptance-profiles-and-feasibility.md)

2026-09-30 审阅补充。状态：**待实现的规范性设计**，不是现有 API、模型支持或认证声明。
本页把 02–07 的候选收敛为逐阶段合同；本页与 10、11 对首发范围的限定优先于前文的
概览/示例。既有生产 streaming 的安全规则不被覆盖。设计约定变更必须更新 revision；
不允许为了让某次实验通过而事后更换 profile。

## 1. 发布单元和实现顺序

`R*` 是交付里程碑，`P*/C*/A*/G*` 仍是 07 的代码工作包，两者不是同一编号体系。

| 里程碑 | 必须交付 | 本阶段明确不承诺 | 对应工作包 |
| --- | --- | --- | --- |
| R0 合同与基线 | fixture binding、dtype trace、纯 CPU oracle、配置/plan 负例、验收 profile 冻结 | 新格式生成或性能提升 | P0、P4 合同部分 |
| R1 Q8 最小纵切 | 一个 Z-Image Turbo DiT Q8_0 + 浮点 mixed 文件；CPU 直填；先 p=0 单槽再 p=1 双槽；GPU-only 完整请求 | encoder GGUF、K/IQ、p=2、tile、ANE、LoRA | P1a/P1b 子集、P2、P3a、P4 实验接入 |
| R2 格式与 encoder | Q4_0/1、Q5_0/1、Q4_K/Q5_K/Q6_K；Qwen3-4B conditioning；encoder-only/DiT-only/组合三组 | 所有 Unsloth 架构、IQ4、视频 | P1b 扩展、P3a/P3b |
| R3 更紧预算 | p=2、packed-streamed、projection/N-tile、完整阶段账本；正式入口的已认证 tuple | 所有预算均能执行、请求内改布局 | P2 扩展、P4 发布部分 |
| R4 ConvRot/异构浮点 | shared H、butterfly、两种有界展开独立比较；runtime FP16 直填 | W8A8、必然超过 GPU | C1/C2、A0 |
| R5 条件性 INT8 | A8 oracle；静态 ANE、M5 GPU、动态 ANE 各自完成可行性判定，再决定集成 | 无实机认证；保证动态 ANE INT8 可用或更快 | C3、A1/A2、G1；成功后 A3/G2 |

R1 的 source 固定为 `packed_resident`：保留原始压缩 payload，不能同时常驻整套
MLX repack 副本。若压缩数据加 encoder/VAE/激活的阶段峰值仍放不下，R1 必须拒绝，
不能声称支持该预算；R3 才增加 packed-streamed。R1 的“一层”包含该 block 的
attention、FFN、modulation 等全部所需字段，不是只计算 FFN 的 225 MiB。

R1 使用现有 safetensors encoder/VAE，但在整体预算中照常计费；encoder 完成后在
消费者独立持有 conditioning 的前提下释放其权重。没有 encoder streaming 不等于
encoder 内存为零。固定小张量与少数非循环矩阵单列 resident 字段，不藏在 baseline 中。

R1 可以返回 `qe_budget_floor`，暂不接受装不下一整层的配置。这是阶段性不支持，
不是 03 中 tile 目标已经完成。R2 格式兼容不等待 R5。IQ4 和其他 encoder 各建后续
milestone，不把“文件能解析”标记为模型可生成。

## 2. Fixture 身份与本次只读目录核对

本次仅解析本地 GGUF header/tensor directory，未解码 payload、未全文件哈希、未生成。
下表是找到的候选，不是完成内容校验的 golden；路径只用于定位，文件名/大小不能授予资格。

| 候选文件（相对仓库） | bytes / tensor 数 | 目录事实 |
| --- | --- | --- |
| `models/z-image-runtime-gguf-q8/z_image_turbo-Q8_0.gguf` | 7,224,707,136 / 453 | v3；F32=245、BF16=28、Q8_0=180；`general.architecture=lumina2` |
| `models/z-image-runtime-gguf-q4-main/z_image_turbo-Q4_0.gguf` | 4,509,425,472 / 453 | v3；BF16=273、Q4_0=180；无 metadata；pad token 为 rank 1 |

两个文件的小 tensor dtype 不同，不能直接拿它们的速度差解释成 Q4 vs Q8 kernel 差异。
Q8 的 pad token 为 `[1,3840]`，Q4 为 `[3840]`，adapter 仅允许下面列出的这一等价 reshape。

每个 campaign 必须绑定以下清单，再计算 canonical profile digest：

1. DiT/encoder/VAE 的全部文件 content SHA-256、字节数、split 集合和 per-type histogram。
2. 原始 BF16 对照、source-quant 对照及量化来源；不知道共同 base 时明确不可作同源画质结论。
3. tokenizer/config、prompt template、mapping/precision/decoder revision、oracle commit。
4. 当前源码与二进制身份、设备/OS/库、完整实际 request、layout、原始噪声身份。

使用现有 `SourceLease::capture_verified` / `capture_preverified` 的原生证明机制。
首次内容校验的 I/O、时间和内存要单列；目录 digest 不能替代 payload digest。
只持有 fd 也不能防止原地修改：继续执行现有 open-file/path/generation revalidation。

本轮没有取得 Qwen3 encoder GGUF 或 K/IQ 真实 fixture 的 binding；其验收状态是
`not_run`，R0 可先为 R1 完成 binding，其余绑定在对应 milestone 前完成。模板里的
null hash 是阻止误认证的占位，不允许 verifier 自动补成通过。

## 3. Z DiT adapter：`z-comfy-gguf-v1`

### 3.1 架构和命名

首个 adapter 接受核对过的 Comfy 风格 tensor namespace，不做通用的前缀猜测删除。
`lumina2` 仅作为该 Z checkpoint 的已知 metadata 标签，不能据此加载任意 Lumina2。
识别必须同时通过模型绑定、下面的几何和完整 required-key 集合。

无架构 metadata 的 Q4 本地样本仅能经显式 legacy import、完整内容 binding 和
`z-comfy-gguf-v1` 映射验证进入实验。不能通过“缺 metadata 就按 Z 猜”开放生产导入。
额外的 Diffusers 别名/分离 QKV 是独立 mapping revision，不属于 R1。

GGUF `ne[0]` 是连续输入维；二维磁盘 dims `[K,N]` 对应内部逻辑 `[N,K]`。
反转 shape 描述不等于转置 payload；只在 recipe 明确要求时重排数据。

| 源 key / 内部 key（R1 同名） | 内部逻辑 shape | 约束 |
| --- | --- | --- |
| `layers.{0..29}.attention.qkv.weight` | `[11520,3840]` | 按 Q、K、V 三段，每段 3840；30 heads × 128 |
| 同 block `attention.out.weight` | `[3840,3840]` | 不转置重解释 |
| 同 block `feed_forward.w1/w3.weight` | `[10240,3840]` | w1=gate，w3=up |
| 同 block `feed_forward.w2.weight` | `[3840,10240]` | down |
| 同 block `adaLN_modulation.0.weight/bias` | `[15360,256]` / `[15360]` | 4×H；bias 必须保留 |
| 同 block `attention.q_norm/k_norm.weight` | `[128]` | head-local norm |
| 同 block `attention_norm1/2.weight`、`ffn_norm1/2.weight` | `[3840]` | 四份 norm，不能漏 |
| `noise_refiner.{0,1}.*` | 对应 main block shape | 有 modulation，分别作为可流式 block |
| `context_refiner.{0,1}.*` | 对应 attention/FFN/norm shape | 无 main-block modulation；独立 layout class |
| `x_embedder.weight/bias` | `[3840,64]` / `[3840]` | stage resident |
| `cap_embedder.0.weight`、`.1.weight/bias` | `[2560]`、`[3840,2560]` / `[3840]` | 连接 encoder 的 2560 hidden |
| `t_embedder.mlp.0.weight/bias`、`.2.weight/bias` | `[1024,256]` / `[1024]`、`[256,1024]` / `[256]` | 原 timestep 路线不变 |
| `final_layer.adaLN_modulation.1.weight/bias` | `[3840,256]` / `[3840]` | final stage |
| `final_layer.linear.weight/bias` | `[64,3840]` / `[64]` | 输出 patch |
| `x_pad_token`、`cap_pad_token` | `[3840]` 或 `[1,3840]` | canonical `[1,3840]`，仅 reshape |

真实计算顺序服从 `z_image.cpp`：context refiners 与 image/noise refiners 不能因表中
顺序而重排。主层 30 个，加两个 noise 和两个 context refiner，不能将现有 32 个
hybrid block ID 误当作文件只有 32 个 transformer block。

缺必需 tensor、重复 alias、多余未解释 tensor 或 shape 不符均拒绝；可忽略项必须列入
adapter allowlist 和 report。逐 tensor type 均通过 parser/registry，未使用不等于允许损坏目录。

### 3.2 后续 QKV assembly

若新增分离 Q/K/V 的 adapter，decode task 直接写目标 QKV backing 的三个已证明
不相交的 row spans，全部成功后发布 Ready。禁止先分配 Q/K/V 三张 dense 再 concatenate
出第四张而不记峰值。Q/K permutation 在读取 recipe 中显式编码；shape 相同不代表布局相同。

## 4. Qwen3 conditioning adapter：`qwen3-z-gguf-v1`

只接标准 dense Qwen3-4B 文本 backbone；MoE、Next、3.5、VL、embedding/reranker
不因名字含 Qwen 而放行。配置取自绑定的原始模型 config，与 GGUF metadata 双向核对；
不根据 GGUF 覆盖当前模型的 heads/RoPE/epsilon 以“使其能够运行”。

首个 consumer 固定现有 `Qwen3Conditioning::z_image()`：heads=32、kv_heads=8、
head_dim=128、rope_theta=1,000,000、epsilon=1e-6、hidden=2560。
FFN 宽度 F、vocab V、模型总层数由绑定 config 验证，至少存在 block 0…34。
运行 block 0…34，在 **block 34 residual 完成后**取值，不经过模型 final norm；
residual/RoPE 保持现有 FP32 合同，最终 conditioning 转 BF16。不是 HF hidden-state
列表的任意“第 34 项”，不执行 lm_head，不推理 block 35 来替代这个 tap。

| GGUF 名称 | 内部名称 | shape |
| --- | --- | --- |
| `token_embd.weight` | `model.embed_tokens.weight` 的 source handle | `[V,2560]` |
| `blk.{i}.attn_norm.weight` | `model.layers.{i}.input_layernorm.weight` | `[2560]` |
| `blk.{i}.attn_q.weight` | 同层 `self_attn.q_proj.weight` | `[4096,2560]` |
| `blk.{i}.attn_k/v.weight` | 同层 `self_attn.k_proj/v_proj.weight` | `[1024,2560]` |
| `blk.{i}.attn_q_norm/k_norm.weight` | 同层 `self_attn.q_norm/k_norm.weight` | `[128]` |
| `blk.{i}.attn_output.weight` | 同层 `self_attn.o_proj.weight` | `[2560,4096]` |
| `blk.{i}.ffn_norm.weight` | 同层 `post_attention_layernorm.weight` | `[2560]` |
| `blk.{i}.ffn_gate/up.weight` | 同层 `mlp.gate_proj/up_proj.weight` | `[F,2560]` |
| `blk.{i}.ffn_down.weight` | 同层 `mlp.down_proj.weight` | `[2560,F]` |
| `output_norm.weight`、`output.weight` | 本 consumer 不绑定计算 | 若存在仍做目录/type/shape 安全验证 |

命名参考为 02 固定的 GGML 快照及其 `conversion/qwen.py`、`gguf-py/gguf/tensor_mapping.py`。
普通 Qwen3 converter 路线没有此处需要自行添加的 Llama Q/K permutation；recipe 首版为
`qk_layout=hf_half_split_v1`，仍须由 Q/K 输出、RoPE 后结果与逐层 hidden fixture 验证。
其他来源 exporter 必须声明并验证自己的变换，不能照名字应用同一个 recipe。

embedding 只按有效 token IDs gather，保持 token 顺序/重复次数；可以去重读取，但
需要有界去重表与恢复索引。不得为取 `.shape(1)` 继续要求整张 dense embedding：
模型从逻辑 source descriptor 取 hidden，`mx::take(full_dense_embedding)` 改成 adapter gather。

继续使用 Z tokenizer、`z_image_prompt`、特殊 token、padding 与 causal mask。
GGUF chat template 不自动覆盖图像 conditioning 模板；tokenizer 不兼容直接拒绝。
Flux2 的 `{8,17,26}` tap 和拼接需另一 consumer identity，不继承本资格。

## 5. 精度 profile：存储相同不等于执行相同

| 对象 | `z-source-mixed-v1`（R1 正确性起点） | `z-dense-bf16-v1`（独立优化候选） | `qwen3-z-source-mixed-v1` |
| --- | --- | --- | --- |
| 量化权重 | 原 codes/scales，FP32 重建后 RNE 一次写 BF16 | 同左 | 同左；不把 encoder residual 改成 BF16 |
| 文件内浮点权重/norm/bias | 保留源 dtype | Z 执行浮点字段显式 RNE 转 BF16，逐字段记录 | 保留源 dtype |
| GEMM 输入/输出 | 原表达式及真实 dtype promotion；trace 记录 | 支持段的输入/权重/输出 BF16 | 按现有 FP32 residual 表达式，允许 FP32 GEMM |
| norm 累加 | 现有 FP32 累加、结果转回输入 dtype | 同左 | FP32 residual 对应 FP32 输出 |
| residual/SiLU | 保持原算子边界，不额外降精度 | BF16 段须保持声明的边界 | 保持 FP32 residual |
| 图选择 | eager/兼容分段，不声称已恢复整体 fusion | 新验证的 BF16 compiled segment | 首版逐层提交，不套 Z DiT 图 |

`source-mixed` 指小 tensor 保留源 dtype，不承诺与旧 packed QMM bit-exact。
量化权重转换到 BF16 本身有一次目标 dtype 舍入；它的 execution oracle 是同源解码、
同 profile 的 dense 路线。旧 QMM 和原 BF16 模型另作画质/性能对照。

现有 `z_block` 的 `bf16_graph` 同时检查 bias/norm 等字段，因此首版不自动放宽它。
`z-dense-bf16-v1` 对原 FP32 小 tensor 的转换是显式数值变更，须
`allow_approximation=true` 和独立质量资格；不是 decoder 的隐形优化。
另一候选是保留 FP32 小 tensor 的 compiled mixed segment，需独立 profile，不伪装 homogeneous BF16。

decoder 的 BF16/FP16 转换统一 round-to-nearest-ties-to-even；不允许 flush/clamp
掩盖有限值溢出，CPU/GPU 实现均对照 bit pattern。模型 mask 中约定的 `-Inf` 不属于
非法权重/输出；非有限 weights/scales、logits 异常或非有限生成结果仍失败。

ConvRot 分别命名 `convrot-source-f32-scale-v1` 与 `convrot-legacy-packed-bf16-scale-v1`，
避免把旧打包 scale 舍入和新算子误差混在一起。ANE FP16 headroom、A8、逆旋转和
requantization 各自增加 recipe；只改后端也不能跨 profile 继承质量资格。

## 6. Definition of Done

每个 milestone 必须同时交付实现、正/负 fixture、可重放命令、原始 receipt 和状态更新。
R1 通过不表示 R2/R3 完成；R5 的可行性否定结论可以关闭研究工作包，但对应功能仍是
unsupported。没有真实 fixture/设备的项明确列依赖，不能用合成数据代替产品完成条件。

上述合同可以开始 R0 与 R1 子集；绑定文件、阈值和计划是对应 campaign 的启动条件，
不是要求先完成所有未来架构才能开发。下一步最小 PR 应只建立 Q8/F32/F16/BF16 oracle
和固定槽单层测试，再穿透到完整请求，之后扩格式。
