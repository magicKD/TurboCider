# 11 · 固定验收配置、判定器与硬件可行性退出

[目录](README.md) · [范围/精度](09-release-scope-and-component-contracts.md) · [执行合同](10-execution-and-product-integration.md)

2026-09-30。状态：待实现的验收规范，**本页阈值是工程决策，不是测得的精度或速度保证**。
将 07 的广覆盖测试拆成可判定 profile；既有 08 回执仍仅代表它实际测过的组件。
下述 runner/verifier 名称均是计划新增，除明确写“已有”的命令外不可当作现成工具执行。

## 1. Campaign 冻结和完成状态

每次 campaign 在看 candidate 结果前固定：fixture content identity、参考路径、
precision/transform recipe、输入、设备 tuple、预算、布局、阈值、样本顺序、失败规则。
绑定后保存 canonical JSON digest；工具拒绝 null、TBD、未识别字段或运行中变更。

开始开发允许使用未绑定模板；它不能用作资格凭证。本轮提供
[Z Q8 模板](validation/z-q8-qualification-template.json)，其中 source hashes、设备、预算、
噪声与二进制身份尚未绑定，`binding_state=unbound`，没有伪造 PASS。

binding 工具先填齐所有必需字段，再置 `binding_state=bound`、`executable=true`
（仅表示 campaign runner 可运行，不是产品 generate 资格）。digest 对去除自身
`profile_digest` 字段后的 canonical JSON 计算；之后 profile 不再改动，所有 verdict
写入独立 receipt。模板的 `qualification_status=not_run` 仅表示创建时未运行，不能
在签名后的 profile 内改成 pass。路径用于本地解析，稳定身份使用相对逻辑 ID 和内容 hash。

结果分别保存：

| 字段 | 可选状态 | 判定 |
| --- | --- | --- |
| correctness / quality / managed_memory | pass / fail / inconclusive / not_run | 任一必需项非 pass，不能资格化对应功能 |
| whole_request_memory | layout_validated / bounded_certified / unknown / fail | managed 槽有界不自动变为 whole-request hard cap |
| performance | accelerated / low_memory_only / regression / inconclusive / not_run | 更省内存但更慢可以是明确 fallback，不叫加速 |
| placement / arithmetic | evidenced / unknown / contradicted / not_run | 分开判断，QDQ 不填入硬件 arithmetic proof |
| feasibility | supported / unsupported_on_tuple / inconclusive / not_run | 负结果可完成研究调查，不完成加速功能 |

正式生成资格、INT8 宣称和默认开启是不同决策。没有 M5/真实低容量机器/对应模型时
是 not_run；选定 runner 缺依赖或证据损坏则 inconclusive/失败，不把 required test 的 skip 当 pass。

## 2. Oracle 和可执行数值判定

### 2.1 三种参考必须同时标识

- O0：固定 GGML commit 的 CPU 解码 oracle，原 codes/scales；禁止用被测 decoder 自证。
- O1：O0 解码并按 09 的目标 dtype 舍入后，执行同 profile 的 dense 参考；可逐层运行，
  不要求在低内存设备常驻全模型。它隔离 pager/布局/调度错误。
- O2：旧 source-quant GPU 路线及同源原 BF16 模型，用于额外数值变化与源量化损失。
  没有相应来源证明时该项 not_run，不能据 O1 正确推导“与原 BF16 画质一致”。

R2 若旧 native 不支持某个 K/IQ type，可绑定独立参考实现的 source-quant golden，
记录其模型语义/精度和来源；它只用于离线验收，不成为产品静默 fallback。无可信
source-quant golden 时仍可开发和做 D0/D1，但不能标记该组件质量资格完成。

所有 tensor 比较按相同有效元素和逻辑轴；padding 不纳入，但 padding 写入范围单独测试。
以 FP64 离线计算 `relL2 = ||a-b||2 / max(||b||2,1e-12)`、cosine 和
`max_norm_error = max(abs(a-b)) / max(max(abs(b)),1e-6)`。
零参考 tensor 单独检查 `max_abs_error<=1e-7`，不依靠相对误差掩盖异常；两者全零 cosine=1。
记录每个 tensor 的结果，不能平均掉失败的某层/step/文本 rows。

| Profile | 硬判定 | 用途 |
| --- | --- | --- |
| D0 source decode | 与固定 scalar O0 FP32 bit-exact，再与目标 RNE bit-exact；±0 按声明 canonicalize | 所有 GGUF types 的第一门 |
| D1 schedule-only | 同 O1 运算顺序下，层输出、conditioning、最终 latent bit-exact | 单槽/双槽、reader 生命周期；先不改 fusion |
| N1 float transform | 每层 relL2≤0.01、cosine≥0.9995、max_norm_error≤0.05；最终 latent relL2≤0.03、cosine≥0.999 | BF16 小字段转换、butterfly、tile/归约或融合改变 |
| N2 added A8 | 每投影/FFN relL2≤0.05、cosine≥0.995、max_norm_error≤0.15；最终 latent relL2≤0.10、cosine≥0.99 | 在同源量化模型上新增 A8，不代替媒体门 |
| Encoder extra approximation | 每个返回 tap/最终 conditioning relL2≤0.02、cosine≥0.999 | N1/N2 若改 encoder，附加而非替代上述检查 |

N1/N2 是首轮冻结的项目准入阈值，未以本机实验声称其可达。失败保留失败结论；修改
策略必须新 revision、新 campaign，并重跑所有候选，不能只为一个 after 放宽。
不同低比特源相对原 BF16 的差异另报告，不用 N1 的实现容差否定源本来已存在的量化误差。

R1 用 D0/D1 证明实现正确；由于 BF16 目标重建可能不同于旧 packed QMM 的实际
算术，还须对旧 source-quant 执行 N1 和媒体门，不能只比较自己选择的 O1 就宣布
与既有输出一致。原 BF16 对照用于区分源量化损失；不要求 GGUF 恢复未量化模型。

D0 使用固定 reference compiler flags（关闭 fast-math 和未声明 FMA contraction），保存
oracle 二进制 hash。优化 decoder 无法通过时修改运算序列，而非拿画质过关替代 block 正确。
D1 先运行三次同 reference 证明目标 tuple 的重复性；reference 自身不稳定则调查或
显式另立数值 profile，本次 D1 inconclusive，不能自动切换宽松门槛。

### 2.2 INT8 数学与归一化 ANE 的不同 oracle

ConvRot per-row A8 首版固定：在所选 H recipe 输出后以 FP32 求 maxabs，
`sx=maxabs/127`；全零 row 的 sx=1、codes=0；RNE 后 clamp 到 [-127,127]。
权重保留原 signed [-128,127] 与 FP32 sw。非有限值失败；不隐式 outlier fallback。
static clipping/group A8 各自新 recipe，不与这个 oracle 混用。

用 INT64 reference 检查整数点积，再证明 plan 的每段上界可装入 INT32；整数结果要求
exact。M5 epilogue 的基准固定为 INT32→FP32、依次乘 sx/sw、加 bias、一次目标转换；
输出距同顺序 reference ≤1 个目标 dtype ULP。±0 等价，其余按单调 bit 编码计算 ULP。
不得用 N2 的模型误差容忍 kernel 解码/点积错误。

06 的归一化 ANE 候选若通过 FP16 输出 `C=I/16384`，该出口已经损失部分整数精度，
不能再要求从 C 恢复 exact I，也不能与 FP32 GPU epilogue 冒充相同算术。
单独 reference：`I64 → FP32(I/16384) → RNE_FP16 → FP32 scale restoration`，
对 C 检查 ≤2 FP16 ULP，并同时报告恢复结果相对未舍入整数 oracle 的误差。
scale restoration 次序固定为先在 FP32 计算 `sx*sw`，再乘 16384 与 C，有限值范围单独测；
gate/up 必须先恢复再 SiLU。若 exporter 的合法 precision 合同不同，在看结果前换 recipe。

该 oracle 的数值正确不证明硬件 INT8；还需第 7 节。若额外 FP16 出口误差不通过 N2，
候选失败，不能以“INT8 本来就近似”解释出口 bug 或过大的损失。

## 3. 固定输入和媒体质量门

R1 campaign：Z-Image Turbo，image.generate，4 steps，512×512 与 1024×1024，
seeds=`[42,1234,20260930]`，共 8 prompts×3 seeds×2 shapes=48 配对输出。
采样器、scheduler、guidance、dynamic_text=true、VAE 与输出处理固定为绑定 request 的
完整 resolved 值，不能靠未记录默认。保留 token IDs、实际/有效/padded rows、噪声 hash。

| ID | 固定 prompt（UTF-8 原文） | 审核关注 |
| --- | --- | --- |
| portrait | `A studio photograph of an adult ceramic artist, both hands visible while holding a small blue cup, neutral background, natural skin texture.` | 人像、双手、杯子 |
| text | `A straight-on photograph of a small bookstore entrance with a clearly readable sign saying OPEN TODAY, daylight, no other signs.` | 文字、符号，不把 OCR 分数等同完整语义 |
| glass | `A clear glass pitcher half filled with water beside a red apple on a wooden table, daylight passing through the glass, realistic refraction.` | 透明、反射 |
| geometry | `An architectural photograph of a hallway with repeating black and white floor tiles and evenly spaced square windows, strong perspective.` | 重复纹理、结构 |
| objects | `Five distinct objects on a plain table: a red ball, a blue cube, a yellow mug, a green book, and a silver spoon, all fully visible.` | 多物体、颜色/关系 |
| long | `A wide photograph of a quiet coastal workshop at sunrise. In the foreground an adult craftsperson repairs a wooden model boat. A blue notebook and three small brass tools lie to the left. A sleeping orange cat rests under the table. Behind the table, tall windows reveal a harbor with two fishing boats and a distant lighthouse. Shelves contain folded cream fabric and plain ceramic bowls. Warm sunlight enters from the right, casting long shadows across the stone floor. Preserve the spatial arrangement, distinguish every material, and keep the scene natural and detailed without decorative text.` | 关系、长文本；不能声称覆盖所有最大长度 |
| dark | `A night photograph of a cyclist waiting beside a wet street, a small warm shop light and distant cool streetlights, detailed dark clothing and reflections.` | 暗部、动态范围 |
| colors | `A close-up photograph of layered translucent colored paper arranged in curved bands, smooth gradients, fine edges, soft diffuse light.` | 渐变、边缘 |

encoder 另测 valid token 长度边界和模型最大长度，固定 token IDs fixture，不通过偷偷
截断来满足预算；非法超长输入必须按原产品语义拒绝。上述 long prompt 不替代该边界测试。

输出保存无损图片、latent、逐层统计与文件 hash。实现错误先判 fail；N1/N2 媒体额外
采用盲化 A/B，两名审核者分别记录新出现的语义/结构/文字/伪影问题，不展示后端名：

- 任一确认的新增严重问题（主体缺失、显著结构破坏、语义关系错乱、崩坏/空图）失败。
- 新增轻微退化的配对比例不得超过 5%（48 张最多 2 张）；其余为持平或更好。
- 意见不一致由第三人裁决；未完成人工记录是 inconclusive，不能仅 PSNR 自动通过。
- source-quant 对原 BF16 的既有缺陷单列，不计为 candidate 新错误，也不能隐去。

这是一份可操作的初始质量政策，不是图像质量存在通用单指标的断言。D1 exact 的新增
slot 数只需对已完成质量审核的同 profile golden 确认输出身份，不必重复盲测相同图片；
首次建立该 profile 仍必须完成 source-quant 对照。
视频/其他模型需要独立 profile，本页不授予其资格。

静态 A8/SmoothQuant 的 calibration 与上述 48 个 holdout case 隔离。另绑定至少
16 个不同 prompts、每个 2 seeds、两个目标 shape、所有 4 steps 的校准清单，分别
收集 caption/image 和 gate/up/hidden/down 范围；数据 hash 与量化参数一起保存。
不能用 holdout 调 clipping/SmoothQuant 后仍把它作为独立验收集。dynamic per-row
A8 不需要训练 scale，但同样需 holdout/outlier 评估；校准/捕获内存也须有界。

## 4. 内存预算与负例

campaign 绑定现有 guard 的 Y/X/S，得到 `B=floor(Y*(100-X)/100)`；模板中的 null
预算禁止运行正式认证。预算取值在看 after 前固定，比较 arms 使用相同 B/系统余量。
每种布局登记完整 `planned_upper`、managed/retained/pending bytes 和未知 envelope。

R1 正例覆盖 p=0、p=1；R3 加 p=2、projection-tile、packed-streamed。负例包括：
目标 slot 容量比实际字段需求少一个字节、packed source 超预算、VAE 峰值超预算、
layout-class 切换峰值、ANE recovery floor 不足。负例通过意味着正确拒绝，不意味着生成通过。

独立 memory campaign 从 prepare/首次校验→load→encoder→DiT→VAE→export→release
开始采样，目标采样间隔 10 ms，报告实际最大 gap；超过 50 ms 的 run 不授予采样证据资格。
高频采样仍不是连续时间上界证明，managed allocation trace 与 required-site closure
不可省略。observer 不与正式计时混用；两种 run 的 plan/binary/workload digest 必须一致。

检查所有唯一 backing，dense capacity 不超过计划字段容量之和；瞬时额外整层分配
即失败，不能只看最终 cache。header-only scan、首次内容哈希、CPU buffers、MLX
retained cache、Core ML/worker 和恢复路径都在范围内。mapped/shared 页说明统计口径，
不直接把父子 RSS 相加当唯一物理量；不可观测 driver 项记录 unknown。

有 required unknown upper 就不能写 bounded_certified，可保留 layout_validated。
guard 下 swapout 增长按既有规则失败；确有外部噪声则整 run inconclusive，原样保留。
大容量机限制 B 只认证该计划，不冒充真实小容量硬件。释放后仍 retained 的对象不能
标记为已归还预算；session retention 尚未资格时不得跨请求保留 dense bank。

## 5. 性能采样与推广门

功能兼容没有“必须更快”的通过条件；`accelerated` 才要求下面所有条件。基线必须是
同 source checkpoint、同组件输入、同预算且已优化的 GPU 路线。比较 A8 等新增近似时
只允许 manifest 声明的 math/profile 差异，先通过质量门；不是假称两者精度完全相同。

正式 warm campaign：两个独立进程会话，每会话每 arm 3 次 warmup，然后 6 个四元组，
依次 ABBA/BAAB 交替；每会话每 arm 12 个、总计每 arm 24 个请求样本。
同一四元组输入/seed/cache-hit 策略一致；各 arm 不得互相共享未声明的 decoded cache。
另外每 arm 5 个新进程 cold 样本；cold-process 不代表冷 OS page cache，禁止未经说明清系统缓存。

上述样本数是**每个性能 cell**的数量，不在不同 shape/token/cache 状态间混池。
R1 最少四个 cells：portrait/long × 512/1024，seed=42，prompt cache miss；cache hit
是额外独立 cells，不继承 miss 资格。实际 token IDs/rows 在绑定时记录；24 不是把
全部输入合起来仅跑 24 次。每个申请推广的 cell 都要独立满足以下门槛。

- warm median 比值 candidate/baseline ≤0.95。
- 按会话分层、以完整四元组为单位重采样 10,000 次（seed=20260930），
  重新计算 median 比值；双侧 95% CI 的上界必须 <1。
- p90 使用 nearest-rank `ceil(0.9*n)`，比值 ≤1.05。
- cold request median 与 load/prepare median 分别 ≤1.10；首次 artifact import/hash
  单列，不伪装 warm 或从 cold-total 抹掉。
- 无 candidate correctness/fallback 异常；同会话前/后半 baseline median 漂移 >5%
  则 campaign inconclusive，不选择较快半段报告。

基线与候选均关闭详细同步 profile；算子 trace 使用另一次同身份诊断 run。
有异常不删除样本；重跑保存新 campaign ID 和原失败记录。昂贵 workload 的小样本只作
screen，不满足以上发布门。无法容纳同预算 baseline 时可以报告功能/容量收益，
不据不同容量 resident 成绩授予 accelerated。

## 6. 计划新增的 runner/verifier

下列为**工具实施合同，当前未创建**：

| 工具 | 输入/输出 | 必须的失败行为 |
| --- | --- | --- |
| `bind_quantized_profile.py` | 本地 artifact、版本/设备/预算 → bound profile + digest | 缺文件/内容证明/基线/配置则失败；不下载、不生成 weights |
| `test_gguf_decode` | 固定 oracle + block/slice fixtures → D0 receipt | 未实现 type、越界、舍入错误失败，不能 skip required types |
| `test_quantized_slot_lifetime` | fake readers、故障脚本、真实设备子集 → L1/D1 receipt | stale/提前复用/无界分配/未 drain 被检测 |
| `run_quantized_campaign.py` | bound profile + arm commands → 原始 samples、媒体和 trace | 不改配置/阈值，不因某 arm 慢而中途换 baseline |
| `verify_quantized_campaign.py` | profile + receipts + 证据文件 → 分项 verdict | schema/hash 缺失、unknown 冒充 pass、样本不足则非 pass |

runner 的接口至少含 `--profile`、`--output`；verifier 用 `--profile`、`--receipt`。
正式 PR 应提供真实命令并更新 08，不能将本段尚不存在的命令写成复现成功。
verifier 退出码约定：0=所请求资格全部通过，1=观测失败，2=输入/证据不足或不合规；
JSON verdict 保留更细的 not_run/inconclusive，不靠一个 exit code 混淆它们。

verifier 本身须有负 fixture：一个空 hash、篡改样本、丢掉慢样本、不同 seed、预算漂移、
未授权 dtype 变化、假 INT8 evidence、缺人工审核、未取得设备却 mock-pass，全部拒绝认证。
receipt 采用 07 字段并新增 profile digest、oracle identity、verifier revision、required
test IDs、evidence hashes 和各分项状态。变更格式先 version，再升级生产 reader。

## 7. M5 / ANE：证据类型和退出实验

### 7.1 不能相互替代的证据

| 证据 | 可以支持 | 不能支持 |
| --- | --- | --- |
| QDQ/MIL inspection、导出成功 | 预期量化数学被表达 | ANE placement、物理 INT8 |
| compute plan 的 op/device 报告 | 该编译结果的设备分配意图 | 每次请求实际执行轨迹或 INT8 算术 |
| 运行期 Core ML/设备 trace，绑定实际 prediction | 被观测 op 的设备执行 | 仅凭 ANE 活动推断乘法位宽 |
| M5 shader/pipeline capture 与可信 integer lowering 证据 | 特定 pipeline 的整数矩阵路径 | 其他 shape/SDK 的资格 |
| 可解释的编译/硬件 arithmetic 诊断，绑定 graph/op/版本 | 覆盖 op 的 INT8 实现证据 | 未覆盖算子的全模型 W8A8 |
| WA/WB/WA + exact/ULP oracle | 换权与数值正确 | 物理 INT8，或者更快 |
| benchmark 更快、功耗变化、CPU-only 对照 | 性能和定位线索 | 单独证明 INT8 |

工具能力因 OS/SDK 而异，实施者必须记录工具版本、采集命令、原始 trace 和 op 覆盖率。
如果公开工具无法给出可解释的 arithmetic 证据，填 unknown；不预设一定存在可用计数器。
可继续报告“runtime QDQ 候选的实测速度”，不可注册 `native_w8a8` 或宣传已获 INT8 加速。

官方参考：Apple Core ML 的量化性能文档说明 M4 INT8×INT8 的适用方向，但并未承诺
本项目这种动态权重图必然 lower。此处以“可观测证据不足则不授资格”为产品策略。

```text
https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html
```

### 7.2 动态 ANE 的有限搜索

A2 不等待 A1 的完整静态 FFN：先制作单 MatMul frozen QDQ 控制 B，然后测试 06 的
A/B/C/D 四 arms。M=256/512/1056（固定形状尾处理单列），K/N 使用真实投影几何，
另加小尺寸 signed/scale/tail fixtures。首轮仅两种 C recipe：规范化 FP16 I/O + 固定
QDQ；若工具链支持，再验证图内动态 scale 归一化。变动一次只改一个 recipe。

每个候选必须执行 WA→WB→WA，改变非均匀 scales/codes/bias，至少 10 个循环；
记录图 hash、compile/load 计数、staging/预测/恢复时间。稳态不允许因换层重编译。
单投影执行至少 5 warmups、30 个交错样本/arm，用作筛选，不替代第 5 节完整请求。

失败处理预先固定：

- frozen 控制尚不能合法导出/放置：停止动态性能结论，报告 target/toolchain prerequisite。
- 动态换权错误、需每层编译、已证实浮点或 CPU 执行：该 recipe 为 unsupported_on_tuple。
- placement/arithmetic 无法观察：inconclusive，不反向宣称硬件绝不支持。
- 数值或含 staging 的收益不合格：保留原始数据，停止集成该 recipe；继续 runtime FP16。
- 筛选成功：再做含 H、gate/up scale 恢复、SiLU、hidden A8、down 和 GPU join 的完整 FFN；
  最终须经 R5 的全请求/预算/质量门才有 A3 资格。

静态 S1/S2 各自绑定 checkpoint/校准，runtime FP16、runtime W8A8 的资格不互相继承。
M5 无实机时 G1 只能完成 CPU oracle、构建隔离、能力 false→fallback，G2 为 not_run。
所有研究负结果都可归档关闭调查，但 09 的对应加速功能状态仍未完成/不支持。

## 8. 本轮文档完成与后续开工清单

本轮补充了范围、映射、精度、配置/冲突、所有权、退路预算和冻结验收政策，没有实现
上面的工具或执行新的模型/ANE/M5 测试。开工顺序：

1. R0 绑定现有 Z Q8、encoder/VAE/tokenizer、O0/O1/O2 和 profile；缺项明确列出。
2. D0 CPU decoder 与负 fixture；typed pager 单槽的 D1/lifetime 测试。
3. 一个完整 Z 请求，p=0→p=1，先 correctness/memory，再性能优化。
4. 每加格式、encoder、tile 或后端，登记独立 capability/acceptance 增量，不能一把打开所有组合。
