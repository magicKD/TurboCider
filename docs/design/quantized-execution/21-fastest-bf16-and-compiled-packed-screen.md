# 21 · 最快 BF16 对照、packed 编译图与 ConvRot runtime（2026-10-02）

[目录](README.md) · [新验收目标](11-acceptance-profiles-and-feasibility.md#9-2026-10-02-用户补充fastest-bf16-low-memory-v1)

本阶段是显式实验和严格负结果筛选，**没有满足新的速度目标，没有推广为默认加速**。
生产`build/native`未替换；完整线程目标保持active。以下每项绑定它实际使用的库，
不同二进制的诊断与速度数据不混池。

## 1. 对照与内存预测

512²、4steps、portrait、seed42、34有效文本tokens（combined padded rows=1088）。
当前生产BF16的≤1056-row小shape默认不能直接代表这个cell的最快路径。
先对照生产默认与已有MPP SwiGLU/projections/QKV/context等显式额外优化：默认约3.78s，
额外优化约3.73s。最终屏幕对照使用后者，相同优化env应用于各arm；没有禁用BF16优化。
这是本机已测候选的选择，不是穷尽所有可能kernel的证明。

真实目录/头部的权重payload预测，不包含encoder/VAE/activations/framework：

| 表示 | bytes | BF16 DiT占比 |
| --- | ---: | ---: |
| BF16 DiT，453 tensors / 6,154,908,736 params | 12,309,817,472 | 100% |
| Q8 raw GGUF | 7,224,676,608 | 58.69% |
| Q8 compute-ready affine | 7,563,825,408 | 61.45% |
| Q4 raw GGUF | 4,509,395,072 | 36.63% |
| Q4 compute-ready affine | 4,848,543,872 | 39.39% |
| ConvRot raw safetensors | 6,200,910,288 | 50.37% |

Q8_0单矩阵raw=34/64=53.125%，affine=36/64=56.25%；Q4_0为18/64=28.125%
和20/64=31.25%。本地mixed文件保留较大的浮点refiners/固定矩阵，因此Q4整模型不能
按30%预测。ConvRot raw占比不授予运行时packed表示或whole-request内存资格。
现有CPU-direct ledger实际容量符合compute-ready预测及对齐上界；整体进程必须另测。

## 2. 原算术的参数化 packed 图与独立 retention

`TURBOCIDER_Z_GGUF_COMPILE_PACKED=1` / runner `--compile-packed`：

- 要求experimental CPU-direct resident GPU、无LoRA/ANE/guard，且显式近似授权。
- 六个affine projections的codes/scales/biases和所有小字段作为graph参数，保留现有
  QMM、源norm/bias dtype、算子边界；不捕获layer内容或常驻dense bank。
- 只编译main blocks，浮点refiners保持既有路线；没有把F32/F16字段默默转BF16。
- eager/capture/详细profile等冲突拒绝，不把被关闭的compiled路径误报为已执行。

`TURBOCIDER_Z_GGUF_RETAIN_PACKED=1` / `--retain-packed`是另一独立实验：

- 仅保留不可变压缩bank（不是dense层），继续cache0；它与VAE重叠，因此比默认
  VAE前释放多占内存，必须独立测量，不能替代stage20的默认生命周期。
- 未缓存prompt前同步并验证旧bank归零再释放；失败/取消也drain→release，不沿用
  partial bank。每个请求/导出前仍检查source generation/path，改变managed ceiling拒绝。
- 记录`session_packed_retention`、实际`reused_packed_bank`、graph recipe和当前请求
  import bytes/time。bank最初导入的累计数据保留，不冒充每次都重新读取。
- whole-request/生产/session-retention资格均未取得；没有绕过`memory_constrained`。

真实Q8和Q4 compiled+retained对各自原native CPU-direct参考：conditioning、初始/
逐step/final latent、decoded及PNG exact；同引擎第二次请求确实复用bank且输出重复exact。
tensor180取消status2、无PNG，重试status0且所有张量exact。
这些是所测cell的实现/调度证据，不是48-case质量/媒体完成。

## 3. 对照工具与速度负结果

`run_z_image_gguf_quantized.py --baseline-bf16 --prefetch -1`用同一harness运行纯BF16；
`--bind-components`在timed调用外绑定实际Comfy encoder/VAE/tokenizer内容并核对generation。
第一次内容hash开销单列，预读取会影响OS cache；这些不是cold-process/cold-disk资格。
timing关闭callback/observer，memory另进程10ms目标采样，保留warmup/实际gap/VM。

`screen_quantized_against_bf16.py`拒绝不同实际库/组件/prompt/seed/tokens/cache/GPU tuning、
不同layout/consumer/retention/source、非BF16 compiled baseline、失败/NaN/PNG不稳定、
计时混入observer、gap>50ms和swap未知/增长。输出preferred1.10/hard1.20和观测内存比，
始终`production_qualified=false / whole_request_memory=unknown`，不自动授予fastest/质量资格。
最后升级screen v2：绑定`tc_system_json`实际GPU/物理memory/OS/MLX；跨tuple拒绝，
旧reports缺设备字段时仍可保留比值负结果，但`device_binding=unknown_legacy_reports`、
不能给target-pass。旧v1回执的来源/verifier hash保留，不倒填设备字段。

首组最终同binary屏幕（`945f8253…`；各4个热样本、1保留warmup，独立内存）：

| arm | 热全请求median | denoise median | 观测进程peak bytes |
| --- | ---: | ---: | ---: |
| 当前更快的纯GPU BF16 | 3.73534s | 3.50017s | 23,011,731,128 |
| Q8原算术compiled+retained | 5.34283s | 4.81935s | 11,315,419,400 |

Q8 wall=1.43035×，**43.03%较慢，硬门失败**；process footprint=49.17%，较低内存子项
通过。比约7.55s的逐请求重导入路径更快不等于达到BF16目标；编译图本身尚未取得
显著加速证据。压缩bank复用消除了热请求导入，但GPU math约4.82s与VAE约0.51s仍在。
BF16的生产allocator策略与实验cache0不同，故49.17%的process比不能解释为仅量化
占比；raw/prepared权重的58.69/61.45%预测是另一口径，尚无完整process upper预测。

观测11.32GB在预留10%的16GB/16GiB可容纳，不满足6/8/10GB或10GiB；仅empirical fit，
不是这些小机器实机或hard-cap认证。更小预算继续stage20的streamed/verified-release
候选，但它们仍未通过此BF16速度门，不能因为省内存就标成目标已完成。

[首组同binary回执](validation/fastest-bf16-q8-screen-20261002.json) ·
[数值/负结果](validation/compiled-packed-diagnostics-20261002.json) ·
[取消重试](validation/compiled-packed-cancel-20261002.json)。
最后又修正actual-reuse计数位置、当前请求import字段和graph标签；最终库重新构建并
独立重测，另存`receipt-final-*`及新的同binaryscreen，不把上述旧SHA改写为新SHA。
最终库`0afd7dfe…`的[独立同binary回执](validation/fastest-bf16-q8-final-screen-20261002.json)：
BF16/Q8 warm wall为3.73511/5.33762s、denoise为3.49946/4.81759s、process peak为
23,009,978,328/11,357,673,736bytes。wall比1.42904、memory比0.49360，速度仍失败；
最大memory gap15.43/15.80ms，swapout0。最终Q8每个热请求明确reused=true、当前请求
source read=0、plan=`compiled_affine_blocks`，没有把旧bank首次导入当成重复开销。
设备元数据新增后再次独立测量，生成[device-bound screen v2](validation/fastest-bf16-q8-device-bound-screen-20261002.json)，
同一最终库/Apple M4 Max 64GiB/macOS26.6.2/MLX0.32.0，BF16/Q8 wall为3.73381/
5.32803s，peak23,011,239,680/11,089,172,744bytes；wall1.42697×仍失败，memory48.19%。
这是另一组观测，不覆盖此前较高峰值；跨run波动和未观测upper仍不能当RAM cap。

最终库另外跑不同prompt A→B→A：三次均未命中conditioning、均释放重建bank，A首次/
返回后的全部张量与PNG exact，B PNG不同。raw总PNG集合非exact是不同输入的预期，
不是same-input不稳定；[独立ABA回执](validation/compiled-packed-prompt-aba-20261002.json)。

## 4. BF16密集槽候选：功能通过，质量失败

显式`z-dense-bf16-v1`接入既有有限槽和BF16 compiled blocks，仅允许packed_streamed、
allow_approximation/compile_gpu授权。所有小浮点字段一次CPU RNE到BF16；固定字段使用
原1MiB read buffer直填单个目标，不额外保留整张raw矩阵。main0/1/2前瞻、单refiner
槽、source/ticket/reader/ledger安全规则不放宽，没有整模型dense展开。

实际Q8，p1，120main fills / 16refiner fills，managed peak749,456,000 bytes；解码
active4.49s、exposed ready wait2.37s，diagnostic denoise6.08s。该诊断有observer/dump，
不计性能资格。相对旧Q8-native final relL2=0.0585004、cosine=0.9982994，**N1失败**，
与stage13已保存的source-mixed负结果一致。不能推广这个profile或放宽3%门。
固定F32/F16/BF16跨chunk RNE、无raw duplicate、wrong-profile/dtype/alias负例和真实
Metal ledger释放通过；数学/内存接口通过不替代模型质量。

## 5. ConvRot runtime FP16：真实执行，但N1失败

新增显式`ConvrotAffineView`，从旧MLX Q8 affine恢复signed codes，验证整H256、row
uniform stored scales和`bias=-128*scale`；整数Comfy H256 inverse→stored rounded
scale→headroom→一次FP16 RNE，单256-int scratch。不使用原F32 scale冒充旧BF16尺度。
广播metadata仅在预admit后物化scales/offsets，不复制codes或full dense W。

`TURBOCIDER_Z_RUNTIME_CONVROT=1`仅experimental resident GPU+ANE runtime、显式授权、
无LoRA/guard/quantized execution。GPU attention/math保留，FFN inverse-FP16走runtime。
这**不是W8A8**，placement/arithmetic仍unknown。

首轮因为metadata非contiguous而全部GPU fallback；成功图片不算ANE执行。修复后：
128hybrid blocks/128ConvRot stage submissions、45,056ANE rows、130runtime calls（2次
headroom重试）、无fallback，slot241,336,320bytes。最终latent relL2=0.0468795 /
cosine=0.9989006，decoded relL2=0.0344975，**N1失败，未推广**。
纯GPU与runtime诊断使用不同实际binary且有observer，不能据其wall授予速度结论。
raw在`outputs/quantized-execution-convrot-runtime/`，包括最初fallback与修正后的失败质量。
[可携带实际执行/质量回执](validation/convrot-runtime-model-diagnostics-20261002.json)。

## 6. 工程检查与剩余目标

实验/普通library builds通过；普通构建在真实GGUF延迟generate和ConvRot初始化入口
拒绝四条显式实验routes且无PNG。首个测试误把GGUF wrapper create当成实际初始化，
因此修正测试进入generate；未放宽后端门禁。Host7项、packed ASan/UBSan、真实
Core ML CPU_ONLY/CPU_AND_NE 2项（含legacy A→B→A十轮和failed stage恢复）通过。
新增profile/parser、Metal固定槽/转换、12项probe/screen/component tests及原W8A8
数学/判定器12项通过；M5实机probe没有运行，不把pytest未收集脚本算通过。

下一步优先：原计算精度的更快GPU quantized/dequant consumer；缩减/重叠解码供给；
ConvRot保持旋转域的runtime图/consumer，避免已失败的inverse FP16舍入；明确的GPU
W8A8能力接口及M5 SDK compile-only隔离；ANE W8A8必须继续含H/A8/scale/SiLU/down/
join并证明实际算术，不能用本次FP16执行替代20–30%加速目标。Formal ABBA/cold/
长文本/1024²/媒体、whole envelope、tiles/IQ/产品接入仍未完成。
