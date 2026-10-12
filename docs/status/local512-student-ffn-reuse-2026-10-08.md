# Qwen512²：真实六步 LoRA 的末步 FFN 复用与 B 投影候选

2026-10-08，Asia/Singapore，M4 Max64GB / macOS26.6.2。接续
[BF16 source ranks](local512-bf16-rank-operands-2026-10-08.md)。上一轮为
progress：实现、实测并提交了候选，但没有 Qwen 整图 ANE 盈利。本轮
继续完整目标；不是用小算子或单个scene替代Z/Qwen base、LoRA、1–2
参考图、encoder和GGUF/ConvRot的最终加速要求。没有下载或模型改写。

## 结果：减少工作量能提速，但32层画质失败，16层收益小

用户允许更宽松的delta误差，性能按完整请求判断，不要求逐位等价；
仍不放宽非有限值、source/load/memory检查或完整GPU回退。
本轮没有把5%的component delta预算冒充最终latent/图像自动通过。

新显式开关 `TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE` 默认关闭：
`0`关闭；`16`仅复用末16层；`1`/`32`复用全部32层。32档是研究
对照，**不推荐**。32档速度较好，但茶壶出现明显块状纹理、轮廓/
钮和把手细节变粗。16档主体/布局更接近，仍有可见的釉面粗化和
格纹，两参考图更明显；不足以推荐默认启用。这是实图判断，不是
被一个很小的数值等价阈值阻止采用。

### 同库的六步真实模型、四路线完整请求

本地原BF16 DiT、Viggle v0.2.1 r256、512²六步、seed29，每臂独立
进程一冷两热，三个fresh prompts且conditioning全部miss。相同
encoder source retention，encoder都是GPU；hybrid为Private DiT5120，
两hybrid臂共享gate/up ranks，关闭A-operand实验、FP16 ranks、down
split及W-code cache。不能把source retention算给新缓存，也不能称
combined encoder/DiT加速。100ms memory采样，无strict-load资格。

| scope | GPU off / on warm median s | hybrid off / on warm median s |
| --- | ---: | ---: |
| 一图，32层，v2 | 10.934522 / 10.359241 | 11.965776 / 11.122859 |
| 两图，32层，v2 | 13.175451 / 12.471592 | 14.555602 / 13.818644 |
| 一图，末16层，v3 | 11.001060 / 10.763400 | 11.899194 / 11.579124 |
| 两图，末16层，v3 | 13.118711 / 12.853617 | 14.535465 / 14.375744 |

32档GPU分别快5.26%/5.34%，hybrid快7.04%/5.06%，但画质退化
明显，不提升默认。16档GPU快2.160%/2.021%，hybrid快2.690%/1.099%；
**hybrid仍慢于同样优化后的GPU**。不能把hybrid对自己的小幅收益
当作胜过GPU，或把32档速度与16档画质组合宣传。

v2一图顺序hybrid on→off→GPU on→off，两图GPU off→on→hybrid
off→on；v3一图GPU off→on→hybrid off→on，两图hybrid on→off→
GPU on→off。每个cold/warm样本保留；不同版本/workload不合并取有利
分母。PNG不逐位相同；v2、v3各12张off臂PNG均与上一轮off
基线byte exact，证明这些匹配case的默认输出未变，不推广到全仓。

## 实现与生命周期：复用的是完整FFN，不是遗漏LoRA

沿用已有Transformer的request-local FFN cache。倒数第二步正常
计算并缓存完整`gate/up→SwiGLU→down`输出，包含全部该层LoRA；末步
仍重算attention及当前调制，只在指定层复用上一步FFN输出。因此
它是**时间近似**，不是在新输入上精确执行LoRA，也不是新快GEMM。

Private callback的输出独立copy/eval，不能借用随后被覆盖的ANE/
CoreML共享backing。GPU路径也先完成捕获。最后一步latents实际eval
成功且finite后，才将request的捕获/复用数汇总到可发布receipt。
缺少完整preceding cache拒绝；conditioning变更/reset清空缓存。
bank owner在denoise结束后、VAE之前clear，逻辑释放不自证driver/
allocator物理释放。prefix key区分16/32，不跨input/adapter/request复用。

当前两档均捕获32层，逻辑bank上界256MiB；16档不是128MiB。
开始捕获前要求MLX active+bank仍留8GiB system reserve，逻辑计数
再检查256MiB上限。不声称这是完整process RAM cap。profile拒绝
memory budget/guard、streaming、prompt enhancement、其它时序缓存
及未匹配的LoRA arithmetic组合；允许resident512六步、一个unmerged
adapter，GPU或Private固定非零channel runtime，最多两张ref512。
合法base开关忽略，非法值拒绝；原base旧开关门禁未放开。

plan显式标记`qwen21_student_final_ffn_reuse`；实际receipt包含requested
layers、captured/reused blocks、peak logical bytes和owner清理范围。
32档每请求capture/reuse32/32；16档32/16，均268,435,456 bytes。
这是成功graph/request证据，不是物理kernel数量、INT8 MAC或overlap证明。

16档每请求实际ANE calls：一图off224/on208，两图off256/on240；
on成功channel blocks176，共享176组/352个rank arrays。32档calls：
一图on192，两图on224，channel blocks160。所有matched请求227个
实际adapter bindings，6个actual steps；failure/fallback/overflow retry0。

## 画面、内存和限制

48张PNG及36组whole/detail sheets保留。末16档18个whole pair均
目视检查；另检查一/两图GPU off/on case0的各三同坐标裁剪，共6个
detail pair。其余detail虽已生成，不冒称已逐一检查。32档抽查7个
whole pair即可明确看到新增块纹和粗化；原失败图不删、不改标记。
固定工具标题写GPU reference/ANE candidate；GPU off/on的右图实际
也是GPU，hybrid off/on的左图实际是Private hybrid，以manifest为准。
自动visual manifest保持pending；没有用户批准或多scene/seed资格。

16个memory reports complete；16档八窗口swap-in/out0。32档一图
hybrid on观测系统swap-in393,216 bytes，其余窗口0，swap-out均0，
不把系统计数归因给单一backend。16档phys-footprint peak约38.30–
43.18GB，on比off增加约1.8–2.2GB，远超过256MiB逻辑bank本身。
这里只观测差异，未证明是compiled constant banks或其它driver分配
中的哪一项；不拿logical bytes遮盖whole-process增加的工作集。
完整scope是进程树load+cold+warm+exit，不含外部服务/driver归因。

目录：`outputs/local512-qwen-edit{1,2}-student-reuse-v2-diagnostic-20261008/`
及`outputs/local512-qwen-edit{1,2}-student-last16-v3-diagnostic-20261008/`。

## B投影：接受约0.17%delta误差的GPU kernel候选，未自动接入

新增bounded layer0 adapter probe，只读真实BF16 A/B，先用合成
BF16 X产生原F32 ranks；计时包括rank→BF16、直接BF16 B/F32
accumulator/output和F32 scale。对照为原F32 B projection，没有
跳过alpha或额外拖慢对照。warm3/hot15、六tiles循环换序，M1056/
2113/3137，gate GPU7168、ANE correction5120、down4096。

| M / projection | 原F32 B ms | BF16 operands/F32 delta16×128 ms |
| --- | ---: | ---: |
| 1056 / gate7168 | 0.606875 | 0.513208 |
| 1056 / gate5120 | 0.441834 | 0.407625 |
| 1056 / down4096 | 0.390333 | 0.372750 |
| 3137 / gate7168 | 1.483209 | 1.337416 |
| 3137 / gate5120 | 1.050333 | 0.974500 |
| 3137 / down4096 | 0.861917 | 0.815250 |

delta relL2约0.143–0.169%，低于用户允许的5%component预算；不等于
最终模型误差。部分早期样本有明显host-span变化，全部保留，不称
稳定性能资格。observer完整、最大gap约206ms；没有GPU timestamps。
它未接入本轮模型，也不是末步复用提速原因。下一步可研究B的
scale/base-add/cast epilogue融合，避免大F32 delta及额外dispatch；
不能用这些小B算子收益代替整次请求盈利。

## 构建、回归、提交与清理

最终Private41项、Public43项selected regressions通过，无skip；
包含实际compiled full-LoRA GPU/split cache、全/后半层消费、missing
cache拒绝、恢复与conditioning reset，原projection/shared/down，
实际Private channel晚期full-GPU fallback、ConvRot LoRA、MPP/affine
及实际Public encoder。不是全仓或Public实模型完整矩阵验收。

两份v3 native-only library构建exit0，adjacent thin CLI完成；Public
actual release-binary guard通过。各494个source input hashes匹配本轮
工作树。v2库是其当时snapshot，后续v3 source变更后不冒称仍匹配
当前source。所有库含原ConvRot drafts，是working-tree snapshot，
不是clean staged-only provenance。v3 library SHA：

```text
Private 73a981710de6b3df12c0a46e1c2393f04c1cdf502d2cb664fbbd1f2931d997a5
Public  897423a13dfaac38220f2b2f73691c3b057d52a569885fa19e2718fa87ea3d9e
```

v1逻辑byte上限表达式在实模型前review修正，未跑模型；v2首次夹具
空event造成bad_function_call，后续ref512 flag错误作用于T2I/base
造成合法拒绝，均只修正夹具，失败logs保留。没有放宽旧ref512门禁。

所有owned jobs终止后，清理v1/v2/v3 Private及v3 Public的851个
可重建`.o`，67,111,856 logical bytes（约64MiB），以及四个空
module-cache目录；库/CLI/probes/logs/PNG/manifest和模型保留。
原ConvRot drafts保持未提交；本轮selective commit只含owned hunks。
相对路径机器记录见[证据](../design/validation/local512-student-ffn-reuse-20261008.json)。

下一轮仍针对实际critical path、B epilogue/compiled constant ownership、
实模型GPU/ANE盈利分区及encoder，继续GGUF真实复用/eviction验证。
本轮不把时间近似、局部小幅收益或counter标签当作完整线程目标达成。
