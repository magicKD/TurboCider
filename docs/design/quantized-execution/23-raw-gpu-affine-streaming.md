# 23 · Raw GGUF 槽、GPU affine packing 与低内存供给（2026-10-02）

[目录](README.md) · [22](22-fp16-packed-compute-and-block-validation.md) · [生命周期合同](10-execution-and-product-integration.md)

本阶段新增明确版本的**raw Ready → owner GPU consumer**实验，不把CPU只读完raw
冒充09/10的dense/affine decoded Ready。GPU/CPU职责、reservation和读者完成分别记录。
真实Q8/Q4输出与22已验证的FP16 packed consumer exact；但本阶段速度仍不满足BF16
≤20%目标，未推广/default/catalog/whole-request认证，完整目标继续active。

## 1. 不可混淆的Ready表示

`z-raw-gpu-affine-f16-v1`：mode=`bounded_raw_packed`、backend=`cpu_io_gpu_affine`，
显式packed_streamed/compile_gpu/allow_approximation。legacy CPU profiles不改。
该revision的slot FieldSpec明确U8 raw block bytes/磁盘几何，conversion为
`gguf-raw-gpu-affine-v1`；descriptor绑定`ready_representation=raw-gguf-v1`和GPU consumer。
它不是10中将GPU decode提前提交后错误publish为decoded Ready的路径。

1. owner创建固定raw pool，worker只pread到已捕获的CPU pointer；量化字段不repack，
   不调用MLX/提交GPU，仍检查source generation、取消、bytes与ticket。
2. owner bind消费raw ticket，先reserve当前block全部GPU output/status上界，再提交
   Q4_0/Q4_1/Q8_0 packing；按block批量eval/status check/fence。
3. 全部成功后事务性bind计算可用codes/scales/biases，调用22的FP16 QMM/FP32 glue。
   raw slot的最后reader覆盖GPU packing和block compute，未完成不可回填。
4. 一个当前GPU bank，不常驻整套affine或dense模型；p0/1/2对应1/2/3个raw slots，
   refiners沿用interleaved单槽/旧浮点计算边界，bank barrier后释放重建。

parser拒绝mode/backend混用、resident/隐式source、无授权/compile、whole guard、LoRA
或ANE。普通构建generate仍capability拒绝。K/IQ没有在GPU分支静默CPU fallback；
此GPU packer仅支持上述三种完整rank2 blocks，其余已有CPU格式不受影响。

## 2. Packing oracle、容量与所有权

`gguf_gpu_affine.hpp`：原FP16 scale bit pattern，Q8 signed byte XOR0x80、Q4 nibble
重排，Q4_1原bias保留；Q4_0/Q8 bias一次FP16 RNE。逐group valid同时检查finite
metadata和FP16 consumer重建范围，非法值失败不clamp。单projection与batch failure
均不发布partial bank，GPU完成后先拆multi-output sibling/source引用再交消费者。

每个输出先reserve、后按实际allocator buffer capacity commit。StorageLease装入其
array Data deleter，逃逸view/lazy reader继续持有claim；不是只从producer dictionary
删除就报告释放。状态归约自身的GPU引用也需要完成清理，否则status会额外retained。
component tests发现这个真实引用问题，最终显式completion/detach解决，不隐藏其bytes。

非零MLX cache会复用比逻辑请求大的backing，首个512MiB cache模型因此正确触发
capacity拒绝。最终`ExactCapacityCacheScope`只在受管理buffer/packing输出的分配阶段
使用fresh bins，始终恢复原hint；没有把actual overflow事后改成更大planned upper。
raw profile额外`--raw-gpu-cache-bytes`0…1GiB仅是compute cache hint，不是RAM cap。
该精确分配策略的性能成本在下表保留，后续应研究固定GPU输出槽/allocator治理。

## 3. 已验证

- 实际Metal27 geometries：3types×rows1/7/33×K32/64/256，GPU codes/scales/biases
  与独立CPU `pack_native_affine`逐byte exact；budget/type/shape/nonfinite/FP16 overflow
  拒绝；mixed batch partial failure、干净恢复、escaped/lazy claims和cache sentinel通过。
- ASan/UBSan同component通过。真实StageExecutor K1/K2/K3、两次pass、8fills/readers、
  GPU projection exact、stale ticket、无source reread/无CPU量化repack、ledger归零通过。
- parser7项和measurement/identity/screen20项通过；普通库实际raw generate gate拒绝。
- Q8/Q4真实512² portrait/seed42/4steps、p1，conditioning/initial/every-step/final
  latent/decoded/PNG与22对应同源consumer exact。不是只以final N1子门代替全部endpoint。
- p2 block5取消status2且无PNG，同engine retry status0、全张量exact；原始失败保留。

## 4. CPU供给改善，但相对最快BF16速度门失败

逐projection准备首版Q8诊断denoise6.285s、GPU prepare host wall .945s；改为每block
批量准备后诊断denoise5.822s、prepare .507s。Q4对应5.742s / .437s。
CPU decode约.508s（原浮点refiners/小字段），不再做180个量化矩阵的CPU repack。
这些有observer/dump且使用各自实际binary，不混入最终关闭observer的计时。

最终同binary=`733ea1c716b45e97ba9f2ed48272c6b53d6ae3218af615eba766a2909581813d`，
M4 Max64GiB/macOS26.6.2/MLX0.32.0，相同原BF16 encoder/VAE/tokenizer、GPU tuning、
34valid tokens/combined1088、cache-hit，每arm4个热样本/1保留warmup，内存另进程：

| arm | warm request median | 观测process peak bytes |
| --- | ---: | ---: |
| 更快纯GPU BF16 | 3.733259s | 独立screen中的原始采样 |
| Q8 raw GPU p1/cache512MiB | 6.282110s | p1独立memory未跑 |
| Q8 raw GPU p2/cache512MiB | 6.302686s | 9,083,427,488 |

p2对BF16 wall=1.68825×，**68.83%较慢，硬门失败**，memory约39.47%。p2不比p1快，
不据更多slots宣称持续供给变快。小样本不是正式ABBA/CI/cold/全部shape资格。
最终p2 managed peak812,041,472bytes，GPU current upper209,207,296bytes；120main
fills、720 GPU projections、16refiner fills。`decoded_bytes`是旧ABI字段名，新
`slot_filled_bytes`/ready representation明确它包含raw填充，不称已CPU解码。

原BF16 encoder的完整请求peak约9.08GB不能代表6/8GB执行。另起进程用已有bounded
Qwen3 Q8 GGUF encoder（streamed p2/1GiB managed ceiling）组合实测完整request peak
4,165,174,496bytes，encoder managed606,845,952、DiT812,041,472bytes，gap≤15.42ms、
swapout0，可在6/8GB扣10%余量**observed fit**。但其组件/conditioning与纯BF16不同，
不塞入相同组件的BF16性能screen，不授予小机器、质量、RAM hard-cap或速度资格。

[同binary BF16负screen](validation/raw-gpu-q8-bf16-screen-20261002.json) ·
[原始hash绑定诊断/失败/恢复](validation/raw-gpu-affine-progress-20261002.json)。

## 5. 尚未完成的真实目标

raw-GPU机制及数值/ownership通过不等于“6/8GB且不慢超过20%”已经实现；speed仍fail。
下阶段需要避免分配/全局同步和cache flushing成本、固定GPU output bank并证明mutation/
最后reader协议、projection tiles及whole-request allocations/envelope/admission闭包。
22的resident FP16仍是本cell16GB预算的较快候选，不能用新低内存fallback取代它。
IQ/更多组件与48-case媒体、正式perf cells、ConvRot旋转域、GPU/ANE W8A8/M5分层验证
仍在原完整目标内，没有用本stage的小subset重新定义完成。
