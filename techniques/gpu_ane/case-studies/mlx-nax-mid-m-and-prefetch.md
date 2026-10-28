# MLX M5量化QMM：mid-M split-K、16-row schedule与预取

<!-- knowledge
id: technique-gpu-ane-case-studies-mlx-nax-mid-m-and-prefetch
type: case-study
stage: ffn, prefill, decode
operator: gemm, gemv, quantization
backend: metal, mlx
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [GPU量化机制](../gpu-gemm-quantization.md) · [硬件边界](../hardware-and-tensorops.md) · [来源](../sources.md#mlx-nax-dispatch)

核查日2026-10-17。本页将此前只读patch关键hunk的[MLX #4682](https://github.com/ml-explore/mlx/pull/4682)推进到**精确base/head的最终组合源码**：

- base `06eb7483f03bca49a80e596f4089d02a507a2cf9`，tree `a230b2cb5d73a59a6ad072b8fff130d305ad1e9c`。
- head `6e7ab39b4b1042d53860507e2e00e4854a152658`，tree `2815b80d446a2b138e59c95041d21f5a8a2dfd14`；前一commit为`6b2af68f46505466df5da6f4175ccd259bda83bf`。
- 当日GitHub API为open、merged=false、merged_at=null。非空`merge_commit_sha`是API合并预览字段，不能单凭它认定已合并。新下载的两段patch与第一轮归档字节相同；本页不是新发现一个独立PR，也不声称最新main/release。

本机没有M5执行证据。实际只编译/运行两个从head**逐字提取**的纯CPU策略函数；未编译MSL、运行MLX或上游Python测试。本页将源码事实、作者报告、数学见证和本机执行分账。

已有本地MLX checkout固定`36969c094354086e15d2a327a9c50feb20b9cf11`，不能冒充PR base/head；所需两ref文件从固定URL独立取得并核官方tree blob，仅复用与head blob相同的本地helpers/许可/贡献指南。缺对象查询的首次自动promisor fetch失败，随后以`--no-lazy-fetch`确认缺失，不继续fetch或切换共享checkout；参考源码保持不变。

## 1. 问题不是只有“decode还是prefill”

`X[M,K] @ W[N,K].T`在M为8到数百、N不很大时，可能既不像单token GEMV，也没有足够输出tiles填满GPU。base的`QuantizedMatmul::eval_gpu`先按架构/尺寸的`get_qmv_batch_limit`分流；小于阈值仍走qmv/qmv_wide，达到阈值且B=1的转置权重先试旧qmm_splitk，最后才可能到unsplit NAX。

PR针对这个中间地带做两件独立的事：

1. K分区增加输出tile不足时的threadgroups，写float32 partials再归约。
2. 无论是否split，对符合条件的转置affine NAX QMM换成4个16×64 simdgroups、预取与双buffer加载。

因此“没有走新增split分支”不等于“没有获得新schedule”；反过来，“进入split入口”也不保证最终split>1。

## 2. 三层门槛，不能只抄M≥8

head `quantized.cpp:1174–1207,1346–1351,2073–2116`可归纳为：

| 层 | 条件 | 不足以证明什么 |
| --- | --- | --- |
| 能力/格式 | NAX guard；affine；N、K均64对齐；K能整除group；不是float32；有affine bias输入 | 并不表示实际多分区或所有低比特格式通用 |
| 原向量区的新入口 | 转置；`8≤M<vector_limit`；`M*N≥16384`；输出恰为M*N；四输入；再过route门槛 | 不是所有batched/gather/MoE路径都重路由 |
| route门槛 | 64列tiles，用M≤32时BM32、否则BM64估算；两倍tiles≤256；K可被`2*max(64,group)`整除 | K虽可平分，但每段可能不足256，实际仍unsplit |
| split选择 | 实际BM64；从1倍增到最多4；总TG≤256；分区至少256列且满足BK64/group对齐 | 不是根据GPU core数动态求得的通用最优值 |

原矩阵区`M≥vector_limit`、转置且B=1时，新增候选在旧qmm_splitk之前；条件不满足才保留旧路线。旧路线约512TG、BK32的策略仍存在，与新增256TG/BK64上限不同。`vector_limit`本身仍按架构与K/N变化，不是固定的32。

几个从原始CPU函数得到的边界：

| M/N/K/group | route helper | 实际split helper | 含义 |
| --- | --- | --- | --- |
| 16/2048/256/64 | true | 1 | 进入新入口后仍用unsplit NAX |
| 16/2048/384/64 | true | 1 | 可二等分不代表每段满足≥256 |
| 16/2048/512/64 | true | 2 | 深度256/分区 |
| 16/2048/1024/64 | true | 4 | 四分区、每段256 |
| 16/8192/4096/64 | true | 2 | 128输出tiles只能二分才不超256TG |
| 16/2048/2880/64 | false | 1 | K本身64对齐，但不能按128对齐二分 |

两个helper不含dtype、设备、M≥8或输出batch检查；直接调用helper的true不能冒充完整dispatcher真的选中了kernel。M=1/7也纳入CPU边界扫点，正是为了保留这个差别。

## 3. offset是byte：复制成本发生在不同分支

`use_qmm_t_nax_rows16:822–830`要求转置、N64对齐、affine、FP16/BF16、`x.offset()%8==0`。`array.h:361–372`把offset加到char指针，单位是**byte**，不是8个元素；对16-bit输入是4元素对齐。

一般unsplit `qmm_nax`对不对齐offset保留旧schedule。新增`qmm_splitk_nax:1223–1230`则先插入`contiguous_copy_gpu`并保留temporary，再递归调用；即使随后的split helper返回1，这次copy也已经发生。`ensure_row_contiguous_matrix`保证末两维连续不代表offset必为8-byte对齐，也不代表整个batch布局连续。新schedule可用于batched unsplit QMM，但新增split重路由只适用所读的单输出batch分支，不能推广到gather kernels。

迁移时必须单列strided-copy与offset-copy；仅做aligned warm kernel benchmark会漏掉真实view消费者的成本。这里分析的是host分支，未在本机运行这些copy。

## 4. “4-bit NAX”这里仍是浮点解码后的MMA

数据链是：

```text
packed weight words + scale/bias → registers
  → affine decode到T（FP16/BF16）→ threadgroup weight tile
  → cooperative A/B fragments → MPP matmul2d → float accumulator
  → float split partials → float归约 → 最终T输出
```

`quantized_nax.h:572–675,1027–1113`复用原dequantize，允许device输入/thread输出；先取uint32 words和metadata，解码时再拆byte，最后按8个T写threadgroup。3/5/6-bit分别有跨byte packing，不是任意“uint4 tensor”可直接消费。`steel/gemm/nax.h:387–456`最终调用simdgroup作用域的MPP 16×32×16 cooperative matmul，operand是已解码T、destination是float。本路线不因模型以INT4存储就获得“原生整数矩阵乘”资格。

对BN=BK=64、128线程的staged loader，每线程负责同一weight row内32个连续逻辑值，恰好覆盖64×64一次；读取word数随bit数变化，1/2/3/4/5/6/8-bit分别为1/2/3/4/5/6/8个uint32。32/64/128三种group下，这32个值不跨group；group128的相邻两次K64步可共享metadata组。这里是地址/packing合同，不是设备带宽计数。

## 5. 预取、双buffer和barrier的真实顺序

head `quantized_nax.h:1115–1274`中四个simdgroup各计算16行×64列，而非旧2×2网格各32×32。输出tile仍64×64；host以flat `(128,1,1)`线程组提交，这仍是128线程，不是额外四倍线程。

流程为：初始fetch/decode/store第0步并barrier；循环先fetch下一步packed words/scale/bias到registers，同时取下一组activation fragments，再执行当前步MMA；随后把下一步解码到另一块Ws，最后全threadgroup barrier并交换buffer。尾步的activation预取退回当前K位置，避免读到K结尾之外。每个循环末仍有barrier，不是“无同步”，更不是host异步copy指令。

显式Ws占用为`buffers * BN * (BK + 16/sizeof(T)) * sizeof(T)`：

| T | 单weight buffer | 双weight buffer |
| --- | --- | --- |
| FP16/BF16 | 9,216 B | 18,432 B（18 KiB） |
| FP32 | 17,408 B | 34,816 B（34 KiB） |

FP32双buffer超过该路径意图遵守的32KiB预算，且rows16实现有`sizeof(T)==2`静态约束，故FP32保留旧schedule。占用按**解码后的T**而非packed bits算；INT4文件更小不让Ws减半。表中仅显式weight数组，不含register spills、编译器临时或其他资源，不能据此宣称occupancy已测得或预取一定盈利。

M尾部将越界fragment行读到本tile最后有效行；无有效行的simdgroup跳过activation/MMA/store，但仍参与weight加载及barrier，直到循环结束后才return。只有有效输出行写回。不能为了省无用工作把return挪到barrier之前。N尾部不走新schedule；仍由原aligned=false路径处理。

## 6. split-K：改迭代长度，不改原始row pitch

split kernel `quantized_nax.h:1631–1703`先按partition移位x、packed w、scale/bias和float输出plane，再传入**原K**与独立`k_partition_size`。原K用于x row stride、weight packed row stride和metadata row stride；partition长度只限制K循环。若把K直接改小，第一row也许对，第二row就错——本库CPU负控分别覆盖activation、packed4-bit和group metadata寻址。

scratch为`split_k*M*N*4`字节，多一次归约dispatch。partial保float避免先舍入每段FP16/BF16；`gemm_splitk_accum:172–193`再按plane累加并最终转换。但split改变浮点加法顺序，“最后只舍入一次”并不保证与unsplit bit-identical。

独立CPU负控：partials为2048.5与−2048，最后转half得0.5，逐partial先转half则得0；float32序列`[2^24,1,1,−2^24]`顺序累加为0、二分累加后归约为1。它们是实数/浮点控制，不是MPP误差测试。PR作者“同split保持bit-identical”的结果与“换split等同unsplit”是不同命题，本机没有验证前者。

## 7. 实际执行与未执行的证据

observer只调用head `quantized.cpp:1188–1192,1198–1207`两段逐字提取的pure C++函数。提取include保留MIT notice、函数range/hash，仅留ignored outputs；wrapper加namespace与输出，不改函数体。编译binary仅声明libc++/libSystem依赖，无Metal/MLX库。

一次有界CPU进程跑14种M、7种N、12种K、3种group，去除K/group不合法组合后共**3430点，每点调用两个函数**。全部actual与独立Python模型相等；分发JSON保留count、规范化全部case的SHA-256、stdout hash及terminal receipts，原始全部cases在ignored stdout。离线测试重算整组digest，不把小量展示值当全量验证。两个函数不执行实际分配、copy、dispatch或MMA。

另有21种bit/group的loader覆盖、16×16 fragment坐标、M=1–257的尾行地址、内存公式和舍入负控。全部为独立CPU数学/地址见证，没有调用Metal或上游runtime。

上游`test_qmm_mid_m`参数静态展开为1530个有效组合，`test_qmm_t_batched_and_strided`为72组合；这些是**源码中的测试设计**，本轮运行数为0。source tolerance用同一packed权重dequantize后的float32参考，以及相对于最终输出舍入误差的1.1×/2.0×门槛，不是模型质量合同；均值门槛也不能替代全值极端误差、OOB或长跑验证。

## 8. 上游性能报告怎样读

PR作者对M5 Max 40-core的一次85-token d1-3B forward报告约2.07×，同时说明长输入/大batch更接近compute-bound、收益较小；依赖链operator测试、模型forward和脚本quick benchmark边界不同。正文还明确默认TF32时main与候选存在大量共有FP32测试失败，不能摘要为“完整套件无条件通过”。本轮未重跑这些数字，也未下载模型。

Apple的[2025-11-19 M5/MLX文章](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)说明GPU Neural Accelerators经TensorOps/MPP使用，并以特定M4/M5机器比较长prompt TTFT与生成速度；它不是此2026年PR的base/head A/B，也不能把跨芯片差异归因于本PR。MLX这条GPU/NAX路线仍不是独立ANE协同。

迁移到GPU+ANE搜索时，应先重建更强的完整GPU基线，再重搜ANE share：split-K增加并发度与scratch，双buffer增加threadgroup占用，offset copy改变暴露准备时间；这些都可能改变与ANE同时执行时的DRAM/功耗竞争。不能将作者GPU倍率乘到本库的历史hybrid倍率上。

## 9. 复测合同与来源

native CPU重放需按[来源清单](../sources.md#mlx-nax-dispatch)先取得相同base/head源文件与tree元数据到新ignored目录，再用capture工具`--prepare`生成带notice的精确include；原编译/运行完整argv保存在。工具不会联网、安装、运行GPU或启动MLX。原始源与网页留`outputs/gpu-ane-mlx-nax-round17-20261017`，核心Wiki不分发第三方源码/编译物。

真正M5复测需要匹配source build/JIT选项、同packed权重、TF32配置、M/N/K/group/bit/offset/stride、完整copy+scratch+归约、独立输出oracle和同输入成对反序计时。按GPU guard连续5次一秒占用采样均低于5%才准入；缺计数拒绝，忙时等待。还需全消费者/模型质量、cold/peak和hybrid争用验证。本机本轮没有发起任何设备测试，不为M5专门路径以M4替代验证。

固定入口：[head host选择](https://github.com/ml-explore/mlx/blob/6e7ab39b4b1042d53860507e2e00e4854a152658/mlx/backend/metal/quantized.cpp#L1171)、[staged loader与rows16](https://github.com/ml-explore/mlx/blob/6e7ab39b4b1042d53860507e2e00e4854a152658/mlx/backend/metal/kernels/quantized_nax.h#L1027)、[kernel实例化](https://github.com/ml-explore/mlx/blob/6e7ab39b4b1042d53860507e2e00e4854a152658/mlx/backend/metal/kernels/quantized_nax.metal#L83)、[新增测试](https://github.com/ml-explore/mlx/blob/6e7ab39b4b1042d53860507e2e00e4854a152658/python/tests/test_quantized.py#L686)。范围/验证另见[第十七轮](../../MAINTENANCE.md)。
