# Qwen Image 2.1 profiling 工程

## 独立runtime conservative缓存tier

`runtime_cache_collect.py --route gpu|runtime --cache off|conservative --case generate|edit`
保留完整40次更新，但单独核对实际跳过的middle blocks；原full-work collector/auditor
仍要求1280块，不借新近似档放松。actual skip callback必须退役layer8未来W并预取
下一步layer0；两固定ANE图不改。`--requests 4`为prime73/hit73/hit74/hit73恢复，
不得作标准三请求速度分母。所有模型窗结束后用`runtime_cache_audit.py`核全来源/
PNG/原始memory与load、实际selectors、数值预算及恢复；pilot旧字节需显式archive。
`--seed`可显式选择另一组相邻seed，默认73不变；三请求为s/s+1/s，四请求为
s/s/s+1/s。代码整理时的三组编辑holdout与当前构建/回归见
[集成验证](../../../docs/status/runtime-ane-integration-2026-10-19.md)。
目前编辑有26.16–26.51s原始信号，生成累计质量预算未过，默认均不变；
见[完整对照与限制](../../../docs/status/qwen-base512-runtime-conservative-cache-2026-10-19.md)。

## 共享compiled complement完整请求对照

`collect.py --compiled-complement`为显式base40/fixed6144/joint QKV研究开关，
X和原W显式输入，一份compiled函数跨层/请求复用；与GPU-dual/family互斥。
实际回执核对prefill/decode覆盖、完整hidden/F32 partial验证、trace1及零回退。
全部模型窗结束后，用`function_request_audit.py --complement-pair LABEL OFF ON`
核完整RGBA、唯一环境变化、prefix命中/算术namespace、source、原始memory链及
全部cold/hot分母。负载失败的正式speed仍null。
本轮首组小收益未在反序保留，默认关闭；见[完整验收](../../../docs/status/qwen-base512-compiled-complement-session-2026-10-19.md)。

## 显式family的完整40步生成/编辑

`collect.py --function-rows 512,1024`或`512,1024,1536,2048`，仅base40 resident512
原BF16的显式Private runtime研究域，manifest rows必须等于family最大值；
其他QKV/lookahead/prefix参数保持与fixed控制一致。默认不设置该flag，旧环境
与默认算术identity不变。实际输出包含function_rows、row_alignment与完整
capability调用次数。所有function初始化验证，热请求不得compile/load。

模型窗全部terminal后用`function_request_audit.py --native-library LIB
--native-source-archive ARCHIVE --window ROOT ... --pair LABEL FIXED FAMILY
--output FRESH.json`独立核全PNG、实际40步/FFN覆盖、全SHA、native、memory及
selector；stock缺省不保存progress JSON，不能冒充timeline进度。正式速度比
仅允许干净stock三请求，外部load失败返回null。见[完整请求结果](../../../docs/status/qwen-base512-function-session-2026-10-19.md)。

## 真实W8 family完整FFN消费者

`w8_functions_real_collect.py`配合`build_ane_w8_functions_real_probe.sh`，直接用
原Qwen BF16两层W和历史冻结FFN输入，四arm比较完整GPU、R1024、R2048及
512/1024/1536/2048 family。`--case generate|edit --mode stock|timeline --repeats
16`，`--reverse`作独立反向；cold/setup与每shape warmup另列。长rows为明确
标记的冻结数据拼接，不是自然prefill/完整DiT block或图片速度。

所有设备窗terminal后，`w8_functions_real_audit.py --native-library LIB
--native-source-archive ARCHIVE --window ROOT HELPER ... --output FRESH.json`
核完整typed数组、全部FP32 join、完整K的F64 down见证、全SHA、memory、实际
NN selectors。历史工具字节必须显式传`--tool-source-archive`，不放宽native
archive。外部CPU load失败仍保留功能/质量审计，但不产生正式性能资格。
详见[真实组件结果与未完成的Session gate](../../../docs/status/qwen-base512-w8-function-family-real-2026-10-19.md)。

## 多procedure底层能力，不是图片跑分

`multi_function_collect.py`配合`build_ane_multi_function_probe.sh`验证一个真实
model、两个固定行数function、共享input/W/output surface及实际global symbol
映射。运行后再用`--audit ROOT --native-source-archive ARCHIVE --output FILE`
做独立完整F64 oracle；提供同一`--library`/`--probe`，不能跨native ABI链接。
能力采集与数组复算必须串行，默认图片/FFN路径保持不变。新
`test_ane_procedure.py`与`test_ane_multi_function.py`均为无设备CPU负控。
详见[能力边界与下一道gate](../../../docs/status/runtime-ane-multi-procedure-2026-10-19.md)。

V7新增`build_ane_w8_functions_probe.sh`及`multi_function_collect.py --kind w8`，
用`--activation exp|sigmoid-complement`固定一个进程的配方。它包含真实GPU
W8/A8 staging、ANE完整合成SwiGLU和GPU FP32 restore，比较legacy补齐与共享
最大pitch的family；仍不是Qwen图片或真实H4096/F12288跑分。设备窗结束后
再`--audit ROOT --native-source-archive ARCHIVE --output FILE`独立复算全部K维。
两个arm各有一个model，不能把两个控制模型数误写成candidate需要两个模型。
`test_ane_w8_functions.py`保存default byte golden，`test_ane_w8_functions_oracle.py`
覆盖finite/tail/policy/增量delta/signed-zero负控。Surface新增列view导致C++ ABI
变化，不能复用V6 helper；W8 helper必须核编译时和实际native ID。

## 跨模型固定图功能窗口与独立审计

`../runtime_image_specs.py`可生成原BF16 Z-Turbo八步文生图spec，或明确标记的
Qwen普通base20步合成rank2 capability fixture。后者不是训练adapter，不应
作为普通风格LoRA的速度/质量证据。生成器仅写fresh输出目录，不能覆盖权重。

```sh
.venv/bin/python -B tools/profiling/runtime_image_specs.py \
  --family z-turbo --route runtime --model models/Comfy-Org-z_image_turbo \
  --rows 1152 --channels 5120 --output outputs/FRESH-z-spec
.venv/bin/python -B tools/profiling/runtime_image_collect.py \
  --library build/YOUR-EXACT-BUILD/libturbocider.dylib \
  --spec outputs/FRESH-z-spec/spec.json --mode timeline \
  --output outputs/FRESH-z-window
```

`--shape-cycle`生成同owner的512²→768×512→1024²→512²，固定descriptor不换。
仅功能/compile检查，不能把异构请求混成一个热中位数。当前Z入口仅文生图，
不接收参考图。按尺寸最佳bucket与共享procedure池是不同问题，不能混称已实现。

collector绑定当前native build manifest、原source代际/prefix、真实进度与PNG、
完整block/预测/健康计数、实际Private selector和process-tree memory；timing
stock与timeline分开，外部推理负载或采样缺口严格拒绝性能资格。不得停止共享
服务来“修好”窗口。每个失败窗口保留，不以更慢、受污染的GPU分母宣布收益。

所有模型窗结束后运行`../runtime_image_audit.py`，传`--library`、
`--native-source-archive`、逐个`--window`和fresh `--output FILE.json`。它独立
复核exact native archive、所有原文件完整SHA、原始memory hash-chain、完整
PNG、同seed恢复及实际预测与capability调用。`--quality-pair LABEL GPU ANE`
永不产生倍率；`--pair`仅允许完整、同source、同任务、干净stock重复窗。
`--allow-contaminated-functionality`只允许原推理完整成功但load检查失败的功能/
图片审计，仍不允许把它用作性能分母。合成LoRA、单请求、变尺寸序列均无倍率。

原扩展实验存在活跃的外部ComfyUI短时进程，Z性能尚未过该gate，App/default
未改变。源码整理与测试提交不代表Z速度或多参考资格已通过。
中间spec/PNG/trace/权重全部留在已忽略的outputs。

## 扩展：Turbo、GGUF、ConvRot、encoder与变尺寸

`expanded_collect.py`保留完整native请求、全部PNG/denoise进度、首/热请求、
实际source身份、process-tree内存和Private selector。普通计时用`--mode stock`；
`--mode timeline`另测实际compile/load/prediction，不混入stock分母。

例如，当前研究库的ConvRot单步功能检查（不是图片质量/速度资格）：

```sh
.venv/bin/python -B tools/profiling/qwen_base/expanded_collect.py \
  --library build/qwen-expanded-convrot-v4-20261019/libturbocider.dylib \
  --model models/Comfy-Org-Qwen-Image-2.1-ConvRot \
  --variant base --format convrot --route runtime --steps 1 --requests 1 \
  --output outputs/FRESH-convrot-smoke
```

正式base使用40步、Turbo固定8步；每个stock窗口3个请求。`--width 768 --height
512`和`--width 1024 --height 1024`分别测完整尺寸；`--scenario shapes --requests
4 --mode timeline`检查同一owner的512²→768×512→1024²→512²，无热换图。
固定512-row图可通过`--manifest
outputs/qwen-expanded-runtime-c512-h4096-f12288-v1-20261019.json`独立启动，不能
在同一热owner内按尺寸切换descriptor后声称无重编译。

`--encoder-ane --encoder-manifest PATH --fresh-prompts`检查实际fresh conditioning；
`--retain-encoder-weights`是另一独立source策略，collector要求它实际保留/复用，
不只验证环境变量。当前单参考encoder候选仍慢于GPU，不默认采用。

ConvRot独立使用Comfy H256、原F32系数、Exp激活和F32 partial join；GGUF使用
独立FP16-IO/F32-join关闭配方。不能借用普通BF16的资格。显式固定Program禁用
失败重建/自动重试；失败GPU回退不算成功ANE加速。`expanded_audit.py`在所有
模型窗结束后独立检查原文件全SHA、exact native源码归档、原始内存记录和RGBA。
用`--pair LABEL CONTROL CANDIDATE`作同mode比较；跨stock/timeline只用
`--quality-pair`，该入口只检查完整图片/来源一致性，永不产生速度比。
详细数据与边界见[持续研究报告](../../../docs/status/qwen-turbo-runtime-ane-2026-10-19.md)。

## 共享compiled GPU complement：先测完整混合FFN

`compiled_complement.hpp`及GPU/`_hybrid`两套probe/collector比较真实native原回调
与一个显式X/gate_up W/down W输入的共享compiled函数。无权重转换/副本，无逐层
捕获；原BF16阶段/F32 partial不变。GPU生成32约10.685→10.630ms，完整staged
混合FFN约12.405→12.337ms，编辑约0.55%同向；尚无attention/QKV/full40资格，
没有主native/App/default采用。

八窗384 measured调用、240次实际ANE workload、100完整数组、768个完整K的
F64 down坐标见证、原576MiB W/full-source/memory独立复核通过，177项host测试
通过。受测全部增量bits exact、换X/W trace1；混合timeline一个实际model且
hot compile/load0。不将graph trace计数当GPU shader编译次数或物理INT8证明。
首混合pilot只差collector elements guard，原source另封存并严格绑定。详细范围、
系统swap边界、复跑及下一道gate见
[共享compiled报告](../../../docs/status/qwen-base512-compiled-complement-2026-10-19.md)。

## 后续down精度筛选：FP16 hidden / 原BF16 W

`mixed_hidden.hpp`和`mixed_hidden_{probe.mm,collect.py,audit.py}`比较原BF16、
FP16再转BF16的数值对照、ready混合和包含每调用hidden cast的混合down；不转换/
复制W，保留F32 partial。实际生成加倍样本原3.704563ms、ready3.702875ms、
含cast3.739334ms，编辑也无稳定净收益，未接入主runtime或改变默认。

五窗448 measured down、85完整数组、1024个完整K的F64坐标见证与独立source/
memory审计通过；168项host测试通过。合成显式保留half underflow相对误差1，
不冒称zero exact。这里只是冻结真实hidden/down，不是完整FFN或新40步图片。
当前helper为`build/qwen-mixed-hidden-probe-v4-20261019`；复跑与边界见
[混合hidden报告](../../../docs/status/qwen-base512-mixed-hidden-down-2026-10-19.md)。

## 后续精度/输入调度筛选：Private FP16 QKV

`fp16_qkv.hpp`与`fp16_qkv_{probe.mm,collect.py,audit.py}`用冻结的实际base40
gen/edit normalized输入，比较完整GPU、当前native W8、Private固定512-row
FP16 tiled/single。直接从原三份BF16 W写当前FP16槽，无逐层packed副本。FP16
约13.55–14.45ms，W8约5.10–5.19ms；只留single图加倍样本也没有改善。

进一步对齐输入ready边界，`--single-only --compare-prepack`共用同一FP16图/
bank，实际将14.279→11.205ms且完整bits不变，但仍慢于W8 5.167ms。没有主native/
App/default采用或新整图速度声明。六正式窗416 measured投影、406 Private预测、
81份完整数组、实际hot compile/load0、原权重/冻结source和CPU复算已独立核对；
161项host测试通过。不是新的40步图片或物理精度/overlap资格。

当前helper必须用CAPI_v4构建；旧v2/v3结果按各自库/源码归档审计，禁止ABI混用。
详见[FP16与输入调度报告](../../../docs/status/qwen-base512-private-fp16-qkv-2026-10-19.md)。

## 后续W producer筛选：单CB＋scale写回融合

新增`qkv_stage_batch.hpp`，同一bank比较原native三CB、batch三code、融合scale写回
三dispatch及单dispatch，直接读取三份原BF16 Q/K/V，无packed W副本。W-only
stock准备约0.899→0.831ms、独立加倍约0.912→0.862ms，但完整Private QKV消费者
约5.118→5.115ms、加倍5.071→5.084ms，未稳定获益，未接入主native/default。

`qkv_stage_collect.py`只测真实W producers（ANE预测0）；
`qkv_stage_consumer_collect.py`用真实W＋合成1024行输入、同一固定512-row ANE图
测完整projection，不是FFN并行或40步图片。两个auditor分别独立复核全codes/
scales与完整raw QKV；全部100663296个实际W code通过严格独立CPU复算。正式
CB/selector、51份完整projection数组及原始memory证据通过，153项host测试通过。

scope末端校验污染的首个失败timeline保留，修正后新helper重采；旧stock通过
`--cohort-library ROOT LIB`/`--tool-source-archive`绑定各自原件。完整scope、磁盘
无损去重、边界和复跑见
[W staging报告](../../../docs/status/qwen-base512-qkv-w-stage-batch-2026-10-19.md)。

## 后续调度筛选：QKV producer前移没有稳定净收益

`frontier_probe.mm`/`frontier_collect.py`使用现有native回调，在同一固定FFN/QKV
owner下比较original/dual GPU × post/pre/hidden三个QKV权重提交位置。捕获真实
base40首decode完整block，再接依赖它的QKV投影fixture；后接norm复用layer0调制，
**不是两个自然连续block或完整40步图片**，当前层QKV仍GPU。主native/default未改。

六个正式窗、864 measured消费者、两operator各1080次workload调用，240份完整
BF16数组、原8份payload、两种case独立selectors/memory审计以及146项host测试
通过。前移减少QKV wait却增加FFN wait，排序在独立重复中反向，没有新开关采用。
初pilot带有sample间CPU扫描，扰动风险单列保留；正式校验位于计时循环前后。

`frontier_audit.py`在模型窗全结束后，用`--cohort`、`--probe-library`、
`--native-library`、`--native-source-archive`和fresh `--output`独立复核。
完整边界、计数与复跑命令见
[提交前沿报告](../../../docs/status/qwen-base512-qkv-frontier-2026-10-19.md)。

## 当前源码：GPU complement融合的完整请求验收

当前native为`build/qwen-base-gpu-dual-v1-20261019`（661 inputs）。新增默认OFF的
`--gpu-dual-swiglu`，只替换base512/fixed6144/joint QKV/互补激活路线的1024行
GPU complement：原buffer成对gate/up＋typed BF16 SwiGLU，无新增权重副本，
长prefill保留原GPU回调。首次真实source hidden/F32 partial逐位验证一次，
失败sticky回退；GPU策略隔离prefix recipe，ANE两个固定Program不变。

13窗41完整请求/1640实际步的独立RGBA、source、selector、原始memory hash-chain
审计和140项host负控已完成。生成热请求约快0.13–0.20%，编辑约0.30–0.89%，
材质提示词约0.13%；信号很小，没有默认采用或稳定加速资格。四个timeline均
两个model、hot compile/load0。FFN包络缩短的同时，QKV weight join暴露约从
0.01–0.02s升至0.18–0.20s，需按完整请求裁决，不能只报融合FFN局部收益。

保持下节QKV前看强基线，添加`--gpu-dual-swiglu`进行on实验；off去掉该flag。
生成仍需`--runtime-text-prefix-snapshot --text-prefix-qkv-step0 gpu`，编辑用
`--runtime-prefix-snapshot`。验收后再读
`gpu_dual_request_audit.py --help`，用`--pair LABEL OFF ON`、`--window on ROOT`、
`--frozen LABEL CURRENT_OFF HISTORICAL`、`--library`、`--native-source-archive`
和fresh `--output`独立复核。旧C++ helper与新RunResult/Session ABI不可混用。
完整数据和复跑命令见
[融合验收报告](../../../docs/status/qwen-base512-gpu-dual-runtime-2026-10-19.md)。

## 保留的完整请求基线：FFN提交后的单组QKV权重前看

前看实验匹配`build/qwen-base-qkv-lookahead-v1-20261019`（659 inputs）。新增默认OFF的
`--private-qkv-prefetch`：在当前FFN提交后提前stage下一次QKV的原权重，只复用
QKV现有bank，不新增Program或checkpoint副本。必须joint QKV＋fixed-program，
CPU future核对generation/layer/rows/实际Tensor IDs并有界drain；长prefill/GPU
计划在搬W前跳过。旧默认/App不变，text-prefix仍必须保留GPU首步QKV。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-qkv-lookahead-v1-20261019/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --fixed-program \
  --ffn-activation sigmoid-complement --private-qkv-prefetch \
  --case generate --mode stock --runtime-text-prefix-snapshot \
  --text-prefix-qkv-step0 gpu --output outputs/FRESH-QKV-LOOKAHEAD
```

同库off只去掉预取flag。完整40步生成29.49→28.84s、重复编辑36.98→36.22s，
反向与独立重复保留信号；增量完整RGBA exact。两图在生成/编辑命中/变长独立
timeline均hot compile/load0。10窗31请求/1240步与105项host测试通过，但非独占
小样本、原W8仍近似，不称物理INT8/overlap或全部瓶颈资格。

`qkv_lookahead_collect.py`另跑真实固定QKV图的合成状态探针（不是图片基准）；
`qkv_lookahead_audit.py`在模型窗结束后独立复核全部数组oracle、实际selector、
计数、memory和matched完整PNG。用`--pair LABEL OFF ON`、`--window on ROOT`、
`--probe ROOT`、`--frozen LABEL CURRENT HISTORICAL`配合`--library`和
`--native-source-archive`及fresh `--output`。原件与边界见
[前看报告](../../../docs/status/qwen-base512-qkv-lookahead-2026-10-19.md)。

## 后续诊断：H512因式分解没有采用

当前native/library/完整40步基线未变。新增`rotation_collect.py`/`rotation_audit.py`
对实际C6144/M1024及小shape的H512分组Conv、两级Conv/MatMul/无transpose
版本做独立FP16算子筛选。四窗1272次实际evaluate、240份完整输出F64 oracle及
112项host测试已完成；八种候选实际shape的原始wall均更慢，wide模式还存在
逐点预算超限，因此没有接入runtime或新增整图速度声明。小shape编译拒绝单列，
不计成0ms或GPU性能；不把20个诊断程序当产品逐层模型。见
[旋转因式分解报告](../../../docs/status/qwen-base512-h512-factorization-2026-10-19.md)。

## 后续诊断：hidden量化与5632/6656相邻分工

主native与最快完整请求配置仍未变。新增`hidden_quant_collect.py`/
`hidden_quant_audit.py`比较原除法和两种倒数复用，含单输出codes/scale、零/tiny/
wide/有限源overflow与独立oracle；更完整旋转链没有稳定收益，且保留原/新共有
的tiny参考偏差。另用`ffn_projection_collect.py --channels ...`与新
`build_qwen_ffn_share_probe.sh`筛选5632/6144/6656；完整block分别约24.91/
24.14/25.04ms，新份额未胜出。`ffn_share_audit.py`严格匹配原GPU、输入、payload、
同分工重放及独立selectors。124项host测试通过；没有新的整图速度或默认变更。
详情及全部scope见
[归一化与相邻分工报告](../../../docs/status/qwen-base512-hidden-quant-neighbor-share-2026-10-19.md)。

## 接入前的局部筛选：原buffer dual gate/up＋typed SwiGLU

以下为原局部筛选，当时主native/default未改；本轮接入与完整请求结果见页首。
新增`dual_gup.hpp`的register/shared两个GPU核，不预拼接
每层W，保留BF16各阶段；合成20输出对MLX/MPP的40个位控制、独立全投影oracle、
真实两层A/B/A与原/融合hybrid完整block已复核。register生成block有约0.35–
0.56%微弱信号，编辑约0.08%；还不是完整40步图片收益或采用资格。

`dual_gup_collect.py --kind register|shared`使用对应新helper，同窗四arm同时
保留原hybrid强控制，`dual_gup_audit.py`独立重读14份BF16 block＋10份混合GPU
数组、原payload、完整源/时间线/内存；`dual_gup_kernel_collect.py`及其auditor
只作无checkpoint合成资格。130项host测试通过。完整边界与下一阶段见
[dual-GUP报告](../../../docs/status/qwen-base512-dual-gup-block-2026-10-19.md)。

## 上一轮：text-prefix保留首步GPU QKV

该轮匹配`build/qwen-base-text-prefix-qkv-step0-v1-20261019`（659 inputs）。
已用同库O/A/G三arm隔离此前反例：仅让缓存命中第0步QKV留在GPU，后39步仍
Private。茶壶/材质两提示词的6张GPU/off完整RGBA exact，热请求约少0.28–0.37s；
这是有限样本的显式研究选项，不改产品默认，也不称普遍无损。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-text-prefix-qkv-step0-v1-20261019/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --fixed-program \
  --ffn-activation sigmoid-complement --case generate --mode stock \
  --runtime-text-prefix-snapshot --text-prefix-qkv-step0 gpu \
  --output outputs/FRESH-TEXT-PREFIX-GPU-STEP0
```

新env为`TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_QKV_STEP0=gpu`，必须与text flag及
joint QKV同时选择；默认`ane`保留旧研究行为，**仍有已知画质反例**，不是推荐
预设。正确方案需要上面两项text参数，旧editing flag不扩大。

计划在stage前直接GPU并记credit；旧`plan(rows)`保留，新增permission overload。
CPU step随forward更新，同shape的GPU/external-QKV compiled-block槽分开；首步
policy进入prefix recipe，不跨策略偷用bank。G hot实际FFN1280/QKV1248/GPU-QKV32，
不是旧A的1280/1280/0，collector与真实per-model selector分别核对。两timeline
仍两个固定model、hot compile/load0；不据此证明物理INT8/overlap。

模型结束后运行`text_qkv_step0_audit.py --off ... --ane ... --gpu ... --library ...
--native-source-archive ... --output FRESH`。可加`--reverse-off`、`--repeat-gpu`、
`--timeline`、`--matrix`、`--edit`及`--additional-pair LABEL OFF GPU`；
`--require-gpu-exact`把全部同seed完整RGBA exact作为明确gate，不能只看SSIM。
旧trial参数只作像素/实际request回归，不借历史时间作分母。

完整31请求/1240步、97项host测试及native policy/lifecycle各1项已通过。
原件、边界与复现命令见[首步QKV修正报告](../../../docs/status/qwen-base512-text-prefix-qkv-step0-2026-10-19.md)。
下文按各自历史库保留，不把旧秒数或失败候选重标为当前构建。

## 历史源码：runtime text-prefix，存在画质反例

本节native对应`build/qwen-base-text-prefix-v1-20261018`（659 inputs），不是旧
activation-v3的同一内容身份。新flag默认OFF，**缓存命中的seed73有可见把手
几何退化，不能当作无损或默认加速方案**。原editing开关及默认输出未改，旧
库/原件均保留。结果与限制见[2026-10-19报告](../../../docs/status/qwen-base512-text-prefix-2026-10-19.md)。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-text-prefix-v1-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --fixed-program \
  --ffn-activation sigmoid-complement --case generate --mode stock \
  --runtime-text-prefix-snapshot --output outputs/FRESH-TEXT-PREFIX-RESEARCH
```

新native env为`TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT=1`，只准入显式
resident512 base40、无参考/LoRA/其它路线的固定Private runtime。旧
`TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT`仍edit-only，不能混称。off对照同库
只去掉新flag；新CLI明确给该env0，旧`environment()`默认返回值保持兼容。

`--scenario text-prefix-matrix --requests 4 --mode timeline`执行A73 miss→A74
hit→B73 miss→A73 miss，真实40步、prompt/PNG与per-model调用核对，不pool median。
命中FFN1312→1280，但QKV1248→1280；首步从GPU转W8是待隔离的画质原因之一。
两图固定、hot compile/load0不能推出输出逐位无损。

全部模型结束后，`text_prefix_audit.py --off ... --on ... --reverse-off ...
--timeline ... --matrix ... --edit ... --repeat-on ... --library ...
--native-source-archive ... --output FRESH`重读来源/实际输入/完整RGBA/memory/selectors。
`--historical-gen`/`--historical-edit`只作旧trial的request/PNG exact回归，不借旧
时间作分母。自动审计通过与人工语义审查分开；本轮人工反例另存visual-review。

七窗22次完整请求/880步已完成，95项host＋native准入/lifecycle各1项通过。
文生图hot29.60/重复29.47s、off29.90/反向29.82s是小样本信号；默认和产品
未改，不以这约1%收益掩盖画质回退。以下章节属于各自历史构建，不重标秒数。

## 最新：F32 down tile与FP16 gate/up独立筛选

新增`build_qwen_ffn_down_probe.sh`。显式选择C5120/6144及`m32n128d`等policy，
BM16/32/64、BN64/128、d/s为dynamic/static extent。只改变诊断GPU down的tile，
原BF16 W/物理pitch12288/F32 output及ANE图不变，不创建compact/widened W bank。

```sh
TURBOCIDER_NATIVE_OUT=build/FRESH-DOWN \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen-base-ffn-activation-v3-20261018 \
bash tools/native/build_qwen_ffn_down_probe.sh 5120 m32n128d m64n64d

.venv/bin/python -B tools/profiling/qwen_base/handoff_collect.py \
  --native-library build/qwen-base-ffn-activation-v3-20261018/libturbocider.dylib \
  --library build/FRESH-DOWN/libqwen_ffn_down_m32n128d_c5120.dylib \
  --observer build/qwen-ffn-handoff-probe-v3-20261018/libqwen_handoff_observer.dylib \
  --channels 5120 --down-policy m32n128d --mode stock --output outputs/FRESH-DOWN
```

仍为真实base40首decode block捕获，三arm强控制；另测gate/up、down、原down
独立完成边界。不完成40步、无VAE/PNG，不能外推整请求。`TURBOCIDER_FFN_GATE_PRECISION=fp16`
仅构建显式有损探针，调用时还须`--gate-up fp16`，库名多`fp16_`；运行时权重
参数的每次转换计入，不预存half bank，也不混入packed GPU策略。

全部模型结束后，用`down_audit.py`配合`--native-library`、`--native-source-archive`、
重复`--probe-dir`（含实际observer目录）、重复`--cohort`、历史`--tool-source-archive`
及全新`--output`复核。新增严格BF16-hidden/F32-partial reader，保留旧BF16-only
reader；完整数组比对加原物理W的32点/完整K独立F64 oracle，不是全CPU GEMM。

十八窗均完成，91项host测试通过。12种tile无可靠速度收益；FP16连转换更慢，
两类方案均未接入native/default。细分显示C5120 gate/up约8.28ms、down约4.27ms，
不能将整段GPU complement都归给down。范围/结果/下一候选见
[down筛选报告](../../../docs/status/qwen-base512-ffn-down-tiles-2026-10-18.md)。

## 最新：restore encoder GPU采样与GPU complement消融

新增`handoff_observer.mm`，复用未改动的基础observer，记录真实Metal event
identity/value，并对wait之后的compute encoder采两个GPU timestamp。真实
shared-event数据自检与约20ms host延迟放行负控，验证采样没有直接包含此前的
event等待；时钟两端标定、错误值/drop/drain/scope/消费者计数均严格检查。
encoder elapsed仍不是shader busy time、独占kernel成本或物理ANE时间。

`build_qwen_ffn_handoff_probe.sh`生成两档份额×两种GPU策略的独立诊断helper，
链接当前native但不改它；stock不加载observer，timeline才更换encoder创建路径。
每窗真实base40的第一step后捕获首decode layer0，三arm各2warm/8measured，另
测ready FFN输入下的GPU complement alone；**不完成40步、无VAE/PNG**。

```sh
TURBOCIDER_NATIVE_OUT=build/FRESH-HANDOFF \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen-base-ffn-activation-v3-20261018 \
bash tools/native/build_qwen_ffn_handoff_probe.sh

.venv/bin/python -B tools/profiling/qwen_base/handoff_collect.py \
  --native-library build/qwen-base-ffn-activation-v3-20261018/libturbocider.dylib \
  --library build/FRESH-HANDOFF/libqwen_ffn_handoff_c6144.dylib \
  --observer build/FRESH-HANDOFF/libqwen_handoff_observer.dylib \
  --channels 6144 --gpu-complement separate --mode timeline --output outputs/FRESH-HANDOFF
```

packed使用`libqwen_ffn_handoff_packed_c6144.dylib`和`--gpu-complement packed`。
每进程固定一种图/策略；不要在性能窗口并行build、测试或CPU数组审计。
全部模型结束后，`handoff_audit.py --cohort ...`（可重复）配合`--native-library`、
`--native-source-archive`、`--probe-dir`（可重复）、历史`--tool-source-archive`
和全新`--output`复核全部source/payload/十份完整BF16数组与counter。

十二窗已完成：5120的restore encoder约3.53ms，6144约0.20ms；GPU gate/up
打包合并未带来完整block收益，新增权重副本不接入产品。最新85项host测试通过。
范围、完整结果与下一候选见[encoder报告](../../../docs/status/qwen-base512-ffn-handoff-encoder-2026-10-18.md)。

## 最新分工复核：输出完成等待需单独分析

当前659-input activation-v3库未改。在`--fixed-program --ffn-activation
sigmoid-complement`下，FFN5120比6144的完整40步生成/重复编辑慢约6.11%/5.07%，
保留6144速度候选；不要直接继承旧Exp的分工排序。反向6144恢复速度与原PNG。
新增[输出交接报告](../../../docs/status/qwen-base512-complement-share-handoff-2026-10-18.md)。

`share_audit.py`支持上述显式policy，并检查实际完整环境仅CHANNELS不同。
`--reverse6144`/`--reverse5120`与`--timeline5120`必须恢复相同请求和完整PNG。
新增每请求`completion_deltas`区分FFN worker prediction等待、output completion
等待和caller finish等待，外加逻辑输出字节、pre-FFN/QKV等计数；拒绝计数回退、
缺失、非有限值与错序。它是原始native累计计数的独立差分，不改变采集器。

新结果的output wait约0.5→4.5s/请求，而prediction wait减少约2s，逻辑输出
字节相同。不能把这些重叠host spans相加成wall，或据此唯一归因ANE计算/
restore kernel/物理带宽。5120编辑timeline确认两个稳定模型、hot compile/load0，
但5120生成只有stock，不从另一任务借用selector证明。全部模型结束后再运行
CPU质量审计/host测试；最新79项通过，完整来源/证据位置见报告。

## 历史源码：固定FFN激活表达

当前native源码匹配`build/qwen-base-ffn-activation-v3-20261018`（659 inputs），
原mixed/projection/fixed-program库及全部原件保留，但不匹配当前源码；下文旧
命令属于各自封存版本，不重标成当前性能。

新增`--ffn-activation exp|sigmoid|silu|sigmoid-complement`，native env为
`TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION`。默认Exp/unset保持旧emitter/identity，
非默认只准入fixed-program、授权Private W8、base Sylvester/ungrouped和
descriptor投影，不与其它投影配方混用。owner捕获不可变策略，actual recipe/
prefix命名空间分开，形状/权重/IO ABI和溢出停用＋完整GPU重算不变。

原生`silu`及`x*sigmoid(x)`在本机被原数值自检拒绝，不是有效ANE性能样本。
独立FP16原始数组显示小值区间两者逐位相同而有明显近似误差；不要关闭或放宽
5%自检门槛。互补表达`x*(1-sigmoid(-x))`有更好的小值数值表现，并已通过完整
block和真实溢出负控；不是相对原Exp逐位无损，完整请求/质量范围见
[激活报告](../../../docs/status/qwen-base512-ffn-activation-2026-10-18.md)。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-ffn-activation-v3-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --fixed-program \
  --ffn-activation sigmoid-complement --case generate --mode stock \
  --output outputs/FRESH-ACTIVATION-REQUEST
```

匹配基线同库，只改为`--ffn-activation exp`。编辑双方同时使用
`--case edit --runtime-prefix-snapshot`，标准1024参考不变。timeline另开，
反向恢复基线，变长矩阵也可携带激活策略；`activation_request_audit.py`
在所有模型结束后复核实际source/native/request/PNG/memory/selectors，只允许
激活环境不同。`quality_collect.py --ffn-activation sigmoid-complement`使joint
侧启用相应fixed配方，普通完整GPU控制不变；单请求质量窗不授予hot速度。

`ffn_projection_collect.py --activation ... --projection descriptor`是完整block
筛选，不导出PNG；`ffn_projection_audit.py --same-capture`复核完整输入/GPU控制
和同配方十份数组重放。`fixed_program_collect.py --policy fixed --activation ...`
及其auditor验证实际合成溢出、captured policy、GPU oracle和零热重建。失败窗
用`activation_failure_audit.py`检查已拒绝子进程与原始采样，绝不当成功性能。

独立`activation_op_probe.mm`/`activation_op_collect.py`不加载checkpoint，只对
两种固定FP16 shape、identity/多种激活做A/B/wide/A。builder为
`tools/native/build_qwen_activation_op_probe.sh`，`activation_op_audit.py`重读
全部FP16文件并独立F64 oracle。12个operator诊断程序不是产品逐层模型，API
窗口不证明物理INT8/设备重叠；原始算子精度也不代替整图语义资格。

## 上一轮：gate/up与down独立策略

本节历史native为`build/qwen-base-ffn-mixed-projection-v2-20261018`。仍为658
inputs，但内容/runtime ID与上一轮不同；不以文件数量代替内容身份。旧库及
原件保留，下面旧构建命令属于各自历史版本，不能与当前源码混用。

新增`up-single|down-single`，仅改变对应投影的K策略，保留另一投影的原字节
（包括小K的identity slice）。仍需fixed-program、授权Private W8和base-only
Sylvester准入，权重/IO ABI不变，recipe/prefix命名空间分开，不热切图或逐层
编译。两个新策略目前只完成真实block筛选，均更慢，没有新的完整图片性能。

```sh
.venv/bin/python -B tools/profiling/qwen_base/ffn_projection_collect.py \
  --native-library build/qwen-base-ffn-mixed-projection-v2-20261018/libturbocider.dylib \
  --library build/qwen-ffn-mixed-projection-probe-v1-20261018/libqwen_ffn_projection_probe.dylib \
  --projection up-single --case generate --mode stock --output outputs/FRESH-MIXED-BLOCK
```

此模式首步后捕获首decode layer0，完整block三arm旋转，各2warm/8measured；
`--repeats 16`加倍样本。QKV仍GPU，没有VAE/PNG或完整40步；不要外推整请求。
timeline另开进程。全部设备窗口结束后，`ffn_projection_audit.py --factorial`
前四`--cohort`必须依次descriptor/up-single/down-single/single、相同repeat的
stock窗口。额外cohort允许加倍/反向/timeline，但同策略十份完整数组须exact；
所有源FFN输入/GPU controls也须exact。完整payload、source、memory及实际
selectors仍独立复核；interaction是完整block差，不是硬件operator成本。
见[混合投影报告](../../../docs/status/qwen-base512-ffn-mixed-projection-2026-10-18.md)。

## 上一轮FFN投影粒度与完整请求筛选

本节历史native为`build/qwen-base-ffn-projection-v1-20261018`（658输入）。
固定Program657、joint656及更早二进制/原件均保留，但不匹配当前源码；下文历史
命令不要重标为当前构建。新构建保持默认emitter六类fingerprint完全一致。

`--ffn-projection descriptor|k4096|single`只在初始化时选择FFN图内部K分片。
非默认要求`--fixed-program`、显式Private授权、原base/ungrouped Sylvester
配方；拒绝LoRA、grouped、Comfy及其它BF16边界配方。正常权重/输入/输出ABI
不变，实际source recipe与prefix身份包含非默认policy，不跨配方复用prefix。
H4096/C6144时图内MatMul数7/4/3，不是每block调用数，更不是物理硬件tile。
减少节点不保证更快：完整block与完整请求均须测量，不能套用QKV single的收益。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-ffn-projection-v1-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --fixed-program \
  --ffn-projection k4096 --case generate --mode stock \
  --output outputs/new-base-ffn-projection-request
```

基线同一库，改为`--ffn-projection descriptor`；编辑双方均加
`--case edit --runtime-prefix-snapshot`，标准参考1024不变。timeline另开进程，
两模型身份与真实compile/load/evaluate分别核对，诊断秒数不进入stock分母。
不在一个owner热切换策略或创建逐层checkpoint模型；没有变更产品默认。

`ffn_projection_probe.mm`＋`ffn_projection_collect.py`是另一级真实block筛选：
首步完成后捕获第一decode的layer0，普通compiled GPU为主分母，stagedGPU须
exact，Private每次实际stage生产FFN权重并完成GPU complement/ANE/F32join。
三arm旋转各2warm/8measured，再查激活A/B/A、原layer0/1/0权重、旧输出所有权；
QKV仍在GPU，退出不导PNG，不称完整40步图片。实际probe builder为
`tools/native/build_qwen_ffn_projection_probe.sh`，CPU emitter兼容性builder为
`tools/native/build_qwen_ffn_projection_describe.sh`。

所有模型窗口结束后，`ffn_projection_audit.py`重读十份完整BF16数组、原checkpoint
两层四个FFN payload、源代码/库/内存与可选selector；不是完整CPU denseFFN oracle。
`projection_request_audit.py`在完整请求中只准许projection环境不同，比较同库
stock/独立timeline/反向baseline以及增量PNG。改变reduction允许像素变化；同配方
重放才要求exact。可给`--gpu-reference`只查质量，绝不借历史GPU秒数作新分母。
完整方法、分层结果和适用边界见[FFN投影报告](../../../docs/status/qwen-base512-ffn-projection-2026-10-18.md)。

```sh
.venv/bin/python -B tools/profiling/qwen_base/projection_request_audit.py \
  --library build/qwen-base-ffn-projection-v1-20261018/libturbocider.dylib \
  --native-source-archive outputs/qwen-base-ffn-projection-native-source-v1-20261018.tar.gz \
  --baseline outputs/YOUR-DESCRIPTOR-STOCK --candidate outputs/YOUR-SINGLE-STOCK \
  --projection single --case generate --timeline outputs/YOUR-SINGLE-TIMELINE \
  --reverse-baseline outputs/YOUR-DESCRIPTOR-REVERSE --output outputs/FRESH-AUDIT
```

`--historical-descriptor FROZEN_TRIAL_DIR`只检查旧同算术request/PNG exact；
`--gpu-reference FROZEN_GPU_TRIAL_DIR`只检查同prompt/seed/geometry的质量。
两者均不读取旧wall作为本轮性能分母。实际报告分别保留冷prepare、stock、诊断、
反向、默认兼容性与未取得的硬件/图像语义资格。

## 显式fixed-program：禁止热重建

本节记录`build/qwen-base-fixed-program-v1-20261018`（657输入）的合同与历史实验；
该合同已包含在上面的当前activation构建。旧数据/库/源码archive保留，不重标版本。
使用显式`--fixed-program`，原App/产品默认仍关闭：

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-fixed-program-v1-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --fixed-program \
  --case generate --mode stock --output outputs/new-base-fixed-program
```

编辑可同时使用`--case edit --runtime-prefix-snapshot`；变长矩阵使用
`--scenario sequence-matrix --requests 4 --mode timeline`。对照只去掉fixed开关，
必须使用同一新库。新原生env为`TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM=1`，不自动
授权Private，也不替代manifest和approximation准入。unset/0保留旧行为和身份。

策略在owner创建时捕获，FFN溢出不再build新Program；完整GPU重算、释放/停用
失败路径，同配置下一请求不重新装载。初始FFN失败时QKV用无Device/Program的
失败占位。shape不变、正常算术不变，失败不发布runtime prefix。新receipt区分
requested与实际enabled，记录Program构造/blocked重建/disabled状态；这些
host计数不能替代真实compile/load选择器，也不覆盖不可见driver内部编译。

`fixed_program_collect.py`＋`fixed_program_probe.mm`是真实小模型溢出对照；
`fallback_collect.py`运行三次完整base40初始准入失败负控，不能当成功ANE性能。
`fixed_program_audit.py`复核原数组CPU oracle、实际selectors、完整请求off/on/
timeline/reverse和GPU fallback PNG。最新[固定Program报告](../../../docs/status/qwen-base512-fixed-program-2026-10-18.md)
列明范围、状态转移、原件、初始compile与后续零热compile/load；成功PNG对照
exact并不使原W8路线变成相对完整GPU无损。

## 联合配方筛选与多提示词检查

后续[分工/K分片/质量报告](../../../docs/status/qwen-base512-joint-share-quality-2026-10-18.md)
保留同一native库的参数筛选。`share_audit.py`比较joint5120/6144，只准许CHANNELS
环境变量变化；`tiling_audit.py`比较固定MIL K分片，只准许descriptor的tile_k变化。
actual40步、无fallback/retry、相同请求、完整PNG/source/memory和独立反向控制
分别复核。内部MIL K分块不是物理ANE kernel tile，也不是每tile一次predict。
一次进程固定一种配方，不在热路径编译/调参。当前筛选没有优于默认6144/K2048
的速度信号；descriptor中4096被native emitter限制为2048，不用标签替代源码。

```sh
.venv/bin/python -B tools/profiling/qwen_base/quality_collect.py \
  --library build/qwen-base-joint-qkv-ffn-v1-20261018/libturbocider.dylib \
  --channels 6144 --output outputs/new-base-joint-quality
```

该脚本串行执行文字海报、成人肖像、材质排列、标准参考matte编辑，各有同库
完整GPU/joint配对；每进程1个真实40步请求，原prompt/脚本身份/完成记录保存
在`recipe.json`。不运行LoRA/student、不下载语义模型、不产生hot timing分母。
全部窗口结束后用`quality_audit.py --pair LABEL GPU_DIR JOINT_DIR`（可重复），
同时指定`--channels`、`--library`、`--native-source-archive`和全新`--output`。
像素分数与人工检查分开：SSIM高不等于字符正确、脸/材质保持或总体语义资格。

## 固定Private QKV＋FFN联合实验

当前observer源使用20万条有界记录，默认构建目录为
`build/qwen-base-profile-observer-v2-20261018`；可按下文CMake命令在此目录构建。
旧10万条v1二进制仍保留，但不能与修改后的observer源hash混用。joint首六窗
绑定的旧工具已封存于`outputs/qwen-base-joint-profile-tools-v1-20261018.tar.gz`；
审计这些历史窗口须传`joint_audit.py --tool-source-archive`，不修改旧source.json。

新库`build/qwen-base-joint-qkv-ffn-v1-20261018`支持显式`--private-qkv`。
仍默认关闭，不修改App或产品默认。只有base40、resident512、cache-off runtime、
健康Hadamard W8/F32 FFN配方可准入；拒绝LoRA/student、weight prefetch、其它
encoder/量化/temporal路线。支持`stock/timeline`及标准1/2参考，不能与旧public
CoreML QKV模块混称。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-joint-qkv-ffn-v1-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --case generate --mode stock \
  --output outputs/new-base-joint-qkv-ffn
```

匹配对照使用同一新库，只去掉`--private-qkv`。编辑两侧同时添加
`--case edit --runtime-prefix-snapshot`，保留强基线。单独timeline进程验证两个
固定workload model identity（FFN一个、QKV一个），热compile/load均0；原有
单模型observer合同不放宽，joint必须显式选择，按每模型/每请求核对真实预测。
joint图不是两颗ANE，host API spans也不是硬件重叠证明。

QKV固定512-row/H4096/N12288，无checkpoint常量，一份约62.77MiB surface槽位，
逐层直接stage原Q/K/V权重；GPU计算另512行，原attention消费者收到完整QKV。
仅1024目标rows使用QKV ANE，其它长prefill先在GPU；FFN继续用固定1024图分块。
因此可同进程变换参考长度，不通过热重编译适配。FFN/QKV共享原2GiB optional
预算，288MiB QKV估计不是实测RSS；整进程memory另外采集。

`qwen_private_qkv`回执包含成功预测/完整block/stage、GPU计划、失败回退、首次
真实projection的GPU数值检查与累计spans。数值校验不是额外ANE预测；每个原
checkpoint generation一次。失败后sticky关闭，不在热路径重建/重试，完整GPU
重算并计数；collector拒绝这样的窗口成为成功ANE样本。公开build保留同一PImpl
布局，未启用Private时返回不可用，不暴露Private实现。

普通生成每请求QKV1248次；prefix编辑首请求1248，命中后1280。
FFN仍为生成1312、编辑miss1440/hit1280，首请求能力检查的3次evaluate单列。
`stage_lifetime_seconds`包含norm/前序GPU完成等待，不是纯staging kernel时间；
QKV/FFN指标不可相加当关键路径。组件约5%的结果不能外推joint整请求收益。

全部设备窗口结束后，使用`joint_audit.py --off ... --on ... --case generate|edit
--library ... --native-source-archive ... --timeline ... --reverse-off ... --output FRESH`
重读原PNG/请求/source/memory/selector/native计数，比较增量像素与强基线。
可用`--gpu-reference FROZEN_PNG_DIR`只增加相对原GPU的质量观察，不借旧时间作
新QKV开关分母。W8仍需多prompt语义验证；工程完整不等于所有硬件counter都有。
异构`--scenario sequence-matrix`也可添加`--private-qkv`；其审计按两个固定图
核对实际调用，长prefill不假报成QKV ANE工作。

历史命令中的库须与其封存源码匹配。当前新增QKV捕获准入的构建为
`build/qwen-base-qkv-capture-v1-20261018`；旧v8/prefix-v1的结果不重标为新库
性能。要在当前源码上复跑，使用匹配当前sources的新库并建立新窗口。

目标是用完整请求定位瓶颈，再以独立、无观测的匹配窗口确认优化。只使用原
BF16 base、512²、40步，不允许 LoRA/student；图生图保留单张标准1024参考。
不修改 App、产品默认、安装的 MLX、系统设置或其他进程。

不启用temporal DiT cache。GPU重复编辑保留原本默认的immutable prefix
snapshot（首轮miss、随后hit），runtime路线现有规则不准入snapshot；报告从
实际selection记录这一区别。生成无参考、不存在编辑snapshot。不要为了增加
hybrid倍率关闭既有GPU强基线，也不要把两者称为同prefix状态的局部消融。

## 固定 size 的 runtime ANE

固定图为1024 token rows /4096 hidden /4096个 ANE FFN通道，GPU完成另8192
通道；同一无 checkpoint 常量的图接受每层 gate/up/down 权重输入，跨32层和
全部40步复用。首步较长文本/参考序列分块消费同一形状，不裁参考，不按层
重编译。Private W8/H128-H512 与 F32 partial join 是显式近似，不是 BF16
逐位无损或已证明的物理 INT8 执行。
`--channels 5120|6144 --route runtime`可选择另一个固定分区：仅在进程初始化时选择，
随后全部层/步不改变shape；不是每次请求动态重编译。默认仍4096。cache路线
限定4096，避免把未测完整cache bank的较大分区误当已测组合。

实际 hot `compileModel`、`loadModel` 必须为0，实际 evaluate 的 model身份须
只有一个；同时拒绝 overflow retry、headroom变化和 GPU fallback。初次编译/
加载独立报告，不能把它们隐藏后声称冷启动零成本。Private driver内部不可见
编译不在 selector计数的证明范围。

## 覆盖的观测层

| 模式 | 采集内容 | 用途和限制 |
| --- | --- | --- |
| `stock` | 完整请求、原生阶段/计数、load/memory、PNG | 首项独立，后两项热请求；不加载采集器，作为性能分母 |
| `timeline` | 上述＋真实 compile/load/unload/evaluate API spans、Metal CB GPU时间窗、进度与40个采样步 | 找编译/装载、队列间隙、交接与关键路径；不是纯 shader/ANE硬件时间 |
| `gpu-ops` | GPU block0 的 norm/QKV/attention/out/FFN等原生诊断边界 | 插入同步并改变该block编译路径，不能作普通GPU性能分母或外推全部层 |
| `runtime-layers` | 全部 runtime FFN的layer/rows/stage/GPU/ANE/join诊断 | 关闭 fixed-async，使用有同步的诊断路线；不能与stock数字混用 |

完整请求已包括采样、原 encoder/参考处理、VAE、PNG；脚本的后置JSON/trace
处理不在请求分母。runtime计数按同一 graph-owner 的 session累计量做逐请求
差分，重置/回退拒绝。stage work与等待可以重叠，绝不相加冒充关键路径。
`graph_build_seconds` 在 runtime callback中包含实际同步/设备等待，不是纯CPU
构图成本；保留原件，分析不据此归因CPU瓶颈。

Metal使用公开commit完成回调和 `GPUStartTime/GPUEndTime`，先用真实64-byte
blit验证完成、数值与host/GPU时钟包络。只覆盖所发现具体CB类的当前进程实例；
queue编号进程内有效，GPU窗口取并集不重复累计重叠。窗口包括buffer内间隙，
不是所有shader busy time，未捕获其他类/框架/外部进程的工作。
初版分析中 `api_host_span_seconds["metal.command_buffer"]` 是commit到completion
的累计存活区间，包含排队并可互相重叠，绝不是CPU执行秒数；不能与request
wall相加或据其大于wall判断CPU拥堵。各ANE API键才是对应host调用包络。
当前字段已改名为`api_or_cb_lifetime_seconds`，trace也将CB标为completion
lifetime类别。旧原件保留，不替其改写字段或源码身份。

Private selector观察器只转发原方法一次。它与completion回调、保留queue、锁
和计数都有开销，故仅诊断子进程安装，退出排空后恢复原IMP。若原方法来自
父类，新增的子类method slot仍留到子进程退出。未注入系统或其他应用。

## 构建与运行

从仓库根执行。所有输出必须是新目录；无下载或模型转换。

```sh
cmake -S tools/profiling/qwen_base -B build/new-qwen-base-profiler \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2
cmake --build build/new-qwen-base-profiler -j 2

.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen512-async-step-pair-v8-20261018/libturbocider.dylib \
  --observer build/new-qwen-base-profiler/libqwen_base_profile.dylib \
  --route runtime --mode timeline --output outputs/new-base-runtime-profile
```

route为`gpu`、`runtime`、`cache`。后两者同一固定图，仅`cache`使用1600MiB
converted-code native-surface cache；在既有2GiB optional budget内保留图/
scratch空间，不扩大产品预算。直接给缓存2GiB会被图/bank admission拒绝，
不能将其随后走GPU的图像叫成功ANE试测。

先跑`gpu/runtime/cache`的`stock`，再在新进程分别跑诊断模式。`--case edit`
测试编辑，默认参考为既有蓝茶壶；`--requests 1`只作首轮诊断，不授予hot无编译
证据，默认3次为73→74→73。同一候选至少再反序匹配复测，确认RGBA恢复、
cache开关一致性及相对GPU的累计画质，才决定是否采用。

另可用同一图检查标准参考编辑的长prefill分块：

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen512-async-step-pair-v8-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --case edit --mode timeline \
  --output outputs/new-base-runtime-edit-profile
```

不会为了取得安静窗口停止用户ComfyUI。持续竞争CPU负载、完整process-tree
memory与swap保留；安静CPU也不证明GPU独占，当前工具始终不授予正式性能资格。
模型窗口期间不并行自己的编译、测试或CPU数组审计。

## 产物与复核

- `trial/request-*.json`、`report.json`、PNG：真实请求与完整结果。
- `trial/observer.json`：有界、完整、无drop的原生selector/CB记录；缺记录拒绝。
- `stderr.log`：原生步骤、block/layer诊断JSON，不与stock分母混合。
- `profile-analysis.json`：逐请求编译/加载、CB窗口并集、阶段及runtime差分。
- `trace.json`：Chrome Trace/Perfetto格式，可导入timeline viewer查看host/queue。
- `summary.json`、`source.json`、load与memory证据：实际库/源码/observer/模板身份。

模型运行全部结束后，可单独CPU重算：

```sh
.venv/bin/python -B tools/profiling/qwen_base/analyze.py outputs/new-base-runtime-profile
.venv/bin/python -B tests/native/test_qwen_base_profiling.py
```

历史原件重导出时添加`--output outputs/new-derived-profile`；新目录保存派生
analysis/trace，不改历史目录。当前Chrome/Perfetto输出使用整数pid/tid，request
与progress为host lane0、GPU各queue独立编号；原始事件和时钟单位保持不变。

`audit.py`在所有模型窗口结束后重读原始PNG、实际request、selector/CB记录、
native差分、memory digest、库/工具当前或封存源码。它不重跑硬件。首版工具
归档和后续版本分开校验；失败的2GiB缓存窗口另记GPU fallback，不进入成功
runtime分母。不要把诊断的phase总和、CB存活区间之和当成request wall。

当前不提供每个Metal shader counter、ANE硬件时长、物理GPU/ANE重叠或外部服务
归因。这些未知不能靠API名称或时间段相交补齐。工程的覆盖和每条模式的实际
运行资格分开记录；一个self-test通过不等于全部模式/图像质量已验证。

## 同进程变长序列检查

`--scenario sequence-matrix --requests 4 --mode timeline --route runtime`
运行四个完整请求：文生图→单张1024参考编辑→两份相同1024参考编辑→原文生图，
全部seed73、40步。固定6144或4096分区不变，检查模型身份、每次compile/load、
实际reference tokens及分块数；首尾生成必须RGBA exact。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen512-async-step-pair-v8-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --scenario sequence-matrix \
  --requests 4 --mode timeline --output outputs/new-base-sequence-matrix
.venv/bin/python -B tools/profiling/qwen_base/sequence_audit.py \
  outputs/new-base-sequence-matrix \
  --library build/qwen512-async-step-pair-v8-20261018/libturbocider.dylib \
  --output outputs/new-base-sequence-audit.json
```

不同任务的耗时不pool成hot median；该字段刻意为null。重复同一参考只验证长序列
和状态恢复，不等于两个不同主体的语义质量。普通`repeat`模式仍是1/3请求。
`--prompt`可指定单进程固定提示词；默认保留已有茶壶任务。

## vpipe参考路线

`vpipe_probe.cmake/.cpp`在独立build目录链接未修改的参考源码；
`vpipe_compare.py`绑定checkout revision、全tracked源码、submodule revision、
编译/链接recipe、库hash、输出数组与process-tree/load证据。只调用Qwen2.1
BF16 base，无LoRA或量化checkpoint；输入是合成latent/conditioning。
每进程一种路线，分别测首轮与3轮prefill/decode，不包含encoder、sampler、
VAE或PNG，绝不能同上面的完整请求秒数直接作倍率。

本地Comfy权重的`gate_up`与vpipe分开的`gate_layer/proj`命名不兼容。
`vpipe_view.py`用APFS `clonefile`创建两份CoW header view，不复制或改写payload，
再用原source的symlink与index组合出vpipe可读的矩阵范围。无physical-copy fallback。
两个view有约26.5GiB**逻辑**大小，数据extent由APFS共享；不能按逻辑大小估算新增
物理占用。view保留未使用payload gaps，仅适用于此vpipe reader，**不是通用
safetensors导出**。原文件不改动，名称/shape/offset、文件generation及每投影
边界字节单独检查；没有全payload hash的声称。

```sh
cmake -S ../references/vpipe -B build/new-vpipe-reference \
  -DCMAKE_BUILD_TYPE=Release -DVPIPE_BUILD_PYTHON=OFF \
  -DCMAKE_PROJECT_vpipe_INCLUDE="$PWD/tools/profiling/qwen_base/vpipe_probe.cmake" \
  -DTC_VPIPE_PROBE_SOURCE="$PWD/tools/profiling/qwen_base/vpipe_probe.cpp"
cmake --build build/new-vpipe-reference --target qwen_vpipe_reference -j 4
.venv/bin/python -B tools/profiling/qwen_base/vpipe_view.py \
  --output outputs/new-vpipe-header-view
.venv/bin/python -B tools/profiling/qwen_base/vpipe_compare.py \
  --library build/new-vpipe-reference/libqwen_vpipe_reference.dylib \
  --checkpoint outputs/new-vpipe-header-view --case edit --arm ffn-qkv \
  --mode stock --reps 3 --output outputs/new-vpipe-ffn-qkv-edit
```

分别用`--arm gpu|ffn|ffn-qkv`、`--case generate|edit`收集；另开进程用
`--mode timeline --case edit --arm ffn-qkv`取得实际selector/CB观测。
vpipe默认2048-row图对512生成/1024-row decode不执行模型ANE工作；长prefill
初始化后armed也不等于decode有预测。AUTO控制器会改变工作份额和混合精度，
重复合成输入不强求ANE路线输出逐位相同；需保留每轮actual calls/误差。

全部设备窗口结束后运行`vpipe_audit.py --cohort ... --view ... --output FRESH_DIR`；
GPU两类任务的cohort须包含在内。历史工具改版用`--source-archive`指定封存源码。
审计重读数组、native credit和实际Private selectors，导出整数lane trace；hot
两图共享、零compile/load与物理INT8/硬件时长是不同结论。脚本status=complete
只表示证据复核完成，不能代替速度/质量资格。

## Base runtime-prefix复用实验

该功能需要新构建的`qwen-base-runtime-prefix-v1`或后续库，旧v8没有这个实验。
默认关闭，不改变普通GPU snapshot或App设置。新选项只接受resident512、base40、
标准1024参考的cache-off runtime `stock/timeline`重复编辑：

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-runtime-prefix-v1-20261018/libturbocider.dylib \
  --route runtime --channels 6144 --case edit --runtime-prefix-snapshot \
  --mode stock --output outputs/new-base-runtime-prefix-on
```

对照使用相同新库、相同命令但不加`--runtime-prefix-snapshot`，不能拿旧库差异
当开关收益。首请求实际建库并报告published；随后要求hit、40个真实target-only
步骤和1280次模型预测，不能用陈旧PNG或少算采样步代替。新receipt为
`qwen_runtime_prefix_snapshot`，包含admitted/hit/published/bytes/recipe digest。

Transformer的默认callback guard没有放宽。Session完成runtime drain、清除
callbacks后才导出叶值/紧凑独立K/V，snapshot标记runtime arithmetic identity；
默认GPU导入拒绝这个身份，反向或配方不匹配也拒绝。后端、分区、manifest与
executor策略、checkpoint generation、conditioning/refs/geometry变化会失效；
runtime失败、headroom/retry变化、取消、异常也不发布新bank。
原有min(8GiB,物理内存/8)和export headroom限制仍在；不是外部服务的内存硬界限。

现有`--resident-prefix`调用保留整个Transformer的旧入口，两种prefix方式互斥。
本机实际被公共runtime准入规则拒绝，保留作负控，不能标成成功速度对照。
它不是compact snapshot，也不借用后者的recipe receipt或独立owner证明。
长序列支持、首/热wall、内存和画质均需分别测量；单参考命中不代表fresh prompt、
文生图或不同主体的多参考也获得同样收益。

全部设备窗口结束后使用`runtime_prefix_audit.py`复读matched off/on、可选timeline、
reverse-off、resident及native source archive；只读旧v8 PNG作为disabled回归，
不把旧时间作为新开关的分母。即使增量RGBA exact，原W8路线与GPU BF16仍可能近似。

## Private QKV真实base组件

`qkv_collect.py`在原base40调度的首个decode block0捕获真实输入，比较原完整GPU
block、拆分GPU控制、已备W与每次重备W的Private候选。它在模型step1 layer0
退出，不导出PNG，也不是完整40步图片benchmark。FFN在GPU，不能替代与当前
FFN-ANE组合的整请求验证。支持固定256/512行，single/tiled两种图布局。

```sh
TURBOCIDER_NATIVE_OUT=build/new-base-qkv-probe \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen-base-qkv-capture-v1-20261018 \
  bash tools/native/build_qwen_base_qkv_probe.sh
.venv/bin/python -B tools/profiling/qwen_base/qkv_collect.py \
  --library build/new-base-qkv-probe/libqwen_base_qkv_probe.dylib \
  --native-library build/qwen-base-qkv-capture-v1-20261018/libturbocider.dylib \
  --case generate --chunk 512 --layout single --mode stock \
  --output outputs/new-base-private-qkv
```

编辑使用`--case edit`保留标准1024参考；`--mode timeline`另开进程观测真实
compile/load/evaluate和CB。Observer初始化后，在任何source加载/encoder动作前
须设scope(-1)，-2只属于blit self-test；不得把source阶段误作额外校准事件。

`qkv_audit.py --cohort ... --probe-library ... --native-library ...
--native-source-archive ... --output FRESH_DIR`重读完整数组、原checkpoint两层六个
QKV payload、实际计数与源码。`--cpu-oracle`另做全部1024行CPU F32/BF16参考；
CPU/NumPy警告保留，只有设备窗口全部结束后再运行。`--failed`保留前置拒绝或
观测校验失败，不将它们算成性能样本。支持`--tool-source-archive`校验历史工具。
