# Qwen base512：固定ANE图下的GPU complement融合验收

2026-10-19，M4 Max/64GiB。仅原BF16 base、512×512、实际40步文生图/图生图；
无LoRA/student/Z Image，无temporal DiT cache。接续
[QKV单组权重前看](qwen-base512-qkv-lookahead-2026-10-19.md)和
[dual-GUP完整block筛选](qwen-base512-dual-gup-block-2026-10-19.md)。

本轮已完成13个串行设备窗口、41次完整请求/1640实际采样步，以及独立全量验收。
增量完整RGBA exact，但生成收益只有约0.13–0.20%，编辑约0.30–0.89%，不足以
授予稳定速度或默认采用资格。新增功能保留默认OFF；推荐研究基线仍是原
fixed FFN/QKV＋one-hop QKV前看，不改变App、安装库或产品默认。

## 固定图与动态长度

ANE仍然只有两个operator各自的一份Program：FFN1024/H4096/C6144和
QKV512/H4096/N12288。原gate/up/down或Q/K/V作为动态输入跨32层复用，
没有给每个transformer block嵌入一套常量权重，也没有新增ANE bank。

FFN按intermediate通道分担，GPU/ANE各6144，F32 partial join；长prefill按
同一1024-row图分块并padding尾块。QKV按目标行分担，GPU512/ANE512；长prefill
保留GPU。sequence变化只改变调度次数，不为每个参考数创建新的ANE图。
text-prefix命中仍保留step0 GPU QKV，后39步Private，不能重引入已知画质反例。

W8/A8是本实现的输入表示和量化数据流，不是物理INT8 MAC证据。小Program
不等于整个进程只有这一点内存：原BF16权重、IOSurface、GPU/ANE staging、
encoder/KV和driver均需保留在process-tree峰值中。零热编译结论限定实际观察到
的Private client compile/load selectors，不承诺opaque driver内部行为。

## 本轮实现

新增`TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU=1`，collector对应
`--gpu-dual-swiglu`。只准入显式joint resident512/base40/fixed6144、F32 join、
sigmoid-complement、untimed fixed1且有实际submission fence的原dense BF16路线。

只替换1024行GPU complement：同一custom kernel内直接读取原gate_up physical
pitch，执行两次MPP投影并融合typed BF16 SwiGLU；固定register/BM32/BN64。
不预拼接每层权重，不增加checkpoint副本；长prefill仍走原GPU回调。ANE图、
权重slot、QKV one-hop lookahead以及原F32 down projection均不变。

每个owner对真实model source generation只验证一次：首个实际hidden与F32
partial对普通GPU比较完整bits并检查finite。失败立即返回原GPU结果、sticky
停用，不靠重试或重编译续跑。source改变也停用，需要新owner；显式unload重置。
GPU策略进入prefix arithmetic namespace，停用/失败不能发布可复用prefix。

JSON `qwen_gpu_dual_swiglu`记录实际attempt/completed/ordinary/fallback、
session累计validation calls/time和`extra_owned_weight_bytes=0`。健康开启的
miss为1248 fused+32 ordinary，prefix hit为1280 fused；关闭时1280 ordinary。
验证不是每层每步重做，也不是只检查一个伪造fixture后授予真实模型资格。

## 同库stock完整请求，不是稳定加速结论

同库stock三请求seed73→74→73，完整40步、encoder/VAE/PNG均计入wall。
双方保留QKV前看和正确text-prefix/GPU-step0强基线，仅新开关不同。

| 配对 | off热中位 | on热中位 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 文生图首配对 | 28.840978s | 28.803325s | 0.131% |
| 文生图反序off/独立on | 28.876982s | 28.818377s | 0.203% |
| 单参考重复编辑 | 36.304769s | 35.980719s | 0.893% |
| 编辑反序off/原on | 36.088638s | 35.980719s | 0.299% |
| 材质提示词生成 | 28.867942s | 28.829452s | 0.133% |

首配对first31.533620/30.415805s，不能将首调差异归给约60ms的首次真实验证，
也不混入热中位。编辑first44.951012/44.991761s；材质first31.118436/30.637248s。
不根据first差异声称cold-start收益。每个on owner首次实际source验证约23.3–60.1ms，
calls一直为1、累计seconds在后续请求严格不变。off没有验证或fusion credit。

五组配对的15次完整RGBA比较全部exact（编辑on按两组控制各比较一次）；同recipe
反序/独立重复/timeline另有36次完整重放通过。新off与旧冻结生成/编辑的6张完整
RGBA也exact；历史时间仅作回归背景，不作本轮开关分母。pilot原审计保留，最终
全量审计见`outputs/qwen-base512-gpu-dual-full-audit-v1-20261019/audit.json`。
这些是本轮增量位控制，不消除原W8相对纯GPU的近似差异，也不授予全部prompt
的语义质量资格。收益很小，不能把局部block收益乘1280写成整图节省。

## 瓶颈：缩短FFN也会改变QKV准备窗口

真实stock回执的每热请求累计差分显示，生成FFN caller约14.65–14.76s降到
14.41–14.47s，async wait约13.81–13.91s降到13.55–13.64s。但QKV weight join
暴露等待同时从约0.011–0.014s升到0.182–0.194s；材质提示词也出现相同方向。
编辑off约0.013–0.025s、on约0.194–0.203s。新融合保留相同one-hop命中计数，
但命中不等于producer已完成，不能只看命中率宣称成本没有变化。

因此局部FFN包络缩短与更多QKV准备暴露并存，完整请求净收益远小于FFN累计差。
这是由实测支持的调度取舍，不是已唯一定位GPU带宽/硬件占用率。各host span可
重叠且含设备等待，不能相加成wall、硬件ANE时间或物理并行比例。热生成denoise
仍约28.1–28.4s，VAE约0.57–0.61s；继续压缩VAE并非当前主机会。

下一步应单独筛选QKV producer与GPU complement的提交前沿，保持单bank和原
completed/drain/source合同，验证是否能保留FFN缩短而不重新暴露权重等待；
还不能直接打开更远预取/更多bank。长参考仍有attention/KV、padding和内存成本。

## 四个独立timeline：变长不新建ANE图

生成、重复编辑、sequence矩阵和text-conditioning矩阵各在独立进程采集。每窗
实际evaluate model身份均为2；后续请求实际compile/load均0，native成功调用数
与每model observer计数匹配，首轮3次FFN capability evaluate单列。Metal完成/
时钟标定通过，记录无drop/pending并真实drain。初次load仍存在；本轮没有把
文件缓存命中的初始化称为严格冷启动或driver永久零编译。

同一owner的实际sequence矩阵，不开prefix复用：

| 请求 | prefill rows | FFN/QKV预测 | 完整wall |
| --- | ---: | ---: | ---: |
| 文生图 | 1048 | 1312/1248 | 30.753247s |
| 一份标准1024参考 | 5155 | 1440/1248 | 44.265628s |
| 两份标准1024参考 | 9257 | 1568/1248 | 57.344628s |
| 恢复原文生图 | 1048 | 1312/1248 | 29.109924s |

每次fusion1248、ordinary长prefill32；首尾完整PNG exact。两份参考为同一文件
重复，只证明长度/状态恢复，不授予多主体语义资格。不同任务的hot median为null，
没有off矩阵，不从这些时间或上一轮矩阵推算变长融合加速百分比。

text矩阵依次为原prompt miss、换seed hit、换prompt miss、恢复原prompt miss。
单prefix bank被替换，最后必须真实重算，不能伪报hit；FFN1312/1280/1312/1312，
QKV均1248，fusion1248/1280/1248/1248。实际prompt/请求核对及首尾完整PNG恢复
通过，没有因缓存身份不同而跳过采样步。首次GPU验证在两种矩阵内都不重复。

## vpipe对照与灵活调度的边界

只读复核`../references/vpipe@18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`，
checkout干净，未改参考工程。此前本机GPU/ANE FFN/FFN+QKV对照、实际selector、
误差和完整证据见
[profiling与vpipe报告](qwen-base512-runtime-ane-profile-2026-10-18.md)。

其默认2048-row chunk对1024-row decode没有合法非零share；打开开关不等于有
实际ANE预测。此前本机生成三路线decode约1.0617/1.0619/1.0622s，ANE预测均0；
编辑长prefill有调用，但AUTO后来退GPU并周期probe。这是合成DiT forward，
不能直接与包含encoder/sampler/VAE/PNG的40步图片秒数作倍率。

适合本任务的组合是固定operator图、动态原权重、有界bank、按phase/rows/precision
隔离的调度状态、stage前zero-share判断，以及真实完成后的bank复用。两个图
仍竞争同一ANE资源；不将FFN与QKV单项倍率相加，也不以host时间段相交证明
物理并行。不同参考长度首先改变padding、attention/KV和内存，不能只优化GEMM。

## 构建与复跑

native为`build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib`，SHA256
`7da7a7125bd7cb401344bf885c2425ec7eacc14d81b0563fd45317a94b4b260a`。
Runtime ID为`tc-runtime-build-v1-288ff68716a59f774623c4c84abd985e60f89e92afcdcdd3ccc1531d7d3c08ed`。
661项native input封存于`outputs/qwen-base-gpu-dual-native-source-v1-20261019.tar.gz`，
SHA256 `cab9b206dfaf782c7da7d4f3f91bd64500e1c9511b00a51f1e2c0d6ea8342e36`。

本轮改变内部C++ `RunResult`/Session ABI；旧C++ helper不能混用，必须重新构建
并绑定native身份。C JSON API collector和独立ObjC observer不受这个ABI影响。
构建native时显式设置`TURBOCIDER_BUILD_PACKAGE_ONLY=1 TURBOCIDER_NATIVE_ONLY=1
TURBOCIDER_ENABLE_PRIVATE_ANE=1`，避免继续进入可选App编译。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --private-qkv-prefetch \
  --fixed-program --ffn-activation sigmoid-complement \
  --case generate --mode stock --runtime-text-prefix-snapshot \
  --text-prefix-qkv-step0 gpu --gpu-dual-swiglu \
  --output outputs/FRESH-GPU-DUAL
```

off只去掉`--gpu-dual-swiglu`。编辑改`--case edit --runtime-prefix-snapshot`并去掉
两项text参数。timeline独立进程；`sequence-matrix`用4请求且不带prefix复用，
`text-prefix-matrix`保留text参数。所有模型窗串行，禁止自有build/test/CPU数组
审计/看图干扰；模型窗全部结束后再运行`gpu_dual_request_audit.py`。

## 验收、内存与交付边界

最终auditor逐窗重读完整PNG、实际request、661项native源/封存源码、当前库/
工具/实际model generation及selector。除digest外，重跑process-tree JSONL hash
chain/身份/完整性验证，结果必须与memory report、summary和correlation一致；
GPU验证累计时间要求首后完全不变，不只接受单调增长。

进程树峰值physical footprint约17.54–30.05GiB；系统swap-in/out全部0。sequence
矩阵观察到约1733.25MiB compression，它是系统计数，不归因单独ANE或本进程。
没有新增owned W副本不等于RSS必降，样本间峰值差不当作fusion净内存收益。

最终140项profiler host测试通过：`outputs/qwen-base-gpu-dual-host-tests-v2-20261019.log`。
native严格准入/source sticky状态1项、真实MLX lifecycle1项分别见
`outputs/qwen-gpu-dual-native-policy-v2-20261019.log`和
`outputs/qwen-base-gpu-dual-lifecycle-v2-20261019.log`。后者覆盖24次释放循环、
8种variant和prefix/compiled-graph隔离；不合并为142项图片/硬件资格。

工具、测试和本报告封存于`outputs/qwen-base-gpu-dual-tools-final-v1-20261019.tar.gz`，
摘要索引为`outputs/qwen-base-gpu-dual-artifacts-v1-20261019.json`。保留pilot、
旧库、所有失败日志和原始输出。native构建完成后曾意外进入可选App编译分支，
仅中断自有App build进程；native library和source archive随后重新核验。没有
安装App、替换默认库或停止ComfyUI；磁盘仍约9GiB可用，无需下载或删缓存。

全部性能窗串行，自有测试/数组审计均在设备campaign终止后。窗口存在竞争
推理CPU负载，无设备独占、统计/thermal或物理INT8/overlap资格，统一
`qualification_passed=false`。本轮融合仅保留显式研究入口，不默认采用；更可信的
完整请求基线仍是上一轮约28.84s生成/36.22s重复编辑的QKV前看路线。总体优化
目标ACTIVE，不称全部瓶颈已解决。
