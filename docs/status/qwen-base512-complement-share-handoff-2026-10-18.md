# Qwen base512：互补激活下的分工与输出交接瓶颈

后续已完成[restore encoder GPU时间戳与gate/up合并消融](qwen-base512-ffn-handoff-encoder-2026-10-18.md)。
设备侧encoder区间也复现变长，但仍不等于独占shader成本；packed GPU候选无
新收益，当前native保持不变。本页原始完整请求数字及范围不改写。

2026-10-18，M4 Max/64GiB。继续[固定FFN激活优化](qwen-base512-ffn-activation-2026-10-18.md)，
仅原BF16 base、512²、实际40步文生图/普通参考编辑，无LoRA/student/Z Image，
无temporal DiT cache。当前659-input native未改动、未重新构建。

结论：保留FFN6144＋QKV512作为当前速度候选。新激活下缩减到FFN5120反而慢
约5–6%；不能将较长的`async_ane_wait`简单解释成ANE计算变慢。新审计已把
worker预测等待、输出完成等待和caller等待按请求分别导出，未改写历史原件。
这是瓶颈定位与负结果，不是新一轮加速成功，也不声称所有瓶颈已解决。

## 固定图、匹配请求与完整耗时

两档都固定FFN1024 rows/H4096、QKV512 rows/H4096/N12288，跨32层/40步共用
两份无checkpoint权重常量的runtime图。每进程只在初始化时选FFN通道数，之后
不切shape；5120的GPU complement为7168，6144的为6144。都使用
`--fixed-program --ffn-activation sigmoid-complement`、descriptor投影、F32
partial join、cache off、prefetch off。两档只允许CHANNELS环境变量不同。

stock不加载observer；每进程seed73→74→73，首请求单列，后两次取中位数。
完整wall含原encoder/参考处理、40步采样、VAE和PNG。编辑双方都保留标准1024
参考及相同runtime-prefix snapshot；热请求均hit，不靠降低参考分辨率提速。

| 任务/FFN份额 | 首次完整请求 | 后两次完整请求的中位数 |
| --- | ---: | ---: |
| 生成6144 | 34.091825s | 30.029953s |
| 生成5120 | 34.036539s | 31.864343s |
| 编辑6144 | 45.531184s | 36.805848s |
| 编辑5120 | 47.678713s | 38.672756s |

5120的wall分别增加6.11%/5.07%。独立反向6144控制回到生成30.048203s、编辑
36.897854s，全部同配方PNG恢复逐像素一致。不能将旧Exp下5120/6144接近的
排序直接继承到新激活，也不借旧完整GPU秒数作本轮分母。

四个stock和两个reverse窗口均完整3×40步。生成每请求FFN1312/QKV1248预测；
编辑miss1440/1248、hit1280/1280。每请求成功FFN blocks1280，无fallback、
overflow retry或headroom变化，两个owner都保持一个Program。

## 关键发现：等待从prediction移到输出完成

以下是每次实际请求相对前一次session累计量的差，再取后两次中位数，单位秒。
不能将三请求累计值当成一张图，更不能把这些不同线程/有重叠的span相加成wall。

| 请求 | FFN worker prediction等待 | FFN worker输出完成等待 | FFN caller finish等待 |
| --- | ---: | ---: | ---: |
| 生成6144 | 13.568396 | 0.514790 | 14.038282 |
| 生成5120 | 11.461867 | 4.519562 | 15.985903 |
| 编辑6144 | 13.224242 | 0.489429 | 13.654294 |
| 编辑5120 | 11.240971 | 4.452375 | 15.642927 |

源码边界：`private/ane_w8_executor.mm::run`在`request.finish()`周围累计prediction，
在随后`io.finish()`周围累计output；后者等待GPU restore/validation的完成。
`ane_ffn.cpp`在主调用方等待更宽的后台完成路径，记录async wait。三个数字
不是三段互不重叠的硬件时间，也不是纯ANE计算、纯拷贝或CPU构图成本。

两档相同任务的逻辑输出字节完全一致：生成每请求22011707392 bytes（20.5GiB），
编辑hit21474836480 bytes（20GiB）。这是多次FFN的逻辑输出累计量，不是峰值
内存或测得的物理DRAM流量/带宽。按intermediate通道切分后，两支仍各返回
完整H4096 partial，所以减少ANE通道不会减少这一输出维度。

5120让prediction侧暴露等待约少2秒，却让output侧约多4秒，整体较慢。
QKV join生成约4.54/4.52s、编辑约4.68/4.65s，基本没有同等级变化；FFN的
pre-FFN边界也未增加。这支持先查FFN输出交接及其与较大GPU complement的
关系，不支持“更小ANE图计算更慢”或“同字节量的拷贝带宽下降”这一唯一解释。

`ane_gpu.mm::Transfer::submit`提前提交ready/ANE-done依赖及restore command
buffer，`finish`等完成通知。当前observer只标识队列/CB，并没有把每个CB映射
成FFN restore、stage或complement；CB window还可能包含event等待。因此
queue竞争、调度和完成回调都是待隔离假设，不能从现有host wait确定物理根因。

`share_audit.py`新增`completion_deltas`和scope声明，原始采集器、analyzer、
observer与native均未改。完整负控拒绝missing/type/bool/非有限值、计数回退和
request index错序；v2 audit把每窗三次差分分别保存，未把累计量伪装成单请求。

## 实际selector与输出复核

5120编辑独立timeline完整结束，94961条记录，dropped/pending均0：

| 模型/请求 | 首次编辑 | 第二次 | 第三次 |
| --- | ---: | ---: | ---: |
| FFN evaluate | 1443（含3次能力检查） | 1280 | 1280 |
| QKV evaluate | 1248 | 1280 | 1280 |
| compile/load API | 1/2 | 0/0 | 0/0 |

只有两个稳定模型身份；首次compile约0.968126s、load合计0.005936s单列。
timeline热中位38.592251s只作诊断，不替代stock38.672756s。全部timeline PNG
与5120 stock exact。两份v2独立审计都重读source/native/archive/request/memory
及PNG，并核对反向恢复。5120生成只有stock证据，不借编辑或6144 timeline
给它授予实际selector零热编译证明；默认6144的生成/编辑及变长证据见上轮报告。
client API观察也不覆盖不可见driver内部活动。

两张新contact sheet均实际查看。5120和6144的生成seed73壶嘴/把手几何有差异，
seed74接近；编辑大结构一致而有局部纹理差异。增量生成RGB SSIM为
0.959373/0.986032，编辑0.997636–0.997822。相对冻结完整GPU，5120生成SSIM
0.924519/0.989226，6144为0.902176/0.977323；5120在这两个seed上较接近GPU，
不推导通用质量更优。W8与互补激活仍有损；高SSIM不授予材质、身份、文字或
hard-mask inpainting资格，原饱和红/强高光风格也不归因于本次分工变化。

## vpipe与灵活runtime的取舍

只读复核本地vpipe仍为`18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`，worktree干净。
本轮没有重跑其forward；已有本机同版本ANE开/关实测见
[固定runtime与vpipe报告](qwen-base512-runtime-ane-profile-2026-10-18.md)。

其默认2048-row chunk在1024 decode无合法share，实际ANE预测0；编辑prefill
曾预测，AUTO后来退回GPU并周期probe，未观察到稳定forward收益。开关开启不等于
硬件做了工作；forward也不能与含encoder/VAE/PNG的40步整图wall算倍率。

当前选择固定FFN1024和QKV512两图，长参考prefill重复FFN chunk，QKV长prefill
交GPU，host按合法shape路由；不是逐block/逐sequence编译模型。图小也不会消除
原GPU权重、动态weight slots、prefix/KV、长attention或encoder的内存/耗时。

后续优先做能区分原因的消融，而不是继续盲减ANE份额：给restore/validation
和complement建立显式对应的诊断完成跨度，比较各自独立与并发完整block，再
测试安全的队列/consumer融合。必须保留producer、ANE真实完成、有限性检查、
取消/失败drain、F32 join和完整GPU重算；减少同步不能以提前复用slot换取。
任何native修改需要新构建/source archive、同库强基线及完整图复验。

另一条只读候选是重复文生图的text-prefix复用：源码中prefix使用零timestep
调制，文本segment只attend自身/此前tokens，现有snapshot结构支持空reference。
但当前runtime snapshot准入明确edit-only，还没有新的text-prefix开关或设备
off/on证明。不能静默扩大编辑flag或声称已经获得收益；缓存key必须包含实际
conditioning、source/recipe、形状与紧凑叶值所有权，且不依赖seed。

## 原件、构建与测试

- 原件：`outputs/qwen-base512-complement-share-c{6144,5120}-{gen,edit}-stock-v1-20261018`。
- 反向：`outputs/qwen-base512-complement-share-c6144-{gen,edit}-reverse-stock-v1-20261018`。
- timeline：`outputs/qwen-base512-complement-share-c5120-edit-timeline-v1-20261018`。
- 最新审计与contact sheets：`outputs/qwen-base512-complement-share-{gen,edit}-audit-v2-20261018/`；旧v1保留。
- native：`build/qwen-base-ffn-activation-v3-20261018/libturbocider.dylib`，SHA256
  `e978d1fec176366c402e20fff296c0839c3a7960875f6452a6d64020b2f8605e`。
- runtime ID：`tc-runtime-build-v1-ab01cef2a455d9aa65e19dc3ee05f1992227877a769ff8299f67794f3c943b3e`。
- 659-input source：`outputs/qwen-base-ffn-activation-native-source-v3-20261018.tar.gz`，
  SHA256 `afc3606df103b1035f2948a6300f4da18ace6e93b4208911b9aa24ddd8379cf2`。
- 本轮最终工具/tests封存：`outputs/qwen-base-complement-share-tools-v2-20261018.tar.gz`，
  摘要独立保存于同名`.sha256.json`；旧activation-tools-v4保留。初次工具包装v1
  含macOS AppleDouble metadata，路径逐项校验因此失败；v2禁用该metadata后重包，
  不涉及native或任何模型测量的失败/重跑，v1亦保留。
- 最新79项profiler host测试通过，日志
  `outputs/qwen-base-complement-share-host-tests-v2-20261018.log`；不是79项硬件测试。

七个进程共21次完整请求/840个实际steps全部exit0；最后串行wrapper session69462
已终止，不应重启它。自有CPU数组审计/测试/看图均在这些模型窗口结束后进行。
进程树memory证据完整，峰值physical footprint约17.45–20.12GiB；系统swap-in/out
均0，部分后续窗口有系统compression/decompression，不作单模型独占归因。
全部窗口仍记录竞争CPU推理负载，无设备独占、thermal或统计资格，统一
`qualification_passed=false`。没有下载/删除、停止ComfyUI、改references/App/默认。

优化目标仍ACTIVE；6144仍为显式有损研究候选，默认Exp和普通GPU保持不变。
