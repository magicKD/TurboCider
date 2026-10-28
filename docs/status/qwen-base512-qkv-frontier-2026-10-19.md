# Qwen base512：QKV权重提交前沿，等待减少不等于净提速

2026-10-19，M4 Max/64GiB。仅原BF16 base、512²生成/标准1024参考编辑来源，
无LoRA/student/Z Image。接续
[GPU complement融合的完整40步验收](qwen-base512-gpu-dual-runtime-2026-10-19.md)。

本轮完成六路调度fixture、生成加倍样本/独立重复、编辑及两种独立timeline。
提前准备QKV确实减少了它的暴露等待，但同时拉长FFN等待；完整消费者没有稳定
新增收益，因此**不新增runtime默认或研究开关，主native、App、完整请求基线未改**。
本轮不是新的完整40步图片纪录，整体优化目标仍ACTIVE。

## 可证伪假设与实际范围

上一轮融合GPU complement后，每热请求的QKV weight join暴露约从0.01–0.02s
升到0.18–0.20s。假设：不增加bank，只把同一组QKV W8 producers提前到GPU
complement之前，或插在hidden与down之间，能保留FFN缩短并减少这个等待。
反例：提前的producer与当前关键路径竞争，减少的等待转移到FFN或其他边界。

同一进程一份固定FFN1024/H4096/C6144和一份固定QKV512/H4096/N12288，动态
读取原权重；三个frontier各配ordinary/dual GPU complement，共六arm：

- `post`：当前native顺序，ANE launch与GPU complement async提交后准备QKV。
- `pre`：ANE launch后，在GPU complement callback构建/提交前准备QKV。
- `hidden`：提交hidden，准备QKV，再让原F32 down partial进入GPU提交。

只使用现有`HybridFfn::ChannelGpu`和`AfterSubmit`接口构建独立helper，不改native。
保持untimed fixed1、launch fence、F32 join、sigmoid-complement、单QKV future、
cache/prefetch-bank off；单future并不等于没有真实W8 producer工作。

来源是实际base40请求完成第0步后捕获的首decode layer0，保留普通GPU QKV、
attention、prefix K/V、完整FFN、残差与完成。其后接一个**依赖该block输出的QKV
投影fixture**，使用真实layer1 Q/K/V权重；前置norm刻意复用layer0 modulation。
因此不是自然模型的layer0→layer1完整forward，不冒充两个真实连续block、
joint当前层Private QKV或40步图片。这个边界只筛选FFN→下一投影的调度机会，
候选若胜出仍必须接入真实pipeline并验收完整40步；本轮没有这样的稳定候选。

## 正式stock结果

每arm2 warm，随后16或32 measured，六arm按rep轮转；输入A/B交替。单位ms，
包括当前完整block、真实FFN重备W、QKV producer、依赖norm、Private预测与输出
完成，不是只计一次join。每窗用自己的`original-post`或`dual-post`强控制，不
借普通GPU或历史图片秒数放大收益。

| arm | 生成16 | 生成32 | 生成32独立重复 | 编辑32 |
| --- | ---: | ---: | ---: | ---: |
| original-post | 28.550625 | 28.589938 | 28.563000 | 34.258459 |
| dual-post | 28.451250 | 28.502105 | 28.547063 | 34.239625 |
| original-pre | 28.555875 | 28.551542 | 28.592125 | 34.374001 |
| dual-pre | 28.534333 | 28.576604 | 28.530604 | 34.190104 |
| original-hidden | 28.512792 | 28.557875 | 28.607230 | 34.276292 |
| dual-hidden | 28.532916 | 28.544625 | 28.588812 | 34.207000 |

例如dual-pre相对同窗dual-post，生成前两窗分别慢约0.29%/0.26%，独立重复仅快
约0.058%；编辑约快0.145%。ordinary前移也在生成重复与编辑反向。不能把每窗
各自最小值拼成新配方，或把微秒差乘1280宣称整图收益。没有统计/thermal资格。
两窗timeline只验证状态/归因，不进入stock排名。

## 具体归因：等待移位

生成32同窗的每次fixture中位数，单位ms：

| arm | FFN caller | FFN async wait | QKV weight join wait |
| --- | ---: | ---: | ---: |
| original-post | 11.3802 | 10.7311 | 0.0074 |
| dual-post | 11.1260 | 10.4610 | 0.2079 |
| dual-pre | 11.3536 | 10.6860 | 0.0112 |

提前提交让QKV等待少约0.197ms，但FFN caller增加约0.228ms。权重命中数、预测
数、bank数完全相同，实际执行的成本没有因“更早hit”而消失。hidden位置也未
稳定解开这项取舍。观察与资源/队列竞争一致，不能据host spans唯一定位物理
带宽或硬件占用率；不将这些累计边界相加成wall或ANE硬件时间。

这进一步缩小了上一轮的假设：只调整producer前后位置不能免费获得FFN和QKV
两侧最短等待。后续应研究减少producer本身或真正的依赖/交接成本，保留完成/
generation/单bank合同；没有证据支持新增更深预取或更多常驻权重副本。

## 保留的pilot与测量修正

`outputs/qwen-frontier-gen16-v1-20261019`是首个调试窗，输出/调用检查完成，但
每次sample后立即CPU hash完整block/QKV数组。虽然hash不在该sample计时区间，
它仍改变下一sample的cache/内存/调度状态；该窗耗时约37–41ms、表观5–6%的
frontier收益不在修正后的窗口复现。不能证明checksum是唯一原因，但不能当作
只要“移出timer”就无扰动的性能资格。

v2只在preflight做完整A/B/换W/A的六arm校验；正式循环内部保证真实设备完成、
新输出与成功计数，保留各arm最后独立owned结果，循环结束后统一hash。没有
通过删掉完成等待来“提速”。最后六arm输出各32MiB，共192MiB激活owner；这是
fixture校验workspace，不是权重副本或产品内存收益。原pilot/源码快照/失败编译
日志均保留，不改写成v2数据。六个正式窗没有其它自有build/test/CPU数组审计。

## 数值、源权重、真实selectors与内存

每正式窗40份完整BF16数组，共240份：原GPU A/B、真实FFN输入A/B、三种输入/
权重fixture×六arm×block/QKV。六arm对应输出完整bits全部exact，输入A/B变化、
layer1→layer0 QKV换W及A恢复通过；换QKV权重不改变其上游block。
同case不同repeat/stock/timeline的全部数组也exact。当前FFN block相对普通GPU
relative-L2约0.00072–0.00086；这是原W8近似，增量调度exact不消除它。
没有独立CPU全QKV/FFN GEMM oracle，也没有多主体图像语义资格。

独立auditor从原checkpoint重读layer0 gate_up/down和两层六份Q/K/V，完整8份
BF16 payload、共480MiB，和真实helper源hash匹配；再验证native661-input archive、
helper ABI绑定、实际工具源码/库/model generation及全部数组extent/dtype/finite。
原始process-tree JSONL hash chain重算必须与report/summary/correlation一致。

正式六窗共864 measured消费者（其中192属于timeline），每个operator1080次
workload预测，共2160次；另有六窗18次FFN capability evaluate。每窗换策略、
输入和QKV层权重时只用原两个Program。生成、编辑独立timeline各264次workload
evaluate，与每scope两个实际model各1次完全对应；hot compile/load0，Metal完成/
时钟自检、drain和记录完整性通过。不是物理INT8、引擎并行或driver永久零编译证明。

进程树峰值physical footprint约16.22–18.31GiB，系统swap-in/out全0；不能将其与
完整40步图片窗口作净内存收益。窗口存在竞争推理CPU负载，无设备独占，统一
`qualification_passed=false`。本轮只完成七次来源请求的首step捕获（含pilot），
没有导出PNG或新的完整40步请求。磁盘约4.8GiB可用，保留原始数据/权重/旧库，
未清缓存、下载、停止ComfyUI或修改App/default。

## 工程与复跑

- `frontier_probe.mm`：完整block＋合成依赖边六路对照，动态原buffer、单bank，
  scalar-only统计与fresh output检查，校验不挤在计时sample之间。
- `frontier_collect.py`：实际132或228次每operator完整计数、六arm严格轮转、
  source/receipt/memory与stock/timeline分离。
- `frontier_audit.py`：独立原payload、全数组位比较、同case重放、selectors与
  memory原始证据；`tests/native/test_qwen_base_frontier.py`新增6项host负控。
- 最终146项profiler host测试通过：`outputs/qwen-frontier-host-tests-v2-20261019.log`。
  不将host测试数与864个fixture或2160个evaluate相加成设备资格数。
- helper：`build/qwen-frontier-probe-v2-20261019/libqwen_frontier_probe.dylib`，SHA256
  `622b8d649b16950df367497888789ed82aa6f633681f3ff3e11f9d887f3e474f`。
- native仍为`build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib`，SHA256
  `7da7a7125bd7cb401344bf885c2425ec7eacc14d81b0563fd45317a94b4b260a`；没有新内部ABI。
- 独立审计：`outputs/qwen-frontier-full-audit-v1-20261019/audit.json`；最终工具封存
  `outputs/qwen-frontier-tools-v1-20261019.tar.gz`，摘要索引
  `outputs/qwen-frontier-artifacts-v1-20261019.json`。

```sh
TURBOCIDER_NATIVE_OUT=build/FRESH-FRONTIER \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen-base-gpu-dual-v1-20261019 \
  bash tools/native/build_qwen_frontier_probe.sh
.venv/bin/python -B tools/profiling/qwen_base/frontier_collect.py \
  --library build/FRESH-FRONTIER/libqwen_frontier_probe.dylib \
  --native-library build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib \
  --case generate --repeats 32 --mode stock --output outputs/FRESH-FRONTIER
```

编辑用`--case edit`；timeline独立进程。设备窗结束后用`frontier_audit.py --help`
指定所有`--cohort`、匹配`--probe-library`、`--native-library`和原native source
archive及fresh输出。不要将旧helper与当前不同ABI的库混用，也不把本轮fixture
秒数与完整图片wall比较倍率。完整请求推荐仍保留原QKV lookahead配置。
