# Qwen base512：Private runtime FFN × immutable prefix K/V

2026-10-18，M4 Max/64GiB。仅Qwen Image2.1 base编辑、512²、40个真实采样步、
单张标准1024参考，原BF16 encoder/VAE与原master权重，不启用LoRA、student、
temporal DiT cache或Z Image。当前是显式实验，App/default未变。

## 本轮实现

目标是消除重复编辑中每次约6秒的长prefill，而不改变已测的固定6144通道FFN
分担。计算图仍为一个1024-row Private W8/H128–H512图，跨32层/所有步骤共享，
GPU完成另6144通道，F32 partial join、fixed async，无weight code cache或预取。

新增`TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT=1`，默认0/未设。只对resident
512²、base40、标准1024参考、无memory override的runtime编辑准入；源checkpoint
必须原BF16，现阶段native允许1–2参考，**本轮性能/图像验证只覆盖单参考**。
实际executor还须为健康的Private Hadamard W8 channel分担、4096/5120/6144之一、
F32 join，无fallback、overflow/headroom改变、S1或强制GPU层，固定1024原base
descriptor。不是开启flag就自动算成功。

保留`Transformer::plain_prefix_route`的原有callback拒绝规则。Session先完成
40步、drain executor/source scope并清除callbacks，再导出独立紧凑K/V叶值。
snapshot新增runtime arithmetic identity；普通GPU、错误recipe或反向未标记bank
不能隐式导入。Source Transformer、compiled closures、执行器、权重与target noise
不由这个snapshot持有。原GPU/frozen/student组合的默认允许参数不改变。

identity包含现有manifest/hash/executor策略、partition、S1/available信息、采样
配方和checkpoint文件generation；conditioning tensor/refs/geometry仍有原来的
身份检查。换条件/来源/路由、取消、异常、runtime失败或recipe不健康时清理或
拒绝发布；不在热路径创建新ANE图。物理文件generation不是完整payload hash。

新receipt `qwen_runtime_prefix_snapshot`明确记录admitted、hit、published、bytes、
recipe digest及decline reason。CLI `--runtime-prefix-snapshot`只接受stock/timeline
重复单参考编辑；工具会拒绝缺receipt、静默fallback、未发布的首轮或少执行步骤。

另检查既有`TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV=1`：Session调用点虽存在，
真实请求却被`qwen21_module.cpp`的runtime准入规则拒绝。尝试保留在
`outputs/qwen-base512-runtime-prefix-retained-stock-v2-20261018`，0张完成图片，
不作为速度/内存样本。没有为了取得数字放宽旧路线。collector的互斥
`--resident-prefix`保留为这个负控入口；它不是本轮新compact snapshot。

## 先取得的匹配完整请求

相同新库，seed73→74→73，唯一区别为runtime snapshot开关。每个请求包括
encoder/reference处理、全40步、原VAE和PNG；首项分列，后两项取中位数。
不是严格系统冷启动，不拿不同库或观察器的时间充当开关性能分母。

| 路线 | 首请求 | 后两次完整请求 | 热中位数 |
| --- | ---: | ---: | ---: |
| 同库snapshot off | 50.618414s | 45.561864 /45.546881s | 45.554372s |
| compact snapshot | 49.744718s，miss+publish | 40.303905 /40.372588s，hits | 40.338246s |

该窗口热请求耗时下降11.4503%。首项有不同load/cache状态，不声称snapshot让冷
启动变快。on/off三张PNG逐像素RGBA相同，包括首轮miss、换seed及返回seed73；
disabled新库的三张也与冻结v8的已有6144输出一致。这里只证明本例两个种子的
增量PNG结果，不证明所有中间latent、prompt、人物/材质或多参考语义逐位一致。
原W8路线相对普通GPU BF16的近似仍在，没有借增量exact消除其既有误差。

反向关闭窗口另得45.404944/45.469870s，中位45.437407s，仍比candidate约慢
5.10s；相对此反向控制耗时下降约11.2%。三张RGBA也恢复到相同baseline。
这不是独占设备或广泛语义质量资格，仍保留所有竞争负载记录。

miss实际完成1个5155-row prefill+39个1024-row decode，共1440次模型预测；hit
实际完成40个1024-row decode，共1280次预测。每请求1280个FFN均执行，没有
跳采样步或复用旧target。首项另外3次能力检查evaluate不混进模型预测。

额外独立timeline3请求观察到同一个ANE model identity，hot compile/load均0，
无drop/pending；设备selector的独立复核另见audit。证明范围是捕获到的Private
client selectors，不承诺不可见driver内部永远不编译，也不声称物理INT8 MAC、
ANE硬件时长或GPU/ANE物理重叠。`qwen21_step.hybrid`是旧frozen字段，不能据它
否认实际runtime调用，应读新receipt、native phase与真实selector。

## 内存与边界

新增compact K/V逻辑bank为2165833728 bytes，约2.017GiB；至多一个bank。
保留min(8GiB,物理内存/8)上限和export的额外8GiB headroom检查。这不是ANE
graph变大，也不是整个进程/外部服务内存的硬上限。

前两组process-tree peak footprint：off 19616001232 bytes、on21499391256 bytes；
采样覆盖load/请求/退出，100ms、max gap500ms。系统swap-out均0，on窗口系统
swap-in65536 bytes；不能将系统计数全部归因本进程。两个窗口均检测到竞争推理
CPU负载，GPU独占未证明，所以`qualification_passed=false`。

这是重复conditioning的收益，不是fresh prompt、首次编辑、文生图或所有多参考
都变快。跨请求前缀的可复用性来自模型的causal prefix/t=0调制，不等于扩散
target或DB residual可以跨请求缓存。

## 构建、回归与复核入口

- 新库：`build/qwen-base-runtime-prefix-v1-20261018/libturbocider.dylib`。
  SHA256 `31c771bce2b010c25ca191bbb65b2b8dabc62a587dfb9ddde79719bfc00a780e`。
- Runtime ID：`tc-runtime-build-v1-e9b6386a5972163080eb1af5e0dbb7b53a392f787ba850f31b8f56e56aeb134d`。
- 653个native/build输入与封存源码一致；归档
  `outputs/qwen-base-runtime-prefix-native-source-v1-20261018.tar.gz`，SHA256
  `42bd9b5e533386949961498bbf3c41796708b74f146fcce1358e88c19d88eaad`。
  旧v8库/652输入归档保留，不能继续把当前源码称为与v8相同。
- CPU request/backend recipe负控通过；真实MLX lifecycle回归通过，包含24 cycles /
  8 variants，以及新tagged snapshot默认拒绝、错误recipe拒绝、不保留source
  owner和matching fixture数值检查。tiny fixture不是实际ANE性能/质量证明。
  全部32项host回归重新通过，日志为
  `outputs/qwen-base-runtime-prefix-host-tests-v2-20261018.log`；生命周期重跑记录为
  `outputs/qwen-base-runtime-prefix-lifecycle-v2-20261018.log`。
- matched stock：`outputs/qwen-base512-runtime-prefix-{off,on}-stock-v1-20261018`。
- 真实selector/CB：`outputs/qwen-base512-runtime-prefix-on-timeline-v2-20261018`。
- 独立CPU审计驱动：`tools/profiling/qwen_base/runtime_prefix_audit.py`，重读native
  source archive、actual requests、PNG、receipt、phase/chunk counters及memory。
  最终审计`outputs/qwen-base512-runtime-prefix-audit-v3-20261018.json`包含反向
  control、真实selector和失败resident负控；旧v1/v2审计均保留。

复现命令见[profiling工程](../../tools/profiling/qwen_base/README.md#base-runtime-prefix复用实验)。
本轮没有下载/改写权重、停止用户ComfyUI、替换已安装App或修改产品默认。
4个成功窗口共12个完整编辑请求，另1个resident负控未完成图片；所有owned
build、test与模型进程均已终止。磁盘约余8.4–8.5GiB，未执行删除。

下一步仍是不同prompt/参考的失效与实际取消恢复、多参考资格，以及512² Private
QKV/更快FFN组合；不能将本次重复编辑收益外推成全部生成任务或全局瓶颈已解决。
