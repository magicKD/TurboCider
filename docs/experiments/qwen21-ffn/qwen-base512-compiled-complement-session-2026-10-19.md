# Qwen base512：共享compiled GPU complement的完整40步验收

2026-10-19，M4 Max/64GiB、macOS26.6.2。接续
[显式动态W的组件实验](qwen-base512-compiled-complement-2026-10-19.md)。
仅原BF16 Qwen Base512文生图和单full1024参考编辑，无LoRA/student、无减步或
DiT residual cache。优化目标仍active，App/default不变。

## 结论：功能成立，但局部信号没有成为稳健整图收益

组件约0.5%的信号已接入真实attention、Private QKV/FFN与one-hop前看，并完成
正反序完整请求。首轮生成/编辑分别少约0.214/0.143s，反序却基本持平，因此
**不取代原fixed FFN/QKV基线，不叠加到当前最快配置，不默认开启**。

单位秒，各窗口均三个完整请求seed73→74→73，首请求另列，后两次取中位数：

| 同库配对 | 原GPU complement | 共享compiled complement | 观察 |
| --- | ---: | ---: | --- |
| 文生图首组，off→on | 28.909705 | 28.696142 | 首组小幅减少 |
| 文生图反序，on→off | 28.650046 | 28.652624 | 持平，未复现 |
| 重复编辑首组，on→off | 35.802170 | 35.659595 | 首组小幅减少 |
| 重复编辑反序，off→on | 35.753033 | 35.745796 | 约7ms差，未形成稳定收益 |

各任务按上述顺序使用独立进程，期间还交错另一任务及独立timeline，不称紧邻
连续ABBA或强统计资格。八个stock首请求生成约30.24–30.74s、编辑44.20–44.42s，
不作为严格OS冷缓存比较。两个on timeline热28.846329/35.762628s只作诊断。
十窗均检测到外部ComfyUI短时CPU负载，正式speed字段全部null，没有设备独占、
thermal或置信区间资格。没有因反序结果不利而删除窗口或选择最快单张。

## 具体实现与失效边界

新选项`TURBOCIDER_QWEN21_COMPILED_COMPLEMENT=1`，默认关闭；collector为
`--compiled-complement`。仅显式joint base40/resident512/fixed6144、descriptor、
sigmoid-complement、F32 partial join、untimed单chunk，拒绝LoRA、packed source、
GPU-dual/family组合及其他尺寸。此轮实际只测单参考，不授予双参考性能资格。

`compiled_complement.hpp`的body与受测组件逐字一致：原X、完整原gate_up W和
down W显式传入；GPU gate/up仍独立BF16 MatMul、typed SiLU×up、原pitch的
BF16×BF16→F32 down partial。没有拼接/复制W、逐层捕获W或新增ANE model。
一个Session保留一份compiled函数；long prefill仍原回调，目标1024行才使用它。

首个真实hidden与F32 partial完整bits/finite检查，source generation绑定后只做
一次；失败sticky停用，回原GPU callback，不能发布健康prefix。函数/状态在unload
清空；本轮未做真实driver hang/reset资格。新算术namespace包含独立recipe，
关闭时原prefix key不变，不能把旧policy的缓存冒充新policy。

`qwen_compiled_complement`回执记录实际attempt/completed/ordinary/fallback、
一次validation时间和graph-body trace数；无新显式owned weight bank。
旧GPU-dual关闭回执的ordinary表示非dual路线；新compiled路线应读取自己的回执，
不把逻辑feature计数误当物理kernel数量。

## 实际执行、图片与编译

独立审计通过：十窗30张完整RGBA、1200个实际采样步、38400个成功混合FFN block。
每请求1280个完整block；FFN生成首请求1312次预测、编辑首请求1440，热均1280。
QKV生成每请求1248，编辑为1248/1280/1280，原lookahead全部健康平衡。

六个on owner均实际跨32层与三请求保持trace1、validation1；首请求1248个compiled
complement＋32个ordinary prefill，热请求1280个compiled，无fallback/retry。
两条on timeline各两个稳定ANE model（FFN/QKV各一），hot compile/load0。
Trace count不是GPU shader/library编译次数，Private selectors也不覆盖opaque
驱动内部，不从这些计数宣称物理INT8 MAC或硬件并行比例。

四组策略配对、每组三张完整RGBA全部exact；六组同配方重复/timeline重放、
每组三张也全部exact。新关闭控制的六张完整PNG另与V9生成/V10编辑原控制
逐字节相同；历史时间没有作为本轮分母。

生成seed73的miss/hit恢复exact。编辑保留原先首步QKV placement差异，旧/新两侧
miss/hit SSIM均约0.998314，不冒充同一路由逐位恢复；匹配hit状态的新旧策略仍
exact。已查看生成/编辑完整样本，本次没有新增像素差异，但原W8相对纯GPU的
误差、编辑形状/材质/高光和任意prompt语义资格不由此自动解决。

## Profiling：不能把少量FFN包络改善直接外推为整图

反序stock的每请求hot累计host包络均值：

| 边界 | 生成off /on | 编辑off /on |
| --- | --- | --- |
| FFN caller | 14.623757 /14.591902s | 14.638776 /14.583102s |
| pre-FFN | 7.908826 /7.930176s | 14.809481 /14.831885s |
| async ANE wait | 13.793619 /13.776255s | 13.799008 /13.748780s |
| QKV暴露W join wait | 0.011875 /0.011840s | 0.016124 /0.020773s |

FFN caller仅少约32/56ms，同时pre-FFN略增，完整wall没有保留首组的明显差值。
这些边界可能重叠，不能相加、乘层数或当成物理ANE/kernel时间，也不能唯一
归因CPU构图、GPU带宽或某一条同步。QKV暴露W等待已只有约12–25ms/请求，
本轮没有支持继续盲增权重缓存/预取距离的证据。

八stock进程树physical peak约18.87–21.53GB，system swap-out均0；首gen-on
stock有约417.6MB system swap-in与71.0MB压缩活动，其余窗口另列原记录。
系统计数不归因单独ANE或当前进程，更不因为没有显式W副本就称整进程零增长。
编译/测试/CPU数组审计均在模型窗口之外，未停止共享服务。

## 工程、来源和复跑

- Native：`build/qwen-compiled-complement-session-v11-20261019/libturbocider.dylib`，
  SHA256 `06e290fa07ad44282fdde9f1a45c5858d1b691b7a38d6941a627ffd0ea0adeab`。
- Runtime：`tc-runtime-build-v1-ec9eb8aaa542a65d0f4442ee401386bc5871546f53507acfee9708ca48e1e829`。
- 668输入exact源码归档：`outputs/qwen-compiled-complement-session-native-source-v11-20261019.tar.gz`，
  SHA256 `644aa8d3d67eb75ca7ecb1cdbeb9410f9e23d6f328ab2f4fce6359d4d52a8354`。
- 全量审计：`outputs/qwen-compiled-complement-session-audit-v1-20261019.json`，
  覆盖原模型/encoder/VAE全SHA、真实工具字节、完整请求、PNG、memory原始链和selectors。
- `function_request_audit.py --complement-pair LABEL OFF ON`要求唯一开关、全部RGBA
  exact、匹配prefix hit且recipe隔离；即使无速度资格也重算全部cold/hot分母。
- 50项相关CPU测试通过，含原组件body逐字比对、域/默认、source sticky、实际
  counts、trace/revalidation拒绝、完整图片/prefix/分母负控；日志
  `outputs/qwen-compiled-complement-session-host-v12-20261019.log`。
- 新库C API141项：140通过、1跳过；日志
  `outputs/qwen-compiled-complement-session-contract-v11-20261019.log`。
  不是全仓或所有设备/模型数值资格。C++布局已变，不能混用旧helper与新库。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-compiled-complement-session-v11-20261019/libturbocider.dylib \
  --route runtime --channels 6144 --fixed-program \
  --ffn-activation sigmoid-complement --private-qkv --private-qkv-prefetch \
  --case generate --mode stock --runtime-text-prefix-snapshot \
  --text-prefix-qkv-step0 gpu --compiled-complement \
  --output outputs/FRESH-compiled-complement
```

控制只去掉`--compiled-complement`；编辑改`--case edit --runtime-prefix-snapshot`并
去掉text-prefix两项。模型全部退出后才运行独立审计，不把timeline或受污染窗
混入正式速度分母。原件统一为`outputs/qwen-compiled-complement-*v11-20261019`。

所有中间产物已由现有gitignore覆盖，约32GiB可用；本轮没有下载或删除模型/
用户缓存，没有部署App、改默认或commit。当前最强研究配方仍原fixed6144 FFN＋
Private QKV＋one-hop＋相应prefix，不因这次局部编译成功就改变速度选择。
