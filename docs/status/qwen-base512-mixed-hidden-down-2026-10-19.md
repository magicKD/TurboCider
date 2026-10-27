# Qwen base512：FP16 hidden × 原BF16 down权重

2026-10-19，M4 Max/64GiB，macOS26.6.2/25G83。只关注Qwen Image 2.1原BF16
base、512²文生图/图生图；不涉及LoRA/student/Z Image。

结论：本轮没有稳定加速，不接入主runtime、不改变App/default。完整40步推荐仍为
fixed FFN C6144＋Private QKV512＋one-hop QKV lookahead，GPU-dual OFF；生成仍须
text-prefix snapshot和首步GPU QKV。总体优化目标ACTIVE，不把组件实验当成图片收益。

## 假设与实际边界

不同于此前gate/up每调用转换权重的负结果，本候选只把GPU互补FFN hidden存为
FP16，down继续读取原BF16权重，保留F32 partial。没有W cast、packed副本或额外
权重bank。安装的SDK26.5 MPP头有half/bfloat→float混合分支；本轮实际执行了
该路径，但不从SDK标签推断物理MAC精度或性能。

实际shape为M1024/K6144/N4096，原W物理pitch12288。捕获来源是既有真实base40
生成/单参考编辑首decode layer0的GPU互补hidden；这里只重放down，不重新生成
40步图片，也不包含完整gate/up、hybrid block、encoder/VAE/PNG。A/B为同来源输入
及其原有扰动，不是两个新seed或prompt。历史时间不充当本轮分母。

同窗四arm：original-bf16；FP16后转回BF16的rounded-bf16（隔离格式误差）；
ready FP16的mixed-ready；包含每调用hidden cast的mixed-with-cast。原/新同样
完成F32输出；每arm两次warmup、轮转顺序、A/B交替；计时区间不做CPU数组扫描。

## 实测结果

单位ms，中位数；timeline仅诊断，不与stock混成速度样本：

| 窗口 | 原BF16 | rounded BF16 | mixed ready | mixed含cast |
| --- | ---: | ---: | ---: | ---: |
| 生成16 | 3.700229 | 3.681834 | 3.670729 | 3.731313 |
| 生成32 | 3.704563 | 3.702208 | 3.702875 | 3.739334 |
| 编辑32 | 3.695479 | 3.692042 | 3.689958 | 3.738855 |
| 生成timeline16 | 3.707396 | 3.705522 | 3.699000 | 3.744250 |

初次ready约0.8%的信号在加倍样本后只剩约0.046%；编辑约0.15%。含cast约慢
0.84–1.17%。不据此开发更大范围的双格式GU融合，不宣称FP16自动更快。

## 完整数值与独立复核

五个正式窗共448 measured down（384真实＋64合成）、85份完整数组；真实四窗
另有1024个完整K6144的独立F64坐标见证。auditor核对原96MiB完整down W、
冻结输入与原F32 partial完整bytes，以及同case重复/timeline的全部数组。

实际hidden范围max约2.03–3.33，FP16 roundtrip relative-L2约7.57e-8–8.17e-8，
约18k/6.29M微小值改变；partial relative-L2约7.55e-8，最大绝对差约1.91e-6。
受测mixed与rounded-BF16对照全部bits exact，ready/含cast也全部bits exact；
这不是对原BF16无损，也不是32层/全部输入范围或完整图片质量资格。

无checkpoint合成fixture为M33/K128/N257，W pitch320、row begin5、column
begin17，覆盖非零view和tail。独立整数dot全输出oracle覆盖normal及domain：
可表示subnormal与真zero exact；低于half范围的underflow relative-L2=1，明确
保留为格式损失，不称zero exact。wide BF16→FP16非有限在down前拒绝。另有六项
非法dtype/range负控、W及输入A/B/A恢复，原件与所有数值门槛保留。

最初MSL联合auto声明的half/bfloat推导失败，后拆开声明；这是源代码问题，不是
设备不支持混合类型。另一次未eval的MLX reduction修正为显式eval；audit v1的
合成输入oracle漏batch轴，修正shape，没有放宽误差门槛。失败日志均保留。

## Profiling、内存与交付

GPU-only observer核对每个sample scope的实际CB完成/时钟，无ANE调用；因此本轮
不能证明ANE精度、物理INT8或GPU/ANE overlap。原始process-tree memory JSONL
hash-chain独立重算通过，真实窗口峰值约0.709GiB、合成约0.202GiB，swap-in/out0。
存在竞争推理CPU负载，统一qualification_passed=false，不具备独占/thermal资格。
设备窗串行，所有自有CPU审计/测试在设备窗结束后进行，没有停止ComfyUI。

- 工具：`tools/profiling/qwen_base/mixed_hidden{.hpp,_probe.mm,_collect.py,_audit.py}`。
- helper：`build/qwen-mixed-hidden-probe-v4-20261019/libqwen_mixed_hidden_probe.dylib`。
- 最终审计：`outputs/qwen-mixed-hidden-full-audit-v3-20261019/audit.json`，complete，5窗。
- 168项profiler host测试通过：`outputs/qwen-mixed-hidden-host-tests-v1-20261019.log`。
- 主native661源码/ABI未改，库SHA256仍为
  `7da7a7125bd7cb401344bf885c2425ec7eacc14d81b0563fd45317a94b4b260a`。
- 工具/测试/本报告归档：`outputs/qwen-mixed-hidden-tools-v1-20261019.tar.gz`；
  摘要：`outputs/qwen-mixed-hidden-artifacts-v1-20261019.json`。

本轮先对三个既有QKV consumer相同完整数组做cmp＋硬链接去重，所有原路径/bytes/
hash保留，消除788532424重复bytes；审计后对混合hidden三个生成窗同样无损去重，
所有编辑数组独立保留。这不是runtime内存/推理速度收益。保持2GiB磁盘reserve，
不删除模型、旧库、失败原件或图片。

```sh
TURBOCIDER_NATIVE_OUT=build/FRESH-MIXED-HIDDEN \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen-base-gpu-dual-v1-20261019 \
  bash tools/native/build_qwen_mixed_hidden_probe.sh
.venv/bin/python -B tools/profiling/qwen_base/mixed_hidden_collect.py \
  --library build/FRESH-MIXED-HIDDEN/libqwen_mixed_hidden_probe.dylib \
  --native-library build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib \
  --fixture generate --repeats 32 --output outputs/FRESH-MIXED-HIDDEN
```

编辑改`--fixture edit`；无checkpoint合成用`--fixture synthetic`。所有模型窗退出后
用auditor的`--cohort ROOT`（可重复）、`--probe-library`、`--native-library`、
`--native-source-archive`及fresh `--output`独立复核。

## 下一个可证伪方向

检查发现真实base无LoRA的GPU channel callback仍分开project_slice(gate/up)、
`silu(g)*u`、F32 down；silu本身已经compile，不能称全是eager。此前prototype的
compiled runtime_ffn::channels并不等于真实pipeline的回调。

下一步可比较原callback与一个共享、显式runtime X/gate_up W/down W输入的compiled
函数，先保留原BF16阶段/F32 partial与原W，不捕获逐层权重或构造新W bank。先做
真实冻结输入/W0/W1的完整GPU互补对照；只有稳定净收益再测完整block/joint/full40。
目前尚无该方向的性能证据，也不改变固定ANE图与热路径零compile/load契约。
