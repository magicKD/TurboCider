# Runtime ANE v1：Qwen 1024² base 专用图与 Z 组件筛选

同一原生库 `875b0d3dcf5c766feeced21d71f1a366acf0a1238bc3f0ed332098aa3b27b8cd`。
本轮没有更改 native 计算、默认选路、checkpoint 或已有冻结图；只为无 adapter
的 Qwen base 请求重用 exporter 已支持的 v1 接口。v1 返回 FFN 输出，不接收
LoRA gate/up 修正，也不返回供 down-LoRA 使用的 hidden；**带 adapter 必须继续
使用 v2**，不可将 base 专用 v1 图用于 LoRA。v2 无 adapter 时虽然传零修正，
仍需要完整 v2 接口。

## 组件与两个提示词的 v1/v2 交错筛选

真实 Qwen BF16 block0 权重、合成 4096-row 激活、c1792/K1024/N512，
一段 1792 rows 给 ANE，剩余 2304 rows 给 GPU。两图只改变 v1/v2 接口，
各独立进程 10 次测量，顺序 v2→v1→v1→v2；四组数值 relative L2
均约 0.002160，零 overflow retry。池化组件中位：v1/v2 并行 FFN
47.809/50.717 ms，含 staging 50.480/53.630 ms。真实权重不等于真实
捕获激活，组件速度不能直接换算整请求。完整原始记录在
`outputs/runtime-ane/qwen-c1792-v1-v2-component-screen/summary.json`，
状态 complete，绑定输入前后 SHA 不变。

相同图与库的 resident 1024²、40 步 base 请求，Q/K norm-RoPE 融合在所有
trial 开启，`chunks=auto`、profile 关闭。每条路线独立一冷两热；每个
提示词 v2→v1→v1→v2，每配置四个热样本：

| 提示词 | v1 四热中位 | v2 四热中位 | v2/v1 |
| --- | ---: | ---: | ---: |
| 雪中狐狸 / seed42 | 145.397 s | 147.708 s | 1.016× |
| 成年肖像 / seed43 | 145.377 s | 147.870 s | 1.017× |

原始记录分别在 `outputs/runtime-ane/qwen-c1792-v1-v2-model-abba-875b/`
与 `outputs/runtime-ane/qwen-c1792-v1-v2-portrait-abba-875b/`，均 complete。
每 trial 3,648 次 prediction/hybrid block、192 次正常 GPU probe，
无失败回退或重试；采样完整且无 swap-in/out。相同提示词的 v1/v2 热 PNG
逐字节一致。两组驱动各自验证所绑定模型/图/工具输入的前后 SHA 一致；
不能把两个提示词当作广泛画质资格，也不能跨实验拼接 GPU 分母。

## 同库 GPU、v1、冻结图正反序

同一只雪中狐狸 / seed42、1024²、40 步、BF16 base、resident、Q/K 融合
统一开启。v1 c1792/K1024/N512，`chunks=auto`；冻结图是原有匹配
4096-row W8A8 manifest。各路线独立一冷一热、独立 100 ms 进程树采样；
正序 GPU→v1→冻结图，反序 冻结图→v1→GPU，两个热样本取中位。
计时是完整请求墙钟（含 VAE/PNG），不含冷请求。

| 路线 | 正序热请求 | 反序热请求 | 两热中位 | GPU/路线 |
| --- | ---: | ---: | ---: | ---: |
| GPU | 183.072 s | 183.099 s | 183.086 s | 1.000× |
| runtime v1 | 144.836 s | 144.292 s | 144.564 s | 1.266× |
| 冻结 base 图 | 161.320 s | 159.249 s | 160.285 s | 1.142× |

此工作负载 v1 比同组冻结图少约 9.8% 请求耗时。两个 v1 trial 各有
2,432 次 prediction/hybrid block、128 次正常 GPU probe、零失败回退和
overflow retry；两个冻结图 trial 各 2,528 次调用。六 trial 的内存采样
verifier 都 complete，无 swap-in/out；两个冻结图窗口有约 46/74 MB 进程树
compression，故不能据此断言无内存压力或物理 ANE 驻留。
`compute_units=cpuAndNeuralEngine` 是请求配置，遥测中的实际 residency
仍为 unknown。不能把不同路线的进程树 footprint 与 MLX allocator 峰值
相混，也不能将两热样本推广到所有硬件和提示词。

证据：`outputs/runtime-ane/qwen-v1-base-40step-forward-875b/summary.json`
和 `outputs/runtime-ane/qwen-v1-base-40step-reverse-retry-875b/summary.json`
均 complete，库 SHA 相同；六张末次热图在同一路线正反向 SHA 一致。
肉眼检查狐狸主体、姿态、构图正常；GPU 与 v1 很接近，冻结图局部毛发与
树枝细节不同。三路 PNG 不完全相同，不承诺逐像素或普遍语义等价。

## Z-Image：组件信号不足，未升为整请求候选

同几何 c352/K1024/N512 的 Z BF16 v1/v2 图、noise_refiner.0 真实
checkpoint 权重和合成 1024-row 激活，352 rows 给 ANE、672 给 GPU。
顺序 v2→v1→v1→v2，每 trial 十次，汇总中位如下：

| 接口 | GPU FFN | 并行 FFN | 并行含 staging | Core ML prediction |
| --- | ---: | ---: | ---: | ---: |
| v1 base-only | 16.499 ms | 11.907 ms | 13.991 ms | 10.906 ms |
| v2 共图 | 16.503 ms | 12.092 ms | 14.160 ms | 11.111 ms |

数值 relative L2 均为 0.001369，四个 trial 均一次 prediction/样本、
无 overflow retry；内存 verifier complete、无 swap-in/out，绑定输入前后
SHA 不变。含 staging 的组件收益约 1.2%，不值得直接宣称完整 Z 请求
提速；本轮不开展 Z v1/v2 整请求对照，继续保留已验证的 v2 LoRA
与匹配冻结 base 路线。原始记录在
`outputs/runtime-ane/z-c352-v1-v2-component-screen-retry/summary.json`。
另在更长的 1024² 序列上，v1 **相对 GPU** 的双提示词整请求对照取得
超过 5% 收益，见 [Z 1024² 独立记录](runtime-ane-z-1024-v1-2026-09-30.md)；
它不是 v1/v2 比较，也没有匹配冻结图分母。
初次驱动错误地要求 GPU 后 join 耗时严格大于零；一次合法的零等待使
该次 summary 保持 incomplete。修正为有限且非负后在新目录完整重跑，
失败原始记录保留在 `outputs/runtime-ane/z-c352-v1-v2-component-screen/`；
未放宽数值、调用或内存门槛。

## 复现与边界

图由 `tools/coreml/export_runtime_ane.py` 的既有 v1 协议导出，不传
`--lora-inputs`：

```sh
.venv/bin/python3 tools/coreml/export_runtime_ane.py \
  --kind swiglu --rows 1792 --hidden 4096 --width 12288 \
  --tile-k 1024 --tile-n 512 --output outputs/runtime-ane/new-qwen-v1-base
```

整请求使用 [统一 screen](runtime-ane-validation.md)，为三路提供各自匹配的
manifest，参数 `--size 1024 --steps 40 --routes gpu,runtime,frozen`
与反序 `frozen,runtime,gpu`、`--chunks auto --qwen-qk-norm-rope`
`--warm-repeats 1 --sample-memory --timeout 1200`。每次 output 是新目录；
不要直接使用此报告中依赖本机的 manifest 路径或把历史 v2 分母拼入新库。

v1 与 v2 在原生 runtime 中均是显式 optional；LoRA/编辑仍保留共图 v2，
512² 的最快已测 Qwen/Z base 路线不因这组 1024² 数据或 Z 组件而变。没有修改
默认 GPU/冻结图策略，也没有新增真实 LoRA、1–3 图编辑、多提示词画质、
低内存或硬件级重叠资格。后续若更改图、chunk、库或融合设置，须重新
做同组完整请求对照。

## 收尾校验

正反序 model screen 与 Z 组件重跑均 status=complete。原生库 SHA 保持
875b；`make test-acceleration-contract` 沙箱外完整通过，布局 8 项与
独立性 7 项通过，`git diff --check` 无警告。未执行全量 `make test`，
未重编译 native、stage/commit 或移除已有模型/图/图片；已有暂存删除不动。
