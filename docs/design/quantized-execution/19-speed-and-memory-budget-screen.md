# 19 · 6/8/10/16 GB 的速度与内存筛选（2026-10-01）

[目录](README.md) · [固定验收](11-acceptance-profiles-and-feasibility.md) · [refiner 银行](18-streamed-refiners-and-bank-boundaries.md)

用户新增优先级：每档预算寻找最快可行路径，不以最低内存路线代替速度目标。
本阶段新增分离的 timing/memory probes 与严格经验筛选器，实测同源 Z Q8 的七个
候选，发现原 native packed 更快但进程 footprint 超出所有目标预算。
这不是自动生产 planner，不授予 whole-request hard cap 或 accelerated 资格。

## 1. 新工具与测量边界

`run_z_image_gguf_quantized.py`：

- `--measurement timing` 不启动内存 sampler、不传进度 callback、不写事件内容，
  不允许 dump/cancel。VM counters 在 timed 调用外读；不会把观察者开销混进测速。
- `--measurement memory` 单独以 10 ms 目标间隔采样 Darwin phys_footprint 和
  kernel lifetime peak，报告实际最大相邻采样 gap。原 diagnostic 模式仍保留。
- `--warmup` 明确标记并保留原始请求，不删除慢 warmup；screen 仅排除标记的 warmup。
- `--prompt-cache miss` 为每请求新建 engine，保持原 prompt 不变，并检查实际未命中。
  因此这是 **warm runtime / fresh engine** 的全请求 cell，不是常驻权重的 steady-state。
  hit 是独立 cell；screen 拒绝把未命中的 timing 样本冒充 hit。
- 显式 encoder GGUF/config/tokenizer、source mode、p 和 managed ceiling；tokenizer
  必须与 Z root 实际使用文件相同，不能用不存在的环境 override 改它。已有冲突环境拒绝。
- native packed 的 source 内容 SHA 在请求计时外独立核对，时间单列；不伪装 cold total。
  后续诊断事件新增 monotonic timestamp，便于定位后端 retained/峰值阶段。

`screen_gguf_memory_budgets.py`：必须同时有成功 timing 与 memory reports，并核对
实际库/source/encoder/layout/profile、输入与 PNG hash。slot/fill 数不符、诊断混入、
错 cache cell、缺样本/observer、NaN/空 hash 或换 seed 等均拒绝。swapout 增长或 gap
超过50ms 的候选不能成为 observed fit。原始 raw reports 的 SHA 与 verifier SHA 均绑定。

按默认10%余量，`available=floor(limit*90/100)`；只在观测 peak 可容纳的候选中
按 timing median 排序。GB=10^9 bytes、GiB=2^30 bytes 显式区分，绝不把两者混用。
任何结果始终 `whole_request_memory=unknown`、`production_qualified=false`、
`bounded_certified=false`；这不是现有 memory_constrained guard 的替代品。

## 2. 真实完整请求

Apple M4 Max / 64 GiB，macOS26.6.2 (25G83)，MLX0.32.0；实际 library SHA 在回执。
Z Q8 DiT 与 Qwen3-4B Q8 encoder，512²、portrait prompt、34 valid tokens、seed42、
4steps、dynamic_text=true、compat-affine。encoder streamed p2 / 1 GiB **managed
weights ceiling**，先 drain/release 再执行 DiT；不是额外1 GiB免费配额或整请求限制。

两组 timing 顺序：streamed `p0,p1,p2,p2,p1,p0`；resident
`native,p0,p1,p2,p2,p1,p0,native`。每个 visit 一次 warmup + 两次记录，因此每候选
4个计时样本和2个保留 warmup；另7次独立 memory 请求，共49次真实生成，全部成功，
同一 PNG 内容 SHA。timing/memory 不混池，未清 OS cache，也不称 cold SSD 测试。

| DiT 候选 | 全请求 median 秒 | 观测 footprint 最大值 bytes | 约 GiB |
| --- | ---: | ---: | ---: |
| streamed p0 | 14.988 | 3,565,536,480 | 3.32 |
| streamed p1 | 10.786 | 3,756,639,480 | 3.50 |
| streamed p2 | 10.817 | 3,756,639,480 | 3.50 |
| resident p0 | 16.712 | 9,620,035,704 | 8.96 |
| resident p1 | 12.411 | 9,620,035,704 | 8.96 |
| resident p2 | 12.204 | 9,620,035,704 | 8.96 |
| original native packed | 9.142 | 21,057,870,640 | 19.61 |

表内数值仅指本次 sample/kernel lifetime peak 的最大值，不是连续时间或
所有输入/driver 的已证明上界。resident 多个 p 共用 memory 进程，lifetime peak
可能由较早 arm 决定；保守采用，不减去旧峰值。native packed 用全新进程独立采样，
所以其19.61GiB并非继承 resident 的历史峰值。所有 memory 请求 gap≤16.4ms，
系统范围 swapout 增量0；不能将64GiB上的经验结果称为6/8GiB实机认证。

## 3. 各预算的经验结果

GB 与 GiB 两套分析都得到：6/8/10/16 下 **streamed+p1 是本次七个已测候选中最快
的 observed fit**，约10.79s，peak约3.50GiB。p1/p2仅差约0.03s，样本不足以证明
这个细微差异显著；暂不修改默认或强制p1。较大预算仍应继续尝试更快的新候选。

native packed 虽快，却超出16GB/16GiB以及余量，不作为可行推荐；resident 虽在
10GiB/16GiB可 observed fit，却慢于streamed，不因为预算大就自动选它。
10GB与10GiB的可容纳集合不同：9.62GB peak超过10GB扣10%后的9GB，而小于
10GiB扣10%后的9.66GB。二者最终最快候选相同，但不得丢掉这个单位差异。

native 的MLX active仅约7.90GB、MLX peak约11.28GB，而进程当前最大/末尾
footprint约21.06/20.88GB。因此问题不只是一个已经释放的加载临时峰值；**框架/
源 backing/allocator retained 的额外驻留仍须归因**，不能只看managed/MLX数字。
`Weights::load_gguf_file()` 当前调用全文件 `mx::load_gguf`；下一阶段检查其ownership
与 retained 来源，评估有界直填 MLX affine packed bank 的导入路径，避免原压缩源与
全模型repack副本/转换临时同时保留。不先宣称已经证明所有额外 bytes 的单一原因。

## 4. 回执和复现

[GiB 回执](validation/gguf-budget-gib-screen-20261001.json) ·
[GB 回执](validation/gguf-budget-gb-screen-20261001.json)。raw reports 在
`outputs/quantized-execution-budgets/`，含每个请求、全部 warmup、sample、错误/VM、
实际library身份、PNG。首个stream timing先于runner hash字段新增，未倒填旧runner hash。

```sh
.venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib --model models/z-image-runtime-gguf-q8 \
  --output outputs/new-budget-timing --source-residency packed_streamed --prefetch 0 1 2 2 1 0 \
  --runs 2 --warmup 1 --measurement timing --prompt-cache miss --precision z-mlx-compat-affine-v1 \
  --encoder-gguf models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf \
  --encoder-config models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json \
  --encoder-tokenizer models/Tongyi-MAI-Z-Image-Turbo/tokenizer/tokenizer.json
# memory另起进程：相同组件/profile/input，--measurement memory --prefetch 0 1 2 --runs 1 --warmup 0
# resident timing顺序 -1 0 1 2 2 1 0 -1；memory仅0 1 2；native -1 memory另起新进程
.venv/bin/python tools/validation/screen_gguf_memory_budgets.py \
  --reports outputs/quantized-execution-budgets/q8-stream-timing/report.json \
  outputs/quantized-execution-budgets/q8-resident-timing/report.json \
  outputs/quantized-execution-budgets/q8-stream-memory/report.json \
  outputs/quantized-execution-budgets/q8-resident-memory/report.json \
  outputs/quantized-execution-budgets/q8-native-packed-memory/report.json \
  --budgets 6 8 10 16 --budget-unit GB --output outputs/new-budget-screen.json
.venv/bin/python -m pytest -q tests/native/test_gguf_memory_budget_screen.py \
  tests/native/test_gguf_probe_measurement_options.py
```

13个host tests通过，含数值/identity/采样/预算单位/CLI冲突负例；真实测速关闭observer，
独立内存observer通过。最终runner另跑一次真实单GGUF文件路径smoke：显式encoder、
194个带timestamp进度事件、963个内存samples、gap15.32ms、PNG与矩阵exact；这次
额外验证不混入上表49请求矩阵或计时统计，raw在`final-runner-file-path-smoke/`。
本阶段不修改native kernel/数学/发行catalog，不下载或删除模型。

## 5. 不能由本阶段替代的验收

只有一个512² portrait/seed42 cell，未跑第二session、正式ABBA/24样本、cold campaign、
1024²/long/cache-hit/Q4矩阵；不注册 accelerated，也不推论所有模型同一winner。
PNG exact不替代首次profile的source-quant/原BF16/人工媒体审核；旧负结果仍保留。
whole-request admission、managed required-site closure与driver envelope仍未闭合。
tiles、ConvRot模型/ANE integration、W8A8与M5条件路线继续原目标，线程保持active。
