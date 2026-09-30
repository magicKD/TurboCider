# Runtime-weight FFN：两模型集成与性能筛选

2026-09-28，接续[单层原型](runtime-ane-component-2026-09-28.md)。
现已接入 Qwen-Image-2.1 和 Z-Image Turbo 的 **base BF16** 生成路径，
显式 `hybrid_mlp_mode=runtime`；不改变 GPU / 冻结图默认选择。
这是实验性路线，v1 仅 base、无 streaming。后续已补齐 Q4/Q8 affine
staging 组件及微图测试，并接入 Z 原生 GGUF 路由；真实 Q8_0 的整模型
结果见文末。Q4_0/Q4_1 仍缺真实 checkpoint 验证，不能从 Q8 成绩外推。
已有完整 runtime LoRA 继续使用冻结 base 图 + GPU 动态修正，不受影响。
后续可选 graph-v2 已接入 BF16 Z/Qwen 的完整 runtime LoRA，仍不合并权重；
性能和共图切换见[LoRA 激活接口](runtime-ane-lora-2026-09-28.md)。
Qwen 新路线当前变慢，不替换原 GPU 或冻结图；GGUF runtime 仍限 base。

## 已实现

- `native/backends/ane_ffn.{hpp,cpp}` 作为模型侧接口，拥有借用 tensor
  的生命周期，在 attention 提交前启动 staging，在 FFN 处 join。
  GPU 保留 head token rows，Core ML 处理整数 chunk 的 tail rows。
- Qwen 对 prefill 和 decode 使用同一按形状共用图；GPU 仍用 fused
  gate/up matmul。Z 保留 GPU projection / fused SwiGLU 优化，context
  refiners 留在 GPU，32 个 image/unified FFN 层共用一个 Core ML 图。
- 任意 Core ML chunk 失败时重算**整个 tail**，不拼接部分成功结果；
  staging 失败时全 FFN 走 GPU。后续层降级 GPU，错误原因写入结果。
  取消/析构先 drain 再释放借用存储。预算不够时不分配图槽，走 GPU。
  模型 kind/H/F 在内存准入前检查，不能用低内存回退掩盖选错图；
  Core ML 模型加载能力错误与 receipt/几何错误分开，前者可报告后回退。
- `ane_scheduler.hpp` 与执行器解耦，按 layer / row-count 独立计时；
  warmup、GPU probe、EMA/hysteresis、整数 chunk 平衡及周期 reprobe。
  on/off 比较包含 pre-FFN 时段，避免忽视 staging 与 attention 的带宽竞争。
  这是同一 split-boundary 路线的 GPU 对照，不代替整模型默认 GPU 比较。
- 结果区分 `runtime_weight_token_row_ffn` 与旧通道补集路线；输出
  staging/wait/join、GPU/混合块数、rows、失败/溢出恢复及槽大小。
  数据为 **session 累计**，需差分得到每请求量。
- staging 和输出 FP16→BF16 恢复都有 SIMD 路径及 exhaustive 位型测试。
  不删 SiLU，不融合 adapter 到 checkpoint 或编译图。

普通 native 构建现在链接显式路由所需的 C++/Core ML 实现；Python 导出器、
探针和实验脚本仍不打包为生成依赖，默认 GPU 请求不创建 runtime 图/线程。
计划中的 Core ML 额外内存预留由冻结 Qwen 的 10 GiB 改为 runtime 的
2 GiB；Z runtime 同样预留 2 GiB。实际 constructor 再按剩余物理内存
与槽预算检查。它仍是保守估计，不是统一进程内存硬上限。

## 第一版真实整请求对照（输出 SIMD 优化前）

M4 Max 64 GB，macOS 26.6.2，512×512；同一二进制、fox prompt、seed 42，
resident，第一请求为冷/预热并排除。脚本每路运行前检查其他活跃推理进程，
未保证独占设备；先 GPU、再 runtime、再 frozen，尚非 ABBA 资格验收。
runtime 固定单个 256-row chunk、内部 tile K/N=1024、FP16 图；
各图先通过自身实际几何的 sparse weight-switch 自检。

| 模型/工作负载 | GPU 热请求 | runtime 热请求 | runtime / GPU 加速比 | 现有冻结图热请求 |
| --- | ---: | ---: | ---: | ---: |
| Z-Image，8 步（两次取中位） | 7.009 s | 6.765 s | **1.036×** | 5.360 s（W8A8 1024-row / 5120-channel） |
| Qwen，40 步（一次热请求） | 42.780 s | 38.945 s | **1.098×** | 30.252 s（W8A8 1024-row / 6144-channel） |

本表只是 initial screen，**没有证明 runtime 比冻结图快**。Z 尚未达到
5% 整图门槛；Qwen 需要更多交错重复。两者没有 Core ML 运行失败。
Z 全局 headroom 最终为 64（最初三次溢出重试），Qwen 为 1。
两组 GPU/runtime 图已肉眼对照：狐狸姿态、主体、树林和雪地接近，无黑图、
明显颜色偏移或纹理崩坏；细毛/雪点有变化。只覆盖一提示词、一种子，
不等于跨题材验收。早期 Qwen 五步图仍很粗糙，不作为 base 画质资格。

脚本证据在本地 `outputs/runtime-ane/z-model-screen-a/` 与
`outputs/runtime-ane/qwen-model-screen-a/`：请求、PNG、进程快照、完整 JSONL、
stderr 与包含 native library SHA256 的 `summary.json` 均保留，未纳入 Git。
结果报告当时仍用了旧的 GPU graph/weight qualification 标签，后续已修正；
不能用旧标签声称新后端经过冻结图的资格验收。

MLX 峰值约 Z 16.03 GB，Qwen GPU/runtime 20.61 GB、冻结图 25.45 GB。
这些不是进程 RSS：runtime 另有 Z 239,861,760 / Qwen 306,184,192 字节
显式 Core ML 槽及框架开销；尚未做整请求 wired/compression/swap 验收。
累计 staging 等待约为 staging 时长的万分之一量级，说明主机看不到明显
暴露等待；不能据此证明 GPU attention 没受共享带宽竞争，或物理 ANE 驻留。

## SIMD 输出恢复与 chunk 复测

2026-09-28，在完成上述整理后继续优化；同一 fox/seed 42、resident，
profiling 关闭。只检测到空闲 ComfyUI（进程 CPU 为 0），未终止其他进程；
进程快照保留，但这不是设备独占证明。

| 配置 | runtime 热请求 | 同轮 GPU | 结论 |
| --- | ---: | ---: | --- |
| Z c256，SIMD，runtime/GPU/GPU/runtime，各两次热请求 | 6.736–6.762 s | 7.007–7.015 s | 约 1.039×；顺序反转仍有小幅收益 |
| Z c320，SIMD，各两次热请求中位 | 6.678 s | 7.011 s | 约 1.050×，仅一次顺序筛选，刚到 5% 附近 |
| Z c320，原自动 controller，两次热请求中位 | 6.776 s | — | 尚慢于固定一个 chunk |
| Qwen c288，SIMD，40 步，一次热请求 | 38.914 s | 42.816 s | 约 1.100×，仍需更多交错重复 |

证据分别在 `outputs/runtime-ane/z-model-simd-c256-screen/`、
`z-model-simd-c320-screen/`、`z-model-simd-c320-auto/`、
`qwen-model-simd-c288-screen/`（后三者也在 `outputs/runtime-ane/` 下）。
各目录 `summary.json` 记录实测 library SHA256；这些结果均早于下述
controller 修改，不混称为修改后成绩。

Z c256 的 771 次预测累计输出处理从旧版约 1.339 s 降至 0.213 s；
但整图收益很小，因为它大多在 GPU 分支内被覆盖。此次 c256 PNG 与
优化前 c256 的 PNG SHA256 完全相同，证实该样本没有新增数值变化。
c320 每热请求的 GPU FFN 约 3.12 s（c256 约 3.34 s），同时 join 等待
升至约 0.15–0.17 s；说明增加 ANE 行数降低了 GPU 计算，却开始暴露
更多 Core ML 时间，不能继续按行数线性外推。

两模型 c320/c288 图均与同轮 GPU 图肉眼对照：主体、姿态、树枝和雪地
接近，细纹理有变化，无黑图/色偏/明显崩坏。仍只有一个提示词/种子。
复测的系统 swap 使用量为已有的 958.38 MiB，前后没有增长，swap-in/out
累计计数也没有变化；这只说明所记录窗口未观察到新换页，不是完整
进程峰值、wired 或所有后台活动的内存资格验收。

原 c320 自动策略每热请求有 53–56 / 256 个块走 GPU（包含探测及
不利层关闭），中位比固定 chunk 慢约 0.10 s。因此继续调整 controller：
首次 hybrid 编译同时从 on/off 与 row-rate 估计中排除；发生 headroom
重试的调用不训练调度器；周期探测从同一 layer/shape 每 8 次改为每
32 次。初始两次 hybrid、两次 GPU probe 以及不利时关闭仍保留。
修改后整模型结果须另行记录，不能据此预先宣称加速。

controller 修改后 c320 的 runtime/GPU/GPU/runtime 对照（每路两次热请求），
四次热态中位为 runtime **6.723 s**、GPU **7.012 s**，约 **1.043×**。
每热请求全 GPU 块降为 22–32 / 256，且无 Core ML 失败；比旧自动策略
有小幅改善，仍未稳定达到 5% 门槛。
证据：`outputs/runtime-ane/z-model-controller32-c320/`。

继续将 Z chunk 设为 **288**（图几何 H3840/F10240、tile1024 不变）：
固定一个 chunk 两次热请求中位 **6.488 s**；自动策略同样做
runtime/GPU/GPU/runtime，各两次热请求，汇总中位 runtime **6.574 s**、
GPU **7.011 s**，约 **1.066×**。四次 runtime 为 6.489–6.634 s。
它比 c320 更好，但并非 ANE 行数越多越好；GPU head 的实际 GEMM
形状及共享带宽也影响结果，目前不能将差异完全归因于某一个因素。
证据：`z-model-controller32-c288-fixed/` 与 `z-model-controller32-c288-auto/`
（均在 `outputs/runtime-ane/`）。这些自动结果仍早于下面的长序列候选修正。

## 1024²、多 chunk 与候选切分修正

Z 1024²、8 步，换用 lighthouse 提示词和 seed 7；复用同一 c288 图。
首轮冷+热各一次，热请求 runtime **31.665 s**、GPU **31.480 s**，
**没有加速**。热请求有 112 个混合块、144 个 GPU 块，Core ML
448 次预测（每个混合块 4 个 chunk），没有运行失败；相同图的循环
执行已经真实覆盖，而不是仅做小图或静态 shape 检查。
图像肉眼对照主体/光照接近，灯塔局部、屋顶/烟囱和海浪纹理有变化。
两边 MLX 峰值约 25.63 GB，Core ML 槽额外约 240 MB；系统未观察到新增
swap。运行结束附近出现新的 ComfyUI 活动，因此此轮只作诊断，不作为
严谨负收益资格结论。证据：`outputs/runtime-ane/z-model-1024-controller32-c288/`。

该测试促成一项调度修正：on/off 的旧单 chunk 慢样本不能否定刚根据
row-rate 选出的新多 chunk 候选。切分改变后清除旧 hybrid 均值，先让
新候选运行并测量；每个新 GPU head 形状的第一次编译也排除。
平衡器改为先找全局最小预测成本，再应用一次 5% hysteresis，而不是
对枚举过程的每一个候选都施加 5% 门槛。新增测试覆盖旧切分亏损但
新候选值得实际测量的情况。最终模型复测另记，不能把这次源码修正
本身算作速度提升。

修正后的反向顺序 GPU→runtime 复测（同一灯塔输入，各冷+热一次）：
GPU **31.444 s**、runtime **30.012 s**，约 **1.048×**。热请求现在有
233 个混合块、23 个 GPU 块，**932 次成功预测 = 233 × 4 chunks**，
没有失败或重试；这说明新候选确实获得了测量/执行机会。相比初轮
31.665 s 有改善，但仍未稳定达到 5%，不能外推为全部 1024² 请求更快。
staging 约 1.230 s、暴露等待约 0.000098 s、join 约 0.066 s，均是
热请求 session 计数差分、host 侧时长，不是硬件 timeline/ANE 驻留证据。
GPU/runtime 灯塔、海岸和夕阳观感接近，附属房屋及海浪有可见差异，
未见崩坏；这是生成任务的宽松视觉判断，不是精确保留参考图的编辑验收。
证据：`outputs/runtime-ane/z-model-1024-candidate-c288/`。

## 本轮最终候选：两模型自动策略

最终三组对照使用同一个 native library：
`b7be4b9e19f5906362ed25e3d6358d19d4002e52762bd8dbae2c8ac2b152b550`。
均为显式 runtime、c288/tile1024、auto、resident；冷请求排除。

| 工作负载 | GPU 热请求 | runtime 热请求 | 中位加速比 |
| --- | ---: | ---: | ---: |
| Z 512²，8 步，fox/42，每路两次 | 7.009 s | 6.608 s | **1.061×** |
| Qwen 512²，40 步，fox/42，每路两次 | 42.771 s | 39.183 s | **1.092×** |
| Z 1024²，8 步，lighthouse/7，每路一次 | 31.444 s | 30.012 s | **1.048×** |

Z 512² 最终复测证据：`outputs/runtime-ane/z-model-candidate-c288-final512/`；
两次 runtime 为 6.646 / 6.570 s，无运行失败。前面的 c288 顺序反转实验
支持其收益，但不把不同版本的样本混算为本表中位。
本表支持两模型 512² 在这台 M4 Max 上超过 5% 的候选收益，**不等于
完整 Phase 1 交付**：runtime LoRA、Q4/Q8、更多输入、内存资格仍未覆盖。
也不证明冷启动更快；默认/已有更快冻结图不变。

同一最终候选二进制、c288、auto、512²/40 步，runtime→GPU 各一次冷请求
和两次热请求：runtime **39.069 / 39.296 s**，GPU **42.753 / 42.790 s**；
热态中位 **39.183 / 42.771 s**，约 **1.092×**。
runtime 三请求共 3553 次 Core ML 调用，没有失败、溢出重试或错误回退。
两次热请求的 GPU-only 块为 59 / 100（每请求 1280 块），说明自动策略
保留探测/关闭逻辑，并非简单强制固定比例。图像肉眼与同轮 GPU 接近。
系统 swap-in/out 与使用量前后没有增长；仍不是完整内存资格验收。
证据：`outputs/runtime-ane/qwen-model-candidate-c288-auto/`。

以上 automatic 与先前固定 c288 的 1.100× 属于不同的调度条件，不应
只挑更快的一个数字当作默认保证；已有冻结 W8A8 图仍明显更快。

## 使用与重现

先按单层文档用公开 coremltools 导出 checkpoint-independent 图。
不要把 runtime manifest 与冻结 base manifest 混用；显式无效产物应报错。
当前 M4 Max 的 c288 候选可离线导出如下；其他硬件仍须自行验证。
目录必须不存在，导出器不会覆盖既有产物。

```sh
.venv/bin/python3 tools/coreml/export_runtime_ane.py \
  --rows 288 --hidden 3840 --width 10240 --tile-k 1024 --tile-n 1024 \
  --output outputs/runtime-ane/local-z-c288
.venv/bin/python3 tools/coreml/export_runtime_ane.py \
  --rows 288 --hidden 4096 --width 12288 --tile-k 1024 --tile-n 1024 \
  --output outputs/runtime-ane/local-qwen-c288
```

```sh
TURBOCIDER_NATIVE_ONLY=1 make build
# 无 SDK / 无推理的 host 回归也包含在默认 make test 中
make test-runtime-ane-host
# 下项额外需要 Core ML SDK，运行临时微图，不加载真实 checkpoint
make test-runtime-ane

# 使用任意匹配几何的本地 runtime manifest；Qwen H4096/F12288，Z H3840/F10240
build/native/turbocider plan examples/requests/qwen21-base-512.json \
  --hybrid-mode runtime --ane-manifest path/to/runtime-qwen/manifest.json
build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-base-512.json \
  --hybrid-mode runtime --ane-manifest path/to/runtime-qwen/manifest.json
```

请求必须明确允许近似且 `residency=resident`；Z 可用已有
`examples/requests/z-image-turbo-512.json` 加同样 CLI 参数。输出文件名固定，
重复运行前应改为独立路径。Qwen base 示例默认为 GPU，不包含 LoRA。

显式 runtime 模式的环境选项：

| 选项 | 含义 |
| --- | --- |
| `TURBOCIDER_RUNTIME_ANE_CHUNKS=auto`（默认） | 分层自适应 controller；不是固定比例资格保证 |
| `TURBOCIDER_RUNTIME_ANE_CHUNKS=1` | 固定一个 chunk，仅初版和明确标注 fixed 的 screen 使用；最终候选使用 auto |
| `TURBOCIDER_RUNTIME_ANE_CHUNKS=0` | 所有 FFN 走 GPU，但保留 split 边界；排查桥接/图切分成本 |
| `TURBOCIDER_RUNTIME_ANE_PROFILE=1` | stderr 输出逐层 host timing；正式计时保持关闭 |

`tools/validation/runtime_ane_model_screen.py` 可生成同条件独立请求和 PNG：

```sh
.venv/bin/python3 tools/validation/runtime_ane_model_screen.py \
  --model models/Comfy-Org-Qwen-Image-2.1 --model-id qwen-image-2.1 \
  --runtime-manifest path/to/runtime-qwen/manifest.json \
  --frozen-manifest path/to/frozen-qwen-w8a8/manifest.json \
  --output outputs/runtime-ane/new-screen --routes gpu,runtime,frozen \
  --steps 40 --warm-repeats 2 --chunks 1
```

反向顺序用新的 output 目录；脚本拒绝覆盖已有目录。有其他推理负载会等待，
持续繁忙则停止，不终止他人任务。脚本不对图片做自动视觉放行。
可显式传 `--prompt '...' --seed 7 --size 1024` 做新的匹配工作负载；
两路必须使用相同值，不能跨尺寸/提示词直接比较秒数。
结果门禁检查各路实际 backend、有限正值计时、失败/回退计数；
错误降级时保留原始 JSONL/PNG 并报错，不记入成功的性能汇总。
调度器正常选择 GPU 和 `chunks=0` 消融不属于错误降级。
进程检查只是启发式，不保证检测到所有后台 GPU/ANE 负载。

## BF16 最终候选的收尾状态（量化组件接入前）

native 构建及 `make test-runtime-ane`（8 项）、`make test-qwen21`、
`make test` 成功退出；默认测试里的缺失夹具/专用构建用例按规则跳过。
上述命令在最终候选源码上重新执行通过；benchmark 报告/输入检查
7 项 CPU-only 测试也通过。三组最终对照的 library SHA256 与当时的库
逐一核对一致，原始 JSONL 通过结果门禁。6 页文档本地链接及
staged/unstaged 空白检查通过；此前移出索引的 249 个产物仍在磁盘上。
本轮不修改默认路由，不删除现有最快冻结图或本地实验产物。
“第一版”表是 SIMD 前的历史数据；“最终候选”表才对应本轮最后的同二进制
整模型复测，不能把正确性测试通过本身称为速度提升。后续量化组件改动
属于新二进制，不自动沿用该 SHA 或性能资格。

## Q4/Q8 affine staging 组件（后续进展）

2026-09-28：参考 VPIPE 的逐行 staging，将 packed Q4/Q8 直接转换到
单层共享 FP16 IOSurface；**不是 ANE INT8 计算**，也不改变 checkpoint。

- `AffineView` / `WeightView` 明确区分 dense 和 MLX affine uint32 codes；
  保留逻辑 `[out,in]`，低位 code 在前，组大小 32/64/128，支持独立 row stride。
  scale/offset 可为 FP16/BF16/FP32，offset 缺省为零。raw GGUF block、
  ConvRot 和 NVFP4 不属于该 ABI，不能误解码。
- `ane_runtime_quant.hpp`：Q4 每组 16 项 FP16 LUT、每次解一个 packed word；
  Q8 使用 NEON。公式是 FP32 `fma(scale,code,offset)`，随后乘 headroom，
  最后舍入到 FP16；不在 CPU 转置、不额外分配完整 dense 权重。
  只检查实际使用的 LUT 值是否溢出；NaN/Inf metadata、存储不足和不合法
  几何拒绝，沿用 staging 错误后 GPU 完整重算的路径。
- `HybridFfn::stage_weights` 接受混合 dense/affine 投影，持有 packed、scale
  和 offset 的 MLX tensor，直到 worker join/drain；普通 dense `stage`
  仍可使用。此阶段公共模型入口仍限 BF16；后续 Z GGUF 接入见文末。
- host 测试覆盖 Q4/Q8 所有码值、三种 metadata dtype、三种 group、
  有/无 offset、非对齐指针与 padded stride、headroom、越界/溢出拒绝。
  真实 SwiGLU 微图 H64/F96/c32 的 12 组测试，对照原生 MLX packed GPU FFN，
  最差相对 L2 为 **0.00658136**（门槛 0.03），无错误回退；每次执行两个 chunk。
  同图继续切换到 dense/quant 混合层、释放调用者的源 tensor、注入错误
  metadata 后完整 GPU 回退也已通过。

转换 oracle 使用 MLX 解包，但先把 metadata 提升到 FP32。首次测试发现，
BF16 metadata 的 `dequantize` 即使请求 FP32 输出也带有 BF16 舍入，
会混入额外舍入边界；修正 oracle 后仍保持原转换误差门槛，未放宽测试。
原生 packed GPU FFN 对照仍使用原 metadata dtype，没有改成 dense GPU 冒充。

构建后的库 SHA256：
`36f48b2833f449a763f0054abd75992580988095bdeaa789d18f2dc126672e3a`。
该版本 `TURBOCIDER_NATIVE_ONLY=1 make build`、`make test-runtime-ane`
（4 项 host + 5 项图/集成）、`make test-qwen21` 和 `make test` 均成功退出。
缺失夹具/专用构建的旧用例仍按门禁跳过，不是全部模型验收。
首轮编译曾因构建期间源码变更被 identity 校验拒绝，固定源码后已完整重建；
上面的 SHA 是通过校验的最终构建，不使用失败构建进行性能测试。
该阶段本地 Z 模型目录尚未找到原生 Q4/Q8 GGUF 夹具；BF16、ConvRot INT8、
NVFP4 均不能冒充已验收的量化 checkpoint。下一步需保留量化 GPU attention/FFN
路径地接入真实模型，并分别测试性能、内存和视觉质量。
上述组件通过不等于 Q4/Q8 整模型或 runtime LoRA 已完成。

2026-09-28 收尾补记：后续下载的 `jayn7/Z-Image-Turbo-GGUF` 中
`z_image_turbo-Q8_0.gguf` 已完成文件大小（7,224,707,136 bytes）、SHA256
及本项目原生 GGUF 目录/支持类型校验。SHA256：
`f163d60b0eb427469510b8226243d196574a18139a2e40c017409cfbda95ecfe`。
下载完成时仅为本地待用 checkpoint，尚未接入 runtime 或生成/测速；
后续接入与真实 Q8 测试见文末。Q4_0/Q4_1 的真实模型验证仍缺失，
不因文件校验通过而视为整模型验收。

### 量化组件接入后的 BF16 回归

保持同一构建 SHA、Z 512² / 8 步 / fox / seed42 / resident，c288/auto。
首组 GPU→runtime→frozen 筛选得到 7.005 / 6.573 / 5.352 s 热中位，
但 runtime 结束后、frozen 开始前检测到新 ComfyUI 进程（72.6% CPU），
脚本等待后才继续；启动时间接近边界，因此这组只作诊断记录。
证据留在 `outputs/runtime-ane/z-affine-staging-dense-regression/`。

确认后台进程退出、常驻 ComfyUI 空闲后，新目录重做 runtime/GPU/GPU/runtime：
每个 trial 排除首个冷请求，再测两次热请求，共每路四个热样本。

| 路线 | 热请求中位 | 样本范围 |
| --- | ---: | ---: |
| GPU | 7.005 s | 7.005–7.008 s |
| runtime-weight（BF16 源） | 6.561 s | 6.546–6.640 s |

配对汇总比 **1.068×**，没有 Core ML 失败或错误回退；此轮各路运行前与
额外进程检查未见其他活跃推理，不保证设备独占。它支持本次 shared staging
接口调整没有明显拖慢已测 BF16 路线，不把相对旧 1.061× 的小差异归因于
新的加速算法。冻结图仍更快，不改默认选择。
证据：`outputs/runtime-ane/z-affine-staging-dense-abba/`。

GPU PNG 与此前候选的 GPU PNG SHA256 相同；当前 runtime/GPU 图肉眼主体、
姿态、雪地和色调接近，毛发/枝叶细节有差异，无黑图或纹理崩坏。
仍是单提示词/种子的生成样本，不是量化 checkpoint 或编辑质量验收。
MLX peak 两路约 16.03 GB；测试窗口 swap 使用量和 swap-in/out 计数未增长，
不是完整进程 RSS/wired 内存资格。Qwen 的该新构建未重新整模型测速，
不能把前面旧二进制的两模型表改称为本次新构建成绩。

## Z 原生 GGUF Q8_0：整模型接入与 chunk 筛选

本轮库 SHA256：
`a81bab7f984d7d6839c6161ab2a33093b4227fdbb364612bc5f7250390beb740`。
复用上文已校验的 Q8_0 checkpoint，tokenizer/text encoder/VAE 以相对链接
复用现有组件，没有复制权重。GGUF 实际包含 180 个 Q8_0、28 个 BF16 和
245 个 F32 tensor；main blocks 为 Q8_0，noise refiners 的矩阵为 BF16。

### 接入和正确性修正

- `z_runtime_block` 的 GGUF 分支为三投影构造 dense/affine 描述符，在 GPU
  attention 前启动 staging；GPU attention/FFN 继续使用 `Weights::project`
  的原生 packed 投影。不将整模型解量化、不把 uint32 当作 dense GEMM 输入。
- 保持原生 GGUF 的浮点 dtype promotion。若残差为 FP32，ANE headroom
  输出经 BF16 恢复指数范围后转回 FP32；新增真实微图验证输入有限、FFN
  输出超过 65504 时不产生 Inf 或错误回退。BF16 旧路径不改数值边界。
- 首次实际 GPU 运行发现混合 BF16 矩阵/F32 bias 的 refiner 被错误选入
  全 BF16 融合 Metal 图。现按 tensor dtype 检查融合图适用条件，混合层
  保留原生运算语义走兼容路径。原有 Comfy BF16 checkpoint 的 453 个
  tensor 全为 BF16，继续满足原快路径条件。失败记录保留在
  `outputs/runtime-ane/z-gguf-q8-c288-first/`，不记作性能成绩。
- plan、实际输出和 benchmark 验证均区分
  `mlx_cpp_metal_gguf+coreml_runtime_weight` 与 dense backend；同时报告
  checkpoint 量化和 runtime FP16 近似。GGUF 内存计划额外预留 2 GiB。
  请求须显式 resident/base-only/GPU+ANE/manifest/允许近似，拒绝 LoRA、
  encoder ANE 及不支持的混合模式；默认 GPU/冻结图选择不变。

### 初步整请求筛选

M4 Max 64 GB；512²、8 步、fox/seed42、resident；tile1024、SiLU exp、
chunks auto。各路线排除首个冷请求、取两次热请求中位，包含 VAE/PNG。
每路开始前检查常见活跃推理；仅发现空闲 ComfyUI，不保证设备独占。

| Chunk rows | 原生 Q8 GPU | Q8 runtime | 配对加速比 | 本地证据目录（`outputs/runtime-ane/`） |
| --- | ---: | ---: | ---: | --- |
| 288 | 9.187 s | 8.398 s | 1.094× | `z-gguf-q8-c288-native/` |
| 320 | 9.213 s | 8.282 s | 1.112× | `z-gguf-q8-c320-screen/` |
| 352 | 9.189 s | 8.160 s | 1.126× | `z-gguf-q8-c352-screen/` |
| 416 | 9.180 s | 8.151 s | 1.126× | `z-gguf-q8-c416-screen/` |

c352 与 c416 的整请求差别仅约 0.1%，不能作为明确胜负。c352 每热请求
staging 约 0.56–0.61 s，暴露的 stage wait 仅约 0.07–0.08 ms；GPU FFN
约 3.56 s，Core ML prediction 约 3.01 s，join 约 0.8–1.7 ms。
c416 虽将 GPU FFN 降到 3.24–3.32 s，prediction 却增至 3.43–3.46 s，
join 增至 0.33–0.34 s，抵消了 GPU 少算的收益。暂以更小、等待更少的
c352 作为复测候选，不把这种近似持平说成持续提高 ANE 比例就一定更快。
这些是 host 依赖/等待计时，不是物理 ANE 驻留或硬件 timeline 证明。
在本样本中进一步做双缓冲不能解决主要瓶颈，因为 staging 已基本隐藏。

上述初步筛选无错误回退。c288 的 GPU/runtime 狐狸图片已肉眼检查：主体、
构图、雪地和色调接近，毛发/树枝等细节有差异，无黑图或纹理崩坏；
单提示词不能代替广泛质量验收。MLX peak 均约 11.284 GB，额外 Core ML
不在该计数内。测试窗口系统 swap 用量及 swap-in/out 计数未增长；
退出时系统 wired 可能暂时升高，随后回落，不能用这些快照代替峰值内存验收。
这是 Q8 对自身原生 GPU 的提升，不意味着比更快的 BF16 冻结图路线快。

### c352 交错复测

同一二进制、输入及 c352/auto，runtime/GPU/GPU/runtime 四个独立 resident
trial，各排除冷请求后两次热请求，即每路线四个热样本：

| 路线 | 热请求中位 | 热样本范围 |
| --- | ---: | ---: |
| 原生 Q8 GPU | 9.199 s | 9.184–9.215 s |
| Q8 runtime c352 | 8.097 s | 8.071–8.133 s |

中位数比 **1.136×**。未见错误回退，首个冷请求的三次 headroom 重试将
scale 调至 64，热请求没有继续重试。两个 runtime trial 的自动 GPU-only
块数不同，说明控制器可选择不同 refiner 切分；所有行仍完整计算。
最终 GPU/runtime 的狐狸样本肉眼主体、姿态、色调及背景接近，细节不同。
证据：`outputs/runtime-ane/z-gguf-q8-c352-abba/`；不能外推其他设备、LoRA
或不同输入的固定收益。

### 1024² 长序列初测

同一构建、同一个 c352 图、auto 调度，1024² / 8 步 / 灯塔提示词 / seed7。
GPU 后 runtime，各一个冷请求加**一个热请求**，不是交错重复资格验收：

| 路线 | 热请求（含 VAE/PNG） | 热 denoise |
| --- | ---: | ---: |
| 原生 Q8 GPU | 41.041 s | 40.082 s |
| Q8 runtime c352 | 35.607 s | 34.671 s |

单样本比 **1.153×**。runtime 热请求在 256 个混合块内执行 1014 次预测、
处理 356,928 行，验证长序列会循环复用同一图，而非截断 token。
没有错误回退或新增 headroom 重试；槽大小 241,336,320 bytes，估算 runtime
开销 571,408,384 bytes。两路 MLX peak 同为 20,889,011,368 bytes，
不含 Core ML/OS；系统 swap 使用量及 swap-in/out 未增加，仍不是峰值内存验收。

已肉眼对照灯塔、悬崖、海浪和日落的主体、构图与色调，整体接近；
附属建筑烟囱/屋顶/窗户、云和岩石波浪细节有可见差异，无黑图或崩坏。
不能称为结构或像素完全不变，也不能外推 1024² 编辑/LoRA。
证据：`outputs/runtime-ane/z-gguf-q8-1024-c352/`。

### 同构建 BF16 / 冻结图回归

Q8 接入后用相同库重新运行原 Comfy BF16 checkpoint，512² / 8 步 /
fox / seed42 / resident，GPU→runtime→frozen，每路排除冷请求、两次热请求：

| 路线 | 热请求中位 | 相对本轮 GPU |
| --- | ---: | ---: |
| BF16 GPU | 7.008 s | 1.000× |
| runtime c288 / auto | 6.540 s | 1.072× |
| 冻结 W8A8，1024-row / 5120-channel | 5.361 s | 1.307× |

无 Core ML 错误或回退，GPU PNG 与上一 affine 构建的 GPU PNG SHA256 相同。
这支持原 BF16 快路径保留；两次顺序样本不说明新改动带来额外 BF16 加速。
冻结图仍是这里最快的已测路线，不以新 runtime 或 Q8 取代。
本轮未重新整模型测试 Qwen，不能把历史 Qwen 数字标成此构建的结果。
证据：`outputs/runtime-ane/z-gguf-integration-bf16-regression/`。

### 可复用 CLI 入口

先准备本地 GGUF、tokenizer/text encoder/VAE；导出的图不包含 checkpoint
或 LoRA 权重。输出目录必须是新目录：

```sh
.venv/bin/python3 tools/coreml/export_runtime_ane.py \
  --rows 352 --hidden 3840 --width 10240 --tile-k 1024 --tile-n 1024 \
  --output outputs/runtime-ane/local-z-c352
build/native/turbocider plan examples/requests/z-image-gguf-base-512.json \
  --hybrid-mode runtime --ane-manifest outputs/runtime-ane/local-z-c352/manifest.json
build/native/turbocider generate path/to/z-gguf-model \
  examples/requests/z-image-gguf-base-512.json --hybrid-mode runtime \
  --ane-manifest outputs/runtime-ane/local-z-c352/manifest.json
```

不传后两个开关则走原生 GPU。`--ane-manifest` 沿用显式近似授权契约，
`plan` 不加载模型，不能代替真实生成。c352 是 Q8 本机候选，不覆盖 BF16
的 c288 候选或已有更快冻结图。生成时无需 Python 或邻接仓库。

## 本次整理与验证

保留冻结图、完整 runtime LoRA 与显式 runtime-weight 的独立入口；
默认选择不变，实验性 chunks/profiling 仅影响显式 runtime 请求。
便携请求留在 `examples/requests/`，原始 JSONL/PNG/编译缓存留本地，
未删除模型或本地产物。代码职责和 optional 清单统一见
[维护入口](runtime-lora-acceleration-2026-09-28.md#代码维护边界)。

上述 `a81bab7f…` 构建已通过 `make test-runtime-ane`（4 host + 5 图/集成）
及 `make test-qwen21`。全部计时完成后重跑 `make test` 成功退出，包含
4 host / 10 benchmark 报告 / 11 CLI 契约（含 GGUF 便携示例的新断言）。
旧用例中缺失模型夹具或专用构建的项目仍按门禁跳过；不等于全面模型验收。

## 剩余工作

输出 SIMD、chunk 筛选、Z 自动策略和 Z 1024² 多 chunk 已有上述真实证据，
但更广的 tile/设备调优、Qwen 1024² 与编辑、Q4 整模型验证、runtime LoRA 的
性能优化与广泛质量资格、
不可变 graph artifact lease、进程峰值/换页和多提示词质量验收仍需继续。
更快的冻结图保留，不用新的 runtime 方案覆盖它。整体目标仍未完成。
