# Z-Image Q8 GGUF 1024² base：runtime v1 双提示词同库对照

接续 [BF16 base 对照](runtime-ane-z-1024-v1-2026-09-30.md)。本轮使用原生
Q8_0 GGUF checkpoint，验证设计稿中的 Q8→FP16 runtime FFN staging 在
完整文生图请求上的收益；没有修改 native 代码、模型、图、默认路由或 LoRA。

## 条件与结果

M4 Max 64 GB；`z-image-turbo-gguf`，resident，1024×1024，8 步、
无 adapter；runtime 使用 base-only v1 c352/K1024/N512 图、`chunks=auto`，
profile 关闭。原生库 SHA-256 为
`875b0d3dcf5c766feeced21d71f1a366acf0a1238bc3f0ed332098aa3b27b8cd`，
manifest SHA-256 为
`087d461c171b25e1dc66128a16d1944ffe860a8d22a4505cd80696c469aca5e8`。
GPU 与 runtime 的 `plan` 均可执行；后者标明 GGUF GPU + runtime FP16 FFN。
每个提示词按 GPU→runtime→runtime→GPU，四个独立 batch 各一冷两热；
每 batch 独立 100 ms 进程树内存采样与完整性验证。下表是四个热请求的
`request_wall` 池化中位数，包含 VAE/PNG，不含载入/冷请求：

| 提示词 / seed | GPU 四热样本 (s) | runtime 四热样本 (s) | 四热中位 GPU / runtime | 倍速 |
| --- | --- | --- | --- | ---: |
| 灯塔海岸 / 7 | 40.885 / 40.898 / 40.862 / 40.896 | 35.151 / 35.259 / 35.155 / 35.240 | 40.891 / 35.198 s | **1.162×** |
| 成年肖像 / 43 | 40.890 / 40.899 / 40.874 / 40.898 | 35.240 / 35.345 / 35.214 / 35.177 | 40.894 / 35.227 s | **1.161×** |

两组各减少约 13.9% 请求墙钟时间，超过设计稿的 5% 整请求初筛门槛。
这是同库、同模型、同提示词的配对对照；不能把历史旧库单热
41.041/35.607 s 当作本轮样本，也不能与 BF16 的 31/28 s 跨 checkpoint
直接算加速。原始 JSON、请求、PNG、内存采样与 `status=complete` 的汇总分别在：

- `outputs/runtime-ane/z-gguf-q8-1024-v1-abba-875b/`
- `outputs/runtime-ane/z-gguf-q8-1024-v1-portrait-abba-875b/`

四个 runtime batch 的 `runtime_failed=false`、错误 fallback 为零、
hybrid block 均为 704，正常 GPU probe 为 64。灯塔各有 2,583 次
Core ML prediction；肖像正反序分别 2,603/2,583 次，自动分块决策
不完全相同。每个 batch 都在冷请求发生三次 FP16 headroom overflow retry，
后两热请求没有新增 retry，稳定 scale 64。四组采样 verifier 全部 complete，
swap-in/out 都为零；进程树峰值 physical footprint GPU 约 43.59–43.65 GB，
runtime 约 44.67–44.94 GB，范围覆盖加载、冷、热和退出，不是 Core ML/ANE
专属内存。肖像第一组 GPU batch 采样到 56.7 MB compression，不能宣称
这次完整活动无内存压力，更不能外推到小内存设备。

已打开两组 GPU/runtime 末次热 PNG：灯塔/海岸/夕阳与浪花、成年人脸部、
卷发、雀斑、绿毛衣和窗口均保留，局部岩石/发丝/织物细节有差异，
没有观察到明显崩坏。灯塔同一路线正反序 PNG SHA 相同；肖像 GPU
正反序相同，**runtime 两次正反序 PNG SHA 不同**，但肉眼构图相近。
肖像 runtime 的预测次数也有差异，尚未证明二者的因果关系；不把这两张
图片的非逐字节一致忽略为完全确定性，也不把两种提示词当作广泛画质资格。
`quality_validation_enabled=false`，没有整模型数值真值校验。

`compute_units=cpuAndNeuralEngine` 表示请求的 Core ML 调度策略，遥测
`observed_ane_residency=unknown`；预测成功和性能收益均不能单独证明
物理 ANE 驻留或严格 GPU/ANE 并行。Q8 GGUF 权重、内存和质量资格均
未扩展到别的量化、尺寸、步数、设备、prompt 或带 adapter 请求。

## 复现和保留

同库、同肖像 prompt/seed 的固定分区探索各仅一个独立 resident batch：
`chunks=4` 两热 35.193/35.437 s，中位 35.315 s；`chunks=3` 两热
37.129/37.251 s，中位 37.190 s。两个 batch 的内存 verifier 都
complete、无 swap-in/out、错误 fallback 为零，冷请求各三次 headroom
重试且热请求未增加；各自的两张热 PNG SHA 相同。固定 4 的两热与本页
auto 35.227 s 接近，固定 3 明显更慢，但它们**不是交错 ABBA**，
不能将微小固定 4 差额判作性能结论，也不能由固定分区的批内相同 PNG
推断 auto 的跨 batch 输出问题已被解决。不改变自适应调度。原始证据在
`outputs/runtime-ane/z-gguf-q8-1024-portrait-fixed4-pilot-875b/` 和
`outputs/runtime-ane/z-gguf-q8-1024-portrait-fixed3-pilot-875b/`。

使用 [统一 screen](runtime-ane-validation.md)，本机 Q8 checkpoint 与
base-only c352/K1024/N512 v1 manifest，在空输出目录运行；肖像实验
改用上述肖像 prompt、seed43，另用一个全新输出目录：

```sh
.venv/bin/python3 tools/validation/runtime_ane_model_screen.py \
  --model models/z-image-runtime-gguf-q8 --model-id z-image-turbo-gguf \
  --size 1024 --steps 8 --seed 7 \
  --prompt 'A white lighthouse on a rugged sea cliff at sunset, waves breaking against dark rocks, detailed stonework, dramatic clouds, photorealistic.' \
  --runtime-manifest path/to/z-v1-c352-k1024-n512/manifest.json \
  --routes gpu,runtime,runtime,gpu --chunks auto \
  --warm-repeats 2 --sample-memory --timeout 900 \
  --output outputs/runtime-ane/new-z-q8-1024-abba
```

Q8 base 1024² 可保留 v1 作为显式 optional 候选；默认 GPU 不变。
下一步是更广泛的图像质量与自动分块确定性核查、内存压力资格及
硬件级放置实测；没有证据足以自动开启 Q8 runtime 或双缓冲权重槽。
本轮未执行全量 `make test` 或重建原生库；`git diff --check` 通过。
