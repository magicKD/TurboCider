# Runtime-weight ANE：不合并 base 的 LoRA 激活接口

2026-09-28，接续[base/GGUF 集成](runtime-ane-integration-2026-09-28.md)。
这是显式 `hybrid_mlp_mode=runtime` 的可选 graph-v2 接口，不替换更快的冻结图，
也不修改 GPU/`execution=auto` 的默认选择。它不是旧的 `lora_suffix`。

## 计算与权重复用

设计稿 §19 的 staging 权重合并不采用：用户要求 base 权重独立于 LoRA。
graph-v2 仍只有 base `wg/wu/wd` 三个运行时权重输入，新增：

- GPU 低秩计算得到的 `dg/du` 激活输入（chunk rows × FFN width）。
- 完整非线性后的 `h` 输出，供 GPU 做 down-LoRA。

逻辑为 `h = SiLU(x Wgᵀ + dg) * (x Wuᵀ + du)`，
Core ML 计算 `h Wdᵀ`，GPU 加上 `h Adᵀ Bdᵀ` 的适配器贡献。
GPU 行补集和 attention/modulation 继续执行完整 runtime LoRA。
不生成 `B @ A` 全矩阵，不改 checkpoint，不向权重槽写 adapter，
图与 adapter rank/内容无关；同一形状的图可用于 base 和不同 LoRA。

FP16 headroom 同时缩放 base up 与 `du`；每次溢出重试都重新按新 scale
填入 `du`。`y` 和 `h` 都恢复范围后才返回 GPU。任一 chunk 失败，GPU
重算整个 tail，不能把早先成功 chunk 与失败后残留的 hidden 拼在一起。
base 请求显式清零两个修正输入，防止继承上一个 adapter/layer 的激活。

graph-v1 保持兼容、无额外修正输入/hidden 输出。携带 LoRA 的请求必须
使用 v2；几何/LoRA 接口在内存准入前检查，不能以低内存回退掩盖选错图。
v2 也能执行 base，但多出的输入/输出可能使 base 比专用 v1 更慢。

## 模型与实现边界

- Z：resident Comfy BF16，`lora_strategy=inference_time`；base 和 runtime
  LoRA 共用图。GGUF 新 runtime 路线仍限 base；ConvRot/NVFP4/streaming 不开放。
- Qwen：resident BF16；LoRA 沿用 512²、六步 student schedule、一个 transformer
  adapter 的显式近似门禁。其他 adapter/strength 仅是实验能力，不代表已验收画质。
  不混用 frozen W8A8、DBCache 或 prefix 复用。当前没有编辑性能资格。
- `RuntimeGraph` 负责可选激活 IOSurface、worker、双输出恢复；`HybridFfn::Adapter`
  回调负责 GPU 低秩计算。所有输入 tensor 存活到 join，失败时不执行残留 hidden 的 down-LoRA。
- 更换 adapter 时保留 Core ML executable、清空调度计时模型；不沿用前一个
  adapter 的 GPU/ANE 平衡判断。同 adapter 的 resident 热请求保留调度样本。
- Qwen 的 GPU FFN 和 gate/up、down 修正使用 request-local 编译函数，
  不把前一 adapter 的捕获权重带入下一请求。此融合尝试尚未证明整请求加速。

## 整请求初测与负结果

M4 Max 64 GB，macOS 26.6.2，512²，fox/seed42，resident，c288 / tile1024 /
auto。各路线排除首个冷请求，再取两个热请求中位，包含 VAE/PNG。
事前检查常见 AI 进程；检测到下载/推理忙时等待，不终止其他进程。
进程快照只作启发式干扰检查，不是设备独占证明。

| 构建与工作负载 | GPU | runtime v2 | GPU/runtime |
| --- | ---: | ---: | ---: |
| 初版，Z distill patch，8 步 | 8.580 s | 8.295 s | 1.034×；尚未达到 5% |
| 初版，Qwen Viggle v0.2.1 r256，6 步 | 8.203 s | 10.778 s | 0.761×；变慢 |
| 补齐 GPU 编译融合后，Qwen 同配置，反转顺序 | 8.125 s | 10.811 s | 0.751×；未挽回整图损失 |
| 同一后续构建，Z c352，反转顺序 | 8.598 s | 8.293 s | 1.037×；与 c288 接近，不推广更大 chunk |
| 同一后续构建，Qwen `chunks=0` 拆图消融 | 8.158 s | 10.762 s | 0.758×；没有执行 ANE prediction 仍变慢 |

初版库 SHA：`dbac0a1611a9e0dd324318a3d199c50c66dda4068099b2fba5934bbabae0a2a9`。
编译融合/调度身份隔离后：
`ed7d8c8381e85b1b8be8f4d20fe0000e8bb984f638a8f7a586706838f33fdc9a`。
不同构建的行不作为严格单变量因果比较；后一组只证明当前 Qwen 仍慢。
本地证据（均在 `outputs/runtime-ane/`）：`z-runtime-lora-v2-c288-screen/`、
`qwen-runtime-lora-v2-c288-screen/`、`qwen-runtime-lora-v2-c288-compiled/`。
Z c352 证据：`z-runtime-lora-v2-c352-screen/`。更大 chunk 的 GPU FFN
累计时间减少，但 join 累计从初版 c288 约 0.070 s 增到 c352 约 0.578 s，
整图没有可靠的额外收益；跨构建筛选不是严格单变量实验。

以上无 Core ML 错误回退。Z c288/c352 与 Qwen 的 GPU/runtime 狐狸图已肉眼查看：
主体、姿态、构图和色调接近，毛发/雪点/松枝细节有变化，无黑图或纹理崩坏。
仅为单提示词/种子，不能推广到编辑身份、其他 LoRA 或广泛画质。

Qwen 编译融合后的第二个热请求仅 30 个混合块、162 个 GPU 块，说明自动策略
确实减少了混合计算，却没有回到普通 GPU 的速度：当前仍保留 attention/FFN
之间的同步和拆图边界，而不能只归因于 ANE 算力。
该热请求 staging 约 0.179 s，暴露的 stage wait 约 0.018 ms；继续做权重
双缓冲不是优先项。还存在 pre-SiLU 修正依赖、hidden 返回与 GPU down-LoRA，
以及任意 adapter 模式每请求的内容哈希检查。denoise 本身仍是约 9.65 s，
高于 GPU 约 7.51 s，说明哈希开销也不能单独解释退化。

### 关闭 ANE 后仍慢：拆图消融

同一 `ed7…` 构建的 `qwen-runtime-lora-v2-chunks0/`：GPU 两个热请求
8.123/8.192 s，runtime 拆图路径 10.732/10.793 s，中位分别
8.158/10.762 s；约多 2.605 s（31.9%）。denoise 中位为
7.504/9.603 s，差约 2.099 s，不能由模型加载或请求哈希单独解释。
三请求累计 runtime prediction、staging、hybrid block 和错误回退计数均为 0，
GPU block 为 576；初始化 self-test 仍会运行，但不在热请求计时内。

该实验定位的是**整条拆图 GPU 路径**的额外成本，不是精确测得某一个 fence
耗时 2.605 s。源码中 callback 安装后就选择 split block，调度器返回零 chunk
也不会恢复原来 unsplit 编译块；桥接仍 materialize 输入并同步 FFN。
这与自动模式约 10.81 s 的负结果相符，但两次小样本 screen 不能相减来
证明 ANE 净成本只有 0.05 s，也没有证明本机每个算子确实驻留 ANE。

下一步应先让调度决策早于 block 图选择，维护独立的 full/split 编译缓存，
并在重新探测时隔离此前排队的 GPU 工作；不能直接在复用同一编译缓存时
切换 `split_this`。本次收尾不引入未验证的调度重写，保留现有 GPU 默认。
后续已实现此 Qwen 改进，另见 [unsplit GPU 回退与新构建测试](runtime-ane-unsplit-2026-09-28.md)；
上表仍是 `ed7…` 历史构建，不改写成新结果。

### 内存与结论边界

记录的 MLX peak 不含 Core ML、wired memory 或 OS 文件缓存，不代表整进程峰值。
Z c352 和 Qwen chunks0 各自 screen 内 swap usage 均为 958.38 MiB，
swapins 保持 2726；更早 screen 为 2718，故不能宣称整个实验期间零换页。
还未完成内存压力/多模型常驻验收。这里是单提示词、单种子的小样本筛选，
没有新增 1024² LoRA、1–3 参考图编辑或第二个训练 LoRA 的质量/速度资格。

## 共图状态检查

扩展 `tools/validation/runtime_lora_shared_graph_switch.py` 支持 v2：
固定一个 chunk，先暖 base，再执行 base → adapter A → 合成 B → base。
合成 B 将 A 的 LoRA B tensor 乘 0.5，只用于状态隔离，不是第二个训练 adapter。
`--output` 可保留 PNG、原始 JSONL、请求和摘要；临时 B 文件在检查结束时清理。

Z 已通过：Core ML 累计调用数 259/515/771/1027/1283，加载时间不变，
绑定投影数 0/0/238/238/0；首尾暖 base PNG SHA256 相同，A/B PNG 不同。
证据：`outputs/runtime-ane/z-runtime-lora-v2-shared-switch/`。
Qwen 也通过：调用数 192/384/576/768/960，投影数 0/0/227/227/0，
图未重载，首尾暖 base PNG 相同、A/B PNG 不同。
证据：`outputs/runtime-ane/qwen-runtime-lora-v2-shared-switch/`。
Qwen 的六步 base 仅用于状态比较，不作为未蒸馏 base 的画质资格。
固定 chunk 是排除计时调度变动的状态检查，不是性能推荐。

## 可复现入口

```sh
# Z：替换为 Qwen 时使用 hidden=4096、width=12288，并用对应六步请求
.venv/bin/python3 tools/coreml/export_runtime_ane.py \
  --rows 288 --hidden 3840 --width 10240 --tile-k 1024 --tile-n 1024 \
  --lora-inputs --output outputs/runtime-ane/local-z-lora-v2
build/native/turbocider generate models/Comfy-Org-z_image_turbo \
  examples/requests/z-image-runtime-lora-512.json --hybrid-mode runtime \
  --ane-manifest outputs/runtime-ane/local-z-lora-v2/manifest.json
```

导出目录必须不存在。不开 `--lora-inputs` 的 v1 图不能接 LoRA；冻结
`lora_fused` 图也不是 runtime v2。Qwen GPU 仍为推荐 LoRA 对照，冻结图的
已有收益不被此实验覆盖。后续重点是关闭无收益层时恢复真正 unsplit GPU、
控制修正/hidden 传输开销，以及更广泛的生成与编辑视觉验证。

## 收尾验证与保留策略

graph-v1/v2、完整 gate/up/down LoRA、headroom 恢复、失败 tail 重算与共图
切换保留；新 runtime-weight 路由仍为 explicit optional，旧冻结图与普通 GPU
默认不改。离线导出/探针不进入推理时依赖。测试入口与当前验证状态统一记录在
[维护总览](runtime-lora-acceleration-2026-09-28.md#验证入口与历史边界2026-09-28)，
历史构建的整图数字不作为当前构建的回归成绩。
