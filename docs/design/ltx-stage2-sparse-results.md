# LTX‑2.5 Stage‑2 sparse attention：交付摘要

## 结论

目前没有测得同时满足 **Stage‑2 ≥1.5×** 和 **接近 dense 参考画质** 的方案。
已实现六种实验路由并接入 TurboCider；生产默认继续使用 dense。
这是对已测配置的结论，不是对所有 sparse attention 的不可能性证明。

测试机为 Apple M4 Max，64 GiB 统一内存。下表的完整视频实验使用同一 fox prompt、
seed 42、121 帧、24 fps、开启音频和 component-staged 驻留；另补充了人像
prompt／seed 7，结果见下文。
“480p/720p”是输出桶名称，实际尺寸分别为 **768×448 / 1280×704**。

| 已测配置 | 480p Stage‑2 | 720p Stage‑2 | 参考 RGB 质量 |
|---|---:|---:|---|
| all-exact 稀疏路径控制 | 0.950× | 0.922× | 两者通过，但比 dense 慢 |
| structured pooled，frame radius 1 | 1.061× | 1.203× | 两者失败 |
| direct top‑k32 | 1.042× | 1.238× | 两者失败 |
| pooled top‑k32 | 1.038× | 1.227× | 两者失败 |
| CiderSol-like，frame radius 4 | 0.995× | 未测完整视频 | 480p 失败 |
| Sol tau=0 | 1.008×（仅中间步） | 1.096×（全步） | 两者失败，步范围不匹配 |

各数字来自各自配对的 dense 基线，不是同一轮跨策略排名。仅 480p pooled
top‑k32 为 ABBA，其余表内数据为预热后的单次 AB pilot。720p top‑k32 含同帧
保留区域，480p 对应实验没有该区域；因此不能把跨分辨率差异只归因于尺寸。
完整配置及端到端数据见 [研究账本](ltx-stage2-sparse-research.md)。

720p direct top‑k32 是目前最大的已观察 Stage‑2 比率：125.29→101.21 秒，
但整次请求仅 1.106×；RGB 平均相关性 0.87390、MAE 18.93/255、LPIPS 0.29907。
pooled top‑k32 的 LPIPS 为 0.27153，仍明显高于 all-exact 控制的 0.05746。
首尾步保留 dense 能改善 pooled top‑k32 画质（LPIPS 0.14428），但 Stage‑2
只剩 1.066×，RGB 门槛仍未通过。

## 已集成什么

| `ltx_sparse_mode` | 行为 |
|---:|---|
| 0 | Sol 阈值路由，远端 pooled correction |
| 1 | 结构化精确区域，丢弃其余块 |
| 2 | 结构化精确区域，远端 pooled correction |
| 3 | CiderSol-like 局部／anchor 保留与动态路由 |
| 4 | pooled-QK top‑k 精确块，丢弃未选远端块 |
| 5 | pooled-QK top‑k 精确块，加远端 pooled correction |

这些是借鉴参考实现的 Metal-friendly 策略，不是 STA/LVSA/VSA 等论文的完整复现。
核心采用连续 64-token tile、SIMD-group matrix 运算和在线 softmax；不 gather
或复制选中的 K/V。路由 scratch 已改为 packed u32 bitmask，包含统计预留后，
480p 为 32,768 bytes，720p 为 197,120 bytes。这只降低 scratch，不代表等量
降低模型总内存；仍有 materialized block connectivity，并非完全无路由表。

请求必须明确开启 `allow_approximation` 和 `ltx_sol_stage2`。示例为实验配置，
不是建议开启的生产预设：

```json
{
  "allow_approximation": true,
  "ltx_sol_stage2": true,
  "ltx_sol_dense_edge_blocks": 1,
  "ltx_sol_dense_edge_steps": 0,
  "ltx_sparse_mode": 5,
  "ltx_sparse_keep_blocks": 32,
  "ltx_sparse_radius": 0,
  "ltx_sparse_anchor_stride": 0,
  "ltx_sparse_tokens_per_frame": 880
}
```

该 token/frame 值仅适用于本次 1280×704 几何；768×448 是 336，0 表示按连续块
解释 radius。完整请求还需模型、输出尺寸、prompt 等普通生成字段。

## 为什么 core 很快，整阶段仍不够快

真实晚层 QKV 的 720p direct top‑k32 core 约 4.05×，完整视频 video-self 分支
却只有约 1.74×，Stage‑2 为 1.238×。独立合成分支探针进一步测得，720p 的
QKV、gate、输出投影仍需约 140 ms，不能被稀疏 core 消除；FFN、text/cross
attention 等 Stage‑2 工作也保持执行。

已探索而未采用的改动包括 query fragment 缓存、按模式跳过摘要读取、输出
ConvRot/INT8 图融合、均匀细化远端摘要及固定逐-head 预算。部分只有 CPU
可行性诊断，不能当作 GPU 加速证据。逐-head 离线最优分配在同一 QKV 上能
降低误差，但固定用于另一 step/block 后退化；不能直接硬编码为模型策略。

## 复现入口与证据

在 TurboCider 根目录运行；使用本地已有权重，不将模型或生成物提交到 Git。
完整构建要求托管依赖，或显式设置 `MLX_ROOT` 指向包含 `include/`、`lib/`
的安装目录。Python 环境须有测试依赖；Metal 测量需要可访问 GPU 的会话。

```sh
TURBOCIDER_NATIVE_ONLY=1 bash tools/native/build.sh
bash tools/native/build_ltx_attention_probe.sh
python3 tools/native/benchmark_ltx_attention.py --parity-only --report /tmp/new-ltx-parity.json
python3 tools/native/benchmark_ltx_sparse_stage2.py \
  --model models/LTX-2.5 --output outputs/new-720p-pooled-topk32 \
  --width 1280 --height 704 --frames 121 --mode 5 --keep-blocks 32 \
  --radius 0 --anchor-stride 0 --profile
```

最后一条默认跑 ABBA；`--pilot` 才是单个 AB 对。输出目录必须是新目录。
生成后可用 `evaluate_ltx_sparse_run.py` 检查音频／视频张量，
`evaluate_ltx_lpips.py` 和 `evaluate_ltx_temporal.py` 补感知与时序诊断。
LPIPS 需作者源码和权重，参数见工具 `--help`；时序诊断依赖 OpenCV。

证据位于 `outputs/ltx-sparse-research/`，关键入口：

- `evidence-summary-v13.json`：17 组完成的配对报告，其中 14 组为 121 帧。
- `packed-route-parity.json`、`packed-route-abba/report.json`：路由格式正确性与重放对比。
- `720p-direct-topk32-pilot-121/report.json`、`480p-pooled-topk32-abba-121/report.json`：完整视频。
- `lpips-comparison-v1.json`、`temporal-comparison-v1.json`：独立质量诊断。
- `branch-attribution-v1.json`：合成分支耗时，不是最终视频或全 Stage‑2 基准。
- `final-native-parity-v1.json`：当前源码重建后完成的 41 项 Metal core 正确性检查。

新增第二场景验证：`480p-portrait-seed7-pooled-topk32-pilot-121/` 使用
121 帧人像转头 prompt、seed 7 和同一 pooled top-k32 配置。Stage-2 为
40.879→39.345 秒（1.039×），请求 wall 为 1.028×；latent relative L2
0.3000、cosine 0.9546。全帧 RGB 平均相关性 0.97588、最低 0.96694、
平均 MAE 6.077/255，仍未通过 RGB gate；LPIPS 0.08413，参考流残差 MAE
1.7997/255，flow endpoint 差 0.2307 个评估像素。Stage-1 video/audio 与
Stage-2 input video 保持 byte-exact；Stage-2 audio 随联合 latent 改变，
不作为 lip-sync 结论。`paired-frames.png` 仅用于人脸／头发／轮廓的定性抽查。

参考源码已检查 SolAttn MPS、attn-bench、LVSA、FastVideo；另已下载并使用
PerceptualSimilarity。准确 checkout revision、后续实验、失败记录和指标定义
均保留在研究账本中。

本次交付审计：原生构建成功；123 passed、1 skipped、96 subtests；manifest
列出的 13 个源文件 hash 全部匹配；`git diff --check` 通过。跳过项是缺少 Wan
fixture，不能当作测试通过。Metal parity 覆盖六种模式的短序列、尾块和部分
帧边界；更大路由字边界的证据来自先前 packed-route 对比，不能与本次 41 项
混称为同一轮测试。

## 尚未完成的验证

不能宣称找到可上线的近-dense 稀疏预设：原 fox 单 prompt 的 11 组近似
121 帧实验均未通过；新增人像样本同样未通过
既定 RGB 参考保真门槛。LPIPS／光流结果只是描述性指标，没有经校准的通过阈值。
仍缺少更多匹配配置的全策略双分辨率覆盖、多提示词／seed、身份压力集、
解码音频同步／lip-sync 和重复计时置信度；硬件带宽／占用率计数也未采集。
未来若有候选同时改善这些差距，应重新做匹配 dense 的完整视频验证，而不是
用本次孤立 core 的速度或离线 oracle 误差作为替代。
