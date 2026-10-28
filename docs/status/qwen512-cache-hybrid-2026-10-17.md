# Qwen 2.1：512² 冻结 FFN × 保守残差缓存

2026-10-17，M4 Max / 64 GiB。只推进 Qwen Image 2.1 的文生图和参考图编辑；
Z-Image 未修改。目标仍未完成，本页记录一个显式近似组合候选，非默认推广。

## 机制和范围

沿用原 BF16 DiT / encoder / VAE，以及已有 source-bound frozen W8A8 图：
6144 个 FFN intermediate channels 交 Core ML，剩余 6144 个留 GPU。
再叠加既有 conservative DiT residual cache：前 8 blocks 始终计算，前 8 steps
预热，最多连续复用 1 step，relative-L1 threshold=0.15，末步完整计算。
不是新 GEMM kernel，也不是减少采样步数。

本次每个候选都完成 40 个采样步骤，15 步复用中间 24 blocks；冻结 prediction
从每请求 1248 降到 888，减少 360 次。逐请求检查 session counter 增量，
不拿累计计数当单次工作。原 checkpoint SHA 验证成功，runtime failure=0。
`cpuAndNeuralEngine` 是计算单元策略，不是所有算子物理驻留或重叠的证明。

实际输入为蓝茶壶生成、将已有蓝茶壶改成红色；编辑保留标准 1024 参考处理，
输出均为 512×512 RGBA PNG。未换权重、LoRA、参考尺寸或 scheduler。
seed 29 的生成 pilot 后，seed 73 同时跑生成和编辑。

## 同一窗口诊断结果

旧库为 `build/main-into-dev-20261015-after/libturbocider.dylib`。
每条路线独立 session；每个任务执行 off warmup → off → conservative → off。
首项排除，两个 off 控制取均值；候选每任务仅一项。所有计入比较的请求都命中
conditioning 缓存，但真正执行采样、VAE 和 PNG 导出；prefix snapshot 两路均关。
因此这是 repeated-prompt warm request，不是 fresh encoder 或严格冷启动排名。

| 路线 | 文生图 / s | 图生图 / s | 相对纯 GPU off 的 RGB SSIM（生成 / 编辑） |
| --- | ---: | ---: | --- |
| GPU，cache off | 42.694 | 54.621 | 1 / 1 |
| GPU，conservative | 31.109 | 41.011 | 0.9880 / 0.9974 |
| Frozen W8A8，cache off | 30.486 | 42.373 | 0.8987 / 0.9200 |
| Frozen W8A8，conservative | 22.228 | 33.804 | 0.8926 / 0.9198 |

相对自己的 frozen off 控制，组合候选生成耗时少 27.09%，编辑少 20.22%；
相对纯 GPU off 分别少 47.94% / 38.11%。这些比例来自真实整请求，不乘历史倍率。
seed 29 生成 pilot 为 30.488 → 22.464 秒，方向一致。

**不能称为近乎无损。** 组合对自身 frozen control 的 SSIM 是 0.98965 / 0.99743，
但对完整 GPU 已是 0.89262 / 0.91981；大部分累计差异在已有 W8A8 路线上就出现了。
对完整 GPU 的 RGB MAE 为 7.791 / 5.898（0–255），alpha max abs 为 2 / 7。
生成图可见壶把、壶嘴和釉面变化；编辑任务本身也未取得通用材质保真资格。
没有用增量近似的小误差代替整体画质结论。

四组 off-before/off-after 的 RGBA 均逐像素一致，未观察到跨请求残差污染。
这只覆盖本次输入，不证明所有 prompt / seed / reference / LoRA 的生命周期与画质。

## 负载、内存与失败记录

两条路线的完整进程树内存采样均通过 100 ms / max gap 500 ms 校验。
加载＋cold＋warm＋退出的 process-tree peak：GPU 32,279,239,960 bytes，
frozen 37,622,849,104 bytes；并非逐预设独立峰值，也不含外部 Core ML 服务 / driver 归因。
frozen 窗口系统 swap-out 184,811,520 bytes，GPU 窗口 0；不将系统换页归因到单一进程。

**两条路线 continuous competing-load 检查均失败。** 捕获到既有 ComfyUI Python
进程活动，未停止或修改用户进程。保留原始记录，`qualification_passed=false`；
该组不能据此升级为正式稳定性能排名或自动路由。

早期 sandbox 启动未取得采样能力；第二个 pilot 完成生成对照后，编辑因未显式
开启 full-reference W8A8 diagnostic 被拒绝。两个失败目录保留，后续修正工具参数，
没有移除产品校验或把失败回填为完整通过。

原始记录：

- `outputs/qwen512-cache-hybrid-pilot-20261017/`：sandbox 失败。
- `outputs/qwen512-cache-hybrid-pilot-v2-20261017/`：seed 29 生成完成、编辑 admission 失败。
- `outputs/qwen512-cache-hybrid-confirm-20261017/`：seed 73 两路线、16 PNG、完整内存记录。
  `summary.json` 保存自身路线观察，`joint-quality.json` 单列对完整 GPU 的累计差异。

## 实现和验证边界

新增显式 conservative 组合入口，而不是改变默认配置或放宽数值参数：

- 只允许 resident、512×512、40 步、base、原 BF16 组件、frozen W8A8。
- 必须明确 `execution=gpu_ane`、`qwen21_w8a8=true`、manifest 和近似许可。
- loader 仍核验原 source、32 blocks、1024 rows 和 6144-channel partition。
- 不接 LoRA、encoder ANE、GGUF、runtime-weight、W8A16 suffix、GPU fallback blocks、
  其他 canvas / 步数、memory-constrained 或 streaming 请求。
- 全尺寸参考编辑仍需已有 `TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1`；
  不借缓存入口绕过它。512 参考组合尚未接入。
- balanced / fast 仍只在 GPU 上开放；App 仍保留 GPU-only cache 控件，默认 Off。

独立构建为 `build/qwen512-cache-hybrid-20261017/libturbocider.dylib`，启用 private
代码编译但本候选实际使用 public frozen Core ML 路线；未替换已安装 App。
130 项请求合同中 129 通过 / 1 跳过，8 项组合工具 host tests 与 1 项 C++ options
编译执行通过。初次 sandbox 合同测试的 GPU 标识无法匹配 profile；在可识别设备的
环境重跑通过，未放宽硬件校验。没有声称完整 `make test`。

新库 named-preset 的真实请求复核另保存在
`outputs/qwen512-cache-hybrid-preset-20261017/`；该轮运行期间不并行 build 或 pytest。
该轮已完成 8 张图、每 session 9264 次实际 prediction，全部 40 步。
8 张输出（包括 cold、候选、前后 off 控制）与旧库等价 diagnostic 入口的 RGBA
逐像素相同；证明接入没有额外更改本批数学，不证明相对完整 GPU 无损。
独立核对在 `outputs/qwen512-cache-hybrid-verification-20261017/preset-parity.json`。

新库热态生成 30.352 → 22.208 秒（少 26.83%），编辑 42.365 → 34.842 秒
（少 17.76%）。编辑候选较上一窗口的 33.804 秒慢约 1 秒，不能只选较快窗口；
两轮当前观察范围为生成约少 27%，编辑约少 18–20%。新轮负载检查同样失败；
完整 process-tree peak 37,614,346,024 bytes，系统 swap-out 786,432 bytes。
因此仍没有正式性能资格或默认推广。

新库 SHA256 为 `1f768a335fa7fedf9ab84ba2a317e61c0b01d7217752916f9140eb93416f52c5`，
runtime ID 为 `tc-runtime-build-v1-ac7234205b6d3fd820d42f6bc5ad8f0a143f08e7eb3962eaffdd649c3c610860`。
所有本轮 owned inference/build 进程均已终止，未留后台测速。

复现诊断组合：

```sh
.venv/bin/python -B tools/validation/qwen_cache_hybrid_screen.py \
  --library build/main-into-dev-20261015-after/libturbocider.dylib \
  --output outputs/new-qwen-composition --routes frozen,gpu \
  --modes conservative --cases generate,edit --seed 73
```

新库去除 cache 诊断环境变量的复核：

```sh
.venv/bin/python -B tools/validation/qwen_cache_hybrid_screen.py \
  --library build/qwen512-cache-hybrid-20261017/libturbocider.dylib \
  --output outputs/new-qwen-preset --routes frozen \
  --modes conservative --cases generate,edit --seed 73 --cache-api preset
```

## 继续方向

当前可保留为显式速度 / 画质候选，不升默认。下一步区分 fresh encoder 与热态，
对照已有 512-reference 路线，扩大场景 / 多参考 / seed，研究更保真的 ANE 分区、
prefix-snapshot 与 cache 的合法组合以及 decode prefix-KV 消费成本。本表为了隔离
机制而统一关闭 snapshot，不是已开启所有产品优化的最强 GPU 重复编辑分母。
六步 Viggle student 保持独立来源、步数和质量口径。
磁盘约剩 19 GiB，本轮无下载、无删除权重或用户缓存。
