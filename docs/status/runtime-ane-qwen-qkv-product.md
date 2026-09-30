# Qwen 1024² runtime QKV：产品接入与同库整请求初筛

结论：三份独立 BF16 Q/K/V 权重已能直填一个 checkpoint-independent
Core ML MatMul 槽，产品 block 的 Q/K norm、RoPE、attention 和完整 GPU FFN
保持原路径。**40 步整请求只观察到 1.013× 单样本收益**，慢于既有 FFN
runtime 与冻结 W8A8，且占用更多内存。保留显式研究路由，不纳入默认选择；
组件级 1.411× 不代表整请求收益。

## 产品边界

请求必须显式指定 `execution=gpu_ane`、`allow_approximation=true`、
`hybrid_mlp_mode=runtime_qkv`、QKV MatMul manifest、resident 1024×1024
base 文生图。无 LoRA、fused QKV、冻结/运行时 FFN split、FFN 步缓存、
DBCache 或 tiled prefill。启动时检查 32 层的三个 checkpoint 权重均为
无 bias 的 `[4096,4096]` BF16/FP16 矩阵；当前固定每 block 一个
1536-row ANE tail chunk，其余 token 在 GPU 计算。可用
`TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS=1...128` 显式选择 chunk 数，
默认 1；没有自动 QKV 调度。

模型侧在注意力输入归一化前发起三个权重源 staging；归一化完成后
并行提交 GPU head 与 Core ML tail。尾部 Core ML 失败时完整重算
**所有尾行**，不消费部分成功输出；每次归并完成并持有独立 MLX
输出，才允许下一层复用 Core ML buffer。取消/异常先 join worker，
再 drain 已提交的 GPU head。请求开始、宿主输出扩容前使用现有
Mach/MLX 准入，图与 scratch 的可选额度上限 2 GiB；此估算不覆盖
全部驱动或 MLX 工作集，内存采样另列。QKV 有独立的 `qkv` 回执，
不冒充 FFN 的 `hybrid.runtime_weight` 计数。CPU+NE policy 不能证明
物理 ANE 驻留。

## 最终库的 40 步同库对照

库 SHA256：`a7026013a131a761cf6553dd550ea02c6024c8a0b520b143befd884f96ddd69d`。
M4 Max 64 GB，BF16 Qwen base，1024×1024、40 步、seed42、相同狐狸雪景
提示词；无 LoRA/编辑/其他 GPU 诊断。一条路线一个 resident CLI batch，
各一次冷请求、一次热请求，顺序 QKV→GPU→FFN runtime→冻结图。
四路均开启同一个 100 ms 进程树采样器，逐 trial verifier 均为
`complete=true`、最大间隔低于 118 ms；库前后 SHA 相同。

| 路线 | 冷请求 s | 热请求 s | 热 denoise s | GPU/路线 | 进程树峰值 footprint GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU | 190.377 | 187.768 | 185.710 | 1.000× | 29.050 |
| QKV runtime + GPU FFN | 188.271 | 185.322 | 183.272 | 1.013× | 30.521 |
| FFN runtime v2 | 169.919 | 164.680 | 162.648 | 1.140× | 29.719 |
| 冻结 W8A8 FFN | 173.699 | 163.274 | 161.238 | 1.150× | 37.488 |

QKV 冷/热各有 1280 hybrid block、1280 次预测（32 层×40 步，
每层一个 chunk），无 GPU-only block、失败或尾行回退。QKV 的
graph slot 为 150,994,944 bytes，graph 估计总额 432,013,312 bytes；
实际进程树 footprint 峰值比 GPU 高 1.470 GiB，MLX allocator 峰值高
2.015 GiB。这些不是同一时刻的净资源归因，不能把估计 graph 字节
当成进程总增量。四路采样窗口 swap-in/out 均为 0；冻结图采样窗口
有约 114.6 MB compression，其他三路为 0。外部 Core ML 服务/驱动
内存与物理设备放置仍未归因。

GPU/QKV 的热请求 PNG 已目视比对，狐狸位置、背景和总体细节接近；
8-bit RGB 平均绝对差 0.619、RMSE 1.684，并非逐像素一致，也不是
多提示词的质量资格验收。仅一条热样本/路线且顺序未反转，1.3% 差距
可能受设备状态和请求波动影响；冻结图和 FFN runtime 在本次屏幕
明显更快，但不据此扩大默认路线或组合两个 Core ML graph。

完整逐请求 JSON/PNG、原始采样、验证报告与 summary：
`outputs/runtime-ane/qwen-qkv-product-1024-s40-a70260-screen/`，
`summary.status=complete`。另有不同库 `bb663001…` 的 5 步接入初筛
`outputs/runtime-ane/qwen-qkv-product-1024-s5-bb6630-screen/`，
四路热请求依次为 GPU 25.261 / QKV 24.852 / FFN 22.932 /
冻结 22.303 s；**不可**将其当成最终库的重复样本合并统计。

复现最终库的四路 screen（输出目录须尚不存在）：

```sh
.venv/bin/python -B tools/validation/runtime_ane_model_screen.py \
  --model models/Comfy-Org-Qwen-Image-2.1 --model-id qwen-image-2.1 \
  --size 1024 --steps 40 --warm-repeats 1 --sample-memory --timeout 1200 \
  --qkv-manifest outputs/runtime-ane/qwen-qkv-packed-c1536-k1024-n512/manifest.json \
  --runtime-manifest outputs/runtime-ane/qwen-lora-c320-k1024-n512-v2/manifest.json \
  --frozen-manifest results/qwen21/ane-compiled-4096/manifest-1d0b913e073fc44130c667a55f40c715622f7eb783e3e32ee7707250c6adb509.json \
  --routes qkv,gpu,runtime,frozen --chunks auto --qkv-chunks 1 \
  --output outputs/runtime-ane/qwen-qkv-repro-new-directory
```

最终库的 `TURBOCIDER_NATIVE_ONLY=1 make build`、115 项请求契约
（1 项因 fixture 缺失而跳过）、5 项 runtime host、12 项 Core ML/MLX
集成、加速契约及 Qwen 编译块 prefill/decode/路由切换回归通过。
下一门槛是反向顺序多热样本的完整 block/on-off 观察与内存策略，
再讨论自动分区或与 FFN ANE 的联合调度/准入；此版本明确不并存。
