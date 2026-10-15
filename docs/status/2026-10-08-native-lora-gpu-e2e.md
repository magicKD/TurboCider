# Native GPU LoRA：窄域请求与真实六步对照

日期：2026-10-08。机器：M4 Pro／48 GiB。分支：`dev-verify`。

**完整 Native GPU Viggle r128 路线可运行，当前实现没有性能收益。** 一组本地 512²／六步／同 seed 对照中，MLX request 为 20.939 s，native 为 25.437 s，native 慢 21.48%；两张最终图片目视接近，RGB 相对 L2 为 1.035%。保持显式 `experimental-unqualified`，产品默认路线不变，不据此开放 Native hybrid LoRA 或 App 自动选择。

本轮没有下载模型、复制 checkpoint、导出全模型或运行 ANE。只运行 MLX→native 各一次真实请求；没有 warmup、自动重试或第二组整图。较早的 generated 组件证据保留在[原状态](2026-10-08-native-lora-gpu-components.md)，不重写历史回执。

## 1. 构建与准入

新构建为 `tc-runtime-build-v1-6d914980eefe568b7acc3c919a50e145b14b35d35e44c6482b7b305488841748`，目录 [native-lora-e2e-20261008](../../build/native-lora-e2e-20261008/)。CLI、动态库和 manifest 来自同一次完整构建；567 个运行时源摘要一致。Session／pipeline／结果序列化严格 host 检查通过；新库 metadata API 的 12 项路由测试、监督器的 14 项 CPU mock 测试通过。完整构建只出现既有三项 Metal `fastMathEnabled` 弃用警告。

新增候选准入仅接受：

- Qwen Image 2.1，`image.generate`，512²、单帧、无参考图；
- `qwen21_denoiser_backend="native-metal-graph"`、纯 GPU；
- 已有 Viggle v0.2.1 r128、一个 transformer adapter、strength=1、六步；
- `inference_time`（模型默认 `auto` 解析到该策略）、允许该固定蒸馏学生；
- 近似 DiT cache、MLX compile、PE、legacy ANE／量化／streaming 路线关闭。

metadata planning 不读取 adapter；Session 从加载用的同一 held descriptor 验证既有 SHA-256 pin，再物化 factors、校验真实 owner 与完整 target。r256、任意 adapter／strength、多 adapter、编辑和 Native hybrid LoRA 仍明确拒绝。实际六步 sigma 来自原 Viggle schedule，没有把 20 步 Base 简单减为六步。

新增 planning 的 1 GiB LoRA 余量覆盖约 648 MiB factors、共享 scratch 与 headroom，是启发式估计，不是完整请求容量认证。

## 2. 实际执行证明

新增结果 `native_denoiser_lora` 来自 retained source、resolved snapshot、adapted importer 和物理完成的 run，不是请求字段回显。本次实际记录：

| 字段 | 实测 |
|---|---:|
| variant | `viggle-r128-6step` |
| adapter／rank／原始 strength | 1／128／1 |
| source generation／source kind | 3／`verified-held-fd` |
| resolved physical bindings | 195 |
| logical projections／ordered terms | 227／227 |
| completed steps | 6 |
| shared low／gate／up scratch | 52,047,872 B，约 49.64 MiB |
| GPU graph | `native_full_dit_euler_runtime_lora` |
| runtime precision | `bf16_gpu+runtime_lora_fp32_rank` |

adapter 内容 pin 是 `bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。实际 revision／verified artifact digest 和原始结果见[完整回执](../../outputs/native-lora-e2e-20261008/e2e-pair/receipt.json)。保持 FP32 low／因子解码、有序逐 adapter BF16 cast、bias 最后相加的独立 recipe。

Native phase 成功完成、未 taint／quarantine；新增存储数／字节、预留、待释放和 unknown 均为零。原 MLX source baseline 另计，不能把这个零存储计数写成完整进程内存已归零。现有 MLX baseline 仍使用其 pathname pin 检查；监督器只对五个原始资产前后 stat，不为 MLX 冒称 same-fd 来源证明。

## 3. 完整成本与图像

本地 Qwen BF16、相同 Viggle r128、seed 42，提示词为 `A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.`。各路线一个新进程，固定 MLX→native；系统 source／driver cache 未重置，不称冷启动配对。

| 指标 | MLX GPU | Native GPU |
|---|---:|---:|
| request wall | 20.938971 s | 25.436790 s |
| denoise | 14.299567 s | 17.647532 s |
| native prepare（已计入 request） | — | 1.296343 s |
| text encode | 2.878046 s | 2.694143 s |
| VAE decode | 0.922286 s | 0.985866 s |
| 外部 supervisor wall | 21.525431 s | 25.585870 s |
| `time -l` peak footprint | 17.04 GiB | 14.36 GiB |

Native request 多 4.497819 s，其中单列 prepare 占差额约 28.82%，denoise 多 3.347965 s。Native denoise 包含最终 latent 复制、来源复核、phase finish／释放；它不是 GPU 纯计算时间。prepare 不含更早的模型／adapter hash、绑定与物化；其余 request residual 仅是派生余项，不能作为独立 load／compile 实测。

已有进度日志显示首步 MLX 2.641 s、native 4.048 s，后五步均值分别约 2.332／2.719 s，native 稳态也慢 16.62%。Native phase trace 未溢出，六组 before／after checkpoint 完整，跨度为 4.047441、2.718429、2.719325、2.722195、2.717121、2.718526 s。它们包含 host 提交／等待／来源与观察成本，不归因为某个 GPU kernel；结果足以说明差额并非只发生在准备阶段。

两张图的蓝色茶壶主体、盖子／把手／壶嘴、桌面构图和光照目视接近。RGB 相对 L2 **1.035%**、MAE **0.668/255**、PSNR **44.64 dB**，仅作此对最终图诊断，不能替代中间张量或多场景质量资格。原图：[MLX](../../outputs/native-lora-e2e-20261008/e2e-pair/mlx.png)、[Native](../../outputs/native-lora-e2e-20261008/e2e-pair/native.png)；[图像检查](../../outputs/native-lora-e2e-20261008/e2e-pair/image-comparison.json)。

两个 CLI 的 process swaps 为 0；外部采样的 system swap used 保持 3305.75 MiB，swapins／swapouts 前后计数未增加。系统此前已有 swap，不能写成“系统无 swap”；native 内部 swap observation unavailable，也不用于证明零活动。`pmset` 未记录 thermal/performance warning，不独立证明恒定频率或没有降频。

## 4. 下一步优化，不追加本轮整图

静态审查优先级如下；这些是候选，尚无 operator 计时归因或预期倍数：

1. **低秩 GEMM。** 当前 8×32 scalar FP32 kernel 用长 K tile 循环和 threadgroup barrier；尤其 down 的 `X×A` 输入宽度 12288。先比较保留 FP32 low／因子精度的矩阵 tile 实现，维持 tails／小行 fallback；归约次序变化必须用新 precision recipe／revision、独立数值与完整路径筛选。
2. **FFN 拆分与物化。** gate/up 两次 base GEMM、两个 correction、独立 SiLU 取代 Base fused FFN。后续 epilogue fusion 要保留 pre-SiLU correction、每个 adapter 的有序 BF16 边界和 down 使用 corrected hidden；不能只减少 dispatch 就称正确。
3. **依赖与准备。** 当前逐 dispatch 的全 buffer barrier 及首次请求准备可能值得优化；先建立可信读写范围和独立持久资产，不能直接删除 barrier、缓存旧 request hooks 或复用一次性 graph／jobs。

Native GPU LoRA 改善后，再推进独立 hybrid correction／hidden ABI。当前 Base W1S 收益不转授六步 LoRA；Splash 式 W2 两 block 设备调度、编辑 encoder、Z-image 和 App 均需各自完成域、成本与质量验收。完整设计见[16](../design/qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)和[17](../design/qwen-image-gpu-ane-command-graph/17-native-lora-integration.md)。

## 5. 回执与清理

完整日志、结果、图片、构建 manifest 和运行库保留；编译中间对象／静态 archive／该新构建的 module cache 在物理完成与进程结束后按 owned 清单删除。没有清理模型、外部目录或系统 driver cache。

机器可读记录：[分析](../../outputs/native-lora-e2e-20261008/analysis.json)、[构建与执行验证](../../outputs/native-lora-e2e-20261008/verification.json)、[owned 清理](../../outputs/native-lora-e2e-20261008/owned-cleanup.json)。
