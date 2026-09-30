# Runtime ANE：真实 Z-Image Q4 与 padding 兼容

更新：2026-09-29。本次只完成现有 Q4 工作的正确性收尾和 c288 小样本，
不继续 chunk 搜索，不改变最快冻结图或默认路由。
后续 c352 筛选及完整 block 调度见 [Z 调度接续](runtime-ane-z-block.md)；
本页保留修复阶段的原始结果，不作跨构建替换。

## Checkpoint 来源

从已有 `z_image_turbo_bf16.safetensors` 本地转换；没有下载新模型，
tokenizer、text encoder 和 VAE 通过相对软链接复用。
有效文件为 `models/z-image-runtime-gguf-q4-main/z_image_turbo-Q4_0.gguf`，
4,509,425,472 bytes，SHA256：
`687360c35c468ae1054f8a848d40a6c852c3c67d81f962dafcfbeff09e66d7d0`。

原始 BF16：12,309,866,400 bytes，SHA256：
`2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6`。
转换器 `sd-cli` SHA256：
`5c8ffa063b4cac12e91ef28440fb1ff4d2e018b6a363729b8e45d2ee313a4120`。
其 version/commit 输出未知，不仅凭安装目录名宣称对应某源码版本。

复现时指定自己的转换器与源文件路径：

```sh
path/to/sd-cli -M convert -m path/to/z_image_turbo_bf16.safetensors \
  --type bf16 \
  --tensor-type-rules '^layers\.[0-9]+\.(attention\.(qkv|out)|feed_forward\.w[123]|adaLN_modulation\.0)\.weight$=q4_0' \
  -t 4 -o path/to/new-z_image_turbo-Q4_0.gguf
```

GGUF 目录共 453 tensors：180 个 Q4_0、273 个 BF16；90 个主层 FFN
矩阵均 Q4_0。`cap_pad_token`/`x_pad_token` 在磁盘上为 BF16 `[3840]`。
MLX 加载后浮点常量为 FP16，矩阵仍为 packed uint32 + FP16 scales/biases；
不要混淆磁盘 BF16、加载 dtype 和 ANE FP16 staging。

先前全局 `--type q4_0` 误将 padding 常量也量化，不能用来比较性能。
该错误中间文件已在前一阶段删除，释放约 3.23 GiB，可从原始权重重建；
有效 Q4、原始 BF16、Q8 与历史失败日志保留。

## 修复与测试边界

初次有效 Q4 运行在 concatenate 时失败：GGUF 省略 padding 的 leading
singleton dimension，模型假设 `[1,H]`，实际为 `[H]`。失败证据保留在
`outputs/runtime-ane/z-q4-main-c288-screen/`，不覆盖。

`padding_token_row()` 仅接受 FP16/BF16/FP32 的 `[H]` 或 `[1,H]`；
前者 reshape，后者返回原 tensor，不改值/dtype。错误宽度、rank 和整数
packed token 拒绝。caption/image padding 共用这一个 helper。
新增原生测试覆盖 dtype、数值、现有 row 的 identity、重复拼接及非法输入。

构建 SHA256：
`f20b8e4093f519f336f4652dda64c17b020c2bf898c064887e0611f7c7de976e`。
构建成功；`make test-runtime-ane` 的 4 host + 8 图/集成测试通过。

## 真实整模型初测

M4 Max 64 GB；512²、8 步、狐狸雪景提示词、seed42、resident，无 LoRA。
顺序 GPU → runtime c288/tile1024/auto；每路一冷两热，冷请求不计中位。
每路运行前检查竞争推理；不是整个测试期间设备独占的证明。
依据：`outputs/runtime-ane/z-q4-main-padding-c288-screen/summary.json`。

| 路线 | 两次热请求 | 热中位 |
| --- | --- | ---: |
| GPU | 9.151510 / 9.102643 s | 9.127077 s |
| runtime | 8.606355 / 8.588249 s | 8.597302 s |

相对自身 Q4 GPU 约 **1.062×**。没有反向/交错复测，不能称为 Q4 最佳配置
或通用资格验收；更不能当成比 BF16/冻结图快。runtime 累计 639 hybrid、
129 GPU blocks、642 predictions，3 次 overflow headroom 重试至 scale64，
没有错误回退。Q4 GPU 仍执行 packed 投影；ANE 输入权重是 FP16，不是 INT8。

已肉眼比较最后 GPU/runtime PNG：狐狸主体、姿态、毛色、雪景构图接近，
树枝和背景细节不同，没有明显生成崩坏。只验一条提示词，不外推人脸、
文字、编辑或其他训练 LoRA；GGUF runtime 仍仅支持 base。
两路前后 swap usage 958.38 MiB、swapins 2786 未变，但 wired 增加；
MLX peak 不含 Core ML/OS，不据此宣称完整内存资格通过。

## 原 BF16 与最快冻结图回归

同一 `f20b8e4…` 构建、相同狐狸/seed42、512²/8 步，每路一冷两热，
顺序 frozen → runtime c288/auto → GPU，三路串行且逐路竞争预检。
证据：`outputs/runtime-ane/z-padding-cleanup-regression/`。

| 路线 | 两次热请求 | 热中位 | GPU/该路线 |
| --- | --- | ---: | ---: |
| GPU | 7.007920 / 7.004962 s | 7.006441 s | 1.000× |
| runtime | 6.574485 / 6.528099 s | 6.551292 s | 1.069× |
| frozen W8A8 | 5.357531 / 5.354760 s | 5.356145 s | 1.308× |

GPU/frozen 最后 PNG 与 `bc1…` 对应输出 SHA 完全相同；runtime 的自适应
切分输出允许不同，不用像素一致作为画质门槛。冻结图仍最快，没有明显回归。
本次也肉眼查看了 BF16 GPU/runtime PNG，主体、姿态与构图接近，局部细节有差别。
这次没有重跑 Qwen 整请求性能，不把此前 Qwen 数字标为此构建的测试结果。

## 整理收尾检查

- `make test-acceleration-contract`（随全量测试执行）：4 host、18 报告/预检、
  11 CLI 通过。`make test-qwen21` 和完整 `make test` 成功退出。
- 仓库布局 8 项通过，新增便携请求路径和 inference-time LoRA 检查。
  CLI 契约统一清除外部实验环境变量；Qwen/Z 集成缺库时分别正确 skip。
- 缺失夹具、专用 audit/test-hook 构建和 GPU opt-in 用例仍有既有跳过项，
  不当成实际模型、广泛质量或性能验收。
- 暂存区摘要保持不变，249 项原有暂存移除对应产物均仍在本地；未提交 commit。
  本次收尾没有继续删除模型、编译缓存或历史结果。
