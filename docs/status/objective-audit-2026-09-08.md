# TurboCider 原始目标完成度审计

更新时间：2026-09-08

本文按原始目标文件 `pasted-text-1.txt` 的四项要求逐项检查当前 `dev` 分支，不把静态代码存在、单次局部 benchmark 或旧 reference 工程的结果误认为完整验收。

## 总结

当前 TurboCider 已是可构建、可独立启动的 Apple-native runtime，正式构建不链接 private ANE API，FLUX、Z-Image、LLaDA、H3/LTX 以及 GGUF 的主要路径和统一请求/计划/生命周期已经接入。近似路径也已正确区分“允许尝试”和“质量/性能已验收”：默认 GPU，`gpu_ane` 必须显式许可和 manifest，自动策略只使用有实测证据的设备/shape/checkpoint 桶。

但是，原始目标还没有全部完成。主要未闭环项是：GGUF streaming 缺少多轮 ABBA 和更高分辨率矩阵；LTX hybrid 的 RGB/感知质量仍未通过；LLaDA LoRA、H3 动态 pinned-prefix、FLUX 9B hybrid 和跨机器/多 seed 验收尚未完成。当前分支不应被标记为“所有模型和所有 ANE 路径已经生产级完成”。

## 逐项证据

| 原始要求 | 当前证据 | 状态 | 说明 |
|---|---|---|---|
| TurboCider 自包含、整理并可提交 | `make build`、`make test`、`make package`；`docs/status/independence-and-dependencies-2026-09-06.md`；`tests/repository/test_independence.py` | 已完成（有边界） | H3/LTX runtime 已 vendored；模型权重、FFmpeg、FastMetal Python worker 仍是明确外部资源边界 |
| GGUF 量化和低内存 streaming | `native/models/z_image_gguf_module.cpp`；`docs/design/z-image-gguf.md`；`validation/z-image-gguf-streaming-2026-09-08.json` | 基础完成，矩阵未完成 | sd.cpp disk-backed streaming 已工作；Q3_K_S 256² 物理 footprint 约降 36.1%，但目前每路线仅单 warm 样本 |
| Core ML/ANE 启动开销、预加载和隐藏 | `docs/design/coreml-ane-startup.md`；`validation/coreml-startup-overhead-2026-09-08.json` | 已完成 | load-only、zero-input warmup、first/subsequent prediction 已分开；尚需更多机器和多轮缓存 ABBA |
| private ANE 实验但不进入正式产品 | `experimental/private-ane/README.md`；`validation/private-ane-shipping-isolation-2026-09-08.json`；shipping binary 审计 | 已完成（研究结论有限） | 正式 binary 无 `_ANE*` 符号/私有 framework；private 路径仅保留实验记录，不能当产品性能保证 |
| Transformer tensor/sequence/head/CPU/GPU/ANE 对比 | `docs/design/transformer-heterogeneous-report.md`；`validation/transformer-heterogeneous-2026-09-08.json` | 主要实验完成 | FFN channel split 是当前推荐；attention/sequence/head split 的负结果和 Amdahl 限制已记录；仍需更多 block/shape/机器复测 |
| 不要求逐 bit/逐像素完全对齐 | `native/runtime/plan.cpp`；`tools/native/quality_gate.py`；`tools/native/video_quality_gate.py` | 机制已完成，模型资格分层 | 允许近似不等于无条件通过；LTX hybrid 当前明确失败并保持 GPU 默认 |

## 近似正确性审计

统一计划在显式 `gpu_ane` 下要求 `allow_approximation=true` 和 ANE manifest。输出仍必须满足 shape、帧数、steps、operation、finite、checkpoint/LoRA provenance 等硬约束。图片默认离线门禁是 correlation ≥ 0.99、cosine ≥ 0.995、MAE ≤ 5/255；视频还要求最低单帧 correlation ≥ 0.95、motion-energy 误差 ≤ 15%。这些阈值是 workload 策略，不是数学定律。

为避免把对齐 RGB 指标误认为唯一质量定义，新增了可选的公开 Vision feature-print 诊断。该诊断不下载外部模型、不在运行时跑参考推理，也不会自动放宽路由；必须按 OS、Vision revision、模型和 workload 校准后才可使用。当前 LTX 样本的 Vision 距离均值 `0.1862`、最大值 `0.2079`，但 RGB gate 仍失败，因此仍不能自动启用。

## 性能证据边界

当前正式可引用的 M4 Max 64 GB resident warm 结果：FLUX 4B GPU+ANE `1.392×`、Z-Image BF16 GPU+ANE `1.212×`、LLaDA GPU+ANE `1.203×`。这些数字只适用于记录中的 shape、checkpoint、manifest 和设备；LLaDA 仍是显式候选。H3/LTX 的局部 MLP 或历史 hot-path 加速不能直接写成完整点击到媒体输出的保证；LTX 704×448/97 帧 hybrid 当前完整视频质量未过门禁。

## 尚未完成的明确工作

1. GGUF streaming 多轮 ABBA、1024²、多 seed 和多量化完整矩阵。
2. LTX GPU+ANE 的多 prompt/seed 质量修复，以及低内存 16/24/32 GB 验收。
3. H3 动态 pinned-prefix 和正式 GPU+ANE 端到端资格。
4. LLaDA 独立 LoRA 文件、in-memory/inference-time 分支和多机器验证。
5. FLUX 9B GPU+ANE；FLUX/Z-Image/GGUF LoRA 的异常、取消、多 adapter、多尺寸矩阵。
6. 以统一的感知/语义指标补充 RGB 门禁，但仍需保留 tensor、media、性能和 provenance 门禁。

## 当前可提交性

工作树在本审计开始时干净，`dev` 相对 `origin/dev` 领先两个提交。新增的 Vision 诊断、门禁测试和本审计文档完成后，应先运行：

```sh
make build-vision-quality
make test
git diff --check
```

模型权重、`outputs/`、编译产物、Core ML cache、Python 环境和本机 profile 仍按 `.gitignore` 排除，不应随提交进入仓库。
