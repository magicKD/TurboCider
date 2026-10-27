# Runtime ANE：集成与 profiling 入口

当前模型、来源、尺寸、实际步数、首/热请求和质量限制统一记录在
[最快已测配置](../../docs/benchmark/CURRENT_FASTEST_CONFIGS.md)。这些是研究配方，
不是 App 默认路由或跨模型质量等价的排行榜。产品默认仍是 GPU、DiT cache off；
Private ANE 和所有近似路径都需要显式授权。

## 已接入主 native 的实现

| 路径 | 主要入口 | 选择边界 |
| --- | --- | --- |
| Qwen Base 完整 40 步 | `native/models/qwen21/pipeline.cpp`、`runtime_private_qkv.cpp`、`runtime_prefix_snapshot.hpp` | 固定 FFN R1024/C6144＋QKV R512、one-hop W staging、生成/编辑各自的 prefix 策略；约 28.7–28.8s 生成记录，不使用残差缓存 |
| Qwen Turbo | `checkpoint_recipe.hpp`、`scheduler.hpp` | 原始 Turbo 文件名和作者 8 步 sigma；独立模型根，不能把 Base 任意截成 8 步 |
| GGUF / ConvRot | `convrot_weights.hpp`、`native/backends/ane_ffn.cpp` | 格式独立配方；ConvRot 保留 Comfy H256 旋转基、W8 codes 与 F32 系数，不二次旋转 |
| LoRA GPU 融合 | `native/models/qwen21/metal/`、`diagnostic_options.hpp` | 显式、限定 adapter/shape/schedule 的 direct QKV 与 typed SwiGLU；不能推广为任意 LoRA 都获益 |
| Runtime conservative cache | `runtime_cache_options.hpp`、`transformer.cpp` | 默认关闭的近似档；仍做 40 次采样，但跳过部分 middle blocks；不与完整计算纪录混排 |

FFN / QKV 的程序与权重 bank 由 session 复用，不按 transformer 层创建模型。
便携的[shape descriptor](../../examples/coreml/README.md)已随源码提供，作为
collector 的 `--manifest` 传入；不必保留历史 `outputs/` 中的同内容配置。
`TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM=1` 禁止 owner 的失败重建；初始化 compile/load
单列，热请求的无重编译承诺必须用实际 selector observer 验证，不只检查计数器。
更换尺寸、来源、LoRA 或 recipe 的资格各自验证，不承诺热切换任意 descriptor。
GPU-dual、共享 compiled complement 和多 procedure family 保留为实验开关，
没有稳健完整请求净收益的组合不加入最快配方。

## 构建与回归

所有命令从仓库根运行，先完成依赖安装。使用新输出目录，不把旧 C++ helper
链接到新 ABI 的库；Python C API 测试也显式指定实际库目录。

```sh
TURBOCIDER_NATIVE_OUT=build/runtime-ane-check \
TURBOCIDER_ENABLE_PRIVATE_ANE=1 TURBOCIDER_BUILD_LIB_ONLY=1 \
TURBOCIDER_NATIVE_ONLY=1 bash tools/native/build.sh

.venv/bin/python -B -m unittest discover -s tests/repository
make test-qwen21-runtime-host

TURBOCIDER_TEST_NATIVE_DIR=build/runtime-ane-check PYTHONPATH=tests/native \
  .venv/bin/python -B -m unittest test_contract test_qwen21_lifecycle \
  test_qwen21_scheduler test_qwen21_convrot_runtime test_qwen21_vae_norm \
  test_qwen21_async_step_pair test_qwen21_qkv_prefix_snapshot
```

发布库另在新目录使用 `TURBOCIDER_ENABLE_PRIVATE_ANE=0` 构建，并运行
`tools/native/check_release_binary.py --library LIB --manifest MANIFEST`。
Private 研究库不能用于稳定版打包。当前集成验证记录见
[代码整理与验证](../../docs/status/runtime-ane-integration-2026-10-19.md)。

## 采集与分析

- [Qwen profiling 详细说明](qwen_base/README.md)：`collect.py` 为完整计算；
  `expanded_collect.py` 覆盖 Turbo、格式、encoder 和尺寸；`runtime_cache_collect.py`
  单独采集改变实际计算量的缓存档。具体 fastest 命令见上面的配置索引。
- `runtime_image_specs.py` / `runtime_image_collect.py` / `runtime_image_audit.py`：
  跨模型固定图功能与形状恢复；Z 当前最快仍是 Public frozen，不能用新 runtime
  功能通过替换其速度结论。
- `native_source_archive.py`：按库的实际 source manifest 校验并封存源码。
  原始 PNG、trace、内存采样、source archive 全部写新的 `outputs/` 子目录。
- `qwen_base/analyze.py` 与各 `*_audit.py`：复核实际请求、完整输出、权重来源、
  内存、编译/加载/提交与质量预算。stock 计时和 timeline 观察分开；外部推理
  负载不合格时保留原始时间，正式速度比保持 null。

不要在完整模型性能窗口中同时构建、跑测试或做重型 CPU 图片审计；不要停止共享
App/ComfyUI/ANE 服务来改善测试条件。历史一次性清理工具
`retire_obsolete_exports.py` 默认只预览；显式 `--apply` 会删除固定名单里的
派生产物。它仅归档小型元数据，大型派生权重必须从原模型重新生成，不能从归档
恢复；不要把它作为日常构建步骤运行。

`outputs/`、`output/`、`Output/`、`results/`、`build/`、权重和编译缓存均被忽略；
历史已跟踪的结果不因新增 ignore 规则而删除。便携请求放 `examples/requests/`，
小型测试 fixture 放 `tests/fixtures/`，结论放 `docs/`；不提交原始实验目录。
