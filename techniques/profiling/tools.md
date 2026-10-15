# 工具箱：优先复用已有观察器

[总目录](../README.md) · [测量规则](measurement.md) · [新轻量工具](../scripts/README.md)

以下均从仓库根目录运行。`--help`用于确认当前版本参数；需要模型/NumPy/MLX/CoreML环境的工具不属于无依赖测试。输出使用新路径，先确认磁盘与无其他自有GPU benchmark运行。

## 只读环境与能力

```sh
git rev-parse HEAD
git status --short
sw_vers
uname -m
sysctl -n hw.memsize
dist/cli/turbocider doctor
```

doctor依赖已有CLI；不自动下载或package。记录真实加载library/runtime身份，而不仅当前Git HEAD。

## 性能 / 内存 / 生命周期

| 工具 | 用途 | 边界 |
| --- | --- | --- |
| [benchmark_native.py](../../tools/native/benchmark_native.py) | native C ABI完整请求/cold/warm | 具体参数、模型资格读help；不是任意recipe都支持 |
| [process_tree_sampler.py](../../tools/native/process_tree_sampler.py) | 独立Darwin进程树采样、hash-chain JSONL | 不制造压力、不改swap；driver/外部服务仍可能缺失 |
| [verify_process_tree_samples.py](../../tools/native/verify_process_tree_samples.py) | 采样完整性、identity/gaps校验 | complete不是低内存资格 |
| [benchmark_with_footprint.py](../../tools/validation/benchmark_with_footprint.py) | native benchmark self-process footprint | 需要task-footprint dylib；不覆盖CoreML其他进程；不是独立sampler替代 |
| [profile_z_image.py](../../tools/native/profile_z_image.py) | fusion/stages诊断 | 强同步profile不作正式E2E |
| [benchmark_ltx_env_abba.py](../../tools/native/benchmark_ltx_env_abba.py) | 单env候选ABBA/BAAB | LTX限定11steps，使用有效现有候选变量 |
| [branch_merge_benchmark.py](../../tools/validation/branch_merge_benchmark.py) | 两份library同任务配对 | 保留旧构建，不把新源码当旧binary |

先检查help：

```sh
python3 tools/native/process_tree_sampler.py --help
python3 tools/native/verify_process_tree_samples.py --help
.venv/bin/python tools/native/benchmark_native.py --help
.venv/bin/python tools/native/profile_z_image.py --help
```

采样新启动的已授权命令（同时会真正生成，非只读dry-run）：

```sh
python3 tools/native/process_tree_sampler.py --launch \
  --output "$TC_RUN/memory.jsonl" --correlation-id new-model-baseline \
  --interval-ms 100 --max-gap-ms 500 \
  -- dist/cli/turbocider generate "$TC_MODEL" "$TC_REQUEST"
python3 tools/native/verify_process_tree_samples.py "$TC_RUN/memory.jsonl"
```

`TC_RUN`须由调用者先创建为新目录，`memory.jsonl`本身必须不存在；变量不是默认值。采样间隔/允许gap应预声明，这里是示例，不是所有campaign的标准。`--pid`只附着；确认PID及出生身份，不能凭旧PID信号用户进程。

## 数值与媒体

| 工具 | 用途 | 注意 |
| --- | --- | --- |
| [compare_tensors.py](../../tools/native/compare_tensors.py) | paired safetensors CPU比较 | 当前小工具按reference目录枚举；要另外核对完整文件集合，不能以缺失未枚举tensor当pass |
| [quality_gate.py](../../tools/native/quality_gate.py) | decoded RGB corr/cosine/MAE | 支持的PNG格式有限；RGB不等于alpha；默认阈值需按任务先固定 |
| [video_quality_gate.py](../../tools/native/video_quality_gate.py) | 配对帧级/运动基本门禁 | 不代替人工时间一致性/AV同步 |
| [evaluate_ltx_lpips.py](../../tools/native/evaluate_ltx_lpips.py) | LPIPS诊断 | 需明确已有作者源码/权重；不自动下载 |
| [evaluate_ltx_temporal.py](../../tools/native/evaluate_ltx_temporal.py) | temporal/光流诊断 | 依赖OpenCV，metric不自动授予视觉资格 |
| [seedvr2_vae_phase_summary.py](../../tools/validation/seedvr2_vae_phase_summary.py) | VAE高水位归因 | trace span/去重owner范围明确，非全进程RAM |

```sh
.venv/bin/python tools/native/compare_tensors.py "$TC_REFERENCE" "$TC_CANDIDATE" \
  --report "$TC_RUN/tensors.json" --require-exact
python3 tools/native/quality_gate.py "$TC_REFERENCE_PNG" "$TC_CANDIDATE_PNG"
```

比较集合、shape、finite与exact位含义要按工具实现确认；strict bitwise包括signed zero时应比较原字节，而非仅`array_equal`。

## GPU / Core ML / ANE 组件

现有probe和screen按目标选择：

- [Metal Z组件](../../tools/native/benchmark_z_image_metal.py)、[Z shape sweep](../../tools/native/benchmark_z_image_shapes.py)、[LTX attention](../../tools/native/benchmark_ltx_attention.py)。
- [runtime ANE screen](../../tools/validation/runtime_ane_model_screen.py)、[完整FFN](../../tools/validation/qwen21_real_ffn_screen.py)、[混合调度](../../tools/validation/hybrid_runtime_schedule_screen.py)。
- [CoreML FFN bridge](../../tools/native/benchmark_coreml_ffn_bridge.py)、[SeedVR2 shared-tile](../../tools/validation/seedvr2_vae_shared_tile_screen.py)、[serial block stream](../../tools/validation/seedvr2_block_stream_screen.py)。
- 离线模板在 [tools/coreml](../../tools/coreml/export_runtime_ane.py)，原生推理不应启动这些Python导出工具。CoreML工具环境与runtime环境分离，版本按 [环境说明](../../docs/public/ENVIRONMENT_SETUP.md)选择。

部分probe需要独立build脚本/实验flag。先读文件顶部/`--help`和对应证据文档，不把示意参数当新API，不为了执行研究probe替换正式App。

## Instruments / GPU trace / 电源观测

需要设备归因时，用可用Xcode的Metal System Trace/GPU capture、CoreML Instruments；记录工具版本、capture影响和支持的计数器。仅CLT环境不保证这些GUI工具存在。CoreML compute plan先看supported/preferred device，不把它当runtime trace。

仓库 [IOReport diagnostic](../../tools/native/z_image_ioreport_diagnostic.py)属于诊断路径，不授予物理频率/占用保证。功率/频率采集若需额外权限，先说明再授权；本知识库不自动sudo、改电源/wired/swap、清cache或终止其他应用。

## 实现参考与实验

可读实现：[参考代码](../labs/summarize.py)；[实验步骤与边界](../labs/profiling.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
.venv/bin/python -B techniques/labs/ane/bench.py --kind frozen-fp16
```

本项控制：actual calls vs unsupported/no-device；setup/validation outside timed operator。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new hardware counters or instrumentation overhead。历史负结果限定于其来源/shape/边界，不关闭该方向。
