# Qwen2.1 优化报告 4：DBCache 跨步残差复用

定稿：2026-10-16。单项实测：2026-10-02；最新代码整合实测：2026-10-15 至 16，Asia/Shanghai。

DBCache 在相邻去噪步变化较小时复用部分 Transformer 层的残差，从而跳过这些层的计算和权重读取。历史单项实验中，阈值 0.25、最多连续复用 4 步的配置将端到端中位耗时从 **52.711 s 降至 32.019 s，减少 39.26%**，每组 2 次。它是有损近似，图像细节会改变；当前 PR 的公开 `fast` 档保留这个强度。

四项组合在最新 main 上的正式对比为 **53.829 → 25.785 s，减少 52.10%**，每组 3 次。这是组合结果，不能归给 DBCache 单项，也不能将各份报告的收益相加。其余三项见 [GPU FFN suffix W4](01-gpu-suffix-w4.md)、[Attention W4](02-attention-w4.md)、[Q/K norm + RoPE](03-qk-norm-rope.md)。

## 适用条件与计时口径

实测设备为 Apple M5 Pro、24 GiB 统一内存；Qwen Image 2.1 BF16，512×512，40 步，狐狸提示词 `A red fox standing in fresh snow, soft morning light.`，seed 42。所有对照保持无 LoRA、无参考图，`gpu_ane`、`component_staged`、layer-streamed text/DiT 开启。ANE 使用原始 W8A8 FFN 前缀模型，32 层、1024 rows、hidden 4096、FFN 12288，ANE 覆盖 `[0,6144)`，GPU 覆盖 `[6144,12288)`。

此 PR 对流式混合路径的新增支持限于精确 M5 Pro / 24 GiB 设备及代码准入条件，包括 512×512、20–40 步、允许近似、已验证的 ANE 分区；不推断其他 M5 芯片或内存容量也适用。Core ML 配置为 `cpuAndNeuralEngine`，有真实调用及零失败记录；没有采集物理 ANE 利用率。

每个性能样本为新进程内的首次生成，取 `request_wall_including_export`：包含请求内加载、ANE setup、量化打包（若启用）、去噪、VAE 和 PNG 导出，不包含进程启动和 engine 构造。文件系统缓存未清空，不能解释为冷 SSD 时间。MLX peak 仅代表 MLX 分配峰值，不等于进程或系统总内存。

## 优化点与有效原因

实现位于 [transformer.cpp](../../../native/models/qwen21/transformer.cpp)，参数解析位于 [diagnostic_options.hpp](../../../native/models/qwen21/diagnostic_options.hpp)。每个 decode 步先计算前 8 层，得到前段残差，再与上一 decode 步的前段残差比较：

```text
change = sum(abs(front_now - front_previous))
         / (sum(abs(front_previous)) + 1e-6)
```

当 `change < threshold`、已过 8 步 warmup、不是最后一步，且未达到连续复用上限时，直接把最近一次完整计算的后 24 层残差加到当前 hidden 上，跳过这 24 层。否则完整计算并刷新缓存。首步完整 BF16 prefill 与 prefix KV 不走这种复用，最后一步也完整计算；采样仍执行 40 步。

这同时省去被跳过层的 GPU attention、GPU FFN suffix 和 Core ML FFN prefix，以及相应权重访问。在本工作负载中，未缓存时 Core ML 调用数为 `39 × 32 = 1248`；命中 K 步后为 `1248 - 24 × K`。Fast 命中 25 步，所以调用数为 648，而非关闭 ANE 获得加速。

缓存使用独立复制并求值的残差，避免保留未完成 MLX 计算图或共享 Core ML 输出缓冲；请求结束或取消后清理。前段相对 L1 变化只是后段可复用的启发式判断，**不是最终图像误差的上界**。完整路径在早期步骤已经加载过权重，因此 DBCache 主要减少重复工作，不能期待同等比例的峰值内存下降。

## 历史单项实测：公开 Fast 强度

2026-10-02 的 `qwen21-quantization-20261002/hybrid-dbcache-aggressive` 实验保持 attention、GPU suffix 为 BF16，关闭 Q/K 融合，仅改变 DBCache。执行顺序为 Off、max2、max4、max8、max8、max4、max2、Off。下表列出全部 8 次端到端耗时：

| 配置（阈值均为 0.25） | 两次耗时 / s | 中位耗时 / s | 较 Off 减少 | 命中步数 | Core ML 调用数 |
| --- | --- | ---: | ---: | ---: | ---: |
| Off | 54.103050、51.318275 | 52.710662 | — | 0 | 1248 |
| max2，对应 Balanced 强度 | 35.786173、35.718716 | 35.752444 | 32.17% | 21 | 744 |
| max4，对应 Fast 强度 | 31.988437、32.049579 | 32.019008 | 39.26% | 25 | 648 |
| max8，诊断配置 | 29.747363、29.335848 | 29.541605 | 43.96% | 28 | 576 |

Fast 的去噪时间为 21.909 / 21.958 s，对照为 42.242 / 41.233 s。进程读取计数的中位值从 227.876 GB 降至 113.843 GB；这是进程 I/O 计数，不能当作 SSD 实际物理读量。MLX peak 从约 5.665 GiB 增至 5.689 GiB，增加约 24 MiB 残差缓存。首个对照样本还记录了系统级 swapout 增量 909,180,928 B，其他样本为 0；系统计数不能归因于单个进程，样本数也不足以排除运行状态波动。

另一组早期 `hybrid-dbcache` 实验每组 2 次：Off 为 53.991512 / 52.601086 s，中位 53.296299 s；0.25/max2 为 36.521833 / 36.371608 s，中位 36.446720 s，减少 31.61%。0.08/max2 两次均零命中，53.110121 / 52.986950 s，没有证据表明该低阈值带来有效加速。

这些是历史实验二进制的单项结论，不是当前 PR 对所有单项重新进行的多轮正式消融。

## 精度代价与档位选择

历史狐狸样本相对 BF16 对照的 RGB RMSE（像素范围 0–255）为 max2 **2.9364**、max4 **5.0849**、max8 **8.7351**。视觉检查中狐狸和雪地语义保留，max8 毛发及雪地更平滑。RMSE 衡量像素变化，不直接代表审美或语义质量。

历史文字留出样本为 `A clean white poster with the exact words "BLUE SKY" in large dark-blue capital letters, centered, no other text, flat graphic design.`，seed 13，各 1 次：Off 51.049 s、max4 32.028 s、max8 29.559 s。三个输出均保持 BLUE SKY 正确拼写、白底蓝字，但字缘及间距有变化。单条英文短语不能证明中文、长文字或其他 seed 都可靠。

当前组合中 max8 在狐狸的三个探索样本为 24.050 / 24.262 / 24.135 s，但文字样本为 28.414 s，慢于 Fast 的 25.502 s；物体样本为 27.440 s，也慢于 Fast 的 25.845 s。后两项各 1 次，不能证明 max8 必然更慢，但足以说明尚未建立跨工作负载的一致优势。因此本 PR 保持公开 Fast 的 max4，未把 max8 变成公开默认档。

## 最新代码整合结果

实现提交 `242befe983d5b17439daecb5ef2ab44de8fa5a8f`，main 对照 `2a4c2b5b03b8dba4c53f0f8e8f37d0dfa1e6a441`。MLX 0.32.0，仍使用上述无 LoRA、流式加载、分阶段释放和 ANE 条件。正式 ABBAAB 顺序及原始耗时如下：

| 顺序 | 配置 | 耗时 / s | DBCache 命中 | Core ML 调用 |
| ---: | --- | ---: | ---: | ---: |
| 1 | main + Off | 54.713867 | 0 | 1248 |
| 2 | W4 suffix + W4 attention + Q/K 融合 + Fast | 26.400967 | 25 | 648 |
| 3 | 同上组合 | 25.351826 | 25 | 648 |
| 4 | main + Off | 53.828678 | 0 | 1248 |
| 5 | main + Off | 53.724513 | 0 | 1248 |
| 6 | 同上组合 | 25.784862 | 25 | 648 |

中位耗时 **53.828678 → 25.784862 s**，减少 **28.043815 s / 52.10%**，约 **2.09×** 加速。组合 MLX peak 中位数从 6,083,208,370 B 降至 5,829,385,454 B；不能把这个内存下降归给单独 DBCache。

同一最终二进制的探索性消融中，关闭 W4 与 Q/K 融合后，Off 为 53.383300 s，单独 Fast 为 33.061841 s、25 次命中、648 次调用。两者各只有 1 次，应视为当前路径可用的佐证，不替代多轮统计。组合狐狸姿态、后腿和尾巴有变化，不能将变化单独归因于 DBCache；文字及红方块/蓝杯/黄球顺序在小样本中保持正确，字体与材质有变化。

同一 engine 的 10 项生命周期检查涵盖文本第 4 层取消、去噪第 2/12 步取消后重试、关闭并重新开启优化，以及诊断 max8 第 12 步取消后重试。4 次取消没有输出 PNG，6 次完成均与各自正常配置的参考 PNG 字节一致；调用方 wired limit 从 256 MiB 提升到去噪阶段 6 GiB 后恢复，最后恢复原值 0。最大空闲 MLX active 为 169,252 B，engine 释放后为 5,412 B。这些是功能检查，耗时未计入正式性能统计。

## 如何启用与复现

客户端选择 Qwen2.1 BF16，512×512、40 步、无 LoRA/参考图/提示词增强，打开 ANE 并采用 component-staged，同时开启 **“Qwen 固定量化分区（实验）”**并选择匹配的已编译 ANE manifest。符合设备及请求准入时，DiT 缓存选择 **“快速”**（`fast`）；W4 suffix、W4 attention 和 Q/K 融合由该路径默认开启。只关闭 DiT 缓存会关闭 DBCache，其他三项仍然生效。

要只测试 DBCache，启动新进程前清除所有 `TURBOCIDER_QWEN21_*` 环境覆盖，再设置 `TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS=0`，两组仅切换请求的 `qwen21_dit_cache` 为 `off` / `fast`。总开关不覆盖显式 suffix/attention/QK 参数，所以应清除旧环境值。诊断 max8 必须省略请求的缓存字段，再设置 `DBCACHE_DIAGNOSTIC=1`、`DBCACHE_THRESHOLD=0.25`、`DBCACHE_MAX_CONSECUTIVE=8`（均带 `TURBOCIDER_QWEN21_` 前缀）；显式 `off` 会优先关闭诊断缓存，公开 `fast` 与诊断覆盖同时出现会报冲突。

以下从仓库根目录调用已有 [benchmark_native.py](../../../tools/native/benchmark_native.py)。先按仓库构建流程生成动态库，并准备同一 BF16 模型与匹配的原始 W8A8 ANE manifest。把下面三个路径替换为实际路径；脚本在临时目录生成请求和结果，每次子进程只运行一次。它使用当前实现进行独立 DBCache 对照，不会重新生成历史二进制的精确结果。

```python
import json, os, pathlib, subprocess, sys, tempfile

library = pathlib.Path("build/native/libturbocider.dylib").resolve()
model = pathlib.Path("/path/to/Qwen-Image-2.1-BF16").resolve()
manifest = pathlib.Path("/path/to/verified-w8a8/manifest.json").resolve()
output = pathlib.Path(tempfile.mkdtemp(prefix="qwen21-dbcache-"))
env = {k: v for k, v in os.environ.items()
       if not k.startswith("TURBOCIDER_QWEN21_")}
env["TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS"] = "0"
for index, mode in enumerate(("off", "fast", "fast", "off", "off", "fast")):
    cell = output / f"{index:02d}-{mode}"
    cell.mkdir()
    request = {
        "model": "qwen-image-2.1", "operation": "image.generate",
        "prompt": "A red fox standing in fresh snow, soft morning light.",
        "width": 512, "height": 512, "steps": 40, "seed": 42,
        "frames": 1, "audio": False, "dynamic_text": True, "loras": [],
        "execution": "gpu_ane", "residency": "component_staged",
        "qwen21_w8a8": True, "allow_approximation": True,
        "ane_manifest": str(manifest), "qwen21_dit_cache": mode,
    }
    request_path = cell / "request.json"
    request_path.write_text(json.dumps(request, indent=2))
    subprocess.run([
        sys.executable, "tools/native/benchmark_native.py",
        "--library", str(library), "--model", str(model),
        "--model-id", "qwen-image-2.1", "--request", str(request_path),
        "--output", str(cell), "--runs", "1", "--record-vm",
    ], env=env, check=True)
print(output)
```

检查每份 `report.json`：`plan.lora_count=0`、`plan.residency=component_staged`、`acceleration_selection` 包含 layer-streamed text/DiT、`hybrid.compute_units=cpuAndNeuralEngine`、`checkpoint_sha256_verified=true`、零 runtime failures。单项实验应无两类量化 receipt，且 Q/K 融合未启用；Fast 的 `qwen21_dbcache` 应记录 front 8、warmup 8、max4。检查实际 40 步和 `1248 - 24 × cached_steps` 调用数，避免将路径回退或条件未生效混入性能结论。

## 证据索引与限制

核心原始耗时已逐次列在本文。原始 JSON、PNG、模型和临时驱动保留在本地实验记录，未作为 PR 附件上传；以下 suite/run ID 与 SHA 用于定位同一批证据，不是仓库内链接。

| 证据 | 标识 |
| --- | --- |
| 历史单项 suite | `qwen21-quantization-20261002/hybrid-dbcache-aggressive` |
| 历史 native SHA256 | `be9f8a3aee67dd74805ddb744bb9bc38cf9d33e103daf8d6a29cc3a24cf73f25` |
| 历史 identity SHA256 | `fcd511484a0c8bc6394972c09e784254b9ed6c261ecb496c6477dcd0c128b932` |
| 当前 suite | `qwen21-m5-24gb-integration-20261015`，`formal` / `screen` / `quality` / `lifecycle-combined` |
| 当前 native SHA256 | `36305cc4999e6dca2ec0ecaa3c81263d24a49592602b2b6514fadc0704c43d53` |
| main native SHA256 | `394f27d1e4b653867846b8ae4484e8f66710071a68c574a18d96805847ca0f53` |

关键 `report.json` 的 SHA256 前 16 位如下；完整散列可由原始文件重新计算：

| suite / run | SHA256 前 16 位 |
| --- | --- |
| historical / `00-baseline` | `0633138e833d00c4` |
| historical / `02-db025x4` | `85fddd52e66ce41f` |
| historical / `05-db025x4` | `1ea4a6eb876a3e3f` |
| historical / `07-baseline` | `fcc568084226c2d3` |
| current formal / `1792079228692-upstream-fox` | `438a9d5d86424de7` |
| current formal / `1792079283508-fast-fox` | `50795defd252ef98` |
| current formal / `1792079310343-fast-fox` | `6b764c13ed5fb878` |
| current formal / `1792079335789-upstream-fox` | `68b13d3d46597b3a` |
| current formal / `1792079389774-upstream-fox` | `ad6edca4b8b22e5a` |
| current formal / `1792079443624-fast-fox` | `65f42f183f99216a` |
| current screen / `1792079609213-dbonly-fox` | `e456d3c16473d578` |
| current screen / `1792079692325-disabled-fox` | `602978d73f834729` |

此结论限于上述硬件、固定小样本及路径。单项多轮数据来自历史实现，当前正式多轮验证的是四项组合；没有覆盖长文本、大分辨率、多 seed 或更广泛提示词质量集。源码中的 [设备与默认策略](../../../native/models/qwen21/memory_policy.hpp)、[混合执行路径](../../../native/models/qwen21/pipeline.cpp) 和参数解析共同限定实际启用范围。
