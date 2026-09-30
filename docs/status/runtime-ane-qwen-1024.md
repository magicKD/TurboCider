# Qwen 1024²：runtime-weight 与 GPU/冻结图对照

当前 v2 共图的重复对照另见[1024² v2 记录](runtime-ane-qwen-1024-v2.md)
（已完成，当前 runtime 1.115×、冻结图 1.147×）。本页继续保留旧构建
与单热样本初筛，不改标，也不跨构建归因某项优化的收益。

本轮为 **base 文生图**，不是图像编辑或 LoRA。M4 Max 64 GB、resident、
1024×1024、40 步、seed42、狐狸雪景提示词，每路一次冷请求、一次热请求。
顺序 runtime → GPU → frozen，整组统一开启进程树内存采样。
只有一个热样本/路线，属于初筛，不是重复性能资格验收。

原生库 SHA256：
`bdc78921f6b07c170faae8d66fd3f57ff137fa94a87cfe3c2613bf0ea0c3936c`。
本轮整理仅修改 Python 测试工具和文档，没有重编译或改变这份原生库。
本轮测试进程已提前加载旧脚本，不受新增 summary 标记/default 影响；
本轮原命令已显式指定 `--chunks auto`。

## 配置与证据

- checkpoint：`models/Comfy-Org-Qwen-Image-2.1`。
- runtime：`outputs/runtime-ane/qwen21-c288-t1024-exp/manifest.json`，
  bucket288，原有图通过多 chunk 服务更长序列，没有为 1024² 再导出图。
- frozen：`results/qwen21/ane-compiled-4096/` 下的 all32 manifest，SHA 文件名
  为 `manifest-1d0b913e073fc44130c667a55f40c715622f7eb783e3e32ee7707250c6adb509.json`。
  同目录另一份 block0-only manifest 不是这次整模型对照。
- 整请求结果与 PNG：`outputs/runtime-ane/qwen-base-1024-c288-sampled-screen/`。
- 采样流：`outputs/runtime-ane/qwen-base-1024-c288-sampled-memory.jsonl`。
- 冻结 1024² 路线仍显式开启 `TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC=1`，
  不提升为普通请求默认。

## 结果

三路全部成功完成，无错误回退；进程树采样独立校验通过。
时间为 native `request_wall`，含 VAE/PNG，不含启动加载，冷请求另列。

| 路线 | 冷请求 | 热请求 | 热 denoise | GPU/该路线 |
| --- | ---: | ---: | ---: | ---: |
| GPU | 191.110 s | 188.399 s | 186.320 s | 1.000× |
| runtime c288/auto | 187.243 s | 181.206 s | 179.048 s | 1.040× |
| frozen W8A8 all32 | 182.268 s | 163.859 s | 161.780 s | 1.150× |

runtime 热请求真实增量为 531 hybrid、749 full GPU、32 full GPU probes
（包含在 GPU 数内）、456 untimed hybrid、2028 次 runtime predictions；
无错误回退。2028/531 大于 1，确认长序列路径确实循环执行多个 chunk，
不是仅复用后端标签。自动调度仍将不少 block 交回 GPU，不能假定 sequence
越长就一定全部适合 ANE。
冻结路线累计 2496 次 Core ML 预测、无错误回退，仍是本组最快路径；
runtime 的共图灵活性尚未抵消相对冻结图的额外运行成本。

速度收益约 3.8% 的请求时间减少，仅能说明此配置初测有小幅收益。这里既有
attention 成本，也有完整 block 调度、预测/传输/同步成本，不能由 FFN 或
INT8 理论吞吐推导整请求同倍加速。runtime staging 为 FP16，不是 INT8。

肉眼已查看 GPU/runtime 最后 PNG：狐狸主体、姿态、面部和雪地松枝构图接近，
有细微毛发/背景差别，未见明显崩坏。冻结图最后 PNG 也已查看，狐狸身份、
姿态和雪景构图接近，毛发/树枝细节差异更明显，没有明显生成失败。
仅一个提示词，不外推到文字、编辑或
其他参考身份质量。没有要求像素级一致。

## 采样和边界

所有路线都带相同采样器，时间只与本组比较；不要拿未采样的历史单路值拼表。
预检起始等待一次，记录中的 Python PID 是采样父进程，不能把这条等待当作
已经确认有其他 AI 竞争。预检是启发式而非独占设备保证。

采样器与 benchmark 均成功退出；独立 verifier 的 `complete=true`，
11051 个 sample、最大间隔 110.461 ms（上限 500 ms），无未知子进程或
采样缺口错误。进程树峰值 RSS 为 17428234240 bytes，footprint 为
40266986696 bytes。按实际 inference PID 对应路线的采样峰值：

| 路线 | CLI 峰值 RSS | CLI 峰值 physical footprint |
| --- | ---: | ---: |
| runtime | 14.371 GiB | 29.726 GiB |
| GPU | 14.000 GiB | 28.328 GiB |
| frozen | 16.210 GiB | 37.490 GiB |

各列是各自峰值，不保证同一采样时刻；不是 MLX allocator 的 peak 指标。
采样窗口系统 swap-in/out 增量均为 0，compression 增量 0，decompression
增量 3178496 bytes；不等于系统已有 swap 为 0。process wired 返回 0，
不能据此声称没有设备/驱动 wired 内存。
进程树 RSS/footprint 和系统 swap/compression 不等于完整 GPU/Core ML 服务
归因；即便采样通过，也不构成全部物理内存或 ANE residency 认证。

本组未验收 LoRA 或 1–3 图编辑；编辑测试入口与边界见
[复现指南](runtime-ane-validation.md)，默认路线选择见[当前结论](acceleration.md)。
