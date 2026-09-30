# Runtime v2：同构建 GPU / 共图 / 冻结图对照

接续 [v2 base hidden 优化](runtime-ane-v2-base.md)。上一轮只测 v2
before/after，未在新库重测 GPU/冻结图；本次补齐同构建、双向顺序的
对照，不用历史 v1 或不同构建的时间计算新的 GPU 加速比。

## 条件与口径

保留原生库 SHA：
`b55419e636ada12e54dc197b7099811225b361d63d2c1d16e6e0560449104dba`。
M4 Max 64 GB，resident，512×512，狐狸雪景/seed42，无 LoRA。
Qwen 40 步、runtime c320/K1024/N512；Z 8 步、c352/K1024/N512。
runtime 使用支持完整 LoRA activation corrections / hidden 输出的 **v2**
base-only 图，无 adapter 时输入零修正；不合并 LoRA，不跳过 SiLU。
冻结图沿用此前匹配的 base artifact，没有重新导出或更换分区。

每模型依次 GPU → runtime → frozen → frozen → runtime → GPU；每个
trial 独立 resident batch 一冷两热，各路线共四个热样本。下表中位数
直接取四个热请求，包含 VAE/PNG、不含冷请求。`chunks=auto`，profile 关闭。
模型间串行，编译和测试不与计时重叠；每 trial 启动前检查竞争推理。
这不是设备独占、热稳态或多提示词资格。

此前[转换任务粒度候选](runtime-ane-staging-partition.md)已恢复，未替换
产品库，本次不是四组转换的整模型性能实验。

## Qwen：完整对照

证据：`outputs/runtime-ane/qwen-base-v2-current-palindrome/summary.json`，
状态 complete，库 SHA 与上方一致。

| 路线 | 四个热请求（s） | 热中位数 | GPU/该路线 |
| --- | --- | ---: | ---: |
| GPU | 42.651751 / 42.702894 / 42.684023 / 42.658996 | 42.671510 s | 1.000× |
| runtime v2 | 36.926688 / 36.964424 / 36.954172 / 37.060998 | 36.959298 s | 1.155× |
| 冻结 base 图 | 30.157761 / 30.225641 / 30.160383 / 30.234506 | 30.193012 s | 1.413× |

runtime 比自身 GPU 少约 13.4% 耗时，但比冻结图仍多约 22.4%。因此共图
复用已具有 base 性能价值，不代表替代最快冻结图，也不代表 LoRA 运行
本身获得该加速比。

两个 runtime trial 各累计 3648 predictions/hybrid blocks、192 GPU
blocks（含冷请求），无错误回退；两个冻结图 trial 各 3776 predictions。
六个 trial 系统 swap-in/out 页数均无增量。MLX peak 不含 Core ML/OS，
系统快照不等于完整进程峰值内存或物理 ANE residency 证明。

已肉眼查看三路末次热请求：狐狸主体、姿态、构图和色调接近，runtime
局部毛发/背景细节不同；冻结图尾部/毛发/枝叶差异更明显，整体正常。
各路线正反向末次热 PNG SHA 分别一致，runtime PNG 也与上一轮 v2
优化后的相同；不将单一样本推广为所有输入无损。

## Z：完整对照

证据：`outputs/runtime-ane/z-base-v2-current-palindrome/summary.json`，
状态 complete，原生库与 Qwen 相同。

| 路线 | 四个热请求（s） | 热中位数 | GPU/该路线 |
| --- | --- | ---: | ---: |
| GPU | 7.001986 / 6.997085 / 6.996468 / 6.991164 | 6.996776 s | 1.000× |
| runtime v2 | 6.456249 / 6.431457 / 6.466523 / 6.418768 | 6.443853 s | 1.086× |
| 冻结 base 图 | 5.345351 / 5.342793 / 5.344630 / 5.339090 | 5.343711 s | 1.309× |

runtime 比自身 GPU 少约 7.9% 耗时，比冻结图仍多约 20.6%。两个 runtime
trial 各累计 688 predictions、685 hybrid blocks、83 GPU blocks（含冷
请求），无错误回退；两个冻结图 trial 各 768 predictions。六个 trial
系统 swap-in/out 页数均无增量，仍不替代完整内存与物理 overlap 验收。

已肉眼查看三路末次热请求：狐狸正面主体、构图、雪地和色调接近，runtime
局部眼部/毛发有差别，冻结图背景枝叶更不同，未见明显新增崩坏。各路线
正反向末次热 PNG SHA 分别相同，runtime PNG 与上一轮 v2 相同。

## 结论与复现

这组结果补齐了 **同一 LoRA-capable v2 图运行 base** 的两模型 GPU
对照，超过各自 GPU 5% 耗时改善的筛选门槛；不能用它取代其他验收项。
冻结图仍是已测最快 base 路线，runtime v2 的优势是共图/跨层动态权重
复用，不是目前的绝对最低延迟。此次没有新的产品性能改动，不能将已保留
的 v2 优化重新宣传为又加速了一次。

使用 [统一复现工具](runtime-ane-validation.md)，将 `--routes` 设为
`gpu,runtime,frozen,frozen,runtime,gpu`、`--chunks auto --warm-repeats 2`，
传入相同模型的 v2 和匹配冻结 base manifest。Qwen/Z 分别使用 `--steps 40`
和 `--steps 8`；默认 512²、狐狸 prompt、seed42 与本次一致，每次使用
尚不存在的 `--output` 目录。不要把不同构建、不同图接口或不同步数混比。

## 收尾验证

两模型的六-route screen 均返回 0、summary complete；计时结束后才运行
`make test-acceleration-contract`（4 host + 27 报告 + 8 切换工具 + 11 CLI），
以及布局 8 项、独立性 7 项，全部通过且无 skip。没有重跑全量 `make test`
或微图集成；候选只修改过独立 probe 对应的转换任务粒度，最终源码恢复，
产品库 SHA 保持不变。文档本地链接目标与 `git diff --check` 检查通过。
未删除模型、缓存或图片，未 stage/commit，既有 249 项暂存移除及暂存
diff SHA 与本轮前一致。

## 适用边界

当前性能只适用于上述 base 生成工作负载。没有新增 LoRA、1–3 参考图
编辑、1024²、多训练 adapter 或更多提示词资格。Qwen 六步 LoRA/编辑
仍优先 GPU；保留最快冻结 base 图，runtime v2 仍为显式 optional。
阶段性对照完成不等于设计中的所有 runtime、overlap、memory 和后续
QKV/双缓冲/INT8 目标已完成。
