# Qwen512 编辑首步通道分工：单图比例未定，双图4096有正反序信号

2026-10-10，Asia/Singapore，M4 Max 64GB / macOS 26.6.2。
接续[首步权重缓存筛选](local512-qwen-prefill-weight-cache-2026-10-10.md)。
本轮完成单、双参考图的通道比例筛选及共有三臂反序确认：48次实际请求、
16份独立进程树内存证据。全部是竞争负载下的诊断，不升级默认路线。

## 相同工作负载，只改变首步 FFN 的通道比例

同一 generation-phases-v1 Private CLI/库、本地原 BF16 Qwen Image2.1、
Viggle v0.2.1 r256/strength1、512²、6步、seed29。每臂独立进程、一冷
两 fresh warm；三个提示词只改变 sunlight 修饰，conditioning 全 miss。
所有臂使用 GPU encoder、相同原权重保留、joint BF16 A/B 与 F32 ranks；
原 adapter 真实应用227个 bindings，没有合并或改写原模型/参考图。

首步全部32层 FFN 按 intermediate channels 拆给 GPU/Private ANE W8A8，
ANE4096/5120/6144/7168分别搭配 GPU8192/7168/6144/5120，F32 partial join。
并非整个首步交给 ANE，也不是 attention 和 FFN 的依赖链被强行并行。
后五个 KV-hit 步骤都走原完整 GPU，实际 ANE calls/blocks 为0。
W-code cache、prefetch、A8 lookahead、层筛选和时间复用均关闭。

单图首步2096 rows，原 c2112/eager join；双图3144 rows，原 c3168/deferred
join。第二张原图为1024² blue RGBA PNG，native reference_size512；未重写
原图，两张有效 reference tokens 各1024。GPU control 保留完整 block 路径。

工具新增 `--prefill-share-screen --joint-ab`，限定原 LoRA 编辑、baseline5120；
禁止与 generation/frozen/layer/cache 混用。每臂的原生 env、receipt validator
及 `requested_ane_channels` 都使用实际比例，不靠改标签冒充不同执行。
支持含 GPU 的显式 subset 和匹配 `--order`；双图对所有 hybrid 臂统一 deferred。

## 同任务正反序结果

表内为两次 fresh warm 中位数，单位秒。括号内是同臂首步时间；后续五步
约6.09–6.11s（单图）、6.34–6.37s（双图），未发现新的后续阶段赢家。

| workload / version | GPU | ANE4096 | ANE5120 | ANE6144 | ANE7168 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单ref / v1 | 10.512879 (2.436003) | 10.258946 (2.254574) | 10.310326 (2.259002) | 10.359085 (2.352287) | 10.579328 (2.515393) |
| 单ref / v2 | 10.586863 (2.429480) | 10.299745 (2.300982) | 10.256970 (2.236887) | — | — |
| 双ref / v1 | 12.695630 (3.744798) | 12.141717 (3.253152) | 12.229772 (3.346569) | 12.461035 (3.637639) | 12.674230 (3.946719) |
| 双ref / v2 | 12.740456 (3.744954) | 12.174593 (3.250799) | 12.224656 (3.356029) | — | — |

单图 v1 顺序 GPU→4096→5120→6144→7168；v2 为5120→4096→GPU。
双图 v1 顺序7168→6144→5120→4096→GPU；v2 为GPU→4096→5120。
各 workload 的 GPU/4096/5120 共有三臂确实反序；6144/7168仅有首轮，
不声称它们获得反序资格，更不把单图和双图互当 ABBA。

单图4096对5120整请求先快约51ms、后慢约43ms；首步先快约4ms、后慢
约64ms。单图保留原5120候选，不追逐一个窗口的最小值。5120相对各自
GPU 首步少7.27–7.93%，整请求少1.93–3.12%，是整条已存在的并行路线
收益，不能全算给本轮 share 工具。

双图4096对5120整请求分别少88/50ms（0.72%/0.41%），首步少93/105ms；
两窗口方向一致。对各自 GPU，首步少13.13–13.20%，整请求少4.36–4.44%。
因此4096是双图值得继续确认的显式候选，但还不是自动选择器或正式默认。
双图4096冷请求21.318/20.886s，5120为17.165/17.198s，GPU为17.340/17.366s；
热请求优势不能说成冷启动优势，也不把冷差额唯一归因于 channel size。
6144/7168的首步均输给小比例；增加 ANE 工作占比不等于必然更快。

## 实际执行、内存与图像

全部 hybrid 每请求实际32次 Private prediction /32个成功 channel blocks；
后五步0，failure/fallback/overflow retry0，headroom1。原 shared gate/up
ranks、joint A/B 和 crypto source proof 独立重放通过。cold 读取原 adapter
完整内容，warm 原 source-proof 复用；没有省掉任一路独立 LoRA A。

四个 enclosing continuous CPU-load checks 全部失败，qualification始终false。
软件 async/phase receipts 不证明物理 GPU/ANE overlap、设备独占或全场景稳定
倍率。所有冷/热样本和 PNG 保留，未拼各臂最佳 phase 成未执行过的请求。

16份100ms进程树证据的 hash-chain、原 report、binary/source/模型 generation
及 PNG SHA 全部核对。最大 tree phys-footprint 40,722,409,896 bytes；单图v1
GPU窗口有 system swap-in65,536 bytes，其余0，全部 swap-out0。存在系统
compression/decompression，不冒称无内存压力或把系统计数归因于 ANE。
scope 为 process-tree load+cold+warm+exit，不含 external driver/service 归因。

此前v1有限观察之外，本轮查看v2单/双图 warm case1 的 GPU/4096/5120整图。
壶形、壶嘴/把手/盖钮、米色/蓝色、左右位置、木桌和暖光阴影很接近；
有少量纹理/高光差异，未见明显新增色块、棋盘格或断裂。不是多seed/scene
质量验收，也不要求严格 latent 逐位一致。

## 整理与验证

新增 share contracts，整理两份新测试的可读性、subTest 和有界 build timeout；
拒绝 frozen/fused-B/不匹配 order 等组合。最终44项 host tests通过、无skip：

```sh
.venv/bin/python -m unittest -v \
  tests.native.test_qwen_prefill_share_screen \
  tests.native.test_qwen_prefill_weight_cache \
  tests.native.test_qwen_ffn_phase_screen \
  tests.native.test_qwen_encoder_residency_screen \
  tests.native.test_qwen_frozen_base_screen \
  tests.native.test_qwen_prefill_busy_diagnostic
```

另有[paired-rank 独立研究与实Metal回归](local512-qwen-paired-ranks-2026-10-10.md)，
但其大行数在线 delta 未稳定盈利，不接入模型。生产 native 源码/库本轮未
重编译，不改 guard/完整GPU fallback，也不默认启用任何新 share/cache。
所有 owned 模型与 probe jobs 已结束；temporary test build自动清理，保留的
小 probe build 无 `.o`/module-cache。原用户草稿、模型/adapter/ref、已有库、
日志/失败证据/PNG均保留，没有清用户/driver缓存或向外部进程发送信号。

[机器记录](../design/validation/local512-qwen-prefill-shares-20261010.json)
绑定四个窗口的完整冷/热/phase、source、PNG、memory与资格边界。
