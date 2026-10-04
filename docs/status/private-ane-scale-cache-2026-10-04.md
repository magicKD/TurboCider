# Private W8 compact scale cache 接续

接续 [prefetch](private-ane-prefetch-2026-10-03.md)。记录目录使用用户提供的
Asia/Singapore local-date 标签2026-10-04；原任务仍 active，四格≥1.2×、
完整 LoRA/latent/media、带宽/内存/物理 overlap qualification 尚未完成。

## 实现与安全边界

- Shared weight/matrix view 增加弱 allocation-generation identity。
  Model adapter 从已 eval 的 MLX `Data` shared allocation 提取稳定身份；
  checkpoint-only `DeviceWeightView` 显式承诺 immutable generation。
  Metal raw-address-only key、mutable source 和 A8 activation 不可 cache。
- Device cache key 覆盖 generation、buffer/extent/offset/full physical pitch、
  encoding/dtype/group、metadata generation/layout、logical slice、H block/seed。
  新 generation、过期 allocation、slice/recipe 变化 miss；source与metadata
  strong owner 从 cache key 移除，不保持旧模型/adapter checkpoint生命期。
- 仅保存紧凑 FP16 row scales，GPU copy 恢复当前 bank 的 physical row pitch。
  miss做原 decode/Hadamard/scale + codes；hit只 copy scale + decode/rotate/codes。
  不保存全模型W8、dense副本，不修改base/LoRA，recipe和量化边界不变。
- 成功GPU completion且validation flags=0后才能发布cache entry。
  hit仍检查codes pass的nonfinite/rotation/scale错误，不绕过失败检测。
  Active ticket持有cache entry，LRU eviction不释放仍在用的GPU storage。
- Mutex保护cache，entry有效位atomic。LRU≤128 entries、总metadata≤4MiB；
  W8 graph memory estimate预留4MiB，缓存不会无限扩大。
- `TURBOCIDER_PRIVATE_ANE_SCALE_CACHE=0|1`/screen `--private-scale-cache` 做
  同库消融；默认W8启用，Public默认与private API隔离不变。JSON输出
  enabled/hits/misses/entries/bytes/evictions，screen验证完整、有界metadata。

## 验证

完整private/public build成功，13 private host/hardware/MLX、13 public
Core ML/MLX/receipt、7 host、49 screen测试通过；Public class/link隔离通过。
Mutable input和A8不误命中；9 source/dtype、H128/H512的cached与fresh
FP16 scales/I8 codes逐位一致；recipe变更、弱generation/address reuse、
不延长source寿命、140-key LRU churn/cap、cache-hit nonfinite拒绝通过。
已有base/A/B/base、完整hidden单次down-LoRA、late-chunk整段GPU重算仍通过。
不是full make test、整体模型画质或原目标完成证明。

Private pilot库：`3a8369a65a303793a882b1e21f6408bdeafda5252f959f5ec9eebc637fdede29`。
Public regression：`0a8c2279af71d7c8598e6e44c976ee4f1310bd40b46b1896d1abf8ef8ab0171e`。

standalone BF16 full FFN weights gate/up/down，先排除一次first-use，8热样本：

| 几何 | cache off | cache on | staging倍率（不是整请求） |
| --- | ---: | ---: | ---: |
| Z H3840/F10240 | 6.931 ms | 4.208 ms | 1.647× |
| Qwen H4096/F12288 | 8.818 ms | 5.280 ms | 1.670× |

每组cache on有3 misses/24 hits，metadata分别48640/57344 bytes。

## 同库Z512整请求：正收益，仍不足1.2×

fox/seed42、8步、Fa3072/c1056/chunks=1、prefetch=0、GPU→Private、
一冷两热。request wall含VAE/PNG、不含冷请求，关闭profile。

| 模式 | GPU hot median | Private hot median | 对GPU倍率 |
| --- | ---: | ---: | ---: |
| scale cache off | 6.986317 s | 6.649499 s | 1.05065× |
| scale cache on | 6.987864 s | 6.467730 s | 1.08042× |

完整回执通过、无错误回退、所有channel blocks实际执行；cache on累计
96 misses/2208 hits、96 entries/638976 bytes、0 evictions。不是把GPU decline
当ANE加速。目录 `outputs/private-ane-scale-cache-z512-{off,on}-20261004/`。
单向两热pilot不是正式campaign，不外推1024²、Qwen或LoRA，不借旧库分母。

继续：更紧的GPU→ANE submission ordering、多chunk activation staging、
带宽校准与全模型share/bucket搜索、两模型分辨率/LoRA/媒体/latent资格、
四格matched正反序≥1.2×。Source generation contract仅针对immutable resident
allocations，不声称完整streaming source-lease/低内存/driver认证已完成。

未改发行build/native、用户模型、参考仓库/设计稿，未stage/commit。

## 首次提交 ordering 接续

Private W8新增显式 `TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE=1`，也进入
executor identity/receipt。Host仅等首个ANE request与leading ready producer
提交，**不等ANE结果**；输入/alias/late-chunk失败路径通知finished，避免
launch前置等待死锁，保持原finish/整段GPU fallback。13 private（含显式
fence错误/换权/lease）、13 public、7 host、49 screen回归仍通过。

同新库 Z512/Fa3072/c1056/cache=1/prefetch=0、GPU→Private/两热：
off GPU=6.986740 s、Private=6.474668 s；on GPU=6.986114 s、Private=
6.423169 s。首轮有小幅改善，仍未达1.2×；不据先提交就宣称硬件overlap。
目录 `outputs/private-ane-launch-z512-{off,on}-20261004/`。
Private库 `cb720a15d594bd06161cf1e46bb45e1d53dd15d77160cf4460b3364f923b7692`，
Public `78a98619d95e108599bc301e7091e002c2560fe1463e23a95a49cd61346e1673`。
下一步需两模型完整分辨率/LoRA测试、bandwidth-aware calibration与实际trace，
不是新的默认资格或目标完成。
