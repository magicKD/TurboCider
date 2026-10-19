# Qwen 2.1：保守 DiT 缓存 × 跨请求参考前缀复用

2026-10-17，M4 Max / 64 GiB。继续优化 Qwen Image 2.1；本轮只改变重复编辑的
前缀处理，不修改权重、参考尺寸、采样步骤或 GPU 算术，也不启用 ANE。
完整优化目标仍未完成。

## 实现

先前 GPU conservative cache 与跨请求 prefix snapshot 互斥。
现在显式设置 `TURBOCIDER_QWEN21_PREFIX_SNAPSHOT=1` 时，可对 resident、
512×512、20–40 步、base GPU conservative 编辑组合两者；保留标准 1024
参考编码，最多三张参考，不接 LoRA、PE、ANE、streaming 或 memory-budget override。
未设置该变量时保留原行为；App / 默认缓存设置未变，已安装 App 未替换。

snapshot 只导出完整、紧凑、独立持有的 conditioning K/V、文本和参考 owner，
不导出 target noise、DB front/middle residual 或 Transformer/compiled closure。
接收者先导入静态 prefix，再配置新的 request-local DB 状态；换种子不会复用旧
target。组合单独加入 snapshot runtime key；条件、顺序、geometry、来源、路由变化，
取消、失败、unload 继续走既有失效/清理逻辑。

`Transformer::export_prefix_snapshot` 的新允许参数默认 false；普通 DBCache 调用
仍拒绝导出，只有已限制的组合路径显式放行。没有把普通导出 guard 全局移除。
借鉴的是 techniques 的 immutable cache ownership / explicit state 原则，
不是把 compile cache 与模型激活缓存混为一谈。

## 真实 512² / 40 步测试

原 BF16 DiT / encoder / VAE，标准 1024 参考处理。已有蓝茶壶参考，指令为改成
红色并保持形状、把手、壶嘴、桌面及背景。实际 4096 reference tokens + 35 text
tokens；两个种子 73、74。仍完成全部 40 个采样步骤、VAE 和 RGBA PNG 导出。
所有对照在同一个串行 session，除首项外 conditioning 均命中；不是 fresh prompt
或严格冷启动排名。

预先固定 9 请求顺序：warmup → snapshot-off control A → populate → hit A →
hit B/new seed → snapshot-off control B → snapshot-off control A repeat →
既有 GPU/cache-off snapshot populate → 既有 GPU/cache-off snapshot hit。
无并行 build、pytest 或图像比较；首项和 populate 不进入命中速度分母。

| 比较 | 对照 / s | 候选 / s | 请求耗时减少 | 候选对同种子对照 RGBA |
| --- | ---: | ---: | ---: | --- |
| conservative，seed 73；前后控制均值 | 41.043 | 36.205 | 11.79% | 逐像素相同 |
| conservative，seed 74；反向匹配控制 | 41.028 | 36.223 | 11.71% | 逐像素相同 |

snapshot populate 为 40.994 秒；不把首次 miss 算作命中收益。两次 seed 73 的
snapshot-off 控制为 40.967 / 41.119 秒，RGBA 完全相同；关闭后 allocator active
也恢复。两个种子的候选图像 hash 不同，不是忽略新种子或返回同一张缓存图。
全部 conservative 请求实际复用 15 steps / 360 middle blocks，snapshot 命中
没有额外跳过 sampler 步骤或改变 DB skip count。

另测现有 **GPU、DiT cache off、snapshot hit** 为 50.008 秒，同 seed conservative
组合为 36.205 秒，即少 27.60%。这包含 conservative 本身的近似收益：RGB SSIM
0.997401、MAE 0.656638/255、alpha max abs 1，并非整体无损。新加的前缀复用在
本批两个种子上没有额外像素误差，不外推所有内容/设备必然逐位等价。
人工查看候选图符合本批红茶壶编辑；该合成参考原有的强纹理不能作为通用摄影或
材质保真资格。

预先记录的局部观察门槛：每对净减少至少 5%，增量 SSIM≥0.98、MAE≤3/255、
alpha abs≤4；CPU 重算全部通过，前后控制恢复通过。它们不代替总体性能/画质资格。

## 内存、负载和限制

命中期间额外持有 2,166,358,016 bytes（约 2.017 GiB）紧凑 K/V。
MLX active 从 14,973,996,274 到 17,140,354,290 bytes；关闭 snapshot 后回到前值。
仍有原来的 min(8 GiB, configured memory ceiling / 8) cap 与 export headroom 检查。
这不是无内存代价的加速，也不是 OS 内存硬上限。

完整 process-tree 100 ms / max gap 500 ms 采样通过，观察峰值 21,362,617,672
bytes，窗口内系统 swap-out 0；不含外部服务 / driver 独立归因。与上一轮包含
生成、ANE 或不同加载生命周期的峰值不能直接作内存优化倍率。

竞争 CPU-load 检查仍失败，保留记录且 `qualification_passed=false`；未停止或修改
用户 ComfyUI 进程，没有物理 GPU 独占或设备时间线资格。每 seed 仅一项命中样本，
同 session 的两个 seed 不是两个独立设备窗口。因此不升级 App 默认或自动选择。

本轮没有重新测量文生图速度，不借此声称其获得新增收益。更保真 ANE、完整多参考
矩阵、独立场景与反序窗口、fresh encoder 和六步 student 路线仍需继续推进。

## 构建和验证

- 库：`build/qwen512-prefix-cache-20261017/libturbocider.dylib`。
  SHA256 `a1b9e369ec54a33031e9ea69a045b246957015c811f40e8c5c6798b95e846b94`。
  Runtime ID `tc-runtime-build-v1-86857ad9f1268b4c6eede95e76f73c40c7ba925639688cc2c1079291a523d01d`。
- 130 项请求合同：129 通过、1 跳过。
- 真实 MLX lifecycle 回归通过：包含已有 24 cycles / 8 variants，新增 40-step DB
  snapshot、两组新 noise 的逐步完整输出和 skip decision 对 fresh prefill oracle
  一致，以及关闭 DB 后恢复；全部 owner 销毁后 active 回到既有容差。
- C++ options parsing / route guards 通过；5 项新 benchmark/hook host tests、
  8 项上一轮 cache-hybrid host tests 通过。没有声称完整 `make test`。
- 原始请求、9 PNG、完整结果、source snapshot、内存与 load：
  `outputs/qwen512-prefix-cache-pilot-20261017/`。当前工具增加了门槛结果打印，
  已采集原始 `summary.json` / `source.json` 保留原版本，不回填旧 script identity。

复现需要新运行库和已有本地权重，未下载新模型：

```sh
.venv/bin/python -B tools/validation/qwen_prefix_cache_screen.py \
  --library build/qwen512-prefix-cache-20261017/libturbocider.dylib \
  --output outputs/new-qwen-prefix-cache --seed 73
```

本轮 owned build / inference 已结束。磁盘约剩 19 GiB，没有清除权重、用户缓存
或前一轮证据。后续优先扩大场景与负载复核，再考虑与 ANE decode 的合法组合。
