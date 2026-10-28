# Qwen 2.1：冻结 GPU/ANE × 保守缓存 × 前缀复用

2026-10-17，M4 Max / 64 GiB。继续推进 Qwen 2.1 的 512² 文生图 / 图生图；
本页记录 GPU/ANE 编辑组合和独立文生图档位诊断。完整优化目标没有完成，未修改已安装 App
或默认配置，也未下载新权重。

## 实现范围与关键不变量

沿用 [GPU 前缀组合](qwen512-prefix-cache-2026-10-17.md)，将同一显式
`TURBOCIDER_QWEN21_PREFIX_SNAPSHOT=1` 开关扩展到 resident / base / 512² /
40 步 / conservative / 原 BF16 组件、6144-channel source-bound frozen W8A8。
保留标准 1024 参考编码；全尺寸 ANE 编辑仍需原有 full-reference diagnostic。
不开放其他 share、步数、runtime-weight、LoRA、encoder ANE、streaming、低内存
或改变参考编码尺寸的组合。

- 首步始终完整 GPU，之后才安装 ANE decode FFN callback；snapshot hit 也不例外。
- 对完整采样步骤和实际 ANE prediction 覆盖核对后，清除当前 Transformer 的 decode
  callback，再导出只含静态 conditioning K/V 的 snapshot。
- Transformer 的“有 live callback 不可导出”guard 没有放宽。snapshot 不包含旧
  target noise、DB residual、ANE 输出、回调或 compiled closure。
- 新请求仍重新配置自己的 DB 状态；source / manifest / LoRA identity、geometry、
  instruction 和有序参考继续参与失效或 runtime key。默认和未显式开启的行为不变。

## 实际 9 请求编辑验证

既有蓝茶壶参考，改成红色并保持形状、把手、壶嘴、桌面和背景。
标准 1024 参考，输出 512×512；原 BF16 DiT / encoder / VAE 与既有 frozen W8A8。
seed 73 / 74，同一个串行 session：warmup → off-control A → populate → hit A →
hit B/new seed → off-control B → off-control A repeat → 纯 GPU/cache-off snapshot
populate → 纯 GPU/cache-off snapshot hit。

除首项外 conditioning 均命中，所有请求完成 40 个 sampler 步骤、VAE 和 PNG。
没有并行 build / pytest / 图片比较，首项和 populate 不进入命中速度分母。

| conservative + frozen 编辑 | snapshot off / s | snapshot hit / s | 耗时减少 | 对同 seed 控制 RGBA |
| --- | ---: | ---: | ---: | --- |
| seed 73，前后控制均值 | 33.171 | 29.266 | 11.77% | 逐像素一致 |
| seed 74，反向匹配控制 | 33.259 | 28.418 | 14.56% | 逐像素一致 |

populate 为 33.114 秒，不把它算作命中收益。两个 seed 73 的 off 控制也完全相同；
两个 seed 的输出 hash 不同。每个 frozen 请求都复用 15 steps / 360 middle blocks，
恰好 888 次 prediction，共 6216 次；没有因为 snapshot hit 在首步增加 32 次 ANE，
没有用 fallback 或漏步骤代替完整请求。source SHA 已核验，runtime failure=0。
末尾两个请求明确转回纯 GPU，结果不带 hybrid 指标。

**不等于整体无损。** 同窗口既有纯 GPU、cache off、snapshot hit 为 49.850 秒，
seed 73 组合为 29.266 秒（少 41.29%），但对该完整 GPU 的 RGB SSIM 为 0.919811、
MAE 5.897755/255、alpha max abs 7。W8A8 / temporal cache 原有的累计差异仍存在。
本次新加的前缀复用仅在本批两 seed 上未增加任何像素误差，不能外推为所有内容。

## 内存和资格限制

增加一个 2,166,358,016-byte（约 2.017 GiB）紧凑 K/V bank。
MLX active：off 19,814,223,090 → hit 21,980,581,106 bytes；关闭 snapshot 后
回到前值。转回纯 GPU 后因保留普通 GPU prefix bank，active 为 17,140,354,290
bytes；不把不同执行器生命周期的差值称作本次内存优化。

完整 process-tree 100 ms / max gap 500 ms 采样通过；观察峰值 26,735,898,264
bytes，系统 swap-out 0。它不含外部服务 / driver 的独立归因，不能同 MLX 统计相加。
竞争 CPU-load 检查仍失败，未停止用户 ComfyUI 进程；没有物理 GPU/ANE 独占或并发
trace。因此 `qualification_passed=false`，不升默认或自动选择。

每 seed 仅一项 hit，属于同 session 的局部观察，不是独立窗口置信区间。
预先保存的局部门槛（净少 5%、增量 SSIM≥0.98、MAE≤3、alpha≤4）及 off 恢复均通过；
这不授予对完整 GPU 的整体画质资格。人工查看本批红茶壶符合测试指令，但仍是单一
合成参考，更多场景、多参考、冷启动与反序窗口需继续。

## 构建、测试和原件

库：`build/qwen512-frozen-prefix-cache-20261017/libturbocider.dylib`。
SHA256：`ffb1d4336e404d740dff8a8616f7a0a1b571efe11b2bc756a718247a4e137eab`。
Runtime ID：`tc-runtime-build-v1-5fa1dfa5d868d0f08de29f26d02ffa0ccba9051b683555b6cfcefac7472943d7`。
构建含 private 代码，但本轮实际是 public frozen Core ML，不能称作 private 收益；
`cpuAndNeuralEngine` 也不是所有算子物理驻留的证明。

- 130 项请求合同：129 通过、1 跳过。
- real-MLX lifecycle 通过：源请求在首步后装入不同的目标 decode callback；有回调时
  即使显式 DB composition 仍拒绝导出，清除后导出；两个新 noise 的恢复请求逐步
  对 fresh GPU oracle 数值 / skip decisions 一致。另有既有 owner/销毁/关闭回归。
- C++ route/option 回归和 6 项 benchmark/hook host tests 通过。
- 完整请求、9 PNG、source snapshot、counter、memory/load、质量与门槛：
  `outputs/qwen512-frozen-prefix-cache-pilot-20261017/`。
- 未执行完整 `make test`；未覆盖任意 LoRA / runtime-weight / 其他设备。

```sh
.venv/bin/python -B tools/validation/qwen_prefix_cache_screen.py \
  --library build/qwen512-frozen-prefix-cache-20261017/libturbocider.dylib \
  --output outputs/new-qwen-frozen-prefix-cache --seed 73 --route frozen
```

## 独立文生图：更激进 residual-cache 档位

使用同一个新库，原 BF16 来源、512²、40 步、seed 73、蓝色陶瓷茶壶提示词，
无参考 / LoRA / PE，固定 source-bound 6144-channel frozen 图。
每条路线独立进程，各 off warmup → off → conservative → balanced → fast → off，
共 12 张完整图。首项排除，所有计分请求命中 conditioning；没有 prefix snapshot。
每候选仅一个样本，off 两个 bracketing control 取均值，非独立窗口统计资格。

| DiT cache | GPU 完整请求 / s | GPU 对 full-GPU SSIM | Frozen 完整请求 / s | Frozen 对 full-GPU SSIM |
| --- | ---: | ---: | ---: | ---: |
| off | 42.673 | 1 | 30.419 | 0.89872 |
| conservative | 31.058 | 0.98801 | 22.246 | 0.89262 |
| balanced | 26.387 | 0.96652 | 19.511 | 0.88722 |
| fast | 23.148 | 0.93209 | 17.493 | 0.87011 |

Frozen fast 比同组 conservative 少约 21.37%，对 full GPU off 名义少 59.01%。
不是 kernel 同工作量提速：保留 40 个 Euler/sampler 更新，但 conservative / balanced /
fast 分别复用 15 / 21 / 25 steps，即少算 360 / 504 / 600 middle blocks，
实际每请求 ANE prediction 为 888 / 744 / 648，而 off 为 1248；全 session 6024。
所有 source verification / counter / runtime failure 检查通过。

**快速不代表画质资格。** Frozen fast 相对它自己的 already-quantized off control
SSIM 为 0.94735，而对 full GPU 为 0.87011；后者 MAE 7.772994/255、alpha max abs 2。
人工查看仍符合蓝茶壶主体，但壶把、壶嘴、釉面高光和局部形状改变。不能只报前一个
增量分数、用近似许可代替验证，或称作普遍更优的配置。GPU fast 也有可见
近似，MAE 4.371490/255，不是无损控制。

两路线前后 off PNG 的 RGBA 各自完全一致。窗口 memory sampling complete：
frozen process-tree peak 24,499,007,080 bytes，GPU 19,033,845,040 bytes，
系统 swap-out 均 0。两条 continuous load 检查仍失败，资格保持 false；
不将它们与不同任务/加载生命周期的上一节峰值直接作内存排名。

balanced / fast 的 frozen 数据来自**已有 diagnostic 环境入口**，没有放宽
产品 named-preset admission：公开请求的这两个档位仍 GPU-only，App / 默认不变。
这部分是配置探索，不伪装成已新增一个 native kernel；其误差或倍率不借给前缀复用。

原始请求、12 PNG、source、memory/load 和自身控制比较：
`outputs/qwen512-generation-cache-sweep-20261017/`。
`joint-quality.json` 单列对 full GPU 的累计误差。补齐了 joint analyzer 对
generation-only case 子集的处理及对应 host 回归；未修改原始采样记录/源码身份。

```sh
.venv/bin/python -B tools/validation/qwen_cache_hybrid_screen.py \
  --library build/qwen512-frozen-prefix-cache-20261017/libturbocider.dylib \
  --output outputs/new-qwen-generation-cache-sweep --routes frozen,gpu \
  --modes conservative,balanced,fast --cases generate --seed 73
```

本轮 build / inference 已退出。后续继续做反序、多场景与更保真 ANE/精度路线，
并独立评估六步 student / runtime-weight；不把本轮单个提示词当完整目标完成。
磁盘约剩 19 GiB，未删除旧证据。
