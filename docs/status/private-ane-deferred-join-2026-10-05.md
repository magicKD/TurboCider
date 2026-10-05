# Private ANE：owned deferred channel join 初筛

接续 [1024² LoRA](private-ane-lora1024-2026-10-05.md)。实现提交为
`424da59`。四格 base ≥1.2×、LoRA 加速及完整质量/内存/设备并发目标
仍未完成；本轮不提升默认路由或资格门槛。

## 实现与推荐

新增默认关闭的 `TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN=0|1`，screen
对应 `--defer-channel-join 0|1`。开启必须已有获授权的 Private channel
executor、正固定 chunks、`fixed-async=1` 且关闭 profile；配置进入
executor identity。Public 默认、source recipe、精度与模型/adapter 均不变。

仅固定异步 channel 的成功路径可以不执行最终 `mx::eval(merged)`。
ANE/restore 已完成，output/hidden 分别由 MLX allocation 独立持有；
join 及一次 full-hidden down-LoRA 留给正常模型 consumer。下一任务复用
ANE y surface或销毁 executor 不得影响已返回但尚未消费的结果。
Measured、失败及异常 cleanup 仍保留 fence，失败 chunk 仍完整 GPU
重算，不发布部分 scratch。之后正常消费 lazy graph 产生的 GPU 错误
仍属于整请求失败，不能凭一次 FFN 返回或计数宣称整请求成功。

回执新增 enabled、actual deferred block session counter 与
`post_join_scope`，区分 evaluated / deferred / mixed host spans。
Deferred 的 post-join/FFN host wall 不包含之后 GPU 消费时间；性能判断
只使用完整 native `request_wall`（含 VAE/PNG，排除冷请求）。这些
host spans 不是独立 ANE kernel timer 或物理设备 overlap 证据。

**推荐保持默认关闭。** 同库 ON/OFF 热中位差仅 0.004899084s（约0.084%），
且两次 route 顺序不同，没有完成 ON/OFF 自身的反序重复；不能视为
统计显著收益，也不将跨库差异全部归因于 deferred join。

## 同库整请求结果

M4 Max / 64GB；Z-Image Turbo BF16 base，512²、8步、fox/seed42；
c1056/v1、Fa4096、W8A8 GPU I/O、chunks1、scale-cache1、launch-fence1、
stage-specialize1、fixed-async1，W-prefetch0、A8-lookahead0。每路线一冷
五热，全部声明 hot samples 保留；各 case 的 GPU 分母来自同一 binary。
连续 CPU load verifier 均通过，但不证明 GPU/ANE 独占。

| 版本 / 开关 / 顺序 | GPU median (s) | Private median (s) | 中位倍率 | min(GPU)/max(Private) |
| --- | ---: | ---: | ---: | ---: |
| 前库 / 无新开关 / Private→GPU | 6.999797 | 5.828962 | 1.200865× | 1.192981× |
| 新库 / ON / GPU→Private | 7.000034 | 5.815350 | 1.203717× | 1.200232× |
| 新库 / OFF / Private→GPU | 6.999319 | 5.820249 | 1.202581× | 1.194608× |

新库 ON 实际累计1536个 deferred blocks；OFF为0，两者累计1536个
channel blocks，每请求256次 Private calls，均无 retry/fallback。
ON/OFF 首个热 PNG 的 SHA-256 同为
`6d3f01c35392246eedc75fddc76c8dc6d0048ad9ec42e339fbaa7bdc20aaf7cf`。
这只是同 Private recipe 的固定样本逐位一致，不是 GPU vs W8A8 的
latent/感知/语义资格，也不能把 ON 的窄裕量推广到多提示词。

原始 case/hot samples/summary与load哈希见
[机器记录](../design/validation/private-ane-deferred-join-pilot-20261005.json)。
原始目录仍在 ignored `outputs/`，不提交生成图、模型或 build artifacts。

## 验证与剩余工作

- Private 隔离库 `build/private-ane-deferred-join`：
  `e5e0527165fb3e3d4b8b58c0f4f5ba8e6ed9b81c6f7b69630a70505f24458363`。
- Public 隔离库 `build/public-ane-deferred-join`：
  `527e5a115f2c1e371f682476eac5a7a25b6d768599ce35389eca38ac5a61191c`。
- 本轮13 Private host/hardware/MLX、13 Public Core ML/MLX/receipt、60
  screen host通过；新增 mixed/zero/all scope receipt 又单独实机通过。
  Public 实际 flags/private class strings/direct links 发行检查通过。
- Native fixture 覆盖 eager/deferred 的取消、GPU/down回调异常、late
  LoRA chunk整次重算及fallback自身异常；前3类清理后可复用。失败
  不累计成功/deferred counter。另测 BF16/FP16/FP32 base及人工
  full-hidden correction 在 executor销毁之后消费，以及先复用y再消费。
- 验证库含独立 ConvRot working-tree修改，未从 clean staged tree重建；
  本提交只包含本专项hunks，ConvRot修改仍留在工作区。
- 未执行完整`make test`，不声称全suite绿色；既有Qwen3 stale source
  字符串断言未在此修改。没有观察或声称 native INT8 MAC placement。

本轮测量窗口结束后，重新核验原 downloader 的 birth、executable、
kernel argv与 stopped state，仅向原PID发送SIGCONT，观察到RN。
`outputs/private-ane-deferred-join-download-pause-ledger.json` 已记录resumed；
没有更改下载文件，不遗留暂停状态。

当前 channel auto controller仍只在固定share上选择开/关，不是设计
§59–68 的 bandwidth-aware multi-share/prefetch calibration。下一项需
实际采样 GPU part alone / ANE part alone / concurrent、多个模型深度、
1-vs-4 block固定成本扣除及memory约束，选择near-optimal最小share，
并按model/geometry/encoding/backend/SoC/OS/build/kernel与graph ABI
隔离cache。不能用 async host wait替代ANE-alone，或只写纯成本模型
就宣称完成。最新库四格反序/多提示词、latent/媒体/内存/device trace
及尚未正收益的 Qwen512 LoRA 仍需继续。
