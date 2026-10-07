# 原生 channel auto：构造期校准、实际候选试验与缓存隔离

本轮 UTC 2026-10-05，本机 Asia/Singapore 日志已为 2026-10-06。
接续 [channel 成本初筛](ane-channel-cost-calibration-2026-10-05.md)。
完整双后端、四格 base ≥1.2×、真实 LoRA 加速和正式质量/内存/
设备 trace 目标仍 active；本轮接通实际 native 模型，不宣布目标完成。

## 新路径

新增显式 `TURBOCIDER_PRIVATE_ANE_CHANNELS=auto`。Public 默认和现有
数值通道配置不变；auto 必须已有获授权的 Private 后端、Sylvester
`w8a8` recipe 和模型提供的 `CalibrationWorkload`。不得将 -1 sentinel
传给 graph geometry；未提供 workload 的直接构造拒绝。选中宽度用
显式 constructor/factory 参数传递，不临时改全局环境变量，也不覆盖
用户指定的数值宽度。

Z-Image 与 Qwen 的实际构造点提供 model SHA256、resident source
allocation ID/weak owners、原始 FFN weights、完整优化 GPU callback
和物理范围 GPU callback。Z 的原 compiled tuned GPU FFN 提取成
共享函数供旧 runtime fallback 与校准共用，避免换成更慢分母。
Qwen 保留 fused gate_up 完整 GPU callback，channel head 保持原
separate gate/up 算法。校准完成以后才建立 Qwen 的 request-local
LoRA range captures；请求中间不更改 share。

原生校准使用五个深度源、0.4/0.8 附近 512-aligned shares，完整
GPU/ANE/Both one/four 独立测量，两次 warmup、七次 hot、循环换序与
median。GPU arm 包含 W/A staging、独立 frozen restore/join 和所选
prefetch 流量，不读取本次 live ANE 输出。原始实际 rows 与 ANE
bucket 分离：GPU 用实际 rows，ANE 为相同 input 追加零 rows；GPU
join 只消费实际 rows。四个 immutable W/Y banks 用于绑定，第五个
源只供未来 staging；mutable W/A 双槽仍独立。

fit 仍只在已测范围选择，比较独立 full optimized GPU baseline，
预测收益不足 5% 就 GPU-only。proposal 之后实例化**新的真实推理
executor**，经过实际 graph admission/self-test、stage/run_channels/
restore/join，并独立比较四组 FFN output（rel L2≤0.03、cosine≥0.999）。
再做完整 four-FFN GPU/candidate 串行换序窗口：两次 warmup、七次 hot；
actual calls、无失败/fallback/retry以及窗口 median 至少 5%收益均通过
才采用。试验强制固定一段、async head、eager join；它对应稳定 FFN
窗口，不替代 adaptive whole-block 或整请求资格。

auto 的 base executor 明确裁去 unused LoRA inputs/hidden output；
显式旧通道配置与旧 LoRA graph ABI 不变。实际 helper 仍需完整 GPU
重算任何失败 operation，不能发布部分 scratch。初始 W producers
部分提交异常也 drain 全部已提交 tickets，保留首个异常。

## 缓存与边界

八项有界、线程锁保护的 **in-process cache**，只登记独立实际试验
通过的正通道选择。Key 包含 model hash、adapter identity、encoding/
precision、backend/recipe、SoC、OS build、runtime build、Metal 配置/
ABI、graph ABI/bucket、source generation、actual rows、H/F、tiles 和
prefetch。Z 额外绑定 relevant tuned-GPU flags；Qwen 绑定 prefix 长度。
source weak owners 过期就失效，不为缓存 pin 第二份模型，graph 重建
仍重新做实际 memory admission。不导入离线 proposal，也未做跨进程
磁盘 policy cache。选择使用 aggregate hot vectors；本轮 native
generation receipt 尚未导出这些内部 calibration/trial raw vectors。

无收益或能力/内存/试验拒绝是明确 GPU-only，不计为 runtime failure
或失败 scratch fallback。`usable_configuration()` 使 resident 请求
能保留这种有效配置，不每次重跑无收益校准。adapter/rows 变化纳入
模型的 executor identity。

当前 auto 不把 base 成本模型套到真实 LoRA 或多 row-chunk：这些情况
返回 GPU-only。1024² 要用覆盖 complete actual FFN rows 的大 bucket
继续试验，不能拿小 bucket 的单次 ANE slope 当成多个 chunks 的成本。
Comfy `convrot_w8a8` 不在本 auto recipe 内；其已有质量失败仍有效。
现有显式数字通道的 GGUF/Q4、LoRA、Comfy、lookahead/prefetch 路线
未删除，通用 Affine source views 可供后续 auto 验证；本轮没有真实
GGUF auto、Comfy auto 或 adapter-aware auto 资格声明。

## 实机与回归

最终 v3 Private library SHA256：
`b1b378f2f91c32fc986abc39c68472804abfb5f75892c2bed7a164dbc5719232`。
Public library：
`2a6f8f50c72dcdeed4a81e420440b5049c23e68f2a8864b173dce6c0d2741a28`。
都是保留原有 ConvRot drafts 的 working-tree builds，不是 staged-only
完整重建。已单独 syntax-check staged Z cpp；selective staging 对比
原始 dirty changed-lines 完全一致，没有夹带那些草稿。

- 14 项 host（11 shared runtime + 3 Private host）通过；cache test
  覆盖 19 个 identity 分支、quality/gain/completion gates、weak source
  失效与 LRU。新 missing-workload 试验最初误用 runtime_error catch，
  已改为库真正返回的 invalid_argument，不改生产错误语义。
- Public Core ML/MLX/receipt 13 项通过；最终 Public actual flags、private
  class bytes/direct-link 发行隔离检查通过。
- Private suite 20 项：19 pass、1 个 prepared-calibration opt-in skip。
  显式 LoRA base/A/B/base、一次 full-hidden down correction、晚 chunk
  整次重算、取消/GPU/down异常、deferred ownership等实机检查通过。
- 新实机 auto fixture 实际调用五个 sources、45 次 full GPU callbacks、
  180 次 complete GPU head callbacks，完成 prepared W8 one/four 流量；
  该小 fixture 收益不足，正确选择 GPU-only。adapter 和超 bucket rows
  不再消费 base callbacks；不是模型性能证据。

模型条件：original BF16 base，fox/seed42，512²；Z 8步、Qwen40步；
c1056/k1024/n512、chunks1、fixed-async1、scale-cache/launch-fence/
stage-specialize1，prefetch/A8-lookahead/deferred0。每模型一冷一热，
同一最终 CLI/library。全部 owned build/test 与两个模型的 timed
constructor/generation arms 串行，不与它们重叠。

| 模型 | 自动采用 Fa | 实际 Private calls / 请求 | 冷 wall s | 热 wall s |
| --- | ---: | ---: | ---: | ---: |
| Z-Image | 4096 | 256 | 20.132181 | 5.871521 |
| Qwen 2.1 | 5120 | 1280 | 56.768905 | 34.183665 |

两模型都报告实际中间通道路线、accepted complete-runtime FFN
candidate；每次生成无失败/retry/GPU fallback。热请求直接保留
resident executor，不是又做一次 cache lookup。未验证强制重建后的
实际模型 cache hit；该分支目前是 deterministic host 验证。
这些是带连续 host observer 的 **generation diagnostics**：没有同库
GPU 对照、反序、多热样本或严格 competing-load gate，不计算加速比，
不能将历史 GPU 分母除以上表。没有证明物理 GPU/ANE overlap或
native INT8 MAC。native quality validation calls=0仍不作为质量验收。

Z冷/热PNG相同，且与以前相同 Private recipe 的
`6d3f01c35392246eedc75fddc76c8dc6d0048ad9ec42e339fbaa7bdc20aaf7cf`
一致；Qwen冷/热同为
`e3155bcc06afd7220c16ae7459cbc2fb42869566dec0b9df132f49cb5f297e23`。
是固定样本回归一致性，不是 GPU-vs-W8A8 latent/媒体资格。

首次 Z diagnostic 揭露 empty-LoRA 配置的非空序列化 identity，被误判
成 adapter 后直接 GPU-only；修正模型调用点，只有真正存在 adapter
才提供该 identity。另一次 fixture 揭露独立 Program 收到空 cache path；
新增 Private default cache root helper，保持原 absolute-path/permission/
digest guards，最终 fixture 确实执行了全部 sampling callbacks。
初始 raw receipt 保留；首次 Z cold PNG 被重复输出替换，旧 hot PNG
保留为 `runtime-v1-hot.png`，不把它误标成最终 Private 输出。模型、
adapter、参考源码与设计原稿不改。

## 证据与下一项

- `outputs/native-channel-auto-z512-20261005/generation-receipt.json`
  （初始接线失败）：
  `d80e77ebfc653d69794c4ec993b573975dac440df06d91addcd18144d8a54de1`。
- 同目录 `generation-v3-receipt.json`：
  `534737a224c2a57bc467eb918b15306e858bc21815a1a7c51b4933a563d59df2`。
- `outputs/native-channel-auto-q512-20261005/generation-receipt.json`：
  `752125621dfc93ec2418749f8f015b2524566c37aa533768097bf3fb7d83acca`。

继续需要 native raw calibration/trial receipt、actual cache-hit trial、
1024² 大 bucket 和多 chunk、真实 adapter-aware sampling与 share重建、
按 operation/layer 的 Public/Private/GPU recipe 选择，以及最新同库
四格 ≥1.2×、multi-prompt/reverse、latent/感知/语义、process memory/
设备 trace。GGUF bounded decode真实复用与 Comfy敏感算术的 GPU
handoff仍需接续；不以本轮功能接通代替这些完整要求。
