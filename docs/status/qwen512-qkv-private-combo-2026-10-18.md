# Qwen512 QKV + static Private FFN：组合实测及两种等待边界

2026-10-18，Asia/Singapore，M4 Max / 64GiB。仅 Qwen Image 2.1 文生图、图生图；
不涉及 Z Image，不部署 App、不改默认值、不停止用户 ComfyUI。持续优化目标尚未完成。

## 当前决策

本页前半记录最初八窗口；后面的“补充：提交顺序和打包补块”记录其后四窗口。
新增的 GPU-first 和 packed-suffix 两个 primary 均未通过局部性能门槛，不接入默认。

QKV 融合和静态 Private FFN 必须在同一个完整真实 block 内联合计时，不能把历史
收益相乘。最初两次编辑窗口的组合比普通 GPU 少约 4.58–4.97%，但生成未稳定超过
Private-only；后续编辑窗口也出现组合增量反转。因此没有全模型采用依据。

另排除了一个明显亏损的调度：在传输已提交后调用 `Device::wait(ready)`，生成/编辑
完整组合 block 比不等待的组合分别慢约 24.75% / 23.53%。源码确认该 API 在同一
传输队列尾部新建等待命令，排在已经提交的 ANE-dependent restore 后面；它不是
孤立的 input-ready 观察。不能把约 9ms 的该跨度归为 upload 或单纯 ready 延迟。

研究探针改用 `MTLSharedEvent::waitUntilSignaledValue:timeoutMS:` 直接等待借用的
实际事件，不提交新命令、不写 event 值。它消除了上述大幅回退；单对生成/编辑
窗口分别比原 GPU 少约 3.95% / 4.41%，但相对不等待组合的增量仅 0.47% / 2.71%，
Private-only 的直接等待还略慢。保留研究候选，不宣称稳定增益，不接入产品。

八个性能窗口全部检测到竞争性推理 CPU 负载，`qualification_passed=false`。
计数、数值和来源复核通过不等于独占设备、ANE placement 或完整图片质量合格。

## 数学、消费者与所有权

使用原 BF16 Comfy encoder / DiT / VAE、Viggle v0.2.1 六步 r128，strength1、
inference-time，joint BF16 A/B、F32 A ranks。固定 adapter SHA256：
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。

请求 512² / 6steps / seed73，生成蓝茶壶、单参考改红。observer 在首个完整 DiT
step 后，借用 step1/layer0 的实际参数、GPU callable 与参考输出；decode M1024、
H4096、F12288，生成/编辑 prefix 为 24 / 1059。计时包括在线 QKV A/B、norm/RoPE、
attention/prefix concat、output projection、整个 FFN 和完整 down-LoRA、residual。
实际绑定的 227 是全局 logical adapter term 数，不称为单 block 执行 227 次。

静态 Private 模型只保存 layer0 的 2048 个 FFN intermediate channels 的原 base，
BF16 转 FP16 constants，没有 LoRA merge；其余 10240 channels 由 GPU 计算。
gate/up 的在线修正在 SiLU 前加入，完整修正后的 hidden 拼接后，只执行一次完整
down-LoRA。`shared_ranks=false` 在计时前固定。

每个 Private 调用前在计时外用 NaN poison 可复用输出。发布仍要求 ticket
`ok && physical_complete && !retired`、transfer 成功、validation flags 为零。
`physical_complete` 在此是实际 driver callback 证据，不是硬件 ANE placement 证据。
每进程 A→扰动 B→A 预检确认旧 A 不被 B 改写、B 确实改变、恢复 A 后逐位复现。
最后两窗口的三次预检也实际走 direct-event 路径，不复用旧预检结论。

所有预热和计时调用检查 4,194,304 元素完整 block 输出：relative L2≤.001、
cosine≥.999；原 GPU 与普通 GPU staged 控制要求 exact。Private FFN 同时对
matched-input 参考和原始 chain 参考检查 relative L2≤.03、cosine≥.999。
保存的组合 block relative L2 约 .000637 / .000676，FFN 约 .0028，并非逐位等价。
原输入、借用输出、base 和全部 LoRA factors 在前后检查不变。

observer 受控停止于 step1/layer0，**不运行完整六步、VAE decode 或 PNG 导出**。
这里的质量门槛是局部数值门槛，不是最终图片质量门槛。

## 匹配窗口的完整 block 中位数

单位 ms。每行独立匹配原 GPU 分母，不跨窗口拼接最佳 phase。

| 窗口 | 原 GPU | Private | QKV+Private | +queue-tail wait | +direct-event wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| 生成 | 35.7278 | 34.3283 | 34.7122 | — | — |
| 编辑 | 37.1134 | 36.4414 | 35.2680 | — | — |
| 生成重复 | 35.7436 | 34.5948 | 34.9525 | — | — |
| 编辑重复 | 37.1113 | 35.7181 | 35.4119 | — | — |
| 生成 queue-tail | 35.7400 | 34.2549 | 34.0684 | 42.4990 | — |
| 编辑 queue-tail | 37.1416 | 35.7456 | 35.3432 | 43.6600 | — |
| 生成 direct-event | 35.7525 | 34.3831 | 34.5012 | — | 34.3396 |
| 编辑 direct-event | 37.0965 | 35.8608 | 36.4517 | — | 35.4621 |

前四窗口五臂，每臂 3 warm + 15 measured，primary 为 `private-qkv`。
queue-tail 两窗口七臂，每臂 3 warm + 14 measured，primary 事前固定为
`private-qkv-fenced`，局部门槛失败。direct-event 两窗口同样七臂、3+14，primary
为 `private-qkv-event`，通过事前 3% 局部门槛；不因此消除负载和全模型资格限制。

八窗口合计 836 个完整 warm/measured block，其中 692 measured，不是 836 个
独立 prompt/seed。前四各 39 次实际成功 Private 调用，后四各 71 次，合计 440 次，
包含每进程三次预检；不是仅统计 requested/enqueue。queue-tail 各 34 次等待；
direct-event 各 37 次等待（3 preflight + 34 warm/measured）。最后两个窗口的全部
37 次直接等待返回后，Private 完成回调尚未被观察到。它区别于强制排队到 restore
后再完成，但不证明实际物理 ANE/GPU 时间交集。

## Phase 解读：等待变短不等于加速

以带 QKV 的生成臂为例，host span 中位数如下，单位 ms：

| 版本/臂 | producer | corrections ready | submission | 其中 ready wait | exposed wait | post join |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| queue-tail 窗口，不等待 | 12.4004 | .9364 | .2430 | 0 | 19.2099 | 1.2466 |
| queue-tail 窗口，等待 | 12.3380 | 1.0597 | 9.5322 | 9.2176 | .0006 | 19.1801 |
| direct-event 窗口，不等待 | 12.3632 | .9075 | .2435 | 0 | 19.5515 | 1.2479 |
| direct-event 窗口，等待 | 12.4072 | 1.0706 | 1.1642 | .8456 | 11.9102 | 7.3585 |

submission 包含 ready wait，exposed wait 与 GPU 工作重叠；这些中位数不相加充当
物理 compute。queue-tail 的 exposed wait 近零只是等待/工作挪位，wall 反而显著
增加。编辑 direct-event 的 ready wait 约 .7166ms，post-join 约 7.3732ms；同样
不能仅凭 exposed wait 变小声称加速。没有 DRAM/register/device trace，不指定
唯一的硬件资源竞争原因。

相关实现：`native/backends/ane_gpu.mm` 的 `Transfer::submit`、`Device::wait`；
SDK `Metal.framework/Headers/MTLEvent.h` 的直接 event-value 等待合同。后一方法
仍等待真实 GPU signal，所有原发布和 poison/ownership 检查保持有效。

## 来源、成本与复核

研究工具链接的产品库未修改、未重建：

```text
build/qwen512-tokenizer-reuse-v2-20261018/libturbocider.dylib
SHA256 a225f17e0fcbe04f7fa2b2a6b4c8178f2f663cb4bb20e43989022fc531c61e70
runtime tc-runtime-build-v1-2d36a6f10c424a5b670fa1317c73a00e74b4df1aeb7e587892db9d32fdce9e44
```

上述 direct-event 窗口的 binary（当前工具已扩展到 packed-suffix recipe）：

```text
build/qwen21-qkv-private-combo-event-v4-20261018/qwen21-qkv-private-combo-probe
SHA256 3ac7bd2466600b91fe4e99b7387c340052aff492ff5595b6dd467372fde46f4d
```

旧五臂 binary `...combo-v1-20261018/...`，SHA256
`35e42ce00f7f924963bb170d7315cb51553b070dc98aec12542d701107bcb402`；
queue-tail binary `...combo-fence-v3-20261018/...`，SHA256
`a955fffe689412e67129434f230363a792a5fc0ca655cd634d932c99080ac898`。
当前 wrapper 要求 v4 source/descriptor；旧 binary 只对应各自归档 source snapshot。

每窗口完整 memory 采样均 complete，进程树峰值约 17.68–18.33 GB（十进制），
swap-in/out 均 0，少量 decompression 保留。它包括模型 load、全部臂共存、输出
校验和退出，不是每臂峰值，也不包括外部 ComfyUI 或驱动归属证明。

单层 static setup 约 .949–1.021s，QKV pack+逐字节来源验证约 .691–.699s；不当作
免费或全冷启动。单层 QKV bank 为 96MiB，static constants 为 50,331,904 bytes，
另有模型缓存、surface、scratch 和输出；没有导出或估算 32 层 bank 盈利。

原件在 `outputs/qwen512-qkv-private-combo-{gen,edit}[-repeat|-fence|-event]-20261018/`。
独立 CPU/raw/source 复核：

- `outputs/qwen512-qkv-private-combo-audit-20261018.json`：历史四窗口。
- `outputs/qwen512-qkv-private-combo-full-audit-20261018.json`：六窗口。
- `outputs/qwen512-qkv-private-combo-event-audit-20261018.json`：全部八窗口。

最后一份 audit complete：649 项当前 native source 与链接库 manifest 一致；各历史
source snapshot 由自身 SHA 校验，不冒称等于当前源码；22 项实际 layer0 base/factor
来源 hash 跨窗口相同；static MIL/constants 相同；同任务输入与原 GPU 参考逐位相同。
保存的完整 BF16 数组以 FP64 重新算误差并复算 timing/gates、memory 文件 digest。
这不是新一次 GPU 测量或独占设备资格。

工具为 `tools/native/qwen21_{qkv_private_combo_probe.mm,static_lora_prefix.hpp}`、
对应 build script，以及 `tools/validation/qwen_qkv_private_combo_{screen,audit}.py`。
当时 host analyzer 七项测试通过，覆盖旧/新 recipe、真实调用数、preflight、提前 ready
观察、完整消费者数值、强分母、换序、缺失/nonfinite 样本拒绝。

上述八窗口阶段结束时可用磁盘约 14GiB；无新权重下载、无清理用户文件。后续应独立复测
direct-event 增量、比较真实 complement/restore 提交顺序，再评估有界 full-request
盈利与最终图片质量；不根据这一个 layer 直接改 32 层默认调度。

## 补充：提交顺序和打包补块

继续只用同一个原 native library，新增四窗口。没有改产品/native 源码，没有
把局部 block replay 改叫整请求。所有源码和历史二进制分别保留。

### GPU-first 不是净收益

`submission-order-v3` 在相同 nine-arm 窗口保留原 GPU、staged、QKV、Private、
QKV+Private、两条 direct-event 控制，再添加无 ready fence 的 GPU-first 两臂。
GPU-first 真正先 `mx::async_eval(suffix)` 再 `transfer.submit()`；ticket 依旧等待
真实 ready signal，所有输入已物化、两支只读共同输入，输出发布门不变。

每臂 3 warm + 18 measured，18 次循环使九个 visit 位置各出现两次。事前 primary
固定为 `private-qkv-gpu-first`。event 与 GPU-first 各自 A→B→A 预检，共六次
实际 Private 预检，不能用原 schedule 的所有权结论代替新 schedule。

| 任务 | 原 GPU ms | 不等待组合 ms | direct-event 组合 ms | GPU-first 组合 ms | GPU-first 相对不等待组合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 生成 | 35.7217 | 34.3810 | 34.0758 | 35.9368 | 慢 4.525% |
| 编辑 | 37.1084 | 36.7309 | 35.3213 | 36.9448 | 慢 .582% |

相对原 GPU，GPU-first 生成慢 .602%，编辑仅少 .441%，两者均未达到 3% 门槛；
相对同窗口 direct-event 组合分别慢 5.461% / 4.596%。没有因 CPU submit span
变短或工作“更异步”而采用。host spans 仍不证明唯一硬件竞争原因。

每窗口 189 个完整 warm/measured block、132 次实际成功 Private 调用，其中
45 次 GPU-first、45 次 direct-event wait（各含三次自己的预检），其余为不等待
控制。保存两套 complete outputs 与六套完整 FFN chain。新 binary：

```text
build/qwen21-qkv-private-combo-order-v5-20261018/qwen21-qkv-private-combo-probe
SHA256 11ece5c8bc622c7cee98ed567ccb0088d99b3d602a68bf004c33f843437afca8
```

### 一次大 gate/up GEMM 也没有免费收益

`packed-suffix-v4` 保留前七臂，将已经失败的 GPU-first 两臂替换为打包 GPU
补块。只在 setup 将 `[gate[2048:12288], up[2048:12288]]` 原 BF16 base 按行复制，
逐字节验证两个区间。每次调用一次大 GEMM，仍在线计算两个**独立** F32 A-rank，
各自使用完整原 r128 B/scale；B kernel 以真实物理 base pitch / offset 加修正。
保留 base BF16、corrected BF16、SiLU/hidden 和 down-base 舍入，不 merge LoRA、
不让 gate/up 共用不同的 A，也不改 Private 数学和完整 down-LoRA。

原输入和扰动输入上，两个补块输出（partial H4096、hidden F10240）均与原普通
GPU 补块逐位一致；随后单独做 packed A→B→A，旧 A 不变、B 不同、恢复 A exact，
且与原 event 预检 A 的完整 FFN 输出 exact。计时中所有 block/FFN 原门槛不变。

| 任务 | 原 GPU ms | 不等待组合 ms | direct-event 组合 ms | packed 组合 ms | packed 相对不等待组合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 生成 | 35.7631 | 34.2381 | 34.2621 | 35.8522 | 慢 4.714% |
| 编辑 | 37.1029 | 35.7618 | 35.2511 | 36.6289 | 慢 2.425% |

相对原 GPU，事前 primary `private-qkv-packed` 生成慢 .249%，编辑少 1.278%，
均未通过 3% 局部门槛。更少 base GEMM dispatch 不保证更少 full-block wall；
该方案还新增单层 160MiB bank，pack+完整字节验证约 1.153–1.159s，不能隐去。
未继续打包 32 层，也未把 direct-event 和 packed 的历史改善拼接成新收益。

每窗口仍 189 个完整 warm/measured block、132 次实际 Private 调用，含 event
三次和 packed 三次预检；45 次 packed 完整消费者、45 次 direct-event 等待，
GPU-first submissions 为 0。当前 binary：

```text
build/qwen21-qkv-private-combo-packed-v6-20261018/qwen21-qkv-private-combo-probe
SHA256 ebb50c19cfd7f788f625b0f628fa27322d5e5ccdb6c3b209cc03900cc2f27a3d
```

### 完整复核及下一边界

四个新窗口数值、计数、所有权全部通过，但仍检测到外部 ComfyUI CPU 负载，
`qualification_passed=false`。各 memory 窗口 complete，swap-in/out 0；新四窗口
峰值分别 17,816,384,088 / 18,346,979,952 / 17,971,983,128 / 18,552,140,760 bytes。
这包括候选 bank 与检查数组共存，不是单个 arm 的内存收益。

`outputs/qwen512-qkv-private-combo-order-audit-20261018.json` 完成十窗口复核；
`outputs/qwen512-qkv-private-combo-packed-audit-20261018.json` 完成十二窗口复核。
后者仍验证 649 项 native source 与原库匹配，22 项 layer0 来源、static MIL 和
constants 跨十二窗口一致，同任务输入与强 GPU 参考相同；重新算全部 timing、
full BF16 block / FFN errors 和 memory digest。另从
`packed-suffix-controls.safetensors` 复核 H4096 partial 的 4,194,304 元素及
F10240 hidden 的 10,485,760 元素均 exact，保存的 packed/unpacked 完整 FFN 与
最终 block 输出也逐位相同。原 H4096 block reader 未放宽：F10240 控制由按
consumer key 严格限定 shape/dtype/extent、检查截断/重叠/空洞的独立 reader 读取。

十二项 CPU host 测试通过，新增两种 schedule 真实计数/发布预检、submission
order 与跨度、packed recipe 和 raw control ABI 拒绝测试。十二窗口总计 1592 个
warm/measured block（1340 measured）、968 次实际成功 Private 调用，仍不是
独立图片数量。所有 owned build / benchmark / audit 已退出。

结束可用磁盘约 13GiB，无下载或用户文件清理。下一阶段优先把多次约 2.9–3.1%
局部收益的 corrected-QKV-only 路径推进到默认关闭、有界的整请求实验，再看
setup、全部层/step、latents/PNG 和最终图片质量。历史 fused-QKV diagnostic
不是本次在线 r128 A/B + 修正 RMSNorm 的实现，不能直接打开旧 flag 替代验证；
也不能把本次 layer0 的结果乘 32×steps 当整图收益。

后续已完成 [corrected-QKV 整请求实验](qwen512-corrected-qkv-request-2026-10-18.md)：
全部六步/VAE/PNG 的热态观察少约 1.8–2.2%，但 final latent/alpha 事前质量门槛
失败，仍不采用；这替代“尚未测整图”的状态，不替代本页历史原始分母。
