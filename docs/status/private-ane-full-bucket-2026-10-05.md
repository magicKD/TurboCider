# Private ANE：1024² 单 bucket 接续

接续 [安静窗口](private-ane-quiet-window-2026-10-05.md)。原四格 base ≥1.2×、
完整 LoRA/媒体/latent/内存和设备并发目标保持 active，不把更少调用当作达标。

## 实现

Private W8 SwiGLU 的软件行数边界从4096扩为4224，4225仍拒绝。显式
4224-row bucket 可以覆盖1024²图像 token 与最多128个 caption token；
较小 bucket、Public 默认、授权/build gate、原始 weights/recipe、两套
W banks、A8 slots、memory admission 和失败后完整 GPU 重算都保留。
本轮没有移植 Splash 的动态 per-128-row program，也没有把单 bucket
升级成全设备/长 caption 的自动默认。

当前 worker 每个 chunk 都等待 GPU restore ticket 结束，才能复用同一
output surface。因此单 bucket 减少一次 client/restore 交接，而不是
宣称两个 chunk 或 GPU/ANE 已不存在设备争用。

Private self-test 保留原 relative L2<0.05，并追加每个 scalar 的
`abs(actual-expected) <= 0.0003 + 0.08*abs(expected)`。大矩阵的平均 L2
可能掩盖坏的末尾行；新检查不是放宽旧 L2 门槛，仍检查所有行的有限性。
实际 driver fixture 使用4224行、三次换权，在 A8-lookahead off/on 下运行；
host emitter fixture检查4224接受、4225拒绝。

## 已完成的同库 screen

隔离库 `build/private-ane-full-bucket`：
`bc76c398eb170d4dbbb55554d2425403fa463b1a9d56613e8d3e3fdd53286bf5`。
它与前库的 build policy 相同（Private=1，test hooks/audit counters/
experimental probes=0），不是复用旧库的 GPU 分母。

Z1024/8步/fox/seed42，一冷两热，native request wall 含 VAE/PNG；
W8A8 GPU I/O、chunks=1、scale-cache=1、launch-fence=1、specialization=1、
fixed-async=1、weight-prefetch=0、A8-lookahead=0。每请求256次实际 Private
调用（此前两 chunk 为512次），0 failed fallback/overflow retry。

| Fa / 顺序 | GPU hot median (s) | Private hot median (s) | 倍率 |
| --- | ---: | ---: | ---: |
| 4608 / Private→GPU | 31.269852 | 26.120886 | 1.19712× |
| 5120 / GPU→Private | 31.271932 | 26.774163 | 1.16799× |

两个 summary 均 complete，连续负载 verifier通过；这仍是 CPU heuristic，
不是 GPU/ANE 独占证明。Fa4608 的保守 min(GPU)/max(Private)=1.19583×，
**严格低于1.2×**，不四舍五入当作达标。更大 share 反而变慢；不能单调
推断或套用 synthetic GPU head 的百分比收益。

原始目录：

- `outputs/private-ane-full-bucket-z1024-a4608-c4224-quiet/`
- `outputs/private-ane-full-bucket-z1024-a5120-c4224-quiet/`

之前 `outputs/private-ane-full-bucket-z1024-a4096-c4224/` 的 runtime arm
生成成功，但164个load samples中5个匹配竞争进程；summary保持incomplete，
没有跑其 GPU arm，不填倍率。之后在短窗口中重新核验正在运行的
`download_ltx25.py` 的 executable/kernel argv/启动时间/非 stopped state，
仅对该具体 PID 使用获授权的 SIGSTOP。新的恢复 ledger 为
`outputs/private-ane-full-bucket-download-pause-ledger.json`；旧 ledger 已
resumed，不重写旧记录。窗口结束后必须再次核验并 SIGCONT。
本轮所有测量/build/test handles已终止后，已重新核验新进程的 birth、
executable、kernel argv 与 stopped state，SIGCONT后观察到不再stopped；
新 ledger 也已 resumed。下载未修改/删除，不将暂停状态遗留到交接。

## 长行数 GPU head 候选

仅 Z dense base 的显式 Private channel callback 在已限定 M4 Max profile
下测试原有 physical-pitch MPP range kernels 到4224行；普通 GPU block、
Public row executor 和 LoRA/GGUF callback 不走此新增范围。没有使用
几乎持平的 dual-head 候选，也没有复制 compact checkpoint matrices。

新的 row fixture 在33/4128行、FP16/BF16、BM16/BM32、非零 row/column
begin、尾块和 original-vs-compact 逐位比较；BF16 separate vs dual128
保留原舍入边界。长行数 fixture已通过，整请求进一步独立验证如下；
不能根据组件的旧正信号直接推断模型达标。

第二隔离库 `build/private-ane-full-bucket-mpp`：
`86144106019f15a289007f148a1ba1a447ef0f77d98ceb5627457a9fe976fa3f`。
13项 Private host/hardware/MLX、54 screen host、8 layout通过；该库使用
`TURBOCIDER_NATIVE_ONLY=1` 构建，没有重复Swift收尾。新库的同库GPU/
Private整请求已完成，不以第一库的GPU时间作分母。Z1024/Fa4608/
c4224/8步，GPU→Private，一冷三热，load verifier通过：

- GPU hot：31.286677459 / 31.303811292 / 31.309993542 s。
- Private hot：25.780713375 / 25.828577084 / 25.583057542 s。
- 中位倍率1.21423×，保守 min(GPU)/max(Private)=1.21132×。
- 每请求256次实际Private调用，0 retry/fallback；不是GPU decline。

目录 `outputs/private-ane-full-bucket-mpp-z1024-a4608-quiet/`。这是一格、
一个提示词和种子的有效初筛，不是全四格正式资格。MPP新/旧Private首个
热输出逐像素一致（max uint8 error=0）；GPU vs Private RMSE=8.492610、
PSNR=29.549980dB、correlation=0.992436、max abs=205。前者证明本次
GPU head 优化没有改变该Private固定样本，不替代后者的W8A8质量验收。

Qwen1024/c4224/base v1/Fa5120/40步、相同Q/K norm-RoPE的GPU/Private
同库对照已完成，Private→GPU、一冷两热：GPU median=183.193043s，
Private median=145.596478s，1.25822×。连续load检查两路线都通过，
每请求1280次实际调用、0 retry/fallback。目录
`outputs/private-ane-full-bucket-qwen1024-a5120-c4224-quiet/`。
这是一个固定workload的有效初筛，尚无反序和多提示词正式资格。

CPU PNG工具已通过4项回归（含constant亮度解析值、精确图、尺寸/dtype
拒绝及alpha差异）。首个热输出：Z1024 RGB SSIM=0.971125；Qwen1024
RGB RMSE=2.585337、PSNR=39.880459dB、correlation=0.999396、SSIM=
0.994998，alpha RMSE=0.113134/max abs=3。Qwen两幅fox图已查看，
没有观察到该样本的明显结构破坏；仍不等于latent/LPIPS/CLIP/语义资格。

第二提示词灯塔/seed17、Private→GPU、双路线独立memory sampling已完成：
GPU=31.436570s，Private=26.000643s，1.20907×，保守边界1.20435×。
load与memory verifier均通过、0 retry/fallback。该实验同时换提示词
和顺序，不能冒称同一fox提示词的反序。目录
`outputs/private-ane-full-bucket-mpp-z1024-lighthouse-reverse-memory/`。
Private/GPU peak footprint=45,241,165,472 / 45,547,562,584 bytes，swap
in/out均为0；Private有2,686,976 bytes decompression，不是完整driver/
wired/低内存资格。RGB RMSE=5.931773、PSNR=32.667113dB、SSIM=
0.976618、correlation=0.995381、max abs=180，仍不作语义或latent通过。

## 验证范围与待完成项

实现已单独提交为 `73cf497`。验证库是上述working-tree快照（含独立
ConvRot用户改动），不是clean staged tree重建；本轮没有重建Public库。
未变动的现有Public库发行guard再次通过。原完整`make test`的Qwen3
stale source-string断言未在本轮修改或重跑，不声称全suite通过。

- 扩边界前期实机：12通过/1 channel-MLX skip；加逐元素检查后的新库：
  13项 Private host/hardware/MLX通过，包含 source/stride、headroom、
  owned output、cancel与 late-chunk完整GPU回退。
- 长行数 range专项：4通过/9未选硬件测试skip；不是13项全套复跑。
- Native/App/Swift构建成功；不代表运行了App全部集成测试或分发了private包。
- 原模型/adapter/reference/设计稿不改；独立 ConvRot 用户改动保留。

当前还要完成Z/Qwen1024同提示词反序/多提示词、1024² LoRA、
Z512 稳健裕量和两模型多提示词/latent/感知质量、内存、device trace及
bandwidth-aware calibration。Public 默认与未资格边界不变，未声称
native INT8 MAC 已被观测到。所有性能报告只代表所列构建与工作负载。

逐 case summary/manifest/连续load哈希与原始hot样本另见
[机器记录](../design/validation/private-ane-full-bucket-pilot-20261005.json)。
新增 `runtime_ane_image_compare.py` 只进行等尺寸PNG的CPU数值比较，定义
RGB通道均值Gaussian11/sigma1.5/valid-window SSIM，不重采样、不丢alpha、
不下载学习权重。SSIM不是LPIPS/CLIP/latent或语义资格；该工具已在Qwen
完整对照结束后测试并用于图片复核，不与模型benchmark重叠。

1024² Qwen LoRA 的请求门目前仍明确限512²（`qwen21_module.cpp` 的
六步adapter验证）。不能仅因v2模板已导出就称其兼容完成；需要独立
扩展限定生成入口、保留原FP32 rank并明确拒绝未资格FP16/编辑组合，再
做同库实际模型/GPU对照。此门未在运行中的benchmark期间修改。
