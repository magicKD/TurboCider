# Public W8A8：真实 LoRA / 1024 生成与 full-K 控制

2026-10-07，Asia/Singapore，M4 Max / macOS26.6.2。接续
[完整共享FFN接入](public-w8-shared-ffn-2026-10-07.md)。本轮是整模型
兼容性/质量证据和一个explicit图切分实验，**不是**完整加速资格。
全部Z/Qwen base/LoRA、512/1024、GGUF/ConvRot、multi-prompt、正式
matched GPU/load/memory/trace目标仍active，没有缩成下面四个cell。

## 同库、真实执行，不是小图 correction 模拟

同一v2 CLI/dylib，Public backend、真正INT8 runtime weights、GPU IO，
original BF16 models；每格GPU/runtime各独立process，8个process串行，
未与owned build/test/timed benchmark重叠。fox prompt/seed42、resident。
chunks1/fixed-async1，scale-cache/specialize/launch-fence1、prefetch/
lookahead0；512用352-row模板，1024用1056-row模板。每图没有学习到的
checkpoint/LoRA矩阵const，两套W bank沿用共享GPU桥接。

Qwen使用真实Viggle rank256 strength1、6步、FP32 rank，227 applied
projections；1024 diagnostic开关在两个arm一致施加。Z使用真实distill
patch strength1、8步、238 applied projections。未合并adapter权重。

| cell | actual steps / CoreML calls | headroom / retry / fallback | final relL2 / cosine | 历史N1 |
| --- | --- | --- | --- | --- |
| Q512 LoRA | 6 / 192 | 1 / 0 / 0 | 0.0301313470 / 0.9995459561 | fail L2 |
| Q1024 LoRA | 6 / 192 | 1 / 0 / 0 | 0.0098075575 / 0.9999521876 | pass |
| Z1024 base | 8 / 256 | 1 / 0 / 0 | 0.0737120245 / 0.9972840901 | fail |
| Z512 LoRA | 8 / 258 | 16 / 2 / 0 | 0.1077736467 / 0.9941859165 | fail |

Z512 LoRA包含两次真实headroom retry，不是258个独立FFN block；成功
block仍256。所有runtime_failed=false、failure_reason空，device IO
counts绑定每次实际prediction。Public实际CoreML execution为true，
Private ANE execution为false；不根据public label宣称物理NE residency。
conditioning、initial latent逐位一致，全部artifact前后不变；历史N1
及qualification_passed=false保留，最新视觉目标不由N1唯一决定。

## 原尺寸视觉检查

GPU/reference与候选whole、center/top-left/bottom-right三张同坐标裁剪
均按原尺寸检查（共16 sheets），不resize/alpha丢弃。agent观察：

- Q512 LoRA：姿态、眼鼻、松针、尾部和积雪很接近；细毛/背景有小变化。
- Q1024 LoRA：整体和胸毛/雪地裁剪非常接近，未见显著新增伪影。
- Z1024 base：脸部、光照、构图很接近；胸毛/尾毛和局部枝条细节略变。
- Z512 LoRA：整体很接近但不是不可区分；耳部轮廓、胸毛及背景枝条
  有可见变化，没有明显新棋盘格/色块或肢体破裂。

RGB SSIM分别0.9900231、0.9968000、0.9835356、0.9687931，仅数值补充。
这是单prompt/seed的agent review，不是盲测、用户批准或所有图像
普遍质量保证。自动visual manifests仍pending，不静默生成pass。
与用户要求一致：继续以视觉非常接近为目标，latent只保留补充诊断。

目录 `outputs/native-public-w8-v2-{q512-lora,q1024-lora,z1024-base,z512-lora}-20261007/`
各保留request、两臂observed receipt、dumps/PNG、quality和visual。
质量JSON SHA256依上述顺序：

```text
528bde221364cb0fe3e296e79318664dbe1528957cf021ce54636e76569f36f8
295c2506d731ba457f84514bb92a0d595423365ab88e08b9f21394cdcf4160e4
d0f0ad2b0a514f45177471af69f409eb45dd9e83444d5d70e8662f4c3c213a1e
e9ea42974ca5a859446d607a666f741eedfe1eff985c634194cdecdeee70af07
```

## 诊断时间：保留不利结果，不当正式倍率

| cell | GPU / Public denoise s | GPU / Public request-wall s |
| --- | --- | --- |
| Q512 LoRA | 7.947319 / 7.571437 | 14.020673 / 12.170232 |
| Q1024 LoRA | 32.014331 / 31.022667 | 39.488531 / 40.791304 |
| Z1024 base | 30.410738 / 27.081243 | 34.738480 / 34.034859 |
| Z512 LoRA | 8.297414 / 7.853062 | 11.852995 / 10.407405 |

Q1024 LoRA request-wall候选反而更长，不能只挑denoise。四格均带dumps/
observer，未隔离cold load/OS cache状态，没有hot repeats/reverse/
strict load/memory gate；不计算正式speedup，不据此提升任何share默认。

## full-K：图操作减少不等于盈利

Public exporter允许explicit tile_k最高16384，default仍1024。INT8
经1/128 dequantize后各operand在[-1,1]，K≤16384的normalized dot
低于finite FP16范围；原restore/headroom/finite guards不改。切分改变
partial accumulation顺序，不声称bit-exact或和Private相同算术。
新增structure test证明小控制从6个MatMul变3个，ABI/input/output不改，
default仍1024；超范围拒绝。

Q512真实LoRA同库/同input/share另跑full-K（tile16384）candidate，
actual192 calls、headroom1、retry0、failure/fallback0。比较绑定同库GPU
reference，final relL2 0.0349936868、cosine0.9993875431，旧N1 fail。
常规tile1024的relL2为0.0301313470；没有选择更有利的质量样本。

常规/full-K diagnostic denoise为7.571437/7.455464s，仅一次、没有统计
或matched-window资格；累计prediction API span反而从2.543993增至
3.053581s。该span含exposed readiness/handling，不是纯NE kernel时间。
没有可靠盈利证据，不改default，不把1.5%的单次整体差异包装为优化通过。
实验接口保留供后续同binary paired/reverse component校准。

`outputs/native-public-w8-v2-q512-lora-fullk-20261007/`保留所有输入/输出/
receipt/quality，后者SHA256
`6fd982c447cc39d7a17710edfbe2d6983c0e224d211401a115de3c77118f070d`。
模板与1024切分是不同compiler artifacts，前后complete file hashes绑定，
没有把相同W/A representation recipe误当同一compiled graph身份。

## 正式测速：两个方向仍被正确拒绝

Z512 base同v2库一冷三热、continuous load+process-tree memory：

| 方向 | 完成arm | load samples / busy samples | accepted trials |
| --- | --- | --- | ---: |
| GPU→runtime | GPU四次，runtime未开始 | 58 / 1 | 0 |
| runtime→GPU | runtime四次，GPU未开始 | 56 / 2 | 0 |

外部ComfyUI CPU负载触发严格拒绝，两个summary均incomplete，run正确
退出失败。目录 `outputs/native-public-w8-z512-base-{forward,reverse}3-20261007/`
保留全部生成、采样和日志。不能拼接这两个各自被拒绝的arm成为matched
comparison，也不移除样本、杀外部进程或降低gate。当前无新正式E2E倍率。

## 回归、provenance、仍未完成的目标

80项相关host/tool tests通过；隔离coremltools9下5项FFN/export/native
integration tests通过，无skip。full-K边界/graph controls新增，默认与
完整GPU/CoreML/GPU小图失败恢复复跑保留；不是整仓绿色。
Private v2库仍486 source inputs与current native tree匹配、seal一致，
library SHA256
`290c8a1f07181eadea1b5c92fc7379d5e2a5e23b8fb3818846691d218a51df61`。
本轮只改exporter/tests/工具说明，不更换native library、不改模型或
原未提交草稿，不夹带references/adapters/design改动。

下一步仍需Public Qwen base512/1024、Z1024真LoRA、GGUF/ConvRot实模型、
多prompt/seed视觉和完整traffic/backend/partition的profitability选择。
正式要求是实际快于matched optimized GPU，不强制旧1.2×；物理并发/
native INT8/F32 ANE arithmetic仍不可凭API成功断言。整体目标保持active。
