# Qwen512 corrected-QKV：舍入修复、真实block小幅收益，尚不接入整图

2026-10-18，Asia/Singapore，M4 Max/64GiB。只关注Qwen Image 2.1生成/编辑，
不改Z Image、已安装App或默认设置。完整优化目标继续，以下不是整图加速发布。

## 决策

实现了两类研究GPU数据流，并以真实r128/joint普通compiled block作强分母：

1. 原在线A→F32 rank，单MPP kernel完成base+B修正、Q/K RMSNorm/RoPE，
   直接输出head-major Q/K/V；保留base BF16及corrected BF16边界。
2. 把三个base和三个**独立**A矩阵按行拼接，同一GEMM计算base与A，
   FP32累加后在原sole-B输入处舍入BF16，再由第二kernel完成B/norm/RoPE。
   没有把不同A当成同一A，也没有merge adapter、裁token/步数、跳过FFN。

局部完整block热态少约2–3%，新base+A方案没有进一步盈利，均未达到预设3%
局部入口门槛。保留原件/工具，不注册新的产品flag、不打包整套32层bank，
不把局部结果乘32×steps或套用到完整图片。所有完整性能窗口持续CPU-load检查
失败，`qualification_passed=false`，未停止用户推理。

## 最初失败与真正的精度修复

第一版在真实生成block上relative L2=0.001082393，超过事前0.001，虽然cosine
很高也仍拒绝；没有运行完整计时循环或授予速度收益。GPU精确重放及普通MLX
packed-base控制均exact，失败的全输出数组、source、binary与日志保留。

读取MLX固定源码 `ml-explore/mlx@36969c094354086e15d2a327a9c50feb20b9cf11` 的
`mlx/backend/metal/kernels/rms_norm.metal`及`normalization.cpp`，发现原型漏掉
了RMSNorm内部的一次typed rounding。更关键的是增加**实际安装库**见证：

```text
native_RMSNorm_BF16(x, gamma)
  == BF16(native_RMSNorm_BF16(x, ones) * gamma)
  != BF16(native_RMSNorm_F32(F32(x), F32(gamma)))   # 本见证数据确实不同
```

使用33×128个确定性BF16值及BF16 gamma。源码tag本身不当作wheel实现证明；
实际库见证与原生GPU参考共同确认本次修复。新kernel先对normalized值舍入BF16，
再乘gamma并舍入，然后才F32 RoPE→BF16。没有放宽原数值门槛。

修正后40个实际Metal数值case、90个非法contract拒绝全部通过。覆盖H128/256，
R64/128/256，M1/33/65/129，BM16/32/48/64，非零origin、非紧凑/未eval输入、
零值及±scale；完整三路输出对原生GPU参考最差relative L2约6.52205e-9。
其中包含projected输入的完整physical pitch与rank分区，不把小型测试当全模型资格。

## 真实输入与完整消费者

原BF16 Comfy encoder/DiT/VAE、固定Viggle v0.2.1 r128，SHA256
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`，
strength1/inference_time、原227个logical adapter terms、joint BF16 A/B/F32 ranks。
输出请求512²/6steps/seed73，生成蓝茶壶及单ref512改红；没有ANE、残差/FFN时间
复用、encoder融合、VAE近似、权重merge或新下载。

同进程observer借用step1/layer0的真实9个参数、原GPU callable和已完成参考输出，
实际decode M1024/H4096/R128；生成/编辑prefix分别24/1059 tokens。
计入每次在线rank计算、base/B、Q/K/norm/RoPE、attention与prefix concat、
output projection、完整FFN/down-LoRA及两次residual，使用同一调用与完成边界。
原始数据、权重、全部LoRA factors前后hash不变；改变输入重新执行并检查消费者。
新输出在eval前必须lazy，不能把上次已完成tensor当成本次GPU执行。

所有预热/首个重放/计时调用均检查完整4,194,304元素输出，block relative L2≤.001、
cosine≥.999；原GPU和等价普通GPU重放要求changed=0。reference、各候选完整输出
均保存原始BF16数组。修正版真实生成/编辑fused输出relative L2约8.74167e-5 /
1.60555e-5，并非逐位等价模型；没有把这项局部门槛当最终图片质量门槛。

observer在step1/layer0后受控停止，只完成首个完整DiT step，**没有VAE/PNG**。
模型/编译/pack/readback与独立输出检查在热block时间外并单列，不叫完整请求。

## 相同窗口的完整block时间

单位ms，中位数。每个任务各自匹配分母，不能跨窗口相乘或拼接最快phase。

| 版本/任务 | 原GPU | 单kernel fused-B32 | joint base+A MLX+B32 | joint base+A MPP+B32 |
| --- | ---: | ---: | ---: | ---: |
| v2生成，6臂×3warm+12计时 | 35.7353 | 34.7224（少2.834%） | — | — |
| v2编辑，同设计 | 37.0862 | 36.1578（少2.503%） | — | — |
| v3生成，9臂×3warm+18计时 | 35.7656 | 34.8232（少2.635%） | 34.9532（少2.272%） | 34.8564（少2.542%） |
| v3编辑，同设计 | 37.1131 | 36.0481（少2.870%） | 36.3965（少1.931%） | 36.2993（少2.193%） |

普通MLX packed-base连完整LoRA处理及必要contiguous/copy，四窗口均约慢1–1.2%，
不能拿它作弱控制宣布fused更盈利。BM16/48/64未超过BM32；完整原始样本全保留。
v3在测量前固定primary为joint-base-a-mpp-b32，不因单kernel列较好而重标primary。
v3首个真实输入的三路rank BF16 B-input boundary对原joint消费者全部逐位相同，
但移除F32 rank store并不自动获得净收益；还增加base写回/读回和不同GEMM形状。
没有DRAM/register counter，不能将唯一原因断言为某一个流量或资源瓶颈。

四窗口共558个完整warm/计时block调用，不是558个独立prompt/seed。cycle换序；
预检与fresh-input执行不混入热分母。准备和首调用不能当免费：v2单96MiB base
pack+完整验证约0.693–0.696s；v3同时保留96MiB base及99MiB joint bank约
1.413–1.437s；这些包含CPU hash校验，不全是GPU copy。初始forward/编译单列，
部分在后续process已受OS/driver cache预热，不宣称完全冷启动。未导出32层bank。

## 来源、内存及独立复核

没有修改或重建native产品库，研究工具明确链接：

```text
build/qwen512-tokenizer-reuse-v2-20261018/libturbocider.dylib
SHA256 a225f17e0fcbe04f7fa2b2a6b4c8178f2f663cb4bb20e43989022fc531c61e70
runtime tc-runtime-build-v1-2d36a6f10c424a5b670fa1317c73a00e74b4df1aeb7e587892db9d32fdce9e44
```

v3 binary `build/qwen21-qkv-lora-block-v3-20261018/qwen21-qkv-lora-block-probe`，
SHA256 `c3cb4acfd7fd7046cf00ee980fed0a9830707a985d8ff988f10c353d295778ba`。
`--describe`在模型工作前核对编译进binary的cpp/header SHA、recipe/arms、sample
数量和实际linked runtime ID；不把当前源码冒称旧binary的源码。

`outputs/qwen512-qkv-lora-block-audit-20261018.json`独立CPU复核complete：649项
native source仍与链接库一致；历史四份source snapshot各自匹配；从保存的全输出
BF16数组按FP64重算数值，复算完整时间分母，复现初版失败；22份实际layer0
base/factor来源hash在生成/编辑与版本间相同，同任务的输入/GPU参考亦exact。
没有重跑GPU或因CPU audit而授予physical/exclusive资格。

四个成功窗口memory均complete，峰值17,501,270,496 / 18,083,165,072 /
17,664,848,568 / 18,262,520,768 bytes；system swap-in/out0，少量decompression
已保留。它包括模型load、全部candidate共存、数值检查与退出，不是每arm峰值。
不得将96/99MiB逻辑bank或单层峰值外推完整模型的常驻/流量或整图内存收益。

原件按版本保留：

- `outputs/qwen512-qkv-lora-block-gen-20261018/`：首个数值失败，不授予速度收益。
- `outputs/qwen512-qkv-lora-block-{gen,edit}-rms-v2-20261018/`：6臂完整窗口。
- `outputs/qwen512-qkv-lora-block-{gen,edit}-joint-v3-20261018/`：9臂完整窗口。
- 三版binary各在独立build目录，当前源码与历史source snapshot不混用。

代码在 `tools/native/qwen21_qkv_lora_{prepare.hpp,block_probe.mm}`；host analyzer
5项拒绝/计数/强分母测试通过，实际Metal40/90通过。未改native model API，
不将旧135项library回归冒称本轮又重跑。磁盘约16GiB可用，无新模型、无用户缓存
清理、无部署/默认变更，owned构建和设备任务已结束。

```sh
TURBOCIDER_NATIVE_OUT=build/new-corrected-qkv \
  bash tools/native/build_qwen21_qkv_lora_block_probe.sh
build/new-corrected-qkv/qwen21-qkv-lora-block-probe --kernel-contract
.venv/bin/python -B tools/validation/qwen_qkv_lora_block_screen.py \
  --binary build/new-corrected-qkv/qwen21-qkv-lora-block-probe \
  --library build/qwen512-tokenizer-reuse-v2-20261018/libturbocider.dylib \
  --output outputs/new-corrected-qkv-generate
```

编辑另加 `--reference`。下一步更有价值的是与既有static Private FFN分担进行
**同一完整block的真实组合**，再决定是否支付32层bank与整图质量/生命周期成本。
不能将本轮2–3%与旧ANE结果相乘，也不能由未测试的组合宣称整图变快。
