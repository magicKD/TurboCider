# Private W8A8 接续：GPU stager 与 MatMul 闭环（2026-10-03）

接续 [GPU IOSurface I/O](private-ane-gpu-io-2026-10-03.md)。本阶段完成GPU
decode/Hadamard/W8-A8 staging与真实private ANE MatMul闭环；尚未接入
Z/Qwen完整SwiGLU，不是模型级双缓冲交付，也没有新的整模型性能成绩。
此处保留组件阶段的证据边界；后续完整 FFN/Executor 和真实请求接续见
[W8A8 Executor](private-ane-w8-executor-2026-10-03.md)。
原任务四格≥1.2×、LoRA/画质/稳定性验收继续保留。

## 已实现

- 共用`DeviceWeightView`明确immutable physical matrix、buffer/owner、
  offset、完整row pitch、encoding与metadata；`W8StageSpec`的逻辑切片
  不改变原始行寻址，不生成compact全权重副本。
- `Device::stage_w8`支持dense F16/BF16/F32、MLX affine Q4/Q8和原始
  GGML Q4_0/Q4_K/Q8_0/Q6_K。两遍GPU decode+rotate：先计算row scale，
  再直接写INT8 IOSurface。没有全矩阵FP16/F32中间结果，threadgroup
  scratch最多512 floats。W8为out/in，A8为channel-major。
- 当前 recipe 为`sylvester-dh-b128-b512-rne-norm-f16-v2`：R=D*H/sqrt(B)，
  H128/H512、固定seed signs，两边同R保持dot invariant。不是Comfy H256。
  v1 的 H*D 被完整 FFN 的独立 hidden oracle 发现与图内 conv 不一致；
  v2 使用 input signs 后 H，GPU/CPU/ANE conv 三者同序，不沿用 v1 资格。
- scale为RNE_F16(peak/127*128)，codes使用该已舍入scale做signed RNE与
  [-127,127] clipping。zero row用exact scale128/q0；tiny nonzero row
  normalized scale下界2^-24。非法源、rotation/scale溢出失败，不clamp。
- stager使用独立GPU queue和ready event，不能提前推进当前层
  activation/ANE-done timeline。ticket持有source/targets/metadata到GPU
  completion；flags非零不能发布成功，timeout保留可能仍在用的对象。
- `w8_matmul_program`使用INT8 runtime x/w、1/128 dequant、normalized
  FP16出口。GPU epilogue以FP32顺序 `(y*sw)*sx` 恢复，再一次RNE到BF16。
  不在ANE FP16出口先恢复大幅值，不宣称native INT8 MAC已观测。

## 已执行验证

M4 Max64GB：

- 独立dense Hadamard oracle、H128/H512 orthogonality/dot invariant、
  deterministic signs、正负RNE ties/clipping、zero/tiny scale通过。
- 9组GPU source组合（3 dense dtype、2 affine、4 raw GGUF）通过。
  两种B、W/A layouts、宽physical pitch/column slice、zero/tiny rows的
  FP16 scales与I8 codes逐位匹配CPU decoder/butterfly recipe；padding
  不覆盖，短buffer/Inf/scale overflow拒绝，clean refill恢复。
- 真实W8A8 M33/K512/N16闭环通过：GPU X/W staging → private INT8输入图
  → GPU FP32 epilogue。normalized INT64 dot oracle最大1 ULP（门槛2）；
  epilogue与实测normalized出口的独立FP32恢复/RNE oracle bit-exact。
- probe恰有两套W banks，A/B/A换权，准备bank1不覆盖bank0、不推进当前
  ANE event。未采集hardware overlap trace，不据此声称物理重叠已证明。
- 10项private host/hardware、7项既有host、最终public库12项Core ML/MLX
  集成通过；public-only class strings/PrivateFrameworks link隔离通过。
  `git diff --check`通过。未执行完整make test、实模型W8A8或正式campaign。

构建身份：

- Private：`8459a0c00e632f33aef4a7e4a297fca8b3955769427b0449adc8f2095ce40c18`。
- Public-only：`e39c7abaa4e12fa59eb08d11ff3e1e78c5152c56502324b2ae9d0f9abbd3aff9`。

```sh
TURBOCIDER_TEST_PRIVATE_ANE=1 .venv/bin/python -m unittest discover \
  -s tests/native -p test_private_ane.py -v
```

默认发现仅运行host tests；hardware skip不计作通过。未改模型、参考仓库、
发行build/native或App，未stage/commit；没有仍在运行的benchmark。

## 必须继续

1. 完整W8A8 SwiGLU：gate/up scales、SiLU、ANE内H512+A8、down、normalized
   出口/GPU恢复；tiny/zero intermediate和真实Z headroom16/64回归。
2. LoRA corrections、corrected hidden、down-LoRA共用dtype/scale合同，
   base/A/B/base、多训练adapter、媒体与latent资格。
3. 生产HybridFfn仍为单套FP16 slots；probe两bank不是全模型双缓冲。
   需接source leases/scale cache、固定两套W8 banks、层间预取/reuse fences、
   取消/失败整段GPU重算和实际内存准入。
4. 统一row/channel与bandwidth/memory calibration，真实checkpoint与捕获
   activation、模型误差，以及两模型512²/1024²的正反序、多hot匹配纯GPU
   ≥1.2×正式验收。

此前6374库Qwen1024 FP16 GPU-I/O单hot1.216×保留原身份，本轮不把它改名
为W8A8成绩，不外推另外三格或提升默认。
