# Private ANE：对齐读取与 GPU channel-head 接续

接续 [staging 专用化](private-ane-stage-specialization-2026-10-04.md)。原目标保持
active：Z/Qwen base 的 512²、1024² 四格正式 ≥1.2×，完整 LoRA、画质、
内存和实际 device overlap 仍未验收。本轮不填新的有效整请求倍率。

## 实现与边界

- 专用化 W8 dense decoder 在 **binding offset 和 physical row pitch 都满足
  dtype 对齐** 时使用 ushort/float load；其它视图和 generic oracle 保留
  byte decode。FP16 仍走整数转换，不能用 Metal half load 丢失 subnormal。
  H128/H512、signs、scale、RNE、原始 weight addressing 和 recipe 不变。
- 同一个 18-variant 上限和 `--private-stage-specialize` 开关；默认仍为0。
  不新增全模型 W8/dense 副本，不改变两套 W banks、A8 slots、source lease、
  reuse fences 或失败后完整 GPU 重算的所有权边界。
- 增加 `swiglu_dual_gemm_range` 候选与 standalone GPU sweep。它在一个
  dispatch 中做 gate/up，保留 BF16 epilogue、原始 row range 和 tile-tail
  检查。**没有接入模型默认 channel head**：实测相对已有 MPP32 近乎持平。
- `build_private_ane_stage_benchmark.sh` 提供可复现的独立 staging 编译入口；
  组件 JSON 增加实际 specialization/variant receipt，不进入 shipping build。

## 验证

- 新私有库、显式 specialization=1 下的13项 Private host/hardware/MLX
  回归通过。覆盖 FP16/BF16/FP32、独立非对齐 offset/pitch、H128/H512、
  W/A、padding、9种 source/dtype、cache identity、2112-row、A8双槽、
  LoRA hidden/base-A-B-base、late-chunk whole-GPU fallback。
- Dual head 检查非零 row start、128/384/512 channels、BN128/256 尾块、
  original-vs-compact 逐位一致，BN128 与 separate-projection BF16逐位一致，
  invalid ranges/tiles 拒绝，base weight 不变。
- 51 screen、4 load-observer、8 repository layout 通过；`git diff --check`
  通过。未在本轮重建 Public 库或运行完整 make test；提交前已复现的
  Qwen3 stale source-string assertion 不在本轮修改。
- Z512 与 Z1024 完整请求生成成功；新旧 Private 输出逐像素一致，max
  uint8 error=0。仅证明本次 decoder 优化没有改变这些固定样本，不能替代
  GPU baseline 的多提示词、latent/LPIPS/媒体质量验收或 Qwen 新库整请求测试。

私有库：`94c4f87ff8066953880836d8b78e40d731300bc3d07088c8f7fbb08a5d78d8de`，
位于 `build/private-ane-dense-load-runtime`。Public 默认、distribution gate、
用户 weights/adapters 和独立 ConvRot 改动保留；不声称观察到硬件 INT8 MAC。

## 组件测量，不是端到端加速

原始记录：`outputs/private-ane-dual-head-component-20261004/summary.json`。
同一 standalone binary、BF16 full gate/up/down、两套 banks、scale-cache=1，
排除首次 shader/pipeline，八个热样本；每个 arm 都有连续 CPU load observation：

| 几何 | Generic | Specialized，含 typed reads | 证据 |
| --- | ---: | ---: | --- |
| Z H3840/F10240 | 3.71298 ms | 2.27679 ms | Generic 被 CPU 活动拒绝，不算倍率 |
| Qwen H4096/F12288 | 4.75506 ms | 2.83175 ms | 两个组件 arm 完整；不是模型倍率 |

这是既有 format/dtype/block specialization 与 typed reads 的**合并**消融；
没有独立测量 typed reads 的净收益，不能将差异全部归因于 typed load。
观察器是 CPU heuristic，仍不证明 GPU/ANE 独占或 device overlap。

GPU-only Fg6144 head 的10个热样本：1056 rows 的 MPP32/dual128 为
10.2086/10.2014 ms；4128 rows 为38.4808/38.5074 ms。Dual256 更慢。
两组 component load observation 完整、source L2=0；微小差异没有足够
正信号支持切换默认，也不是全模型速度/画质资格。

## 整请求证据为何仍无效

- `private-ane-committed-z512-specialize-on-20261004`：Tongyi 本地目录缺少
  transformer 权重，明确失败；后续改用已有完整 Comfy BF16 checkpoint，
  没有修改或下载模型。
- `private-ane-committed-z512-specialize-on-v2-20261004`：runtime arm 的46个
  采样中有1个匹配 ComfyUI 环境 Python CPU 活动，summary incomplete。
- `private-ane-dense-load-z512-runtime-20261004`：41个采样有1个匹配活动。
- `private-ane-dense-load-z1024-runtime-20261004`：107个采样有3个匹配活动。

后两项生成和数值兼容检查成功，但 load verifier 拒绝，summary 仍
incomplete。CPU 活动不等于已证明 GPU 干扰；也不能为得到倍率而忽略
预先定义的 gate，或将旧库 GPU 分母与这些新库 raw wall 拼接。未终止任何
外部进程。本轮所有 inference/test/build handles 均已 terminal。

机器记录见 [dense-load pilot](../design/validation/private-ane-dense-load-pilot-20261004.json)。

## 后续

继续同库正反序四格、share/prefetch 和带宽 calibration。Qwen LoRA 的
channel 路由目前先请求 full gate/up correction 再切 ANE channels，GPU
partial 又有自己的 adapter 投影；应验证更窄的 correction callback 是否
降低实际成本，保留一次 full-hidden down-LoRA、原精度和完整 GPU fallback。
这是待测热点，不以静态代码推断已经取得 LoRA 加速。
