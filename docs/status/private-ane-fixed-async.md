# 固定分区：untimed plan 与异步 GPU head

接续 [LoRA range](private-ane-lora-range-2026-10-04.md)。运行环境报告本地日期
2026-10-05。原四格 base ≥1.2×、完整 LoRA/1024²/质量/内存/device overlap
目标保持 active；本轮没有新的有效整请求倍率。

## 实现

固定 positive chunks 原来每层返回 measured Hybrid plan，等待 GPU head
完成后再 join ANE，并对完整 block 加计时/求值 fence。固定分区不从这些
样本更新 scheduler；与稳定 auto 的 untimed 路径不同。

`TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC=0|1` / screen `--fixed-async` 允许同库
消融：1只在 positive fixed chunks、profile关闭时，把 Hybrid plan 转为
既有 HybridUntimed，使用已有 async GPU head 与最终 ownership fence。
默认仍为0；zero-chunk probe、auto calibration 和 Public默认不变。
开关进入 executor identity；verifier 检查实际成功 block 的 untimed/async
counters，不以配置或程序 self-test 推断设备并发。

没有改变 graph、weight recipe、dtype、LoRA rounding、双 bank/A8 buffers
或 source owners。取消、down-callback/GPU错误、late chunk failure 仍先
drain，再保留原错误或完整 GPU 重算；不能消费部分 ANE输出。

## 验证

- 新 Private/Public 各13项 host/hardware/Core ML/MLX回归通过。
- Row fixture 覆盖 fixed async与auto、base/LoRA、BF16/FP16/FP32、profile
  冲突/非法参数拒绝、retained output/hidden、取消、down callback异常和
  完整 tail fallback。Channel fixture验证相同输出、untimed counters、
  owned output和第二 chunk失败后的完整 GPU重算。
- 54 screen、7 runtime host、6 memory-runner、4 load-observer、8 layout、
  3 release guard通过；Public库实际 flags/strings/links发行检查通过。
- Z512一冷三热、Z1024一冷一热生成成功；各自warm样本对原Private实现
  逐像素一致，max uint8 error=0。分别累计1024/512个block，全部实际
  untimed/async。它不替代Qwen新模式整请求、1024 LoRA或正式画质验收。
- `git diff --check`通过。未运行全量 make test；既有Qwen3 stale assertion
  未在本轮修改。新开关未升级默认，不声称观测到硬件INT8 MAC。

新库：

- `build/private-ane-fixed-async`：`da79328f140eb65312b1a5e48b9b03911141d2c54e8fbc130c2c1f5c1401638c`。
- `build/public-ane-fixed-async`：`a87b717e0ef1b11617e9294721e8845e17822bb4d7ec98bd1e2e77888516d6e1`。

## 无效性能证据保留

`outputs/private-ane-fixed-async-z512-on/` 的runtime arm：50个load采样，
1个匹配CPU活动；`outputs/private-ane-fixed-async-z1024-on/`：106个采样，
2个匹配活动。两个summary保持incomplete，不补旧库GPU分母，不忽略gate。
所有本轮 inference/test/build handles已terminal，未终止外部进程。

负载检查的解释仍是CPU heuristic，不证明GPU干扰或独占。只读复核还看到
ComfyUI环境Python的inline进程与Codex app-server同源，但旧证据未保存
argv，不能据此证明所有被拒绝PID都非推理任务。本轮没有放宽检测规则、
重写旧summary或追认倍率。

机器记录见 [fixed-async pilot](../design/validation/private-ane-fixed-async-pilot.json)。
后续仍需有效同库正反序四格/LoRA速度、bandwidth-aware share/prefetch
calibration、实际device trace与完整质量/内存资格，不能用host counters
或输出相同替代原要求。
