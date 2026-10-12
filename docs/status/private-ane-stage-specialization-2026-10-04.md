# Private W8 staging：编译期专用化与连续负载观察

接续 [1024² 大 bucket](private-ane-large-bucket-2026-10-04.md)。目标 active；
本轮没有新的有效 end-to-end 速度，不把组件数字或受竞争负载影响的输出
填成四格 ≥1.2×、完整 LoRA/质量或连续 device overlap qualification。

## 实现

- Metal function constants 固定 encoding、dense dtype、H128/H512 block，
  排除当前格式不需要的 decode 分支及动态 rotation loop。泛化 shader
  留作同库 off/on 消融；完整 physical pitch、slice、scales/offsets、
  Hadamard/sign table/scale/RNE/非有限性检查均不变，不保存 checkpoint 值。
- 每个 Device 共享一次 library 编译、以 mutex 保护 pipeline 创建，最多
  18 variants（3 dense dtype + 6 packed encoding，各两种 H block）。
  Generic 模式只有 1 variant；key 不持有 model/adapter/source allocation。
  两个 W banks、compact scale cache、A8/未来 W ownership/fences 不变。
- `TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE=0|1` 与 screen
  `--private-stage-specialize`，目前默认0。开关进入 executor identity，
  JSON 返回 `stage_specialized` / `stage_pipeline_variants`，检查完整性、
  有界、session policy 一致和 variant 单调。Public 默认仍不编译私有实现。
- screen 新增 `--observe-load`，每500ms读 process CPU/parent/command state，
  排除明确 owned launch tree，未保留 arguments/prompts。缺采样、过大 gap、
  已知 competing workload 都拒绝该次比较，保留 raw evidence。
  这是 CPU heuristic，**不证明 GPU/ANE exclusive ownership**。
- `run_owned` 对采样/非采样路径统一保护自己启动的 process group，超时/
  取消仅清理该 owned group，恢复 SIGTERM handler。绝不 signal 其它用户任务。

## 通过的回归

- 13 Private host/hardware/MLX：9 source/dtype、H128/H512、W/A、原始 pitch/
  offset/slice、tiny/zero、signed Q8 与 Q4_K/Q6_K等；generic vs specialized
  的 FP16 scales / INT8 codes / padding / flags 逐位一致。18 variants 上限
  检查、compact cache 的 weak generation/LRU/nonfinite、双 bank、A8两槽、
  LoRA hidden、base/A/B/base、late-chunk whole-GPU fallback 均通过。
- 13 Public Core ML/MLX/receipt，51 screen host，6 memory-runner host，4 load
  observer host，8 layout / 3 release guard 通过。加载失败实际 GPU label 与
  2112-row 回归保留。Public library 通过 actual flags/strings/links 发行检查。
- 观察器测试覆盖 own root/descendants 排除、外部 argv 不记录、raw bytes
  被改拒绝、竞争负载/缺样/gap、成功与 timeout 的 owned cleanup。
- `git diff --check` 通过。完整 make test 的 HEAD 既有 Qwen3 字符串断言
  失败未在本轮修改，不宣称全套通过。

Private library：`a1731573295bb56488adfeea153951d97f168a03acf23608a59b115c20427efd`。
Public library：`645b3c61005a214795434f4aac6e442445d290cd0da2162943e5d0ff6e40e60f`。
隔离目录 `build/{private,public}-ane-stage-specialize`。

## 组件正信号，不是请求加速

同一 standalone binary、BF16 full gate/up/down，cache=1，两套 banks，
先排除一次 shader/pipeline first-use，八个热样本：

| 几何 | Generic | Specialized |
| --- | ---: | ---: |
| Z H3840/F10240 | 3.76875 ms | 2.58310 ms |
| Qwen H4096/F12288 | 4.73252 ms | 2.91110 ms |

每组3 misses/24 hits，metadata 48640/57344 bytes，无 full-model W8 副本。
这些没有连续 load observation 的 standalone 初筛，不作为正式速度批准。
Binary hash：`ea725dd1f6039b5c4cbd8c8cc2c766c33d55b6d34416f91bc81ffda18a4fee58`。

## 三次完整请求比较均被连续观察拒绝

新库 Z512/Fa4096/c1056、cache1/fence1/prefetch0/A8lookahead0、8步、三热、
fox/seed42；以下均有短暂 ComfyUI Python >5% CPU，不能证明实际 GPU
冲突，但不满足本次 continuous quiet-load 验证，summary 保持 incomplete：

- `outputs/private-ane-specialize-z512-off-20261004/`：GPU arm 63 samples，1 busy。
- `outputs/private-ane-specialize-z512-off-retry1-20261004/`：GPU arm 54 samples，1 busy。
- `outputs/private-ane-specialize-z512-on-20261004/`：Private arm 47 samples，1 busy。

没有把 earlier arm、fallback、raw wall 或旧库分母拼成成功比较；所有本轮
inference handles 均已 terminal，未终止/改动 ComfyUI 或其它外部工作负载。

## 后续

需要暂时安静的外部工作负载窗口，再以 `--observe-load` 做同库正反序
Z512/1024、Qwen512/1024 和现有 LoRA 实测；对满足连续观测的数据才填
端到端倍率。仍须 GPU head/epilogue/reuse ordering、bandwidth-aware share/
prefetch calibration、Planning JSON W8/channel labels、完整媒体/latent/内存
以及正式四格 ≥1.2×。不根据组件收益升级默认或缩小原目标。

未改发行 build/native、用户 weights/adapters、reference 或设计稿；未 stage/commit。
