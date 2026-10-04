# 双后端接续：Private ANE 基础执行器（2026-10-03）

目标仍是用户要求的完整双后端、Private W8A8/Hadamard、GGUF/Q4 GPU
转换、双缓冲、shared-event 同步、Z/Qwen base 的 512²/1024² 和 ≥1.2×
端到端性能、现有 LoRA 兼容。本次不是最终验收，也不把 FP16 基础路径
改称 Private High-Performance W8A8。

## 当前实现

- `ane_runtime.hpp::Executor` 定义共用 shape、WeightView、异步 staging、
  launch/finish、调用输出所有权与全段 GPU 失败重算合同。
  Public `RuntimeGraph` 和 Private `PrivateGraph` 均实现它；两模型通过
  既有 `HybridFfn` 和 `RowScheduler` 接入，不复制模型 FFN/attention。
- `ane_backend.{hpp,mm}` 维护 `public/private/auto/off`；默认 public。
  private/auto 要显式授权，auto 对私有构建缺失/能力自测/内存不足回退
  Public，Public 不可用仍走既有完整 GPU 重算。私有执行期失败当前禁用
  本执行器；不是同一层零开销切换到 Public。
- `private/ane_program.mm` 动态加载并检查 private selectors，绑定 FP16
  或 INT8 IOSurface 与共享 Metal buffer，统计 Metal 实际 allocatedSize。
  named bindings 按编译器的符号顺序映射，并处理 `@output` 名字修饰。
  真实 GPU signal → ANE wait/signal → GPU wait 的 command-buffer 往返已测。
- 私有 source cache 的 SHA256 绑定分帧 MIL/常量、OS build、Metal device 和
  ABI；使用 owner-only 目录和进程文件锁，源文件逐字节校验，拒绝同长度
  损坏。driver 编译缓存缺失才编译，旧编译加载失败仅重编一次。
- ticket/callback 保留 request/model/event/surfaces。缺回调的 deadline
  释放等待、返回失败，保留仍可能被 driver 使用的对象；进程级健康门禁
  防止反复提交并积累隔离资源。它不证明能取消底层 ANE 请求。
- `private/ane_mil.cpp` 原生生成固定 FP16 MatMul/SwiGLU 微图，没有
  checkpoint 常量或任意用户 MIL；支持 K/N tiles、尾块、channel-major
  对齐、runtime gate/up LoRA 修正及 corrected hidden 出口。
- 复用 dense/affine Q4/Q8/raw GGUF/原始 ConvRot/显式 legacy packed ConvRot
  的 checked FP16 row converters。只有有界 row scratch，不生成另一张
  完整 dense 权重。该私有基础 stager 仍为 CPU，不是 GPU Q→W8。
- 私有 SwiGLU 的 `sigmoid` lowering 在 M4 Max 的 signed sparse oracle
  失败：scale=-0.25 时 expected=-0.008784、actual=-0.009216。使用公开
  runtime 已验证的 `exp` 公式修复，原容差不变。
- 真实 Z 512² 初次请求又暴露首层 FP16 输出溢出。接入 up-only headroom：
  不缩放 gate/非线性，up 权重和 up LoRA 修正同缩放，hidden/down 同恢复；
  hidden 与 launch output_dtype 一致（包括 BF16），超出恢复 dtype 仍失败。
  不修改 base/adapter 权重，不 clamp；部分 chunk 失败仍整段 GPU 重算。

## 构建与选路

```sh
TURBOCIDER_ENABLE_PRIVATE_ANE=1 TURBOCIDER_BUILD_LIB_ONLY=1 \
  TURBOCIDER_BUILD_OUTPUT_DIR=build/private-ane \
  TURBOCIDER_BUILD_PYTHON=.venv/bin/python bash tools/native/build.sh
# public-only 用 TURBOCIDER_ENABLE_PRIVATE_ANE=0（默认），输出到另一目录。

TURBOCIDER_ANE_BACKEND=private TURBOCIDER_ALLOW_PRIVATE_ANE=1 \
  <private-enabled-cli> batch <model-root> <explicit-runtime-request.json>
```

仍需既有 `gpu_ane`、`hybrid_mlp_mode=runtime`、manifest 和近似授权；
这里重用 Public template 的几何/ABI，不加载或宣称验证其 Core ML artifact。
private 模型源码来自原生 emitter。选择 backend 不会绕过模型的 LoRA/
分辨率/内存门禁。QKV 研究路由本轮仍是独立 Public 路线。

`tools/validation/runtime_ane_model_screen.py --runtime-backend private|auto`
显式设置授权，并检查真实 backend/Executor telemetry 一致。private 不接受
Public 或 GPU fallback 冒名；auto 不允许同一 resident session 混合 backend
后汇总为一个成绩。普通对照默认 public，并清除继承的 private 实验环境。
最终回执的 private compute_units 为 not-applicable，不伪造 Core ML 配置。
内部 C++ 对象新增 Executor vtable；所有引用这些头文件的 native probes/
clients 必须重新编译，不能把旧 C++ probe 直接链接新库。外部 C API 未变。

Private 源文件只在 build flag=1 时加入编译；flag 进入原生 build identity。
Public worker 文件无 private selectors。Apache-2.0 来源/修改声明和完整
license 一并加入 source 与打包资源，见 THIRD_PARTY_NOTICES。

## 验证与证据边界

M4 Max 64 GB；本次已执行：

- 7 项 ANE host 回归通过；12 项既有 Core ML/MLX 小图/模型 wrapper
  集成通过（含 LoRA、headroom、取消、输出生命周期和 GPU 回退）。
- 新私有 6 项 host/真实硬件测试通过：真实事件往返、A/B/A、跨进程 cached
  load、同长度 source tamper、无 producer 的 timeout、tiled/tail/multichunk
  MatMul/SwiGLU、base/A/base、坏 staging 后拒绝旧权重、fresh refill、
  大 hidden/output 的 BF16 headroom、Public/Private factory 与授权门禁。
  factory-off 二进制 `strings` 不含 `_ANEClient`/`_ANERequest`/shared-event
  private class names；最终完整 public-only 库的七类 private ANE 字符串
  检查、`otool -L` 无 PrivateFrameworks 检查亦通过。
- 45 项整请求 screen 合同测试通过，包括 private 不能冒名 Public、
  backend telemetry 不能不一致、同 session 切换 backend 不得混合计时。
- Z 512 对应 c352/H3840/F10240 的真实私有图 sparse A/B/A 通过，实际
  slots=241336320 bytes；这是模型几何自测，不是真实 checkpoint/画质验收。
- 两个初次整请求记录保留在 `outputs/private-ane-z512-first-screen-20261003/`
  和 `outputs/private-ane-z512-comfy-first-screen-20261003/`。前者使用缺 DiT
  的 Tongyi root，加载失败；后者使用实际 Comfy BF16 root，完整 GPU 请求
  成功，但私有首层 FP16 溢出安全回退。summary 均 incomplete；不能据此
  报告私有性能。headroom 修复后的六组完整请求初筛见下。

## 六组真实请求初筛：均通过，不具备性能交付资格

同一 `7a2f9895d82fb7e654e8d5db94d5a774f6a6b77ee7c90bb2afbbab8f8b23c70c`
库、狐狸/seed42，每 arm 一冷一热、GPU→Private、profile 关闭、chunks=auto。
Qwen base 两组都开启相同 Q/K norm-RoPE 融合；LoRA 六步组不套此开关。
完整 `request_wall` 含 VAE/PNG，不含冷请求：

| 请求 | GPU 热请求 | Private FP16 热请求 | GPU/Private | 热请求 private calls |
| --- | ---: | ---: | ---: | ---: |
| Z base 512² / 8 steps | 6.990245 s | 6.991198 s | 1.000× | 0 |
| Z base 1024² / 8 steps | 31.243715 s | 30.383427 s | 1.028× | 248 |
| Qwen base 512² / 40 steps | 41.688284 s | 42.169277 s | 0.989× | 64 |
| Qwen base 1024² / 40 steps | 183.150991 s | 184.314913 s | 0.994× | 64 |
| Z distill-patch LoRA 512² / 8 steps | 8.541183 s | 8.563518 s | 0.997× | 0 |
| Qwen Viggle r256 LoRA 512² / 6 steps | 8.145100 s | 9.813386 s | 0.830× | 32 |

六组 summary 均 complete；私有自测和真实预测已跑过、没有执行失败或 GPU
失败回退。所有 warm 路线仍是自适应混合：Z 512 的 base/LoRA 热请求完全
decline 到正常 GPU，不能称为热态 ANE 加速；Qwen 的 64/32 calls 也只是
周期复测，不能把 session 累计调用当作热请求覆盖率。
Z 512 base 热 PNG 字节一致；Z 1024 单样本目视主体/构图接近、细节有差异。
这不代替 latent/LPIPS/多 prompt/多 adapter 资格。LoRA 只证明两份现有
adapter 的请求/修正/输出流程跑通、未合入 base；不证明所有训练 LoRA。
没有正反序、多热样本或独立进程树硬内存认证，**全部不满足 1.2×**。

便携身份、raw 结果/PNG SHA、session→request 调用差值和负结果见
[初筛回执](../design/validation/private-ane-foundation-pilot-20261003.json)。
原始目录使用 `outputs/private-ane-<model><size>-headroom-screen-20261003/`
和 `outputs/private-ane-<model>512-lora-first-screen-20261003/`，不覆盖失败记录。

六组结束后修正了 native emitter 的**实际 down-projection 节点数量**边界
（不能按三倍 gate 节点数估计不对称 tiles），增加 32768×1 的拒绝回归。
没有改算术/默认或重标上述历史库成绩。最终重建成功：

- Private：`dba9acbe5b65e5faf87fbe3b218fb2ecdd52c1d0bca229a02604e659321b2126`。
- Public-only：`f7f4f43fff45eb354285bf66442622a753465859297fe5b16ddc74c10e202b4c`。

最终源的 6 private/7 host/45 screen/8 build-identity 检查全部通过，最终
Public-only 库的 12 项 Core ML/MLX 集成也重新通过。未执行完整 `make test`；
已有测试专用构建/fixture 的 skipped 项不能计为覆盖。`git diff --check`
和相关 bash syntax 检查通过。没有尚在运行的 benchmark，也未 stage/commit。

所有测试/benchmark 只写隔离 build/临时缓存/被忽略 outputs，不替换
发行 `build/native`，不改模型、参考仓库或现有 ConvRot 实验产物。

## 必须继续，不算完成

1. 私有基础路径修复后的 Z/Qwen 两种分辨率完整生成、匹配 GPU 对照和
   LoRA 实机/视觉/latent **正式资格**；六组 pilot 不代替多提示词、至少
   正反序和足够热样本，1024² LoRA/编辑也未扩展既有门禁。
2. 真正的 W8A8 representation、Hadamard/A8/scale oracle、私有 FFN 图
   与分段 arithmetic；不能以 INT8 surface 分配或 QDQ 图代替它。
3. 原 GGUF/Q4 GPU decode+rotate+requantize 直填 W8、固定两套 staging
   slots/依赖调度、下一层预取与当前 GPU/ANE 重叠，以及内存/带宽校准。
4. MLX producer/consumer 的真正 GPU device-side handoff：本次私有 GPU
   事件往返是真实的，但模型输入/输出仍经过 host transpose/copy。
5. channel split/physical stride/views/fallback 合同；共享抽象的 QKV 与
   request/App 级 backend 配置；多设备和低内存验收。
6. 两模型 512²/1024² 相对同构建、同输入、同精度纯 GPU 的 ≥1.2×
   端到端目标。没有四格完整证据，不标 goal complete 或升级默认。

下一实施顺序由上述实际负结果决定：补 GPU input/output handoff（目前
channel-major 的 CPU transpose/restore），再连通 W8A8/Hadamard/规范化
scale epilogue 与 GPU Q4/GGUF stager、固定两套 slots；统一调度重新校准。
真实 Z 的 headroom 达16/64，W8A8 不能在 ANE 内先把完整输出恢复到 FP16
再声称安全：需规范化出口/独立 scales 和 BF16/FP32 epilogue 的单独验证。
转换成功、INT8 surface 或单算子更快都不能替代原任务的四格 ≥1.2×。
