# ane_pmu_profiler：先验 payload，再解释寄存器

[来源目录](../README.md) · [PMU 实验合同](../../profiling/ane-pmu-lab.md) · [全部本机证据](../../review/ane-pmu-20261016.md) · [关联 capacity 实现](ane-capacity-pmu.md)

固定 `freedomtan/ane_pmu_profiler@20080a945096eb5932490b024120b513ae5427e2`，2026-10-16，原 tracked worktree 干净；不声称最新版本。README316行、`dump_ane_pmu.m`1162行、`coreai_loader.m`496行、header138行与技术报告836行均已读；另读小型 compiler/ODIE C pipeline、headers、Swift bridge、Makefile和entitlements。SHA/范围见[唯一登记](../../references/sources.json)。

## 可学习的接口与实际风险

`_ANEModel` 按 MIL/Espresso/ANECIR/HWX 等入口加载，`_ANERequest` 绑定输入/输出和 `statType=2` 的 IOSurface，load/evaluate 用 `kANEFPerformanceStatsMask`。上游以 `perfCounterData` 的29个 uint64 为预期布局，用 `stringForPerfCounter:` 获取名称。这是固定版本的私有协议研究，不是 Apple 公共规格。

| 已读位置 | 源码事实 | 使用边界 |
| --- | --- | --- |
| `checkAndPrintDriverStatus` | 无条件打印 PMU ACTIVE、返回 YES | 既没有推理，也没有验证 payload；本机 missing-model 控制复现了假阳性 |
| `runLiveInferenceAndCapturePmu` 的 ANECIR 分支 | `client` 初值 nil，分支未赋值就调用 load | 不能用 README 的 user-space 声明证明这个分支能运行 |
| MIL/CoreML/Espresso 分支 | compile/load/evaluate 有 BOOL 检查 | `main` 即使失败仍可返回0；退出码不是资格 |
| `gLiveHwTimeNs` | 实际为多次 host evaluate wall 的平均值 | 后续“silicon latency”、GHz、TOPS不能获得物理计时身份 |
| warmup/final counter 差分 | 先假设累计，下降时夹成0 | request-local、reset、wrap 都可能产生不同解释，必须先验证生命周期 |
| 完整表按 counter name 查 dictionary | 未知下标可能共享同一个名称 | 必须保留下标和独立 raw word，不能合并 `[24..28]` |
| surface sizing / input / output | 有缺省大小；没有独立数值 oracle | 成功调用不覆盖实际 pitch、完整输出或真实 workload |

`coreai_loader` 会将 shape 数据求积并按最大 tensor 估 bytes，找 manifest 的第一个 region/arch，可能读可访问的系统 aned cache；prepare/JIT 还会删除模型目录下已有的 `output_host_jit` 或 `ane_bundle`。MPSGraph warmup 用 async 后立刻找临时 bundle，缺明确完成见证。它不是任意用户模型目录上的只读 inspector。本轮未走这些路径，未加载附带模型或 Apple Intelligence 资产。

## 文档、ABI 与本机观察分层

技术报告含直接 kernel 路径的历史安全降级操作，后段又给出 daemon-proxy 修订。这里不执行或推荐安全降级；README 的“无需 root”仍需在特定 OS/API/workload 实际验证。报告说明其反汇编来源包含另一系统版本，地址/结构不能直接借给本机 macOS26.6.2。

本机只读 selector metadata 显示 `initWithRequestPerformanceBuffer:statsBufferSize:` 的参数是 `^^v` / `^I`，不能把任意 IOSurface base pointer 当它的参数强转。`driverMaskForANEFMask:` 在当前 framework 中的真实只读返回为0→0、2→4、15→15；mask 翻译成功仍不代表有 counter payload。

实际执行：只读原源码编译到 ignored 输出；help、missing-model、一个自有小 conv MIL，以及仅普通 `get-task-allow` 的独立 debug-signature 控制均运行。valid MIL 能执行，但两种签名都报告没有 stats object，原 CLI仍返回0。原创客户端另外对 affine/conv、mask0/2/15、较大 workload、compile-profiling 选项及重复进程作控制：81次完整输出通过，未取得 PMU payload，详见[报告](../../review/ane-pmu-20261016.md)。这是本路径未复现，不是宣称 M4 不具备 PMU。

许可边界：该固定 tree 未找到 LICENSE/COPYING 授权文件，所读源文件也未见对应许可声明。本轮只读研究和本地构建，不将其代码、模型或二进制打包进 Wiki；本地客户端/decoder/oracle为原创。关联 capacity 项目的 BSD-3-Clause 许可不能倒推覆盖本仓库或 Apple 私有接口。

[固定采集器](https://github.com/freedomtan/ane_pmu_profiler/blob/20080a945096eb5932490b024120b513ae5427e2/dump_ane_pmu.m) · [固定 loader](https://github.com/freedomtan/ane_pmu_profiler/blob/20080a945096eb5932490b024120b513ae5427e2/coreai_loader.m) · [技术报告](https://github.com/freedomtan/ane_pmu_profiler/blob/20080a945096eb5932490b024120b513ae5427e2/ANE_Performance_PMU_Technical_Report.md)。
