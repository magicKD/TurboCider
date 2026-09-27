# 11 · 分批实施任务、接口交付与发布顺序

[目录](README.md) · [逐文件迁移](06-migration.md) · [编译规格](09-layout-compiler-spec.md) · [验收执行书](12-acceptance-playbook.md)

状态：规划，不是已完成清单。F0–F9 与 06 一致；本文件补足每批可交付物、依赖和停止条件。
建议先交付 LTX 的一个 exact tuple，再增加模型/参数覆盖，不以“大重构一次完成”作为前置条件。

## 1. 最小垂直切片与范围

第一条端到端候选：LTX native GPU、单一明确 checkpoint/shape/dtype、G=1、P≥1、K=2 或 3、
Q 与 baseline 显式对齐、request retention、guard off、固定 pass，无 ANE/LoRA/未验证输入分支。
实际 tuple 由 metadata 和现有 benchmark workload 决定并写入 fixture，不用本文示例 P=1 当性能推荐。

最小切片包含 parse→describe→compile→fake execute→真实执行→quality/performance→layout registry。
它不包含 bounded-memory release；后者必须再完成 F5 与 L2/L3。首批仍可通过 tooling 展示未认证布局的资源需求。

## 2. 每个 PR 必须交付的共同内容

1. 变更前后的模块边界与唯一职责，新增/修改文件清单。
2. supported/unsupported 参数和错误码；新字段明确 presence 行为。
3. 对应测试 ID、原始日志、命令、build identity；skip 不写成 pass。
4. legacy route 无新增行为的断言；触及 kernel helper 则附旧输出对照。
5. 不改变生产资格的说明，或附独立 release-review artifact。
6. 本目录当前状态更新；未实现工具/示例继续标注 proposed。

不把“测试调用成功”“hook已安装”“trace已生成”当整个模型接入完成。

## 3. F0：schema、presence 与路由隔离

修改：`native/core/contracts.hpp`、拟增 `streaming_contracts.hpp`、`request.mm`、`profile.mm`、`results.mm`、`plan.cpp`。

- 新 config 值类型、明确的 `specified` 与 enabled，旧 residency/memory_budget/offload 增 presence。
- parser/profile merge 按 02；严格重复键检查必须发生在 JSON dictionary 吞掉重复键之前。
  若当前解析库不保留重复键，增加 raw JSON token 检查或可拒重复的 parser 入口，不能只遍历 NSDictionary 假称已覆盖。
- 先选择新/旧 route，再走对应 normalize；新 route 仍执行 shape/dtype/operation 等共同校验。
- profile v1 golden 和默认 results 不改变；v2 reader 单独测试。
- 结果添加显式 `plan_only/unsupported`，但此时执行一律拒绝，不偷偷走 legacy。

验收：配置矩阵全覆盖；旧 guard-only route 不变；显式 streaming=false 不创建 context；无认证时拒绝发生在 unload 前。
停止条件：无法区分默认 residency 与用户显式 residency 时，不合入 manual 执行入口。

## 4. F1：metadata 和纯 compiler

新增：`descriptor.*`、`layout.*`、`registry.*` 的空生产 registry，以及拟议 native plan utility。
session 增 describe 默认 unsupported；不得在默认 load/generate 上调用。

- 按 09 实现 C0–C8、canonical serialization、资源与错误报告。
- 先手写 tiny descriptor/fake metadata reader，再接 LTX header；禁止 load block0 取得 shape。
- golden fixture 与异构 field-capacity 反例必须独立手算验证。
- 明确 stage→recipe/pass 映射，别把 LTX denoiser 参数复制成两个互不关联的 pool。
- F1 PR 冻结 control-plane limits 和错误码；exact plan 不取决于机器当前可用内存。

验收：同输入 digest 稳定；Y 改变不改变 layout；metadata malformed/TOCTOU/整数溢出拒绝；无 GPU 分配。
生产执行资格仍为空。工具能够解释“为什么拒绝”比能打印一个总 bytes 更重要。

## 5. F2：slot runtime 与 fake backend

新增：`context.*`、`slot_pool.*`、`io_executor.*`、`use_fence.*`、`native/core/stream_slot_c.h`。

- C ABI 带 struct_size/version，头文件同时通过 C11/C++ 编译；错误固定容量，异常止于 C++ 边界。
- 依 10 实现 owner pump、pending/in-flight credits、persistent workers、mailbox/latch、generation。
- fake backend 独立读取 backing、延迟 completion，不能仅回显 executor 的 ticket。
- error injection 覆盖每次 allocate/read/submit/thread-create；取消在每个状态可触发。
- 稳态事件计数/低成本 counters 与完整 trace 分开；默认 trace 不开。

验收：无 early overwrite/use-after-free/deadlock；2/3 槽乱序完成和 pass wrap 通过；控制内存不随 steps 增长。
ASan/UBSan/TSan 在可用 host 测试；sanitizer 不支持平台标 skip 并补支持环境，不能以此授予 GPU 资格。

## 6. F3：LTX adapter 的四个小 PR

### F3a · metadata 与 exact construction

在 `ltx_blocks.c` 当前 `ltx_native_create()`/`load_block_weights()` 周围增加新分支；保留旧构造流程。
提取 weight-layout 计算与 backing allocation/fill 的窄 helper，优先留在原文件内，不先拆整份大文件。
新 adapter 可先是原文件的内部入口，待接口稳定再迁到 `ltx_streaming_adapter.c`，避免为访问 static helper 复制 kernel。
owner 按 exact plan 分配 prefix/slots；新分支不调用预算启发式；返回 actual layout 供 framework 对比。

### F3b · fill 与 derived binding

将新分支首次 slot allocation 移到 owner，worker 只填既有 spans；旧 loader 不改。
复用 ranged reader，但审计 mmap、CPU base_values、row buffers、scratch、失败释放。
把 `apply_scalar_conditioning_one()` 接到明确的 owner prepare_group；证明派生数据不会污染另一 step。
新分支不使用旧 per-block pthread fallback，线程失败精确报错。

### F3c · compute 和 completion

复用 `run_block()` 数值实现，新增 executor wrapper；检查 `ltx_gpu.m` batching、video/audio queues 与 last reader。
如需新 completion API，只给新分支使用，保留默认 batch/commit 频率。
低成本 synthetic GPU buffer 测试先发现早复写，再跑真实 checkpoint parity。

### F3d · 完整请求生命周期

修改 `native/platform/apple/ltx_session.mm` 连接两 stage、upsampler、latent handoff 和 VAE；确保 denoiser backing 的释放真实完成。
G=1/P≥1 先限制；K=1/2/3 是否可运行由每个具体候选验证，不因接口接受整数就全部发布。
验收至少覆盖 two-stage shape 变化、cancel、第二次请求、stage2 cleanup；未覆盖 audio/I2V 明确拒绝。

## 7. F4：H3 adapter 的接入边界

新增 `h3_streaming_adapter.c` 或内部窄桥；涉及 `h3_dit.c`、`h3_gpu.m`、`h3.c`、`h3_session.mm`。

- 先 exact K=2/G=1；记录 `stream_ready_slot ^ 1u` 的现有语义，new wrapper 映射到 framework slot。
- 现有 norms/AdaLN 常驻部分不放入 refill capacity；访问/释放仍进 descriptor。
- 审计 next_streamed_block、step gate、first-block cache、跨 block fusion；不能表达就限制候选，不能悄悄关闭优化。
- 前向开始时的 ready layer 和跨 forward 预读按 explicit pass init 处理，避免隐藏 I/O 不计入 plan。
- text/DiT/VAE/export 语义要覆盖，但 bounded 未闭包前仅 layout-only 实验。

单槽不是本批任务。F7 再做 K=1 的独立 init/refill/final/cancel 分支；去掉 xor 不代表自动支持任意 K。

## 8. F5：可选 guard 组合与整请求闭包

主要修改 `native/api/c_api.mm`、现有 memory manifest/plan/execution bridge、对应 session allocator hooks。

- 三次检查区分：metadata/layout eligibility → memory certification → clean baseline 后 same-layout admission。
- preflight 拒绝不得先 unload 旧 session；admission 前确需 clean boundary 时，该代价计入 bounded wall。
- 建立 backing→site→instance→epoch→release 映射；content Vacant 不触发 ledger release。
- 外层 text/conditioning/upsample/VAE/output/control、driver/framework envelope 全覆盖。
- 新 memory event codec 如需 v2，单独兼容 PR，不能破坏旧 v1 tests。
- strict sequence owner 唯一 producer；guard=false 仍执行 slot safety，但不建立“无限预算 memory context”。

完成门：所有 required site 闭包、no unknown upper、Y/X/S 的整数边界、actual cap、取消/drain/terminal；
通过这些逻辑测试仍只代表准备好做 bounded 实机 campaign，不直接注册 production record。

## 9. F6：性能与质量认证，决定是否发布

使用 12 的 campaign：legacy before/after、new same-layout、guard overhead、真实低内存四类结果分开。
必须包含正常目标 workload，tiny smoke 只用于快速排错。每个发布 tuple 有完整 evidence identity。
有 quality/lifetime/default-performance 证据可评 layout_validated；bounded 另外满足 L2/L3。
任何 FAIL 不发布；INCONCLUSIVE 补证据；无设备 SKIP 不通过。

只有本批完成才增加对应 registry record。撤记录仅影响显式新 route，不改默认 resident/旧 streamed。

## 10. F7–F9：后续扩展而非首发债务

| 阶段 | 拆分任务 | 新增风险/独立验收 |
|---|---|---|
| F7 | H3 K1；LTX P0；G>1；Q调优；range合并 | metadata解耦、fusion边界、解包/缓存竞争，每项独立 revision |
| F8a | Flux/Z-Image metadata + component staged | graph 引用、compiled cache、text handoff、真实释放 |
| F8b | ranged reader + block/group streaming | 格式-specific unpack、异构 pool、binding cache 不捕获旧 weights |
| F9a | inspect/plan/simulate/sweep/preset | 推荐仅列已认证 tuple；离线优化不自动改请求 |
| F9b | session retention | root ledger、generation、跨请求所有权、cache eviction、warm ABBA |

先做 F9a 的 inspect/plan 工具底座以支持 F1–F6；自动搜索与发布 preset 仍是 F9，避免把工具基础推迟到最后。
Flux/Z-Image 的 default compiled/eager 路线不受接入进度影响。

## 11. 集成、构建与服务

- `tools/native/build.sh`/Makefile 在各批加入对应源和测试；CPU compiler tests 不链接整个模型 runtime。
- `tests/native/test_contract.py` 保留默认 golden，新增 config cases；不把计划中新字段加进旧成功 fixture。
- service 只在新 route 将 layout/backend/retention identity 纳入 session key；默认 key 与 GPU job 串行行为不变。
- 失败清理后检查 worker 可复用条件；quarantined 不能放回 idle pool。
- request/profile reload 不修改活跃 plan；未知 preset 不回退 resident。
- 不在本任务中更改公共 C 二进制 ABI；若确需扩展，另行 versioned 结构体与兼容测试。

## 12. Review 风险与停止条件

| 风险信号 | 必须采取的动作 |
|---|---|
| 默认路径出现新的每层分支/同步 | 拆出新 wrapper；默认性能 gate 暂停 |
| worker 仍然首次分配 GPU slot | F3 不完成，先分离 allocation/fill |
| callback 直接调用 ledger | 移到 owner mailbox；重跑 race/fault tests |
| P/K 被下层 heuristic 改写 | exact ABI 不合格，禁止发布 |
| mmap/MLX graph 让实际 backing 不释放 | 调整资源合同或明确 unsupported，不修数字掩盖 |
| 速度变快但输出差异/省算 block | quality FAIL，不计加速 |
| Q/P/retention 不同却标 same-layout | 重分类实验，重新补同布局对照 |
| guard upper 仅覆盖 denoiser | 不得 bounded_certified |

建议每批先通过 host tests 再消耗真实模型 GPU 时间；不设没有测量依据的完成百分比或日期承诺。
