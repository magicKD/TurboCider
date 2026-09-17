# 16 · 性能工具链实施与可执行验收路线

[目录](README.md) · [实验原则](07-tooling-and-validation.md) · [发布阈值](12-acceptance-playbook.md) · [整合方案](14-layout-first-integration.md)

日期：2026-09-16。本文细化工具接口、仿真模型、诊断和测试映射；不改变12的P0–P4阈值。
T2/T3 的最小可执行实现已经落地为 `tools/native/run_streaming_campaign.py`、
`tools/native/verify_streaming_campaign.py` 和 `tools/native/capture_streaming_source_identity.py`；
T0/T1/T4 仍未完成。没有真实模型 ABBA 数据和完整 audit/environment provenance，不能声明本框架已达成无回退或快于 swap。

## 1. 性能目标不是一句“不能差”

| 场景 | 必须保持/证明 | 工程约束 |
|---|---|---|
| 新开关关闭、内存充足 | P0：改动前后默认resident非劣 | 一次入口分流，内循环零新增工作 |
| 新开关关闭、旧streamed | P0：旧pager功能与性能非劣 | 不顺手重写旧loader或改变cache |
| 新框架同布局 | P1：统一执行器成本非劣 | 持久pool/workers、固定队列、保留batch |
| 同布局加guard | P2：守卫增量成本和完整wall透明 | 安全检测不能为通过性能而关掉 |
| 真实低内存 | P3：同质量/环境下是否好于OS-managed | 不以大RAM软件预算替代物理低内存 |
| 改槽/前缀/并发 | P4：配置变化带来的实际收益 | 不冒充框架改造收益 |

目标是默认无可检测回退；12中的2%/5%等是带置信区间的阻断线，不是本次实测成绩，也不是允许随意消耗的预算。
任一正式workload失败都阻断该范围发布，低内存加速不抵消默认回归。

## 2. Streaming 为什么可能快于 swap，又为什么不保证

要比较的是两种完整执行方式，不是“read比swap快”这样的单一指令差异。
显式streaming知道未来访问序列，可以控制同时存活的weight数量、提前读下一组、在reuse边界丢弃可重建内容。
它可能避免anon dirty页反复换出/换入和不可预测fault；但仍支付checkpoint读取、转换、绑定与同步成本。

需要通过实测区分的反例：

- resident工作集完全放得下：反复读权重通常是额外成本，不应默认切streamed。
- checkpoint冷读慢/转换贵：K再多也不能补足持续吞吐；增加P或更合适的文件格式才可能有用。
- GPU与CPU读/转换共享带宽：Q提高可能让compute变慢；不能仅最大化overlap百分比。
- activation/VAE才是峰值：把K降到1仍可能不足，slot框架不是所有内存问题的解法。
- OS主要回收clean file-backed页：baseline不一定产生swap写；不能按模型超RAM的字节数虚构swap量。

因此预先接受三种实验结果：更快、差不多、更慢。只发布证据支持的preset；保持默认路线是高内存性能的重要保障。
本段为待检验的性能机制分析，不是TurboCider已经获得的加速结论。

## 3. 离线工具最小实现，不先造调优平台

建议新增 `tools/native/streaming_tool.py` 做编排；一个小native utility导出describe/compile结果。
compiler只存在C++一份，Python不复制layout算法；独立verifier只实现简单不变量/参考算例，避免同源错误自证。

| 工具阶段 | 新文件建议 | 输入/输出 | 实施依赖 |
|---|---|---|---|
| T0 | native plan utility + Python CLI | config+metadata→descriptor/resolved/reject | F1 metadata；可先fake fixtures |
| T1 | `streaming_simulator.py` | resolved+calibration→timeline/uncertainty | T0，独立reference cases |
| T2 | `streaming_benchmark.py` | 固定campaign→原始样本/evidence | 真实adapter，合法执行资格/test build |
| T3 | `streaming_verify.py` | evidence→各gate判定/缺失证据 | T0–T2 schema固定 |
| T4 | preset exporter | 有证据候选→展开后的配置和sidecar | 独立确认集/人工签核 |

工具应返回结构化状态：plan_valid、execution_unsupported、budget_rejected、measurement_failed等；
非零进程退出码不能替代细分状态，SKIP也不能仅因脚本exit0就被CI标PASS。
输出目录新建且拒绝意外覆盖，保留完整失败记录；不要把模型文件、机器私密路径或大型trace写入源码目录。

以下为拟议CLI，当前不要作为可运行命令复制执行：

```text
streaming_tool.py inspect --model MODEL --checkpoint PATH --workload workload.json
streaming_tool.py plan --request request.json --descriptor descriptor.json --output NEW_DIR
streaming_tool.py simulate --plan resolved-plan.json --calibration calibration.json --seed 17
streaming_tool.py sweep --candidates candidates.json --simulate-only --max-candidates 12
streaming_tool.py benchmark --policy campaign-policy.json --execute-gpu --output NEW_DIR
streaming_tool.py verify --bundle NEW_DIR --policy campaign-policy.json
```

inspect metadata模式不做GPU分配；真实内容hash或性能profile必须显式请求并记录成本。
sweep不默认运行GPU；benchmark也不隐式启动pressure或改swap。

## 4. 仿真器的状态与事件

### 4.1 状态表

- immutable plan：group→slot、每field容量、P/G/K/D/Q、pass访问模板、文件ranges、reader revision。
- slot state：Vacant/Loading/Ready/InUse、generation、各queue最后reader、sealed状态。
- 资源：Q个fill服务资源、owner prepare资源、GPU queue/submission credits、disk/CPU/shared-bandwidth模型。
- 外层阶段：prefix、text/upsample/VAE/输出的存活集合和依赖；未校准阶段标unknown，不能当0。
- 观测：logical read、physical read估计、cache model、峰值与每类关键路径等待。

仿真首先必须遵循安全约束，之后才预测速度；不得为了漂亮timeline让同槽CPU写与GPU读重叠。

### 4.2 事件推进

事件队列按 `(simulated_time, deterministic_priority, sequence)` 排序。包含read/decode完成、owner准备完成、GPU最后reader、pass barrier。
事件到达后用与runtime相同的窗口/credit约束推进，但不直接调用生产pump来当独立安全验证器。
同时间ready/fence事件的不同合法顺序不得改变是否early-overwrite；为此另加permutation测试。

owner同时只能执行一项prepare/encode工作；fill worker不能无成本无限并行转换。
prepare写row buffer的耗时与I/O时间分开，不能把owner-only工作虚构成在worker里提前完成。
CPU提交时间与GPU完成时间不同，D相对“下一尚未提交group”变化，不能只看GPU完成下标。

### 4.3 独立参考模型

先验证最小Q=1、单GPU queue、无prefix、无竞争、fence零延迟的模型：

```text
start_load[i] = max(previous_load_end,
                    last_reader_end_of_previous_content_in_this_slot,
                    lookahead_eligibility_time[i])
end_load[i] = start_load[i] + read_decode_time[i]
start_compute[i] = max(previous_compute_end, end_load[i] + owner_prepare_time[i])
end_compute[i] = start_compute[i] + compute_time[i]
```

该式只作简化oracle；多worker、owner争用、多queue用离散事件仿真，不能机械套公式。
submit credit、真正last-reader latency、startup/pass drain在扩展模型中显式添加。

手算fixture（全是合成毫秒，不是模型性能）：6组、每组L=8/C=12，K1时间=120；理想K2/D1=80；无抖动时K3仍=80。
若L=18/C=12，K1=180，理想K2=120，说明更多slot不能突破慢读吞吐。首发每pass drain时不得把两个pass当一条无边界流水线。
这些算例用于测仿真器是否正确，不用于承诺低内存机器快多少。

## 5. 校准与避免“仿真自嗨”

| 参数 | 采集方式 | 不能怎么测 |
|---|---|---|
| read/decode分布 | 对同格式/ranges计时，cold/warm/unknown分层 | 用空文件memcpy带宽代替真实checkpoint |
| prepare/bind | owner区间单独trace | 合入GPU compute时间 |
| compute | backend真实GPU时间/完整完成边界 | host函数返回时间当GPU执行 |
| fence/dispatch delay | callback到达、owner消费、再次派发时间 | 所有completion设0后声称无气泡 |
| Q/K竞争 | 在允许的并发组合下实测服务曲线 | 假设Q倍线程=Q倍磁盘吞吐 |
| file cache/pressure | 标注scope与协议，真实环境校准 | 将软件Y当OS物理容量 |

先建立无竞争下界，再增加竞争/长尾敏感性；没有硬件数据的参数必须显式标synthetic。
仿真器给候选排序、误差带、暴露的I/O等待及峰值构成，不给production认证。
训练/校准trace与验证trace分开；在未参与拟合的shape/layout检查wall误差和排名。
若预测不稳定，输出“无法可靠排序”，让用户看Pareto候选，不强行输出唯一最佳K。

Swap近似仅是独立的trace-driven cache/paging敏感性模型，区分页类别和压缩成本，不宣称实现macOS pager。
若没有实际换页观测，P3结果保持NOT MEASURED；不能往baseline上加一个任意常数惩罚来制造speedup。

## 6. 性能瓶颈到代码行动的映射

| 观测到的瓶颈 | 首先核查 | 可以尝试的新候选 | 不应直接做 |
|---|---|---|---|
| 大量slot reuse wait | reader是否覆盖了无关工作/commit过晚 | 更精确last-reader、显式K候选 | 提前复写或忽略audio queue |
| ready等很久、GPU空闲 | 真实read/decode慢还是worker未唤醒 | ranges合并、reader实现、P候选 | 默认加Q到最大 |
| ready很多、compute变慢 | 带宽竞争、额外commit/失去fusion | 降Q、保留group内batch | 按overlap最高选preset |
| 每pass setup很贵 | 是否重建pool/thread/重新hash | persistent生命周期 | 从wall删除setup |
| metadata很慢 | 重复parse/无必要full hash | request内共享immutable view | 默认路径引入全局cache |
| 高峰不在denoiser | text/VAE/upsample live set | 组件staged、独立已批准tiling | 反复减K掩盖非slot峰值 |
| 新默认变慢 | route、probe、cache、编译/链接差异 | 回退侵入性公共改动 | 用P3加速抵消P0 |

一项优化先独立revision、单因素实验，再进入正式配对确认。trace build仅诊断；release timing两侧观测开销必须匹配。

## 7. 实验卡片与artifact合同

复用12的 `campaign-policy.json` 和现有 [performance-policy.json](examples/performance-policy.json)，不要增加另一套阈值文件。
完整campaign需要先填以下字段，示例policy本身并非可运行campaign：

```text
comparison_kind: P0 | P1 | P2 | P3 | P4
baseline/candidate: build identity + source manifest + config hash
workload: model/checkpoint/format/shape/steps/seed/output/audio/input branches
layout: requested/resolved/actual hashes; P/K/G/D/Q/pass/retention/reader revisions
protocol: cache condition / warmup / ABBA blocks / sample ceiling / timeout
quality: registered metric + tolerance + reference identity
statistics: estimator / interval / resampling / multiplicity / stopping rule
environment: GPU/RAM/SSD/OS/SDK/power/thermal/pressure observation
safety: execution authority / resource limits / cleanup and stop rules
```

new same-layout无法匹配旧Q/lookahead/跨pass预读时，不伪造P1：记录semantic mismatch，将该对比降为P4并补真正P1方案。
只看K相同不够。retained old对request-scoped new的比较有用户意义，但不是纯调度器开销。

原始样本增加failure/timeout/cancel/cleanup状态、load/denoise/VAE/drain时间、logical/physical/swap bytes、worker创建次数、pool/buffer次数。
pool bundle与底层多个MTLBuffer区别计数；logical bytes含重读，不能混成unique checkpoint bytes。
OOM/timeout保留为失败样本；不能仅对幸存成功样本算加速再隐去完成率。

## 8. 现有可运行检查与新增测试清单

已有host入口（从仓库根目录执行；不是完整性能验收）：

```sh
python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread python3 -B tests/native/test_streaming_layout.py
```

脚本现包含compiler/executor/C11 bridge；C11测试对象现已加入与C++相同的sanitizer编译flags。
这只覆盖测试中编译和运行的代码，不能扩大为整个Metal/model库的完整sanitizer覆盖；范围见13。
真实Metal检查已有 `tests/native/test_streaming_metal.py`，需要设备访问权限；沙箱SKIP不等于实体无GPU。

| 待补测试 | 原gate | 必测行为 |
|---|---|---|
| persistent lifecycle矩阵 | RUN-04 | 分开调用多pass、不同step、一次建池、early finish/重复finish拒绝 |
| bridge ABI与所有权 | FLT-01/02 | 坏version/size、partial alloc、错误线程、空/重复destroy、异步sink复制 |
| 非协作fill/drain | FLT-02 | 不能因超时释放活资源；worker隔离不复用 |
| reader/source descriptor | CMP-02 | 多source副本、alias、转换scratch、range越界、文件更换 |
| steady-state allocator audit | PERF-01 | 已有setup后allocation/thread counter；LTX单pool实模型为0/0，multi-pool切换当前会非零并阻断签核 |
| LTX/H3真实adapter | GPU-01 | latent/output parity、最后reader、正常shape/cleanup |
| 默认instrumentation audit | DEF-01 | 新hook/probe/clear/unload全部0，legacy结果不变 |
| simulator references | L0 | 手算K1/2/3、乱序fence、D0、长尾、class/pass边界 |
| bundle verifier反例 | PERF-01/MEM-02 | 缺raw数据/错hash/错quality/semantic mismatch不能PASS |

当前测试结果精确范围和未测项统一记 [13](13-implementation-progress.md)，不要在多份规格中复制会过期的PASS表。

## 9. 本机验收顺序与停点

1. **安全快检**：host、C bridge、sanitizer，验证配置与状态机；不做系统压力。
2. **真实设备最小用例**：synthetic Metal → 一个LTX tiny → normal-target；任何quality/lifetime失败先停。
3. **默认隔离**：source/audit检查，P0 before/after准备可信构建。没有可信before就不做“无回退”结论。
4. **布局同义对照**：固定P/K/G/D/Q/格式/retention，跑P1。若不等价先处理合同，不能借更大prefix过关。
5. **候选探索**：有限P4矩阵与仿真，选Pareto配置；独立确认集验证，避免挑样本最优。
6. **预算扩展**：F5闭包后跑P2/完整L2；然后专用低内存设备或明确授权的有限pressure做L3/P3。
7. **独立验收**：verifier汇总PASS/FAIL/SKIP/INCONCLUSIVE，review签具体tuple；默认仍不自动启用新route。

全系统pressure不是上述普通测试隐含步骤。需要操作者明确批准资源上限、最长时间、磁盘余量与停止规则；不关swap、不清用户缓存、不运行无界分配器。
无低内存设备时可交layout-only证据和simulation，但bounded/L3/P3保持未完成。

## 10. 发布评审必须能回答的问题

- 框架默认关闭时，实际P0结果在哪里，是否同时覆盖resident和旧streamed？
- 新实现和旧实现到底哪些参数/生命周期相同，P1是否真的同义？
- 发生的read、decode、compute、fence等待能否解释wall，而不只给一个speedup？
- 增槽后峰值/实际吞吐是否都改善？没有改善为什么仍推荐？
- 内存上限覆盖的是slot、纳管allocator还是完整请求？未知项是否仍然存在？
- 对swap的结论是实测还是敏感性假设？失败率、cache、压力条件是否完整披露？
- 未认证模型/shape/格式如何明确拒绝，如何撤回资格且不伤害legacy？

这些问题全部有artifact支撑，才算“框架清晰、可实现、可验收”，而不是只有可调的slot参数。

## 11. 已实现的 T2/T3 campaign 工具

### 11.1 runner 的执行边界

`run_streaming_campaign.py` 是协调器，不在主进程加载 native dylib。它为 baseline、candidate
各启动一个独立且持久的 worker；worker 在启动时构造一个 retained engine，随后接受长度前缀的
本地 socket 命令。每次 measured request 返回后，runner 立即将一条 JSON 写入并 `fsync` 到
`raw-samples.jsonl`，因此超时、worker 错误和后续未运行位置都不会从证据中消失。

固定执行序列为交替 `ABBA` / `BAAB` 四位置 block：位置 0/1 组成 pair-0，位置 2/3 组成
pair-1。runner 不并发提交两个 variant，也不在 block 内重排；这样 verifier 可以按整个 block
而非单个 denoise step 做 bootstrap。

runner 明确不会关闭 swap、清 OS cache、启动 pressure、修改电源/thermal 设置或自动改变
request 的 slot/budget。warmup 失败会保留 `warmups.jsonl`，并把完整计划位置写为
`not_run_after_abort`；GPU 不可用时不会伪造零样本 PASS。

### 11.2 evidence bundle

一次执行输出以下文件：

```text
campaign-policy.json       # runner 实际使用的冻结 policy 副本
manifest.json              # policy hash、状态、文件 hash、计划/实际数量
build-identity.json        # binary hash、构造器、source identity
raw-samples.jsonl          # 每个 measured position 一条，包含失败/timeout
warmups.jsonl              # warmup 原始记录，不计入性能统计
quality.json               # pair 级 artifact/latent 质量结果
faults.json                # raw 与 warmup 的失败镜像
environment.json           # 自动平台记录或 operator-supplied 完整环境
audit.json                 # release默认partial；独立audit build可按请求reset/snapshot后覆盖
semantic-equivalence.json  # P1 必须声明且观察到同布局
summary.json               # 独立 verifier 的最终状态
workers/*.log              # 两个 worker 的 stdout/stderr
```

`capture_streaming_source_identity.py` 对 Git worktree 记录 commit、tracked manifest、dirty diff
和 untracked source hash；对无 `.git` 的导出树要求显式 40-hex commit，并计算稳定 source
manifest。verifier 对 P0 拒绝缺失 source manifest 的 bundle，即使时间比值很好也只能
`INCONCLUSIVE`。

### 11.3 verifier 的硬规则

- policy hash、manifest 文件 hash、baseline/candidate binary identity 必须一致；P0 binary 必须不同。
- raw 必须含完整四位置 ABBA/BAAB block，pair 不能跨 block；resolved/actual layout 不一致直接拒绝。
- P1 必须有 `semantic-equivalence.json`，并观察到每个成功 pair 的 actual layout 相同。
- bootstrap 以完整 block 为重采样单元；不是按 pair 或 denoise step 独立重采样。
- `audit.status=partial`、`environment.status=partial`、source provenance 不完整、样本不足或
  campaign 未完整结束时，最多 `INCONCLUSIVE`；不自动降级为 PASS。
- 失败/timeout/cancelled 且仍有成功样本时为硬失败；若整个设备在 warmup 即不可用，bundle
  保留但可归为 `INCONCLUSIVE`，并显示 completion rate=0。
- P0/P1 统计阈值均严格受第12章 median 1.02、P95 1.05 上限约束。

CPU-only synthetic backend 与 verifier 反例测试位于
`tests/native/test_streaming_campaign_verifier.py`；source identity 测试位于
`tests/native/test_streaming_source_identity.py`。入口是：

```sh
make PYTHON=python3 test-streaming-campaign
make PYTHON=python3 test-streaming-source-identity
```

这两个目标验证工具合同，不授予任何真实模型或硬件的性能资格。

### 11.4 audit build 与 release campaign 分离

`TURBOCIDER_BUILD_AUDIT_COUNTERS=1` 只用于路由隔离证据。它启用五类原子计数和私有
`tc_streaming_audit_*` ABI；release build不导出这些符号，所有counter callsite编译为空操作。不要用audit
build替代最终release timing：正确流程是先用audit build运行相同请求并生成`audit.json`，再让release
campaign通过`--audit`引用该文件。

也可让audit build直接作为campaign candidate：worker会在每个请求前reset、完成后snapshot，并自动聚合
P0 counter；这用于检查计数覆盖和稳定性，不替代release binary的性能签核。独立入口：

```sh
TURBOCIDER_BUILD_AUDIT_COUNTERS=1 TURBOCIDER_NATIVE_ONLY=1 \
  TURBOCIDER_BUILD_OUTPUT_DIR=/private/tmp/tc-audit tools/native/build.sh
python3 -B tools/native/run_streaming_audit.py \
  --library /private/tmp/tc-audit/libturbocider.dylib \
  --model-id ltx-2.5-distilled --model /path/to/LTX-2.5 \
  --request /path/to/default-request.json --output /tmp/audit.json \
  --expect default-zero
```

`test_streaming_audit.py`同时验证disabled no-op、enabled原子计数、真实executor pool/worker callsite、三种
构建的符号隔离以及public C header未暴露私有ABI。
