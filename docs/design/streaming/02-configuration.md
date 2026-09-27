# 02 · 布局参数、配置和兼容规则

[目录](README.md) · [运行时](03-runtime-protocol.md) · [预算](05-memory-contract.md)

`execution.streaming`现已接入schema2 request的plan-only解析和profile v2合并；模型执行仍待接入与认证，见 [13](13-implementation-progress.md)。
示例是request的execution片段，不是完整生成请求。
首发只实现 `selection=manual`，自动搜索单独排期。

## 1. 参数精确定义

| 字段 | 含义 | 首版规则 |
|---|---|---|
| `enabled` | 是否进入新框架 | 默认 false；显式 false 不因附带参数变 true |
| `schema_version` | streaming 对象版本 | enabled 时必须为 1；与外层 request/profile 版本独立 |
| `selection` | 参数来源模式 | 首版只接受 `manual` |
| `retention` | pool 跨请求生命周期 | 首版 `request`；不接受 `session` |
| `stages` | 以 adapter stage ID 为 key 的配置 | 未知 stage 拒绝；不得根据拼写相近猜测 |
| `residency` | 该 stage 的 weight 布局 | `resident` 或 `streamed` |
| `block_group_size` (G) | 一次 slot fill 的连续 block 分组上限 | 正整数，不改变 block 顺序；不能跨 layout-class 边界 |
| `slot_count` (K) | 池物理 slot 数 | 精确值，不是上限；由 adapter 能力决定支持范围 |
| `resident_prefix_blocks` (P) | 开头常驻的逻辑 block 数 | 不计入 K，不代表跳过后面的 block |
| `prefetch_distance` (D) | 相对当前需求 group 最远允许启动的未来 group 距离 | 0…K−1；D=0 是 just-in-time |
| `io_workers` (Q) | 同时执行 fill 的最大 worker 数 | 1…K；通常先认证 Q=1，不等于 K |

K/G/P/D/Q 不必全部暴露在普通 UI：UI 先提供 K/P 和具名 preset；G/D/Q 可放高级设置。
API/profile 仍使用同一个显式结构，resolved report 必须显示全部值。

### 1.1 分组规则

descriptor 定义 N 个有序 block。P 个 prefix 不参与 suffix group；余下 block 按兼容 layout class 内连续最多 G 个组成 group。
最后一个 group 可以较短；layout class 切换也结束 group。除此之外不为节省内存偷偷减小 G。

示例 N=10、P=2、G=3、同一 layout class：prefix=[0,1]；groups=[2,3,4],[5,6,7],[8,9]。
每个 group 必须全部填入 slot 后才计算；按原 block 顺序逐个 compute，G 不是 kernel fusion 开关。

如果 suffix 只有两个 group 而 K=3，首版拒绝 `slot_count_exceeds_groups`，不静默改 K=2 或分配空闲假槽。
想要所有 weights 常驻必须显式 `residency=resident`，不能通过 P=N 且 K>0 表达。
多 layout class 首版在确定 barrier 重新建池，每个 pool generation 都必须满足 K≤该区间 groups；局部 ordinal modulo K。
跨 block fusion 若令上述边界不安全，拒绝该 G，不自动关闭 fusion 或重排 block。详见 [09](09-layout-compiler-spec.md)。

### 1.2 默认值与精确性

streamed stage 必填 K、G、P、D、Q；普通 UI/preset 负责补全，runtime 不猜测最佳值。
resident stage 只填 `residency=resident`：resolved K=0、D=0、Q=0、P=N；拒绝额外 streamed-only 字段。
未列出的 stage 使用 descriptor 中固定、版本化的 baseline policy，不能动态按剩余内存选策略；plan 输出每个 inherited stage。
descriptor 没有 fixed policy 时要求用户补全，不能假设无配置的 stage 不占内存。

### 1.3 Lookahead 与 startup

在消费 group j 前，允许填充区间 [j,j+D]；同时 ready/in-use/loading 的总 slot 不超过 K。
j 精确定义为“下一个尚未成功提交 compute 的 group”；提交后可推进，不必等 GPU 完成。D=0 是按需求派发，
并不禁止已提交 compute 与下一 group 的另一槽 I/O 重叠；K=1 才必须等唯一槽 reader 完成后重填。
stage 初始可以在 prefix compute 期间加载这 D+1 个 suffix group。先读 j，未来 group 不得阻塞当前需求。
首发按 suffix group ordinal modulo K 静态分配，同槽复用仍必须等 use fence。
3 slots、D=2、Q=1 是三份内容缓冲加单读线程，并非“三个线程同时读盘”。

## 2. 手动布局，不设置全进程预算

```json
{
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 1,
      "enabled": true,
      "selection": "manual",
      "retention": "request",
      "stages": {
        "denoiser": {
          "residency": "streamed",
          "block_group_size": 1,
          "slot_count": 3,
          "resident_prefix_blocks": 1,
          "prefetch_distance": 2,
          "io_workers": 1
        }
      }
    },
    "memory_constrained": {"enabled": false}
  }
}
```

这里 P=1 是 LTX 接入演示，并非性能最优配置。未来必须有该组合的 layout execution record 才能执行。
只控制 slot 数量、backing 容量和功能安全，结果 `memory_enforcement=none`；不应显示“已保证零 swap”。

## 3. 同一手动布局加上预算

在上述 execution 中把 `memory_constrained` 替换为：

```json
{
  "enabled": true,
  "limit_bytes": 21474836480,
  "buffer_percent": 15,
  "min_free_bytes": 2147483648
}
```

Y=20 GiB、有效 B=17 GiB。这不是另一种 pager：同样的三槽 layout，只增加 whole-request admission 和 guard。
如果需要 18 GiB，则返回峰值阶段、组成和缺口；不改成两槽、不修改 prefix，也不执行默认模式。
诊断可以给出未执行的候选建议，由用户重新提交。

## 4. 内存充足时

最强兼容选项是完全不设置新 streaming；已有 default 行为照旧。
也可显式框架 resident：

```json
{
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 1,
      "enabled": true,
      "selection": "manual",
      "retention": "request",
      "stages": {"denoiser": {"residency": "resident"}}
    }
  }
}
```

这是新的 resident lifecycle candidate，不能直接继承现有 default resident 的 release 资格。
bounded+resident 也要计入 VAE/输出峰值；request retention 不承诺旧 warm cache 的速度。

## 5. Profile / preset 的单一来源

原有 profile v1 使用精确 `match.gpu_name/memory_bytes`、`models`；v1兼容语义保持不变。
当前已加入 profile v2解析/合并，在每个 model 中增加 `streaming`；模型执行资格仍未完成。预算仍放原 `memory_constrained`。不再同时实现旧稿的
`memory_tuning`、`slot_policy`、`slot_candidates`、全局 `memory_defaults` 等并行配置方言。

v2 示例见 [profile-v2.json](examples/profile-v2.json)。它是 64 GiB 实验机器的示例，不代表该配置已认证或适合所有 64 GiB 设备。
顶层 `enabled=true` 只启用该 profile 的解析；model 下 `streaming.enabled` 才开启新框架，二者不能混为一谈。

合并顺序：descriptor fixed defaults < 用户显式选择的 model profile < request 显式字段。
保留每个字段的 presence bit；0/false 与缺省不同。`stages` 按 stage ID/字段合并，合并完成后一次验证，report 标注来源。
例外：request 显式改变某 stage 的 residency 时，以该 request stage 对象整体替换旧 stage，不继承旧模式专属字段；
转 streamed 需重新提供完整 K/G/P/D/Q，转 resident 只写 residency。同 residency 时仍按字段覆盖。
只把 K=3 覆盖为 K=1 而继承 D=2 时必须拒绝，不能自动把 D 改0；UI/preset应提交完整有效布局。
request 中 `streaming.enabled=false` 覆盖 profile true，并完全走 legacy；profile 中独立的 memory guard true 不被隐式关闭。
希望全部回到 legacy 的用户同时显式关闭这两个开关。有效组合不成立时返回冲突，不自动忽略安全要求。

普通机器档位只用于显式 preset 选择界面，不能自动修改当前请求：

| 档位标签 | 提供的选择思路 | 不做的承诺 |
|---|---|---|
| small-memory | 小 P、较小 K、component 串行 | 不承诺 16 GiB 能跑所有模型/shape |
| balanced | 2/3 slots 与测过的 prefix | 不将 M4 Max 数据套到不同 GPU/SSD |
| high-memory | legacy 或显式 resident | 不默认启用框架、不声称 guard 零开销 |

每个 preset 是完整参数文件，不执行脚本；包含 descriptor/layout revision、实验/发布状态和出处。
performance provenance 与 execution certification 分开：换磁盘影响推荐排序，不得让失配 evidence 授予硬预算保证。

## 6. 与旧字段冲突的冻结规则

| 输入 | 处理 |
|---|---|
| 无新 streaming，旧 residency/memory_budget | 完全保持现有逻辑 |
| streaming=false，旧 residency=streamed | 继续旧 streaming，不强制 resident |
| 新 manual + 用户显式旧 `residency`/`memory_budget_bytes`/`streaming_offload` | 返回 config_conflict，即使值看起来一致也不建立两个权威 |
| parser 默认注入旧 residency=resident + 新 manual | 不算用户冲突；必须新增旧字段 presence tracking 区分 |
| 新 manual + `memory_constrained.enabled=true` | 合法；只验预算，不调用旧 heuristic normalization |
| 新 manual + 显式 legacy `max_refill_slots` | 作为额外 K 上限，每个 pool 均核对；不覆盖精确 K |
| 新 manual + 未显式 max_refill_slots（旧默认 3） | 不作为新 config 的隐藏 cap；adapter 支持集限制 K |
| 新 manual + GPU/ANE 或首发不支持的 LoRA/输入分支 | capability/route 拒绝，不静默改路线 |
| 新 manual + unsupported K/G/P 组合 | 精确拒绝；不降低到“最接近”组合 |

预算中的 `allow_quality_preserving_tiling` 是许可，不自动改 explicit layout；未显式选择且 descriptor 固定的 tile 策略必须显示出来。
block count 不等于执行 block 子集；首版不支持“只算前 N 块”的 streaming 参数。

## 7. 验证与结果

整数必须有限、非负且无小数，拒绝 bool-as-number、未知键、重复 JSON 键、溢出及不认识的 schema 版本。
P 满足 0≤P<N；G≥1；1≤K≤suffix groups；0≤D<K；1≤Q≤K。adapter 可进一步限制 P≥1、G=1、K=2 等。
bounded 模式沿用现有 buffer_percent 校验范围，不能因新 schema 悄悄放宽。

结果至少包含：

```text
requested_layout + field_provenance
resolved_stages/groups/slot_assignment/capacity + inherited policies
actual_pinned_blocks/slot_count/group_count/io_workers
layout_digest + model_descriptor_digest + adapter/backend revisions
eligibility=plan_only|layout_validated|bounded_certified
enforcement=none|bounded_request
execution_supported + rejection_code + failed_stage + required/available bytes
```

actual 与 resolved 不同就是 lifecycle/configuration failure，不作为“智能优化成功”。
配置 digest 只绑定行为字段；注释/JSON key 排序不影响 plan digest；完整文件 digest 可另记 provenance。
