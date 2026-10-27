# Owned command graph：少提交，但不丢掉资源所有权

<!-- knowledge
id: technique-designs-owned-graphs
type: method
stage: pipeline
operator: scheduling
backend: metal, cpu-reference
device: gpu, cpu
symptom: launch-bound, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[设计目录](README.md) · [本地 CPU 实现](../examples/reference/contracts.py) · 真实 Metal 实现

## 从 Splash 吸收的核心

Ordered graph 保存 dispatch 顺序、pipeline、geometry、buffer bindings 和参数。参数必须复制到 graph 自己拥有的稳定存储；buffer binding 保留完整 allocation owner。不能保存指向调用栈参数、易扩容 vector 内部或临时 tensor view 的悬挂指针。默认复制 graph 会让内部指针指回原 graph，因此要禁 copy 或显式重建 bindings。

本地 `OwnedGraph.add()`深复制参数、按引用保留实际 backing；`submit()`把命令所有权转给 ticket，并封闭 graph。`__copy__`和`__deepcopy__`拒绝不明确的拷贝。CPU 容器自动保活不等于 native 指针自动保活，但明确表达了应该由谁保活。

## 从 graph 到设备提交

```text
构造 graph -> 验证 bindings/geometry -> encode 多个 dispatch -> commit 一次
                                                 |
              ticket 持有 program / params / allocation owners
                                                 |
                           device completion -> 数值/状态确认 -> 发布结果
```

每个 view 至少需要：allocation identity、allocation bytes、offset、shape、physical pitch、dtype、access mode；验证最大访问位置不越界。逻辑 shape 一样不代表可互换的物理布局。同一 command buffer 中的可见性还取决于 encoder、资源 hazard tracking 和所用 API；跨 queue/设备不能依赖列表顺序，要建立实际 event/fence 边。

减少 submit 节约的是 host/driver 固定开销，不是把依赖关系消掉。bench.mm的 `graph`对比每 dispatch 独立提交和一个 command buffer；`overlap`检查两个 GPU queue 的实际时间区间，不能冒充 GPU/ANE 并行。

## 失败路径是设计的一部分

`timeout/cancel`只表示调用方不再等待或不允许发布，不意味着设备停止。ticket 必须继续持有资源，直到物理完成。逻辑失败可以先记录；allocator/slot reuse 必须后发生。解除等待以防死锁时，还要通过 failure flag 阻止下游把 scratch 当正确结果。

本地 `CpuTicket.cancel_or_timeout()`保留 owners；`finish()`仍执行并 drain，返回不可发布。它特意显示：取消不是已经写过的 scratch 的事务回滚。操作抛异常时也进入 terminal/cleanup，不伪造成功。

生产适配建议：ticket 区分 submitted、physically_complete、numerically_valid、publishable；取消后保留 quarantine 队列并由 completion 回收。若完整重跑 GPU，使用另一套安全 scratch，或先 drain 原消费者。

## 运行、反例与复用

反例测试覆盖参数提交后被改写、graph 拷贝、重复 submit、重复完成、超时仍保活、异常后清理。适合小 kernel 较多的 QK norm/RoPE、FFN epilogue、LoRA projection 链；不适合为了合并而制造巨型无限驻留 graph。新 alias、deferred join、queue、batch、取消策略或 precision 都需要重新验收。

这是 Splash owned-payload 机制的独立教学重述，未链接或复制 Splash runtime。原项目文件和 revision 只作 [可选溯源](README.md)。
