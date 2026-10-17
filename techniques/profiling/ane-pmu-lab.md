# ANE PMU：先证明采到了什么，再解释快慢

<!-- knowledge
id: technique-profiling-ane-pmu-lab
type: guide
operator: scheduling
backend: private-ane
device: ane, cpu
symptom: measurement
semantics: contract-only
representation: fp16, iosurface
boundary: producer-consumer
api: private
-->

[总目录](../README.md) · [原采集器源码卡](../sources/repos/ane-pmu-provenance.md) · [capacity源码卡](../sources/repos/ane-capacity-pmu.md) · [完整本机记录](../ane/case-studies/pmu-payload-and-missing-counters.md)

适用：研究私有 PMU采集、counter协议、CPU/设备计时分界；不作为生产依赖，不要求用户更改SIP/AMFI、boot-args、系统cache权限或安装受限entitlement。

## 1. 需要分开的状态

| 观察 | 可得结论 | 不可补造的值 |
| --- | --- | --- |
| 程序打印ACTIVE或返回0 | 仅状态文本/退出码 | 物理PMU可用、已执行正确 |
| 同步call和独立完整输出通过 | 该图/输入的API与数值边界可用 | counter、逐kernel时间、完整请求收益 |
| stats object缺失 / payload空 | 当前路径没有可解析telemetry | 0 cycles、0 bytes或零物理时间 |
| 自有stats surface仍全零 | 观察到缓冲区未变成非零 | 真实工作量为零、已定位权限/硬件原因 |
| 非零payload和counter名可读 | raw字节/标签可保存 | 字节单位、能耗、clock、累计或per-call语义 |
| 独立缩放、reset/wrap、单位/域关联通过 | 只在对应环境范围内解释指标 | 自动推广其他SoC/OS/backend/request |

`perfCounterData`预期232字节/29个uint64是固定上游描述的候选schema，本机未取得这个payload，不把预期格式写成已测事实。未知name不能合并：`[24..28]`即使同名，仍须按独立index保存。DPE raw值也不能直接叫Joule。

## 2. 累计与per-request不能靠直觉选择

相同工作三次payload均为100，可能是每次100的request-local计数，不能用末次减warmup得到0活动。单调增加也只是兼容累计的一种现象，不是充分证明；其他请求、reset、不同执行分组和overflow仍须排除。

counter_delta只做显式位宽/生命周期决策后的算术：下降默认拒绝，不夹0；只有有独立wrap见证才按模数计算；reset保留null。64-bit存储并不证明有效counter宽度64bit。CPU例子的8-bit 254→2得到4只是数学负控，不声称真实ANE使用8-bit。

## 3. 时间和派生吞吐的边界

`api_wall_ns`是本进程host对private evaluate调用的CLOCK_UPTIME_RAW跨度；`host_prepare_call_readback_ns`另包含自有输入准备、poison和锁定后完整readback。它们不是单个物理kernel elapsed。`hwExecutionTime`即使存在，单位与聚合语义也需重新确认。

只有取得匹配单位/生命周期/作用域后，才可计算cycles/sec、bytes/sec或每core吞吐。固定16除法、名义GHz、理论MAC数和框架标签都不能补上缺失遥测；logical operations / host wall至多是派生接口吞吐，不改名为hardware occupancy/TOPS。

## 4. 当前有界探针

pmu_probe.py默认只准备自有合成MIL，不导入可选库、不构建/执行设备。显式 `--device` 才编译原创native客户端，需要Apple Silicon macOS/CLT，不依赖产品或参考checkout。

exit0表示该模式完成（例如prepare或mask0数值控制；有raw时也不授予语义资格）；exit3明确表示推理数值通过但请求的PMU payload不可用；其他错误非零并保留阶段/receipt。不是执行成功就把capture标为成功。早期收集版本曾以execution-only的0退出，历史记录不改写，最终strict入口实际返回3。

客户端检查selector/关键ABI、真实NCHW FP16 pitch/shape、8MiB per-surface上限，拒绝任意fallback大小。每call换generation、输出poison、完整数值检查；统计surface只清自己的4096 bytes，不重置硬件counter。读取发生在同步返回与IOSurface锁之后，成功unload后释放；失败时不复用映射，unload失败则隔离到子进程退出，不声称全系统device drain。

affine `2*x+0.25`使用精确二进制小数；conv为明确合成dense1×1参数。本地独立Python math.fsum witness重建每个逻辑元素的FP16输入/输出并核对完整SHA；输入有声明的17列周期，CPU验证可复用其精确和，但设备仍计算/写出完整tensor。这不是伪装为真实Qwen activation。

## 5. 本机结果与采用门槛

9组/81次完整输出通过；mask0/2/15、较大conv、compile-profiling和新进程确认都未取得stats object。54次还逐call看到自己的stats surface全零，较早9次profiled记录未采这个surface字段，不回填。原工具的missing-model/无stats正常退出与capacity跳过PMU路径也全部保留。

可吸收的是可查询的协议/计时/缺失与生命周期合同，不是某个SRAM容量、PMU峰值或全模型提速。当前保持 `private-api / counter-semantics-unverified / capture-unavailable-for-tested-paths`；不关闭其他私有入口、新OS、ANECIR/binary格式或公开profiling的继续研究。

实现：CPU合同、数据verifier、离线负控。`private:pmu`仅登记显式device入口，不由CPU all或查询调用。

重开条件：OS/SDK/framework/upstream ref、模型序列化/区域、capture策略、布局或实测payload改变。先保存每次raw与所有失败，再做工作量缩放、单位/时间域、reset/overflow、同进程并发与独立确认；没有counter仍可做严格完整wall-time优化，但不能编造微架构解释。
