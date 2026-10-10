# Native LoRA paired base gate/up：单层筛选未获得收益

2026-10-08。**结论为 `NoGain`：把两个基础 gate/up 投影合成一次 dispatch 后，两次正式完整 FFN 计时分别慢 0.0970% 和 0.5691%；两条路径输出逐位相同。** 本轮保留独立 primitive 与可复现工具，不接入 `NativeBlock`、请求路线、默认配置或 App，不追加 E2E。

这里的 `NoGain` 表示这一输入上没有测到稳定收益，不表示已经证明 paired kernel 普遍更慢，也不证明系统的主要瓶颈已经找到。原始回执的 `screening-pass` 只表示数值与执行检查通过，不能解释成性能通过。

证据来自[真实回放](../../outputs/native-lora-paired-20261008/real-ffn.json)、[运行监督器](../../outputs/native-lora-paired-20261008/real-ffn-supervisor.json)、[运行前来源核对](../../outputs/native-lora-paired-20261008/pre-device-verification.json)。[机器汇总](2026-10-08-native-lora-paired-base-screen.json)保存完整数值、binary／source 身份及小型证据文件 SHA256；没有修改原始测量记录。

## 构建与验证范围

实际运行库为：

```text
tc-runtime-build-v1-e6ea19097dc565035bf6a6fc3a554308a6f1c9d968042098f847c1430aa7f160
```

监督器在设备执行前后核对 **570 个 runtime source，0 个 mismatch**，CLI／library／probe／manifest／工具源码身份保持一致。[完整 native package 构建](../../outputs/native-lora-paired-20261008/full-build.log)成功；它跳过了全量测试和辅助 probe，不代表完整测试套件通过。构建日志仅记录三项既有 `ane_gpu.mm` 的 `fastMathEnabled` 弃用警告。

[Gate/up CPU 与 host 检查](../../outputs/native-lora-paired-20261008/gate-up-cpu.log)通过：122 项绑定合同检查、33 项 source 检查及严格 host 编译。[回放 CPU 合同](../../outputs/native-lora-paired-20261008/replay-cpu.log)通过：10／9 次完整 FFN dispatch、相同的 8 次尾部 dispatch、三个实际 r128 投影形状及六项负例拒绝。这些 CPU 检查没有创建设备、编译 GPU shader 或加载模型。

唯一设备组使用 Apple M4 Pro、本地 Qwen Image 2.1 BF16 和 Viggle v0.2.1 r128、strength=1。只读取既有 capture `75991-1226054446574458` 的 **layer0／step5**，输入为 BF16 `[1,1024,4096]`、FFN width=12288。没有新 capture、下载、checkpoint 副本或扫描。

执行预算为一次实际 MLX 完整 FFN oracle，以及 **warm AB → 正式 AB → 正式 BA，总共六次 native 完整 FFN**，无自动重试。监督器进程 wall 为 3.43035 s、退出码 0，未触发 45 s 超时或发送信号。这里没有运行完整 DiT 或生成图片。

## 对照只改变基础 gate/up

| 项目 | A：split base | B：paired base |
|---|---|---|
| 基础投影 | 同一 `NativeLinear` 分别执行 gate、up | 一次 `NativeGateUp`，分别输出 BF16 gate、up |
| 基础投影 dispatch | 2 | 1 |
| 完整 FFN dispatch | 10 | 9 |
| 基础投影 recipe | `dense-bf16-mpp32x128-fp32-accum-cast-before-bias-v2` | `dense-bf16-mpp32x128-paired-base-separate-bf16-gate-up-v1` |

两路之后完全共用 **gate-low → gate-apply → up-low → up-apply → SiLU/up → down-base → down-low → down-apply**，使用同一个 `NativeLoRA`、同一个 down `NativeLinear`、同一个 FP32 low 工作区、相同原始权重／factor／输入视图及六个 scratch backing。三个 LoRA terms 都实际满足 SGMMA 条件；共同 recipe 为：

```text
runtime-lora-fp32-sgmma8x8-aligned-r128-scalar-tail-ordered-bf16-bias-last-v2
```

A 也先执行两个基础投影，再执行 gate／up 校正，以隔离 paired 投影这一变化。它不同于旧 SGMMA 回放的 `gate-base → gate-correction → up-base → up-correction` 调度，不能把本次 A 的时间当成旧工具或历史 E2E 的复测。

新 primitive 不执行 LoRA 或 SiLU，不合并 adapter 权重，也不改变现有 FFN 的 prepare bundle。只有显式调用 `NativeGateUp::prepare` 才准备一个新增 PSO；当前默认路径增加 **0 个 PSO**。本轮没有给 `NativeBlock` 或其他生产路由增加调用。

## 正式时间与数值

| 顺序 | Split 完整 FFN wall | Paired 完整 FFN wall | Paired 相对 Split |
|---|---:|---:|---:|
| AB | 51.830750 ms | 51.881042 ms | 慢 0.0970% |
| BA | 51.591916 ms | 51.885500 ms | 慢 0.5691% |

完整 FFN wall 从 encode 开始，到提交后收到物理完成为止，包含基础投影、全部 LoRA 校正、SiLU 和 down；不包含分配、PSO 准备、MLX oracle、来源哈希或 CPU 比较。每对两路之间只有明确单列的首输出 **8 MiB CPU 快照**；有限值／误差比较、padding、stat 和账本检查在两路都结束后进行。每路使用相同输出和 scratch，因此快照是保留第一路结果所需的有界复制。

warm AB 为 65.251417 → 51.153625 ms，**排除在性能结论之外**。它不能作为 21.6% 加速证据。GPU span 是 Ticket 中首个 GPUStartTime 到最后一个 GPUEndTime 的范围，不等于内核忙碌时间，也不能与 wall 相加。

六次输出各有 4,194,304 个有限 BF16 元素，三对 A／B 输出均逐位一致。相对同一 `Weights`、实际编译执行的 MLX `runtime_ffn::full` oracle：

- RRMS：**9.935683191904662e-7**，通过预设 0.005 门限。
- 最大绝对差：**0.001953125**。
- 95 个 BF16 元素与 MLX 不同；“逐位相同”仅指本次 split 与 paired 两路之间。

一次 MLX oracle 的 65.767041 ms 包含首次编译／执行和清理，只用于数值参照，不是公平的 warmed MLX 计时路线。

本次 paired PSO prepare 为 **203.250042 ms**；split/down Linear 为 0.463166 ms，共用 SGMMA 为 0.280708 ms。它们依次在一个 Backend 上准备，并共享未清除的 OS／driver compiler cache。新 PSO 与已有缓存状态不同，不能据此排列各实现的独立冷启动开销；这些时间也不在上表 FFN wall 中。

## 来源、内存与物理完成

只 materialize 单层三个基础权重、六个 LoRA factors 和一个输入，source payload 为 **308 MiB（322,961,408 B）**；不是整个模型。MLX oracle 结果另有 8 MiB，记录的 MLX active 为 331,350,016 B。adapter 的 held-fd pin 验证读取 679,604,800 B，验证读取量不等于 materialize 的 tensor 内存。

六个新增 native scratch backing（含 guard／row padding／首输出快照）实际占 **88.875 MiB（93,192,192 B）**。这些分配使用同一 phase ledger；原始 MLX backing 由实际 Data owner 保持，作为产生后、同步后的 retained baseline，不通过 import 重复声称已计入新增原生内存。

账本 process baseline 为 876,053,416 B，`same_phase_peak_committed_bytes=969,245,608 B` **包含该 baseline**，不是额外 scratch 量。物理完成并释放后，新增 storage count／bytes、reservation count、pending release count 和 unknown bytes 均为零；不表示进程内存清零。framework guard 与 full-request memory 均未认证。

每次 native Ticket 均报告 `physically_complete=true`，且 `submitted_buffers=completed_buffers=1`，共六次完整执行。物理完成证据来自 Ticket，不来自监督器退出码或进程退出。不可变输入／权重／factors／oracle 哈希、padding、实际 MLX Data owner 和四份来源文件前后 stat 均通过；库与源码身份也未变化。基础模型采用 held descriptor + canonical stat identity，没有增加完整 checkpoint 内容哈希；adapter 采用 verified held-fd pin。

监督器记录 `vm.swapusage` 前后 used 均为 3297.75M，Swapins／Swapouts 未增加；`pmset` 没有记录 thermal／performance warning。这些有限观测不证明频率稳定或整个系统无压力。

## 决策与证据边界

减少一次 dispatch 并未在这个完整 FFN 中形成稳定时间收益；不进入生产 `NativeBlock`／默认／App／新请求选项，不为这个候选继续跑 E2E。独立 primitive 和回放工具保留，后续如有不同融合方案应重新提出可隔离的假设，不能把本轮结果泛化成所有 fusion 都无效。

本轮不能判断全 DiT 的主要瓶颈，也不能用一个 layer／capture 给其他文本行数、层、adapter、rank 或 strength 授予数值和性能资格。它没有 ANE、LoRA W1、W2、编辑、Z-image 或最终图像证据。[旧 SGMMA 整图结果](2026-10-08-native-lora-sgmma-e2e.md)以及既有 Base W1 结果保留原构建和原范围，不转授给当前 `e6ea1909…` 构建或 paired primitive。

## 清理

主线程已完成[本轮 owned 清理](../../outputs/native-lora-paired-20261008/owned-cleanup.json)：删除 **241 个文件、20,721,076 B（约 19.7612 MiB）**，仅涉及本轮新 build 目录的 `.o`／`.a`／module cache 和 supervisor bytecode。测量 probe、runtime CLI／library、manifest、原始回执／日志、已有 capture 与原模型均保留；没有删除系统／Apple driver 缓存。

清理回执 SHA256：`92b68429fec342fd3c3a4ad2087127346c7b55ee580e50ef6e3c00f91e61b260`。
