# 原始LiteRT CustomEventMetal：有限wait、ready、代次与owner实测

<!-- knowledge
id: technique-gpu-ane-case-studies-litert-original-event-probe
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [V2事件调用链](litert-v2-metal-events-and-buffers.md) · 保存的实测 · [第十轮范围（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，在M4 Max上编译并执行固定LiteRT原始 `CustomEventMetal`，不以相似自写类冒充它。本库只写了有界host client，测试原class的constructor、EncodeSignal、Wait/IsSignaled、Retain/Release和GetNative；**没有运行LiteRtEventT runtime、CompiledModel、ML Drift图、shader、blit、模型或ANE**。

## 1. 实际执行的范围

`litert_common.h`依赖生成的build_config.h。探针照上游BUILD的default copy_file规则，把原始 `build_config_gpu_npu.h`原字节复制到生成include位置，记录alias；不是造一个空stub跳过真实ABI。共9个原始输入文件绑定在receipt，client/driver/executable另有SHA。

环境重新读取为macOS26.6.2/25G83、SDK26.5、Apple clang21.0.0，设备名由native程序返回Apple M4 Max。没有安装框架或依赖，没有修改参考源码、系统缓存、电源或产品默认后端。

## 2. 为什么它不会提交GPU工作

client创建一个有效command queue和**不提交**的command buffer；调用原始EncodeSignal使事件目标变成1，但始终不调用commit。正控制通过host设置MTLSharedEvent.signaledValue，不靠GPU执行信号。

每个进程只有两个各5ms的有限wait；没有无限wait、非法地址、故意挂住GPU或设备错误注入。应用command-buffer提交与GPU dispatch均为0。这个0来自client明确没有提交/dispatch的执行范围，不是声称Metal driver内部绝无任何系统工作。

## 3. 两进程观察

每次先通过原有空闲guard的5次1秒采样，全部为0%；然后由有界supervisor启动client，两个进程均正常退出。粗计数仍不是物理独占或未来无干扰证明。

| 行为字段 | 两次观察 | 结论边界 |
| --- | --- | --- |
| 新event、目标0 | ready=true | 未arm的对象不代表一个模型已完成 |
| EncodeSignal到未提交buffer后 | ready=false | 目标已到1，但没有生产者信号 |
| 原生有限Metal wait | 返回false | 实际发生未就绪返回，不是无限等待 |
| 原始void callback有限wait | 返回；IsSignaled仍false | callback返回不能单独当ready |
| host设置signal=1后 | ready=true | 正控制，不是GPU producer验证 |
| observer Retain后原owner Release | observer仍可查询ready | 单线程受控保活，不证明并发refcount安全 |
| 新一代event尚未发信号 | 新false、旧true | 独立对象的代次不互相冒充 |
| 新一代host发信号 | 新true | 正控制完成 |
| GetNative | 指向wrapper自身 | 不是直接返回MTLSharedEvent指针 |

每进程11个行为字段均符合预期，原始逐进程值和source bindings完整保存。两次相同控制程序不是独立随机模型数据，也没有用wait elapsed算推理速度；该时间只作诊断，没有性能排名。

## 4. 保留的未启动尝试

首次在沙箱内，ioreg返回非零，guard在取得任何占用样本前就拒绝，**没有启动native child**。原始admission receipt保留；随后获准在沙箱外重新取得五次空闲采样，才运行上述两个进程。没有绕过缺失计数直接启动，也没有把这个失败算成忙时等待或设备故障。

## 5. 仍不能证明什么

这个实验验证原始类的有界callback/ready/owner行为，不证明[完整Metal上下文的post-event分支可达](litert-v2-metal-events-and-buffers.md)。该问题仍是源码不变量，没有通过修改私有member或补丁把它人为变成可运行路径。

未验证GPU signal/wait、跨queue可见性、实际TensorBuffer的event采用、request并发、GPU完成错误、取消/drain、timeout后的state事务或消费者数值。没有调用LiteRtWaitEvent；其custom分支如何返回status来自runtime源码，不能把native类调用当成已执行整个C API。

## 6. 复现与只读重算

构建需要显式固定参考checkout与系统SDK；运行路径须为fresh ignored输出，且先经过guard。不要在其它GPU实验运行时启动。

```sh
python3 techniques/gpu_ane/litert_event_probe.py \
  --reference ../references/litert-apple-research-20261016 \
  --build-dir outputs/litert-event-new-build

python3 techniques/gpu_ane/guarded_run.py \
  --output-dir outputs/litert-event-new-admission --timeout 30 -- \
  outputs/litert-event-new-build/litert-event-probe
```

已保存本轮raw artifacts时，下式只重算JSON，不启动设备：

23项新离线测试检查原始类receipt、两次准入、有限timeout、代次/保活、零应用提交、精确依赖与CPU布局/候选公式；不是本机重新运行模型或原始shader的测试套件。
