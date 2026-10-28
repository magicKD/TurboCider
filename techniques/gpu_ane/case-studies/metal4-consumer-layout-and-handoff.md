# Metal 4 实际消费者：buffer-backed tensor 与 compute→ML→compute

<!-- knowledge
id: technique-gpu-ane-case-studies-metal4-consumer-layout-and-handoff
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [模型 pass 入口](metal4-machine-learning-passes.md) · 整理数据 · [来源范围](../sources.md#consumer-handoff)

2026-10-16，继续上一轮的公共API验证。这次不是 host 填 tensor→单独 ML pass→host 读回，而是 **GPU生成 A/B → ML矩阵乘 → GPU消费 C**。本机 M4 Max 的三个尺寸、两个独立进程、96个样本全部通过；只证明本库这个应用层衔接，不证明 ANE 放置、内部无复制或真实模型收益。

## 1. 先补齐一个不能忽略的 offset 限制

Apple [buffer 创建 tensor](https://developer.apple.com/documentation/metal/mtlbuffer/maketensor%28descriptor:offset:%29) 和 SDK26.5 `MTLBuffer.h` 明确要求：当 usage 包含 machineLearning，**offset 必须是0**。普通buffer里任意偏移的view、前置canary、bump allocator slice不能直接沿用。

因此本probe给A/B/C各一个独立buffer，tensor在offset0包装。它检查返回的 `tensor.buffer` 确实是请求的buffer、`bufferOffset=0`、实际stride匹配，再用同一buffer的GPU地址供普通compute访问。仅“调用了newTensor”不足以构成backing已采用的证据。

这里没有测试IOSurface、CoreML predictor outputBacking，也没有测试任意子分配或multi-plane tensor。不要用一个路径的成功代替另一条接口的对齐/所有权合同。

## 2. 元素stride、row-byte对齐和base alignment是三件事

Apple [stride说明](https://developer.apple.com/documentation/metal/mtltensordescriptor/strides) 与本机header要求：stride以元素计，最内层为1；ML usage的第二stride满足64-byte条件，后续维度需符合相应乘积关系。本次只测FP32、rank2，采用 `pitch_elements = ceil(columns*4/64)*16`。

| M / K / N | A[M,K] pitch | B[K,N] pitch | C[M,N] pitch |
| --- | ---: | ---: | ---: |
| 3 / 7 / 5 | 16 | 16 | 16 |
| 17 / 33 / 9 | 48 | 16 | 16 |
| 33 / 65 / 17 | 80 | 32 | 32 |

表中单位均为FP32元素，不是bytes。实际返回的pitch与请求相符。

`tensorSizeAndAlignWithDescriptor`还会返回所需空间和base alignment：本次三图的各backing查询alignment都为4，但这**不取消64-byte row条件或ML offset0条件**。例如A[17,33]需3264 bytes、C[33,17]需4224 bytes，二者的dense logical bytes却都只有2244。不能只按元素总数分配或memcpy。

尾部guard放在 `max(SDK要求空间, rows*pitch*4)` 之后；ML offset0不允许本probe另加前置guard。A/B所有logical元素及row padding均检查，C的logical结果与最终dense consumer全部检查。C padding是否改变仅作为观察，不把其未改承诺成任意后端的公共保证。

活跃网页还列出INT2、FP8等本机SDK26.5枚举没有的类型。来源清单分别保存网页内容hash、header hash和SDKSettings版本，不把“今天网页写了什么”倒推成本机SDK已支持所有overload。

## 3. 三段计算与两条提交策略

原创shader 根据phase生成可精确表示的小型dyadic A/B；ML小包计算C=A×B；最后的GPU kernel按C的真实pitch读取，计算：

`D[r,c] = 0.5*C[r,c] + 0.25*(phase−4) + 0.125*(r−c)`。

phase逐轮改变，以检查槽位复用后是否还读到旧数据。原生客户端 复用相同pipeline、形状和布局，两个arm各持有独立slot：

```text
single-commit:
  producer compute → dispatch→machineLearning barrier → ML pass
                   → machineLearning→dispatch barrier → consumer compute
  一次commit，最后一次CPU完成等待

host-synchronized control:
  producer compute → commit/wait → ML pass → commit/wait
                   → consumer compute → commit/wait
```

两支都没有应用层stage间payload复制，control多了两次CPU等待。它是**有意构造的同步边界对照**，不是最优GPU引擎或所有模型runtime的代表。

同一command buffer不自动串行所有Metal4 encoder；本次在消费者encoder开头设置跨pass的queue consumer barrier，scope分别覆盖dispatch和machineLearning。visibility保守采用 `device | resourceAlias`，因为tensor ID与rawbuffer地址是同一storage的不同访问入口；没有搜索最小barrier，也没有运行删barrier的竞态负例。

每次begin后重新声明residency；scratch heap与tensor/buffer保活到terminal feedback；仅完成之后才reset allocator、CPU poison或重新写参数。两个arm不并发使用同一slot/scratch。这不是多请求队列或取消/drain的验证。

## 4. 实际数值与CPU负例

两进程各3个shape，每shape每arm 2次warmup+6次正式；交替先后顺序。合计96个完整链路记录、72个正式记录，检查：

- ML结果C：23,328个元素；最终GPU consumer D：23,328个元素。
- GPU生成的A/B：133,248个元素；input row padding：65,920个元素。
- 四个buffer的尾部guard：6,144个元素。

上述比较均为finite且exact，mismatch=0；数值只覆盖这个可精确表示的有限输入域，不证明一般FP32 fidelity或实际accumulator位宽。C padding在本次96个记录中未观察到变化，但不作为资格不变量。

独立CPU负例故意将C的physical pitch16错当logical width5：第一行仍正确，后两行的10个输出错误。这说明只检查第一行、checksum或binding创建成功会漏掉consumer ABI错误。负例没有在设备上执行错误或越界kernel。

## 5. 局部计时：减少host边界，而不是宣称ANE提速

host计时从准备好的slot开始编码，直到最后feedback完成；包含ML pass、两compute、barrier、提交与通知。排除编译、分配、CPU初始化与全部验证；不是完整请求。

| seed / M,K,N | single-commit median µs | 三次CPU同步提交 median µs | 逐对baseline/single中位 |
| --- | ---: | ---: | ---: |
| 1 / 3,7,5 | 237.791 | 442.480 | 1.831× |
| 1 / 17,33,9 | 226.271 | 440.708 | 1.945× |
| 1 / 33,65,17 | 237.771 | 456.021 | 1.895× |
| 2 / 3,7,5 | 267.312 | 506.917 | 1.925× |
| 2 / 17,33,9 | 247.708 | 468.333 | 1.909× |
| 2 / 33,65,17 | 235.250 | 484.541 | 2.063× |

每行只有6对正式数据；36对均是single较低，全部逐对耗时保留在JSON。比例是每对baseline/single的中位，非两组中位数之比。它支持这个小客户端中减少CPU同步边界的价值，不建立通用2倍收益或统计/频率隔离结论。

两支模型图和数据完全相同，均未观察到物理推理单元；**这不是GPU-only对GPU+ANE比较**。普通优化GPU GEMM可以是更强且不同的基线；真实模型还会改变compile、packing、shape cache、scratch、驻留和critical path。

feedback区间按commit单列/求和，但含ML pass时不把它自动命名为“纯GPU kernel时间”，更不能当ANE PMU。新进程也不等于cold-cache：模型专门化约1.3–17.6ms，未清系统缓存，不与上一轮cold/warm混算。

## 6. “没有host round-trip”究竟验证到了哪层

本次可以说：payload由GPU生成，经同一请求backing进入模型pass，再由GPU消费者读取；中间没有本库代码调用 `replaceSlice/getBytes`、memcpy或等待CPU。control也不复制payload，只是增加等待。

不能说：Metal内部绝无staging/packing、副本、ANE必定直接访问这个buffer，或GPU与ANE在同一时刻执行。两层消费者本来有顺序依赖；API能组织异构timeline，不等于这些三个stage彼此独立。

最终为了诊断，CPU在terminal后遍历shared backing和输出；这是验证期readback，不在局部计时内。另有CPU初始化sentinel/参数和host编码，所以不宣传“全程无CPU”。

## 7. 重放和后续

使用上一轮已固定SHA的官方ZIP，不重新下载或执行其源码工程。两个成功进程分别在五次 `[0,0,0,0,0]` 占用采样后启动；during采样也为0，仍可能漏掉短burst，不能证明绝对空闲或频率稳定。

```sh
# 默认仅编译host；输出目录必须尚不存在。
python3 techniques/gpu_ane/metal4_consumer_probe.py \
  --archive /path/to/ARCHIVE.zip --output-dir outputs/metal4-consumer-host-new

# 显式设备调用会先执行GPU空闲准入，忙时拒绝。
python3 techniques/gpu_ane/metal4_consumer_probe.py \
  --archive /path/to/ARCHIVE.zip --output-dir outputs/metal4-consumer-device-new \
  --execute --seeds 1 2

# 只读核对原始receipt、数值、源码/binary/package身份。
python3 techniques/gpu_ane/metal4_consumer_probe.py \
  --curate outputs/gpu-ane-metal4-consumer-20261016/device-run
```

后续需要真实模型operator转换、物理placement trace、多个bucket/slot并发、失败回收，以及对优化GPU消费者的完整请求比较。本轮不授予生产启用或ANE/M5资格。
