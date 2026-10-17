# 消费者驱动张量 ABI：stride、segments、resource identity

<!-- knowledge
id: technique-designs-consumer-abi
type: method
stage: ffn, pipeline
operator: layout, scheduling
backend: coreml, metal, cpu-reference
device: cpu, gpu, ane
symptom: wrong-layout, lifetime
semantics: bitwise, contract-only
representation: iosurface, packed
boundary: gpu-ane, producer-consumer
api: public, reference-only
-->

[设计目录](README.md) · [本地地址 / 分段样本](../examples/reference/heterogeneous.py) · [来源阅读](../references/gpu-ane-projects.md)

## 优化对象是图边，不只是 operator -> device

从mac_local_ai的HeteroMem与mac_transformer原生runtime吸收：图边应同时记录logical shape、physical strides、segments、dtype/quant recipe、backing allocation、alias/access mode、producer/consumer和completion epoch。统一内存只说明可共享物理页，不保证两runtime拥有兼容的高效布局。

## 1. 让producer直接写consumer布局

CoreML conv式逻辑输出可为`[1,N,1,M]`，而GPU希望读row-major `[M,N]`。默认连续offset是`n*M+m`；指定element strides为`[M*N,1,M*N,N]`，有：

`offset(0,n,0,m)=n+m*N`，正是GPU的row-major地址。

这把布局转换移入producer写地址生成，而不是预测后再transpose。strides单位是元素，不是字节；需要backing至少覆盖`1+sum((shape_i-1)*stride_i)`个元素，且逻辑元素不得意外alias。实际pitch带padding时还要用真实pitch推导，不能只套密排公式。

本地`strided_offsets`枚举并拒绝重叠地址；`consumer-strides`实际将相同逻辑元素写入两种布局，再由row-major consumer读取。错误默认布局得到相同shape但不同值。它是地址证明，不是物理ANE DMA或当前SDK一定接受任意stride的证明。

设备对照至少四组：producer默认、producer非默认stride、hybrid默认+transpose、hybrid指定布局+join。非默认write可能拖慢producer；只有省掉的consumer bridge大于producer premium才净盈利。`outputBackings`返回对象/pointer匹配只证明客户端backing被采用，不证明没有内部staging。未被采用时应明确拒绝该优化或按实际strides复制并报告，不默默挂零拷贝标签。

## 2. Segmented tensor穿过SwiGLU

N-split projection若给出三段`[gate_d | up_d]`，可以让pointwise消费者直接使用配对segments，避免先拼成完整`[all_gate | all_up]`。gate/up标签、各自span、down列、quant group与source pitch都属于ABI。

三条可测路线：先完整materialize再SwiGLU/down；分段SwiGLU scatter到一个F宽hidden后一次大down GEMM；每分片完成down并返回H宽partial后归约。后两条算子数量、GEMM利用率、通信与scratch不同，没有天然赢家。

本地`segmented-ffn`实现配对segments经过down partial的等价式，使用uneven spans；把gate/up对调的反例不改变shape却改变非线性。若消费者有跨segment的RMS/softmax统计，则不能只独立算局部统计，需要先全局归约。

## 3. Resource identity影响调度粒度

参考工程的特定Metal微基准观察到：一个tracked大buffer的disjoint offsets、同heap的不同subresources，可能触发比数学依赖更粗的序列化；独立allocation的tracked buffers可能更容易并发。这个观察不是所有driver的保证，更不是要求关闭hazard tracking。

先证明真实physical intervals不alias，并记录读写mode与显式依赖；两个逻辑名字不同不证明不同backing。若保留一个allocation的多个views，独立resource wrapper也不能解除真实alias。对untracked资源必须补完整synchronization proof，本目录不自动启用这种策略。

`resource-hazards`用离散事件模拟粗粒度identity与独立lane，同时给出带宽饱和后“overlap更多但wall不变”的对照。`independent_regions`只接受明确allocation身份/字节区间/access mode，拒绝重叠写；它不向Metal提交命令，不能声称测到当前driver行为。

## 4. Binding cache不是数据cache

Bridge可缓存MLMultiArray wrapper、feature provider与prediction options，减少每次构造；缓存pointer只说明下次包装同一地址，不说明内容/source没变。每次预测仍需当前Py_buffer/owner保活、输入producer完成、output唯一借用；no-op deallocator不接管实际allocation。

flux2-engine的复用输出使同engine请求串行；需要并发请求时加独立slots或独立engine，计全部驻留内存，不把同一output数组给两个在途预测。`DualBank`与`Slot`提供本地generation/reader模型；actual consumer完成后才能归还slot。

新shape、stride、padding、consumer、quant group、alias或backing策略改变时重验完整边界和数值。来源内容已提炼到本篇，不需要原checkout；Public native样本另给真实2D shared IO/pitch与实际output处理，不冒充本篇4D SEOP的硬件性能试验。
