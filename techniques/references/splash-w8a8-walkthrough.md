# Splash W8A8实现导读：划分、分段、overlap与固定开销

[总目录](../README.md) · [Splash概览](splash.md) · [量化实验](../labs/quantization.md)

来源是本地只读`../references/splash-ffn`、HEAD `80827e1c3593f7d10c266394e4d4381321b73ad0`。本轮读取AneFfn.cpp、Measurement、Calibration、Handoff真实文件；源码里的特定机器/系统观察仍是作者记录，不提升为本库所有芯片事实或新测量。关键文件hash在`sources.json`。

## 1. 从projection planes到同一logical FFN

`AneFfn::unsupported/units/gpuChannels/segments`先检查模型几何和affine/GGUF plane可用性。GPU保留一部分intermediatechannels，ANE计算另一部分gate/up/SiLU/down，最后相加hiddenpartial。

这版代码用2560-channel segments及与rotation/packing兼容的channelunit。它是该实现为真实shape选择的图粒度，不是所有ANE输入必须2560对齐的硬件定律。新模型可探索其他分段，但要写新graph/stager/ABI与对照。

## 2. W8与A8的basis和scale

GPU从原packed planes准备codes/scales，输入与gate/up权重采用匹配的inputrotation，hidden/down用intermediate rotation。图内保留完整非线性与hidden量化，down输出恢复各token/row scales。

特别注意：该Splash代码使用固定`std::mt19937`生成signs，TurboCider的Sylvester配方使用显式unsigned64mixing。seed数字相同仍可能不同；移植要逐个sign、basis方向、normalizedscale、FP16round和常量blob核对，不靠“都叫Hadamard”。

可以运行 [CPU normalized W8A8](../labs/quantization.py)理解比例因子，再看 [Public实际shared IO](../labs/ane/shared_w8.py)。CPUreference不代替原源码的细粒度FP16lowering。

## 3. 两套W staging与ready/done

源码在当前layer执行时准备下一层W，shared event把本layer inputpack→ANEjob→GPUjoin排序。必须同时保住sourceallocation、codes/scales、program与outputs；“预取命中”不释放当前reader，也不能用旧bank代替failedproducer。

`Handoff`跟踪queued/started/succeeded/failed，ready到达后开始deadline；报告只消费一次，retire后解除设备等待但当前command结果丢弃。**解除等待不是输出有效或surface可提前复用的证明**；request仍须保活未完成owners。TurboCider的failureflag/quarantine/physicalcompletion合同要单独核对，不直接复制signal策略。

## 4. 为什么Measurement不只测一个孤立算子

`Measurement::layer`在split layer前加入与真实mixer窗口相似的GPU stand-in，避免把不真实的调度唤醒模式当模型成本，再扣除相同控制；`gpuPart`比较1层与多层以摊销command submission和首层Wstage。它给出测量设计思路，不证明任何相减都有效。

对新模型先定义相同boundary和producer完成、记录原样本；如果stand-in不再代表真实attention或thermal/load变化明显，可改用真实层链/多点回归/device trace。

## 5. 校准、rowthreshold与运行中breaker

`Calibration`使用两档share与fullGPU控制拟合`T(s)=max(A(s),G(s)+u*A(s))`，该模型含highshare由ANE主导等经验假设。之后选择nearoptimum、最小有益rows，并用一段commands的窗口观察split是否继续值得。

TurboCider采用 [保守成本模型与实际候选试验](../gpu-ane/calibration.md)，不照搬某台机器的u、share或rowthreshold。Bucket、segment、LoRA、weightencoding改变时重新采样，而非把一次长prefill slope套给所有短decode。

## 6. 冷启动、idle与内存

programcompile/load、冷firstfunction、reload、IOSurface与GPUstagingbanks都是不同生命周期；idle unload可以释放ANEprogram但未必释放本方scratch。多rowfunction减少padding税，也增加图/编译/prepare成本，需记录完整磁盘与内存。

历史会话进一步提醒：全层W8/W4导出、编译cache和activationcaptures可能各占多GiB；事先估量，不因缺空间盲目再导一套，也不擅自删跨项目共享cache。表观文件大小不等于实际可回收空间。

## 7. 有意义的后续探索

分段宽度、multifunction bucket、W4A8/LUT、SmoothQuant S1、groupA8、sharedoutput、LoRA ranks、nativegraph deeperoverlap、sourceverified cache和idle策略都可继续研究。每项从一个明确变更条件和匹配控制开始；本版某个shape/quality失败不关闭方向，也不放松finite/source/completion底线。
