# 私有接口共享surface：实际backing、完整消费者与有限错误

[总目录](../README.md) · [机制/资格边界](../gpu-ane/shared-surface.md) · [官方来源](../sources/official/shared-memory.md) · [全部证据](private-shared-evidence-20261016.json)

2026-10-16，Apple M4 Max/64GiB、macOS26.6.2、SDK26.5、Apple clang21；Python3.9.6只做标准库准备/监督/核验。没有导入ANEForge、MLX、NumPy或产品runtime，也没有更改参考checkout、模型、App、系统保护或电源设置。

## 配方与身份

原创MIL为FP16 `y=2*x+0.25`，Metal producer逐元素生成有新generation/seed的精确dyadic，Metal consumer计算FP32 `3*y-0.5`。最终完整期望是`6*x+0.25`，没有真实checkpoint/激活。四个新进程使用相同wrapper/native源码；原件每run保存source、MIL、binary、native JSON与owned-child终态receipt。

从实际NetworkStatusList读取输入/输出NCHW和字节stride；C32/W33为128/128/4096，C64/W65为192/192/12288。每个surface分配32768 bytes，页大小16384，payload之后至少一页guard。Metal no-copy包装失败即拒绝；未使用“指针相同所以全系统无copy”的推理。

初始、每call前后与最终，实际request inputArray/outputArray的ioSurface ID均与目标相同，Metal contents均等于实际surface base；两个feature surface不同。公开只存五项bool，不发布raw ID/地址。producer/consumer每个owned command成功完成，private同步evaluate后才读输出；验证与unload完成后释放owner。所有四进程均正常退出，没有触发quarantine；失败分支有保守实现，但实际取消/失败回收尚未获实测资格。

## 全部正、负控制

每短run为两arm×(first1+warmup2+measured8)=22次；长run为两arm×(1+2+24)=54次，共120次、三个完整逻辑tensor hash/call。独立标准库struct/数学重建全部FP16输入、FP16 ANE输出与FP32 GPU消费者hash，逐一相同；每call输出poison覆盖、完整数值和尾guard通过，不只检查抽样。

另有每run两个实际GPU消费者负控，共8次：紧密stride错误仍全finite，但C32/W33有1011/1056错误；C64/W65分别3991和3990/4160错误。直接消费上一代真实输出却按下一代验证，1056/4160个元素全部拒绝，最大绝对误差12。负控后消费者尾guard仍完整；这些调用不计入120次正样本，也不进入ratio。

## Host诊断：不授予稳定加速

边界为GPU producer encode→GPU consumer完成，三段owned host事件顺序串行。copy控制额外包含两次CPU feature memcpy和相应lock/unlock，总量分别8192/24576 bytes；shared arm在此边界内外部feature copy数0。初始化/poison和CPU验证仍在边界外，runtime内部copy保持unknown。compile/load/pipeline prepare均不在warm计时，不声称cold请求收益。

| shape / seed / measured组 | shared wall中位ms | copy wall中位ms | 诊断paired copy/shared中位 |
| --- | ---: | ---: | ---: |
| C32/W33，17，8 | 0.301750 | 0.324709 | 1.06332 |
| C32/W33，29，8 | 0.327313 | 0.331708 | 1.01050 |
| C64/W65，17，8 | 0.371813 | 0.373896 | 0.98936 |
| C64/W65，29，24 | 0.349438 | 0.339605 | 1.01093 |

paired中位不等于两组中位之比，最后一行符号甚至不同；两seed/shape并无稳定优势。两次copy host窗口合计中位量级仅约2.65–3.56μs，相对约0.3ms完整小图边界不是主项。这里没有optimized GPU-only分母、物理ANE时钟/placement、thermal长时或并发，不把少量copy消失称为GPU+ANE加速。

本批报告采到OS product version、SDK/compiler字符串、GPU名称和private selector encodings，没有另采OS build、runtime framework binary hash或逐arm独立峰值；不回填为当时已采，也不授予严格跨版本复现/内存资格。

## 复核与采用范围

[只读curator](../scripts/curate_shared_surface.py)只读明确run，复核source/MIL/binary/native及退出receipt；[独立合同](../labs/shared_backing.py)重建全部逻辑hash、顺序/时钟/复制量/负控与数值。它不重跑硬件，不是attestation。curator/helper自身内容hash与全部四run native源码身份由[离线反例测试](../tests/test_shared_backing.py)绑定。

将外部alias/stride/owner的know-how纳入知识库，**不改产品默认**；原[真实Qwen copy-bridge](qwen-private-ffn-20261016.md)仍不是shared实现。本轮尚未证明MLX external adoption、真实二维FFN端口布局、完整source/Qwen精度与成本、LoRA/请求质量、异步取消失败/重用或物理并行。

原件及raw日志保持ignored；没有删旧实验/模型/系统缓存。本轮后磁盘观察约0.3GiB，不据此归因所有空间变化，也不继续新增大图/权重副本。完整目标继续沿[工作台](research-campaign-20261015.md)推进。

后续更正范围说明：同日另有[真实Qwen完整shared FFN](qwen-shared-ffn-20261016.md)实测，而不是把本页toy成绩外推；本页原始protocol/数字和不确定项保持原义，不回填成真实模型证据。
