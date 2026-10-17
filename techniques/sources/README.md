# 固定版本来源与源码阅读

[知识库](../README.md) · [怎样读来源](../references/evidence-map.md)

按主题找固定 revision、关键文件与阅读重点；不附机器清单、文件 hash 或实验数据，不声称是最新版本。

- [Metal / GPU](../metal/sources.md)
- [ANE / Core ML](../ane/sources.md)
- [GPU+ANE 协同](../gpu_ane/sources.md)
- [算法与论文](../algorithms/sources.md)

## 专题解读

- [Apple文件clone：隔离常量资产，不当作设备零拷贝](official/clonefile.md)
- [Apple公共Metal counter与capture：先核查能力和时钟](official/metal-counters.md)
- [Apple Metal 4 counter heap、反馈与资源生命周期来源](official/metal4-counter-heaps.md)
- [Apple Metal与IOSurface：外部共享内存合同](official/shared-memory.md)
- [Accelerate/vImage：像素排列、方向与非原地缓冲合同](official/vimage-stage.md)
- [measure_ane_capacity：序列化、fallback 与 PMU 计分](repos/ane-capacity-pmu.md)
- [ane_pmu_profiler：先验 payload，再解释寄存器](repos/ane-pmu-provenance.md)
- [ANEForge：结构搜索、精度预算和 e5rt 边界](repos/aneforge-rewrite-search.md)
- [anemll-profile：先审运行副作用，再读性能数字](repos/anemll-profile-safety.md)
- [Apple ANE Transformers：迁移布局，不替换模型数学](repos/apple-ane-transformers-ffn.md)
- [ccv：图内K分块、surface缓存与output backing](repos/ccv-coreml-rowwise.md)
- [coremltools：计算计划不是硬件逐算子计时](repos/coremltools-compute-plan.md)
- [GGML Metal：块布局、实际摘录执行与tensor政策](repos/ggml-metal-layouts.md)
- [gputrace：时间来源比漂亮时间线更重要](repos/gputrace-timing-provenance.md)
- [KernelWiki检索与版本复查：借合同，不搬硬件别名](repos/kernelwiki-retrieval.md)
- [MLX affine：dispatch、metadata精度与原始模型基线](repos/mlx-affine-dispatch.md)
- [MLX外部内存：尝试导入、共享view与copy的物理区别](repos/mlx-external-memory.md)
- [MLX SDPA：guard、stride、mask与输出布局](repos/mlx-sdpa-dispatch.md)
- [vpipe：图内K/N tiling与runtime权重布局](repos/vpipe-runtime-tiling.md)
