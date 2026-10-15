# 内存与IO实验：从tensor table到有限在途工作

[实验室](README.md) · [IO实现](systems.py) · [header reader](inspect_model.py)

## 1. 真实小文件positional reads

```sh
python3 -B techniques/labs/run.py bounded-io
```

脚本创建自己的小fixture，对比串行读取与恰好一个future的lookahead，checksum全部字节；short read抛错而不是发布partial；`future.result()`显式传播错误。另用private FD演示`dup()`共享内核游标，`pread()`保留独立logicaloffset。

这里Python `os.pread`先产生bytes，不能称native direct-fill/零拷贝；文件很小且OS cache未清，elapsed不用于SSD带宽宣传。生产直接填typedbuffer与取消/error latching见 [ReadAheadWorker](../../native/runtime/streaming/read_ahead_worker.hpp)和 [SeedVR2 reader经验](../../docs/design/seedvr2-block-weight-residency.md)。

## 2. 元数据先行，不为算预算加载整个模型

```sh
python3 -B techniques/labs/inspect_model.py "$TC_SAFETENSORS_SHARD" --block-prefix blocks. --output outputs/my-header.json
```

实际只读8-byte length和有cap的header：重复JSONkey、未知dtype、shape溢出、overlap/gap、越filebounds和source stamp变化都拒绝。统计每block原payload、widestblock、non-block与dtype；记录fileoffset的16/4096对齐。

单shard元数据不是完整模型来源证明，headerhash不是checkpoint SHA，U32仍可能是不同packedformat；多shard按index/tensor identity合并，别把每shard的trunk都重复算一遍。

## 3. Phase与allocation去重

```sh
python3 -B techniques/scripts/interval_peak.py techniques/examples/memory-plan.json
python3 -B techniques/labs/run.py slot-ownership
```

区间包含conditioning/latent跨phase存活；alias使用完整backingbytes。pipeline并发多个request时，应展开联合timeline，不能继续取串行phases最大值。CoreML surfaces/driver、allocatorcache、媒体output、scratch与系统reserve另有项。

## 4. 适合扩展的实验

改变blockbytes、槽数、lookahead、fixedpinnedsubset、coldcache/OS cache状态、量化payload与decode位置。LRU对循环scan差的条件是C<N且访问基本顺序；有真实非顺序/热专家访问时LRU也可能合理，应先用访问trace验证。

不同源格式可比较mmap/no-copy、copied/pread、page-aligned staging和typeddequant；不全局宣布哪个总好。每个case包括真实payload范围、temporarycopy、owner、lastreader、error与cancellation。内存收益至少报MLX和process-tree的独立口径，预算planner只是预测。
