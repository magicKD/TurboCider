# MLX外部内存：尝试导入、共享view与copy的物理区别

<!-- knowledge
id: source-upstream-mlx-adoption
type: source
-->

[来源索引](../README.md) · [共享backing机制](../../gpu-ane/shared-surface.md) · [本机adoption证据](../../metal/case-studies/external-memory-adoption-and-fresh-views.md)

固定读取`ml-explore/mlx@7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`（v0.32.0），MIT、Copyright Apple Inc.；文件内容SHA和选定阅读范围在[同一来源登记](../README.md)。参考checkout只读，不分发其实现；以下是原创概括。

本机执行另绑定已安装0.32.0的383个headers和libmlx/libjaccl/metallib内容hash、OS build/SDK/compiler，以及实际buffer/设备相等bool。**同版本号不是wheel由此tag构建的证明**；source-inspected和installed-behavior-measured保持独立。

## 入口并不保证采用

`array(void*, Shape, Dtype, deleter)`先调用`allocator::make_buffer(data,nbytes())`；返回nullptr则另分配、复制并立即调用caller deleter，成功时包装release再延迟调用deleter。因此“调用了这个构造函数”不能当实际alias证据。

`allocator::make_buffer`有明确no-copy/null失败合同；这种buffer必须经`allocator::release`释放，不是普通`free`，更不能把CPU地址直接塞进`Buffer`构造函数。Metal backend的`Buffer::ptr()`是Metal buffer对象，`raw_ptr()`才取contents；本轮代码仅在明确Metal backend/版本和真实alias检查下访问，不作为跨后端稳定ABI。

`MetalAllocator::make_buffer`用既有区域创建shared、untracked-hazard buffer并加入residency bookkeeping。`release`撤销bookkeeping和Metal引用，不替调用方释放IOSurface owner。外部queue写入和MLX stream之间的完成依赖必须由接入方建立；IOSurface CPU lock不是跨队列GPU fence。

## View与Copy不是同一层的“新对象”

`as_strided`先flatten输入，`AsStrided::eval`要求row-contiguous的原输入，再`copy_shared_buffer`并保存stride/offset。应先包装整块合法page region，再从flat构造已知逻辑view；不能把任意非连续输入交给flatten后就假定仍alias。

`Copy::eval`在这个固定版本直接`out.copy_shared_buffer(inputs[0])`。它产生另一个逻辑array，不承诺独立物理快照；这在不可变array语义内成立，但外部设备会改写backing时尤其重要。本轮首个客户端错误地把`mx::copy`当独立snapshot，实际指针相同而严格gate失败；保留原件，不称MLX硬件不支持或bug。

新的consumer从live view或same-dtype Copy读到新generation；已求值consumer array不会因外部writer而自动失效。不同dtype的FP16→FP32 materialization可以成为本轮独立backing快照控制，但必须实际检查pointer不同并完成求值，不能只凭名字或图节点存在。

## 本机范围与迁移

两个import模式、两个seed、新进程、padded `[32,33]`/pitch64和完整`[1024,4096]` FP16逻辑8MiB范围均有实际alias、全输出hash、owner/view/deleter与代际结果见证。后者仍是明确period-37合成值，不是原Qwen激活、真实FFN图或性能分数；本轮没有ANE调用。

接真实FFN时仍需按实际private二维图descriptor做GPU pack/cast及restore/join、保留完整BF16 GPU分母与source/FP64预算。不要将此处8MiB adoption与上一轮独立private surface toy相乘成一条未跑过的端到端路径。

[固定constructor源码](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/array.cpp) · [固定Metal allocator](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/allocator.cpp) · [固定AsStrided/Copy](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/common/common.cpp)。版本、dtype/layout、owner、外部producer或graph缓存政策变化后重验。
