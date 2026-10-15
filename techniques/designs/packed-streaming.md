# 不可变权重与 typed refill：格式、消费者、owner 一起设计

[设计目录](README.md) · [本地实现](../labs/contracts.py) · [实际 pread 对照](../labs/systems.py)

## 两种路线及适用条件

Splash 的自有 aligned pack 让 format-specific GPU kernel 直接消费；vpipe 的 typed positional refill 更适合通用 checkpoint 中不满足 no-copy 条件的 tensor。应吸收双方的合同，而不是宣布 mmap 或 pread 总更快：

| 路线 | 好处 | 必须支付 / 验证的成本 |
| --- | --- | --- |
| 只读共享映射 + GPU view | 减少显式复制，可让多个 view 共享 owner | 页/offset/pitch 对齐、Metal no-copy 准入、fault/驻留、文件不被原位修改 |
| 有界 pread 到执行 buffer | 生命周期明确，可做 dtype/布局变换 | SSD/OS-cache 读成本、scratch、转换、ready ticket、失败传播 |
| 完全常驻转换后权重 | 热循环不读原盘 | 转换副本可能比原 pack 大很多；退出时可能仍有 async reader |

## 本地 pack 的完整合同

`PackedStore`使用特意命名的教学格式 TCLAB01：magic、bounded JSON header、version/endian、tensor name/shape/dtype、相对 payload offset/bytes；payload 起点与 tensor offsets 按 64 bytes 对齐。它检查大小、shape×itemsize、重复名、overlap 和 bounds，再发布只读 view。

Backing 是不可变 bytes；view 自己保住完整 backing，而不是只保一个裸指针。`source_sha256`能区分内容，但不是授权或签名。生产 native 还要检查整数加乘溢出、实际 FD 身份/稳定来源、mmap 对齐、文件被替换/修改、完整 metadata 与 scale 表；不要拿路径字符串当 source identity。教学 64-byte 对齐不保证任何真实 no-copy API 的页对齐准入。

```text
验证 source -> 验证 tensor table -> 创建 owner -> 创建带 offset/pitch 的 view
          -> 执行消费者 -> last-reader completion -> 回收整个 backing
```

Packed Q4/GGUF 等 formats 还要绑定 group size、bit order、scales、zeros、full physical row stride 和消费者的转置方式。局部 slice 的宽度不能替换母矩阵 pitch。[quantization.py](../labs/quantization.py)提供具体错误对照，不把 TCLAB01 冒充原项目文件格式。

## Typed refill 不能只看字节宽度

本地 `typed_refill`显式区分：`ready`（完整 payload）、`unservable`（该转换不支持，可走独立慢路径）、`failed`（短读/数值转换失败，绝不能发布半个结果）。F16→BF16 是浮点解码和 RNE 重新编码；虽然同为 16bit，直接 reinterpret 是错的。已经是 BF16 的 scale 也不能再做一次 F16 转换。

真实 FD 读取用 [read_exact](../labs/systems.py)：`pread`携带显式 offset，循环处理短读和 EINTR，EOF 抛错；`dup(fd)`仍共享普通 seek cursor，所以换一个 duplicated FD 不等于隔离 `read()`位置。一个 future 的 bounded read-ahead 在取结果时重新抛出 producer 异常，不以 `wait()`结束冒充数据有效。

## 有界预读的算法

设当前执行 block i，最多允许一个 future i+1。future 拥有读出的 payload，consumer 的 last reader 完成后才能复用 slot；最后一层不再提交 future；取消时先 stop 新任务，再 get/drain 当前任务，保住 source FD。若 staging 使用 GPU，还要有实际 GPU ready，而非“host 已完成 memcpy”。

IO 较慢时 lookahead 可能隐藏部分等待，但两个设备同时读写同一统一内存也会争抢。必须比较相同缓存状态的完整 layer/request，不把“prefetch hit”直接换成收益。

```sh
python3 -B techniques/labs/run.py packed-weight-store
python3 -B techniques/labs/run.py quant-formats
python3 -B techniques/labs/run.py bounded-io
```

测试覆盖损坏 header、overlap、越界、不同 dtype、short payload、只读 view 和 owner 保活。重新探索条件：新格式、offset 对齐、source encoding、consumer pitch、buffer pool、SSD/内存条件。设计来源见 [吸收账本](../references/absorbed-designs.json)，不需要其 checkout。
