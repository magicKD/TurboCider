# Apple文件clone：隔离常量资产，不当作设备零拷贝

[来源索引](../README.md) · [真实共享FFN](../../review/qwen-shared-ffn-20261016.md) · [存储合同](../../memory/storage.md)

2026-10-16读取Apple现行Foundation文档和本机SDK26.5的`sys/clonefile.h`；URL、内容SHA和选定范围在[sources.json](../../references/sources.json)。不是镜像Apple headers/文档，也不把其许可变成Wiki MIT。SDK header带APSL2.0声明，本目录只作原创概括/调用。

## 公共合同与本轮策略

[About Apple File System](https://developer.apple.com/documentation/foundation/about-apple-file-system)说明同卷clone共享未修改blocks，后续修改另写；这是文件系统机制，不是GPU/ANE feature接口的alias。[volumeSupportsFileCloning](https://developer.apple.com/documentation/foundation/urlresourcevalues/volumesupportsfilecloning)只表示卷是否支持clonefile(2)，不替代对具体路径/文件/权限的实际错误处理。

Foundation示例中的高层copy在非APFS情形可能转成普通复制；若存储admission预算依赖clone，不能仅凭函数名假定没有完整副本。本轮客户端直接调用`clonefile`，新target、NOFOLLOW、失败不做普通copy fallback；还检查新inode不是hard link，并核对完整MIL/blob SHA。没有修改旧常量或原checkpoint。

## 不能从clone推出什么

新inode/hash相同不等于独立物理blocks；clone成功不证明private compiler不会另写临时文件/cache。`du`可能给多个clone各计逻辑占用，APFS container可用空间也会变化；本轮没有用`df`增减反推实际共享bytes、峰值或归因全部外部空间变化。

clone不免除metadata/新写块/编译产物的预算，不等于硬件feature零copy。新clone/compile与只读prepared/load是两条明确流程，各自admission；后者省去客户端clone和compile selector，也不证明runtime内部无编译或无setup。
