# Apple Metal与IOSurface：外部共享内存合同

<!-- knowledge
id: source-official-shared-memory
type: source
-->

[来源索引](../README.md) · [实际backing技术页](../../gpu-ane/shared-surface.md) · [本机验证](../../gpu_ane/case-studies/surface-strides-and-stale-output.md)

2026-10-16读取Apple官方文档Markdown与本机SDK26.5 headers；原始内容不镜像到Wiki，URL、内容SHA与SDK相对路径登记于[sources.json](../README.md)。公共API知识不覆盖私有`_ANE*` ABI，也不是对所有芯片/图的性能保证。

## 已核查的公共合同

- [makeBuffer(bytesNoCopy:length:options:deallocator:)](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29)：包装既有连续内存，起点与长度形成页对齐区域，原allocation须在单一VM region；返回nil表示失败。nil deallocator只是由调用方继续管理owner，不会自动证明其生命周期安全。
- [IOSurfaceGetID](https://developer.apple.com/documentation/iosurface/iosurfacegetid%28_%3A%29)：取得surface身份。SDK的`IOSurfaceLookup`注释说明同一buffer可由不同`IOSurfaceRef`对象表示，等价判断应比较ID或`CFEqual`，不能只比较Ref指针。
- SDK的`IOSurfaceLock`/`IOSurfaceUnlock`注释将lock界定为CPU访问边界，且注明操作并不便宜；不是GPU/ANE全局device fence。这里所有GPU完成显式检查，private同步返回后的正确性再由GPU消费者验证。
- `IOSurfaceGetAllocSize`与`IOSurfaceGetBaseAddress`给出mapping范围/地址；实际tensor的row/plane/batch stride来自private模型descriptor，不从surface allocation大小或Metal buffer长度反推。

## 本机证据与不确定项

原创客户端实际比较request的inputArray/outputArray所返回ioSurface与Metal mapping，四个新进程120次调用全量数值通过，padding/stale负控拒绝。公开证据只含相等bool和内容hash，不包含raw ID、地址或原模型数据。

这证明**所测外部feature接口**的同backing交接，不证明runtime内部没有复制、不证明ANE物理放置/并发，不替代真实Qwen的layout、MLX adoption、成本和质量验证。selector身份只在本机运行报告中做版本绑定，不冒充官方文档。
