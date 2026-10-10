# Z GGUF：兼容读取不得改变原模型代次

2026-10-11，M4 Max64 GiB/macOS26.6.2。为补齐base格式benchmark，本轮
构建同源Private/experimental库，预检BF16、NVFP4、ConvRot、Q4/Q8的
GPU/Public/Private配置。发现并修复了一个原GGUF文件保护问题；本页
不是新的整模型加速倍率，未把不完整窗口或GPU fallback当混合收益。

## 问题与独立复现

8步正序窗口完成前八条路线后，Q4 stock GPU虽然生成四张完整PNG，
parent generation guard发现原文件ctime变化而失败。独立重跑Q4 stock
再次出现相同变化。mtime、inode、size、权限与provenance xattr保持；
完整SHA仍为`687360c35c468ae1054f8a848d40a6c852c3c67d81f962dafcfbeff09e66d7d0`。
没有推断用户改了权重，也没有忽略ctime、删除坏样本或追认完整性能窗口。

独立Python MLX0.32.0 `mx.load(original.gguf)`同样改变ctime；与native
模型算术无关。原实现调用`mx::load_gguf(path)`。Apple MLX v0.32.0
`mlx/io/gguf.cpp`委托`gguf_open`；其引用的gguf-tools
commit`af7d88d808a7608a33723fba067036202910acb3`实现为：

```c
open(filename, O_RDWR | O_APPEND);
mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
```

这也要求源可写，不适合作为不可变/只读模型artifact的加载边界。
依赖源码只下载到本仓库忽略目录，没有编辑MLX安装、参考仓库或原模型。

## 修复范围

`GgufReadSnapshot`从native held read-only SourceLease fd生成owner-private
压缩文件快照：APFS使用`fclonefileat` CoW，不提前解码或复制dense模型；
其它情况以64KiB heap buffer做压缩字节copy。只有独立的临时inode可设为
0600并交给upstream可写reader，目录0700。clone/copy后与load返回后均
复验原fd/路径代次；缺失、替换或原模型变化仍拒绝。

native执行类型门在这个实际快照上检查，避免先验证旧path再从新generation
读取的缺口。成功/异常时只删除本次自建临时文件/空目录，不递归清目录，
原模型从不chmod、写入或删除。不创建永久sidecar，也不授予内容proof。
非CoW fallback需要临时磁盘容量/复制I/O，不能宣传所有文件系统零成本。

兼容decoder与codes/scales/浮点dtype和GPU算术不变；已有CPU-direct
packed reader仍是原只读路线。本修复不改Qwen的首次SHA注册政策，不新加
每个inference的完整hash，只加小量文件身份复验。

## 当前验证与保留失败

- 独立APFS clone + `mx.load(clone)`，原Q4 inode/size/mtime/ctime全部不变。
- native host fixture覆盖源0444、clone及强制bounded-copy、0600/0700、
  writable shared mmap故意修改快照不影响源、原变更拒绝、cleanup与missing源。
  初版和最终heap-buffer版各一项完整fixture通过。
- 新匹配库`build/z-base-formats-v2-readonly-private`，Private1、probes1，
  514native输入由构建流程绑定。Q4/Q8两步、每路一cold/two fresh/repeat，
  八张完整PNG、两份memory verifier complete，parent全部源代次保持。
- Q4新预检四张PNG像素与原failed-window中的四张逐位相同。旧窗口的失败
  是source guard，不是数值错误；这里只将其用于同源输出回归，不借作性能分母。
- 本轮format host5项、winner/audit host4项通过。未跑全仓make test。

另一个预检曾把ConvRot Public误配Sylvester图，导致实际runtime失败后
GPU fallback；validator拒绝成绩。改用已有Comfy H256图后Public与两种
GGUF的GPU/Private/Public两步预检完成，没有放宽失败/finite/basis检查。
NVFP4实际GPU路径成功，但runtime原生门明确GPU-only，继续标N/A。

原失败日志/PNG和独立复现全部保留：

- `outputs/z-base-formats-v1-precheck-20261011/`：错误Public basis。
- `outputs/z-base-formats-v3-8step-forward-20261011/`：Q4代次变化，非完整对照。
- `outputs/z-base-formats-v4-stock-precheck-20261011/`：相同ctime问题重现。
- `outputs/gguf-mlx-generation-q4-v2-20261011.log`、
  `outputs/gguf-mlx-clone-source-probe-20261011.log`：独立reader/clone复现。
- `outputs/z-base-formats-v5-readonly-precheck-20261011/`：新库实际source保持。

完整8步、所有16个声明配置的正反序正在用新库重跑。只有两窗口完整、
actual source/recipe/calls与memory验证通过后才更新benchmark的正式诊断
表；不从被中止的不同library/window拼接“最佳”。整体目标继续active。
