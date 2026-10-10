# Qwen GGUF：统一 BF16 高 I/O 前置负载的原 GPU 参考对照

2026-10-10，M4 Max64GB / macOS26.6.2。接续
[有界预读](qwen21-bounded-import-read-ahead-2026-10-10.md)。
完整目标仍 active；本轮补齐之前cold encoder对照的文件缓存状态边界，
没有更改生产算术、模型、参考代码或参考原flags。

## 比较方法

同原Q4_K_M encoder/DiT与原BF16 VAE，512²/40steps/seed29、三个相同
fresh fox-light prompts，全部condition miss。三arm分别为native GPU、
原Unsloth pinned sd-server明确GPU conditioner、native GPU/ANE5120。
每arm启动前，**都实际运行相同的三次原BF16 streaming40step请求**，
prefix28/两个slots/managed13GiB，与之前高I/O四路窗口一致。每次前置
负载单独进程、来源/真实fills/层/pass/fence/drain/完整tree/注册命中验真。

所有运行库、原flags文件与模型的完整SHA在整个window开始前计算；
前置负载完成到target launch之间**只做canonical/stat身份检查**，不做
任何GGUF payload hash、GPU warmup、dummy decode或系统cache purge。
退出后再次整体SHA验证。文件mtime/ctime/inode/size/target有变化直接失败。
首注册已在原window完成，native每target source SHA读取0，不扣除timing
中的真实weights import/prefill/model setup。

这让三个target都经历相同**实际前置工作负载**，不等于保证OS每个page
完全同状态或所有文件缓存状态均已覆盖。不同warmup/external系统活动
仍可能变化，precondition各自完整约135.98/137.26/136.80s（原请求wall
相加），native/原sdcpp路径生命周期也不同；不隐藏这些差异。

native库 `build/qwen21-read-ahead-v1-private`，513source inputs匹配；GPU
encoder prefill+retention，8 CPU decoders，4MiB×2 bounded source read ahead，
原verified-import gate。mixed为原Private Hadamard W8A8、GPU IOSurface、
ANE5120/GPU7168固定渠道，两phase均执行；不使用temporal FFN复用。
参考使用只读`../references/unsloth`原 `metal_text_encoder_flags()` 与
`native_speed_flags("max")`，12threads，实际MTL0 compute/params日志验真。

## 完整40step结果

| route | cold encoder/condition s | warm encoder/condition s | cold DiT/sampling s | warm DiT/sampling s | cold request/generation s | warm request/generation s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native GPU | 1.414193 | .091071 | 43.717930 | 43.712706 | 46.171517 | 44.427732 |
| 原 GPU reference | 1.910000 | .130000 | 108.660000 | 107.300000 | 128.730000 | 125.365000 |
| native GPU/ANE5120 | 1.254466 | .090424 | 36.217170 | 36.168671 | 38.906136 | 36.870464 |

reference最后两列是实际native generate_image完成span（另记录HTTP
128.769888 / warm125.397347s），ready .213537s单独记录。native两列是
实际完整request_wall，含源加载/调度/导出，未用相减拼凑kernel倍率。
各两warm中位数；没有把参考lazy loads、首次native模型加载偷偷扣掉。

相同前置条件下，cold encoder native GPU比参考少约25.96%，mixed少
约34.32%；warm少约29.95% /30.44%。两路DiT和完整请求也明显更快。
这补充了以前拿高I/O后native1.207s与独立预hash状态参考1.18s比较的
不匹配；**保留原旧window，不将它重写为成功**。新window本身证明
当前指定负载下cold/warm encoder与DiT均领先，不推广为任意shape/缓存
状态的保证，也不以它代替Public/hard-RAM或编辑/LoRA质量资格。

## 源与运行证据

六个独立process-tree window（各arm precondition与target）全部complete，
owned reference服务器已退出，没有外部进程被杀或参考文件被写入。
target physical peak native GPU15,472,238,128、reference11,035,124,520、
mixed15,301,189,216bytes；这些不是相同内存的reference比较，BF16同内存
结论仍见前文已完成的原四路window。

三target system compression/swap-in/out均0；GPU/reference decompression
各16,384bytes，mixed655,360bytes。属于系统counter，不归因特定模型。
native GPU/mixed六对PNG与已验证4ahead40step结果逐字节相同；原reference
自身三PNG与此前原seed/RNG结果一致，但**跨engine seed不是同noise**，
不声称native/reference图像bit parity。原W8A8 phase/native calls/source与
fallback/failure/retry全部按实际receipts验真。

新runner `qwen21_matched_cold_reference_screen.py` 不在import时执行模型，
host测试4项通过：generation查询不打开payload、同大小mtime恢复修改/
换inode或symlink目标拒绝、前置与target flags隔离、三个真实请求配置。
初次missing-path负例expectation只接受ValueError，原真实resolve抛
FileNotFoundError，保留host失败日志；统一为明确ValueError而未放宽任何
generation/check/no-read门禁，最终4项通过。

[机器记录](../design/validation/qwen21-conditioned-reference-20261010.json)
绑定原source/两引擎、完整六process trees、native与reference实际完整
请求span、zero-SHA source receipts、前置负载与精准图片一致性、原失败。
