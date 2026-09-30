# Runtime ANE：Qwen 1024² 第二提示词三路对照

接续[同库狐狸/seed42 内存准入对照](runtime-ane-memory-admission-2026-09-29.md)
和[512² 双模型肖像](runtime-ane-portrait-and-z-chunk-2026-09-29.md)。
本组改为 Qwen-Image-2.1 BF16 base 1024²/40 步，成年女性肖像/seed43；
没有更换原生库、模型、runtime/冻结图、Q/K 融合设置或默认路由。
它补第二提示词的质量/性能证据，不是广泛人像、文字或编辑验收。

## 设计与身份

M4 Max 64 GB；prompt 是窗边自然光、深色卷发和雀斑的成年女性、绿色毛衣。
与先前 512² 肖像使用同一完整 prompt/seed，但本组重新运行全部三个分母。
GPU→runtime→冻结→冻结→runtime→GPU，共六个独立 resident trial、18 个
请求，每 trial 一冷两热。三路统一显式开启 GPU Q/K norm-RoPE；runtime
graph-v2 c1792/K1024/N512，base 传零 LoRA 修正、auto chunks；冻结图是
已有 4096-row W8A8 base，非同一 runtime-weight 接口。profile 关闭，
整请求 `request_wall` 含 VAE/PNG，不含加载/冷请求。每路启动前检查竞争
推理；预检不能保证整个窗口独占设备。

原生库 SHA256：
`f3f6b6a05ce6eecb476f314f1fa1fccbdb1d2ef24fc09fdc5b72049b0390ca22`；
runtime manifest SHA256：
`308ca0b8f423956a871728c194cefc271d07f04ae93afd85aaf415b78c7b3a9f`；
冻结 manifest SHA256：
`1d0b913e073fc44130c667a55f40c715622f7eb783e3e32ee7707250c6adb509`。
运行器 SHA256：
`7aeae67fd1d38fadfb7e33fba2683ce5414ab0460f0cce8e4be08ef74150f499`。
原始 summary 与 18 份请求/JSONL 核对了 1024²、40 步、seed43、
无 LoRA、实际路线和图身份；运行库前后 SHA 不变。

## 结果

| Trial/路线 | 冷请求 | 热 1 | 热 2 |
| --- | ---: | ---: | ---: |
| 0/GPU | 185.945 s | 183.151 s | 183.172 s |
| 1/runtime | 152.604 s | 148.025 s | 148.841 s |
| 2/冻结 | 163.099 s | 152.435 s | 152.248 s |
| 3/冻结 | 164.378 s | 154.079 s | 153.298 s |
| 4/runtime | 153.140 s | 146.958 s | 147.997 s |
| 5/GPU | 185.393 s | 183.160 s | 183.135 s |

每路四个热请求池化中位：GPU **183.156 s**、runtime **148.011 s**、
冻结图 **152.866 s**。同组 GPU/runtime 为 **1.237×**，GPU/冻结图为
**1.198×**，冻结图/runtime 为 **1.033×**。四个 runtime 热请求均短于
四个冻结热请求；本机第二提示词仍由显式 runtime c1792 最快。
不把狐狸/seed42、512²肖像或 c1664/c1792 chunk 消融的样本拼入本组中位。

两个 runtime trial 各累计 3,648 次真实 prediction、3,648 hybrid blocks、
192 次正常完整 GPU probe；无 runtime failure、错误 fallback 或 overflow
retry。两个冻结 trial 各有 3,744 次 Core ML calls、无调用失败。
GPU 没有 Core ML calls。18 张 PNG 均存在，总 summary complete。

六条 100 ms 进程树采样流重新独立验为 complete，合计 29,428 样本，
最大间隙 111.116 ms；无新增系统 swap-in/out。两个 GPU 和两个 runtime
窗口没有新增系统压缩；冻结窗口分别有 2.085 GB、100.090 MB 系统
compression，以及 1.610 GB、55.181 MB decompression（十进制）。
用采样流 wall time 与冷 PNG 写出时间核对，这些 compression 在各自冷
PNG 写出时已达最终值，两次热请求期间没有进一步增加。这不等于热计时
完全不受冷请求后的内存状态影响，更不能将系统压缩归因于冻结图/ANE。
进程树 footprint 峰值 GPU 31.560–31.630 GB、runtime 32.039–32.132 GB、
冻结 39.931–40.261 GB；范围含加载/冷/热/退出，不覆盖外部服务和全部
driver/wired 内存，不能作为低内存机型资格。

已实际打开反向 GPU/runtime/冻结热图：成年女性主体、眼睛、卷发、雀斑、
绿色毛衣与窗边构图均保留；GPU/runtime 肉眼很接近，冻结图的面部与发丝
有局部差别。相同路线正反向末次热 PNG 各自字节相同，跨路线 SHA 不同；
不要求逐像素一致，也不以一条肖像证明广泛视觉质量或编辑/LoRA 正确性。

本地被忽略的原始依据：
`outputs/runtime-ane/qwen-base-portrait-memory-admission-1024-2026-09-29/`
（完整请求、JSONL、PNG、六份进程树采样/报告及 summary）。
文档更新后 `make test-acceleration-contract` 沙箱外退出 0：5 host、
10 placement、42 screen、6 memory screen、17 switch、12 repository 检查；
`git diff --check` 通过。这些契约测试不代替上述真实生成和采样。
这个对照加强了本机 1024² BF16 base 的显式 runtime 优势证据；
默认仍不改，512²最快仍是匹配冻结图。物理 ANE 驻留、真实低内存、
Q4_1、更多提示词和编辑/LoRA 的资格继续未完成。本组未修改或重建产品库，
未重跑完整 `make test`。
