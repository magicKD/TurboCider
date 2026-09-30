# Runtime ANE：第二提示词与 Z-Image c416 组件筛选

2026-09-29，接续[内存准入和同构建狐狸对照](runtime-ane-memory-admission-2026-09-29.md)。
本轮将第二个 base 提示词扩至两个主力模型；另试 Z 的更大 chunk，
只在组件窗口更快时才值得整模型验证。普通 GPU、冻结图、runtime
默认选路及产品库均未改。

## Z c416/K1024/N512：没有组件收益

使用公开 Core ML exporter 新导出 graph-v1、H3840/F10240、exp SiLU、
K1024/N512、c416 图，保留既有 c352 同接口图。新图没有 checkpoint
权重常量。`make build-runtime-ane-probe` 成功；独立探针 SHA256 为
`c992325af819b4a52467fffd50ee47804facbe5336371114aea904a72f05b7a8`。
Z BF16 真实 block0 权重、探针固定 seed41 的**合成** 1024-row 输入；
这不是捕获的生成 activation。每 trial 排除两个 warmup，交错测十次
GPU/并行 FFN；c352→c416→c416→c352，共四个独立进程、每配置
20 个实测样本。GPU head 分别为 672/608 rows；因而同时改变 chunk
几何和分区，不把差额只归因为 dispatch。计时没有 attention，含 staging
列是完全暴露的 CPU 转换，并不代表整模型 staging 不能被隐藏。

| 配置 | 完整 GPU FFN | Core ML prediction | 并行 FFN | 并行含 staging | GPU 完成后的 join |
| --- | ---: | ---: | ---: | ---: | ---: |
| c352 | 16.495 ms | 10.977 ms | 11.894 ms | 14.064 ms | <0.001 ms |
| c416 | 16.504 ms | 12.991 ms | 13.736 ms | 15.942 ms | 2.723 ms |

并行 FFN c416 比 c352 慢约 15.5%，含 staging 慢约 13.4%。
四 trial 的图形状、失效恢复及真实权重探针均退出 0；各自两次 warmup
有 headroom retry，之后固定 scale16，全部 40 个计时样本每次恰好一次
prediction、无 retry。relative L2 c352 为 0.001369，c416 为
0.001541；配置处理的 ANE rows 不同，误差不能简单视作同一数值样本。
本次不运行 c416 整模型、不替换 c352、不改变产品代码。组件不证明
实际物理 ANE 驻留或图像质量。

原始逐样本、前后内存快照、预检、图/权重/探针/工具 SHA 与 complete
汇总：`outputs/runtime-ane/z-c352-c416-n512-component-2026-09-29.jsonl`；
驱动在同目录 `.py`，候选图为
`outputs/runtime-ane/z-image-c416-k1024-n512-exp-2026-09-29/`。
第一次沙箱内 Core ML 编译因系统临时目录权限失败，随后获权限成功；
没有将失败草稿当成已验证图或删除。

## 两模型 BF16 base 肖像对照

最终库 SHA256 为
`f3f6b6a05ce6eecb476f314f1fa1fccbdb1d2ef24fc09fdc5b72049b0390ca22`，
与先前狐狸实验相同。提示词为窗边成年卷发女性、雀斑、绿色毛衣及自然
光摄影，seed43；这是新于狐狸/seed42 的质量与性能样本，Z 的旧 Q4
肖像不是本轮 BF16 分母。两模型各按 GPU→runtime→冻结→冻结→runtime→GPU
运行六个独立 resident trial，每 trial 一冷两热；每路池化四个热请求
取中位，不跨模型或构建拼接。计时为包含 VAE/PNG 的 request_wall，
排除加载和冷请求，profile 关闭、chunks auto。

- Z-Image 512²/8 步：runtime graph-v1 c352/K1024/N512；冻结用已匹配
  的 image-only W8A8 base 图。
- Qwen-Image-2.1 512²/40 步：runtime graph-v2 c320/K1024/N512，base
  传零 LoRA 修正；冻结用已有 512² W8A8 base 图。三路线统一开启
  可选 GPU Q/K norm-RoPE，不改变模型权重或 SiLU。

| 模型/路线 | 四个热请求（s，按 trial 顺序） | 四热中位（s） | GPU/该路线 |
| --- | --- | ---: | ---: |
| Z GPU | 7.320337 / 7.320049 / 7.317971 / 7.327727 | 7.320193 | 1.000× |
| Z runtime | 6.533177 / 6.489380 / 6.511787 / 6.465406 | 6.500584 | 1.126× |
| Z 冻结 | 5.507351 / 5.502372 / 5.510366 / 5.504446 | 5.505899 | 1.330× |
| Qwen GPU | 41.730280 / 41.766239 / 41.775090 / 41.757093 | 41.761666 | 1.000× |
| Qwen runtime | 35.563877 / 35.655427 / 35.529622 / 35.741841 | 35.609652 | 1.173× |
| Qwen 冻结 | 29.071855 / 29.079686 / 29.070023 / 29.089129 | 29.075770 | 1.436× |

两模型的 runtime 热请求都实际持续调用 Core ML，无 runtime failure
或错误 GPU fallback；Z 两次 runtime trial 末请求累计 668/688 calls，
Qwen 均 3648 calls；冻结的末请求分别为 Z 768、Qwen 3744 calls，
两路均无失败。调用差异包含正常调度和首轮 headroom 探测，不将 GPU
probe 误写为错误回退。18 张 Z PNG、18 张 Qwen PNG 均存在；原始
请求/JSONL 复核 BF16 base、尺寸、步数、seed、无 LoRA、route 与图身份。

六份 Z 和六份 Qwen 100 ms 进程树采样流均独立重验 complete：分别
1353/6673 个样本，最大间隙 110.104/110.019 ms。全部 12 个 trial
均无新增 swap-in/out；Z 与 Qwen GPU/runtime 无新增系统压缩。Qwen
两个冻结窗口 compression 分别约 161.0/55.9 MB，在各自冷 PNG 写出
时已达终值，热请求未再增加；系统计数不可归因于 Core ML 或 ANE，
也不覆盖外部服务、driver/wired 或真正低内存设备。

已实际查看两模型反向 GPU/runtime/冻结的热图：成人主体、卷发、雀斑、
绿色毛衣、窗边构图均保留，未见黑图或明显色偏。GPU/runtime 的面部与
服饰细节接近，冻结图在面部、头发和衣物局部有可见差别。相同路线
正反向末次热 PNG 字节相同，不同路线不是逐像素一致；这仅为第二条
提示词，不能证明人脸/文字/多 adapter 的广泛画质资格。

被忽略的原始证据：

```text
outputs/runtime-ane/z-base-portrait-memory-admission-512-2026-09-29/
outputs/runtime-ane/qwen-base-portrait-memory-admission-512-2026-09-29/
```

最后重跑 `make test-acceleration-contract` 正常退出 0：5 host、
10 placement、42 screen、6 memory screen、17 switch、12 repository 检查。
这些 host/报告检查不能替代上述真实整模型与独立采样；本轮没有重建
产品原生库或重跑完整 `make test`。

结论仍是该本机 512² BF16 base 工作负载冻结图最快，runtime 相对
普通 GPU 有整请求收益；1024² Qwen 的显式 runtime 优势属于另一尺寸。
没有凭第二提示词升级默认选路。完整目标仍缺真实低内存/硬件驻留、
更多提示词、LoRA/编辑和 Q4_1 广覆盖。
