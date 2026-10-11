# 扩展 benchmark：完整目标仍在推进

2026-10-11。本页对应新增要求：充分覆盖本机 FLUX、Z-Image Turbo、Qwen
Image、LTX 的 BF16/可用量化表示与参数，继续探索 GPU/ANE 最佳路线，
保留最快配置，并补常见框架的成功端到端对照。此前18个campaign是已
完成的第一阶段证据，不是这份扩大目标的完成证明。

## 当前覆盖与必须继续做的工作

| 家族 | 已有权威实测 | 扩展缺口 / 下一项 |
| --- | --- | --- |
| Qwen Image 2.1 | 原 GGUF GPU/Private/Unsloth；BF16 resident/streaming；原 BF16 frozen 历史 | BF16/mflux40步对照进行中；mflux affine Q4/Q8、匹配混合、512/1024和share/phase最佳配置再核验 |
| Z-Image Turbo | 五表示16配置GPU/Public/Private/frozen；512/1024分开的历史 | 原 BF16 无损派生布局后对照mflux；补affine4/8、尺寸与channel候选的端到端筛选，不只微基准 |
| FLUX.2 Klein 9B | 原 BF16 native/mflux ABBA | 4/8bit同源派生与mlx-serve成功路径；9B DiT混合分区尚未验证，不能以N/A代表探索完成 |
| LTX 2.5 distilled | 当前完整97帧音视频GPU/INT8/FP16；原组合mlx-serve memory preflight失败 | 本机更快GPU/混合profile、可运行的量化框架布局和完整媒体对照；不能只重用旧2.7%或新0.86%结果 |

除了主表热态延迟，还保留新进程constructor、cold、fresh prompt与
立即重复prompt，512/1024图像与480p/720p-class视频的独立口径，完整
PNG/MP4、memory pressure、执行分区/实际fallback/retry及表示差异。
额外LoRA/PE/VDN附件仍不进入base排名；不同checkpoint不互当分母。

## 本轮实际新增证据

1. 为Qwen原BF16创建只读派生布局：encoder仅键名调整，VAE仅键名与
   已确认singleton空间轴reshape；整个payload字节逐一校验一致。
   不改原权重，不修改只读mflux参考代码，不将格式转换时间扣进或
   加进单次inference。准备encoder约24.256s、VAE约0.917s，单独记录。
2. Qwen完整512²/4步功能预检已成功：mflux BF16首个request+PNG为
   7.519705s，native GPU为8.922268s，两个完整进程树、原token IDs/
   template校对通过。这是单cold短预检，不能变成40步排名、warm倍率
   或生产质量结论。原sourceSHA注册后复用，不在inference再扫模型。
3. 首次正式40步四进程native→mflux→mflux→native窗口在第二进程被
   memory verifier拒绝：实际四张完整图均完成，exit0，但一次采样间隔
   522.920ms超过原500ms上限。native已完成fresh43.226/43.179s，
   mflux已完成58.628/60.036s；这两个进程不能升级为完整ABBA排名。
   保留failed窗口，恢复混合配置后用原门槛在fresh目录重新实测。
4. 以前Qwen frozen最快历史配置的compiled manifest当前不存在。
   原32个W6144 source mlpackage与source manifest仍在；已全部重新编译到
   独立持久cache。新40步单cold完整PNG/memory预检成功，46.877607s，
   实际1248runtime+32warmup calls、failure0、checkpoint SHA通过。
   尚需当前warm/反序实测，不能把旧33.121s冒充当前最佳测量。
   本轮启动前失败保留在
   `outputs/qwen21-mflux-original-v2-40step-palindrome-20261011/startup-error.json`。
5. mflux affine Q4/Q8也实际完成原source512²/4步PNG预检，首request为
   9.059559/8.905662s，两完整进程树。此框架按原定义保留BF16 text
   encoder、不量化该组件；不能与encoder也为Q4_K_M的GGUF组合宣称
   同精度/同内存。两路仅证明可执行，不能宣称量化一定更快。

原始实验路径：

- `outputs/benchmark-model-views/qwen21-bf16-mflux-v1-20261011/layout-receipt.json`
- `outputs/qwen21-mflux-original-v1-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v3-40step-abba-20261011/summary.json`
- `outputs/qwen21-mflux-original-v4-frozen-restored-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v5-affine-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v6-40step-five-route-palindrome-20261011/summary.json`：
  当前完整native GPU/frozen与mflux BF16/Q4/Q8正反序进行中；10个独立
  进程、每进程4请求。仅全部完成后汇总fresh medians，repeat不混入。
- [可携带预检/无损证明/alpha差异记录](../design/validation/qwen21-mflux-layout-precheck-20261011.json)

原生Qwen PNG保存模型生成的RGBA，当前mflux输出RGB，后者只有隐含
opaque alpha。比较明确分别报告raw RGB、完整alpha、白背景composited
RGB；不会悄悄丢掉透明度或宣称两边输出功能完全等价。

尚未运行的量化/参数格子继续标pending，格式/memory失败保留原原因，
但仍尝试安全的同源派生格式，不用失败启动时间、不同精度或其它机器
数字凑胜率。赢/输/无收益都发布。最快配置需留下参数、来源、artifact
和复现入口，且在当前文件状态下能实际加载；不只保存一个旧最小数字。
