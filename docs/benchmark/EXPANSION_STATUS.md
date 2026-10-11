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
2. Qwen v1完整512²/4步预检完成计时：mflux BF16首个request+PNG为
   7.519705s，native GPU为8.922268s，两个完整进程树、原token IDs/
   template校对通过。后续图片检查发现v1遗漏64个真实FFN权重、输出
   为噪声；因此连“正确模型可执行”都未证明，不作排名或质量通过。
   原sourceSHA注册后复用，不在inference再扫模型。
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
5. v1 mflux affine Q4/Q8完成512²/4步PNG预检计时，首request为
   9.059559/8.905662s，两完整进程树。此框架按原定义保留BF16 text
   encoder、不量化该组件；不能与encoder也为Q4_K_M的GGUF组合宣称
   同精度/同内存。v1同样遗漏FFN权重，输出噪声；不作有效模型执行。
6. v6五路线40步回文十进程/40PNG/memory全部完成。有效native GPU/
   frozen四fresh中位43.249697 /31.007695s，省时28.305%。MF三route
   是invalid-output diagnostics：Comfy gate_up未拆成64个required
   gate_layer/proj，stock loader默默保留random initialization，不能
   用56.879/60.699/61.443s宣传跨框架领先。
7. v2完整布局已拆分全部32个gate_up，完整payload仍相同；required
   coverage297 DiT/398 encoder/226 VAE均满。三参考4步输出狐狸；另
   BF16完整40步单cold50.961601s、图片正常、memory complete。新的
   v9五路线正反序正在运行，之后才做实际框架排名。

原始实验路径：

- `outputs/benchmark-model-views/qwen21-bf16-mflux-v1-20261011/layout-receipt.json`
- `outputs/qwen21-mflux-original-v1-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v3-40step-abba-20261011/summary.json`
- `outputs/qwen21-mflux-original-v4-frozen-restored-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v5-affine-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v6-40step-five-route-palindrome-20261011/summary.json`：
  已完成，但MF参考噪声仅作失败诊断，native两route有效。
- `outputs/qwen21-mflux-original-v9-complete-weights-five-route-20261011/summary.json`：
  修复后五路线十进程40步对照正在运行。
- `outputs/z-mflux-original-v4-full-seven-route-20261011/summary.json`：
  七路线完整8步对照串行排在Qwen之后，尚未启动。
- [原生完整结果及invalid参考标记](QWEN_BF16_FRAMEWORKS.md)
- [可携带预检/无损证明/alpha差异记录](../design/validation/qwen21-mflux-layout-precheck-20261011.json)

原生Qwen PNG保存模型生成的RGBA，当前mflux输出RGB，后者只有隐含
opaque alpha。比较明确分别报告raw RGB、完整alpha、白背景composited
RGB；不会悄悄丢掉透明度或宣称两边输出功能完全等价。

## 后续入口与实现核对

- `z_mflux_framework_screen.py`已加入原生GPU/frozen/Public/Private与
  mflux BF16/affine4/8；Private候选支持2048–8192的七个已声明partial
  shares、F16/F32 join，核对actual channels与join receipt，不仅记录env。
  BF16/Q4/Q8短预检图片正常。七路线完整对照尚未启动，新增shares/
  join候选尚未硬件实测，不进入最佳配置或性能表。
- Qwen派生encoder的大副本占用影响下一批模型布局预算。新增held-fd
  APFS clone + compact-header工具，仅缩短reference不消费的352个tensor
  名字，保留全部payload和所有language targets；实际fixture三项通过。
  大模型clone已经完整校验，v1的17.53GB自建重复副本在旧窗口terminal
  后移除，原source和clone保留，换出空间用于正确DiT派生文件。当前
  计时中不改文件或做大批I/O；`st_blocks`不等于独占物理磁盘字节。
- FLUX9B实际为H4096/F12288、8 dual+24 single。现有HybridSession已
  支持index+完整shard SHA，可直接保持原sharded模型，不必创建18GB
  dense单文件。待扩展的是4B-only GPU图/padding/selected checkpoint与
  export geometry，之后用真实block/全图验证再解除9B显式拒绝。
- LTX当前batch走spawned VAE helper，而single-shot CLI/disposable worker
  可走clean exec finalizer。后续把两种完整MP4边界分开测，并筛选现有
  Sol/text256/INT8 profiles；不能把历史clean decode micro当新whole-request
  加速，也不能把换进程cold算成同session warm。

尚未运行的量化/参数格子继续标pending，格式/memory失败保留原原因，
但仍尝试安全的同源派生格式，不用失败启动时间、不同精度或其它机器
数字凑胜率。赢/输/无收益都发布。最快配置需留下参数、来源、artifact
和复现入口，且在当前文件状态下能实际加载；不只保存一个旧最小数字。
