# Qwen 1024²：当前 v2 共图三路重复对照

2026-09-29，**六 trial 全部完成并重新校验**。当前保留库的 Qwen 1024²
base 文生图：runtime v2 对 GPU 为 **1.115×**，冻结图为 **1.147×**，
冻结图仍最快。本轮是长序列性能/进程内存对照，不是新原生优化，
也不是 1024² LoRA、编辑或广泛质量验收。

## 条件与执行

- 库：`e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`。
- M4 Max 64 GB，Qwen-Image-2.1 BF16，1024×1024，40 步，seed42，
  狐狸雪景 prompt，无 LoRA，resident，auto chunks，profile 关闭。
- runtime 复用此前 512² 的 c320/K1024/N512 v2 图：base 输入零 LoRA
  修正，长序列循环 token chunks；没有重新导出图或合并任何 LoRA。
- 冻结 W8A8 使用已有 4096-row、全 32 层图和显式 1024² diagnostic；
  不与 512² 冻结图混用，也不提升为普通请求默认。
- GPU → runtime → frozen → frozen → runtime → GPU，6 个独立
  resident CLI batch，每 trial 一次冷请求、两次热请求，共 18 请求。
  每路线汇总全部四个热样本，含 VAE/PNG，不含加载和冷请求。
- 所有路线同样开启 100 ms 独立进程树采样，最大允许间隙 500 ms。
  每个 trial 前预检竞争推理，繁忙时等待；不杀他人进程，不与构建、
  微图或其他模型测试并行。预检并非全程设备独占证明。

从仓库根目录复现；输出目录必须尚不存在：

```sh
.venv/bin/python3 -B tools/validation/runtime_ane_model_screen.py \
  --model models/Comfy-Org-Qwen-Image-2.1 --model-id qwen-image-2.1 \
  --size 1024 --steps 40 \
  --runtime-manifest outputs/runtime-ane/qwen-lora-c320-k1024-n512-v2/manifest.json \
  --frozen-manifest results/qwen21/ane-compiled-4096/manifest-1d0b913e073fc44130c667a55f40c715622f7eb783e3e32ee7707250c6adb509.json \
  --routes gpu,runtime,frozen,frozen,runtime,gpu --chunks auto \
  --warm-repeats 2 --sample-memory --timeout 1800 \
  --output outputs/runtime-ane/qwen-base-1024-v2-matched-memory
```

`--timeout` 是每个完整 trial 的上限，不是整轮或单张图时限。超时只
清理本工具启动的采样/推理进程组；不重启或覆盖不完整输出。
输出目录保存逐请求 JSON、PNG、原始 JSONL、进程快照、采样流与 verifier
报告。`summary.json` 仍为 `incomplete` 时，不得把部分路线当完整对照。

## 与现有证据的关系

[旧 1024² 初筛](runtime-ane-qwen-1024.md)是 `bdc7892…` 库、c288 v1 图，
每路线仅一个热请求：GPU/runtime/frozen 为 188.399/181.206/163.859 s。
本轮不把旧结果重标为当前库，不跨构建拼接 GPU 分母计算新加速比。
即使本轮更快，也不能直接将差额归因于某一个 native 或图参数改动。

[512² 匹配对照](runtime-ane-matched-memory.md)仍属于 `8d90f74…` 库，
其 Qwen runtime 1.165× 与 frozen 1.412× 不能外推到 1024²。
[LoRA 采样顺序候选](runtime-ane-gpu-first.md)已经撤回，本轮使用恢复后的
原版调度，不包含 GPU-first、FP16 LoRA、DBCache 或其他实验组合。

## 完整请求结果

六 trial 全部退出成功，`summary.json` 为 complete。重新读取六份原始
JSONL（18 请求）通过 backend、有限计时、失败/累计计数验证；逐请求
参数、base 无 LoRA、执行路线、图几何和 PNG 存在性一致。库 SHA 在实验
前后保持上述 e55。下表保留冷请求及每个热请求，不只挑最快样本。

| trial / 路线 | 冷请求（s） | 热请求 1（s） | 热请求 2（s） | 本 trial 热中位数（s） |
| --- | ---: | ---: | ---: | ---: |
| 0 / GPU | 192.633798 | 187.904493 | 187.871430 | 187.887961 |
| 1 / runtime v2 | 174.401138 | 168.085974 | 177.559596 | 172.822785 |
| 2 / frozen | 173.634962 | 163.674498 | 163.742614 | 163.708556 |
| 3 / frozen | 174.996460 | 163.655141 | 163.401857 | 163.528499 |
| 4 / runtime v2 | 175.035023 | 168.843791 | 166.661392 | 167.752592 |
| 5 / GPU | 189.726411 | 187.718151 | 187.727788 | 187.722970 |

| 路线 | 四个热请求中位数 | GPU/该路线 |
| --- | ---: | ---: |
| GPU | 187.799609 s | 1.000× |
| runtime v2 | 168.464882 s | 1.115× |
| frozen W8A8 | 163.664819 s | 1.147× |

runtime 热请求范围 166.661–177.560 s，波动比 GPU/frozen 大。样本少，
不声称统计显著性或稳定热状态。冻结图仍更快，但 runtime 在本次进程
窗口的 footprint 峰值更低；不能把速度与内存取舍简化成一个万能预设。

## 多 chunk 与时间边界

runtime 两个 trial 均无失败、fallback 或 overflow retry，headroom=1。
各自冷/热1/热2的计数增量相同：hybrid blocks 为 1,184 / 1,248 / 1,216，
完整 GPU probe 为 96 / 32 / 64，实际 predictions 为 4,448 / 4,896 / 4,864。
每请求合计 1,280 个 block；预测次数明显大于 hybrid block 数，确认多
chunk 实际执行。热2平均每 hybrid block 为四次预测，不代表每层都相同。

正向热2比热1慢约 9.47 s，但反向热2反而更快，不能用固定请求位置或
累计执行数量解释。四个热请求分段增量（秒）：

| 窗口 | 正向热1 | 正向热2 | 反向热1 | 反向热2 |
| --- | ---: | ---: | ---: | ---: |
| hybrid FFN | 77.687 | 80.567 | 77.898 | 73.977 |
| pre-FFN | 82.837 | 84.916 | 83.537 | 80.275 |
| staging 工作 | 5.571 | 9.681 | 6.087 | 4.841 |
| 暴露 staging 等待 | 0.000819 | 0.001603 | 0.000818 | 0.000605 |
| Core ML prediction | 65.783 | 66.032 | 65.789 | 65.186 |
| 输出处理 | 4.562 | 8.432 | 4.802 | 3.827 |
| post-join | 4.114 | 4.200 | 4.021 | 4.025 |

正向两次热请求首步分别为 4.725 / 4.763 s，不支持把约 9.47 s 的差距
归因于首步编译。输出处理与 staging 总工作增长明显，但不证明具体
热状态或外部竞争原因；暴露 staging 等待不足 2 ms，不宜只靠增大
双缓冲就假定能消除数秒整请求时间。

这些窗口不能全部相加：async ANE wait 与 GPU 重叠；`pre_ffn` 在
untimed 路线可能包含上游 lazy GPU block/残差，不是纯 attention。
完整 block 的 on/off 样本输入 fence 已在两模型源码中核对，这不是
调度窗口污染证据，也不是硬件 overlap 比例或实际 ANE residency 证明。

## 独立内存与环境

重新调用 verifier 校验六份原始采样流，与 summary/报告逐项匹配；
原始文件、报告、采样工具 SHA 与 correlation ID 全部一致。共 31,690
个样本，最大间隙 116.656 ms，小于预设 500 ms。范围含加载、冷/热请求
和退出；以下 GB 为十进制，不是热请求独占峰值，RSS 与 footprint 不可相加。

| trial / 路线 | 峰值 RSS（GB） | 峰值 footprint（GB） | swap-in（KiB） | compression（bytes） |
| --- | ---: | ---: | ---: | ---: |
| 0 / GPU | 15.033 | 31.554 | 0 | 0 |
| 1 / runtime | 15.474 | 31.953 | 0 | 0 |
| 2 / frozen | 17.436 | 40.124 | 0 | 306495488 |
| 3 / frozen | 17.400 | 40.252 | 64 | 0 |
| 4 / runtime | 15.474 | 30.746 | 64 | 0 |
| 5 / GPU | 15.030 | 31.554 | 64 | 0 |

全组无新增 swap-out，但不是零换页/零压缩。系统 swap/compression 不
能全部归因于模型，也不覆盖外部 Core ML 服务/driver/wired；这不是完整
内存压力或低内存设备验收。不能据进程峰值直接下调 admission 安全余量。

trial 4 的预检曾发现一个 ComfyUI 进程 CPU 35.6%，工具按原规则等待
后才启动 runtime；后续进程快照未见该 PID。保留这个环境事件，
不声称全程设备独占，也不因观察等待而重启已完成的 trial。

## 代表图检查

接续整理后，实际打开正向三路最后一个热请求的 1024² PNG：
`0-gpu-2.png`、`1-runtime-2.png`、`2-frozen-2.png`。三张均是雪中坐姿
红狐，主体位置、朝向、松枝背景和整体色调接近；runtime 的脸部、毛发、
雪地细节与 GPU 很接近，冻结图在尾巴轮廓、胸部毛发及枝叶细节上差别
更明显。未见黑图、明显色偏、主体缺失或大面积纹理崩坏。

这是对本 prompt/seed 的视觉检查，不要求逐像素一致；不是数值等价、
多提示词或 1024² 编辑/LoRA 的资格。结束后核对正反向末次 PNG：GPU、
runtime、frozen 各自字节完全相同，反向对应同一已检查图像。不同路线
之间不是逐像素一致，不用哈希替代对不同图片的视觉判断。

## 后续组件筛选

设计稿区分 runtime chunk 与图内 K/N tile；VPIPE 的默认 chunk 不能
直接替代本模型实测。当前计数说明长序列已经循环多次 prediction，
下一项可检验比较是保持 4,096 总 rows、1,280 ANE rows、2,816 GPU rows、
K1024/N512 及同一 v2/SiLU 接口不变，比较 c320×4 与 c640×2。
这样改变的是 chunk 几何/调用次数，而不是同时改变 GPU 分区工作量。
仍需分别测 prediction、并行 FFN、含 staging 窗口及数值误差，不能仅
凭 dispatch 次数减半宣称提速；通过组件筛选后再做整请求匹配实验。

现有 Qwen teapot 捕获只有 1,024 rows，不能复制/reshape 后当作真实
1024² activation。若先用真实 checkpoint 加随机 4,096-row 输入，必须
标为合成激活的组件实验，不能替代完整模型质量。整模型对照终止后才
构建探针、导出 c640 新图；它们不影响本页的 c320 整模型结果。

为此已在独立 `ane_runtime_probe.cpp` 中准备增量遥测：保留原中位数字段，
补充输入转换、输出处理、worker 总窗口，以及排除两次 warmup 后的逐样本
时间、交错顺序、prediction/retry 数、headroom 和几何。这样可以检查
chunk 次数与工作量，而不将重叠的 worker/GPU 时间相加。微图集成测试
同步增加中位数重算、调用数和分段时间关系断言，以及 v2 c16×4/c32×2
同分区的真实微图回归。`make build-runtime-ane-probe test-runtime-ane`
成功：4 项 host、9 项 Core ML 图/集成通过，无 skip；既有临时目录
ResourceWarning 未导致失败。随后 66 项加速契约也通过；本轮未运行完整
`make test`。未修改 native 产品后端或重建产品库。

后续组件筛选已完成，原始 c320/c640 同分区没有并行收益；重新平衡后的
c1792 进入整模型交错对照。另见[chunk 与分区记录](runtime-ane-qwen-chunks.md)，
不把该候选组件成绩或未完成的整模型结果回填成本页基线。
