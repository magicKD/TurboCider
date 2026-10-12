# 性能测量负载来源：只读诊断

四格 base ≥1.2×、完整 LoRA/1024²/质量/内存和并发验收目标未改变，仍未完成。
本轮没有修改 inference 实现或放宽 benchmark gate，没有新的有效倍率。

## 新的直接证据

使用 macOS `KERN_PROCARGS2` 只读查询仍存活的候选 PID，避免 `ps args`
对多行 Python code 的显示/拆词误导。先用本轮创建的短生命 Python probe
验证 argv 解码与已知参数逐项相同；该 probe正常退出，无 GPU/模型工作。

连续30秒捕获两个 CPU>5% 的 ComfyUI环境Python：

| PID | PPID | CPU | argc | 可安全保留的参数元数据 |
| --- | ---: | ---: | ---: | --- |
| 37438 | 1 | 6.6% | 3 | `-u`，脚本 basename `download_ltx25.py` |
| 37538 | 1 | 7.5% | 3 | `-u`，脚本 basename `download_ltx25.py` |

更早35秒观察另有3个短生命候选（37101、37202、37300），其中查询成功，
但当时只保留argc/mode，没有保存脚本名，不能补填其身份。

这证明最近捕获的具体候选执行的是名为 `download_ltx25.py` 的脚本，
不证明其代码绝不调用GPU，也不证明所有历史拒绝PID都是同一下载任务。
既有 classifier 明确将该下载脚本列为 CPU/I/O resource competitor；
不能把其活动写成已证明的 ComfyUI模型推理/GPU争抢，也不能忽略它追认速度。

只输出PID/PPID、CPU、参数数量、flag及脚本basename；没有保存原始argv、
inline code、prompt、token、凭据或完整脚本路径。没有读取或修改下载内容，
没有signal/暂停/终止任何外部进程。诊断 probe/observer handles已terminal。

## 对后续工作的影响

所有原来被 load verifier 拒绝的summary保持incomplete，原始证据不改。
下一步正式同库正反序四格和LoRA timing仍需要满足预先定义的安静窗口；
不能用这些不完整时间拟合/确认实际收益。控制外部下载任务需要用户协调，
不属于本任务可擅自执行的操作。

运行期带宽-aware share/prefetch calibration、Qwen新模式/1024 LoRA、
完整媒体/latent/内存和device trace仍是待完成项；本诊断不是其替代方案。
