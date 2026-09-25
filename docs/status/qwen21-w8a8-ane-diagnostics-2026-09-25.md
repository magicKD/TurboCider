# Qwen-Image-2.1 512² W8A8/ANE 单层诊断

日期：2026-09-25。本文只记录基于实际 512² 生图第二步、缓存 prefix KV 的
block 0 FFN 输入 `[1024,4096]` 做的离线实验，**不是 32 层或完整生图的性能/质量验收**。
32 层校准输入位于忽略的 `results/qwen21/qwen21-w8a8-calibration-512-teapot/`；
文档中的本机模型产物、报告也不进入 Git。可用
`tools/native/qwen21_transformer_probe.cpp` 提取 FFN 输入，以
`tools/coreml/export_qwen3.py` 导出单层隔离模型，并用
`tools/validation/qwen21_a8_boundary_compare.py` 对同一输入比较
Core ML `CPU_ONLY` 和 `CPU_AND_NE`。计算设备策略不是物理 ANE 驻留证明。

## 已测结果

同层 W8A16（仅权重量化）相对 BF16 分支的 RMSE 为 `0.02064`，
Core ML prediction 中位数 `9.26 ms`。原 W8A8 的相对 BF16 RMSE 为
`0.91408`，prediction 中位数约 `8.34 ms`：单层调用略快，**输出不合格**。
即使 CPU_ONLY 相对 BF16 只有约 `0.09255` 的 RMSE，CPU_AND_NE 对
CPU_ONLY 的 RMSE 仍约 `0.91270`。该速度差不能外推到 GPU+ANE 的
32 层/5 步或 1024² 加速比。

进一步隔离同一实测输入（下表均为 `CPU_AND_NE` 相对 `CPU_ONLY` 的
relative RMSE；数值来自忽略目录下的对应 compare JSON）：

| 上投影 / hidden / down | 对比结果 |
| --- | ---: |
| W8，仅 input A8；无 hidden A8，完整分支 | 0.02165 |
| W8，hidden A8，W8 down，完整分支 | 0.91270 |
| W8，hidden A8，FP16 down，完整分支 | 0.91266 |
| FP16 上投影，hidden A8，W8 down，完整分支 | 0.04170 |
| W8，仅 input A8，在 down 前输出 SwiGLU | 0.06130 |
| W8，input/hidden A8，在 down 前输出 hidden | 0.90285 |
| FP16 上投影，input/hidden A8，在 down 前输出 hidden | 0.13686 |

后两组 hidden tap 不含 down，不能替代完整分支画质测试；但它们把当前
`CPU_AND_NE` 失真定位在 **W8 上投影与 hidden A8 的图组合中、进入 down
之前**。把 hidden A8 改成 uint8、把 down 切成两段、把上投影后 SwiGLU
暂转 FP32 均未改善：各自的单层对比结果约为 `0.91267`、`0.91267`、
`0.91270`。简单统一输出增益也不是修复：拟合增益约 `7.14×` 后仍剩
`0.599` relative RMSE。相应离线 JSON 为
`qwen21-a8-amplitude-compare-512.json`、
`qwen21-hidden-input-vs-both-tap-512.json`、
`qwen21-a8-projected-down-isolation-512.json` 等。

这些现象提示 Core ML 对该图的 ANE 路径与 CPU 路径不等价，尚不能在
没有内部中间值/设备轨迹的情况下断言具体是哪一个 ANE kernel 或编译 pass。
后续应先找到数值正确且能证明 ANE 驻留的**双 W8 + 双 A8**单层图；
再测未用于校准的提示词/编辑输入、至少 29/32 层覆盖率、512² 文生图和
1–3 张参考图编辑的实际视觉效果、GPU/ANE 同条件端到端性能。

生产 Session 仍只允许完整 32 层 FP16 manifest；上述诊断产物不能用于
默认 CLI 或声称 W8A8 已合格。即使诊断图采用 FP16 权重，Session 仍须验证
`activation_precision=fp16`，防止研究用 A8 图误入生产路径。
1024² 的 Qwen W8A8 GPU/ANE 配对测试
尚未进行；Z-Image 的历史 1024² 速度不属于本模型。
