# GPU/ANE 验证工具维护入口

产品选路和已完成跑分只维护在[加速状态](../../docs/status/acceleration.md)，
模型计算与生命周期约束见[后端说明](../../native/backends/README.md)。
本页只整理GPU/ANE相关工具，不改变本目录其他验证入口。

## 代码职责

| 文件 | 保留职责 | 不代表什么 |
| --- | --- | --- |
| `runtime_ane_common.py` | 共用环境清理、竞争推理预检、流式SHA、编辑/调用/QK receipt校验 | 不加载模型，不代替图像质量验收 |
| `runtime_ane_model_screen.py` | GPU/runtime/frozen整请求对照、原始结果与PNG、可选独立内存采样 | 单向screen不等于完整匹配对照 |
| `runtime_ane_image_compare.py` | CPU-only等尺寸PNG哈希、RGB RMSE/PSNR/相关与Gaussian11 SSIM、独立alpha比较 | 不重采样，不下载学习权重，不代替LPIPS/CLIP/latent或语义资格 |
| `runtime_lora_shared_graph_switch.py` | 同进程base → A → 合成B → base，检查图复用、实际调用和状态隔离 | 合成B不是第二个训练LoRA的质量资格 |
| `runtime_ane_memory.py` | 采样进程编排、证据绑定、超时处理；复用 `tools/native/` 的采样器 | 进程footprint不是ANE独占或完整driver内存 |
| `runtime_ane_qkv_regression.py` | 使用既有原生库运行一次合成 QKV 失败回退/取消复用检查；保存日志/hash 并清理所属临时图 | 不加载 checkpoint，不证明 ANE 驻留或推理提速；不属于默认构建 |
| `runtime_ane_swiglu_int8_probe.py` | 完整动态 SwiGLU/LoRA 的 FP16/QDQ 对照，包含 CPU 准备、FP32桥接、数值/设备计划与严格 Python 输出边界检查 | 两个小几何的诊断不是生产加速或图像质量资格；不接入 App |
| `runtime_gpu_swiglu_a8w8_probe.py` | 两个受限小尺寸的完整动态 SwiGLU/LoRA GPU 对照：FP16、INT8 存储后 FP16、普通整数 ALU；含准备/同步/回读和独立 CPU 数值校验 | 整数 ALU 不是原生 INT8 矩阵加速；不接入 App，不与不同输入边界的旧 Core ML 跑分直接相除 |
| `qwen21_ane_placement.py` | 冻结/runtime manifest的离线设备计划、嵌套算子与artifact身份 | preferred设备不等于实际硬件执行或重叠 |

新增验证工具应直接复用 `runtime_ane_common`，不要从screen导入runner逻辑。
screen现有helper重导出和switch兼容接口保留，避免破坏已保存的复核脚本。
Private W8A8 初筛使用 `--runtime-backend private --private-gpu-io
--private-data-path w8a8`；tool 明确设置已清理的私有环境，并校验实际
data-path 和所有 prediction 的 device I/O 回执。runtime 每请求调用增量
另存为 `runtime_calls_per_request`；auto 的热态零调用不是 ANE 加速资格。
Private/纯 GPU 的库、工作负载和 Q/K/LoRA 精度仍必须相同。
`--private-a8-lookahead 0|1` 独立控制 chunk-sized A8 双槽，默认关闭。
多块序列校验累计 producer 数、等待 span 与 policy 一致；单块无 future
producer，不能把其它优化的收益算作 A8 pipeline 收益。详见
[A8 接续](../../docs/status/private-ane-a8-lookahead-2026-10-04.md)。
`--private-stage-specialize 0|1` 消融 Metal format/dtype/block function
constants；默认0、同库测量，不改变 recipe 或原始 weight layout。
`--observe-load` 连续记录 CPU 负载并排除 owned launch tree，检测到竞争或
观察不完整就保留 raw output、拒绝比较；不是实际 GPU/ANE 独占证明。
`--private-lora-channel-range 0|1` 对比旧 full gate/up correction 与仅 ANE
子范围 correction；限定 private W8A8 channel LoRA，并验证实际 narrow/full
callback counters，不能将 env 或 self-test 当作模型执行。Public/row和
没有子范围 callback 的调用方保持完整路径；down-LoRA仍用一次完整 hidden。
`--fixed-async 0|1` 仅用于positive fixed chunks且关闭profile的runtime消融。
1复用untimed/async head，0保持固定分区计时；verifier要求实际成功block的
untimed/async counters符合选择。该计数不是物理GPU/ANE overlap证据。
`--qwen-lora-1024` 显式开启六步1024² LoRA生成diagnostic，对每条GPU/
runtime路线采用同样的原FP32 rank，拒绝编辑、frozen和FP16混用；核验
planned与actual 1024 LoRA标记、实际尺寸/步数/未合并binding，不把flag
或self-test当模型执行/画质资格。原512²精度与入口不变。
placement保留历史文件名和冻结图 `--blocks` 接口；runtime只有一个共享图，
不是每层一个图，且不接受 `--blocks`。不要仅为命名统一移动这些入口。

## 可复用工具与一次性实验分开

- 产品CLI不启动这些Python工具。导出在 `tools/coreml/`，组件probe在
  `tools/native/`，对应回归在 `tests/native/`。
- 一次性campaign driver、请求、日志、PNG和编译图留在被忽略的
  `outputs/`、`results/`；稳定结论进入 `docs/status/`，便携请求进入
  `examples/requests/`。不把实验的机器路径复制进产品代码或示例。
- 保留失败候选的报告与有用回归，不把已撤回优化重新加成产品开关。
- LoRA不得合入base权重、Core ML artifact或runtime slots；不要为了
  加速删去完整adapter SHA校验、有限性检查、失败重算或隔离测试。

## 执行与验收

从仓库根目录串行执行，且不得与性能benchmark重叠：

```sh
make test-acceleration-contract  # host数学/报告/路由；可能编译host测试
make test-runtime-ane            # 小型Core ML/MLX图与真实设备工作
make test-qwen21
make test
```

整请求命令、图准备、参考图顺序及内存采样说明见
[实验复现指南](../../docs/status/runtime-ane-validation.md)。不同路线必须匹配
构建、工作负载、Q/K融合与LoRA精度；正式计时关闭profile，保留全部预声明
热样本，不能删除GPU探测请求或拼接旧分母。遇竞争推理等待，不终止他人进程。

运行中的campaign已绑定源码、工具、CLI/库和artifact身份：不要修改这些
输入、重建库或清缓存。完整结束后独立复核原始结果与身份，再查看图片。
`plan`、微图通过、planned placement或incomplete报告均不算性能/视觉通过。
