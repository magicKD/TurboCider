# GPU/ANE 验证工具维护入口

产品选路和已完成跑分只维护在[加速状态](../../docs/status/acceleration.md)，
模型计算与生命周期约束见[后端说明](../../native/backends/README.md)。
本页只整理GPU/ANE相关工具，不改变本目录其他验证入口。

## 代码职责

| 文件 | 保留职责 | 不代表什么 |
| --- | --- | --- |
| `runtime_ane_common.py` | 共用环境清理、竞争推理预检、流式SHA、编辑/调用/QK receipt校验 | 不加载模型，不代替图像质量验收 |
| `runtime_ane_model_screen.py` | GPU/runtime/frozen整请求对照、原始结果与PNG、可选独立内存采样 | 单向screen不等于完整匹配对照 |
| `runtime_lora_shared_graph_switch.py` | 同进程base → A → 合成B → base，检查图复用、实际调用和状态隔离 | 合成B不是第二个训练LoRA的质量资格 |
| `runtime_ane_memory.py` | 采样进程编排、证据绑定、超时处理；复用 `tools/native/` 的采样器 | 进程footprint不是ANE独占或完整driver内存 |
| `runtime_ane_qkv_regression.py` | 使用既有原生库运行一次合成 QKV 失败回退/取消复用检查；保存日志/hash 并清理所属临时图 | 不加载 checkpoint，不证明 ANE 驻留或推理提速；不属于默认构建 |
| `qwen21_ane_placement.py` | 冻结/runtime manifest的离线设备计划、嵌套算子与artifact身份 | preferred设备不等于实际硬件执行或重叠 |

新增验证工具应直接复用 `runtime_ane_common`，不要从screen导入runner逻辑。
screen现有helper重导出和switch兼容接口保留，避免破坏已保存的复核脚本。
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
