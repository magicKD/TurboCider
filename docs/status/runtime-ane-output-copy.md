# Runtime ANE：输出复制消融与收尾决策

结论：直接把 worker 的还原结果写入调用独占的 MLX buffer，通过了微图
正确性/生命周期检查，但没有证明整请求更快。本次整理撤回候选实现，
恢复可复用 CPU scratch → 独立 MLX tensor 的输出路径；保留新增测试。
不增加新开关，不改变 GPU/冻结图默认路由，也不宣称本轮有新性能提升。

## 条件与证据

- M4 Max 64 GB，Qwen BF16 + Viggle v0.2.1 r256 runtime LoRA，6 步。
- 512² 输出、三张有序 ref512，seed29，resident，固定 `chunks=1`。
- v2 activation-input 图，base 权重不合并 LoRA；profile 关闭。
- 每个构建两个独立 runtime trial，每 trial 一冷两热；下表为四个热请求
  的中位数，墙钟含 VAE/PNG。不是 auto 最佳分区或 GPU/runtime 加速比。
- before 库：`9cad19b5d70bfe8b6a15932f07a907ad594b9ba59b5fcd04b3a1f06de3e65e54`。
- 候选库：`8351c3799161bba3263e7d5a7c25cae0b9ef97e1a1d8b34c651716dc150214d7`。
- 原始结果：`outputs/runtime-ane/qwen-edit3-owned-output-before-fixed1/`、
  `outputs/runtime-ane/qwen-edit3-owned-output-after-fixed1/`，两者
  `summary.json` 均为 `complete`。输入来自本地
  `results/qwen21/viggle-v021-r256-edit3-ref512-seed29-request.json`。

| 热请求指标，中位数 | before | 候选 |
| --- | ---: | ---: |
| 整请求墙钟 | 15.932250 s | 15.993678 s |
| hybrid FFN 累计 | 8.453521 s | 8.479162 s |
| GPU FFN 分支累计 | 6.712873 s | 6.731349 s |
| join 累计 | 0.411114 s | 0.447284 s |
| gate/up 修正累计 | 0.701550 s | 0.764291 s |
| post-join 累计 | 0.633938 s | 0.461666 s |

before 热墙钟：16.089365542、15.833831917、15.962980625、15.901520291 s。
候选热墙钟：15.855294666、16.116452834、15.922114375、16.065242125 s。

post-join 名义少约 27.2%，整请求却名义多约 0.39%。分配移动到了 launch
之前，不能把 post-join 的下降全归因于消除复制；各窗口存在重叠，不能相加
代替整请求墙钟。样本少、构建前后顺序运行，差异不能证明候选显著变慢，
但也不足以保留它作为已验证的默认优化。候选仍需从 Core ML 输出还原，
不是 Core ML 零复制输出。

每 trial 共 576 个 hybrid block / prediction（含冷请求），无错误回退或
overflow retry。四个 trial 的系统 swap-in/out 均无增量；MLX peak 约
22.287 GB，不包含 Core ML/驱动/文件缓存，不能作为整进程内存峰值。
工具逐路预检其他推理进程；这是启发式检查，不保证设备全程独占。

两构建全部 12 张 PNG 的 SHA256 相同：
`1e76ba5f78d7c3d67203b3ef2ba05c8ef008c5eab55f72b237bd7cf279749e1b`。
这证明该固定输入的输出未变，不代表新增提示词、适配器或编辑质量验收。

## 代码保留边界

- `HybridFfn` 保留两块可复用 scratch，worker 完成且成功后才复制成
  tensor。typed-pointer 构造保证独立所有权，不用借用 scratch 的视图。
- adapter callback 保留的 hidden/base 与延迟消费者不得被后续预测覆盖；
  同样不得依赖 runtime 对象继续存活。
- worker 失败时重算整个 tail；GPU 抛错/取消必须先 join 再退出，不能消费
  部分写入结果。末尾 eval fence 保留，未把 GPU 工作移出计时窗口。
- 继续保留完整 gate/up → SiLU → down 数学及 down-LoRA 修正；不合并权重，
  不放宽数值测试阈值。

`tests/native/ane_ffn_test.cpp` 新增输出生命周期覆盖：BF16/FP16/FP32、
33/97/65 行切换、callback 保留值、lazy 消费者、runtime 销毁、launch 后
GPU 异常/取消及恢复。`test_ane_runtime.py` 要求对应 PASS 标记，避免测试
入口悄悄漏跑。候选撤回后重新构建和运行这些检查；收尾验证结果记录于
[当前维护入口](acceleration.md#代码维护与验证)。

已有最快 base 路径与 optional 对照见[加速决策表](acceleration.md)。
更严格调度门槛也未采用，见[另一项负结果](runtime-ane-block-margin.md)。
保留负结果文档，避免以后重复把局部计时改善当作端到端收益。
