# TurboCider 实验代码边界

`experimental/` 只保存研究快照、非公开 API 试验和已被正式实现替代的原型。这里的代码不属于 TurboCider 的产品 ABI，也不进入默认 native library、CLI、App 或发行包。

正式构建边界由以下机制共同保证：

- `tools/native/build.sh` 使用显式源文件清单，不递归收集 `experimental/`；
- `tests/repository/test_layout.py` 拒绝正式构建重新引用实验目录；
- Apple private ANE 的 `_ANEInMemoryModel*`、`_ANERequest` 和 `_ANEIOSurfaceObject` 调用只存在于实验快照；
- 正式 H3 runtime 链接 `h3_ane_disabled.c`，任何历史 private-ANE 入口都会 fail closed；
- 公共 GPU+ANE 路径使用 Core ML、显式 manifest、checkpoint/LoRA provenance 和 caller-owned output backing，不依赖 private API。

目录说明：

- `transformer/`：M4 Pro 上独立 1–3 层 Transformer 的 tensor、FFN sequence-row、完整 attention-head 并行研究，包含 public/private API 对照、原始数据和 [完整技术报告](transformer/notes/REPORT.md)；
- `video/`：H3/LTX 迁移早期快照，不应继续同步模型数学；
- `private-ane/`：private API 的适用范围、实测结论和不可发行原因；
- `tools/experimental/`：只构建研究快照的显式入口，与 `make build`、`make package` 无关。

完整 Transformer CPU/GPU/ANE 结论见 [异构并行技术报告](../docs/design/transformer-heterogeneous-report.md)。
