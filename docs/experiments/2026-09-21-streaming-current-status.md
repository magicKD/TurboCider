# Streaming 当前完成情况与提交快照

更新：2026-09-21。本报告整理截至实验第 69 轮的代码、已有实机证据和发布缺口。当前处于内部验证阶段，**GPU 实际执行已跑通，产品发布尚未完成**。不以代码量估算完成百分比，M0/M1 和整体 GPU/ANE 目标均未宣告完成。

## 完成情况

| 范围 | 已完成及验证 | 尚未完成 |
|---|---|---|
| 模型准备 | Flux.2-klein-4B、Z-Image-Turbo 已下载并校验来源 | 发布资格所需的跨安装验证 |
| Streaming 框架 | 布局编译、SourceLease、slot 执行、actual receipt、来源与实际计划绑定 | 完整失败矩阵和 unsafe drain 隔离验收 |
| 真实 GPU 执行 | 两模型均通过 public C API 的 exact streaming 生成与 denoise 取消；使用内部 test catalog | 正式发布 campaign、多 prompt/seed 质量验收 |
| App / CLI | worker 查询与生成、持久 intent、进程身份与启动日志、取消清理、进度事件、结果校验与事务发布、重启恢复已有实现和测试 | 使用合格生产记录的真实 App 成功出图全流程及完整 GUI 验收 |
| 性能探索 | 已完成 Z 512² 的 D0/Q1 与 D1/Q2 配对测量，保留全部样本及失败/不确定证据 | 稳定端到端加速、完整 P2/P3 资格；没有新的默认布局结论 |
| GPU/ANE | typed hybrid stage 和 Core ML / GPU suffix 私有实验已运行；FP16 bank 局部检查 32/32 通过 | 完整 ModelEngine/public route、全组件 receipt、engine 级故障隔离、整图质量与收益认证 |
| 发布工具 | strict 合同、calibrated 冻结合同、验收文件与原始 campaign 重验工具已实现 | calibrated record/schema、builder/native/resolver、打包与 App 标识接通；真实评审验收 |

生产目录 `native/runtime/streaming/bundled_catalog.json` 仍是 `tc-streaming-catalog-empty-v1`，**records = 0**。公开档位没有可授权记录，继续不可用。内部 test catalog 出图不等于生产目录或 App 产品放行。Flux 在本机 16 GiB 上的 resident guard 也未因 streaming 单例成功而降低。

## 已有实机结果

设备为 M1 Pro / 16 GiB。以下是不同测量范围的观察，不能相互替代。

| 单次 public C API smoke | Z-Image | Flux4 |
|---|---:|---:|
| 输出 | 512²，9 步 | 512²，4 步 |
| 完整脚本 wall，含来源验证与清理 | 60.296 s | 21.919 s |
| native request | 48.633 s | 13.025 s |
| 进程树采样峰值 | 8.321 GiB | 5.984 GiB |

两模型均验证 actual plan / source lease / drained receipt，输出 PNG 可解码；第 1 步取消均无图片并完成清理。Flux 取消到清理返回约 0.095 s，Z 未单独记录取消延迟。该轮采样完整且未观察到 swap-out，但不是多样本内存保证、零 swap 保证或正式 P1/P2。详见[真实 GPU 验证记录](2026-09-21-m1-public-semantics-smoke.json)。

Z 512²/P7 的四组预取配对中，D1/Q2 完整 wall 中位数只缩短 **0.35%**，denoise 缩短 1.89%，最后一组完整 wall 反而慢 2.18%。8 张图片逐字节一致，但 3 次采样间隔超限，整体内存资格保持 **INCONCLUSIVE**；因此维持 D0/Q1 默认判断，不宣称稳定加速。详见[预取测量记录](2026-09-21-m1-public-prefetch-validation.json)。

GPU/ANE 的 FP16 分支局部检查通过不代表整图通过：真实 prompt 的最终 latent 与 GPU 参考相对 L2 约 0.0972，尚无整请求质量/性能资格，也未找到可公开承诺的最佳分工。迁移一致性证据与质量认证分开记录，详见[逐轮实验日志](2026-09-20-m1-streaming.md)第 55–56 轮。

## 本次代码整理和验证

新增两个职责独立的模块：

- `tools/native/verify_streaming_acceptance.py`：检查六类人工验收报告的覆盖、commit/policy/binding、原始文件哈希与大小。拒绝路径越界、符号链接、FIFO、超大 JSON、重复键和变化中的文件。
- `tools/native/verify_streaming_release_evidence.py`：重跑原始 P0/P1/P2（可选 P3）campaign verifier，对比持久 summary、来源 commit 与策略绑定，再汇总评审状态。输出固定 `production_authorized=false`，不生成或安装生产记录。

测试位于 `tests/native/test_streaming_acceptance.py`。本次整理后重跑验收测试 **10 项**和策略回归 **8 项**，均通过。端到端工具测试实际执行 synthetic campaign verifier；数据仍是合成 fixture，不是模型或 App 验收。此前原生回归为 80 PASS / 3 skip（第 65 轮），本轮没有改动 native/Swift，也没有重跑原生构建或模型测量。源码与日志哈希见[本轮验证记录](2026-09-21-release-evidence-validation.json)。

验收报告声明仍需人工核查；reviewer 名称和摘要不是身份认证或数字签名，哈希一致不自动证明行为正确。binding 的完整嵌套 schema 和执行权限仍由后续 catalog/native 集成负责。

工具调用示例（所有输入均须是独立准备并审阅的真实证据）：

```sh
python3 tools/native/verify_streaming_release_evidence.py \
  --frozen-policy /path/to/frozen-policy.json \
  --binding /path/to/catalog-binding.json \
  --p0 /path/to/P0 --p1 /path/to/P1 --p2 /path/to/P2 \
  --acceptance /path/to/acceptance-bundle \
  --output /path/to/new-verification.json
```

`--p3` 可选；退出码 0 为 READY_FOR_REVIEW、1 为 BLOCKED、2 为 INVALID。输出文件须不存在。READY_FOR_REVIEW 只表示进入评审的工具结果，不是发布授权。缺失、未执行和不确定的验收项继续阻断。

## 剩余工作顺序

1. 将 calibrated 合同、证据读取与 record/schema、builder/native/resolver、打包及 App 标识一起接通，保留 strict 的 P0–P3 要求。
2. 完成真实来源/质量/生命周期/App/第二安装/打包撤回验收和正式 campaign；补齐评审后才产生生产记录。
3. 用该记录完成真实 App 成功发布与 GUI 验收；再根据合格的测量决定默认布局和产品声明。
4. 继续完整 GPU/ANE 接入及整图质量、内存、性能比较；局部 bank 或阶段收益不能替代整请求收益。

当前分支为 `feat/stream`。截至此前 fetch 的 `origin/dev` 为 `61c08495815d645bb54d75ae9dbea466f0648b2d`，已是当前分支祖先；本次核对本地远端引用，未再次 fetch。此次提交是代码与文档快照，不表示已合并至 dev 或已发布。
