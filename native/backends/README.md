# GPU/ANE 后端维护说明

本页说明代码职责与不可破坏的边界，不维护另一套跑分。
选路、性能、optional 开关及未完成项统一见
[加速维护入口](../../docs/status/acceleration.md)；调用方式见
[CLI 文档](../../docs/public/USAGE.md)。

## 保留的实现

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 模型计算 | `../models/qwen21/`、`../models/z_image/` | 完整 GPU block、模型自己的 LoRA 修正与残差；不在通用后端复制模型实现 |
| 冻结图桥接 | `coreml.mm`、模型 hybrid 实现 | 保留已测最快 base 图和独立的 base-only `lora_fused` 图；两类 manifest 不混用 |
| FFN 编排 | `ane_ffn.hpp`、`ane_ffn.cpp` | token-row 分区、staging/输入就绪、GPU head 与 Core ML tail、输出所有权、失败后完整 GPU 重算 |
| 调度 | `ane_scheduler.hpp` | 按 layer/rows 采样完整 block，调整 chunks、周期复测；adapter 切换重置状态 |
| Core ML 执行 | `ane_runtime.hpp`、`ane_runtime.mm` | 单个固定形状 runtime-weight 图、单层 slots、持久 worker、逐 chunk 执行与有限性检查 |
| QKV 研究路由 | `ane_qkv.hpp`、`ane_qkv.cpp`、`ane_qkv_scheduler.hpp` | Qwen 1024² base 三份独立 Q/K/V 权重直填单一 MatMul 图；GPU 保留 norm/RoPE/attention/FFN，单独计数与失败重算 |
| 可选内存准入 | `ane_memory.hpp`、`ane_memory.cpp` | 图加载前、resident 请求与 host scratch 扩容前的机会性余量检查；压力不足安全释放并回 GPU，非内存认证 |
| 数据转换 | `ane_runtime_convert.hpp`、`ane_runtime_quant.hpp` | dense/affine SIMD staging、dtype/headroom 处理；不是 INT8 ANE 算术 |

冻结图主要分配 FFN intermediate channels；runtime-weight 分配 token rows。
长序列在同一 runtime 图上循环 chunks，不需为每层或每个 adapter 编译图。
v1 仅 base，v2 的 SwiGLU 输入支持 runtime LoRA；GGUF 集成目前仍限 base。
几何匹配不等于所有 checkpoint/adapter 已通过质量验收。

## 计算与生命周期约束

SwiGLU 的逻辑顺序（忽略 dtype 转换和内部 headroom 处理）：

```text
g = x Wg^T + LoRA_gate(x)
u = x Wu^T + LoRA_up(x)
h = SiLU(g) * u
y = h Wd^T + LoRA_down(h)
```

- checkpoint、冻结 base 图、runtime 权重槽均只含 base 权重。不要以减少
  同步为由合并 adapter 或跳过 SiLU；base 请求必须提供零修正。
- `RuntimeGraph` 的 borrowed 权重/输入/输出在异步工作结束前必须有效。
  `HybridFfn` 发布给 GPU/adapter 的结果必须独立拥有，不能引用下次复用的 scratch。
- runtime compiled graph 逐文件校验后复制到私有临时快照，仅从该快照
  加载 Core ML；先 drain worker/释放模型，后删快照。不可改回从用户原图
  延迟加载，否则源文件在校验后被替换会产生验证—使用间隙。真实移动源图
  后继续 prediction 的回归与旧/新库比较见[图快照记录](../../docs/status/runtime-ane-artifact-lease.md)。
- 私有快照的新格式在 0700 目录中持有独占 `flock`；仅在锁已释放且
  marker 的 uid/device/inode 与目录及锁文件一致时回收崩溃遗留。回收及析构删除
  使用 descriptor-relative、no-follow 操作，拒绝外部链接、不明条目和
  身份变化，失败写入 stderr。每次建图最多回收 16 个；旧格式、无完整
  marker 的极早期崩溃空目录、最后移除 marker 后留下的空目录及权限受损
  目录保守保留，不自动推断归属。
  最后 rmdir 失败且原目录身份仍可证明时，重新独占创建并锁定 marker
  以供重试；恢复失败或该短窗口崩溃仍可能留空目录，不承诺任意点零残留。
  临时目录扫描最多 4096 项，单图删除最多 4096 项/32 层；超过限制保留
  未删内容并记录诊断。快照子目录/文件显式采用 0700/0600，独立于 umask。
  这是临时图生命周期保护，不承诺 Core ML 系统缓存完全不落盘，也不
  改变权重、精度、选路或硬件资格。对应纯 host 回归不加载模型。
- 部分 chunk 失败也不能发布半成品；保留整段 GPU 重算和失败计数。
- 同 adapter 的 resident 请求可保留调度状态，切换 adapter/返回 base
  必须隔离。保留数值、非有限值、headroom、失败后状态回归。
  Qwen ref512 的 batch 诊断开关也允许显式 runtime 的 base 请求，便于
  同图 base/A/B/base；GPU/冻结图的 base 门禁不变。三图实际切换及工具
  的 prediction 增量验证见[编辑记录](../../docs/status/runtime-ane-qwen-edit-chunks.md)。
- `pre_ffn` 可能包含此前 lazy GPU 工作，不能叫作纯 attention；
  `async_ane_wait` 与 GPU 重叠，不能计作暴露 ANE 等待。测量完整 block
  的 on/off 样本先清除上游依赖；profiling 不用于正式性能结论。

`cpuAndNeuralEngine` 是执行策略，不是物理 ANE residency 证据。
图/精度近似允许视觉接近，但不允许关闭有限性、错误或隔离检查。
内存准入仅保护显式 runtime-weight，当前代码和证据见
[内存准入接续](../../docs/status/runtime-ane-memory-admission-2026-09-29.md)。

## Optional 与清理边界

runtime、`lora_fused` 都需显式选择，不提升为普通请求默认。固定 chunks、
profile、FP16 低秩、Metal 实验、缓存近似保留各自的 opt-in 和兼容门禁；
不自动组合成未经测量的“最快”预设。旧 `lora_suffix`/`lora_merged`
接口仅保留兼容，不作为本任务的完整共图方案。
Z 的真实 fused QKV 权重与合成 1024-row 输入的四分区组件筛选
[未跑赢 GPU-only](../../docs/status/runtime-ane-qkv-component.md)，没有接入 Z。
Qwen 则已把三份独立 Q/K/V 权重的 MatMul micrograph 接入显式
`runtime_qkv` 研究路由；固定一 chunk 的
[早期整请求初筛](../../docs/status/runtime-ane-qwen-qkv-product.md)只有
1.013× 单热样本，后续 `QKV_CHUNKS=auto` 的
[875b 同库 40 步正反序](../../docs/status/runtime-ane-qwen-qkv-auto.md)
相对 GPU 约 1.001×。auto 只决定一 chunk 开/关，不自动寻找 chunk 数；
两者均不升级默认，也不与 FFN ANE 图并存。

Qwen Q/K norm-RoPE 复用 `../models/qwen21/metal/qk_norm_rope.hpp`，不在
ANE 后端复制 kernel。`qwen21_module.cpp` 维护显式 opt-in 与形状/精度门禁，
pipeline 在最终路线描述之后附加实际 kernel 标记，避免 runtime 覆盖标记。
screen/switch 共用 `runtime_ane_common.py` 的环境设置与计划/实际 receipt
校验；对照的 GPU 与 hybrid 必须使用相同设置。512² 三图只有小幅收益，
保持 optional；当前构建/测试与性能依据见[组合报告](../../docs/status/runtime-ane-qwen-qk.md)。
接续[1024²组合](../../docs/status/runtime-ane-qwen-qk-1024.md)只扩展resident
base文生图的门禁，kernel本体和512²范围不变；1024² LoRA/编辑仍拒绝。
六配置36请求已完成并独立复核，runtime相对同融合GPU为1.242×、冻结图
为1.091×；单prompt视觉接近，仍不升级默认或外推1024² LoRA/编辑。

保留稳定层异步 head、readiness 合并、base hidden 免复制和 SIMD staging。
输出恢复只保留 host 可测的串行 `restore_fp16_matrix` 布局遍历，连续列
继续使用原 SIMD 行转换；scoped Core ML 读取与拥有所有权的输出拷贝不变。
四组并行输出恢复虽减少局部耗时，但两模型整请求均无收益，已撤回；
大矩阵、非连续布局、非有限值和只校验路径的回归保留。
GPU-first、输出直接写入等未采用候选，以及没有整模型 after 的整数输出
转换/零修正缓存，不恢复到执行路径；负结果与有用回归保留。

离线导出在 `tools/coreml/`，组件 probe 在 `tools/native/`，可复用对照与
共图切换在 `tools/validation/`，测试在 `tests/native/`。原生生成不启动
这些 Python 工具。一次性请求、PNG、日志、图和编译缓存只在被忽略的
`outputs/`、`results/`，便携模板在 `examples/requests/`。
验证工具的共用模块、兼容接口及证据边界见
[工具维护说明](../../tools/validation/README.md)；运行中的campaign绑定输入
保持不变，代码重构、测试和缓存清理应在计时结束后进行。
便携请求集中在[请求示例索引](../../examples/requests/README.md)，明确区分
普通GPU模板、显式 `lora_fused` 模板与 `runtime` 覆盖；不要为实验另建一套
产品路由或将机器manifest路径写入模板。

性能 screen 与共图切换共用 `runtime_ane_common.py` 的编辑 receipt 校验，
避免 operation/reference-size/token 门禁出现两套实现。工具各自保留工作
负载资格与接口；有效 receipt 不等于图像质量验收，也不改原生计算路径。

组件 probe 只保存一份 `ProbeSample` 序列，结束后从相同样本计算中位数
和加速比，避免汇总与原始证据各维护一套计时数组。保留 JSON 字段、两次
排除 warmup、交错顺序和精度/失败门禁；奇偶样本的汇总均有小图回归。
这是离线工具整理，不是新的产品推理优化或性能 after。

验证入口从仓库根目录运行，不能与 benchmark 重叠：

```sh
make test-acceleration-contract  # host math、路由和报告；可能编译 host 测试
make test-runtime-ane            # 显式小型 Core ML/MLX 集成
make test-qwen21
make test
```

准备新图另用 `tools/coreml/export_runtime_ane.py`；组件探针另用
`make build-runtime-ane-probe`。不要把导出/组件运行接入默认产品构建，
不要把 `plan`、单层结果或已有测试重标成新的整模型性能/视觉验收。
