# 独立 Native Metal Graph：首轮组件实现与验证

日期：2026-10-07。设备：Apple M4 Pro。本记录对应 [设计路线 N](../design/qwen-image-gpu-ane-command-graph/10-native-metal-graph-backend.md)，数值与实际源码摘要见 [JSON 回执](2026-10-07-native-graph-components.json)。这不是完整原生 DiT、GPU/ANE 或 App 的验收记录。

这是首轮 FFN/Foundation 历史回执；随后 Linear、attention、block 与 W0 桥的扩展和当前源码状态见 [2026-10-08 记录](2026-10-08-native-block-and-ane-handoff.md)。以下 source 摘要和构建身份对应当轮测试，不宣称覆盖后续修改。

## 已实现的边界

- `native/backends/metal_graph/backend.{hpp,mm}` 独立持有 Metal device/queue，准备 PSO，记录 dispatch/signal/wait，复制参数，提前检查队列容量，并返回保留所有资源的异步 ticket。没有 MLX 调度调用。
- `native_ffn` 在同一原生图中编码 gate/up+SiLU 和 down；支持本地 dense BF16 Qwen/Z-image 几何、原始矩阵步长、合并权重偏移、动态行数和明确 FP32 部分输出。
- `mlx_bridge` 保留实际 MLX allocator owner，以原始 Metal buffer/offset 导入权重和输入；不调用 eval、不复制张量。输出仍由 MLX allocator 分配，再导入原生执行器，避免新增一套完整权重和内存私账。
- `mlx_ffn_island` 在准备阶段建立 PSO；执行阶段导入、分配、编码、提交并等待物理完成后返回 MLX 输出。这是 **W0 FFN 纵切**。attention、norm、residual、采样和 ANE 都还不由这个新图控制。

allocation hooks 已提供，但没有把 Qwen/Z-image 的生产 `MemoryExecutionContext` 接入完成；导入不重复计费也不能替代完整资源账本认证。当前独立代码没有进入产品默认路由。

下一步投影所需的 `NativeLinear` 也已建立，BF16 源与 FP32 累加，支持动态行数及原始 view/stride；bias 明确拒绝，不能漏加 checkpoint bias。当前 K/N 限于 128 对齐的大矩阵，小输入/输出投影及不对齐尾部仍需独立支持，不能称已覆盖全部 DiT 投影。其 53 项 CPU 布局/范围合同与主机语法检查已通过，尚未跑实际 Linear shader 数值验证。新模块已列入标准 native 构建源清单；后端选择和模型 delegate 仍未接入，没有因为编译进入库就启用未验收功能。

接线审查还修复了 [private ANE ticket 的退役状态](2026-10-07-private-ane-ticket-retirement.md)：超时/拒绝后的迟到 callback 不再复活成功结果，新增非阻塞观察与独立完成证据。CPU 状态合同和严格主机语法通过；这仍没有接通新原生 handoff，旧 force-signal/预排 restore 的故障认证另行处理。

## 有界验证

主机严格编译通过。命令层设备合同测试使用 32 个 FP32 元素，覆盖参数复制、signal 处分段、外部 owner 寿命、导入不重复分配计费、范围与容量前置拒绝、host timeout 与迟到完成。实际设备测试退出码 0。

FFN 使用三种很小的几何、六次 dispatch，检查 10,368 个有限值。hidden、down 和完整输出与独立 CPU BF16 参考误差为 0；物理填充、源不可写及九种非法绑定拒绝通过。四种真实模型几何只准备 PSO，不执行真实模型。首次 prepare 因默认 Metal 语言标准不支持 MPP 失败，未提交任何命令；显式 Metal 4 与 safe/precise math 修复后通过。

真实输入仅复用已有 Qwen base 捕获：layer 0、step 1/19、`[1,1024,4096]` BF16。模型来源匹配现有 canonical/stat identity；该身份没有宣称完整 checkpoint 内容哈希认证。只求值这一层需要的权重，不物化第二份完整模型。每处输入先预热一次，各测两次，顺序为 GPU→native、native→GPU。

| FFN | 既有 GPU 平均 | 原生 wall 平均 | wall 减少 | RRMS 范围 |
|---|---:|---:|---:|---:|
| 完整 12288 通道 | 41.976 ms | 41.267 ms | 1.69% | 0.108%–0.151% |
| 6144 通道补集 | 21.374 ms | 20.879 ms | 2.32% | 0.129%–0.186% |

native wall 包含输出/hidden 分配、导入、编码、提交和物理完成等待；PSO 准备单列。两次观察到的 prepare 为 86/326 ms，驱动编译缓存状态未控制，不能称为首次冷启动开销。数据量很小，没有置信区间；这点优势不足以给出稳定加速资格，更不能推算 32 层或整图倍率。

本轮没有新模型下载、checkpoint 副本、ANE 程序导出或完整生图。原始回执在 `outputs/qwen-real-ffn-20261007/native-ffn-base-{full,half}.json`；小张量日志为同目录 `native-ffn-component.log`。

## 对设计的影响

单个 FFN 改为自己提交，并未获得明显收益。继续开发独立后端的主要理由是让完整窗口的 projection、attention、FFN、A8、staging 和 join 可统一编排，之后分别量化融合、host 间隙与真正并行的成本。现有较优 GPU/hybrid 仍须作为验收对照，不能只选择较慢的原生 GPU 对照来放大 ANE 倍率。

下一步依次为：

1. 原生投影与真实 attention/mask，再验证完整 block；不把 naive SDPA 用作 1024 query 的生产基线。
2. 接入现有资源准入、source lease 与缓存代次；不复刻独立的 Splash 内存池。
3. 给 private ANE 提供 typed event/surface lease 和物理完成许可，先做 W0 安全交接，再做有限 W1。现有 `Program::prepare/submit` 尚绑定旧 GPU device event，不能直接假装已接到新 command graph。
4. 组件完整路径明确有价值后，才用预算内的一组 base 配对 E2E；base 成立后再引入 LoRA，编辑 encoder 后续独立验证。

用户允许约 5–8% 数值差异作为探索参考，并接受同条件图像的肉眼比较。这里将它用于筛选，不作为每层或整图无条件质量资格；内存访问、同步和模型语义的正确性不受该容差放宽。

## 重现入口与缓存

`tests/native/test_native_ffn_component.py` 默认仅 CPU 合同检查；Metal prepare 与六次小张量 dispatch 分别要求显式测试开关。`tools/native/build_native_qwen_ffn_replay.sh` 使用现有依赖及既有 GPU runtime library 构建组件程序；执行方式：

```sh
build/native-graph-validation/native-qwen-ffn-replay \
  <local-BF16-checkpoint> <existing-base-capture-directory> \
  <fresh-output.json> 12288
```

将最后参数改为 `6144` 可测通道补集。不要在尚未通过小张量合同前跳过门槛直接跑模型。

统一全量构建已通过：private-enabled native 库、CLI、Swift App 与 model-library helper；构建过程中验证 runtime identity/catalog。采用过程内的 `DEVELOPER_DIR=/Library/Developer/CommandLineTools` 避开本机默认 Xcode loader/SDK 问题，没有改全局开发工具配置。新库实际包含 `Backend::submit`、`MlxFfnIsland`、`NativeLinear` 和 private `Ticket::status` 符号。该结论是编译与链接通过，没有进行 App 运行验收或完整生图。

构建输出为 `build/native-graph-20261007/`，日志及 artifact SHA 在 JSON 回执中。本轮 `.o/.a/module-cache` 和三个小组件临时构建目录已清理，删除约 **222 MB** 文件，保留约 **24.7 MB** 可用构建产物、身份清单和小型回执；重现组件时先执行构建脚本。没有删除已有模型、`dist` App、旧 runtime、原始 capture 或其他实验缓存。
