# dev → dev-verify 整合与实现评估

日期：2026-10-04（Asia/Shanghai）。整合父版本为 `a27e8ec`（dev-verify）与
`27c6b59`（dev）。本轮没有下载模型，也没有修改用户模型、模型登记或 App 草稿。
验证使用新的 `build/merge-dev-20261004`；单独的本地测试包位于
`outputs/merge-dev-20261004/dist/TurboCider.app`。

## 整合与修复

预检发现六个冲突文件：`ane_ffn.cpp`、`ane_runtime.hpp`、`ane_runtime.mm`、
Qwen `pipeline.cpp`、Apple `request.mm` 与 `package.sh`。没有整文件偏选一方。

- Public Core ML 的 Prepared/异步提前准备、artifact lease、阶段计时与新 Executor
  接口共存。准备和绑定都包含 packed conversion scratch 的内存估算。
- 提前准备仅允许 Public 或没有私有授权的 Auto；Private、授权的 Auto 与 Off
  保留工厂选择流程。提前准备和常规请求使用相同的后端配置缓存身份。
- 自检失败后释放执行器、立即结束初始化，保留完整 GPU 重算；避免在
  `degrade()` 之后继续读取已经释放的执行器。
- 保留 dev-verify 的 Qwen 分阶段驻留、提前准备、Viggle 区分、DiT 缓存和
  编辑偏好；请求解析同时接受 `qwen21_dit_cache` 与 `quantized_execution`。
- 打包继续支持自定义输出目录；新增第三方 notice 也复制到该目录，避免写入
  固定的根目录 `dist/cli`。默认构建和稳定打包仍排除 Private ANE。
- 修正 W8A8 host fixture 的编译警告、Qwen3 重构后的旧变量断言；LoRA slice
  测试可绑定本次新构建，避免误测旧 `build/native`。
- 新增 `make test-quantized-host` 并纳入 `make test`，覆盖 GGUF 和 W8A8 基础合同。
  超分检查加入 App 的构建与 `test-app`，其测试和用户文档与当前“新建创作保留
  用户选择”行为一致；初始默认关闭，显式关闭不会被新建操作重新启用。

## 本轮验证

证据保存在 `outputs/merge-dev-20261004/`；下表的测试组有重叠，不相加为唯一用例数。

| 检查 | 结果 / 证据 |
| --- | --- |
| 原生引擎、Swift App、模型库 helper | 新构建通过，`build.log` |
| 常规 `make test` | 484 项：463 通过、21 跳过；`make-test-final.log`。随后新增的 quantized target 单独完成验证 |
| `make test-quantized-host` | 58 项：52 通过、6 可选集成跳过；`quantized-target-tests.log` |
| 新库请求/兼容合同 | 134 项：131 通过、3 跳过；包含两个分支的解析合同，`request-contract-tests.log` |
| Qwen 合成回归 | 46 项：44 通过、2 跳过；LoRA、编译权重释放、文本流式、取消与重试，`qwen-contract-tests.log` |
| GGUF Metal / Qwen3 state | 4 项通过；raw/fixed/dependency kernels、packed bank、pager、escaped views、预算和释放，`metal-synthetic-tests.log` |
| Public Core ML / MLX | 13 项通过；runtime LoRA、尾部失败重算、取消、输出拥有权、源图快照与释放，`public-ane-tests.log` |
| Prepared CPU 微型图 | 1 项通过；所有权转移、ABI、绑定预算和同步等价，`prepared-tests.log` 与 `prepared-probe/` |
| App 专项 | Studio、超分 x2/x4 合成检查与 Qwen 编辑工作流通过；`studio-tests.log`、`upscaler-tests.log`、`workflow-tests.log` |
| 打包 | 自定义目录、Public release guard、严格代码签名校验通过，`package.log` |
| 冲突/格式 | 无残留冲突标记；`git diff --check`、打包及 App 构建脚本语法检查通过 |

部分测试首先受到 sandbox 的 Metal/剪贴板/系统缓存访问限制，之后在已授权的
系统访问下重跑并通过。首次常规测试的 CLI 组误指向旧构建，最终记录显式绑定了
本轮的新 CLI。上述结果采用最终完成记录，未把中途失败或 skip 计为通过。

### 真实 Z-Image Turbo

使用已有登记中的 BF16 模型，新 CLI 同一 engine 连续生成两张 512×512 图片，
seed 42/43、同一狐狸提示词，均实际执行 8 步并发布可解码 PNG。第一张目测与
提示词相符；这是运行冒烟检查，没有做完整画质资格或分支性能对照。

| 请求 | 文本缓存 | request wall |
| --- | --- | ---: |
| 第一次 | miss | 20.153 s |
| 第二次 | hit | 15.069 s |

实际后端 `mlx_cpp_metal`，精度 `bf16`。回执与 PNG 哈希见
`generation-check.json`、`z-generation.jsonl`。这些时间来自集成验证期间的小样本，
不能解释成 dev 相对原 dev-verify 的加速比例或正式冷/热性能认证。

## 剩余缺陷与使用边界

1. **新增 GGUF 功能尚未完成 App 产品集成。** 有界执行、raw GPU affine、
   CPU-direct、保留 bank 等主要由显式实验构建、请求和环境开关控制。
   当前 `StudioDraft.request()` 仍要求普通 GGUF 使用 resident；新有界路线不是
   App 中已经可选的通用流式配置。新的 quantized execution 合同限制为 Z GGUF
   GPU 文生图，拒绝 LoRA、参考图和 ANE；格式/组件/形状支持不能按 GGUF 名称外推。
2. **节省权重内存并不保证更快。** dev 中已有的 M4 Max 64 GiB、512²/4 步
   同组件初筛：Q4 fused 路线 wall 比 BF16 慢约 34.9%，观测进程峰值约为其 40.8%；
   fixed/dependency Q8/Q4 路线慢约 31.6%/28.4%。这些不是本轮 M4 Pro 的新测速。
   供给、转换、同步、encoder/VAE 与 cache 生命周期仍是整体瓶颈，局部 kernel
   或保留 bank 的热态成绩不能替代完整请求。
3. **内存管理仍缺整请求容量保证。** managed ledger 约束权重和指定缓冲；
   GPU 激活、encoder/VAE、driver 和系统缓存还影响进程峰值。已有文件明确标记
   `whole_request_memory=unknown`；某些预算 observed fit 不等于真实低容量机器资格。
   稳定 bundled streaming catalog 仍为空，本机实验配置的既有边界保留。
4. **Private W8A8 仍是研究后端。** 默认/稳定包不包含私有 API。本轮未编译或运行
   Private 硬件矩阵；Host policy/math 与 Public 集成通过不能替代其资格。
   dev 的 M4 Max 历史初筛中，Z1024 A8 lookahead 后仍慢于 GPU，Qwen512 的 Viggle
   LoRA 约 9.263 s 对 GPU 7.946 s；base 的收益不能推广到 LoRA。
   W8/A8 数据路径和 MIL 算术并不证明已观察到物理 INT8 MAC。
5. **质量与选路尚未闭环。** Private 的 share、W/A prefetch、scale cache 与
   packed compute 等组合仍需同模型、尺寸、LoRA、设备的性能/数值/媒体验收。
   当前没有完整的跨设备自动“最快”策略，也未完成 GGUF 的 48-case 媒体资格。
6. **缺少本机 GGUF 实模验证。** 本机没有测试要求的 Qwen3 Q8 GGUF 及配套
   路径；metadata 实模测试在夹具检查处停止，未执行模型推理，也未计入通过。
   没有为此下载模型。当前正确性结论覆盖已完成的合成数据、Public 图与 BF16
   Z 实际生图，不包括新增 GGUF/Private 全模型、所有 LoRA 或整套容量资格。

历史数据来源：
[fused affine](../design/quantized-execution/26-lossless-fused-affine-import.md)、
[fixed/dependency](../design/quantized-execution/24-fixed-gpu-bank-and-dependency-ready.md)、
[Private A8/share/LoRA](private-ane-a8-lookahead-2026-10-04.md)。

当前默认 GPU 与 Public 受限路线可以继续集成验证。新增 GGUF/Private 能力应保留
实验门禁；工程整合通过不等于这些能力已成为全场景正式加速功能。
