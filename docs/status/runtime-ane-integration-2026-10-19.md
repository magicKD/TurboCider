# Runtime ANE 源码整理与集成验证

2026-10-19，Apple M4 Max /64GiB，macOS26.6.2。

## 结论与范围

此前最快的 Qwen Base 完整计算配方已进入主 native 构建：固定 FFN
R1024/H4096/F12288，ANE/GPU 各6144通道、F32 partial join、sigmoid-complement；
Private QKV R512、one-hop W staging、生成 text-prefix＋首步GPU QKV、编辑 prefix。
没有把少算 block 的 conservative cache 混入这条完整40步路线。

Turbo 原文件名与作者8步 schedule、GGUF、Comfy H256 ConvRot 的独立来源配方，
以及限定域的 LoRA GPU 融合也已纳入源码和回归。没有证据支持的 GPU-dual、
shared compiled complement、多procedure组合不加入最快配方。各模型的历史最快
配置、尺寸与限制继续以[配置总表](../benchmark/CURRENT_FASTEST_CONFIGS.md)为准。

App/default 保持 GPU、DiT cache off。Private ANE 与近似路径仍需显式选择；
本次不部署 App、不重下载模型、不恢复已经清理的大型派生包、不推送远端。

## 工程整理

- 主构建显式包含 QKV 实现；未为语言选择重写算法或改变调度。Apple API 仍由
  backend 封装，公开发布与 Private 研究构建保持隔离。
- 新增 `make test-qwen21-runtime-host`，显式列出无模型/无设备的集成回归，并接入
  `test-acceleration-contract`；GPU 生命周期与模型窗口不混入 host 目标。
- 十二个新探针的默认库目录统一为 `build/native`，仍可用
  `TURBOCIDER_NATIVE_LIBRARY_DIR` 指向新构建；不再默默链接旧实验 ABI。
- 提供[便携 shape descriptor](../../examples/coreml/README.md)，R1024配置与本次
  实际使用的历史descriptor逐字节一致；R1536/R2048用于已有Turbo尺寸特定配方。
  它们不包含权重、编译模型或机器路径，不代表新增尺寸资格。
- `.gitignore` 覆盖 `outputs/`、`output/`、`Output/`、所有新 `results/`，继续忽略
  build、权重、trace与缓存。原已跟踪的证据不删除；源码、便携JSON与小fixture
  仍可跟踪，正/负向 ignore 回归均覆盖。
- profiling脚本、探针、审计、测试与提炼文档入库；原始图片、日志、模型、源码
  tar归档与编译产物全部留在忽略目录。本次无需额外删除文件回收空间。

入口与命令见[profiling索引](../../tools/profiling/README.md)。本轮也发现旧基准
测试把“原BF16模型存在”误当成“可丢弃的Q4派生包存在”。历史报告元数据继续
严格检查；实际派生包存在性改为显式
`TURBOCIDER_TEST_FLUX9_DERIVED_ASSET=1` 的独立测试，缺fixture不冒充已验证。
首次失败日志保留，修复后另存重跑日志，不改写历史性能记录。

## 当前构建与回归

两种库都从当前源码重新完整构建，未复用旧C++ helper：

- Private：`build/runtime-ane-integration-private-v2-20261019/libturbocider.dylib`。
  Runtime ID `tc-runtime-build-v1-12070d42505a60d20ead504e72928e0565d54a4de2622d45709f97ed456c7fc3`。
  库SHA256 `291ed489ae4e3429d64cdf838da79661bdf62c3ffc8c08700e3a42c94b22954e`。
- Public：`build/runtime-ane-integration-public-v2-20261019/libturbocider.dylib`。
  Runtime ID `tc-runtime-build-v1-4e3e5ef345ece2ba867781b49e4a5598caf337cc59b1c94bd86d6d7c3e265029`。
  `check_release_binary.py`通过，包括实际符号、字符串、动态链接与build policy检查。
- 新建observer：`build/runtime-ane-integration-observer-20261019/libqwen_base_profile.dylib`。
- Private exact-manifest源码归档包含669个输入，
  `outputs/runtime-ane-integration-native-source-v2-20261019.tar.gz`，
  SHA256 `dd529385f2763cc18c057a80d140161a8fe1d11f586fd1bc884d2cd397a5d62d`。

构建日志为`outputs/runtime-ane-integration-{private,public}-v2-build-20261019.log`。
仍有编译器的弃用/未使用符号警告，不能称为零警告构建；两种构建均成功。

| 检查 | 结果 | 本地日志（均在outputs/） |
| --- | --- | --- |
| 新增host集成suite | 364项通过 | `runtime-ane-integration-host-v2-final-20261019.log` |
| 仓库、忽略规则、边界与知识库 | 45项通过 | `runtime-ane-integration-repository-v2-final-20261019.log` |
| 原runtime/量化host与base benchmark | 343项执行，328通过、15显式跳过 | `runtime-ane-integration-existing-host-rerun-20261019.log` |
| Private C API与真实MLX回归 | 153项执行，152通过、1缺Wan fixture跳过 | `runtime-ane-integration-native-v2-final-20261019.log` |
| Public C API | 141项执行，140通过、1缺Wan fixture跳过 | `runtime-ane-integration-public-v2-contract-20261019.log` |

Private实际覆盖Transformer释放/快照隔离、五/六/八步sigma、ConvRot F32来源、
VAE融合/完整FP32重试/取消、异步step drain、direct QKV/typed SwiGLU prefix。
encoder fixture另实际执行64组Metal RMS/NeoX/GQA与packed投影/来源lease检查。
所有新shell构建脚本语法、相关Python语法与diff空白检查通过。
跳过项不算通过；未运行完整App/Swift测试或所有模型的整图/整视频性能矩阵。

## 新库完整请求验证：9张图片

所有构建和测试停止后，串行运行三个timeline窗口，每个seed73→74→73：

| 配方 | 完整请求 | 实际稳定ANE模型数 | 热compile/load |
| --- | --- | ---: | --- |
| Base生成，完整40步、cache off | 3张512²；每张1280个成功混合FFN block | 2（FFN、QKV） | 0 /0 |
| Base编辑，完整40步、标准1024参考、cache off | 同上；真实prefix miss/hit | 2 | 0 /0 |
| Turbo生成，作者8步、cache off | 3张512² | 1（FFN） | 0 /0 |

Base生成QKV每请求1248次，编辑1248/1280/1280次，符合首步与prefix策略；
无FFN fallback或overflow retry。没有用失败GPU回退冒充ANE成功。
实际selector观察只证明被观察到的API行为，不是opaque driver内部行为或物理
INT8 MAC/硬件并行的证明。

新Base六张PNG与V12完整计算off控制逐字节相同；Turbo三张与原最快完整窗口
逐字节相同。这是当前fixture的回归证据，不外推任意prompt/尺寸逐位等价。

原始窗口为`outputs/runtime-ane-integration-base-generate-v2-20261019/`、
`runtime-ane-integration-base-edit-v2-20261019/`与
`runtime-ane-integration-turbo-generate-v2-20261019/`。
独立base审计为`outputs/runtime-ane-integration-base-independent-audit-v2-20261019.json`，
Turbo为`outputs/runtime-ane-integration-turbo-independent-audit-v2-20261019/audit.json`。

本次timeline原始热中位数约28.661s /35.723s /7.546s，但均有外部推理CPU负载；
本次用于集成回归，不产生新的正式倍率或最快纪录，不用timeline覆盖历史stock。
没有停止共享ComfyUI、App或ANE服务。

## 之前启动的编辑holdout：保持近似档默认关闭

沿用预先固定的三组prompt/seed、同一参考、GPU off/runtime off/runtime
conservative三臂，九窗口27张完整图片；独立审计实际请求、原来源全SHA、原始
memory/load、实际执行/跳层、全部RGBA与固定数值预算。

| 指令 | Runtime缓存对完整GPU的RGB SSIM范围 | alpha最大差 | 暂定联合预算 |
| --- | --- | ---: | --- |
| 透明玻璃材质，seed173/174 | 0.992175–0.993621 | 18 | 未过；三张中两张alpha超≤4 |
| 添加柠檬，seed211/212 | 0.989898–0.994871 | 2 | 三张通过 |
| 写TEA，seed307/308 | 0.991333–0.995555 | 2 | 三张通过 |

玻璃组对原runtime的增量alpha也达5/7；不能仅用较高RGB SSIM宣称缓存无损。
没有调宽RGB SSIM≥0.98、MAE≤3/255、alpha≤4预算，也不把数值通过当语义合格。
三个缓存热窗口原始约26.33–26.51s，少算middle blocks，正式speed均null。
完整审计：`outputs/runtime-ane-integration-edit-holdout-audit-20261019.json`。

所以此前红茶壶样例的通过不能泛化为所有编辑。近似缓存继续默认关闭，原完整
计算配方不改；普通LoRA、多参考、其他尺寸与真实驱动失败/超时仍须各自验证。
