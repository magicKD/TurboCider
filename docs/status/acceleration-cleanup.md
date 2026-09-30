# GPU/ANE 实验代码收尾记录

当前选择与性能统一维护在[维护入口](acceleration.md)。本页按阶段记录整理，
避免将历史测试算作本次新增验收。
2026-09-30 的代码结构、最新 BF16/Q8/QKV 对照、整理/提交边界和
本轮构建回归见[当前代码与进度](runtime-ane-current-2026-09-30.md)。
以下各节保留原执行时点的状态与测试口径，不代表今日的运行库。

## 历史整理：当时候选、已验证性能与维护边界

2026-09-29，本轮按进度总结与整理请求收尾，不再新增优化候选。
当前运行库为55e5，3986是已完成512²匹配性能/视觉对照的保留构建；
修正维护入口中“当前3986”和测试数量的过时描述，不把旧成绩归给新库。

- 保留普通GPU、已测最快冻结base、完整 `lora_fused`、runtime v1/v2，
  以及异步head、readiness合并、SIMD staging、token-row多chunk和失败重算。
- 继续按已有显式入口区分runtime、Q/K融合、c1792图几何、FP16低秩、
  固定chunks/profile和缓存近似；不增加“自动最快”预设或恢复失败候选。
- 新增 `tools/validation/README.md`，集中说明共用校验、整请求对照、
  adapter切换、内存采样和离线设备计划的职责及证据边界；与后端说明、
  状态索引互相连接。保留既有公共helper和工具路径，不破坏实验复现。
- 离线设备计划工具现已实现并有五份报告；它不等于物理ANE驻留或重叠
  证明。旧整理段落中的“尚未实现”只描述当时状态。
- 增加 `examples/requests/README.md`，集中说明普通GPU、最快冻结base、
  `lora_fused` 和runtime v2模板如何选择。明确冲突覆盖会被拒绝、图不能
  混用、LoRA不合并及manifest覆盖才代表近似授权；不复制机器路径。
- 将旧操作指南的bc1及更早性能表、1024²进度和验证缺口明确标为历史，
  消除“最新/当前”措辞与维护入口冲突；原始数据、负结果和工具路径保留。

1024²六配置正反向对照仍在运行。本轮只改未被campaign绑定的维护文档，
不改推理源码/共享验证工具，不运行构建、测试、导出或其他推理，不移动或
删除图、缓存、PNG和原始结果。性能快照仍使用已完成的匹配对照，完整
实验未结束前不发布55e5的1024²加速或画质结论。

55e5构建、91项加速契约、微图、Qwen专项及完整回归是启动计时前的
验证，见[候选报告](runtime-ane-qwen-qk-1024.md)，不是本轮重跑。
本轮只做文档链接、机器路径、diff格式及绑定输入/暂存区身份检查；
未stage/commit，249项既有暂存移除及其本地文件不动。
实际检查：五份维护文档的108个本地链接目标存在，无机器用户目录路径；
`git diff --check`通过，25个campaign绑定输入SHA不变，暂存diff SHA仍为
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`。
检查时1024²父报告为incomplete、已完成1/12 trial；没有重启或中止队列。
代码整理收尾与仍在运行的性能实验分开，不表示所有研究目标已完成。

以下按阶段保留原始记录；“当前”“本轮”和测试数量只指该阶段。

## 前次整理：Q/K 可选组合与统一维护入口

2026-09-29，按最新的总结与整理请求收敛维护范围，不继续新增候选。
当前运行库是3986，1080只是此前保留版本；修正概览中容易混淆的版本
叙述和索引中的“最新”标签。当前性能表只保留各工作负载的代表匹配结果，
旧构建、低秩精度消融与短/长 batch 的细节仍在专项报告，不删除历史证据。

- 普通 GPU、最快已测冻结 base、完整 `lora_fused`、runtime v1/v2 及
  已有异步/转换优化全部保留，默认不变，LoRA 不合入 base 权重或图。
- Q/K norm-RoPE 复用已有 GPU kernel，runtime 组合显式 optional；CLI
  使用文档补齐开关、可移植命令、兼容边界及与开发工具 flag 的区别。
- 后端说明补齐 kernel、模型门禁、pipeline 实际标记、共用验证函数的
  维护职责；不为整理搬动稳定代码或恢复已撤回的输出并行等候选。
- 本轮不扩展尚未实现的 runtime 设备计划检查工具；计划偏好也不能
  替代物理 ANE 执行证据。可复用工具/测试继续在现有目录，一次性
  campaign driver、原始结果、PNG和编译缓存继续留在忽略目录。

3986库的构建、微图、79项加速契约、Qwen专项和完整回归已在前一实验
阶段通过，见[组合记录](runtime-ane-qwen-qk.md)，不称为本轮重跑。
接续的Qwen base对照仍运行，保持库、绑定源码、driver和图身份不变；
不并行构建、测试或其他模型工作。未完成的正反向结果不进入最终性能表。

本轮只做维护文档整理与静态检查；不删除模型、缓存、图片或原始结果，
不stage/commit，249项既有暂存移除及本地文件不动。整理完成不代表
更多LoRA/编辑/分辨率或物理ANE驻留等研究目标已完成。

整理检查时Qwen base已完成7/12 trial，父报告仍为incomplete；队列继续，
未重启或终止。五份维护文档126个本地链接目标存在，无机器用户目录路径，
`git diff --check`通过。运行库SHA保持3986，暂存diff SHA保持
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`；
249项暂存移除对应的本地文件全部存在。这些是静态检查，不是新增性能
或完整回归验收。本轮不改推理源码，后续完整campaign结果另行归档。

## 前次收尾：输出恢复候选撤回与保留库验收

2026-09-29，响应“总结实验进度、整理代码、关键加速保留、实验 optional”
的收尾请求。本轮不继续性能搜索，不新跑整模型 benchmark，不改默认选路、
模型权重或图；接续并确认已有串行构建/回归正常结束。

### 保留与 optional

- 保留普通 GPU、已有最快冻结 base 图、完整 `lora_fused`、runtime v1/v2，
  以及 token-row 多 chunk、自适应调度、异步 GPU head、SIMD staging、
  base hidden 免复制、完整失败重算与 adapter 隔离。
- LoRA 始终 runtime 运行，base checkpoint、Core ML artifact 和 runtime
  base slots 均不含 adapter；完整 SiLU 与 gate/up/down 修正顺序不变。
- runtime、共图 LoRA、c1792 图几何、固定 chunks、profile、FP16 低秩、
  Metal 实验与缓存近似保留各自显式入口和兼容门禁；没有自动组合为
  “最快”预设。选用原则与性能只维护在[决策入口](acceleration.md)。
- 并行输出恢复局部约3.69×，但Qwen三图全热请求慢0.62%，Z base慢1.20%。
  四组GCD候选已撤回，不另留开关；保留串行矩阵布局 wrapper、SIMD行转换、
  大矩阵/非连续布局/NaN/溢出/只校验回归。完整负结果及构建身份见
  [专项报告](runtime-ane-output-restore-parallel.md)。

### 本轮代码与文档

隔离构建 screen 的 `--cli` 在原来的文件/相邻库检查上补齐执行权限检查；
缺CLI、目录CLI、不可执行文件与缺相邻库均在创建证据目录前拒绝。测试
覆盖四种错误及有效隔离构建的调用/哈希归属，不改变默认工具或产品路径。
没有删除 SHA 校验，也没有修改既有 campaign 的源码/工具身份记录。

维护入口、后端职责、验证指南与负结果报告补齐撤回后状态；历史性能仍
绑定原构建，当前保留库不能冒用旧成绩。原始请求/PNG/日志与一次性runner
继续留在忽略目录，复用工具/测试留在 `tools/`、`tests/`。

### 实际验证与边界

接续的 `TURBOCIDER_NATIVE_ONLY=1 make build`、
`make test-runtime-ane test-acceleration-contract test-qwen21`、完整
`make test` 均正常结束（exit 0）。保留库 SHA：
`1080afeb54e0d20e4f608c94d65c1e33f51b7236c6eb0b7d3f1ba6b00aa420c2`。
包含4项host、9项Core ML/MLX微图、76项加速契约（含同一4项host），
Qwen 38+7+30项及三个native/工作流入口。原日志位置见专项报告。
缺夹具、专用构建或GPU opt-in的skips不算新增覆盖。

本轮工具修改后的契约/Qwen与完整回归另行串行执行，均已exit 0；
76项加速契约及Qwen专项全部通过，完整回归仍保留上述skips边界。日志为
`outputs/runtime-ane/acceleration-cleanup-contracts.log`、
`outputs/runtime-ane/acceleration-cleanup-full-test.log`。
六份维护文档104个本地链接目标存在，文档及修改的工具/测试无机器用户
绝对路径，`git diff --check` 通过。运行库SHA不变，暂存diff SHA保持
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`，
249项暂存移除对应的本地文件全部存在。
没有在1080库上重测整模型性能/视觉；更多训练LoRA、1–2图/1024²编辑、
物理ANE驻留及完整内存资格仍未完成。整理不等于研究目标全部完成。

未stage/commit，未删除模型、缓存、PNG或原始结果；既有249项暂存移除
及其本地文件保持不动。以下各节是历史阶段记录，其“本次”只指各自阶段。

## 前次收尾：三图长 resident 完成与编辑校验去重

2026-09-29，按“总结进度、整理代码、保留关键路径、加速 optional”的
请求收尾。接续运行的 36 请求 ABBA 已正常退出；没有启动新的性能搜索，
没有重建或更改产品原生库、调度参数、权重、图或默认选路。

### 当前结论与保留边界

- [三图编辑报告](runtime-ane-qwen-edit-chunks.md)补齐新 8e 库结果：全部
  热请求 GPU/runtime 为 **12.895099/12.826079 s（1.005×）**，预声明
  热6–8 为 **12.909628/12.682185 s（1.018×）**。小收益不升级为默认；
  不删除 GPU 探测样本，不拼接旧库分母，不把较晚窗口叫作全面收敛。
- 36 条原始结果、工作负载、精度、实际调用、driver 身份与四份独立
  内存流全部重验；4,801 个样本，trial2 有 983,040 bytes 系统 swap-in。
  末次 runtime 确有 31 次 prediction，和匹配 GPU 图片肉眼接近，蓝壶
  透明质感仍共同不符合预期；不扩展为多提示词/多 adapter 质量验收。
- 普通 GPU、已有最快冻结 base、完整 runtime LoRA、token-row/chunk、
  自适应调度、异步 head、SIMD staging 与失败重算保留。LoRA 从不合入
  base checkpoint、Core ML artifact 或 runtime base slots，完整 SiLU
  及修正顺序不变。固定分区/profile/FP16/Metal/缓存近似独立 optional。

### 本次代码与文档整理

将 screen 与共图切换重复的 operation/reference-size/token 校验收敛到
`tools/validation/runtime_ane_common.py::validate_edit_results`。screen
继续导出旧 helper，switch 保留原 helper 和断言式失败接口；两者各自的
工作负载资格与图/adapter 身份检查不变。增加共用实现、调用委托、生成
跳过、空结果与非整数参考尺寸回归；不新增实验开关或推理依赖。

维护入口、状态索引、CLI、后端说明、验证指南与专项报告同步更新；
历史报告/负结果保留。原 driver 的工具哈希记录运行时版本，本次整理
发生在 benchmark 结束后，不能覆盖历史哈希或称为新的性能 after。

### 本次实际验证

`make test-acceleration-contract test-qwen21` exit 0：75 项加速契约
（4 host + 37 screen + 6 memory + 16 switch + 12 CLI），Qwen 专项
38 + 7 + 30 项与三个 native/工作流入口通过。仓库布局 8 项、独立性
7 项全部通过；本次没有重跑完整 `make test` 或 Core ML 微图套件，
上一阶段的完整回归/真实共图切换证据留在专项报告，不重标为本次执行。
去重后的两套工具接口再次验证全部 36 条已保存编辑结果；七份文档的
125 个本地链接目标存在，变更工具/测试/文档无机器用户绝对路径，
`git diff --check` 通过。它们是静态/保存证据检查，不是新增推理跑分。

运行库 SHA 保持
`8e533d4daf9edd3e45889ea15463c9bff33be85479d0941cd1debebc670cdf9b`；
暂存 diff SHA 保持
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`。
未 stage/commit，未删除模型、图、缓存、PNG 或原始结果；249 项既有
暂存移除及其本地文件保持不动。更广的编辑/adapter、物理 ANE 驻留、
完整内存资格与 INT8 runtime 仍未完成，整理收尾不等于研究目标全部完成。

## 前次收尾：chunk 对照完成与探针整理

2026-09-29，按本次“总结进度、整理代码、保留关键路径、实验 optional”
请求收尾。接续的 c1792/c320 ABBA 正常结束后再运行测试；未启动新一轮
性能搜索，未改产品推理源码、运行库、默认选路、checkpoint 或 base 图。

### 保留与精简

- GPU、512² 已测最快冻结 base 图、完整 `lora_fused`、runtime v1/v2、
  token-row 多 chunk、自动调度及失败 GPU 重算全部保留。LoRA 不融入
  checkpoint/Core ML artifact/runtime base slots，SiLU 与修正顺序不变。
- 将组件探针的九套重复计时数组收敛到一份 `ProbeSample` 序列，汇总在
  计时结束后统一从原始样本取中位数；原 JSON 字段、两次 warmup、交错
  顺序、数值与失败检查不变。这不是新的产品加速或组件性能 after。
- 小图测试覆盖奇数/偶数样本中位数及两种 speedup 的汇总一致性；保留
  CPU reference、多 chunk、LoRA/失败后状态隔离等既有回归。
- 精简决策入口与状态索引的重复逐轮叙述；历史报告、负结果与原始证据
  不删。后端维护说明补充探针统计边界，CLI 文档记录 c1792 的显式入口。
- c1792 仅为 Qwen 1024² base 的 optional 图几何；固定 chunks/profile、
  FP16 低秩、Metal 实验和缓存近似继续独立 opt-in。不把它自动套用于
  512²、1–3 图编辑或任意 LoRA，也不新增一个未经测量的“最快”预设。

### 实验结论与复核

[c1792/c320 交错对照](runtime-ane-qwen-chunks.md)四 trial、12 请求全部
complete。每配置全部四热样本中位数为 c320 **164.999891 s**、c1792
**153.049528 s**，即 **1.078× / 耗时减少 7.24%**。产品库 e55 不变。
该组没有普通 GPU/冻结图，不能拼接上一组分母发布新三路加速比。

原始结果、请求与图身份、累计调用重新核验；无失败、fallback 或重试。
四份独立内存流/报告/工具哈希重验通过，共 19,336 个样本、最大间隙
115.172125 ms。trial2 有 64 KiB 系统 swap-in，全组无新增 swap-out 或
compression；不是完整 driver/ANE 内存资格。查看反向 c320/c1792 末次
热图，狐狸主体、构图与细节观感接近；只有一个 prompt/seed，不外推到
编辑或训练 LoRA。完整数据及冷启动、PNG 哈希边界留在专项报告。

### 本次实际验证

benchmark 正常退出且竞争推理预检为空后，执行
`make test-runtime-ane test-acceleration-contract`，exit 0：4 host、9 Core ML
图/MLX 集成通过；加速契约共 66 项（其中 4 host 为同一依赖，只执行一次），
没有 skip。测试在临时目录编译整理后的探针；未重建产品运行库。既有
Core ML 临时目录 ResourceWarning 未导致失败。本轮没有重跑完整
`make test`，不将以前的完整回归记录算成本次执行。

另行执行仓库布局 8 项、独立性 7 项，全部通过；六份本轮维护文档的
115 个本地链接目标存在，探针/测试及这些文档无机器用户路径，
`git diff --check` 通过。产品运行库 SHA 仍为
`e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`，
暂存 diff SHA 仍为
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`；
249 项暂存移除的本地文件全部存在。本轮 benchmark 与测试均已结束。

未 stage/commit，未删除模型、缓存、PNG 或原始结果；既有 249 项暂存
移除与本地对应文件保持不动。研究目标仍有 LoRA 编辑、大 chunk 的真实
ANE 修正路径与更广视觉覆盖等未完成项，整理收尾不等于全部研究完成。

以下为更早整理阶段的原始记录，其“本次”只指各自阶段。

## 进度汇总与维护入口整理

2026-09-29，响应最新的进度总结与代码整理请求。本轮只整理维护文档，
不继续开展候选搜索，不改推理源码、调度、CLI 默认、权重或 Core ML 图。
已有 1024² 六 trial 对照继续运行，不重启；GPU/runtime/frozen 正向 trial
已完成，完整性能、反向一致性和视觉检查仍待结束后验收。

- 精简 `acceleration.md` 中累积的逐轮叙述，只维护当前选择、匹配跑分、
  optional 和未完成项；负结果、历史构建与详细验证仍保留在原专项报告。
- 新增 `native/backends/README.md`，明确模型 GPU 计算、冻结桥接、共享
  FFN 编排、调度、Core ML worker 和转换的职责。不为文档整理搬动稳定代码。
- 记录完整 LoRA/SiLU 顺序、base-only 权重、异步 buffer 生命周期、部分
  失败完整重算、adapter 状态隔离，避免后续优化破坏共图要求。
- 保留 GPU、最快已测冻结 base 图、完整 `lora_fused`、runtime v1/v2；
  近似优化仍显式 optional。已撤回候选不恢复，也不新增实验开关。
- 更新 1024² 阶段记录，明确三个已完成正向 trial 和请求间波动；不使用
  incomplete summary 发布最终加速比，不将单层或 base 收益外推到编辑。
- 修复原生目录使用文档入口，连接统一状态与代码维护说明。

本轮不运行构建、测试或其他推理，以免影响仍在运行的 benchmark。
此前通过的 66 项契约、微图/Qwen 专项和完整回归保留原验证日期与范围；
不能写作本轮通过。没有新增图像质量验收，也没有新性能优化的主张。
本轮仅检查文档链接、机器路径、diff 格式及库/暂存区身份。
实际检查：六份文档的 100 个本地链接目标存在，无机器用户目录路径；
`git diff --check` 通过。运行库 SHA 仍为 `e55b1a06…`，暂存 diff SHA
仍为 `694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`；
249 项暂存移除对应的本地文件全部存在。这些是静态整理检查，不是测试套件。
未删除缓存、模型、图片或原始实验；未 stage/commit，既有 249 项暂存
移除及对应本地文件保留。整理完成不等于长时实验或研究目标完成。

## 本次整理

2026-09-29，按本次请求停止扩展性能搜索，整理代码与现有证据。
本次不改变原生计算、调度顺序或默认路线，不新增候选开关。
此后接续的优化轮单独记录在[LoRA 初始采样消融](runtime-ane-gpu-first.md)，
不把后续候选的构建/测试算作本次整理。

- GPU、最快已测冻结 base 图、完整 `lora_fused`、runtime v1/v2 及其
  token-row 多 chunk、自适应调度、失败 GPU 重算全部保留。
- 近似 FP16、GPU Metal 实验、DBCache、固定 chunks/profile 仍分别
  optional；未验证的组合不包装成“最快”预设，LoRA 不融入 base。
- 将 screen、共图切换、独立内存工具重复的文件 SHA256 读取统一到
  `runtime_ane_common.sha256_file`；库文件改为流式哈希，不整文件读取。
  保留原 `sha256` 调用名、凭据格式和前后校验，不删完整性检查。
  这只是验证工具整理，不是推理加速，也不是不可变 artifact lease。
- 增加三项回归：共用同一实现、空/二进制/多块文件哈希一致、缺失文件
  失败且不创建文件。无新增第三方依赖，生成时不需要这些 Python 工具。
- 未实现的 LoRA GPU-first 初始采样候选暂停；保留普通 hybrid-first
  调度，不将刚结束的原版基线算作候选 after 或候选的负结果。
- 原始 PNG/请求/日志、一次性 runner 仍在忽略目录；未删除模型或缓存，
  未 stage/commit，已有 249 项暂存移除及对应本地文件保持不动。

### 刚结束的原版 LoRA 基线

接续上一轮已运行的 benchmark，等其正常结束后才修改验证工具及运行测试；
未重建原生库。两模型均为库
`e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`，
M4 Max 64 GB、512²、resident、auto chunks、profile 关闭、100 ms 独立采样。
每模型 GPU → runtime → runtime → GPU，每 trial 一次冷请求、两次热请求。
下表中位数取每路线全部四个热请求，含 VAE/PNG，不含冷请求或模型加载。

| 工作负载 | GPU 热请求中位数 | runtime 热请求中位数 | GPU/runtime |
| --- | ---: | ---: | ---: |
| Qwen Viggle v0.2.1 r256，6 步，三张 ref512，FP32 低秩 | 12.912022 s | 14.001179 s | 0.922× |
| Z distill patch LoRA，8 步 | 8.584159 s | 7.766461 s | 1.105× |

冷请求分别为 Qwen GPU 16.470/16.637 s、runtime 18.046/18.180 s；
Z GPU 11.820/9.227 s、runtime 8.846/8.843 s。冷请求不能替代加载到退出
的总耗时，route 顺序也可能影响缓存，故不据此宣称普遍冷启动优势。
Qwen 两个 runtime trial 的热请求 hybrid block 增量为 42/10、47/15；
Z 均为 248/248。Qwen 自适应减少混合层仍未抵消整请求开销。

两份 summary、8 份原始 JSONL（24 请求）重新通过 backend、LoRA、
计时与失败遥测校验；Qwen 精度标记、参考 SHA/顺序/token 数一致。
重新运行独立内存 verifier，原始文件、报告、工具 SHA 与 correlation
均匹配：共 2,834 个样本，最大间隙 110.020 ms，无新增系统 swap-in/out。
这是进程树加载/冷/热/退出范围，不是完整 driver/ANE 内存资格。

原始证据（目录名沿用候选计划，但只有原版 before）：

```text
outputs/runtime-ane/qwen-edit3-gpu-first-before/
outputs/runtime-ane/z-lora-gpu-first-before/
```

本次没有新增肉眼质量验收，不用哈希/遥测通过代替视觉检查；代表样本的
既有视觉结果见[低秩精度实验](runtime-ane-qwen-lora-rank.md)。base/frozen
未重测，沿用各自历史构建的匹配成绩，不跨构建计算提速。

### 本次验证

`make test-acceleration-contract`：4 host + 37 screen + 6 内存 wrapper +
8 共图工具 + 11 CLI，共 66 项通过，无 skip。
完整 `make test` 成功结束（exit 0），包含布局、独立性、便携路径、
其他模型与 streaming 回归；缺夹具、专用构建或 GPU opt-in 的 skip
不计为新增覆盖。既有临时目录和 macOS 链接目标版本 warning 未导致失败。
四份维护文档的 91 个本地链接目标存在且无机器路径，`git diff --check`
通过。暂存 diff SHA 保持
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`，
249 项暂存移除对应的本地文件全部存在。
本次未修改或重建原生库；未另跑 Core ML 微图或整模型 after 实验。
基线与测试均已结束，没有遗留本轮推理任务；整理完成不代表研究目标全部完成。

## 历史整理：撤出未完成候选，保留状态回归

按用户要求总结并收尾，不继续扩大性能搜索。已有 GPU、最快冻结 base 图、
完整 `lora_fused` 与 runtime v1/v2 保留；近似 GPU kernel、direct-FP16、
DBCache、内存采样等继续 optional，不改普通请求默认，不合并 LoRA。

- 移除 `ane_runtime.mm` 的零修正输入缓存状态与失效分支、
  `RunResult` 的专用清零流量计数，恢复每次 base launch 清零。
  候选仅完成微图和原版 before，未完成 after；不是已证明无收益。
- 将候选测试整理成独立的 `correction_input_isolation_tests`：保留
  CPU/NE、scalar/SIMD、层/多 chunk、正负 adapter、部分失败及 headroom
  后的 base 隔离检查；不再断言已撤出候选的缓存/流量实现细节。
- 增加[零输入候选记录](runtime-ane-zero-input.md)，概览与索引明确
  已测、optional、兼容诊断、未验收的区别；避免给失败候选逐个添加开关。
- 复核六份既有 summary、24 份原始请求 JSONL（72 次请求），均 complete，
  遥测和热请求中位数与性能表相符。这是复核已有证据，不是新跑 benchmark。

`TURBOCIDER_NATIVE_ONLY=1 make build` 成功，重建库 SHA 为
`7502bc307551307030ebf57a0b1e6a924bfcb3a2624e765831e9e7e869ba2c4a`。
不称其与性能实验 `8d90f74…` 库字节相同，也不把旧构建的性能/视觉结果
重标为本次执行。恢复逐次清零不改变 base/LoRA 计算顺序或有限性检查。

未 stage/commit，未删除模型、缓存、参考图或本地原始结果；保留已有暂存状态。

本次实际验证（2026-09-29）：

- 构建成功；`make test-acceleration-contract` 的 4 host + 31 报告 +
  6 采样 wrapper + 8 共图工具 + 11 CLI，共 60 项通过，无 skip。
- `make test-runtime-ane` 的 4 host、8 Core ML/MLX 微图与集成通过；
  新状态隔离测试通过，LoRA 最坏 relative L2 仍为 0.00533939，未放宽阈值。
  运行前竞争推理预检为空，没有同时运行 benchmark。
- 完整 `make test` 成功退出（exit 0），包含布局、独立性、便携路径、
  其他模型与 streaming 回归。缺夹具、专用构建或 GPU opt-in 的 skip
  不计为已覆盖。已有临时目录 ResourceWarning 与 macOS 链接目标版本
  warning 未导致测试失败。本次没有另行执行 `make test-qwen21`。
- 六份维护文档的 87 个本地链接目标存在且无机器路径，`git diff --check`
  通过。249 项暂存移除对应的本地文件均存在，暂存 diff SHA 仍为
  `694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`。

构建和测试均已结束；本次没有新增真实整模型生成、视觉验收或性能结论。
1024² 重复验证、更多编辑/训练 LoRA、物理 ANE placement 与完整内存资格
仍是未完成项，不因整理或契约测试通过而标为完成。

## 此前整理：保留计算路径，拆分验证工具

本轮按用户要求收尾，不接续新的长时间性能/内存实验。保留库为
`8d90f74ad94e63d4a1dfc1f4ce9cf5220f27b0f1491a3a2ddd2c25a722843296`，
未修改原生源码、路由默认、模型权重、Core ML 产物或误差检查。

- 将遥测校验、chunks 参数、竞争推理预检、实验环境清理及系统内存快照
  集中到 `tools/validation/runtime_ane_common.py`，只依赖标准库。
- 整模型 screen 负责请求/结果编排；共图切换工具直接引用共享模块，
  不再依赖整模型 CLI 脚本。原 screen 的函数名继续导出，兼容本地分析脚本。
- 增加“两个工具共享同一实现”和“无 SDK/无启动进程/无等待导入”回归，
  将新共享模块纳入机器路径检查。未移除原有有限性、累计计数或失败检查。
- 保留 GPU、最快冻结 base 图、完整 `lora_fused` 与 runtime v1/v2；
  近似 GPU kernel、direct-FP16、DBCache 等仍分别 optional。
  没有新增模式，没有把 LoRA 融进 base，没有恢复已撤回候选。

本轮已执行 `make test-acceleration-contract`：4 host + 31 报告/共享模块 +
8 共图工具 + 11 CLI，共 54 项通过，无 skip。
重新读取并校验八份已存在实验 summary 的 28 份原始 JSONL（84 个请求），
均 complete 且通过抽取后的校验器；重新计算热请求中位数与性能表一致。
包含两模型 async before/after、LoRA ABBA 和历史 b554 三路对照。
这不是新跑 benchmark，也没有新增肉眼质量验收或完整内存资格。

完整 `make test` 已成功结束（exit 0），包含上述契约、布局/独立性、其他
模型与 streaming 回归；缺夹具、专用构建或 GPU opt-in 的 skip 不算通过
覆盖。仍有既有临时目录 ResourceWarning 和 macOS 链接目标版本 warning，
未导致失败。本轮未重建原生库、未重跑完整模型或 runtime ANE 专项微图。
另行检查 8 项布局、7 项独立性、71 个文档链接及 `git diff --check` 通过；
库 SHA、暂存 diff SHA 不变，249 项已暂存移除对应的本地文件均仍存在。

当前库 base runtime 的 Qwen/Z 热请求为 36.663/6.332 s；GPU/冻结图的
完整匹配 base 表仍属于历史 b554 库，不能跨库计算新加速比。当前 LoRA
ABBA 为 Qwen 三图 0.908×、Z 1.102×。性能及未完成项以维护入口为准。

未 stage/commit，未删除缓存、模型或本地原始结果；既有暂存记录保持不动。

## 历史整理：异步 head 实验之前

### 范围

本轮按“总结进度、保留关键代码、实验 optional、记录文档”收尾，不继续
扩展优化搜索、不新增近似策略，不将研究目标全部标为已完成。
当前选择与性能只维护在 [acceleration.md](acceleration.md)，历史详细报告
通过[状态索引](README.md)保留。

### 代码取舍

- 保留普通 GPU、原有受限 auto 与已测最快冻结 base 图；不改变默认路由。
- 保留 base-only `lora_fused` 与 runtime v2 共图、完整 LoRA/SiLU 顺序、
  token-row 多 chunk、自适应调度、失败重算、SIMD/Q4/Q8 staging，继续
  使用显式入口。没有把 LoRA 合入任何 base 权重或产物。
- 撤回尚未完成真实模型 after 测量的 FP16→BF16 整数转换候选，不额外
  维护未验证开关。保留 65,536 位模式 × 13 个 headroom 的穷举测试，
  包括 NaN 不变 Inf、signed zero、subnormal、scalar tail 与失败状态。
  [候选报告](runtime-ane-output-convert.md)区分局部潜力与未验证整请求收益。
- 不搬动成熟后端、不制造重复实现。导出器、组件 probe、真实微图测试仍
  是独立 optional；一次性脚本/图片/结果留在被忽略的本地产物目录。
- 精简概览、状态索引与 CLI 文档的重复历史叙述；保留兼容矩阵、开关默认、
  代码职责、测试入口及未完成项。

### 实际验证

原先运行的 baseline benchmark 已正常结束，Qwen/Z 两份 summary complete；
它们都使用原库，不是整数转换候选的 after。见[补充基线](runtime-ane-output-convert.md)。
结束后才运行构建与测试，没有与这轮 benchmark 重叠。

- `TURBOCIDER_NATIVE_ONLY=1 make build` 成功；重建库 SHA 与保留库完全一致：
  `4d864b4439adb23cd53aa90dfe2884a3ced46027a4d24b4ac3e5ba2126ee6eac`。
  这验证撤回候选未改变此前已测产品二进制。
- `make test-acceleration-contract`：4 项 host、28 项报告/benchmark、
  8 项共图工具、11 项 CLI 契约全部通过，无 skip。host 在恢复后的实现上
  执行新增穷举测试；CLI 检查 optional 不提升普通 GPU 请求为 runtime。

- `make test-runtime-ane`：4 项 host、8 项 Core ML/MLX 微图与集成通过。
  LoRA / Q4-Q8 最坏 relative L2 为 0.00533939 / 0.00658136，未放宽
  阈值；临时目录 ResourceWarning 仍存在，但测试成功。
- `make test-qwen21`：38 项 Qwen 专项、7 项采样报告、29 项请求契约与
  三个原生/工作流测试入口通过；不是新的真实图像质量验收。
- 仓库布局 8 项、独立性 7 项通过，覆盖 optional 边界、便携路径、
  请求模板与 runtime LoRA 不合并。224 个本地文档链接目标均存在，
  新增/重写状态文档无机器路径，`git diff --check` 通过。

- 完整 `make test` 成功退出（exit 0），包括上述加速契约、仓库边界及
  streaming/其他模型回归；缺少模型夹具、专用构建或显式 GPU opt-in 的
  测试按原有门禁 skip，不计为已覆盖。没有新增模型 benchmark 或质量资格。

上述各项均为本次实际执行；构建和测试全部结束，无遗留本轮测试进程。

### 数据与 Git 边界

未清除模型、缓存、参考图或实验原始结果；本次是代码整理，不是磁盘清理。
未 stage/commit，保留既有 249 项暂存移除及其本地文件。暂存 diff SHA：
`694224ebd2c05cbe764f86dbc90727737ca0e47ebdafc9ee479d5f01cc331854`。
新文档与示例不写机器绝对路径，原始被忽略的运行 receipt 保留本机 provenance。

### 没有新增的主张

没有新的 GPU/runtime/frozen 配对性能结论，也没有新增真实图像质量验收。
重建库相同，保留此前已验证路径；这不等于所有 1–3 图编辑、1024²、
训练 LoRA、物理 ANE placement 或峰值内存都已获资格。
