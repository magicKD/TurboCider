# 首版多模态输入、种子与高级参数设计

日期：2026-09-05，第二轮设计修订。状态：待实施产品方案；本轮只阅读代码与更新文档，未运行模型。

配套：[整体产品](app-product-design.md) · [视觉规范](app-visual-system.md) · [实现方案](app-implementation-plan.md)。本文细化输入工作台，并收紧首版为一次一个主输出。发生范围冲突时，以本文首版约束为准。

## 1. 首版范围：多输入、单输出、单个活动生成

- 每次点击只生成 **一张图片或一个视频**。视频可内含音轨；缩略图、预览与元数据属于辅助产物，不增加生成数量。
- 多张参考图共同参与一个请求，不是给每张图分别生成，不自动笛卡尔积、不自动遍历种子。
- 首版 App 同时只提交一个活动生成任务，运行时可修改下一份草稿；按钮显示“正在生成”，另设“取消”，完成后手动再次生成。没有批次数量、加入队列、自动连发或种子范围。
- 底层服务可保留既有队列，应对 CLI/其他客户端竞争；App 如收到 queued，显示“等待本机资源”，不谎报运行或立即失败。App 单任务限制不等于整个系统已实现跨进程互斥。
- 结果条是多次手动生成的历史，标“本次创作的历史结果”；不是一次请求的候选组。
- 原生 v2 `native/core/json.mm` 已校验 `outputs.count == 1`；兼容请求的 `output` 也是单个对象。UI 与契约保持一致。批量探索、任务重排放到首发之后。

## 2. 本轮代码核对

仓库持续演进，旧 README 的描述不一定等于当前源码。以下按本轮读取的文件说明，不把静态检查视为质量/性能验收。

| 文件 | 已有行为 | UI 工作与约束 |
|---|---|---|
| `Sources/TurboCiderKit/Models.swift` | `TCInputAsset` 含 type/role/path/strength/frameIndex/audioPath；request 含有序 inputs | 用资产绑定表达输入角色，保留顺序；不能将所有图都当 init_image |
| `Sources/TurboCiderApp/AppModel.swift` | `addInput` 文件选择、多选 reference、staging；固定 seed 默认 42 | 新增统一插入/拖放/粘贴入口、排序、随机种子模式 |
| 同上 `applyModeConstraints` | 删除不兼容输入并删除其暂存路径 | 改为保留未使用绑定；任务引用期间不能删除文件 |
| 同上 `maxReferenceImages/addInput` | 未声明上限时默认 1，reference 计数也可能覆盖视频/音频 | 按 type+role+operation 校验；未知上限不能标为模型仅支持 1 张或无限 |
| `src/turbocider/adapters/flux2.py` | img2img 恰好 1 张 init_image；edit 用有序 reference；透传 guidance 等参数 | 单图变换与参考编辑分开；多参考无独立权重参数，不画每图权重滑块 |
| `src/turbocider/adapters/ltx.py` | 单 first_frame，默认 strength 1.0；8+3 步；明确拒绝 guidance | 不提供多图插值或 CFG 控件；不能统一用 App 当前 0.75 初始化所有模型 |
| `native/models/flux_module.mm`、`flux.mm` | 已声明并实现 image.generate/transform/edit 路径，最多 8 图 | 上轮“编辑尚未实现”已过时；应按 operation 进行真机/一致性验收再开放 |
| `native/models/h3_module.mm` | executor=true，weight_validation=pending；首尾帧与 reference 互斥；4 步/24 fps | 已有执行入口，不等于权重与媒体已验收；分开关键帧/参考模式 |
| `native/models/ltx_module.mm` | executor=false；默认 704×448；宽高 64 倍数，帧数 8n+1 | 与兼容推荐 704×480 不同；按运行时取默认，不能共用硬编码预设 |
| `native/core/json.mm` | v2 单输出，有序 inputs；sampling 仅 seed/steps；参数仅 dynamic_text | JSON 可解析字段不等于高级参数全模型支持；不能把兼容 engine_options 原样传 native |
| `native/swift/TurboCiderNative.swift` | NativeInput 有 kind/role/path/strength；id 目前由 path+role 拼接 | UI 改用独立 binding UUID，排序、重复素材、角色变更不能导致身份碰撞 |

H3 最大参考图片数尚不能从当前 descriptor 推导：通用 native parser 的 32 资产上限不是 H3 每种角色的可用上限。增加角色上限 schema 前采取保守 UI 限制并标明是 App 限制，plan/engine 再校验，不能以默认 1 冒充模型能力。

## 3. 输入工作台：素材与提示词共同组成请求

保持中央媒体查看器、右侧参数。将原来“提示词框附带一排小附件”升级为明确的 **输入区**，默认一直显示“添加素材”和粘贴提示；有图片时展开角色槽与缩略图。图片是生成条件，不插入 TextEditor 成为富文本，也不替代结果预览。

输入区层级：

1. 操作子模式：文生图 / 单图修改 / 多图参考编辑，或文生视频 / 首帧视频 / 首尾帧 / 参考生成；只列当前模型实际可执行操作。
2. 素材角色区：原图、首帧、尾帧或有序参考列表；常规显示 80×64 pt 缩略图，附角色、序号、名称和错误状态。
3. 文本提示词：描述输出及参考用途；保留正常文字复制粘贴。
4. 提交摘要：`3 张参考图 → 1 张图像 · 512×512 · 固定种子 42`，或 `1 张首帧 → 1 个视频 · 97 帧 · 种子本次随机`。
5. 生成按钮与当前任务状态。素材导入中/校验未完成时不能提交；错误靠对应素材显示。

空素材区仅一行“添加图片 · 可拖入或 ⌘V 粘贴”。展开后最多两行缩略图，再提供“管理全部参考图”弹层，避免 8 张图把提示词和画面挤没。单图替换拖到角色槽；拖到整个输入区默认新增待分配素材，不能覆盖已有图。

### 模型对应配置

| 操作 | 输入区域 | 发送契约 | 配置 |
|---|---|---|---|
| FLUX 文生图 | 无活动图，仍可添加并选择转入编辑 | 兼容 text_to_image / 原生 image.generate | 无 strength |
| FLUX 单图修改 | 1 个“原图”槽 | image_to_image / image.transform，role=init_image | strength，默认遵从 operation，兼容当前 0.75 |
| FLUX 多图参考编辑 | 1–8 个有序“参考图”槽 | image_edit / image.edit，role=reference | 排序与用途说明，无未经实现的逐图权重 |
| LTX 首帧视频 | 1 个“首帧”槽 | image_to_video / video.image，role=first_frame | strength 默认兼容 1.0；原生执行待完成 |
| H3 首尾帧 | 首帧/尾帧两个独立槽，按运行时校验合法组合 | 原生 video.keyframes；兼容 adapter 翻译相应输入 | 不与 reference 混用；无通用 strength 滑块 |
| H3 参考生成 | 按图像/视频/音频分组，组内有序 | 原生 video.reference；兼容 adapter 保留角色与音轨配置 | 参考视频可选原音轨/静音/外配音频，仅在当前契约支持时开放 |

“图片输入”不意味着“仅图片、无提示词”已支持：兼容 `GenerationRequest.validate` 与 App.generate 都要求非空 prompt。首版仍要求文字指令；图片导入不自动编造提示词。未来仅图输入由 operation 的 prompt required 属性明确开放。

### 一张图还是多张图：不靠数量猜用户意图

- 文生图中插入一张图：出现一次就地选择“修改此图”或“作为参考”；选择后绑定角色。不自动把图片加入纯文生图请求。
- 已选单图修改时插入第二张：进入待分配区，提供“替换原图”或“转为多图参考编辑”。转换后显示计划变化，不能静默切换。
- 已选 LTX 首帧时插入第二张：保留待分配，提示“当前模式只使用一张首帧”，可替换。不能伪造尾帧/插值功能。
- H3 首尾帧组与 reference 组互斥：切换将原组存入未使用输入，保留文件和顺序；确认新的活动组合后重新计划。
- 切模型不清空草稿；不支持图片的新模型显示“2 张图片未参与本次生成”。用户可删除或切回恢复。

## 4. 插入、拖放、剪贴板统一导入

### 交互入口

- “添加图片…”：NSOpenPanel，当前允许多参考才支持多选；独立“从素材库选择”。
- 拖放：Finder 文件、App 素材、系统可提供文件/图像数据的拖放提供者，统一走 AssetImporter。
- ⌘V：输入工作台获得焦点时接收图片；“粘贴图片”菜单提供明确入口。无可用图像时禁用或说明。
- 提示词获焦时，纯文本正常粘贴；剪贴板只有图像数据时把图片导入素材区并保留文本光标；同时有独立文本和图像时默认由焦点选择，用户可通过“粘贴图片”显式取图，不双重插入。
- 复制图片的 TIFF/PNG 等多个 representation 是同一个剪贴板 item，不能生成多份附件。多个文件 URL 对应多个独立素材，保持可解释的顺序并允许重排。
- 外部 HTML、URL 文本不能自动下载网络图片；只有真实图像数据/文件提供者可导入。不会后台监听或保存用户剪贴板，只有粘贴动作才读取。

### AssetImporter 拟实现步骤

`入口 → 候选提供者 → 读取/解码 → 临时受管副本 → 元数据/格式校验 → 角色分配 → 原子提交绑定 → 重算计划`。

用 NSItemProvider/NSPasteboard、UTType、ImageIO 完成导入。解码及复制在后台，有单次导入 ID；相同 item 优先读取一份最合适 representation。无文件名的位图以“粘贴图片 01”显示，保存为引擎可解码的受管格式，不拿剪贴板临时路径直接做任务输入。

显示像素尺寸、方向、透明度及将采取的模型预处理。导入层保留原始素材；方向标准化、格式转换要有派生记录，不静默改宽高、裁边或压掉 alpha。引擎不支持透明输入时明确合成背景策略；动画/多页图片首版要求选一帧或提示不支持，不能无说明取首帧。

操作不应承诺任意图片格式都能直接进入模型。UI 文件类型限制与实际 decoder 能力取交集；真实解码校验而非只看扩展名。超大图先读元信息、限制解码内存，再创建缩略图，不阻塞主线程。数值上限由设备/decoder 预算配置，不在未测量前写死“支持任意大小”。

超限选择不默默截断：导入预览显示已选、剩余槽位、超限项，用户选择参与本次请求的图片后提交；未选项不进入请求。不再沿用当前 `prefix(availableSlots)` 自动只取前几张的方式。读取失败按素材显示原因，成功项可保留，失败临时文件清理。

替换/删除支持撤销；这只移除草稿绑定。已提交任务引用资产时保留文件，取消任务并不等于立即删除在途素材；任务/草稿/历史无引用后才回收受管副本。

## 5. 多组图片：界面分组与模型语义分开

首版“多组”优先表达原图组、首尾帧组、参考组，以及参考组内的有序图片。不加入让用户误以为可独立控制的多个 conditioning batch。

对于 FLUX，现有两条路径都将参考图按顺序传入，没有“人物组权重”“风格组权重”的通用字段。可以在管理弹层给图加“主体”“风格”“背景”等**备注**，但必须标明备注只用于组织/提示词编写，不作为引擎硬控制。自由命名组可后续加入；不能暗中拼贴成一张图或擅自生成多组请求。

提示词引用建议用明确“参考图 1”“参考图 2”；UI 提供插入引用文字的按钮，提交时仍是普通文本，不暗示模型支持真正的 asset-ID token。维护稳定 `bindingID` 与当前显示序号。重排后，对手工写下的编号显示“参考顺序已变化，请核对提示词”；不得任意改写用户的自然语言。

若后续模型真正支持嵌套组/时间锚点/每图权重，通过 operation schema 增加 group cardinality、role、ordering、weightRange，再新增受控原生编辑器。今天的 `inputs` 有序数组不是任意多组语义的证据。

## 6. 种子：默认 42，可编辑，可每次随机

将种子移到右侧常用参数底部，默认直接可见，避免核心可复现控制藏进高级 JSON。

```text
种子       [固定 ▾]   [42          ]   [随机一次]
```

- 默认 `固定` + 42；新建草稿复位到 42，恢复旧草稿保留旧值。
- 固定模式输入有效整数；“随机一次”生成一个有效值填入输入框，并保持固定模式，方便之后围绕该 seed 调整。
- 可切换“每次随机”：输入框变成“提交时确定”，保留上次固定值；点击生成时只解析一次，冻结到 RequestSnapshot。界面立即显示“本次种子 123456”，结果详情永久保存实际 seed。
- 同一逻辑请求的传输重试沿用该 seed 与提交身份，不因重连重随机。明确新一轮“重新生成”才产生新的随机值；“复用参数”默认固定为历史实际 seed。
- seed 随机化不代表批量生成；首版无递增序列、区间、多 seed 对比和自动下一张。
- 不发送 `-1`、`random` 或空值给整数 API。当前通用 App 推荐范围 0…2147483647，匹配已记录原生范围；未来使用 descriptor 的范围并以 planner 为准，不能因为 H3 vendor 支持 uint64 就放宽整个 SDK。
- 无效输入就地标注，阻止生成，不四舍五入、不 clamp 后默默提交。当前宽泛 JSON 数字转换不足以代替 UI/服务整数校验。

领域模型：`SeedPolicy.fixed(value)` / `SeedPolicy.randomPerSubmission` 属于 Draft；引擎只接收 `resolvedSeed: Int`。同 seed 只有在模型版本、输入顺序、规格、steps、执行/近似等一致时才有复现意义；不跨模型保证像素相同。

## 7. 高级参数开放清单

原则：字段不仅要能编码，还要被当前执行器消费、有合法范围、有解释，并经过对应路径验证。默认值来自 model+operation+backend；不把模型相关参数做成全局持久偏好。

| 参数 | 首版位置/策略 | 代码依据与限制 |
|---|---|---|
| 输出尺寸/视频帧数与 fps | 常用区；可用预设+受约束自定义 | LTX 原生与兼容 shape 不同；H3 原生 24 fps、4 步、帧数 5+17n；按各自 validate |
| 种子 | 常用区，固定 42，随机一次/每次随机 | SDK/native/Python 均有整数 seed，随机策略由 App 解析 |
| 单图 strength | 仅匹配的操作显示 | FLUX img2img 与 LTX 首帧默认/语义分别验证；多图 edit/H3 不给无效逐图 strength |
| 步数 | 高级区；FLUX 在已验证范围内可编辑 | native 全局校验 1…50 不是全部值已有质量验收；默认 4；LTX 固定 8+3；H3 native Turbo 固定 4 |
| Guidance / CFG | 兼容 FLUX 验证后高级区；其他隐藏 | flux2 adapter 透传，Swift Sampling 有字段但 App.generate 当前未传；native sampling 白名单无 guidance；LTX 明确拒绝；H3 adapter 未消费通用 guidance |
| 音频 | 视频常用区按 required/allowed | H3 兼容 audio_required=true 时锁定“包含音轨”；不能用播放器静音冒充生成不含音轨 |
| 图像预处理 CRF | LTX 首帧高级详情，默认 33，可 0…51 | 兼容 ltx adapter 支持；属于输入预处理，不是输出视频质量开关；native 无同名参数不能透传 |
| 质量/预览 profile | 运行设置，解释会改变什么 | 保留 profile/execution/approximation 三个独立概念；不是越快越好；native 按当前 profile 注册状态 |
| GPU / 混合执行 | 默认 GPU，匹配范围显示已验证/实验 | 新增参考图使 token/shape 变化时重新计划；不能沿用先前无图 ANE 准备状态 |
| BF16 / FP16 | 兼容 FLUX 高级，原生仅显示实际精度 | 兼容 adapter 支持选项；native 不接受通用 precision 参数，禁止放可编辑下拉假装有效 |
| 动态文本长度 | FLUX 高级，默认 true | 兼容 `dynamic_text_length` 与原生 `dynamic_text` 需 adapter 映射，影响 shape/缓存计划 |
| 保留模型/释放内存 | 模型运行偏好或 Models 页面 | Session/资源调度控制；不放每张参考图卡片上 |
| H3 layers/reuse/flow shift/SSD streaming/super | 首版开发者区或暂缓 | 兼容 adapter 有选项不代表原生公共 schema 已开放；改变质量/内部多次尝试的选项必须匹配许可；单输出首版不默认 super 自动搜索 |
| ANE block、attention 实验、路径/环境、tensor dump | 开发诊断，普通创作隐藏 | 引擎工程参数；不能用“更多高级参数”任意覆盖路径、任务输出和安全约束 |
| 负面提示词、mask、LoRA、sampler 选择 | 当前无统一已验证实现则隐藏 | 不从其他工具惯例推断当前支持 |

重复字段只保留一个权威入口：有 `TCInputAsset.strength` 时 UI 不再同时写 `engine_options.flux2.image_strength`。高级 JSON 在兼容开发者模式保留，冲突字段显式报错，不让隐藏 JSON 覆盖可见控件；原生白名单严格映射，未知键阻止提交。

强度特别需要补验收：兼容说明是“允许偏离原图的程度”，当前 native `flux.mm` 按 strength 计算起始 step。首版先用“图像强度（模型参数）”配经验证说明，端点/中点与有效采样步数对齐后再使用“变化幅度”等直觉标签。遥测优先使用 `actual_denoise_steps`，不能把请求 steps 永远当实际采样数。

## 8. 实现拆分与用例

新增/调整领域对象：

- `AssetRecord`：稳定资产 UUID、来源、受管文件、内容身份与元数据。
- `InputBinding`：独立 UUID、assetID、role、order、optional note、strength、active/unused；同资产可多处绑定，不把文件路径当视图身份。
- `InputDraft`：操作、活动绑定、未用绑定、导入状态；切换模式是重新分配，不删除文件。
- `SeedPolicy` 与 `ResolvedRequest`：草稿策略和本次实际 seed 分离。
- `AssetImporter`：文件/粘贴/拖放共用；`InputRoleResolver` 处理分配/数量；`RequestBuilder` 统一翻译兼容 mode 与原生 operation。

建议实现顺序：先 P1 的文件多选+角色槽+seed 编辑与单输出约束；随后同一首版内完成图片粘贴/拖放、排序/撤销、随机 seed；再逐一接入验证后的模型高级控件。输入管理不能等 P5 才做，它是首发图像编辑/首帧视频的基础。

验收用例：

1. 粘贴截图一次只出现一张图；同 item 的 TIFF+PNG 不重复；纯文本不触发图片导入；混合剪贴板按焦点处理。
2. 多选 3 图，完成导入后顺序稳定，拖动/键盘移动顺序能准确传到 adapter；复制的同一图作为两个角色时身份不冲突。
3. 选择 9 张 FLUX 参考图显示超限项，不静默只取前 8；LTX 第二张首帧不悄悄覆盖第一张。
4. 切为不支持图片的模型再切回，原图片、参考顺序、提示词、seed 仍在；运行任务输入不被草稿删除操作清理。
5. H3 首尾帧和参考模式混用在 UI 与服务两层被阻止；实际范围来自当前 backend。
6. 新草稿固定 42；编辑 123 后请求为 123；随机一次填入可见值；每次随机只在新提交时解析；超时重试 seed 不变。
7. 多参考 3 图只得到 1 个主输出；运行中不能从 App 再提交；完成后手动重新生成追加一条历史。
8. native guidance、LTX guidance、H3 每图 strength 无假控件；高级 JSON 冲突有错误；native/兼容 strength 端点、有效步数与说明一致后才开放。
9. 无图数据、损坏图、超大像素、离线文件、导入取消不阻塞主线程；临时数据无引用时回收。
10. VoiceOver 读到“参考图 2，共 3 张”，键盘可以添加、移动、替换、删除；粘贴图片后提示词光标不丢，Cmd+Return 不打断中文输入法组词。

完成此修订的标准是输入流程真实可用、请求语义可解释；不是简单在提示词框旁边增加一个回形针图标。
