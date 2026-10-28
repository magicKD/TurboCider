# ncnn的CPU/Vulkan交接与缓存：空输入、完成门和编译身份

<!-- knowledge
id: technique-gpu-ane-case-studies-ncnn-vulkan-handoff-and-cache
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [INT8/Apple选路](ncnn-vulkan-int8-and-apple-policy.md) · [来源](../sources.md#ncnn-vulkan-policy)

2026-10-17，固定`Tencent/ncnn@c15904bb8918187ec100f1ebe708942af83b84be`。阅读的是实际Net/command/cache调用链；本机没有运行ncnn Net、原始`test_net`或Vulkan设备。这条路线的异构是**CPU+GPU**，不是现成GPU+ANE实现；交接/缓存经验可供其它异构运行时对照。

## 1. fallback不是“GPU出错就无成本换CPU”

[NetPrivate::forward_layer](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/net.cpp#L200)维护host `Mat`与device `VkMat`两份blob表。若两边都空，先递归计算producer；随后按layer的`support_vulkan`决定所需表示：

| producer处理后的状态 | 下游要GPU | 下游要CPU |
| --- | --- | --- |
| 两边都空 | 保持空，不上传 | 保持空，不下载 |
| 只有host有效 | record upload | 直接使用host |
| 只有GPU有效 | 直接使用device | record download，submit/wait，再CPU |
| 两份都在 | 不新增传输 | 不新增传输 |

这是存在性逻辑，不证明两份数据永远新鲜一致；也不把表中最后一行升级成任意原地更新的coherence协议。`lightmode`在交接后释放源blob引用，实际command仍保留必要staging/postcopy owner；释放表项不等于物理buffer已经drain。

读取路径里，layer执行返回非0会传播；没有读到“任意GPU runtime错误自动重试CPU”的保证。有效device不存在时的整网CPU回退、layer不支持Vulkan而预选CPU、运行期错误传播是三类行为。

## 2. #6870：不存在的可选输入不能拿来传输

[PR #6870](https://github.com/Tencent/ncnn/pull/6870)于2026-10-10合入；本轮读取完整173行两段patch及固定源码。修复从“destination空”改成“destination空且source非空”：两边都空的optional input应保留empty，而不是创建一个数值为0的tensor或调用download处理不存在的源。

PR作者描述了first-step empty K/V cache导致的混合推理崩溃，并记录Windows、其它Vulkan设备与软件driver测试；这些是**作者报告**，不是本机Apple GPU重现。上游新增Net/Extractor回归覆盖两方向×lightmode开关×optional存在/缺省，共8组合；`get_gpu_count()==0`直接return0。因而一个没有Vulkan设备的`test_net`成功，不能单凭exit0宣称覆盖这8项。

本轮只用独立CPU合同枚举producer处理后的八种存在性/目的侧状态，并将两边为空与合法零tensor区分。没有执行原始Net回归或复现其segfault。浏览器缓存页曾显示旧的open状态；状态与合入日期依据保存的官方API JSON，并用固定源码确认条件已存在，不从旧网页推断未修复。

## 3. UMA不会消灭staging、packing和host完成门

[record_upload](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/command.cpp#L364)会处理dtype、把大于pack4的CPU布局转换，再创建staging、逐batch memcpy并flush；后续GPU packing/cast依device类型和选项选择。不是看到Apple统一内存就自动省掉这些步骤。

[record_download](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/command.cpp#L445)按consumer packing要求与allocator可映射/可缓存属性选择staging，记录GPU转换和host-read barrier，再挂起host postcopy。源注释给了另一vendor设备的下载/上传带宽数字，本页不把它移植为Apple测量。

[submit_and_wait](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/command.cpp#L1978)先queue submit、wait fence，再执行invalidate和逐batch memcpy；某些精度路径还需post-cast。CPU fallback必须越过这个完成门，record_download返回不代表host可以消费结果。

Net另有基于`rough_score`的pending-dispatch提交阈值。源码累计的是dispatch workgroup乘积，不是实测毫秒、bytes或功耗预算；一次自动submit/wait会切断更长的异步区间。不能把“使用command buffer”写成设备自然与CPU/ANE重叠。

固定[macOS CI workflow](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/.github/workflows/macos.yml)含Vulkan构建，但所读测试命令指定SwiftShader ICD。这说明CI配置也需辨认物理driver；本轮没查询这些CI job是否成功，更不拿其存在当Apple Metal执行证明。

## 4. 两层cache分别省掉什么

[pipelinecache实现](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/pipelinecache.cpp)不是一个笼统的“模型cache”：

| 层 | 关键身份 | 复用边界 |
| --- | --- | --- |
| SPIR-V entry | shader index、15个option bits、shader source hash | 复用编译出的模块数据；不同设备选项与源码不能混用 |
| process内pipeline artifact | shader/option、local x/y/z、subgroup、specialization双hash | 返回shader module、descriptor/pipeline layout及pipeline等对象 |
| persisted file/driver cache | magic/version、ncnn版本、endian/pointer size、vendor/device/API/driver、名字hash、UUID等 | load前验证身份、长度、payload hash与SPIR-V/source一致性，再创建/合并Vulkan cache |

local-memory开关在option bit11，cooperative-matrix在bit12，因此变更Apple偏好会形成不同shader cache身份，不应当无关调度小开关。specialization的shape与activation等常量同样参与pipeline身份；warm/cold benchmark必须记录cache状态。

本轮实际编译并执行**未改的**`encode_spirv_cache_opt_bits`，完整枚举15个boolean的32,768种组合，每个实际编码均与独立field-mask一致且在该子空间不重复。也执行原始`get_masked_option`的4,352组mask/input-pattern，并核对35个假设备策略点。这个结果只验证所选纯CPU逻辑；某些组合本身不代表合法device配置，不授予32,768份shader编译或cache load/hit资格。

FNV/Murmur是非加密hash；双hash和header校验不是抵抗恶意输入的密码学证明，也不是全空间无碰撞证明。kernel/pipeline可与具体权重值无关，复用编译物不等于该checkpoint、量化scale、LoRA或request已获数值资格。不能把一次cache hit当跳过质量检查的理由。

## 5. 本轮可重放边界

probe将原始Option头文件/constructor、三份原始CPU函数及一个INT8 packing block放进独立host观察器。设备properties、两个CPU helper和byte-Mat allocation是显式stand-in；ASan/UBSan只约束该观察器的host行为，不测试原始VkMat allocator、GPU同步、MoltenVK或shader的内存安全。

`otool -L`实际读到libc++、ASan runtime与libSystem，无Vulkan/MoltenVK/Metal链接。未安装SDK/依赖、改系统cache/环境、运行设备或修改默认后端。完整native输出、原始打包bytes、数学复算和编译/失败receipt留ignored outputs；[来源18文件范围](../sources.md#ncnn-vulkan-policy)与27项离线测试无需这些outputs也能检查发布数据合同。

下一步若执行完整ncnn，应固定Vulkan/MoltenVK/GLSL编译器版本，先过GPU idle guard，再运行原始8组合回归、有效/空输入、tail形状和正确的quantized图。性能要匹配同checkpoint/输入/精度，并报告staging、等待、首编译/热cache和完整consumer；本页没有速度倍率、ANE支持或GPU+ANE并行资格。
