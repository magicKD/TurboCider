# Runtime ANE：大输出恢复并行候选

2026-09-29，接续[三图长 resident 对照](runtime-ane-qwen-edit-chunks.md)。
本轮继续实现设计稿的 GPU/ANE 重叠与 exposed-overhead 优化，不重新开始
已完成的后端集成。**候选已撤回源码**：Qwen 三图与 Z base 的匹配对照
均未证明整请求收益。保留矩阵布局/有限性回归及隔离构建对照工具；
原版 GPU、冻结 base、runtime 与 LoRA 共图功能不变。

## 为什么选这个位置

重新检查 VPIPE 的 row split、独立 worker、转换池及完整 block 采样实现。
沿用 token-row、固定 runtime 图、base 权重 staging 和 GPU/ANE 并行；
不采用参考代码/设计稿的 adapter staging 合并，因为本任务要求 LoRA
始终 runtime 运行、base 图及 base slots 不含 LoRA。

原版 `8e533d4…` 上单独运行 c1792/K1024/N512 v2 的三参考图诊断：
Qwen BF16 + 原始 Viggle v0.2.1 r256、FP32 低秩、512²、6 步、seed29、
相同三张有序 ref512、resident，一冷六热，auto、profile 开启。
证据：`outputs/runtime-ane/qwen-edit3-c1792-spans-profile/`。

7 请求正常结束，157 条实际 hybrid layer 记录均为 4226 rows、1792
ANE rows，headroom=1；prediction 增量 `32/32/0/0/31/31/31`，没有
失败/回退。全部 hybrid 样本（含冷请求）的 host 窗口中位：

| 窗口 | 中位 ms |
| --- | ---: |
| 权重 staging | 2.819 |
| 输入就绪后的 staging wait | 0.000541 |
| GPU FFN head | 66.027 |
| ANE worker（含转换/输出处理） | 57.422 |
| GPU 完成后的 join | 0.000333 |
| LoRA input readiness，含上游工作 | 68.848 |
| post-join，含拷贝/down-LoRA/拼接 | 9.189 |
| 完整 FFN 窗口 | 75.135 |

累计输出恢复 1.299732 s / 157 predictions，平均 **8.2785 ms**；
它包含在 ANE worker 窗口内，不与上表相加。126 条已计时 hybrid 完整
block 中位 147.232 ms，64 条 GPU probe 中位 171.2275 ms；它们混合
不同层与预热阶段，不是配对纯 kernel 加速比。31 条未计时 plan 仍有
profile 的 layer 日志，故 layer 与完整 block 样本数不相等。

profile 会保留同步 GPU head，因此不能从本轮推导正常异步路径的整请求
收益。没有独立内存采样，系统前后 swap-in/out 未增，不作峰值/驻留验收。
这里优先试输出恢复，不因 staging 总时间非零就盲目增加双缓冲内存。

## 已撤回候选的实现与边界

- `restore_fp16_matrix` 共用原行转换，超过或等于 1,048,576 元素的
  materialized 输出在 Apple 平台分为四个连续行组，同步使用独立 GCD
  转换队列；小输出、scalar oracle、base 未使用 hidden 的只校验路径串行。
- 各任务只写独立目标行和各自的状态槽；全部 join 后才归并有限性。
  所有读访问仍在 Core ML scoped `getBytesWithHandler` 内，不延长
  IOSurface 锁，不借用下一次预测的 scratch，不省略 owning tensor 拷贝。
- 支持 padded/非连续列输入；保持 BF16 ties-to-even、headroom、NaN/Inf
  与恢复后溢出检查。没有改 base 权重、SiLU、LoRA 修正或调度门槛。
- 这与此前撤回的[staging 四组任务](runtime-ane-staging-partition.md)
  不同：这里只并行之前单线程的**输出恢复**，不改变权重/输入 fill 策略。

原版库保存在 `outputs/runtime-ane/output-restore-serial-bin/`，SHA：
`8e533d4daf9edd3e45889ea15463c9bff33be85479d0941cd1debebc670cdf9b`。
候选 native-only 构建成功，SHA：
`7aa60c6d88e0498d871f79c6e94fe56838f28de00ef7ba4070a4117c4d311ec3`。
普通 GPU/冻结图选路不变，候选只在显式 runtime 路线执行；尚不列为
已验收内部优化，也不增加长期实验开关。以下候选成绩只属于7aa构建，
不重标为撤回后重建的运行库。

## 验证与性能门禁

已执行 `make test-runtime-ane test-acceleration-contract test-qwen21`，
exit 0：4 host、9 图/微图集成、76 加速契约（4 host 为同一依赖），
Qwen 38 + 7 + 30 项及三个 native/工作流入口通过。host 新覆盖超过
并行阈值的奇数行、列尾、padded/scattered 输入、只校验、guard、四组
异常与所有源数据不变；原来的完整 FP16 位模式/13 headroom 回归仍在。
微图本身小于并行阈值，不能把它当作真实大输出并行验收。

候选库的完整 `make test` 已结束，exit 0，日志为
`outputs/runtime-ane/output-restore-full-test.log`；缺夹具、专用构建与
GPU opt-in 的既有 skips 不算实际推理覆盖。没有与正式计时并行执行。

对照工具新增 optional `--cli PATH`：要求旁边有运行库，记录所选 CLI
和其旁边库的 SHA，不会将孤立原版构建误标为仓库候选库。原默认不变；
缺库在创建输出前拒绝，host 测试覆盖路径选择、哈希归属与原默认行为。

## 真实三图共图回归：通过

`outputs/runtime-ane/qwen-edit3-output-restore-shared-switch/` 已正常结束。
同一 c1792 v2、固定一 chunk，预热 base → base → 原始训练 A → 合成
半幅 B → base；五个请求的累计 prediction 为 **32/64/96/128/160**，
projection 绑定为 **0/0/227/227/0**，reference tokens=3072，图加载
计时不变。重新读取原始结果核验，无失败、fallback 或 headroom retry。

五张 PNG 与原版 8e 的对应共图测试**全部字节一致**，返回 base 与
先前热 base 一致，A/B 不同；图、库、原始 adapter 和参考身份检查通过。
已查看候选 A 图，主体/构图正常，蓝壶透明材质仍是共同限制。这里不是
第二个训练 LoRA 或普通四十步 base 的质量资格，没有独立内存采样。
真实 shape 超过并行阈值，补齐微图没有触发大输出分支的覆盖缺口。

## Qwen 匹配性能对照：完成，未证明整请求收益

上述构建、全量回归与真实切换全部结束后，启动原版 → 候选 → 候选 →
原版，每 trial 一冷八热，auto/profile-off、各路独立 100 ms 采样，
max-gap 门槛 500 ms。预先固定全部热1–8与较晚热6–8窗口；只比较
同组原版/候选，不把旧 campaign 的 GPU 当作新分母。对照未完成前
不承诺收益，期间不构建/导出/测试，不修改所绑定的源码和工具。

驱动 `outputs/runtime-ane/qwen-edit3-output-restore-abba.py`，证据目录
`outputs/runtime-ane/qwen-edit3-output-restore-abba/`。父报告只有四 trial
全部结束、原始结果与运行库身份通过、前后源码/adapter/图身份不变后
才标记 complete；保留预声明窗口，不从中挑最低值。每子 trial 继续做
竞争推理预检；不是全程独占设备。现四个 trial 和父报告均 complete、
驱动正常退出；36 条原始结果、两构建/工具/adapter/manifest/参考字节
身份、精度与实际调用重新核验通过。

| 预声明窗口 | 每构建热样本 | 原版中位 s | 候选中位 s | 原版/候选 |
| --- | ---: | ---: | ---: | ---: |
| 全部热1–8 | 16 | 12.856890 | 12.935969 | 0.994× |
| 较晚热6–8 | 6 | 12.681290 | 12.650062 | 1.002× |

全部热请求候选慢 **0.615%**，较晚窗口快 **0.246%**；不能凭较晚窗口
宣布成功，也不将小差异全部归因于代码。相同窗口 denoise 中位分别为
原版/候选 11.660077/11.723889 s、11.477230/11.452101 s。

按 trial 顺序的八个热请求墙钟：

```text
serial 0: 12.787581 13.556795 13.657114 12.845426 12.905079 12.628576 12.661494 12.626923
parallel1:12.948048 13.573138 13.566621 12.969371 12.923891 12.639812 12.661194 12.621166
parallel2:12.922704 13.626560 13.629301 12.957777 12.973239 12.620063 12.682315 12.660311
serial 3: 12.899103 13.672916 13.684564 12.868354 12.900121 12.702342 12.701085 12.741469
```

前三者中的 serial0/parallel1，以及反向 serial3，prediction 增量均为
`32/32/0/0/31/31/31/31/31`（含冷请求）；parallel2 为
`32/32/0/0/32/32/32/32/32`。热2/3 是各32层 GPU probe，较晚窗口
每请求31/32个异步 hybrid block。没有失败/回退/重试，headroom=1。

较晚窗口的输出恢复按实际调用加权为 **8.189 → 2.217 ms/call**，
约 3.69×；然而 per-request 中位 async wait **1.715 → 1.567 s**，
post-join **0.324 → 0.480 s**，完整 hybrid FFN **2.128 → 2.117 s**。
async wait 与 GPU 重叠，post-join 可包含未完成的 GPU head：等待位置
变化与小幅 FFN 差异支持“输出恢复原本多数已被隐藏”的解释，不支持
“down-LoRA 本身变慢”或“输出恢复局部收益等于整请求收益”。各窗口中位
也不可相加当作整请求差额。

四份独立内存 hash-chain、报告/工具 SHA 和 correlation 重验通过：
**4,841 样本**，最大间隙 **110.016583 ms**；均无新增系统 swap-in/out
或 compression。峰值 footprint 原版 **22.661–22.673 GB**，候选
**22.655–22.673 GB**（十进制，完整进程窗口）；不是完整 driver/ANE
内存或物理驻留资格。

已查看反向原版/候选的热8 PNG，二者均实际执行 hybrid；主体、构图与
贴纸细节肉眼接近，透明蓝壶仍共同失败。此处 auto 启停可不同，不要求
逐像素相等；前述固定分区共图测试的逐字节相等证据独立保留。

结论：**不将本候选推荐为 Qwen 整请求加速**。随后补齐下节 Z-Image
base 受控对照，两者都无整请求收益后撤回，不更新默认路由或已验证最快推荐。

## Z-Image base 补充资格：完成，同样未获益

Qwen campaign 完全结束并独立复核后，启动
`outputs/runtime-ane/z-output-restore-base-screen.py`，证据目录
`outputs/runtime-ane/z-output-restore-base-matched/`。512²、8 步、seed42、
原狐狸提示词、BF16 resident，无 LoRA、auto/profile-off。

顺序候选 GPU → 原版 runtime → 候选 runtime → 候选 frozen → 反向。
八个独立 trial、每个一冷两热，所有路线100 ms独立采样，预设500 ms
max-gap；每配置池化四个热样本。候选库内 GPU/runtime/frozen 有匹配
分母，原版 runtime 仅用于同组 before/after。不与历史 GPU 拼接，不在
这轮运行中构建/测试/导出。八 trial、24 请求及父报告均正常完成，运行
前后源码/两构建/图/工具身份一致；原始请求、计时、backend 和实际调用
独立复核通过。

| 构建/路线 | 四个热请求 s | 池化中位 s |
| --- | --- | ---: |
| 候选7aa GPU | 6.994160 / 6.992662 / 6.996000 / 6.993263 | 6.993711 |
| 原版8e runtime | 6.334278 / 6.298646 / 6.311978 / 6.309252 | 6.310615 |
| 候选7aa runtime | 6.413148 / 6.351231 / 6.404055 / 6.368242 | 6.386149 |
| 候选7aa frozen | 5.338762 / 5.337394 / 5.348751 / 5.336400 | 5.338078 |

候选 runtime 比原版**慢1.197%**。候选库内部 GPU/runtime 为1.095×，
GPU/frozen 为1.310×；后者仍最快。该runtime/GPU优势来自已经存在的
后端，不是本候选的增益；不使用候选GPU分母将原版runtime重标为同库。

runtime 四个 trial 的最终计数均为688 predictions、685 hybrid、558
async block；冷请求发生3次正常 headroom 探测，最终scale=64，所有热
请求均无新增重试，无错误或fallback。相同调度计数有助于比较，但不代表
所有其他系统因素完全相同。

每热请求中位输出恢复从0.148498降至0.084750 s，hybrid FFN从3.028550
降至2.967553 s，整请求仍更慢。不能将剩余差额全部归因于CPU并发；
当前证据只支持局部窗口减少没有转化成总体收益。

八份独立内存流、报告/工具 SHA 和 correlation 重新验证：**1,751样本**，
最大间隙**110.015667 ms**，全组无新增系统swap-in/out或compression。
峰值footprint（十进制GB）GPU22.946–22.947、原版runtime23.040–23.041、
候选runtime约23.042、frozen约23.071；这是完整进程窗口，不是ANE或
driver独占内存，也没有证明所有内存压力情形安全。

已查看反向原版runtime、候选runtime、GPU、frozen四张末次热图：狐狸
主体、姿态与雪景接近，runtime与GPU/冻结图在眼部、毛发、尾巴等细节
有差异，无明显新增崩坏。一个prompt/seed，不构成普遍质量资格。

## 撤回与保留

四组GCD并行恢复已从源码移除，不增加新的实验开关。原来的串行行转换
仍执行相同有限性和headroom检查；只将二维padded/scattered遍历保留为
`restore_fp16_matrix`，使Core ML scoped读取与host回归共用同一逻辑。
移除了仅候选需要的dispatch/array头文件；大矩阵边界、只校验和异常测试
保留，明确它们不再测试一个并行策略。原生计算精度、SiLU/LoRA顺序、
图复用、GPU/ANE分区和已有异步head没有变化。

撤回后的4项host检查已通过。native-only 重建、专项及完整验证串行结束，
均 exit 0；保留库 SHA 为
`1080afeb54e0d20e4f608c94d65c1e33f51b7236c6eb0b7d3f1ba6b00aa420c2`。
专项包含4项host、9项Core ML/MLX微图、76项加速契约（含同一4项host），
Qwen 38+7+30项及三个native/工作流入口。完整 `make test` 的缺夹具、
专用构建与GPU opt-in skips 不算新增覆盖。日志入口：
`outputs/runtime-ane/output-restore-retained-build.log`、
`outputs/runtime-ane/output-restore-retained-tests.log`、
`outputs/runtime-ane/output-restore-retained-full-test.log`。
这里记录的是撤回后的实际执行，不复用候选库的测试成绩；尚未在1080
库重新运行整模型性能或视觉验收，不能把7aa/8e的成绩重标为当前库。
原实验driver的源码/库哈希代表运行时版本，在campaign正常完成并重验
以后才更改源码；不回写历史身份。未stage/commit，模型/图/PNG不删除，
249项既有暂存移除及本地对应文件保留。

仍需两模型 base/LoRA 更广整模型回归及质量/内存资格；
不将本轮 host 或 profile 进展当作完整目标完成。
