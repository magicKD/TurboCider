# Qwen base512：联合分工、固定K分块与多提示词质量

后续已经完成[显式fixed-program与真实失败分支验证](qwen-base512-fixed-program-2026-10-18.md)。
本文保持原656-input库的参数/质量测量，下文“下一步”是当时尚未实现的历史状态。

2026-10-18，M4 Max/64GiB。继续[固定QKV＋FFN联合实验](qwen-base512-joint-qkv-ffn-2026-10-18.md)，
只使用原BF16 base、512²、完整40步，无LoRA/student/Z Image。本轮native源码和
库均未改变，所有窗口使用`qwen-base-joint-qkv-ffn-v1`，656个输入仍匹配封存源码。
不修改App、产品默认、原权重/参考，不停止ComfyUI；所有新窗口均独立输出。

## 重新平衡FFN份额：没有新增速度收益

原FFN-only的5120档不代表QKV联合路线的最优配方。本轮固定QKV512、FFN1024
rows，分别在进程初始化时选5120或6144通道，此后各层/step/request均固定，
不在热路径切换/重编译。stock无observer，每进程seed73→74→73，编辑两侧均
开启相同runtime-prefix复用；首请求分开，后两次完整请求取中位数。

| 任务 | joint FFN5120 | joint FFN6144 | 5120相对耗时 |
| --- | ---: | ---: | ---: |
| 生成 | 31.753837s | 31.727923s | +0.082% |
| 标准单参考重复编辑、prefix hit | 38.809109s | 38.650952s | +0.409% |

差异不足以支持采用5120作速度优化。新6144编辑比上轮39.35s更快，但native没有
改动，这是另一个竞争负载窗口，不当成新的代码加速收益。所有运行仍无独占/
thermal/统计资格，均发现竞争CPU推理负载，`qualification_passed=false`。

四窗口均完成3×40步，实际每请求1280 FFN blocks，生成FFN1312/QKV1248预测，
编辑miss1440/1248、hit1280/1280；无fallback/overflow retry，headroom保持1。
这里5120只有stock证据，不能借用另一分区的timeline授予其实际零热compile资格。

独立审计重读原PNG、请求、源码/manifest/native/memory身份及计数。
相对冻结完整GPU（只作质量参考，不借其时间当新分工分母）：

| 分区 | 生成seed73/74 RGB SSIM | 编辑RGB SSIM |
| --- | ---: | ---: |
| joint5120 | 0.92029 /0.97293 | 0.99743–0.99778 |
| joint6144 | 0.92195 /0.97766 | 0.99690–0.99750 |

已实际查看生成contact sheet：5120也没有恢复原GPU seed73的把手/壶嘴几何。
这与旧FFN-only5120的较高SSIM不同，说明两个近似组件的传播不能由单组件质量
排序直接推导。编辑略高的像素相似度也不等于材质/身份语义资格。不改默认。

原件：`outputs/qwen-base512-joint-share-c{5120,6144}-{gen,edit}-stock-v1-20261018`；
审计：`outputs/qwen-base512-joint-share-{gen,edit}-audit-v1-20261018/audit.json`。

## 固定MIL K分块：定位吞吐/边界成本

当前native `ane_mil.cpp::w8_swiglu_program_impl::projection`将普通Sylvester
配方的descriptor `tile_k`限制到2048。原manifest虽写4096，发射的是2048
K分片；这不是物理ANE kernel tile大小的测量。

只新增两个descriptor，rows1024/H4096/F12288/tile_n512均不变，`tile_k`改为
1024/512。没有修改native、GPU分工、步数或QKV图。每个进程只用一种固定图；
额外MatMul在同一program内，不是逐tile调用Private predict，也不是热重编译。
6144-channel FFN的gate/up/down投影在源MIL中分别合计7/14/28个MatMul节点；
更小的weight工作集可能有利，但新增partial sum、点算子和边界可能抵消收益。

| 请求的tile_k | 源MIL有效K分片 | 生成热中位数 |
| --- | ---: | ---: |
| 4096（原描述符） | 2048 | 31.727923s |
| 1024 | 1024 | 32.164158s |
| 512 | 512 | 37.148823s |

K1024慢1.37%，K512慢17.09%；独立反向默认控制回到31.869487s，并恢复原PNG。
两个K变体与反向控制的完整PNG/计数/source/memory审计通过，原件在
`outputs/qwen-base512-joint-k{1024,512}-audit-v1-20261018/audit.json`。
不能因为更小tiling可能符合历史片上buffer经验就采用。没有给未观察selector的变体授予
实际热compile/load为0的资格，也不把物理INT8/重叠从图类型推导出来。

K512的FFN累计host ANE wait约20.7s，默认约15.8s；pre-FFN边界仍约8s。这支持
将新增暴露成本定位到FFN路径，不证明是哪一个硬件算子/带宽竞争所致。
K1024相对冻结GPU的SSIM为0.94896/0.97876，seed73比默认0.92195更接近，但
只有两个独立seed，不能把这个较慢变体命名为普遍“高质量模式”。本轮速度
筛选仍保留默认有效2048分片，所有变体原件保留。

新descriptor原件：`outputs/qwen-base-runtime-c1024-h4096-f12288-k{1024,512}-v1-20261018.json`。
新consumer审计：`tools/profiling/qwen_base/tiling_audit.py`，强制仅descriptor
tile_k发生变化，实际request除manifest路径相同，保留冷/热完整请求与PNG。

## 五组同库GPU/联合质量检查

已完成精确文字海报、成人肖像、材质排列及标准参考编辑，使用同一新库的
完整GPU控制与选定joint配方，仍40步、512²、无LoRA。每个提示词/路线独立进程，
一条真实请求；不把冷请求时间叫热性能，不通过降低步数获得“提速”。

`quality_collect.py`按冻结任务列表串行运行八个新进程，将原prompt/配方与完成
记录保存在`outputs/qwen-base512-joint-quality-matrix-v1-20261018/recipe.json`。
本轮选择未改算术的joint6144/默认K分片，不采用刚被否决的参数。

全部八个子进程已完成，另补两个干净参考局部改色进程。每一对使用同一库、
同prompt/seed73/geometry/输入文件；editing均为标准1024参考、40步，未降低
参考尺寸，也未使用prefix hit来隐藏首步。新参考直接取本轮materials-GPU的
原始PNG，不替换旧参考或重写其历史结果。

| 配对任务 | joint相对完整GPU RGB SSIM | alpha最大差 |
| --- | ---: | ---: |
| 精确两行文字海报 | 0.998871 | 4/255 |
| 成人肖像、眼镜、绿夹克 | 0.992698 | 1/255 |
| 红方块/钢球/透明玻璃瓶 | 0.997438 | 1/255 |
| 旧茶壶参考改森林绿/matte | 0.997039 | 1/255 |
| 干净产品参考只将方块改蓝 | 0.998122 | 1/255 |

已实际查看五张完整GPU/joint contact sheets和原茶壶参考：

- 海报两侧都可读为`QWEN IMAGE`和`512 TEST`，正好两行，蓝色方块在下方；
  不只是像素分数高，没有把contact sheet标题当生成文字。
- 肖像保留短发、圆眼镜和绿色夹克，局部皮肤/衣料有小差异。无身份参考输入，
  因此不是人脸身份保持测试，也没有推导人口属性识别资格。
- 材质两侧都为三个物体，顺序正确，钢球反射和玻璃透明性可见。
- 旧茶壶参考本身有很强高光、环纹和粗纹理，绿色编辑两侧都延续了这类风格。
  颜色已改变，不能据此授予严格“去高光/哑光材质”或干净陶瓷质感资格。
- 干净参考的红方块两侧都变蓝，钢球、玻璃瓶及大体布局保留，但背景纹理也
  改变；不声称未指定区域逐像素不变。这是普通条件编辑，不是硬mask inpainting。

所有joint请求都完成1280 FFN blocks，生成FFN1312、编辑1440次预测，QKV均1248，
首次真实QKV relative-L2约0.00353；没有fallback、overflow retry或headroom变化。
全部process-tree memory证据完整，系统swap-in/out均0；仍有竞争CPU推理负载，
不授予设备独占/正式性能资格。新joint峰值physical footprint约16.59–18.28GiB。

首请求wall另列，**不是热性能记录**：三个生成GPU约44.19–44.23s、joint约
33.59–34.73s；两个编辑GPU约58.70–58.74s、joint约48.12–49.18s。没有同进程
提示词切换、热compile/load或多seed语义资格，也没有把这些首请求当新稳态分母。

最新独立审计：`outputs/qwen-base512-joint-quality-audit-v2-20261018/audit.json`，
同目录五张命名PNG；旧四对审计v1保留。干净编辑原件为
`outputs/qwen-base512-joint-quality-clean-edit-{gpu,joint}-v1-20261018`。

`quality_audit.py`重读实际prompt/seed/reference/geometry、PNG、原生完整步骤/
FFN/QKV调用和源码/内存身份。它只作可复读的像素/来源检查，不自动授予文字
准确度、人脸身份、材质保持或总体图像质量资格；须实际看图并明确失败案例。

## 工程状态

native：`build/qwen-base-joint-qkv-ffn-v1-20261018/libturbocider.dylib`，SHA256
`1e88bcd9208d677c196148371e97cf8c977c57ffde0ae015ef829532706194cd`；原656-input
archive仍是`outputs/qwen-base-joint-qkv-ffn-native-source-v1-20261018.tar.gz`。
本轮工具/测试/三个descriptor封存于
`outputs/qwen-base-joint-share-quality-tools-v1-20261018.tar.gz`，SHA256
`e319a01009ea8c781a9913dfed924fb9a70d26b1588da9c6db720ebf830f3cee`。
`joint_audit.window`新增显式channels参数（默认6144不变）；新增share/tiling/
quality独立auditor及CPU负控，不修改collector/native运行算术。
最终53项profiler host测试＋3项CPU options测试通过，含K分片/quality recipe
负控，日志`outputs/qwen-base-joint-share-quality-host-tests-v4-20261018.log`。
它们在相应设备窗口全部退出后运行；无自有编译/测试/CPU数组审计与模型窗口
并行，旧日志均保留。此轮17个进程共31次完整请求/1240步，不是31次组件重放。
所有设备/测试/审计进程已退出。磁盘约6GiB free，无下载/删除。

## 尚需解决的固定图合同

本轮全部输入没有触发FFN overflow/headroom变化，但源码中的
`PrivateW8Graph::run()`仍可能在overflow后`build(headroom*4)`。collector是在
事后拒绝这种窗口，并没有从架构上禁止它。此前观察到的零热compile/load不应
扩大成所有输入/错误路径的无重编译承诺。

下一步优先增加显式fixed-program策略：溢出/无效输出不热重建，完整GPU重算并
粘性关闭有问题的ANE owner；配合实际失败分支/负控与新的native构建验证。
这样才把“本次没有重编译”推进为运行时合同。正常成功路径不应额外等待或
重建模型。随后再研究W-stage批量提交、restore消费者边界，不能仅凭减少CB数
假定整请求更快。当前目标保持active，默认开关仍不改变。
