# Qwen base：多固定行数W8的真实完整FFN对照

2026-10-19，M4 Max/64GiB。目标仍为真实512²生成/编辑；本页是接入生成流程
之前的FFN gate，不是以单组件替代完整40步图片。App/default和Session未改。

## 实际执行范围

原`qwen_image_2.1_bf16.safetensors`前两层原BF16 gate_up/out，H4096/F12288，
ANE/GPU各分6144 intermediate channels，无LoRA/权重merge/转换模型。归一化
input_a来自完整base40的首decode捕获；input_b来自受控BF16扰动后重跑相同
attention/norm，不是第二个自然seed。生成/单full-ref1024编辑分别使用自己的
冻结捕获。超过1024行的输入由这些数据拼接/切片，仅测试真实shape的FFN，
不能称为自然长prefill或新增图片尺寸质量。

四个arm在同一进程轮换，正/反向独立重跑：

| arm | 工作 |
| --- | --- |
| GPU | 原Weights的完整compiled gate/up→BF16 SiLU×up→down |
| single1024 | 固定1024行ANE＋原GPU补充分支＋FP32 join，按1024补齐 |
| single2048 | 相同算术，固定2048行、按2048补齐 |
| family | 一个model的512/1024/1536/2048四procedure，按512补齐和分块 |

每个FFN计时包含实际padding/求值、原权重GPU staging、ANE第一提交、GPU
complement、ANE finish/FP32 restore、最终BF16 join。没有用单独ANE matmul或
裸kernel时间作完整消费者。全部function、source layer和GPU shape先warm；
每shape再排除两轮warmup。profile timeline与stock计时分开，不混用分母。

## 观察到的耗时，不作为正式加速资格

生成/编辑×正向/反向四个stock窗，每臂每shape16 measured，单位ms。下表是
这四窗的热中位数范围，fixed列取该shape更快的原固定bucket：

| 有效行数 | GPU完整FFN | 最佳fixed ANE | family |
| --- | ---: | ---: | ---: |
| 1024 | 21.05–21.07 | 13.10–13.18（R1024） | 13.10–13.15 |
| 1536 | 31.39–31.41 | 21.90–21.97（R2048） | 18.39–18.46 |
| 2048 | 41.72–41.73 | 23.66–23.71（R2048） | 23.65–23.68 |
| 2560 | 51.73–51.75 | 49.51–49.65（R2048） | 35.42–35.46 |

较长/尾块fixture的信号在反序保留；1024/2048完整bucket没有新的稳健增益。
原因与被执行的行数一致：1536不再执行2048 padding；2560由2048＋512服务，
而R2048控制执行4096行。真实GPU分支仍处理全部有效行，所有token保持，未
借减步/省算术得到收益。不能直接把局部差值乘成整模型/图片倍率。

**六窗均捕获到外部ComfyUI Python CPU负载，performance load gate=false。**
这些是有限的同窗观察，不宣布正式加速比，不借受污染GPU分母改默认或commit。
CPU负载记录也不单独证明外部GPU任务或GPU/ANE物理重叠。

## 数值、来源和真正的compile/load

V7/665-input native原件及匹配C++ helper。四stock窗各256 measured完整FFN，
两timeline窗各128，共1280 measured消费者，另有明确warm/check/恢复调用。
每个混合arm一个持续model，family内部不按function另建model。两个诊断窗
实际看到三个不同model，warm、check及timed阶段compile/load均0；三个是
控制实验同时保留的三臂，不是family需要三个模型。

独立审计`outputs/ane-w8-functions-real-audit-v2-20261019.json`通过：

- 六成功窗、270完整typed张量；原model/encoder/VAE全SHA、原冻结捕获、两层
  576MiB原FFN权重payload、native/CPP pin、完整source archive均复核。
- 保存的全部GPU/ANE merged输出独立逐元素复算质量，相对GPU最大relative L2
  约0.730%；family对single2048所有保存case新增误差为0，无放宽先定预算。
- 所有1024行source case的FP32 GPU head＋ANE tail→BF16 join，全部元素bits
  独立验证；576个原物理pitch/完整K6144的F64 GPU down坐标见证通过。
- 对所有长row输入逐位验证切片/拼接来源；改变X/原W的控制返回不同完整输出。
  source/shape恢复由native完整hash确认。所有输入/W原件保持不变。
- process-tree memory原始hash-chain、所有timed真实预测数与actor model身份
  独立复核；近似质量、host时间和物理engine资格严格分开。

首次pilot的stock scope编号未递增，已单独保留、不纳入最终六窗。首个timeline
因setup GPU命令仍被标在校准scope而被严格拒绝；修正scope后新窗重采，未放宽
“唯一真实Metal校准”或compile/load检查。旧工具归档绑定四stock窗，修正后的
工具绑定两timeline窗，不能把当前源码假装成所有历史窗使用的字节。

## 后续接入条件

当前生产HybridFfn仍按graph最大bucket pad，不能只打开family constructor就
宣称真实Session已省掉padding。下一步需要最小row alignment接口、固定family
准入/选择、实际function列表和capability计数、prefix arithmetic recipe隔离，
再运行完整base40生成/编辑及尺寸/参考数矩阵。512主decode1024行当前无新收益，
要特别防止为了变长灵活性反而拖慢主路径。没有合格净收益则保留原固定图/GPU。

后续V9已完成显式Session接入、31张完整40步生成/编辑及独立媒体审计，见
[完整请求报告](qwen-base512-function-session-2026-10-19.md)。原fixed仍保留，
没有把局部FFN差值外推为512热出图提速。

工程：`w8_functions_real_probe.mm`、`w8_functions_real_collect.py`、
`w8_functions_real_audit.py`与`build_ane_w8_functions_real_probe.sh`。两项schedule
负控和三项独立reader/rounder/schema测试通过，连同既有检查共39项CPU测试通过；
旧默认MIL/原权重与App设置未改。最终审计同时核对实际observer二进制与冻结
capture的原model snapshot/任务，不能只检查CPP标签或数组文件名。
所有大型数组、NN缓存和trace留在已忽略的outputs/build，无新增权重下载或删除。
