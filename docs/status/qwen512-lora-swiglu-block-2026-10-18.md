# Qwen512：原生base＋成对LoRA-B＋typed SwiGLU，数值通过、性能未定

2026-10-18，Asia/Singapore，M4 Max /64GiB。仅 Qwen Image 2.1 的512²文生图、
单ref512图生图；不涉及Z Image，不改App或默认值。持续优化目标保持active。

## 当前结论

新增研究kernel保留原生gate/up base GEMM，把两个独立在线LoRA-B修正与typed
SwiGLU合成一个consumer。最终24个Metal配置、120个阶段输出按完整字节一致，
54个非法合同拒绝；包含正负零及实际安装库BF16 sigmoid边界见证。

真实r128 step1/layer0重放中，完整hidden、完整block的首/末数组复核通过；最终
v4每个预热/计时调用还检查并记录完整BF16输出SHA，而非只比较F32数值。

**不能宣布提速或采用。**较早v3四窗口有1.50%–1.79%的增量信号，但加强逐次
字节验证后的v4四窗口出现+1.01%、−1.02%、+11.23%、−9.20%的强基线相对减少，
普通GPU重放控制本身也有明显漂移。全部窗口检出外部推理CPU负载，保留
`qualification_passed=false`。不挑v3或有利v4窗口替代完整结果。

本轮没有修改/重建native产品库，没有新增生产flag，也没有候选整图/VAE/PNG。
这是一个数值已收敛、性能需用连续完整请求继续判断的FFN候选，不是整图速度发布。

## 为什么不是重跑旧四matmul原型

[旧corrected hidden原型](local512-qwen-fusion-precision-2026-10-09.md)同时融合两次
base、两次B和激活，采用独立F32 logistic，组件误差约0.35%，收益有限。其旧5%
component预算和历史结果保持不变，没有被本轮的新标准或新数据回填。

本轮只改变一个边界：

```text
原生BF16 X @ Wgate_upᵀ → 原base[gate|up]
原在线A_gate / A_up → 两个独立F32 ranks
两路原B/scale修正 → BF16 gate、up → typed sigmoid / SiLU / multiply
完整corrected hidden → 原完整down投影及down-LoRA → residual
```

base GEMM不改、原物理base pitch为2F，不复制或预merge权重，不把不同A当成同A，
不预先计算并跨调用复用rank，不省略down-LoRA，也不裁token/层/step。
新dispatch只发布完整hidden，不发布两份corrected gate/up；是否真的节约足够
时间仍由完整消费者决定，不能按逻辑字节数推算DRAM或整请求倍率。

参考已安装MLX的`unary_ops.h`和`bf16_math.h`，`Sigmoid<T>`中的
`exp(abs(T))`是T-valued调用；不是从头到尾F32 logistic最后才舍入。后续sigmoid、
SiLU以及与up相乘的typed边界也分别保留。实际库见证确认native BF16 sigmoid
不总等于F32 sigmoid再cast BF16；20/24组各阶段检查和真实consumer进一步核对。
不只凭相同版本字符串或头文件宣称wheel等价，也不把新旧组件误差相减当作
旧四matmul误差唯一原因的证明。

首次尝试直接include安装目录的unary header，被custom-kernel虚拟头环境拒绝；
v1失败日志保留。随后只采用同样的短typed表达式，再用实际运行结果验算；没有
编辑MLX安装文件。v2/v3是20配置/100阶段/45拒绝；v4增加负零配置和逐次原始
字节检查，为24配置/120阶段/54拒绝。重复运行不重复累计unique case数。

## 真实完整block与七个固定臂

原BF16组件、Viggle v0.2.1 r128 strength1/inference-time、joint BF16 A/B和F32
ranks，固定adapter SHA256：
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。
六步/seed73，蓝茶壶生成及单ref512改红；实际decode M1024/H4096/F12288，
prefix分别24/1059。无ANE、跨请求snapshot或时间复用。

observer只借用真实step1/layer0的九个参数、原GPU callable、原完整输出与权重；
每进程全局227个adapter terms已绑定，不称为一个block执行227项。七臂固定为：

- 原GPU callable、独立普通GPU replay。
- 已有original-buffer direct-QKV强控制。
- 仅成对B/SwiGLU，BM16与BM32。
- direct-QKV＋成对B/SwiGLU，BM16与BM32。

primary事前固定BM16组合，强分母固定同窗口direct-QKV。1%仅是该零新增权重、
严格等价候选的局部信号标记，不修改旧近似ANE实验的3%入口，也不是生产资格。
BM32在部分窗口较好也不事后重标primary。

计时包含每次在线A、Q/K/V、norm/RoPE、attention/prefix concat、out projection、
完整FFN/down-LoRA与residual。原base与factor均只读复用，新增持久weight bytes0。
source/hash和first-replay/JIT/诊断输出在计时外单列，不叫免费准备或零临时内存。

A→扰动B→A检查要求B确实改变，所有候选在B及恢复A上与匹配参考一致，借用
输入/输出和22项layer0 base/factor来源前后不变。全部计时前输出须尚未available，
不能返回已完成旧tensor。单输出hidden taps只作诊断，不替代完整block分母。

## 先保留v3，再加强bitwise验收

v3每次用完整F32比较检查changed=0；CPU另复核保存的首/末BF16字节和hidden，
60次比较通过。但F32的`==`会把+0和−0视为相同，因此不能把这类live数值检查
自动写成所有调用逐位验证。

v3四窗口、binary、summary及source归档全部保留，未回填。v4对每次first/
warm/measured输出记录完整原dtype字节SHA，要求与对应完整GPU参考相同；扰动
和恢复也检查字节，hidden另核对完整SHA。native和host均有错误hash/负零拒绝
控制。检查位于timer结束后，对所有臂一致，但不能因此保证不会影响下一样本的
cache/时钟/主机状态；没有独立消融，不把v3→v4变化唯一归因于hash或外部GPU竞争。

## 完整block计时观察，非整图

每独立窗口七臂×3 warm＋21 measured，循环visit位置轮转；reverse窗口反向visit，
每臂覆盖每个位置三次。不在测量同时运行自己的build、测试、模型或CPU审计。
下表单位ms，中位数；右列为primary对同窗口direct-QKV的减少，负值为变慢。

| 版本/任务/顺序 | 原GPU | direct-QKV | BM16组合 primary | 相对强控制 |
| --- | ---: | ---: | ---: | ---: |
| v3 生成forward | 35.7310 | 34.7881 | 34.1663 | +1.78737% |
| v3 编辑forward | 37.0990 | 36.1720 | 35.5264 | +1.78464% |
| v3 生成reverse | 35.6963 | 34.7535 | 34.2338 | +1.49518% |
| v3 编辑reverse | 37.1107 | 36.1971 | 35.6472 | +1.51912% |
| v4 生成forward | 47.9941 | 46.3625 | 45.8932 | +1.01213% |
| v4 生成reverse | 47.0804 | 45.8134 | 46.2787 | −1.01562% |
| v4 编辑forward | 47.2756 | 48.7525 | 43.2799 | +11.22531% |
| v4 编辑reverse | 47.0873 | 44.7070 | 48.8188 | −9.19720% |

v4普通GPU replay相对原GPU中位数变化依次为−1.032%、−.547%、+3.224%、−5.225%。
连相同数学控制也有漂移，不能把11.2%的有利列当候选稳定优势；也不能因−9.2%
就永久否定该数据流。CPU负载采样不是GPU利用率、每PID归因或物理shader追踪，
两批之间还改变了验证协议，不能作单因素因果比较。

每批四窗口共588 measured＋84 warm＝672个完整block重放；另76个完整preflight
重放和12个partial hidden taps。两批合计1344 warm/measured完整重放，另152个
preflight；不是独立prompt数、完整图片请求或硬件kernel数。真实模型的首个
prefill step及借用层0参考另在正常轨迹内，不从这些计数反推全模型覆盖。

## 独立复核、来源与内存

`qwen_lora_swiglu_block_audit.py`的v4复核complete：当前tree、链接库及封存归档
650项native/build输入一致；当前probe binary/source、MLX typed头快照匹配；重读
保存的首/末完整block和完整hidden，60次数组比较逐位一致；跨窗口同任务的九个
输入、参考、扰动参考及hidden一致，生成/编辑的22项base/factor来源一致。
还重算全部sample、visit顺序、中位数、强分母及memory digest。逐次live SHA记录
被验证，但不是保存/重放每个样本完整数组，更不是完整生成质量或独占设备认证。

```text
未改动的产品库：build/qwen512-qkv-prefix-v6-20261018/libturbocider.dylib
SHA256 48fc47b2586c2682675ee702179b5756a4cffeef9f46cbad250a9d7b82cab936
runtime tc-runtime-build-v1-7a9f94bb76407117f9546b01af0382338eb99250efa3770ad6502d847e229e74

最终probe：build/qwen21-lora-swiglu-block-v4-20261018/qwen21-lora-swiglu-probe
SHA256 1bd56da0f35ca7d036bf515bd30594e9d49288b2fecb2070c706326b4f7ffb80

outputs/qwen512-lora-swiglu-probe-v4-source-20261018.tar.gz
SHA256 72e16b4bd7dfc7e788c3309cdbd1409c097d620450baef239ca1fa4059590187
```

v3 probe SHA为`9576e60c668a2947b5f65483cdb4198a2f433363ce364348bdacd3165178c8d3`；
独立v3 source归档SHA为
`7f0988dd8eac696cb6342bb728cee40f94be1167b12439e935d2d730206a5484`。
source archive包括研究代码及读取过的两个MLX头；库完整native源码另由既有
`outputs/qwen512-qkv-prefix-native-source-20261018.tar.gz`绑定。以后源码变化应
恢复归档到新目录，不把当前源码冒称旧二进制来源。

八窗口memory采样均complete，system swap-in/out0，compression0。v3进程树峰值
physical footprint约17.44–18.05GB，v4约17.42–18.05GB，涵盖模型load、所有臂共存、
验证、退出，不是每臂峰值或OS/driver物理归因。全部八个负载检查均未通过，未
停止用户ComfyUI；没有能耗、DRAM、ANE placement或GPU独占证据。

最终Metal24/120/54、host screen5及raw audit2项通过；知识库检查只证明文档。
未重跑/冒称本轮又重跑整套产品C API或整图质量矩阵；产品库未改变。

原件在`outputs/qwen512-lora-swiglu-block-{gen,edit}-{forward,reverse}[-v4]-20261018/`，
两批audit分别为`qwen512-lora-swiglu-block-audit-20261018.json`与
`qwen512-lora-swiglu-block-v4-audit-20261018.json`。失败/阶段Metal日志、v1–v4
probe及所有模型/历史结果均保留，没有下载或清理用户文件，结束磁盘约10GiB。

## 下一步

数值候选保留，尚不设生产开关、不更改默认。下一轮应在显式研究入口下测连续
完整请求，包含六/五步、文生图/编辑及已有direct＋prefix强路径，核对全部层、
policy/seed恢复、final latent/RGBA和真实request wall。不能把局部约.6ms差值
乘层数/步数当作整图收益，也不把当前不稳定测量改名为稳定加速。
