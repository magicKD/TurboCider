# 真实Qwen attention：原配方重放、三候选与empty-mask负控

[工作台](research-campaign-20261015.md) · [源码卡](../sources/repos/mlx-sdpa-dispatch.md) · [完整证据](qwen-attention-evidence-20261016.json) · [预设候选与门槛](../candidates/qwen-attention-20261016.json)

2026-10-16，M4 Max/64GiB，macOS26.6.2，MLX0.32.0。此轮是实际attention component研究，不是产品加速、图像/LoRA质量或G03全部完成。

## 1. 真实边界，且说清replay不是直接内部capture

复用当前public native library与已存在的`DiagnosticBlockObserver`，只编译[可选小客户端](../integrations/capture_qwen_attention.cpp)，未改生产Transformer、已安装App或默认route。648个native source、386个MLX build input逐项hash匹配，loaded runtime ID也匹配。模型/encoder/tokenizer的source generation前后不变。

每个独立seed42/43使用相同公开固定prompt，实际执行完整step0，再到step1 layer0停止；没有VAE decode或image export，但普通component-staged路线可能预加载VAE权重。observer提供的actual arguments与original compiled callable输出作为控制，客户端用同一原始MLX配方重放并额外暴露Q/K/V/O。

两seed、两phase的完整block输出hash与原callable相同，relative L2与max abs均0；prefix K/V逻辑hash与原GPU cache完全相同，原input/modulation/RoPE/cache存储前后hash不变。**这约束了replay，却不证明额外输出graph的每个内部tensor/物理layout与未改变compiled kernel完全相同**，因此capture_kind明确保留该边界。

prefill为`[1,32,1048,128]`，24个text query causal，1024个target query看完整1048 K/V；steady为1024 Q、1048 K/V。每seed两份BF16文件约65.13MiB，128MiB硬cap；两个seed总约130.25MiB原件留ignored输出，不发布权重、tensor或机器路径。prompt是客户端里的公开固定样例，不读取用户会话。保存的是逻辑contiguous副本，replay strides/offset另外登记，benchmark恢复支持的布局并用真实GPU stride probe见证。

## 2. 所有比较与失败保留

21个ABBA比较，每个4 warmup+8正式编号样本：7个synthetic controls、6 discovery、6新seed confirmation、2个原样graph noop。另有1个四case常量V诊断。数据共252个timing samples；不把controls/noop当候选胜利。

三候选都保留全部heads/queries/keys，输出恢复到原`[1,L,4096]` BF16消费者：

- `single-mask`：用一个显式bool block mask代替prefill分段调用；steady则是额外all-true mask消融。
- `contiguous-v`：调用内做必要的V布局转换，计入wall；steady V本已compact，不把无复制机会标为机制收益。
- `head-chunks`：按8 heads调用后join完整输出，不删head或KV。

| phase / candidate | discovery paired中位倍率 | 独立seed confirmation |
| --- | ---: | ---: |
| prefill / single-mask | 0.965× | 0.961× |
| prefill / contiguous-v | 0.959× | 0.955× |
| prefill / head-chunks | 0.918× | 0.962× |
| steady / single-mask | 0.994× | 0.904× |
| steady / contiguous-v | 0.855× | 0.963× |
| steady / head-chunks | 1.001× | 1.019× |

倍率是各匹配pair的baseline/candidate之比的中位数，不是跨run中位数相除。steady head-chunks确认的探索bootstrap区间约`[0.941,1.218]`，不构成稳定胜利。noop prefill约1.010×、steady约0.996×，后者单pair有约0.654离群；小样本CI即使偶然不跨1，也不能独立证明算法更快。没有晋升默认或性能profile。

所有真实run完整O对原replay reference误差0，独立CPU Python-float/`math.fsum`选head/query、完整K与全部输出列的row witness最大relative L2约0.001391，过预设1%局部门槛。FP64是抽样，不是全输出；完整对照是BF16 reference，不等于request质量。

计时是prepared operator host-until-output-ready，包含候选layout与consumer restore/join；不是GPU逐kernel。几何mask setup与逻辑resident bytes单列，不能称单次prefill的免费准备。两arm共享准备allocation，并未测每arm隔离request peak。原始物理counter资格没有移植到MLX queue。

## 3. 全false行：明确的本地合同拒绝，而非泛化硬件故障

tail、非方阵causal、D80、GQA vector/full controls通过；D64与D96的全false首行没有满足本轮明确选择的零输出约定，两个比较拒绝并不给speedup。legacy case名`empty-fused`/`empty-fallback`是固定源码预测，不是实测dispatch标签。

常量V全1、Q/K常量的补充诊断给出：

| D / K | masked首行 | 非空行对1的max error |
| --- | ---: | ---: |
| 64 / 29 | 0.90625 = 29/32 | 0 |
| 64 / 32 | 1 | 0 |
| 96 / 29 | 1 | 0 |
| 96 / 32 | 1 | 0 |

它缩小了“全空行+tail+有限sentinel”的解释方向，不证明某个已定位kernel或硬件有错。不能把未声明的零约定归咎于所有MLX版本，也不能在自己的zero-contract verifier里把这种行悄悄忽略。诊断return0只表示采集完毕，四case的zero gate仍全false。

## 复现与剩余

入口见[GPU实验步骤](../labs/gpu.md)与[可选产品接入](../integrations/README.md)。[独立curator](../scripts/curate_attention_campaign.py)只读明确选择的raw reports，复核capture内容、source snapshots、真实shape/strides、seed/phase关联、完整ABBA、数值拒绝与ratio；不导入候选benchmark或自动运行设备。

此批不覆盖中晚层、多prompt/编辑references/LoRA、极长KV或全部head_dim/GQA/empty语义，也没有candidate全block/request计时。后续可单独探索带完整cast成本的precision island、projection/attention融合或native backend，但需新候选、强基线与独立确认。完整目标的ANE/混合/媒体/内存与知识工程工作继续，不因本轮负结果停止，也不因测试数量宣称完成。
