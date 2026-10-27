# 原生 Qwen 完整请求：DBCache 的速度、RGB 与 alpha 取舍

<!-- knowledge
id: technique-algorithms-case-studies-native-cache-request-study
type: case-study
stage: pipeline, dit, vae
operator: sampling, layout
backend: mlx, metal, cpu-reference
device: gpu, cpu
symptom: stale-cache, precision-loss, measurement
semantics: approximate, contract-only
representation: bf16, fp32
boundary: request, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [FBCache/DBCache机制](../methods/block-caches.md) · [Cache-DiT consumer](cache-consumer-callchain.md) · [MLX生命周期](mlx-teacache-lifecycle.md) · [实验](../../examples/README.md)

2026-10-17，第十一轮。M4 Max/64GiB、macOS26.6.2、原生库内MLX0.32.0，实际执行Qwen-Image-2.1 Base、512²、BF16、20步完整生成与PNG导出。两个提示词、两次完整重复/模式，共18次生成（2次不计分warmup、16次计分），360个完成的采样步。

主要观察：固定纯GPU配方下，conservative约1.21×、balanced约1.38–1.39×、fast约1.45–1.46×完整调用加速；**同样的跳层日程并不保证同样的图像误差**。自行车balanced/fast的RGB SSIM降至0.8172/0.8032，fast的alpha最大差达到49/255。没有调高阈值制造hit，也没有按结果选择画质通过线；不修改产品默认、不授予媒体或生产资格。离线数据保留全部请求、step profile、直方图、像素moments、来源和子进程终态，不打包模型、原始tensor或PNG。

## 匹配请求与计时分母

两个独立持久C ABI engine，各自一次完整Off warmup，然后按如下顺序生成，**每次调用前重新取得GPU admission**：

```text
warmup Off → Off → conservative → balanced → fast
           → fast → balanced → conservative → Off
```

茶壶seed42：`A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.`

自行车seed43：`A red bicycle leaning against a brick wall, morning light, realistic photograph.`

全部请求为`image.generate / gpu / component_staged / dynamic_text=true / allow_approximation=true`，无LoRA、reference、PE、ANE或其它实验变换。`PREFIX_SNAPSHOT=0`、`PROFILE_STEPS=1`；跨请求prefix hit均false，同一请求第1步之后仍正常复用条件prefix KV。两种cache不是同一件事。每个session首次prompt编码未命中，后八次命中，属于warm encoding、staged权重重新准备的范围。

为什么Off也给`allow_approximation=true`？固定源码`memory_policy.hpp::layer_staged_t2i`允许某些设备上的false-Off走逐层权重流式路径；只把preset改成true可能改变权重生命周期。本cohort实际optimization profile为`legacy / qwen21_layer_streaming=false`，未观察到该分叉，但仍保持permission相同。全部执行plan去掉`qwen21_dit_cache`和`algorithm_approximations`后逐字段相等；Off近似列表为空，其余仅`qwen21_decode_dit_cache`，同为BF16 `mlx_cpp_metal / qwen21_compiled_prefix_blocks`。允许近似不等于Off启用了近似。

分别保留三个边界，不混用分母：

- `c_api`：同进程`tc_engine_generate`入口到返回，含完整native生成、PNG导出与wrapper收尾；payload构造、Python结果解析/字符串释放、admission、图像指标在计时外。
- `request_wall`：原生session报告的请求wall，不是进程启动/构造/整个实验耗时。
- `denoise`：原生去噪阶段；每步profile还含建图、eval、有限latent检查，不是Metal timestamp或纯kernel时间。

constructor单列；首次完整warmup分别24.776/22.683秒，不进入速度分母。不是冷开机/冷文件cache或App首屏延迟，无人工清系统/model cache。所有模式具有相同step日志仪器化，未测无日志版本。

## 完整请求结果与prompt敏感性

两次计分Off C API耗时：茶壶22.1177/22.1304秒，自行车22.0919/22.0753秒；各自decoded RGBA与PNG bytes一致，warmup图也与Off一致。下表秒数为两次中位数，速度比是**两次配对Off/candidate的中位数**：正向配Off index1，反向配Off index8，不事后挑较快Off；两次重复不是置信区间。

| prompt / 模式 | C API秒 | 配对完整调用× | 配对denoise× | RGB SSIM | RGB PSNR dB | alpha最大差/255 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 茶壶 conservative | 18.346 | 1.206 | 1.219 | 0.994504 | 45.807 | 2 |
| 茶壶 balanced | 15.941 | 1.388 | 1.415 | 0.956956 | 36.786 | 2 |
| 茶壶 fast | 15.208 | 1.455 | 1.489 | 0.961040 | 36.051 | 2 |
| 自行车 conservative | 18.275 | 1.208 | 1.220 | 0.969158 | 32.255 | 14 |
| 自行车 balanced | 15.973 | 1.383 | 1.411 | 0.817221 | 25.080 | 30 |
| 自行车 fast | 15.147 | 1.458 | 1.494 | 0.803190 | 24.231 | 49 |

全部八个prompt×mode重复对的原PNG和decoded RGBA一致；只说明本cohort重复性，不证明所有硬件/seed确定性。native request-wall配对比与C API接近，仍保留独立原始数值。不把第十轮FFN组件速度与本轮cache速度相乘。

茶壶fast多hit而SSIM比balanced更高，PSNR却更低；自行车fast两项均更差。**preset名称、hit数、SSIM、PSNR不是同一质量顺序**。未记录逐步latent-oracle误差或gate标量，不能据末图反推出每次hit的误差预算；原生只检查finite也不证明误差可控。

## 实际省掉什么：软件consumer而非FLOPs估算

固定Git对象与当前worktree逐文件一致，见[S45](../sources.md#s45)的10个selected-file hashes；实际库全部648个构建输入前后通过`source_match`，CLI/library hash不变，无重编译或产品修改。

`diagnostic_options.hpp::db_cache_options`固定F=8/B=0、warmup8：conservative阈值0.15/最多连续1，balanced0.25/2，fast0.25/4。显式Off优先于环境诊断，preset拒绝冲突覆盖。探针移除继承的相关诊断环境，只设置上述profile/prefix控制。

`Transformer::forward`每步仍跑front8，信号是`astype(hidden-db_input,float32)`，**减法在cast之前**；相对L1为`sum(abs(current_front-prev_front))/(sum(abs(prev_front))+1e-6)`。front anchor每步更新，middle residual只在full target decode后更新；首个完整prefix step不能提供可复用middle。

hit时`hidden += old_middle`并eval，增加`db_cached_steps_`，跳过middle loop；full路径保存新middle subtraction并eval，检查是无primitive/inputs/siblings的leaf。配置、reset、trace、step0/重复或不连续step、条件/geometry变化和异常清对应状态。这是源码选段审计；本轮没有把取消/异常注入当正常生成样本。

两个prompt的两次重复均观察到以下**零起始step indices**：

| 模式 | 实际跳过step | 每请求saved middle blocks | 仅由warmup/final/consecutive约束推得的最大hit |
| --- | --- | ---: | ---: |
| Off | 无 | 0 | 不启用 |
| conservative | 8,10,12,14,16 | 120 | 6，实际5 |
| balanced | 8,9,11,12,14,15,17,18 | 192 | 8 |
| fast | 8,9,10,11,13,14,15,16,18 | 216 | 9 |

前8步、最终step19都full；最多连续hit守卫成立。result的cached_steps、saved_middle_blocks=24×hits与每步dit_cache_skipped一致。gate标量未导出，不虚构conservative第18步的数值或唯一拒绝原因。

证据强于只选中开关：源码明确跳过middle循环/建图，计数在完成step中变化，旧residual已独立eval；pipeline仍每步执行noise→Euler更新→eval/finite latent。但计数来自软件，不是逐shader trace/硬件计数，不授予物理GPU执行条数、频率或连续独占证明。

### 阈值0.15不是“最多15%画质损失”

这个threshold约束front residual变化，不直接约束middle误差。原创标量反例：令`F(x)=x+1`、`M(u)=u²`、B恒等。x从0变2时，front residual始终1，相对变化为0；旧middle residual为`1²-1=0`，新路径重用得到3，而full结果为9，误差6。把新x增大可继续放大误差，即便indicator一直0。它不是实测Qwen失败输入，而是说明**没有额外模型/轨迹假设，仅gate本身不能提供输出误差上界**。本轮实际RGB/alpha测量因此不可由“通过阈值”省略。

## RGB不能掩盖alpha

CPU脚本读取原生512×512 RGBA PNG，**不把alpha改成255、不先合成背景**。RGB/alpha分开保留decoded hash、256-bin绝对误差直方图、整数moments、MAE/RMSE/max/PSNR；RGB另报Gaussian SSIM（11×11、sigma1.5、population covariance、valid502×502 interior、三通道均值）及correlation。完全相同的PSNR用null，不写JSON Infinity。

两套CPU blur实现逐元素核对全部SSIM maps，最大差需≤1e-9；不是两种独立感知模型。标准库verifier只重算直方图/moments与SSIM汇总一致性，不读取全部PNG像素；离线aggregate检查不等于独立完整图像重算。

alpha长尾：茶壶Off253–255、cache最低252、最大差2；自行车Off251–255，conservative最低241/最大差14，balanced最低222/最大差30，fast最低205/最大差49。自行车alpha MAE分别0.06618、0.19986、0.21828（0–255标尺）；小均值不能掩盖少数位置的显著差异。

alpha最大差49不直接等于某背景上的合成图最大差49：straight-alpha合成还取决于同位置RGB、背景和颜色空间。需要以实际RGB/alpha配对做黑/白/彩色背景等新对照，不能从一个alpha极值替代合成误差。

离线测试最初错误假设alpha恒255，在实际253处失败；已去掉不透明度前提，增加显式长尾保留测试。**没有修改生成/度量脚本、放松冻结的设备/数值gate，也没有为通过测试重跑生成**。本轮不是alpha无损结论。

实际查看两张四列raw-RGB接触图：茶壶整体轮廓/构图保留，釉面、桌面纹理变化；自行车balanced/fast的车轮/车架细部、砖墙与地面纹理变化更明显。接触图仅去掉alpha显示RGB，不是背景合成验收，无盲评/语义评分。PNG/接触图只留ignored outputs；Wiki仅分发统计/hash。不下载感知模型，不借两prompt宣布人脸、文字、edit忠实度或视频时序合格。

## 资源、身份与可靠性边界

每session加载库前、每次生成前均取3个driver样本，Device≤5%才进入，末样本到调用≤2秒。保留16个先前未通过窗口，全部出现在前一个自有请求之后，不能归因成16次外部作业。附脱敏pid/pcpu/comm快照，无argv，不杀其它进程、不改系统cache；不是per-PID持续独占或热/频率控制。

粗粒度host准入为free+inactive+speculative≥44GiB、磁盘free≥2GiB，只是可回收页估计，不是模型预留/硬内存上限。native MLX allocator字段原样保留，不含Core ML/OS/filecache，不能据此声明各模式物理峰值或精确cache成本。

三个大BF16文件核对bounded header SHA及前后size/mtime/inode/device，config/tokenizer有完整SHA；**未重新计算三个大权重payload的完整SHA**。CLI/library完整hash、source matching和model header/stat是不同身份层，不等于可复现构建/loaded-image attestation。

两owned进程均在900秒预算内正常退出，完成调用后free engine，无跨FFI异常/回调。保留全部失败admission窗口和不利图像；无native请求失败，也无质量门槛“通过”字段。

## 离线复核与后续

探针和像素阶段都需显式`--run`，native依赖、PNG、NumPy/Pillow仅在选择运行后读取。标准库整理器核对完整顺序、匹配输入/plan、hash/终态、360步trace、配对分母、整数统计与负例；默认tests/retrieval不加载产品库、模型、PNG、NumPy或GPU。

少执行完整blocks在当前Apple GPU上取得完整请求收益，也显示同配置跨prompt的质量风险。conservative约1.21×不是普适安全预设，fast约1.46×不是所有图像/视频的免费加速。下一步先冻结RGB/alpha-composite、语义/文字与时间一致性预算，再扩大prompt/seed、edit/video、cross-request prefix基线与短student日程；cache×稀疏/量化组合须独立新对照。当前两prompt保留为研究样本，不授予采用资格。

### 2026-10-17交付检查快照

新增18项测试、全部算法专题179项通过；标准库CLI独立重新整理与公开JSON逐字节一致，来源46条/173个文件hash结构检查通过。匹配历史context展开本页和evidence返回`recorded-context-only`，64k body预算明确截断741755-byte evidence并省略code，不静默声称已展开全部。
