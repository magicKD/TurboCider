# Qwen512：独立CPU tokenizer复用，模型与采样算术不变

2026-10-18，Asia/Singapore，M4 Max/64GiB。只优化 Qwen Image 2.1 文生图/图生图，
不动 Z Image、已安装 App 或默认设置；完整优化目标继续。

## 结果与范围

新增默认关闭的 `TURBOCIDER_QWEN21_TOKENIZER_REUSE=1`，复用一个经过内容与
fd/named-path generation校验的CPU processor owner。它不启用encoder融合、
不保留encoder模型权重、不改GPU图/精度、tokenizer算法、prompt模板、采样节点
或模型张量。不是GPU/ANE并行优化，也不把CPU组件倍率说成整图倍率。

32张首版完整图（24张测速+8张原始张量诊断）表明：每个匹配任务的开启/关闭
结果RGBA逐像素一致，六步与现有五步档位各自的原始条件/latent/pixels也exact。
持续竞争负载检查失败，`qualification_passed=false`。这轮保留显式入口，不改App。

## 为什么选CPU边界

按 `techniques` 的方法→实现/反例→实际shape顺序复查：普通DiT的FFN/rank融合、
跨请求编译图已有小收益或负结果，不重复把局部数字外推；同参考换指令的visual
features及VAE latent原本就有缓存，无需重新实现。

实际缺口是普通BF16路径在每个新提示词重新解析约11MiB tokenizer、构造BPE
rank/decoder表，而现有 `VerifiedEncoderTokenizer` 复用只绑定fused-prefill开关。
将CPU owner与GPU融合解耦，避免为省host工作同时改变GPU算术。使用方法见
`techniques/models/encoders.md`，既有实现/反例见
`qwen21-gpu-encoder-prefill-2026-10-10.md`；历史数字不作本轮分母。

## 来源、边界和生命周期

- 明确opt-in，未设置/0保持原路径，其他值拒绝。仅resident512普通GPU/MLX，
  生成无参考或1–3个ref512/ref1024编辑，无PE/ANE/streaming/预算覆盖。
  native执行要求原BF16组件；不把它冒称GGUF/ANE预设。
- CPU缓存不是算法近似，不单独要求 `allow_approximation`。Viggle、ref512等请求
  自己原有的近似授权/固定SHA/strength/步数要求仍有效。
- 与独立compiled/fused/native/runtime/calibration encoder pilot互斥。原有
  fused-prefill processor lifetime保留；此独立入口退出时释放自己的owner。
- 初次捕获沿用native完整内容验证/认证proof cache与held-fd构造；使用前后复验
  named path和fd generation，digest进入conditioning namespace。不省略验证。
- 新解析的processor source限制16MiB；这是输入大小上限，**不是host RAM硬上限**。
  实际原文件11,422,654 bytes，SHA256
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`。
- 单个会话只留一个CPU owner，不按prompt积累。Session执行异常、取消、unload
  清理；开关切换隔离conditioning/prefix。规划拒绝不是一次模型执行。
- 回执 `qwen_tokenizer_reuse` 报告本次是否构造/复用/分词、实际processor SHA/bytes、
  构造/分词/校验时间。conditioning hit不重放旧分词或encoder执行计数。

## CPU组件，不是完整请求

使用实际processor、英文/中文/编辑/特殊图像标记共6种字符串；每臂1warm、
6项轮转样本。关闭臂计入构造、分词、销毁；复用臂计入前后generation校验与分词。
所有实际IDs及valid数量相同，无模型权重加载或GPU eval。

首版有效组件绑定首版研究库，关闭中位0.113269s，复用中位0.000088s。
首个verified owner构造0.132260s单列，使用既有OS/native-registration缓存，
不叫全冷磁盘或免费准备。owner最终销毁在退出/卸载，不在每次热分词跨度里。
原件 `outputs/qwen512-tokenizer-cpu-v2-20261018.json`，binary
`build/qwen21-tokenizer-reuse-v2-20261018/qwen21-tokenizer-reuse-probe`。
更早一版CPU原型绑定旧五步研究库，原件仍在，不混作新库重复样本。

## 同库24完整请求：关闭→开启→关闭

此节数据唯一来自**首版实测库**，不是后续安全修订库的新测速：

```text
build/qwen512-tokenizer-reuse-20261018/libturbocider.dylib
SHA256 53a5e5845e45485c3829e11df058543aaae7064b857369bb690d02033ebf5639
runtime tc-runtime-build-v1-19fc4fee046b39d12d0e56bc9f5b180b3cc2fb5496bfafd81b4dac579dccac90
```

每臂独立process、8请求：4生成（照片、英文、中文、三物体位置）+4单参考编辑
（改红、玻璃、加柠檬、改桌面）。原BF16组件、固定r128/strength1/inference-time/
227 terms、joint A/B/F32 ranks、512²、实际六次更新、seed73。无ANE/VAE近似、
encoder融合/权重保留、DiT/FFN时间复用或权重merge；计时不并行自己的build/test。

每个prompt均新，所有24项 `prompt_cache_hit=false`；两边都在首个编辑后复用同图
visual/VAE编码，仍完整执行language encoder。prefix snapshot关闭。每任务首个
请求是load/prime，不计入热态得分，但完整保留它的时间/图像/内存。不能把已缓存
condition的7.5s历史时间作新提示词分母。

下表控制为同一任务前后两关闭窗口的均值；范围为3个不同fresh prompt，统计为
对应任务配对，不将异构任务各自中位数直接相除。

| 任务 | encoder配对省时中位 | 整请求配对名义减少中位 |
| --- | ---: | ---: |
| 文生图，3个fresh prompts | 0.099475s | 1.680% |
| 图生图，3个fresh instructions | 0.156122s | 3.289% |

例：中文海报整请求控制8.201700→8.063877s，encoder0.676546→0.577071s；
三物体8.197629→8.059024s，encoder0.685091→0.553954s。编辑玻璃控制
9.806620→9.642347s，encoder0.932659→0.776537s。

不能把编辑全部墙钟差归因CPU缓存：改桌面的两关闭窗口为10.190542/9.676220s，
自身相差0.514s，明显大于CPU约0.113s组件成本；该任务名义0.475s省时不作因果
加速证明。小样本、非ABBA、竞争负载、频率/热态等均未受控。预设局部观察门槛
是encoder中位省≥0.075s、整请求不回归>1%、RGBA exact；本批达到，但不因此
获得正式performance/全任务资格。

首个生成开启10.661073s，两个关闭10.560700/10.660696s；没有冷首请求提速结论。
仅运行一次就销毁worker的场景，没有本轮证明的跨请求收益。

每一任务三臂RGBA exact，共24PNG；关闭恢复没有污染。七个可匹配任务还与旧
稳定六步库的对应PNG exact。只借旧PNG核对不变，不借其时间作加速分母。

三窗口process-tree峰值分别31,615,474,392 / 31,777,430,520 / 31,684,795,120
bytes，采样均complete、系统swap-in/out均0；发生过compression，不宣称无内存
压力。不是单个processor的独立peak/RAMcap，也不把峰值之差全归因CPU owner。

## 六步与五步原始张量各自不变

另8个parity-only请求，生成/编辑×关闭/开启×六/五步，各独立tier，不计速度。
初始noise、完整text、ref latent、denoised latent和FP32 pixels全部逐位相同，
对应safetensors文件SHA也相同。六步PNG复现完整请求对应图，五步PNG复现旧的
五步固定矩阵；不是说五步与六步互相等价。

首版raw目录：`outputs/qwen512-tokenizer-reuse-raw-{six,five}-20261018/`。
两窗口memory complete，峰值31,894,543,184 / 31,908,207,512 bytes，swap-in/out0，
持续竞争负载同样失败。没有停止用户ComfyUI或清系统cache。

## 首版回归、边界修订与证据

首版contract135项：134通过、1个既有Wan fixture缺失跳过；options host1、
runner host4、真实GPU lifecycle1、encoder typed/pack/source测试1（64 numeric
cases及限额/变更拒绝）、原mixed-GGUF prefill会话生命周期1、新BF16 CPU-owner
真实prepare-only生命周期1均通过。测试涉及语义/清理，不充当性能样本。

首次prepare-only测试误以为该API会dump tensor，FileNotFoundError失败；改为用
独立generate raw请求验证，未放宽prepare语义。已有processor内容变化拒绝、
同大小/保留mtime变更、错误恢复、取消重建、unload重建均实际通过。

首版独立CPU audit
`outputs/qwen512-tokenizer-reuse-audit-20261018.json`为complete，绑定649项native
source、原始memory/request/PNG、CPU实际runtime ID、七个历史RGBA、两档raw。
不证明物理引擎独占，不授予32张之外的图像质量资格。

收尾增加两处防护：从既有fused-prefill接收已解析owner时同样检查16MiB上限；
source若在denoise中改变，在PNG导出前拒绝，而不只在已写图后报错。
首版源码另存 `outputs/qwen512-tokenizer-reuse-v1-native-source-20261018.tar.gz`，
对应旧实测库不覆盖、不重标。旧五步库的受影响来源亦已单独归档。

### 当前V2安全修订库

```text
build/qwen512-tokenizer-reuse-v2-20261018/libturbocider.dylib
SHA256 a225f17e0fcbe04f7fa2b2a6b4c8178f2f663cb4bb20e43989022fc531c61e70
runtime tc-runtime-build-v1-2d36a6f10c424a5b670fa1317c73a00e74b4df1aeb7e587892db9d32fdce9e44
```

V2再次构建通过；contract135项仍134通过/1跳过，实际GPU lifecycle1、64-case
encoder typed/pack/source测试1通过，options host1及runner host4通过。
增强的真实BF16会话测试通过：在一个完成的denoise step后，同大小/保留mtime修改
owned processor，明确错误且PNG不存在，再次请求恢复；另以实际16,777,217-byte
合法JSON processor完成旧fused-prefill准备，再切独立入口，既有超限owner被拒绝，
恢复小processor后重新构造成功。只修改temporary fixture，不动原processor/权重。

V2另8个完整parity-only请求，目录
`outputs/qwen512-tokenizer-reuse-v2-raw-{six,five}-20261018/`，两档生成/编辑、
关闭/开启全部复现首版的原始值与PNG；跨版本共36个完整raw arrays核对exact。
不是V2的新测速或新画质提升。两窗口memory complete，峰值32,031,923,168 /
32,056,957,920 bytes、swap-in/out0，仍有竞争推理负载。

`outputs/qwen512-tokenizer-reuse-v2-guard-audit-20261018.json`为complete，
核对当前649项native source、实际新库、全部V2 raw及历史图像。首版649项来源中，
647项与当前一致，两项使用首版当时的精确源码快照核对；不拿当前源码重标首版。
这轮总共40个完整PNG请求：24旧测速+8旧raw+8安全修订raw；受控失败和prepare-only
调用不充当成功图片/性能样本。

当前脚本入口：

```sh
.venv/bin/python -B tools/validation/qwen_tokenizer_reuse_screen.py \
  --library build/qwen512-tokenizer-reuse-v2-20261018/libturbocider.dylib \
  --output outputs/new-tokenizer-full
```

此命令会产生新窗口；上文首版时间仍只属于首版库，不冒充V2已重跑的性能数据。

没有下载/改写模型或adapter，未部署App、修改默认或清理用户文件。主要DiT计算
仍占大部分延迟；本轮是小的CPU固定成本改进，不宣称完整优化目标已完成。
