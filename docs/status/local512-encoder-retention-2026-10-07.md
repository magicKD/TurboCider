# 512²：编码器执行器复用、staging 生命周期与 DiT 复测

2026-10-07 UTC；目录的 `20261008` 使用 Asia/Singapore 日期。
接续 [Qwen encoder 接入](local512-qwen-encoder-2026-10-07.md)。只使用
本地模型、Viggle LoRA 与参考 PNG，没有下载或修改模型/参考工程。
完整 Z/Qwen base/LoRA、1–2 参考图、DiT+encoder 盈利与质量目标仍 active。

## 实现与安全边界

新增显式 `TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME=1`。一个 resident
Qwen Session 最多保留一个语言 FFN executor，不保留完整17.5GB encoder
checkpoint arrays；默认仍 request-local。要求512²、非 constrained/
streaming/legacy memory-budget 请求。GPU-only、关闭开关、manifest digest/
executor configuration 改变、错误或 unload 均释放旧 executor。

encoder 仍受最多1GiB机会性预算与每请求 memory admission；条件缓存命中
也重新检查 resident admission。DiT FFN/QKV 创建预算扣除 retained encoder
estimate；encoder 创建预算扣除现有 FFN/QKV estimate，避免重复使用共享
2GiB optional allowance。这些是估计/机会性保护，不是完整进程 RAM cap。

新增 `encoder_runtime_reuse`：enabled、reused、retained、本请求 actual
calls 和 retained estimate。`encoder_hybrid` 继续是执行器累计 counters；
顶层/plan 的实际执行标签按 request delta 判断，不能用历史72calls把
本次零调用的 GPU probe/cache hit 标成混合执行。cache hit不重放旧 metrics。
当前 Private rows320/Fa3072 estimate243,794,176 bytes；Public FP16
rows320 estimate705,183,744 bytes，均未生成权重 sidecar。

复用审查发现必要的生命周期修复：attention 在 stage之后、FFN launch之前
失败时，`pending_` 尚false，但 staging 已异步读取 borrowed sources。
`HybridFfn::drain()` 现在等待 graph 的 staging/launch，再清 source owners。
新增 `SourceScope`，声明在 sources之后；成功显式finish，异常展开时先
drain再释放权重，保留原异常。Qwen pipeline使用这个guard，不等到外层
catch才清 member executor。八轮staged-only正常/模拟attention异常退出后，
同一真实 Public graph恢复编码通过；没有用线程睡眠伪造时序。

## 真正不同条件的配对工具

新增 `qwen_encoder_residency_screen.py`：GPU / encoder_local /
encoder_retained 独立进程串行；同 native library、adapter、reference
顺序、seed29、512²、六步真实 LoRA，每臂三个不同prompt，**全部条件缓存
miss**。第一个请求单列，随后两个fresh requests全保留；不是拿同prompt
cache hit当作加速。DiT保持原GPU，227真实LoRA projections不变。

工具校验request/cumulative calls、实际 Private W8/channel或Public row
executor、释放与reuse、非有限/非正 timing、zero fallback、相同输入/
runtime bytes；model文件generation stamp和有界header hash变化会拒绝。
这是metadata/layout identity，不是full payload digest或immutable lease。
第一次一参考图forward发生在新增model snapshot之前，保留原记录，不
追认其已绑定model stamp。每臂100ms process-tree sampling完整；这里没有
strict competing-load资格或physical GPU/ANE trace。

| 参考图 / order | GPU fresh median s | local encoder median s | retained encoder median s |
| --- | ---: | ---: | ---: |
| 1 / GPU→local→retained | 10.768094 | 11.262875 | 11.148172 |
| 1 / retained→local→GPU | 10.719059 | 11.265932 | 11.180231 |
| 2 / GPU→local→retained | 12.896907 | 13.488216 | 13.371204 |

复用约省0.086–0.117s，但仍慢于GPU encoder，不改默认。每个一参考图
候选请求36calls；两参考图72calls。retained累计36/72/108、72/144/216，
本次仍36/72；第一次reused=false，随后true。新权重allocation仍导致
scale misses108/216/324、hits0，不把图复用误称完整权重/scale热复用。

这时local executor load已仅约0.007–0.028s，自测约0.05–0.096s；
前次初次compile/load0.37–0.8s不再代表每次新executor的固定开销。
下一步更值得测的是编码权重读取/实际source代次复用和GPU fusion，而非
继续把图加载当作唯一慢因。memory peaks31.1–32.1GB（十进制bytes，
process-tree phys-footprint）随order变化，不将差值全归因于retention；
九个trial的swap-in/out均0，不声称整个系统swap为0。

所有local/retained对应PNG exact。另检查一参考图gentle、两参考图golden
的GPU/retained whole与三处原像素crop：形状、布局、把手、光照很接近，
釉面/高光/纹理有小差异。blue壶仍opaque；不是透明材质保真、多seed或
用户认可。agent review保留 `qualification_passed=false`。

## Public：真实两参考图/LoRA也执行成功，但更慢

新Public-only v2 library，同库GPU与encoder cold诊断、同六步adapter。
rows320、固定一row chunk、retention1、完整GPU DiT：

| arm | text s | request s | actual encoder calls |
| --- | ---: | ---: | ---: |
| GPU | 2.147266 | 16.554299 | 0 |
| Public FP16 | 5.108388 | 17.567945 | 36 |

Public runtime failure/fallback0、227adapter projections、实际6步。
这是Core ML CPU+NE policy，不是物理ANE驻留证明。两图whole/crops很
接近，小纹理/高光差异；未将成功调用视为性能资格或最快路线。

## Z512 base：strict拒绝与独立双向诊断

新Private v2库、BF16本地模型、fox/seed42、8步、Fa4096、rows1056、
fixed async、GPU IOSurface、specialize/fence1、FP32 partial join；
prefetch/lookahead0。原优化GPU实际为 `compiled_fused_blocks`。

strict reverse运行了runtime冷+两热，40个load samples中1个出现外部
Comfy安装的Python进程PID10739、CPU73.8%。GPU未启动，summary仍
incomplete；不能拼接或计算accepted ratio，也不把该观测等同GPU推理
证明。没有signal外部进程或降低gate。

另开独立diagnostic目录，两arm/两order完整执行，保留memory sampling，
**未请求continuous load qualification**：

| order | optimized GPU warm median s | Private warm median s | diagnostic ratio |
| --- | ---: | ---: | ---: |
| GPU→runtime | 6.992746 | 5.679209 | 1.23129× |
| runtime→GPU | 6.990513 | 5.684607 | 1.22973× |

每请求256实际Private calls；失败、retry、fallback0，headroom1。
四个process-tree memory trial完整、swap-in/out0；约22.7–22.95GB peak，
不归因外部ANE服务/driver。fox姿态、眼耳、构图和雪松场景很接近，毛发/
枝条/雪纹有小差异。这是本cell的新诊断，不是全部目标或strict1.2×资格。

## 回归、证据与清理

Private/Public选定Core ML/MLX suite各16通过；真实retained session
switch/recovery/off测试1项通过；新工具5项host tests通过；另3项实际
Private tests覆盖raw GGUF/affine/dense、Comfy raw/packed/group A8、
多chunk/LoRA/generation/failure/4224 bucket。全部上述执行无skip，不是
全仓make test。Public实际release guard通过。

两库488 source inputs各自匹配，构建仍含原ConvRot drafts，提交只包含
本轮owned changes。机器记录见
[evidence](../design/validation/local512-encoder-retention-20261007.json)。
库SHA256：

```text
Private 86954839affea8381cc05240496131a87c2a7529fc6d4feca0c1111ad279c154
Public  5678b3d76a66b73ab42923afecd1bf7de2e53c02b535315c89b79650e4e32d31
```

完成全部owned jobs后，仅清理三个retention v1/v2 build的637个`.o`，
49,580,808 logical bytes（约47MiB），可重建。保留库、CLI、manifest、
log、PNG、模型和adapter；没有下载、新dense模型副本或大activation dump。

继续完整目标：Qwen/Z base与LoRA的盈利分区、两参考图DiT+encoder组合、
可预算且公平匹配GPU的encoder source复用、GGUF真实提前decode复用与
ConvRot数值/视觉修复。stage组件和本轮retention不能替代这些未完成项。
