# Qwen512 corrected-QKV 原缓冲区直读：省去额外3GiB权重副本

2026-10-18，Asia/Singapore，M4 Max /64GiB。仅 Qwen Image 2.1 的512²生成和单ref512
编辑；不涉及Z Image、不部署App、不改默认值，完整持续优化目标保持active。

## 已验证的改动

[上一版](qwen512-corrected-qkv-rope-rounding-2026-10-18.md)融合QKV约少2.3%热请求
时间，但需要把32层Q/K/V物理拼成额外3GiB bank。本轮新增显式direct路径：每个
threadgroup按projection kind读取原有Wq/Wk/Wv buffer，列坐标用`col % H`，原
physical row pitch仍为H。不拼接、不复制base，不把三路LoRA当成同一个adapter。

两种storage使用同一份shader算术body，只替换base的buffer选择与坐标映射。保留
独立在线F32 A-rank、原BF16 B/scale、base/corrected BF16边界、weighted RMSNorm
中的typed rounding，以及局部RoPE contraction修复。首步和prefix K/V仍普通GPU，
后五步全32层decode，成功请求仍要求160个completed decode block。

新增`TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT=1`，须同时显式开启原
`TURBOCIDER_QWEN21_CORRECTED_QKV=1`和joint BF16 A/B。准入仍是原BF16组件、
resident/GPU/MLX、512²、六步、固定Viggle v0.2.1 r128 strength1/inference-time，
生成无参考或单ref512编辑，不支持其他cache/encoder/VAE/ANE/streaming pilot组合。
planner/runtime共享准入；main关闭却direct开启、错误literal或越界请求均拒绝。

direct receipt为`decode-only-corrected-qkv-v3-original-buffers`，`weight_storage`
为`original-buffers`，`bank_bytes=0`、`bank_reused=false`、pack time0。不能把
原buffer复用冒称一个不存在的bank hit。本路径绑定已materialize的原checkpoint
矩阵；lazy flags不是最终physical layout证据，非法stride控制在eval后核对。

这消除的是额外base bank及其pack，不是宣称所有activation/rank/driver staging
均零复制。x/ranks及原有F32→BF16边界仍按原图执行。

## 完整请求与生命周期

canonical raw共十张完整六步/VAE/PNG：每任务off73→direct73→direct74→direct73
→off73。同seed噪声、conditioning、final latents与RGBA逐位一致，换seed重新计算，
返回原policy/seed后恢复。原L2/cosine/SSIM/alpha门槛未改，均通过。

在两个canonical任务中，direct on/off的MLX active bytes完全相同：生成
14,914,718,962、编辑14,931,922,162 bytes。无额外3GiB active weight owner，
不等于保证OS/driver物理内存立即退回。

另完成七次生成混合切换：off73→packed73→direct73→direct74→direct73→packed73
→off73。所有seed73 final latents/RGBA逐位一致，seed74确实改变；实际active
增量依次为`[0,3GiB,0,0,0,3GiB,0]`。两次packed均重新prepare，不误报hit；
进入direct或optout前原bank owner正确释放。本项跨storage验证范围是生成，
编辑direct/optout恢复由canonical窗口覆盖。

独立七任务×两seed矩阵共28次完整请求、14对：照片、英文文字、中文文字、空间
布局、编辑颜色/材质/添加物体。**14/14 final latents与完整RGBA全部逐位一致**，
noise/text/reference亦一致。不把相对joint A/B baseline相同当原模型本身的
instruction-adherence已合格，也不宣称任意dtype/硬件/参考数量的全称证明。

## 测试与当前计时状态

native packed与direct各20个kernel case都对原typed reference逐位一致；direct
fixture使用三个独立Data owner，并覆盖非零origin、lazy/非紧凑x、多个M/H/rank/
tile、零值和正负scale。native packed50/direct65非法合同拒绝通过，原40数值/
90拒绝保留。早期非法transpose测试误把lazy flags作最终stride，改为先eval后
验证物理layout；没有放宽production数值门槛。

137项C API contract中136通过、1项缺fixture跳过，普通Transformer lifecycle
回归通过，request/storage host gate八项通过。新库是独立
`build/qwen512-corrected-qkv-direct-v4-20261018/libturbocider.dylib`，历史库与权重
均保留，没有覆盖App或默认工作流。

## 无dump完整请求速度

四个固定task/arm的独立进程窗口均complete。每个进程执行六次完整请求，首请求
含load/JIT单列，其后五次conditioning-hot取中位数，seed73/74交替并检查同seed
PNG恢复exact。生成off/on、编辑on/off顺序，不并行运行自己的模型、构建或测试。
计入全部六步、VAE和PNG导出，不使用带dump窗口作性能分母：

| 任务 | 原普通GPU热中位数 s | direct热中位数 s | 减少 |
| --- | ---: | ---: | ---: |
| 文生图 | 7.4975514 | 7.3409427 | .1566087s /2.08880% |
| 图生图 | 8.8656934 | 8.7078821 | .1578113s /1.78002% |

首请求分别gen off/on=10.6569/9.0744s、edit on/off=11.1928/11.4032s。系统cache、
JIT和执行顺序不同，不从这些数字声称冷启动倍率；direct实际pack0的结论来自
实现与receipt，不能拿raw/首请求时间差替代。也不跨库套用或相乘上一版packed的
倍率，不声称已经证明direct吞吐超过packed。

当前每任务只有一对独立窗口，全部仍检测到外部推理CPU负载，
`qualification_passed=false`。是新的热态观察，不是独占设备统计性能资格。
source默认值和已安装App保持不变。

## 独立审计与来源

```text
build/qwen512-corrected-qkv-direct-v4-20261018/libturbocider.dylib
SHA256 2d8a74506150537bd707ce0aa1297dcf5f548aa6976875d76f79c5b8c773b024
runtime tc-runtime-build-v1-72f1c95bc56aa637da945a1493738ce141fc5c62948617af8e6a39aaa23eedb7
```

`outputs/qwen512-corrected-qkv-direct-request-audit-20261018.json` CPU审计complete：
650项native/build inputs与当前tree、链接manifest、完整source archive一致；
重算canonical和七任务矩阵的所有原始noise/text/reference/final-latent、RGBA、
L2/cosine/SSIM/alpha；重读全部无dump PNG，确认同task/policy/seed73与raw和
矩阵结果逐位一致；复算完整请求分母和两项百分比，核对全部memory digest。

另独立重算了七次混合storage切换的bank owner增减、重新prepare和输出恢复。
本轮共69次完整请求：raw10、生成switch7、matrix28、无dump24；其中direct35次、
packed2次，累计5920个completed decode block。不是69个独立prompt，也不是硬件
kernel次数。原门槛通过不意味着原模型semantic质量或任意prompt被全称认证。

完整source archive：`outputs/qwen512-corrected-qkv-direct-native-source-20261018.tar.gz`，
SHA256 `41d1c574cfefef687607c62240248f703a06ec5371382b020401c494b3d7cf09`。
以后native tree变化时按归档恢复到新目录，不将新源码冒充该二进制的源码。

## 内存、原件和后续

所有七个完整请求窗口的进程树memory采样均complete，system swap-in/out均0。
canonical raw /matrix峰值31,813,540,688 /31,940,238,256 bytes，系统compression
分别约2.89/3.93GB，不归因于某一进程；生成混合switch峰值22,334,582,040 bytes、
compression0。无dump gen off/on、edit on/off峰值分别18,194,903,832 /
19,139,079,424 /18,936,999,168 /18,782,074,152 bytes，compression均0。

direct清除了额外3GiB持久weight owner，但workspace、allocator cache、JIT等仍有
成本，不能把active-bank差值说成相同幅度的进程树峰值/OS RAM下降。

原件：

- `outputs/qwen512-corrected-qkv-direct-raw-20261018/`：两类任务十张完整对照和数组。
- `outputs/qwen512-corrected-qkv-direct-switch-gen-20261018/`：七次混合storage切换。
- `outputs/qwen512-corrected-qkv-direct-matrix-20261018/`：七任务/两seed的28张图和数组。
- `outputs/qwen512-corrected-qkv-direct-{gen,edit}-{off,on}-20261018/`：四个无dump窗口。
- `outputs/qwen512-corrected-qkv-direct-kernel-ready-20261018.log`：最终kernel合同结果。

结束可用磁盘约12GiB，无新模型下载、无用户文件清理，owned构建/测试/设备/audit
任务均已退出。后续继续重复整请求窗口、检查其他调度/精度组合的真实净收益；
例如原有作者支持五步sampler与这个逐位保持的kernel能否组合，须同窗口另测，
仍明确区分五步质量tier与六步等价工作，不直接相乘历史倍率或改默认。
