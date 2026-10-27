# Qwen512 corrected-QKV 整请求：热态少约 1.8–2.2%，质量门槛失败

2026-10-18，Asia/Singapore，M4 Max / 64GiB。只做 Qwen Image 2.1 文生图和单 ref512
图生图，不处理 Z Image、不部署已安装 App、不改默认流程。持续优化目标保持 active。

## 决策

把此前 layer0 的 corrected-QKV 融合推进到了真正的 32 层、六步、VAE、PNG 请求。
无 dump 热请求窗口中，生成中位数 7.4854→7.3527s（少 1.773%），编辑
8.8790→8.6815s（少 2.224%）。这是整请求观察，不是 block 倍率外推。

**不采用该版**：事前质量门槛是 final latent relative L2≤.01、cosine≥.999、
RGB SSIM≥.99、alpha exact。生成 latent L2=.0177192 超过 .01，两条路径 alpha
最大差均为 1，故两者都判失败。没有放宽门槛、抹掉 alpha 或将失败改称通过。
所有完整窗口还检测到外部推理 CPU 负载，`qualification_passed=false`。

## 真实执行边界

新增默认关闭的 `TURBOCIDER_QWEN21_CORRECTED_QKV=1` 研究开关，仅准入：

- 原 BF16 safetensors 组件，resident / GPU / MLX、512²、6steps、frames1、audiofalse。
- 固定 Viggle v0.2.1 r128、strength1、inference-time，完整 227 个 logical adapter terms。
- joint BF16 A/B operands、F32 A-ranks；生成无参考或单 ref512 编辑。
- 显式允许近似、无 FFN/DB/prefix snapshot 跨请求复用、无其他 encoder/VAE/native/ANE pilot。

首步使用原普通 GPU block，文本/参考 prefix K/V 也走原路径。后五步的全部 32 层
在线计算三个独立 A-rank；融合 kernel 计算原 BF16 base、完整 B/scale、typed
Q/K RMSNorm 和 RoPE，返回 head-major Q/K/V，后续 attention、FFN、residual、
sampler、VAE 和 PNG 都完成。没有 adapter merge、少步数、跳层、裁 token 或 ANE。

成功请求在六步全部 latents 完成后检查 160 个 decode block 的覆盖，才写
`qwen_corrected_qkv.completed_decode_blocks=160`；这是完整执行逻辑覆盖，不是
硬件 kernel 次数或 ANE/GPU overlap 证据。

32 层 base bank 为 3,221,225,472 bytes（3GiB），按原 Q/K/V 排列复制，逐区间
以 uint16 比较原 BF16 位模式，不删除或改写原 source weights。它由 Session 持有，
同 adapter 连续 opt-in 请求复用；关闭开关、adapter 变化、卸载及异常清理路径释放。
本次 raw off→on→换种子→on→off 显示，MLX active bytes 精确增加/回落 3GiB：

| 任务 | off active bytes | on active bytes | 回到 off |
| --- | ---: | ---: | ---: |
| 生成 | 14,914,718,962 | 18,135,944,434 | 14,914,718,962 |
| 编辑 | 14,931,922,162 | 18,153,147,634 | 14,931,922,162 |

这是 MLX active owner 证据，不宣称 OS/driver 已退回等量物理 RAM。首次 pack 与
GPU 位比较约 .116–.159s（raw），两个无 dump on 窗口分别 .1502 / .1515s；
bank-hit 的 pack time 实际为 0。不能隐去 setup 或称 3GiB 为免费常驻。

## 分开的 raw 和性能实验

原 adapter SHA256：
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。
参考 PNG SHA256：
`67c8a9a3cf073214968c6c0b02ef40f9ea8999c2696898c46bc52e12b0763a9a`。
蓝茶壶生成、单参考改红，实际 caption rows 分别 24 / 35，编辑参考为 1024 tokens。

Raw 窗口共十张完整图片，每任务执行：off seed73、on seed73、on seed74、on seed73、
off seed73。相同种子的初始噪声、text、编辑 reference latents 逐位一致；回到相同
policy/seed 后最终 latents 与 RGBA 都 exact；seed74 初始噪声和最终 PNG 确实改变。
开启后的 bank 也正确 reused，未返回上次已完成 tensor。

| 任务 | final latent relative L2 | cosine | RGB SSIM | RGB MAE/255 | alpha max abs | 事前质量门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 生成 | .0177191650 | .9998430136 | .9922324164 | 1.1717428 | 1 | 失败 |
| 编辑 | .0047558285 | .9999887151 | .9987610365 | .3724416 | 1 | 失败 |

目视并排图未见候选相对普通 GPU 的明显结构变化，但该小样本主观观察不能替代
数值门槛或 instruction-adherence 评测。RGB 看起来接近不授权忽略 latent/alpha。
这次只验证两类茶壶任务、两个 seed，不宣称一般图片质量合格。

四个无 dump 性能窗口每个进程一个固定 task/arm，六次完整请求，首次 cold/setup
单列，其余五次 conditioning-hot 取中位数；seed73/74 交替并检验同 seed PNG exact。
没有在请求计时内做 CPU pixel readback，时间包括 VAE 和 PNG 导出。生成按 off/on，
编辑按 on/off 顺序，进程之间不并行；不是每个 arm 内混用 prefix hit/miss 的弱对照。

| 任务 | 原 GPU 热态 s | corrected 热态 s | 少用时间 | 各 arm 首请求 s（off/on） |
| --- | ---: | ---: | ---: | --- |
| 生成 | 7.4853935 | 7.3527037 | .1326898s / 1.773% | 11.1209 / 9.2840 |
| 编辑 | 8.8789958 | 8.6815465 | .1974493s / 2.224% | 11.9160 / 11.8340 |

首次加载/JIT/系统缓存不同，以上 cold 数字保留但不据此宣称冷启动收益；本版额外
pack 本身明确非零。只有一对独立进程窗口/任务，并且存在 competing load，尚无
重复独占设备性能资格，也不把这个约 2% 当作可发布结果。

## 构建、迁移和独立审计

本轮确实改变了 native 原型源码，并完整构建独立、Private-enabled 新库，未覆盖
此前 stable library、历史探针、App 或原始权重：

```text
build/qwen512-corrected-qkv-v2-20261018/libturbocider.dylib
SHA256 c592080ed9a959e2cc54f9eb554cb76950b44578f24fbfed9115064ee2733a00
runtime tc-runtime-build-v1-fb6da525ecdafca9eebb35c4812a605709f73670cd94e7728b4f201055ed10cb
```

首个 v1 build 在编译阶段因局部 result 声明位置出错，未产出库或运行模型；v2 已修正。
新增 native header 为 `native/models/qwen21/metal/lora_qkv_prepare.hpp`；接入与生命周期
在 Transformer/Session 中，metrics 由 C API serializer 实际返回。原 kernel 的
base/corrected BF16 舍入，以及 normalized BF16→gamma→BF16 的修复均保留。

验证结果：

- 原 GPU kernel 40 数值 / 90 拒绝测试通过，worst relative L2≈6.52205e-9；
  迁移到 native 后另有 20 个全输出逐位一致 case、50 个 native 非法合同拒绝。
- C++ request admission 的正/负例通过；全请求 host gate 四项通过。
- 新库下普通 Transformer lifecycle 回归通过，包含 24 次释放/确定性检查和既有变体。
- C API contract 135 项结束，134 通过、1 项因缺失 Wan fixture 跳过，不称全部 fixture 通过。
- 全部 34 个完整六步请求成功：raw10 + timing24；其中候选18，累计 2880 个
  completed decode block，仍不是独立 prompt 数量。

`outputs/qwen512-corrected-qkv-request-audit-20261018.json` CPU 审计 complete：
650 项 native/build 输入与链接库 manifest、当前 tree 及完整 source archive 三方一致；
重算 raw L2/cosine/SSIM、恢复/换种子和 3GiB active bank 生命周期；重新读取全部
PNG，确认 raw 与无 dump 性能窗口的相同 task/policy/seed73 图片逐位相同；复算五次
完整请求时间分母与两对 reduction、核对 memory evidence/report digest。失败质量
结论保留为 false，没有由审计授予性能/semantic/physical qualification。

完整 native 输入归档：`outputs/qwen512-corrected-qkv-native-source-20261018.tar.gz`，
SHA256 `0e290e5a04c88ce33a383a42f9bf4eafffe39d993492293b29d34ad7170bea61`。
以后 native tree 变化时先使用此归档还原到新目录，不把当前源码冒称该库源码。

## 内存与原件

Raw 窗口 process-tree peak phys footprint 31,979,690,904 bytes，系统 swap-in/out0，
系统 compression/decompression 有约 3.91/3.54GB 增量；不归因于单个进程。
无 dump 四窗口峰值分别 19,184,610,584 / 21,998,333,208 / 22,248,582,400 /
19,159,723,264 bytes（gen off/on、edit on/off），均 swap-in/out0、compression0。
这些是包含 load/cold/warm/exit 的进程树窗口，不是单张图片或 GPU/OS 全部归属。

原件：

- `outputs/qwen512-corrected-qkv-raw-20261018/`：raw arrays、十张 PNG、comparison.png。
- `outputs/qwen512-corrected-qkv-{gen,edit}-{off,on}-20261018/`：四个无 dump 完整窗口。
- `outputs/qwen512-corrected-qkv-kernel-20261018.log`：迁移 kernel 合同结果。
- `tools/validation/qwen_corrected_qkv_request_{screen,audit}.py`：完整请求与 CPU 复核工具。

结束磁盘约 13GiB 可用，无下载、无清理用户文件；本轮 owned build/test/model/audit
任务均已退出。下一步应对原生 base GEMM、B、RMSNorm/RoPE 逐项做数值反事实，
降低全步累积误差，或找可逐位保持的图/布局优化；不通过放宽这次门槛来接入默认。

后续已完成 [局部 RoPE 收缩修复](qwen512-corrected-qkv-rope-rounding-2026-10-18.md)：
原门槛未改，canonical 整请求及七类任务/两个 seed 的 final latents、RGBA 恢复
逐位一致。该后续结果不重写本页 v2 的失败质量数据或性能分母。
