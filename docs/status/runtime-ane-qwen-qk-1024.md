# Qwen 1024² base：可选 Q/K norm-RoPE 与 runtime 组合

2026-09-29，接续[512²六配置对照](runtime-ane-qwen-qk.md)。该组runtime自身
有1.028×收益，同样优化GPU后混合为1.170×；不把512²成绩外推到1024²。

## 候选与选择依据

当前1024² c320实验的暴露staging等待低于每请求2ms，不能仅依据CPU
staging总时间就断言双缓冲可以消除数秒整请求耗时。c1792此前已经改善
长序列分支平衡，但未与优化后的GPU/冻结图做同构建六配置对照。
本轮优先验证已有GPU kernel在更长base序列上的组合收益；不是新的ANE
kernel，不跳过SiLU，不合并LoRA，不改变图、dtype或自动调度。

- 原 `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` 增加1024² **resident、
  base、纯文生图**门禁。原512²生成/编辑/LoRA范围不变，不放开1024²
  LoRA、编辑、矩形、streaming、W8A16 GPU或paired-RoPE组合。
- 三条路线共用同一已有kernel；冻结1024²仍需独立的4096-row匹配图和
  既有1024 diagnostic，不能把512²冻结图换个文件名复用。
- screen只在明确无LoRA、无reference时允许1024² Q/K开关；所有受测
  路线统一设置，保留计划/实际kernel标记检查与清理继承环境。
- kernel回归增加4096/4120/4608行，覆盖1024² decode、普通prefill及
  512-text-token上界；沿用relative L2<0.01，不放宽有限性检查。

本轮前3986库和CLI保存在 `outputs/runtime-ane/qk1024-before-bin/`。
新库必须重新构建和验证，不将3986或e55成绩重标为新构建成绩。

## 预声明对照

先构建并串行运行host/路由、Core ML微图、Qwen专项和扩展kernel测试。
通过后再启动模型计时；期间不修改绑定的源码、工具、库、图或driver。
每trial预检竞争推理，繁忙则等待，不终止他人进程。

- Qwen BF16 base，1024×1024、40步、狐狸雪景prompt/seed42、resident。
- runtime复用c1792/K1024/N512 v2图，base输入零修正；chunks=auto，
  profile关闭。冻结图沿用此前已匹配的4096-row/32层artifact。
- GPU关 → runtime关 → frozen关 → frozen开 → runtime开 → GPU开 →
  反向，共12trial/36请求。每trial一冷两热，每配置全部四热样本取中位；
  request_wall包括VAE/PNG，排除冷请求，不剔除GPU探测或慢请求。
- 全部trial独立100ms进程树采样，max-gap500ms；每trial timeout1800s，
  不在观察超时后重启仍存活的进程。
- 分开报告各路线的Q/K关/开，以及同样融合设置下GPU/runtime/frozen比。
  不跨构建/campaign拼接分母，不称小样本结果为统计显著。
- 结束后独立核对原始请求/计时/调用、图/库/源码身份和全部内存流，再
  查看反向六配置末次热图片；不以kernel误差或receipt替代视觉验收。

## 构建与验证

候选库SHA为
`55e5e736ebd07c8ea50fa5ab6250ee7fd2f6238bdb014f4bf60bcb9da375d88a`。
`TURBOCIDER_NATIVE_ONLY=1 make build`、
`make test-runtime-ane test-acceleration-contract test-qwen21`、扩展
`qwen21-metal-qk-test`及完整`make test`均正常exit 0，之后才启动计时。

4项host、9项Core ML/MLX微图、91项加速契约（含同一4项host）、Qwen
38+7+30项及三个native/工作流入口通过。screen现为42项，新增1024²
LoRA/编辑在读artifact前拒绝、base三条路线相同Q/K设置与冻结专用门禁
回归。完整回归的缺夹具/专用构建/GPU opt-in skips仍不算新增覆盖。

九种行数的Q relative L2均为0，K最坏0.00325761；新增4096/4120/4608行
分别为0.00325436/0.00325452/0.00325386，均小于既有0.01。没有修改
kernel数学或误差阈值，这些合成检查不是1024²图像质量验收。

证据与运行入口：

```text
outputs/runtime-ane/qk1024-build.log
outputs/runtime-ane/qk1024-tests.log
outputs/runtime-ane/qk1024-kernel-test.log
outputs/runtime-ane/qk1024-full-test.log
outputs/runtime-ane/qwen-qk-1024-factorial.py
outputs/runtime-ane/qwen-qk-1024-factorial/
outputs/runtime-ane/qwen-qk-1024-driver.log
outputs/runtime-ane/verify-qwen-qk-1024-factorial.py
```

## 完整对照与独立复核（2026-09-29）

原队列正常exit 0，12/12 trial、36/36请求完成，父报告为complete且
`unchanged=true`。25项绑定输入SHA不变；计时期间没有重建或修改库、
源码、共享工具或图，没有重启队列。独立脚本重验原始请求、实际Q/K标记、
调用数、PNG SHA、全部内存流及四热样本/配置的合并中位数，正常exit 0：

```text
outputs/runtime-ane/qwen-qk-1024-verified.json
```

| 路线 | Q/K关，整请求秒 | Q/K开，整请求秒 | 自身关/开 |
| --- | ---: | ---: | ---: |
| GPU | 187.815604 | 182.961481 | 1.026531× |
| runtime v2 c1792/K1024/N512 | 153.789411 | 147.348778 | 1.043710× |
| 冻结base，4096 rows | 163.513712 | 160.718428 | 1.017392× |

同样关闭融合，GPU/runtime为1.221252×，GPU/冻结图为1.148623×。
同样开启融合，分别为**1.241690× / 1.138398×**。runtime相对冻结图为
1.090735×，在本组1024² base工作负载中最快；这与512²冻结图仍最快的
结论不冲突。没有把旧e55或3986的GPU时间当分母，也没有剔除自动GPU探测。
冻结融合的正向热请求159.398/159.300 s，反向162.481/162.039 s，存在
方向间波动；全部纳入统计，不称四个热样本为广泛统计资格。

### 调用与内存

- 四个runtime trial的累计prediction均为1184/2432/3648；每请求增量
  1184/1248/1216。overflow retries、失败和fallback blocks均为0；
  调度器主动GPU探测不等于错误回退。
- 四个冻结trial累计prediction为1248/2496/3744；没有失败记录。
- 60,420个独立进程树样本，最大间隔120.308792 ms，小于500 ms门限；
  包含加载、冷请求、热请求和退出。各路线全trial最大进程树footprint
  为GPU31.627 GB、runtime32.125 GB、冻结图40.245 GB（十进制）。
- 没有新增系统swap-out，但不是全组零内存压力：trial0/6/10各64 KiB
  swap-in，trial7为128 KiB，trial2为1,114,112 bytes；trial2还有
  522,829,824 bytes compression，trial8还有229,376 bytes。
  其余记录为0。系统计数不能全部归因于ANE，也不是完整driver/wired预算。

### 图片与适用边界

已逐张查看反向六配置的末次热PNG（trial6–11）。狐狸主体、坐姿、松枝、
雪景和色调接近；耳形、面部/胸毛、尾部及雪粒纹理存在局部变化，冻结图
的部分细节变化更明显。该单prompt/seed样本未见明显整体质量退化，符合
本轮肉眼接近的筛选标准；不是逐像素一致，也不是多提示词质量验收。

保留c1792 runtime与Q/K组合作为显式optional，不改变默认路由。新模板
`examples/requests/qwen21-base-1024.json`默认GPU，命令行显式选择runtime图。
1024² LoRA/编辑、矩形、streaming仍不在本次Q/K资格内；不能从零LoRA修正
的base成绩推出任意runtime LoRA也有1.24×。物理ANE驻留、硬件重叠和完整
内存压力处理仍未验收。收尾不接续新的优化候选。
