# Qwen512：作者五步采样档位，非等价六步加速

2026-10-18，Asia/Singapore，M4 Max/64GiB。只关注 Qwen Image 2.1 文生图、
图生图；不优化 Z Image，不改 App、默认采样器或已安装库。完整优化目标继续。

## 结论

新增默认关闭的研究入口 `TURBOCIDER_QWEN21_STUDENT_FIVE_STEP=1`，配合请求
`steps: 5`。它使用原有、内容固定的 Viggle v0.2.1 r128，不下载/裁剪权重。
这是**少一次模型求值的独立采样/画质档位**，不是六步等价 GPU kernel 提速。

同窗口完整请求中位数：文生图7.527811→6.382447秒（少15.215%），重复编辑
7.671802→6.466273秒（少15.714%）。计分仅为conditioning-hot；编辑两边均为
实际 immutable prefix hit。各任务、各臂仅3项，不是置信区间或冷启动保证。
持续负载检查发现外部推理，因此 `qualification_passed=false`；未停止用户进程。

28张固定任务图显示：中英文文字、三物体位置关系在这个小样本中可以呈现，
但生成构图/比例、排版和添加物体的位置/细节会改变。SSIM不等于指令完成质量。
不授予等价画质、全任务、身份保持、多参考、透明素材或生产默认资格。

## 作者配方与来源

只读核对作者仓库 `Viggle/Qwen-Image-2.1-viggle-turbo`，固定 README revision
`b667ee50696e8f4ccd161e02f8af0e43bb5665ed`。作者明确给出：

```text
默认六节点：[1, .9375, .875, .75, .5, .25]
可选五节点：[1,        .875, .75, .5, .25]
```

使用既有Qwen分辨率相关mu进行shift，追加0，不做base模型的terminal0.02伸展。
不是取六节点的前五项，也不是base Euler等距节点，更没有换旧v0.1四步adapter。
作者离线质量数字不代替本机检查。

原有权重：

```text
Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors
679604800 bytes
SHA256 bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b
```

文件名仅用于规划入口，Session仍从绑定来源核对固定SHA，227个runtime adapter
投影全部保留，strength1/inference_time。未修改r128/r256文件或六步默认。

## 实现边界

- 仅512×512、resident、显式approximation、普通GPU/MLX；文生图无参考，
  编辑1–3个ref512。原BF16组件；本机实图只覆盖单参考，三参考仅参数契约覆盖。
- flag必须是未设置、`0`或`1`；`1`只接受steps5。未显式开启的Viggle五步仍拒绝。
  六步回到未设置/0；r256、ANE、native Metal、staged、1024和其他已有六步限定不扩大。
- 可用原始LoRA算术或既有joint BF16 A/B/F32 ranks。实际测速固定后者；不把两种
  算术混成分母。排除独立A/B开关、FP16 ranks、VAE近似、残差/FFN时间复用、
  fused QKV/local attention等其他采样近似组合。未增加App开关。
- 计划标记为 `qwen21_viggle_v021_r128_5step_sampler_experimental`；实际selection
  标明authored five-step/distinct quality tier；steps/actual_denoise_steps均实际为5。
  不把五步回执写成six-step distillation。
- Prefix runtime identity同时绑定精确sampler recipe和步数。仍只保存一个有上限的
  immutable conditioning bank；换档失效，重新填充后才能计hit。六步block observer
  和原 `pinned_student_gpu_request` 资格不扩大到五步。

## 完整请求测速，20张

同库、同原始BF16组件、相同r128/joint算术、seed73/74、同提示词和参考，原VAE，
无ANE、无其他近似、无权重merge。8个生成请求+12个编辑请求。首请求、五步warm、
每次换档的prefix prime、额外最终恢复不进入计分。测量不并行自己的build/test/图片
比较；保留完整memory/process-load日志。

| 任务 | 六步3项 / s | 五步3项 / s | 中位数减少 |
| --- | --- | --- | ---: |
| 文生图 | 7.527811, 7.535062, 7.508744 | 6.382447, 6.423634, 6.367982 | 15.215% |
| 编辑，两个档位均prefix hit | 7.629971, 7.675176, 7.671802 | 6.466273, 6.443341, 6.483637 | 15.714% |

五/六档内重复输出各自RGBA exact；六→五→六恢复exact，换档时所有预期miss/hit
均被实际回执核对。四个六步task/seed结果还与旧稳定VAE-v3库的原精度控制逐像素
一致，不把旧窗口的时间拼成新的性能分母。

本窗口process-tree峰值physical footprint31,924,493,256 bytes，独立采样complete，
最大gap109,972,792ns，系统swap-in/out均0。不是每臂独立峰值，不是DRAM带宽，
也没有GPU独占/ANE物理重叠资格。

## 固定画质矩阵，28张，不计速度

7个固定任务×seed73/74×六/五步。每任务按6(seed73)→5(seed73)→5(seed74)→
6(seed74)运行，统一prefix-off、原VAE和同一joint算术。没有挑最快或最好看的seed。
下表是同seed不同采样器的RGB差异范围，不是“通过率”。

| 任务 | RGB SSIM | RGB MAE /255 | alpha最大差 |
| --- | --- | --- | ---: |
| 蓝茶壶照片 | .770549–.855500 | 15.390–27.874 | 2 |
| 英文 GREEN TEA | .785652–.822185 | 18.938–20.705 | 29 |
| 中文 秋日茶会 | .882379–.949374 | 3.936–10.063 | 17 |
| 红方块/蓝球/黄锥，左中右 | .855812–.944637 | 5.080–14.192 | 2 |
| 茶壶改红 | .986697–.989570 | 1.225–1.514 | 1 |
| 透明玻璃材质 | .988416–.991901 | 2.867–3.444 | 8 |
| 左侧加一个柠檬 | .745680–.891640 | 7.862–13.733 | 5 |

人工查看原始PNG及成对contact sheet：英文词语和四个汉字在这些图中可见，
seed74中文五步改为两行；三种物体的数量、颜色及左中右关系可见。照片中壶身比例
和场景变化明显。改色/玻璃材质的两档更接近，但细节并非相同。加柠檬用例尤其会
改变局部位置/形状；两档的一些图都有柠檬贴边/局部裁切、陶瓷质感和参考保留问题，
不把共同存在的问题归因给五步，也不因SSIM接近就宣布保真通过。

本组参考图只有一张，并无身份、人脸、自然透明素材、遮罩或多参考图像资格。
玻璃在RGB中透明，不等于抠图alpha正确；逐像素alpha差异已单列。没有下载视觉
评分模型或把文字肉眼观察当自动OCR评测。

画质组memory complete，峰值31,985,556,448 bytes，最大gap110,015,959ns，
系统swap-in/out0；同样检测到外部推理CPU竞争，不授予性能资格。

## 原始输入与独立CPU复核

另4个parity-only请求：生成/编辑×六/五步，seed73、prefix-off、dump_tensors；
不把dump/readback时间放进速度分母。两档实际BF16初始noise均为[1,1024,64]且
逐位相同；生成text[1,24,4096]、编辑text[1,35,4096]及编辑reference[1,1024,64]
均逐位相同，连safetensors文件SHA也相同。两档的最终denoised latent不同，
不宣称它们算术等价。四张dump PNG各自复现固定画质矩阵对应PNG。

`tools/validation/qwen_student_sampler_audit.py`独立CPU重算请求、14+4对图片差异、
全部时间分母、档内恢复、历史六步RGBA、raw输入、memory原件及来源快照。
`outputs/qwen512-five-step-audit-20261018.json`为complete，核对649项native source
hash和实际库/build manifest，含52个实际请求的证据（20测速+28画质+4诊断），
不改写旧窗口/PNG。测速/画质窗口分别有8/12项load sample记录竞争，不能把
CPU审核当GPU独占资格。raw窗口memory也complete，峰值32,018,799,584 bytes、
swap-in/out0；不作为每臂内存或性能结论。

## 构建、回归与原件

```text
build/qwen512-five-step-20261018/libturbocider.dylib
SHA256 65614d0ace1773dbfad9b8b0c257e10e4296f3d69252fc9d308ae9e63492e537
```

Private-enabled研究库构建通过，实测只用GPU。
contract134项：133通过、1个既有Wan fixture缺失跳过；GPU scheduler1通过，
检查真实五节点shift、默认六→五→六恢复和非法步数拒绝；GPU lifecycle1通过；
options host1通过（含五步边界、原六步observer不扩张）；sampler runner host5通过；
r128固定内容及454张量/227对geometry两项CPU检查通过。不是全仓/长时质量资格。

原件：

- `outputs/qwen512-five-step-timing-20261018/`：20张PNG、请求、回执、source、
  memory/load和完整原始时间。
- `outputs/qwen512-five-step-quality-20261018/`：28张固定矩阵、14对差异、
  `trial/comparison.png`、相同来源和负载/内存证据。
- `outputs/qwen512-five-step-raw-20261018/`：4张诊断图和初始/文本/参考/最终张量。
- `outputs/qwen512-five-step-audit-20261018.json`：新库和上述原件的CPU复核。
- `tools/validation/qwen_student_sampler_screen.py`：独立研究runner，未改写历史六步
  prefix runner的语义或旧结果。

```sh
.venv/bin/python -B tools/validation/qwen_student_sampler_screen.py \
  --library build/qwen512-five-step-20261018/libturbocider.dylib \
  --output outputs/new-qwen-sampler-timing --mode timing --seed 73

.venv/bin/python -B tools/validation/qwen_student_sampler_screen.py \
  --library build/qwen512-five-step-20261018/libturbocider.dylib \
  --output outputs/new-qwen-sampler-quality --mode quality --seed 73

.venv/bin/python -B tools/validation/qwen_student_sampler_audit.py \
  --library build/qwen512-five-step-20261018/libturbocider.dylib \
  --timing outputs/qwen512-five-step-timing-20261018 \
  --quality outputs/qwen512-five-step-quality-20261018 \
  --raw outputs/qwen512-five-step-raw-20261018 \
  --output outputs/new-qwen-sampler-audit.json
```

只新增独立研究构建和小型证据，不下载模型，不清理用户缓存，不替换已安装App。
下一轮继续主要DiT成本/可验证GPU与ANE分工，五步只保留为显式速度—画质取舍。
