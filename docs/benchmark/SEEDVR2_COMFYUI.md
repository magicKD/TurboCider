# SeedVR2：TurboCider GPU vs ComfyUI MPS

2026-10-15整理，复用2026-10-14已完成的8组ABBA记录；本次没有重跑推理。
Apple M4 Max /64GiB /macOS26.6.2，SeedVR2 3B原FP16 DiT与VAE，无LoRA。
这是**图像／视频超分**，与生成模型的历史四家族排名分开。

## 完整请求热态对照

每个任务选默认与SDPA中**实测更快的Comfy预设**，TC也取该组配对中位数；
不独立挑另一组更低的TC耗时。倍率为Comfy /TC，节省比例为1−TC /Comfy，
均由未舍入样本计算；每路6个真正执行的warm请求。

| 超分任务 | 更快Comfy预设 | Comfy完整s | TC GPU完整s | 倍率 | 每次节省 | 耗时减少 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 图像256→512 | SDPA/GPU-only | 3.978 | 0.828 | **4.81×** | 3.150s | 79.2% |
| 图像512→1024 | 默认attention/普通驻留 | 15.535 | 3.506 | **4.43×** | 12.029s | 77.4% |
| 视频256→512，25帧/24FPS | 默认attention/普通驻留，MPS兼容补丁 | 71.190 | 20.586 | **3.46×** | 50.604s | 71.1% |
| 视频512→1024，25帧/24FPS | 默认attention/普通驻留，MPS兼容补丁 | 295.438 | 85.011 | **3.48×** | 210.428s | 71.2% |

不是4.81×的ANE收益：本表TC为**resident GPU路线**。也不是不同框架的
等精度／逐像素一致性证明，原记录仍保留`overall_qualification_passed=false`。
默认与SDPA的差距很小（512图像Comfy约0.18%），不据此声称统计显著或普适最优。

## 默认与SDPA控制均保留

下表每行来自自己的完整ABBA记录。cold按该路线两次独立进程的首次请求列出，
**不含constructor／server-ready**；不能用warm倍率推算一次性请求收益。

| 任务 | Comfy预设 | TC warm s | Comfy warm s | 倍率 | TC cold s（两次） | Comfy cold s（两次） | 记录键 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 图像256→512 | 默认 | 0.827 | 3.985 | 4.82× | 1.567 / 1.411 | 5.266 / 4.959 | `image512-default` |
| 图像256→512 | SDPA/GPU-only | 0.828 | 3.978 | 4.81× | 1.411 / 1.413 | 4.671 / 4.594 | `image512-sdpa` |
| 图像512→1024 | 默认 | 3.506 | 15.535 | 4.43× | 4.123 / 4.075 | 16.772 / 16.547 | `image1024-default` |
| 图像512→1024 | SDPA/GPU-only | 3.504 | 15.636 | 4.46× | 4.053 / 4.064 | 16.201 / 16.203 | `image1024-sdpa` |
| 视频256→512/t25 | 默认 | 20.586 | 71.190 | 3.46× | 21.207 / 21.241 | 72.104 / 72.109 | `video512-default` |
| 视频256→512/t25 | SDPA/GPU-only | 20.598 | 71.337 | 3.46× | 21.230 / 21.271 | 72.291 / 72.302 | `video512-sdpa` |
| 视频512→1024/t25 | 默认 | 85.011 | 295.438 | 3.48× | 85.464 / 85.636 | 296.494 / 296.494 | `video1024-default` |
| 视频512→1024/t25 | SDPA/GPU-only | 84.879 | 298.901 | 3.52× | 85.444 / 85.434 | 297.757 / 297.369 | `video1024-sdpa` |

## 测试口径与版本

- 四个独立进程／服务按TC→Comfy→Comfy→TC运行；每进程1次cold、3次warm，
  seed依次42、43、44、45。每个预设／任务共16个完整请求，合计128个。
- 计时包含导入、bicubic resize、VAE encode、1步Euler/simple采样（CFG1）、
  VAE decode、PNG或无音轨H.264 MP4导出；不做颜色校正。
  不包含进程／模型准备、构造或离线编译；不同API边界仍有各自的调用开销。
- 每请求使用新输入文件名，原生conditioning cache均未命中；Comfy媒体、
  resize、encode、sampler、decode、export节点每次实际执行，允许模型loader缓存。
  两边所有输出均有尺寸检查，视频每次均检查完整25帧、24FPS，未截帧测速。
- TC精度配方为`seedvr2-stable-fp16-fp32-v1`：原FP16权重、
  FP16 DiT投影／attention与VAE，FP32 residual/time/norm/output/Euler；
  默认allocator cache上限4GiB。不是后续block-stream原型或VAE/ANE实验。
- ComfyUI **0.39.0**，commit
  `b0b743566f65daafc423b4fea8a2fbda94b3384a`，PyTorch **2.12.1 MPS**，
  Python **3.13.12**。默认组使用sub-quadratic attention／普通驻留，
  显式`--fp16-vae`；SDPA组另加
  `--use-pytorch-cross-attention --gpu-only --fp16-unet`。
  **两种预设的实际日志均为**
  `model weight dtype torch.float16, manual cast: torch.bfloat16`，
  不能把命令行的FP16标志当作全部算术为FP16的证明。

### 视频参考的兼容性修补

视频Comfy为任务自有副本，仅用
[non-CUDA cache-transfer guard](../../tools/validation/seedvr2_comfy_mps_cache.patch)
在非CUDA设备关闭依赖CUDA streams/events的VAE缓存异步转移。
它改变了MPS缓存转移策略，故应称**MPS兼容Comfy参考**，不是未修改stock视频。
没有修改用户的服务／源码，也没有把stock崩溃当成速度分母；图像组无此补丁。

## 来源与可追溯性

提交的[机器可读样本与身份](../design/validation/seedvr2-comfy-benchmark-current-20261015.json)
保留8组全部cold／warm样本、输入SHA、DiT/VAE SHA、原生library/build SHA，
以及每份配对／子报告和dtype日志的相对记录ID与SHA-256。
原始大体积输出／日志保留在本地ignored目录，不随仓库发布；记录ID不是公开下载链接。
不把后来源码／App版本冒充这些历史二进制的重新配对测量。

| 任务／预设 | 保留配对记录ID（相对于outputs） | 原生library SHA前12位 |
| --- | --- | --- |
| `image512-default` | `seedvr2-stable-image512-paired-default-v1-20261014/report.json` | `de18ca6024dc` |
| `image512-sdpa` | `seedvr2-stable-image512-paired-sdpa-v1-20261014/report.json` | `de18ca6024dc` |
| `image1024-default` | `seedvr2-stable-image1024-paired-default-v1-20261014/report.json` | `de18ca6024dc` |
| `image1024-sdpa` | `seedvr2-stable-image1024-paired-sdpa-v1-20261014/report.json` | `de18ca6024dc` |
| `video512-default` | `seedvr2-staged-candidate-video512-t25-paired-default-v1-20261014/report.json` | `45caa42aba57` |
| `video512-sdpa` | `seedvr2-stable-video512-t25-paired-sdpa-v1-20261014/report.json` | `de18ca6024dc` |
| `video1024-default` | `seedvr2-media-lifetime-video1024-t25-paired-default-v1-20261014/report.json` | `596c160aabcc` |
| `video1024-sdpa` | `seedvr2-stable-video1024-t25-paired-sdpa-v2-20261014/report.json` | `de18ca6024dc` |

## 质量、内存与结论边界

8组记录均为`status=complete`、`speed_gate_passed=true`，
但`overall_qualification_passed=false`：速度门通过不等于整体质量／ANE资格通过。
稳定配方对原参数独立FP32参考的逐层／像素门另见
[SeedVR2验证边界](../public/SEEDVR2.md#validation-boundary)；
不能用这些独立检查声称与Comfy BF16配方bit-exact。

本页没有同口径Comfy进程物理内存采样，**不发布跨框架内存倍率／排名**。
原生MLX active peak不等于进程physical footprint，也不能与Metal allocation直接相加。
VAE tiling、CPU+NE、流式权重等后续实验的内存／耗时必须单独比较，
不能拼入本页历史Comfy分母制造新的加速比。

结论仅覆盖本机、这些权重／版本和四个任务；不外推到所有Mac、第三方插件、
长视频、超过121帧或16GiB设备。测量入口：
[ABBA observer](../../tools/validation/seedvr2_paired_cohort.py)、
[逐请求observer](../../tools/validation/seedvr2_framework_screen.py)。
