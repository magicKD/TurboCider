# Z-Image Turbo：完整框架与GPU/ANE对照

2026-10-11，M4 Max64GiB/macOS26.6.2。原BF16 encoder/DiT/VAE经无损
布局适配给只读mflux0.19.2；512²、8步、seed42，无LoRA。十四进程
七路线完整正反序，每进程cold、两fresh、一次repeat，constructor另列。
56张PNG、14条独立进程树内存证据完整；fresh不混入repeat cache hit。

## 完整PNG墙钟

| 路线 | 四fresh中位 s | cold 正/反 s | repeat 正/反 s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| TurboCider GPU | 7.363311 | 10.231 /7.590 | 6.991 /6.990 | 30.76–31.91 |
| TurboCider frozen Core ML | 5.780123 | 21.587 /11.608 | 5.334 /5.332 | 30.90–32.08 |
| TurboCider Private runtime | 6.139409 | 11.867 /6.646 | 5.753 /5.763 | 30.57–31.64 |
| TurboCider Public runtime | 6.537484 | 8.751 /9.816 | 6.276 /6.165 | 30.93–31.57 |
| mflux BF16 | 9.519604 | 10.337 /9.545 | 9.527 /9.536 | 31.57–31.58 |
| mflux affine Q4 | 9.971474 | 10.054 /10.055 | 10.022 /9.961 | 26.89 |
| mflux affine Q8 | 10.080610 | 10.192 /10.189 | 10.162 /10.076 | 28.67–28.68 |

对mflux BF16，原生GPU为1.293×，frozen为1.647×。frozen对自身GPU
省21.501%，当前七路线热态最快；冷启动明显更贵，不应将它推广为
一次性请求最优。Private/Public不同路径均保留，不互换标签。

参考Q4/Q8覆盖其eligible text/DiT/VAE modules，是独立affine recipe，
不是原GGUF Q4_0/Q8_0或ConvRot。这轮降低观测peak但未降低延迟。
模型组件/allocator生命周期不同，完整框架倍率不等于纯kernel倍率。

## 实际执行和来源

- GPU是原生compiled fused blocks，encoder GPU，不是CPU慢分母。
- frozen使用checkpoint-bound image-only W8A8，ANE5120/GPU5120、
  rows1024；每request256成功runtime calls。配置cpuAndNeuralEngine，
  physical NE residency未观测，不把Core ML成功当硬件占用trace。
- Private为Hadamard W8A8、intermediate channels ANE4096/GPU6144、
  F32 partial join、gpu_iosurface。cold出现已记录的overflow retry；
  四warm均不继续改变headroom，fallback0，不能抹掉初始化retry。
- Public为row partition、352-row模板、gpu_iosurface；实际256个hybrid
  block/request，fallback0。它与Private通道分割不是同一算法边界。
- mflux required coverage：519 DiT、398 encoder、244 VAE全部覆盖，
  不只检查文件存在。mapped VAE buffers也纳入required target。
  stock Z每次重新编码prompt，repeat不能伪称reference cache hit。

原生和参考warm-light输出已查看，为相近构图的正常狐狸图，而非Qwen
旧失败的彩色噪声。完整原尺寸RGBA/RGB差异、alpha/白背景合成和正反
序确定性保留在机器记录；有限prompt/单seed不代表广泛画质资格。

warm-light相对原生GPU，frozen/Private/Public的RGB SSIM分别0.932267/
0.981101/0.989876，MAE4.973/2.027/1.315每255；Private/Public虽稍慢，
此样本漂移更小。mflux BF16 SSIM0.904900、MAE7.394；affine Q4
SSIM0.792111、MAE21.958，不能只报省内存而抹去图像差异。28对正反
序RGBA均exact；这只证明当前固定任务确定性，不是广泛质量parity。

## 复现与范围

[完整便携证据](../design/validation/z-valid-framework-current-20261011.json)
绑定源/库/514个native输入、无损layout receipt、参考源码snapshot、
全部请求与内存/PNG SHA、channel/join/calls/retry/fallback和图像差异。
内存是load+cold+warm+exit的process-tree physical peak，不是硬RAM cap。
system compression/swap-in/out都保留，不只报告MLX allocator。

入口为tools/validation/z_mflux_framework_screen.py，使用原BF16派生
布局与build/benchmark-async-v1-private，8步/4request、private4096/f32：
native-gpu,native-frozen,native-private,native-public,mflux-bf16,mflux-q4,mflux-q8,mflux-q8,mflux-q4,mflux-bf16,native-public,native-private,native-frozen,native-gpu。
初次完整source SHA后generation未变时复用，不在每个inference重扫。

[五种本地表示16配置](Z_IMAGE_FORMATS.md)与本页是独立campaign，不
混用最小单样本。更高/更低Private share、F16 join、1024²仍需当前
完整筛选；[已测最佳配置](best-local-configs-20261011.json)不是全局最优
证明，不自动改变生产默认。raw PNG/模型/compiled graphs不进Git。
