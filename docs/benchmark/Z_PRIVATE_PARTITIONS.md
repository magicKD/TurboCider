# Z-Image Turbo：Private通道/join完整网格

2026-10-11，M4 Max64GiB。原BF16 encoder/DiT/VAE、512²、8步、seed42，
无LoRA。7种partial shares×F32/F16 join，14候选均完成；每候选独立
GPU→frozen→Private→Private→frozen→GPU，每进程cold/两fresh/repeat。
共84进程、336PNG、84完整memory，四fresh/route，不混repeat或跨候选
最小sample。库为build/flux9-compiled-v1-private，514源输入匹配。

## Private完整请求中位数

| ANE channels / GPU complement | F32 join s | F16 join s |
| --- | ---: | ---: |
| 2048 /8192 | 6.832208 | 6.894335 |
| 3072 /7168 | 6.483783 | 6.485205 |
| 4096 /6144 | 6.147741 | 6.192168 |
| 5120 /5120 | 6.676390 | 6.695324 |
| 6144 /4096 | 7.476019 | 7.522188 |
| 7168 /3072 | 8.176816 | 8.184563 |
| 8192 /2048 | 8.933591 | 8.952478 |

该网格Private最快为4096/F32。其自己的GPU7.345378/frozen5.793671/
Private6.147741s，Private对GPU为1.195×、少16.305%，但比同窗口
frozen慢约6.11%。不跨候选挑最低frozen数字当分母。所有候选中frozen
均更快，F16 join没有得到更低时延；更大ANE份额反而拖慢，不能按
“越多ANE越好”选择路由。超过6144的Private甚至慢于自身GPU。

实际ane_channels/gpu_channels、fp32 join、executor_backend、每请求
256 channel blocks、成功calls/retry/headroom与fallback0都独立核验。
cold的overflow retries保留，warm若继续调整retry策略则拒绝候选。
host执行成功不等于hardware占用/原生INT8 MAC或全设备资格。

[完整便携网格](../design/validation/z-private-partition-grid-current-20261011.json)
保留336PNG/raw SHA、各请求阶段/计数、cold/fresh/repeat数组、84memory
和每候选warm-light RGBA差异。full_audit SHA绑定所有原尺寸图片比较
和正反序检查；不缩图、不删alpha，不因允许较宽delta把近似称无损。
4096/F32 warm-light SSIM0.981101、MAE2.027/255；4096/F16为0.985778/
1.684，但没有更低时延。168对正反序RGBA全部exact，只证明这些固定
任务确定性，不代表广泛质量parity。

当前best-local-configs保留既有frozen热态选择，同时补4096/F32作为
该完整网格已测Private最优；不是自动生产路由、所有尺寸或全局最优。
其它checkpoint/ConvRot/GGUF/1024²应独立比较，不套用本网格倍率。
入口为z_private_partition_sweep.py；失败/无summary启动也有原日志
binding和明确N/A，不会被成功样本掩盖。模型/336PNG/84raw日志不进Git。
