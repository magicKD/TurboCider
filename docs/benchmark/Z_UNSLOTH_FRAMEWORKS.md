# Z-Image：原BF16源的GPU/ANE、mflux与Unsloth完整对照

2026-10-12，M4 Max64GiB/macOS26.6.2，512²/8步、seed42、base-only。
五路线完整正序/反序，共10进程、40PNG、10份完整独立进程树采样。
每进程cold/两fresh/立即重复，主中位数只合并自己路线的四fresh。

| 路线 | 四fresh s | cold正/反 s | constructor/readiness正/反 s | physical peak正/反 GB |
| --- | ---: | --- | --- | --- |
| native GPU | 7.356123 | 8.694701 /7.816146 | 0.141959 /0.160430 | 31.016 /31.498 |
| **native frozen W8A8** | **5.955417** | 17.429211 /14.205937 | 0.143926 /0.160806 | 31.889 /31.748 |
| mflux0.22 BF16 | 7.970898 | 9.012036 /8.328598 | 1.447734 /1.285144 | 22.525 /22.481 |
| Unsloth Metal/GPU-text | 18.735300 | 25.653716 /25.145972 | 0.321808 /0.328600 | 23.919 /23.863 |
| Unsloth Metal/defaultCPU-text | 22.599736 | 29.347143 /28.981926 | 0.316283 /0.322301 | 22.691 /22.615 |

采用最快已测Unsloth GPU-text路线作分母：纯GPU **2.547×**、混合
**3.146×**（减少68.213%）；对同组mflux混合 **1.338×**。
不拿较慢CPU-text宣传3.795×，也不拿旧native5.780s拼本组倍率。
混合首请求/内存成本较高；这些配置并不代表一次性任务或全设备最优。

## 来源与实际执行

- 同一Comfy-Org原BF16 encoder/DiT/VAE，source SHA绑定；native直接
  加载，mflux使用已有完整payload相同的lossless键/layout视图，Unsloth
  native后端直接加载原三文件。内部算术/模板/RNG/lifecycle仍不同。
- native实验库MLX0.32.3，514个native build inputs与源码匹配；frozen
  为5120/10240 image-only FFN，实际256 calls/request、失败0、warmup0、
  checkpoint验证通过，CoreML compute units为cpuAndNeuralEngine。
  软件调用计数不是物理INT8 MAC/全部NE占用率trace。
- mflux0.22 commit `31c9640399572ea394d07092763be9a36007a754`，实际1GiB
  cache、原BF16 precision与完整required weights覆盖；不隐启FP16 compute。
- Unsloth只读代码commit `8002a80aaf3474039a7cf53a6a3ac6b3051d487f`；
  mirror build tag `master-813-bfbef5b-u1d02858`，source commit
  `1d028587593a6924f1dc236c4d9a39a0a725b8cb`，2026-09-22发布。
  sd-server SHA `cee598a019362b0f8926543599c7ff73fbc9341b86f66eef207cf68852dff790`。
  原`max` flags与12 threads固定；GPU-text/defaultCPU-text两配置均测。
  实际日志分别确认Conditioner MTL0/CPU、DiT MTL0；每PNG都核验8个
  sampler steps和真实info参数，服务终态退出、listener PID限定于自有进程。

20对正反序同路线/同prompt完整RGBA全部exact；native GPU/frozen另外
8个完整图像比较保留原RGB/alpha/白底合成指标。已检查五路线soft-light
完整图，均为正常雪林狐狸；不同engine不以相同seed冒称相同noise。
CoreML编译锁不属于模型payload，实际compiled artifacts与manifest均绑定。
初次Unsloth预检已出图，但因自有lsof未声明child role而被memory verifier
拒绝；v2按原500ms门槛/完整角色核验通过，失败记录未改写成排名。

[完整逐请求与媒体/内存链机器记录](../design/validation/z-unsloth-mflux-current-20261012.json)、
[收敛性能报告](PERFORMANCE_REPORT.md)、[旧Z参数/格式细节](Z_IMAGE_FRAMEWORKS.md)。
所有模型/原参考代码不改写，source-generation SHA后只做metadata guard。
本轮结束后按用户要求不继续扩展网格，其他实验边界保留在审计。
