# ConvRot 编译 GPU 实验：剩余工作区改动收尾

2026-10-10。上一笔 `9ac3e01` 只提交当轮 Qwen share/rank 工具与记录，
把原有六个 tracked 改动和七个 untracked ConvRot 文件保留在工作区。
本次响应用户对剩余未提交代码的追问，核对并单独收好这一完整实验组，
不撤销、丢弃或用占位实现替换原代码。

## 提交范围与默认边界

保留 Z-Image 的显式 `compiled_dense` / `compiled_butterfly` ConvRot GPU
recipe，所有层的 packed codes、BF16 scales、metadata 都作为动态参数。
使用原 legacy BF16-scale Q8 source，不产生永久 dense 权重副本。

保留 block/final-latent 双轨 source validation、对应 source-profile/result
telemetry，以及0..4GiB的 allocator cache hint 和显式 bins retention。
退出或取消后恢复 caller hint；retention需等待临时 GPU owners完成、按
选定 hint修剪。该 hint不是全请求RAM硬上限，不声称整体内存资格。

非 legacy recipe仍要求实验构建、明确 approximation、resident纯GPU、
无LoRA/ANE/streaming/显式内存约束及冲突调试配置。普通 Public构建必须
拒绝实验入口；默认 legacy GPU、已有 Private/ANE route不改变。
compiled实验结果仍标明未qualified，不因补提交而变成发行默认。

测量工具区分 timing/memory/diagnostic；时间臂不运行内存sampler或事件
callback，dump/source validation只允许diagnostic。组件、source、library、
工作负载、cache配置及PNG身份继续由独立screen核对。

## 历史记录重放，不重写成新性能结果

三份2026-10-02/03的ConvRot/BF16 JSON对原timing/memory reports独立重放，
原report SHA和verifier SHA一致，完整screen结果一致：

| 历史recipe | candidate/BF16 warm wall | candidate/BF16 observed process memory |
| --- | ---: | ---: |
| compiled dense /default | 1.192555 | .796753 |
| retained3GiB | 1.234759 | .562420 |
| retained4GiB | 1.232117 | .610452 |

这些是纯GPU压缩路径的历史速度/内存折中：不是ANE加速倍率，不把小于
1.2的耗时比说成1.2倍加速。qualification仍false、quality未测、whole-request
memory unknown。原JSON仅补入版本控制，没有改数值、门槛或原始文件。

## 本次验证

7项host tests通过、无skip：ConvRot/BF16 fail-closed screen、probe参数/
observer隔离和共用BF16 baseline contracts。当前Private保留库的实模型
compiled-engine test通过、无skip，覆盖caller cache hint恢复、bins上限、
approximation授权拒绝、cancel/retry及prompt A/B/A图像一致。

当前普通Public库的experimental-gate test通过、无skip，覆盖14种显式
未qualified route拒绝，其中包含两种compiled ConvRot recipe，不生成PNG。
Private/Public各502个runtime source input SHA与当前源码均一致；使用
已有库测试，没有新native build，也未将较旧ordinary库当当前树验证。

日志保留在 `outputs/convrot-pending-commit-host-test-20261010.log`、
`outputs/convrot-pending-commit-native-test-20261010.log` 和
`outputs/convrot-pending-commit-public-gate-test-20261010.log`，不加入Git。
测试temporary目录自动清理，保留库无`.o`/module-cache。本次没有下载/
改写模型、adapter/ref，没有删除用户草稿、原证据或向外部进程发信号。

本次是版本控制收尾，不是新的性能推广；本地512²当前收益和后续缺口仍见
[性能进度总览](performance-progress-2026-10-10.md)。
