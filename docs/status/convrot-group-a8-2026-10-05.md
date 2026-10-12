# ConvRot group A8：实际 ANE 图接入与负结果

本轮是进展，不是双后端完整目标完成。接续`5184359`：row A8 + FP32
partial join 的四步N1仍失败，因此实现实际input/hidden group A8，而
不是把GPU oracle的改善当作Private已经合格。

## 实现

`TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE=256`明确选Comfy direct-code
路径；默认0与现有Public/Sylvester/row recipe不变。输入经FP32 radix-4
Comfy H256、原dtype舍入，再按每个H256的peak生成RNE A8与normalized
FP16 scale。原signed W codes（含-128）与stored row scales不重量化。

两套A slots的tx surface变为`[H/256,tokens]`；per-group尺度kernel与
codes pass保留独立ready event、source leases和整体validation flags。
group字段进入selection identity；未知group/basis/weight-staging组合
拒绝。specialized pipeline上限26，旧18项路径仍保留。新增尺度存储与
page slack纳入admission，而不是把额外surface当免费。

hidden graph将H256旋转后的`[1,1,F,tokens]` reshape为
`[1,F/256,256,tokens]`，每组reduce peak、FP16 ratio、int8 quantize，
保留原unrotated corrected hidden输出。down各组normalized dot乘
`group_floor/global_floor`再累加；GPU只需恢复一份global scale。
这不证明原生INT8 MAC、FP32 ANE arithmetic或物理重叠。

试过两版input reduction：

- v1：每个partial恢复W/token scales后再加，总图质量不一致。
- v2：tx group scales先除以token全局scale，normalized partial乘ratio
  后累加，只恢复一次W/token scale。减少重复scale运算、保持normalized
  数值域。recipe是`comfy-h256-direct-q8-source-round-group256-a8-rne-ratio-norm-f16-v2`。

v1/v2都保持原double banks、LoRA correction/hidden ABI、FP32 partial
可选路径与失败后整operation GPU重算。Group尺度与extra flag进入
executor identity，receipt报告actual `a8_group_size`，不只读环境意图。

## 组件与真实driver检查

9组FP16/BF16/FP32、K512/3840/10240、物理strides/slices：group scales、
codes逐位对独立CPU radix-4/dtype/RNE oracle；包含不同peak、zero、tiny、
outlier group与padding。generic/specialized结果逐位一致。原37项direct
W、9项row A8、positive/signed restore验证继续通过。

实际Private driver的raw/packed × serial/lookahead四格，三chunks，
source/LoRA-hidden oracle、negative/zero scale、FP32 final-BF16 control、
future identity、late failure与refill通过。常量blob/group geometry有host
检查。v2 Private17项：15 pass、2个未开启的MLX/channel/calibration集成
skip；Public13、GPU6、host8通过。未运行完整`make test`。

## 四步轨迹：不能提升默认

所有case同binary GPU/reference与Private candidate、同原checkpoint、
同prompt/seed42，Private4096/GPU6144、1056-row bucket、FP32 partial
join开启、四步；128个channel blocks、248/512个driver calls、零fallback，
headroom=1、零retry。conditioning与initial latent逐位相同。

| 版本／尺寸 | final rel L2 | cosine | N1 |
| --- | --- | --- | --- |
| row + FP32 partial／512² | 0.04336734 | 0.99905977 | fail |
| row + FP32 partial／1024² | 0.08632567 | 0.99627542 | fail |
| group v1／512² | 0.07155510 | 0.99744197 | fail |
| group v1／1024² | 0.06652789 | 0.99778597 | fail |
| group ratio v2／512² | 0.05183588 | 0.99865575 | fail |
| group ratio v2／1024² | 0.08450011 | 0.99643390 | fail |

N1 rel L2≤0.03、cosine≥0.999不变，native零validation调用时的passed
字段不用作资格。两版都不是普遍更优；不能把某尺寸改善、toy通过或
较小frozen FFN error推断为采样轨迹通过，更不能借cold diagnostic wall
作为正式1.2×分母。GPU source参考是legacy recipe，不是正式最优GPU
性能资格。保留原PNG/tensors/raw reports与所有失败。

v1 frozen-source512重放的16项Private group FFN error有所降低，但
完整trajectory仍失败；input/hidden的误差方向与FP16 reduction变化
尚未拆清。下一步必须分别开关input与hidden group、检查normalized
FP16 partial累加与原dtype boundaries，并测source-sensitive层GPU/Public
策略。当前证据不能确定哪一项是主因。

## 构建与证据

v1 Private：`80d8dc96ca6c53399f39152636585459f6f71f145b9826a356b252dbd1f822e7`。
v2 Private：`d66cc548a3ccd660c0336b0a33cf85eb5384ac3dc19f9d6fe7d989a3ffd59a94`。
v2 Public：`e3562eb6319f18e97745b32949a37d3e1fee8f80cd1150f5a9c442b9bcbb765a`，
actual release guard通过。Private实验flags开启、Public均关闭。测试
仍是保留原ConvRot drafts的working-tree snapshot，不是clean staged
tree rebuild；原未提交改动不混入本轮selective commit。

目录：`outputs/convrot-group256-{512,1024}-{gpu4,private4096-4}-diagnostic/`，
`outputs/convrot-group-ratio-{512,1024}-{gpu4,private4096-4}-diagnostic/`。
frozen replay：`outputs/convrot-group256-ffn-replay512.json`。其中GPU
normalized-FP16 emulation仍是旧row recipe，不能冒称group emulation。

Z/Qwen base四格matched optimized GPU≥1.2×、实模型LoRA、自动bandwidth
fit/cache/share/prefetch、真实GGUF有界dense consumer与正式媒体/内存/
device trace全部仍需完成。此实验不缩小原目标或改变完成条件。
