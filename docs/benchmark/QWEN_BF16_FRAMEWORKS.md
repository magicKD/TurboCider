# Qwen BF16：本轮原生混合结果与参考加载修复

2026-10-11，M4 Max64GiB/macOS26.6.2。原BF16 encoder/DiT/VAE、512²、
40步、seed29、三个fox-light prompts，无LoRA。相同当前native库，GPU/
frozen完整正反序，每新进程cold、两fresh、一次repeat；constructor另列。
同窗口的mflux BF16/Q4/Q8虽完成计时，却未正确加载64个FFN权重，图片
是明显彩色噪声；它们不是成功生成对照，不作任何跨框架倍率的分母。

## 有效原生对照

| 路线 | cold 正/反 s | 四fresh中位 s | repeat中位 s | tree peak GB（十进制） |
| --- | --- | ---: | ---: | --- |
| 原BF16 GPU | 46.328 /45.302 | 43.249697 | 42.571144 | 30.17–30.78 |
| 原BF16 GPU/Core ML frozen W6144 | 38.595 /40.722 | 31.007695 | 30.162046 | 35.10–35.53 |

混合对自身GPU为1.395×、省时28.305%，不是旧CPU busy窗口的33.121s，
也不是把repeat cache hit混入fresh。原生四个独立memory verifier全部
complete；该十进程完整窗口system compression/swap-out增量都0，
decompression/swap-in另保留，不因此声称全设备或硬RAM资格。

已恢复的32个checkpoint-bound W6144/r1024 source graphs与compiled
manifest放入独立benchmark cache，实际可加载。每40步请求1248次runtime
与首次32次warmup，failure0，checkpoint SHA通过；首步仍GPU，后续
image-only FFN分工。是Public Core ML配置，不称Private/native INT8 MAC
trace。encoder为原GPU/request-local策略，未靠更换或跳过encoder取胜。

已查看warm-light GPU/frozen：狐狸姿态、脸部、雪景/构图接近，局部毛发
略有差异。完整RGB correlation0.993641、Gaussian11 SSIM0.958431、
MAE4.609/255；alpha MAE0.05455/255。RGBA保留，白背景合成另列。
有限图片观察和单seed不代表通用质量资格；用户允许近似，不抹掉误差。

## 不得用失败参考制造领先

原Comfy DiT有32个`img_mlp.gate_up.weight`，每个包含gate/up两个连续
矩阵。mflux0.19.2 mapping要求`gate_layer.weight`与`proj.weight`，而
stock mapper不拒绝缺失field；v1派生布局未拆分这32个tensor，因此
64个required linears留在random initialization。路径可读、步数40、
完整PNG与memory complete都不能证明原模型正确执行。

| 原错误参考路线 | 诊断fresh中位 s | 有效排名 |
| --- | ---: | --- |
| mflux BF16，v1未拆分FFN | 56.879010 | 否，彩色噪声 |
| mflux affine Q4，v1 | 60.698671 | 否 |
| mflux affine Q8，v1 | 61.443489 | 否 |

原始图片、计时和source/memory链全部保留；不修改原失败窗口，也不
把它称为“mflux完整模型比我们慢1.83×”。已修复的是本仓库的加载布局：
原BF16 payload不变，只将连续gate/up byte ranges拆成两个required键。
没有改只读参考代码、权重值、模型步数或推理数学。

## 修复后的真正参考正在重测

v2完整布局所有297个DiT、398个text encoder、226个VAE required targets
覆盖；新增header gate，拒绝缺失真实parameters或空target假通过。
CoW encoder仅缩短不消费的352个名字，全部原payload仍保留。旧17.53GB
重复副本在独立clone验真后移除，原模型未动、可重新生成旧诊断布局。

修复后4步三参考路线均输出可辨识狐狸。另一个完整40步BF16单cold为
50.961601s，图片正常、memory complete；仍不以单cold替代warm排名。
新五路线十进程40步回文窗口
`outputs/qwen21-mflux-original-v9-complete-weights-five-route-20261011`正在
运行，全部完成并检查图片后再发布真正的外部框架比较。

[便携原生结果/原失败参考标记](../design/validation/qwen21-native-cohort-current-20261011.json)
绑定原source/library SHA、514个native inputs、完整各请求阶段、四树内存、
16个原生PNG与原参考invalid-output原因。raw40张PNG、模型与compiled
artifacts不进Git。扩大benchmark目标仍active。

[本机已测最高性能配置与恢复命令](best-local-configs-20261011.json)保留
request、source/compiled manifest、channel分工和实际调用数；不自动改变
生产默认，也不因恢复artifact就跳过完整来源检查。
