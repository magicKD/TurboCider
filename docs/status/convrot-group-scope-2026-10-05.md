# ConvRot input／hidden group scope 对照

接续`a491678`，本轮明确拆分两种A8 grouping，保留默认row路径。
完整目标仍是Z/Qwen四格base≥1.2×、实模型LoRA、双后端、GGUF有界
转换/调度和正式质量/内存/trace验收；下面不是缩小的完成条件。

## 接口与不变量

`TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE=256`配合
`TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE=input|hidden|both`。不写scope
仍是原both，默认group=0仍是row。scope要求明确group256与授权的
Private Comfy direct-code路径；无效值/未授权组合不忽略。

input-only的tx surfaces按H256分组、gate/up用normalized scale ratios，
hidden仍采用全token scale与原K tile；hidden-only保留原row input
staging/gate/up，只有hidden quantization和down partial变为group256。
因此不再用一个全局group bool意外改变未选择的projection K tile。
原H256常量blob、master signed codes/scales、global down epilogue、
double banks、ready/done events、corrected-hidden LoRA ABI和整operation
GPU fallback保持。实际input/hidden group sizes都进入receipt；scope
进入executor identity，独立recipe和MIL/constants hash隔离compiled cache。

Private driver的input/hidden各四格raw/packed × serial/lookahead，连同
原both四格，source/LoRA-hidden、FP32恢复、weight switch、future identity、
late failure/refill通过。host emitter检查两种图确实独立。不宣称原生
INT8 MAC、FP32 ANE arithmetic或物理GPU/ANE重叠。

## 同库四步结果：仍失败

原ConvRot checkpoint、legacy packed BF16 scales、同prompt、seed42，
GPU reference与Private4096/GPU6144、1056-row bucket、FP32 partial join，
相同binary。每格conditioning与initial latent逐位相同。每格128个
channel blocks、248/512次driver calls，零runtime failure／fallback。

| scope／尺寸 | final rel L2 | cosine | N1 |
| --- | --- | --- | --- |
| input-only／512² | 0.05130107 | 0.99868575 | fail |
| hidden-only／512² | 0.05116189 | 0.99869092 | fail |
| input-only／1024² | 0.05748273 | 0.99834759 | fail |
| hidden-only／1024² | 0.07345785 | 0.99730769 | fail |

原N1 rel L2≤0.03、cosine≥0.999不变；native零validation调用时的passed
字段不当作资格。输入分组在1024比row+FP32的0.08632567有所改善，
但两种scope都没有同时通过两个尺寸，不能成为通用推荐。不能根据
某个尺寸或toy通过提升默认/宣称1.2×；cold dump诊断不用于速度分母。

下一步需要源精度边界/FP16 reduction独立控制及敏感层GPU/Public
策略，并重新做完整trajectory与matched optimized GPU性能资格；
当前数据不能证明是单一input或hidden量化缺陷。Qwen/Z base加速与
实模型LoRA、自动bandwidth fit/cache/share/prefetch、真实GGUF消费者
A/B、正式媒体/内存/trace仍未完成。

## 证据与构建

目录 `outputs/convrot-group-scope-{512,1024}-gpu4-diagnostic/` 与
`outputs/convrot-group-scope-{512,1024}-{input,hidden}-private4096-4-diagnostic/`。
保留PNG/tensors/raw报告与独立CPU FP64比较，不删除失败。

Private SHA256：
`6c83bba7e62bddd3383fdbfb00d4adaea60124d82215d643dc4598692bb47e6f`。
Public SHA256：
`7cb8c1cb685542ee76e8d05522da3550327a660bb34f0483477184b1208eff6f`。
Public actual release guard通过。Private18项：16 pass/2显式MLX opt-in
skip；Public13、GPU6、host8通过，不是完整`make test`绿色。
构建仍是保留原ConvRot drafts的working-tree snapshot，不是clean staged
tree rebuild；本轮selective commit不夹带原未提交改动。
